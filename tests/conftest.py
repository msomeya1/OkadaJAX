import os

# Must be set before jax is imported.  JAX grabs 75% of every visible GPU's
# memory up front by default, which is unkind on a shared machine.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The reference data is float64 throughout.  JAX silently truncates float64 to
# float32 unless this is set, which would weaken every tolerance in the suite.
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "golden")


def _devices():
    devs = [jax.devices("cpu")[0]]
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if gpus:
        # index 1 is the GPU reserved for this work when there are two
        devs.append(gpus[1] if len(gpus) > 1 else gpus[0])
    return devs


@pytest.fixture(scope="session", params=_devices(), ids=lambda d: d.platform)
def device(request):
    return request.param


@pytest.fixture(scope="session")
def fortran_golden():
    return np.load(os.path.join(GOLDEN_DIR, "fortran_reference.npz"))


@pytest.fixture(scope="session")
def wrapper_golden():
    """OkadaTorch's wrapper output for well-conditioned configurations.  The two
    wrappers implement the same conventions, so OkadaJAX must reproduce it."""
    return np.load(os.path.join(GOLDEN_DIR, "wrapper_snapshot.npz"))


@pytest.fixture
def okada():
    from OkadaJAX import OkadaWrapper
    return OkadaWrapper()


# ---------------------------------------------------------------- helpers
def rel_err(got, ref):
    """Element-wise error normalised by max(|ref|, 1); see the OkadaTorch suite."""
    got, ref = np.asarray(got, dtype=float), np.asarray(ref, dtype=float)
    return np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)


def put(v, device, dtype=jnp.float64):
    return jax.device_put(jnp.asarray(v, dtype=dtype), device)


def make_coords(device, dtype=jnp.float64, with_z=False, n=5):
    """A small, well-conditioned station grid (km)."""
    x = np.linspace(-180.0, 220.0, n)
    y = np.linspace(-160.0, 210.0, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = {"x": put(X, device, dtype), "y": put(Y, device, dtype)}
    if with_z:
        coords["z"] = put(np.full_like(X, -7.0), device, dtype)
    return coords


def make_params(device, dtype=jnp.float64, rect=True, **over):
    base = dict(x_fault=3.0, y_fault=-11.0, depth=6.5, strike=189.0,
                dip=57.0, rake=101.0, slip=5.62)
    if rect:
        base.update(length=218.0, width=46.0)
    base.update(over)
    return {k: put(v, device, dtype) for k, v in base.items()}


def wrapper_configurations(device):
    """The inputs OkadaTorch's wrapper_snapshot.npz was generated from
    (OkadaTorch tests/generate_golden.py)."""
    x = np.array([-137.0, -41.0, 13.0, 88.0, 211.0])
    y = np.array([203.0, -66.0, 7.0, -155.0, 91.0])
    X, Y = np.meshgrid(x, y, indexing="ij")
    X, Y = put(X, device), put(Y, device)
    Z = put(np.full(X.shape, -7.0), device)

    def p(**over):
        base = dict(x_fault=3.0, y_fault=-11.0, depth=6.5, strike=189.0,
                    dip=57.0, rake=101.0, slip=5.62)
        base.update(over)
        return {k: put(v, device) for k, v in base.items()}

    rect = dict(length=218.0, width=46.0)
    cases = {}
    for origin in ("topleft", "center"):
        for strain in (False, True):
            cases[f"rect_surface_{origin}_strain{int(strain)}"] = (
                {"x": X, "y": Y}, p(**rect), dict(fault_origin=origin, compute_strain=strain))
            cases[f"rect_depth_{origin}_strain{int(strain)}"] = (
                {"x": X, "y": Y, "z": Z}, p(**rect), dict(fault_origin=origin, compute_strain=strain))
    for strain in (False, True):
        cases[f"point_surface_strain{int(strain)}"] = (
            {"x": X, "y": Y}, p(), dict(compute_strain=strain))
        cases[f"point_depth_strain{int(strain)}"] = (
            {"x": X, "y": Y, "z": Z}, p(), dict(compute_strain=strain))
    cases["rect_surface_dip90"] = ({"x": X, "y": Y}, p(dip=90.0, **rect), dict(compute_strain=True))
    cases["rect_depth_nu030"] = ({"x": X, "y": Y, "z": Z}, p(**rect), dict(compute_strain=True, nu=0.30))
    return cases


PARAM_NAMES = ["x_fault", "y_fault", "depth", "length", "width",
               "strike", "dip", "rake", "slip"]


def finite(arrays):
    return all(bool(jnp.all(jnp.isfinite(a))) for a in arrays)
