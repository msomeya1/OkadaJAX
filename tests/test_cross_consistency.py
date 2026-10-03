"""Internal consistency of OkadaWrapper, and agreement with OkadaTorch.

These check relations that must hold between different code paths, which
catches wiring mistakes (wrong rotation, wrong ``fault_origin`` offset, swapped
strain components) that a single-path reference test cannot.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import make_coords, make_params, put, rel_err, wrapper_configurations

# Okada 1985 (surface) and Okada 1992 (z=0) are different formulae for the same
# quantity, so agreement is a genuine cross-check rather than a tautology.
TOL_1985_VS_1992 = 1e-9
# analytic strain vs. autodiff of the displacement: limited only by float64
TOL_STRAIN_VS_AD = 1e-10

NAMES12 = "ux uy uz uxx uyx uzx uxy uyy uzy uxz uyz uzz".split()


@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
@pytest.mark.parametrize("origin", ["topleft", "center"])
def test_surface_1985_matches_depth_1992_at_z0(okada, device, rect, origin):
    coords = make_coords(device)
    params = make_params(device, rect=rect)
    coords_z = dict(coords, z=jnp.zeros_like(coords["x"]))
    a = okada.compute(coords, params, compute_strain=True, fault_origin=origin)
    b = okada.compute(coords_z, params, compute_strain=True, fault_origin=origin)
    for n, u, v in zip(NAMES12, a, b):
        err = rel_err(u, v).max()
        assert err < TOL_1985_VS_1992, f"{n}: 1985 vs 1992 differ by {err:.3e}"


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_analytic_strain_matches_autodiff(okada, device, with_z):
    """The 9 spatial derivatives returned by compute() must equal d(u)/d(x,y,z)."""
    coords = make_coords(device, with_z=with_z)
    params = make_params(device)
    full = okada.compute(coords, params, compute_strain=True)
    for j, arg in enumerate(["x", "y", "z"] if with_z else ["x", "y"]):
        grad = okada.gradient(coords, params, arg=arg, compute_strain=False)
        for i in range(3):
            err = rel_err(full[3 + 3 * j + i], grad[i]).max()
            assert err < TOL_STRAIN_VS_AD, \
                f"d(u{i})/d{arg}: analytic vs autodiff differ by {err:.3e}"


def test_free_surface_conditions_hold(okada, device):
    """At z=0 the traction-free surface forces uxz=-uzx, uyz=-uzy and
    uzz = -nu/(1-nu) (uxx+uyy).  The 1992 path computes these independently."""
    nu = 0.25
    coords = dict(make_coords(device))
    coords["z"] = jnp.zeros_like(coords["x"])
    ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz = okada.compute(
        coords, make_params(device), compute_strain=True, nu=nu)
    assert rel_err(uxz, -uzx).max() < 1e-9
    assert rel_err(uyz, -uzy).max() < 1e-9
    assert rel_err(uzz, -(uxx + uyy) * nu / (1 - nu)).max() < 1e-9


def test_degree_and_radian_inputs_agree(okada, device):
    coords = make_coords(device, with_z=True)
    deg = make_params(device)
    rad = {k: (v * jnp.pi / 180.0 if k in ("strike", "dip", "rake") else v)
           for k, v in deg.items()}
    a = okada.compute(coords, deg, compute_strain=True, is_degree=True)
    b = okada.compute(coords, rad, compute_strain=True, is_degree=False)
    for u, v in zip(a, b):
        assert rel_err(u, v).max() < 1e-9


def test_topleft_and_center_describe_the_same_fault(okada, device):
    """Moving the reference point from the top-left corner to the centre and
    compensating in x_fault/y_fault/depth must give identical displacements --
    the only test that would catch a sign error in the fault_origin offsets."""
    coords = make_coords(device, with_z=True)
    p = make_params(device)
    L, W = p["length"], p["width"]
    ss, cs = jnp.sin(jnp.deg2rad(p["strike"])), jnp.cos(jnp.deg2rad(p["strike"]))
    sd, cd = jnp.sin(jnp.deg2rad(p["dip"])), jnp.cos(jnp.deg2rad(p["dip"]))
    dxl, dyl = L / 2, -(W / 2) * cd
    pc = dict(p, x_fault=p["x_fault"] + dxl * ss - dyl * cs,
              y_fault=p["y_fault"] + dxl * cs + dyl * ss,
              depth=p["depth"] + (W / 2) * sd)
    a = okada.compute(coords, p, compute_strain=False, fault_origin="topleft")
    b = okada.compute(coords, pc, compute_strain=False, fault_origin="center")
    for u, v in zip(a, b):
        err = rel_err(u, v).max()
        assert err < 1e-9, f"topleft/center describe different faults: {err:.3e}"


def test_zero_slip_gives_zero_displacement(okada, device):
    out = okada.compute(make_coords(device, with_z=True),
                        make_params(device, slip=0.0), compute_strain=True)
    for t in out:
        assert bool(jnp.all(t == 0.0))


def test_displacement_scales_linearly_with_slip(okada, device):
    coords = make_coords(device, with_z=True)
    a = okada.compute(coords, make_params(device, slip=1.0), compute_strain=True)
    b = okada.compute(coords, make_params(device, slip=3.5), compute_strain=True)
    for u, v in zip(a, b):
        assert rel_err(u * 3.5, v).max() < 1e-12


def test_wrapper_agrees_with_okadatorch(okada, wrapper_golden, device):
    """The two ports implement the same conventions -- rotation, fault_origin
    offsets, the free-surface relations -- so on well-conditioned geometries
    OkadaJAX must reproduce OkadaTorch's wrapper output."""
    for name, (coords, params, kw) in wrapper_configurations(device).items():
        got = np.asarray(jnp.stack(okada.compute(coords, params, **kw)))
        err = rel_err(got, wrapper_golden[name]).max()
        assert err < 1e-11, f"{name}: differs from OkadaTorch by {err:.3e}"


def test_surrogate_gradient_does_not_perturb_the_value():
    """A-3 uses a straight-through estimator near a vertical fault.  It must be
    bit-exact in the forward direction -- including after XLA's algebraic
    simplifier has seen `s - stop_gradient(s)`."""
    from OkadaJAX.utils import _surrogate_grad
    value = jnp.asarray([1.0, 2.5, -3.75e-8])
    surrogate = jnp.asarray([1e9, -4e7, 6.0])
    use = jnp.asarray([True, True, False])
    for f in (lambda s: _surrogate_grad(use, value, s),
              jax.jit(lambda s: _surrogate_grad(use, value, s))):
        assert bool(jnp.all(f(surrogate) == value)), "the estimator changed the value"
        g = jax.grad(lambda s: jnp.sum(f(s)))(surrogate)
        assert g.tolist() == [1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# D-1: the answer must not depend on the unit the caller happens to use
# ---------------------------------------------------------------------------
# With the old absolute EPS = 1e-6, three of these were classified differently
# in the two unit systems -- e.g. 1e-7 km is below EPS in km (singular) but
# 1e-4 m is above it in metres (regular).
@pytest.mark.parametrize("offset,unit_ratio", [
    pytest.param(o, u, id=f"offset{o:g}-{'km->m' if u > 1 else 'km->Mm'}")
    for o in (1.0, 1e-3, 1e-5, 1e-7, 1e-9) for u in (1000.0, 0.001)])
# Below about 1e-8 of the fault length the offset is not representable to any
# useful precision, so only the classification -- the part D-1 is about -- is
# required to agree there.
def test_result_is_invariant_under_a_change_of_length_unit(okada, device, unit_ratio,
                                                           offset):
    """Every length scaled by the same factor must scale the displacement by that
    factor and nothing else.  Absolute thresholds (`|xi| < 1e-6`) made the same
    physical station singular in kilometres and regular in metres."""
    def run(scale):
        coords = {"x": put([(20.0 + offset) * scale, 5.0 * scale], device),
                  "y": put([0.0, 3.0 * scale], device)}
        params = make_params(device, x_fault=0.0, y_fault=0.0, depth=0.0,
                             length=20.0 * scale, width=10.0 * scale,
                             slip=2.0 * scale, strike=90.0, dip=90.0, rake=90.0)
        out, iret = okada.compute(coords, params, compute_strain=False, return_iret=True)
        return jnp.stack(out) / scale, iret

    a, ia = run(1.0)
    b, ib = run(unit_ratio)
    assert ia.tolist() == ib.tolist(), \
        f"the singular/normal classification changed with the unit: {ia} vs {ib}"
    if offset / 20.0 > 1e-8:
        assert bool(jnp.allclose(a, b, rtol=1e-9, atol=1e-14)), \
            f"the displacement changed with the unit: max|diff| = {jnp.abs(a - b).max():.3e}"


def test_thresholds_follow_the_working_precision():
    """D-1.  A fixed float64-sized tolerance would sit below the float32
    resolution and the guards would never fire; tying it to the machine epsilon
    of the dtype in use keeps them meaningful in both."""
    from OkadaJAX.utils import _rel_eps
    eps64 = _rel_eps(jnp.zeros((), dtype=jnp.float64))
    eps32 = _rel_eps(jnp.zeros((), dtype=jnp.float32))
    assert eps64 < 1e-10 < eps32
    assert eps32 > jnp.finfo(jnp.float32).eps
