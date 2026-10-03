"""Behaviour at and near the singularities of the Okada solution.

1. *Values*.  The original FORTRAN detects singular geometries and returns zeros
   with ``IRET != 0``.  OkadaJAX used to raise the flag but keep computing, so
   callers saw NaN (B-1, B-2); flagged stations now return zero.
2. *Gradients*.  ``jnp.where`` evaluates both branches, so the discarded branch
   can contribute ``0 * inf = NaN`` to the gradient at perfectly ordinary
   stations (A-2).  The forward pass is correct, so nothing looks wrong until an
   optimiser silently stops moving.
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OkadaJAX import DC3D, SPOINT, SRECTF

from conftest import PARAM_NAMES, finite, make_coords, make_params, put



def _grad_sum(okada, coords, params, name):
    f = lambda v: jnp.sum(jnp.stack(okada.compute(coords, dict(params, **{name: v}),
                                                  compute_strain=False)))
    return jax.grad(f)(params[name])


# ---------------------------------------------------------------------------
# ordinary geometries must be clean, in value *and* in gradient
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_regular_geometry_gives_finite_values(okada, device, with_z):
    assert finite(okada.compute(make_coords(device, with_z=with_z),
                                make_params(device), compute_strain=True))


@pytest.mark.parametrize("name", PARAM_NAMES)
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_regular_geometry_gives_finite_gradients(okada, device, name, with_z):
    g = _grad_sum(okada, make_coords(device, with_z=with_z), make_params(device), name)
    assert bool(jnp.isfinite(g)), f"d/d({name}) is not finite for a regular geometry"


# ---------------------------------------------------------------------------
# A-2: stations that trip a singularity *guard* without being singular
# ---------------------------------------------------------------------------
GUARD_TRIPPING_GEOMETRIES = {
    # xi == 0 exactly with a vertical fault: the XI==0 guard and the CD==0
    # branch are active at once
    "vertical_fault_xi0": (
        dict(x=[0.0, 4.0], y=[8.0, 15.0], z=[-2.0, -2.0]),
        dict(x_fault=0.0, y_fault=0.0, depth=6.0, strike=0.0, dip=90.0,
             rake=45.0, slip=3.0, length=25.0, width=10.0)),
    # station on the along-strike axis of an inclined fault (XI==0 guard in A5/AI4)
    "inclined_fault_xi0": (
        dict(x=[0.0, 6.0], y=[12.0, 20.0]),
        dict(x_fault=0.0, y_fault=0.0, depth=4.0, strike=0.0, dip=50.0,
             rake=70.0, slip=2.0, length=30.0, width=12.0)),
    # far down-dip, where the R+eta guard (KET / RET==0) engages
    "down_dip_extension": (
        dict(x=[2.0, 5.0], y=[-40.0, -60.0], z=[-3.0, -3.0]),
        dict(x_fault=0.0, y_fault=0.0, depth=5.0, strike=0.0, dip=30.0,
             rake=90.0, slip=1.0, length=20.0, width=15.0)),
}


def _geometry(name, device):
    c, p = GUARD_TRIPPING_GEOMETRIES[name]
    return {k: put(v, device) for k, v in c.items()}, {k: put(v, device) for k, v in p.items()}


@pytest.mark.parametrize("geometry,name", [
    pytest.param(g, n, id=f"{g}-{n}")
    for g in sorted(GUARD_TRIPPING_GEOMETRIES) for n in PARAM_NAMES])
def test_gradient_is_finite_where_a_singularity_guard_engages(okada, device, geometry, name):
    coords, params = _geometry(geometry, device)
    assert finite(okada.compute(coords, params, compute_strain=False)), \
        "precondition: this geometry must have a finite forward pass"
    g = _grad_sum(okada, coords, params, name)
    assert bool(jnp.isfinite(g)), f"d/d({name}) is NaN/Inf at {geometry} though the value is finite"


@pytest.mark.parametrize("geometry", sorted(GUARD_TRIPPING_GEOMETRIES))
def test_coordinate_gradient_is_finite_where_a_guard_engages(okada, device, geometry):
    coords, params = _geometry(geometry, device)
    assert finite(okada.gradient(coords, params, arg="x", compute_strain=False))


# ---------------------------------------------------------------------------
# B-1 / B-2: singular stations return zero, like the original
# ---------------------------------------------------------------------------
def _on_fault_edge(device):
    """Stations exactly on the top edge of a fault that reaches z=0."""
    coords = {"x": put([0.0, 2.0], device), "y": put([0.0, 0.0], device),
              "z": put([0.0, 0.0], device)}
    params = make_params(device, x_fault=0.0, y_fault=0.0, depth=0.0, strike=0.0,
                         dip=45.0, rake=90.0, slip=1.0, length=10.0, width=5.0)
    return coords, params


def test_fault_edge_station_is_usable(okada):
    """A station exactly on the fault edge is flagged, zeroed and -- because it
    is fed a dummy geometry rather than masked afterwards -- differentiable."""
    coords, params = _on_fault_edge(jax.devices("cpu")[0])
    assert finite(okada.compute(coords, params, compute_strain=False))
    for name in ("depth", "dip", "slip", "x_fault"):
        assert bool(jnp.isfinite(_grad_sum(okada, coords, params, name)))


def test_singular_stations_return_zero_like_the_original():
    T = lambda v: jnp.asarray(v, dtype=jnp.float64)
    x, y = T([0.0, 0.0, 5.0]), T([0.0, 1e-9, 0.0])
    out, iret = DC3D(2.0 / 3.0, x, y, jnp.zeros_like(x), T(0.0), T(45.0),
                     0.0, 10.0, -5.0, 0.0, 1.0, 0.0, 0.0, compute_strain=False, is_degree=True)
    sing = iret != 0
    assert bool(sing.any()), "test geometry no longer triggers IRET"
    for t in out:
        assert bool(jnp.all(t[sing] == 0.0)), "singular stations must return 0, not NaN"


def test_positive_z_returns_zero_like_the_original():
    T = lambda v: jnp.asarray(v, dtype=jnp.float64)
    out, iret = DC3D(2.0 / 3.0, T([1.0]), T([1.0]), T([2.0]), T(5.0), T(45.0),
                     0.0, 10.0, -5.0, 0.0, 1.0, 0.0, 0.0, compute_strain=False, is_degree=True)
    assert bool((iret == 2).all())
    for t in out:
        assert bool(jnp.all(t == 0.0))


def test_rrx_fallback_is_zero_like_the_1992_kernel():
    """RRX = 1/(R(R+xi)) used to fall back to a constant of order 1e6, whose
    meaning changes with the unit.  DCCON2 sets the identical quantity to zero."""
    import OkadaJAX.utils as utils
    code = "\n".join(l for l in inspect.getsource(utils._SRECTG).split("\n")
                     if not l.strip().startswith("#"))
    assert "1.0e6" not in code and "1e6" not in code


def test_rrx_fallback_is_unobservable():
    """Where the RRX fallback fires, Q = Y = D = 0, so its value never matters."""
    T = lambda v: jnp.asarray(v, dtype=jnp.float64)
    out = SRECTF(0.5, T([-8.0]), T([0.0]), T(0.0), T(20.0), T(10.0),
                 T(1.0), T(0.0), 1.0, 0.5, 0.3, compute_strain=True)
    assert finite(out)


def test_up_dip_extension_of_a_vertical_fault_is_finite():
    """Vertical fault, Q = 0, xi = 0, eta = -2: R = 2 and d = -2, so R + d = 0.
    Only the dip-slip component is non-zero, so the physically correct answer is
    the finite dip-slip contribution alone."""
    T = lambda v: jnp.asarray(v, dtype=jnp.float64)
    out = SRECTF(0.5, T(0.0), T(0.0), T(3.0), T(10.0), T(5.0), T(1.0), T(0.0),
                 0.0, 1.0, 0.0, compute_strain=True)
    assert finite(out)


# ---------------------------------------------------------------------------
# B-1: reporting
# ---------------------------------------------------------------------------
def _mixed_stations(device):
    """Three stations: on a fault corner, above the surface, and ordinary."""
    coords = {"x": put([0.0, 1.0, 7.0], device), "y": put([0.0, 1.0, 9.0], device),
              "z": put([0.0, 2.0, -2.0], device)}
    params = make_params(device, x_fault=0.0, y_fault=0.0, depth=0.0, strike=0.0,
                         dip=45.0, rake=90.0, slip=1.0, length=10.0, width=5.0)
    return coords, params


def test_return_iret_is_off_by_default(okada, device):
    out = okada.compute(*_mixed_stations(device), compute_strain=False)
    assert isinstance(out, list) and len(out) == 3


@pytest.mark.parametrize("strain", [False, True])
def test_return_iret_flags_and_zeroes_bad_stations(okada, device, strain):
    coords, params = _mixed_stations(device)
    out, iret = okada.compute(coords, params, compute_strain=strain, return_iret=True)
    assert iret.shape == coords["x"].shape
    assert iret.tolist() == [1, 2, 0], "expected singular / above-surface / normal"
    bad = iret != 0
    for t in out:
        assert bool(jnp.all(t[bad] == 0.0)) and finite([t])


def test_iret_distinguishes_singular_from_genuinely_zero(okada, device):
    coords, params = _mixed_stations(device)
    _, iret_singular = okada.compute(coords, params, compute_strain=False, return_iret=True)
    out, iret_zero = okada.compute(coords, dict(params, slip=jnp.zeros_like(params["slip"])),
                                   compute_strain=False, return_iret=True)
    assert bool(jnp.all(jnp.stack(out) == 0.0))
    assert iret_zero.tolist() == iret_singular.tolist(), "IRET describes the geometry"


def test_surface_and_depth_paths_agree_on_which_stations_are_singular(okada, device):
    coords, params = _mixed_stations(device)
    coords["z"] = jnp.zeros_like(coords["x"])
    _, iret_depth = okada.compute(coords, params, compute_strain=False, return_iret=True)
    _, iret_surface = okada.compute({"x": coords["x"], "y": coords["y"]}, params,
                                    compute_strain=False, return_iret=True)
    assert iret_surface.tolist() == iret_depth.tolist()


@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
def test_low_level_1985_routines_keep_their_old_signature(rect):
    T = lambda v: jnp.asarray(v, dtype=jnp.float64)
    s = T(0.7071067811865476)
    out = (SRECTF(0.5, T([1.0]), T([2.0]), T(3.0), T(10.0), T(5.0), s, s, 1.0, 0.0, 0.0,
                  compute_strain=False) if rect else
           SPOINT(0.5, T([1.0]), T([2.0]), T(3.0), s, s, 1.0, 0.0, 0.0, compute_strain=False))
    assert isinstance(out, list) and len(out) == 3


@pytest.mark.parametrize("name", PARAM_NAMES)
def test_gradients_stay_finite_when_a_station_is_singular(okada, device, name):
    """Zeroing the output is not enough: jnp.where hands zero back to the
    discarded branch, which then evaluates 0 * d(nan)/dx."""
    coords, params = _mixed_stations(device)
    assert bool(jnp.isfinite(_grad_sum(okada, coords, params, name)))
