"""Autodiff correctness: AD gradients must agree with central finite differences.

This is the only class of test that can detect A-1 and A-3 -- both return a
*plausible* value (zero) rather than raising or producing NaN, and both leave
the forward pass untouched, so every other test passes happily.

Tests marked ``xfail(strict=True)`` document known bugs.  When a bug is fixed
the test XPASSes, which pytest reports as a failure -- the signal to delete
the marker.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import PARAM_NAMES, make_coords, make_params, put

FD_TOL = 1e-5



def _functional(okada, coords, params, **kw):
    """One number carrying all three components at every station."""
    ux, uy, uz = okada.compute(coords, params, compute_strain=False, **kw)
    return jnp.sum(ux + 2.0 * uy + 3.0 * uz)


def _ad_grad(okada, coords, params, name, **kw):
    f = lambda v: _functional(okada, coords, dict(params, **{name: v}), **kw)
    return float(jax.grad(f)(params[name]))


def _fd_grad(okada, coords, params, name, **kw):
    v0 = float(params[name])
    h = 1e-5 * max(abs(v0), 1.0)
    out = [float(_functional(okada, coords,
                             dict(params, **{name: jnp.full_like(params[name], v0 + s * h)}), **kw))
           for s in (+1.0, -1.0)]
    return (out[0] - out[1]) / (2.0 * h)


def _assert_close(ad, fd, what):
    scale = max(abs(fd), abs(ad), 1e-12)
    err = abs(ad - fd) / scale
    assert err < FD_TOL, f"{what}: AD={ad!r} vs FD={fd!r} (rel. err {err:.3e})"


@pytest.mark.parametrize("name", PARAM_NAMES)
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_parameter_gradient_matches_finite_difference(okada, device, name, with_z):
    coords, params = make_coords(device, with_z=with_z), make_params(device)
    _assert_close(_ad_grad(okada, coords, params, name),
                  _fd_grad(okada, coords, params, name), f"d/d({name})")


@pytest.mark.parametrize("name", ["depth", "dip", "slip", "strike"])
def test_wrapper_gradient_method_matches_grad(okada, device, name):
    """OkadaWrapper.gradient (jacfwd) must agree with reverse-mode jax.grad."""
    coords, params = make_coords(device, with_z=True), make_params(device)
    g = okada.gradient(coords, params, arg=name, compute_strain=False)
    lumped = float(jnp.sum(g[0] + 2.0 * g[1] + 3.0 * g[2]))
    _assert_close(lumped, _ad_grad(okada, coords, params, name), f"gradient(arg={name})")


@pytest.mark.parametrize("name", ["depth", "slip"])
def test_hessian_matches_finite_difference_of_gradient(okada, device, name):
    coords, params = make_coords(device, with_z=True), make_params(device)
    h = okada.hessian(coords, params, arg1=name, arg2=name, compute_strain=False)
    ad2 = float(jnp.sum(h[0] + 2.0 * h[1] + 3.0 * h[2]))
    v0 = float(params[name]); step = 1e-4 * max(abs(v0), 1.0)
    vals = [_ad_grad(okada, coords, dict(params, **{name: jnp.full_like(params[name], v0 + s * step)}), name)
            for s in (+1.0, -1.0)]
    _assert_close(ad2, (vals[0] - vals[1]) / (2.0 * step), f"d2/d({name})2")


# ---------------------------------------------------------------------------
# Known-bad cases: A-1 and A-3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rake", [180.0, -180.0, 90.0, -90.0])
def test_gradient_wrt_rake_at_the_cardinal_rakes(okada, rake):
    """Only rake=0 actually triggers A-1: sin(180 deg) is -1.2e-16 and cos(90 deg)
    6.1e-17, not zero, so those blocks are never skipped."""
    cpu = jax.devices("cpu")[0]
    coords, params = make_coords(cpu, with_z=True), make_params(cpu, rake=rake)
    _assert_close(_ad_grad(okada, coords, params, "rake"),
                  _fd_grad(okada, coords, params, "rake"), f"d/d(rake) at rake={rake}")


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_gradient_wrt_rake_at_pure_strike_slip(okada, with_z):
    """A-1 regression.  rake=0 makes u_dip exactly 0.0; while the contributions
    were guarded by `if DISLn != 0.0` that skipped the dip-slip block -- the
    dominant term in d/d(rake) there."""
    cpu = jax.devices("cpu")[0]
    coords, params = make_coords(cpu, with_z=with_z), make_params(cpu, rake=0.0)
    _assert_close(_ad_grad(okada, coords, params, "rake"),
                  _fd_grad(okada, coords, params, "rake"), "d/d(rake) at rake=0")


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_gradient_wrt_slip_at_zero_slip(okada, with_z):
    cpu = jax.devices("cpu")[0]
    coords, params = make_coords(cpu, with_z=with_z), make_params(cpu, slip=0.0)
    _assert_close(_ad_grad(okada, coords, params, "slip"),
                  _fd_grad(okada, coords, params, "slip"), "d/d(slip) at slip=0")


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_gradient_wrt_dip_at_vertical_fault(okada, with_z):
    """d(u)/d(dip) is continuous through dip=90, so the value at 90 must be close
    to the value just below it (a finite difference would cross the snap band)."""
    cpu = jax.devices("cpu")[0]
    coords = make_coords(cpu, with_z=with_z)
    near = _ad_grad(okada, coords, make_params(cpu, dip=89.5), "dip")
    at90 = _ad_grad(okada, coords, make_params(cpu, dip=90.0), "dip")
    assert abs(at90 - near) < 0.2 * max(abs(near), 1e-12), \
        f"d/d(dip): {near!r} at dip=89.5 but {at90!r} at dip=90.0"


@pytest.mark.parametrize("with_z,dip", [
    pytest.param(z, d, id=f"{d}-{'depth' if z else 'surface'}")
    for d in (89.999, 89.9999, 89.99999, 90.0, 90.00001) for z in (False, True)])
def test_gradient_wrt_dip_is_smooth_through_vertical(okada, with_z, dip):
    """A-3 regression, the wider half.  The inclined formulae compute d/d(dip) as a
    cancellation of two 1/cos(dip) terms, which used to lose all precision well
    before the snap band (48% wrong at 89.999, 15800% and sign-flipped at 89.9999)."""
    cpu = jax.devices("cpu")[0]
    coords = make_coords(cpu, with_z=with_z)
    reference = _ad_grad(okada, coords, make_params(cpu, dip=89.5), "dip")
    got = _ad_grad(okada, coords, make_params(cpu, dip=dip), "dip")
    assert abs(got - reference) < 0.1 * abs(reference), \
        f"d/d(dip) at dip={dip} is {got!r} but {reference!r} at 89.5"


def test_a_vertical_fault_can_be_optimised(okada):
    """The practical symptom of A-3: an inversion initialised at exactly dip=90
    cannot move, because the gradient there is exactly zero."""
    cpu = jax.devices("cpu")[0]
    coords = make_coords(cpu, n=8)
    truth = make_params(cpu, dip=72.0)
    obs = okada.compute(coords, truth, compute_strain=False)

    def loss(dip):
        u = okada.compute(coords, dict(truth, dip=dip), compute_strain=False)
        return sum(jnp.sum((a - b) ** 2) for a, b in zip(u, obs))

    step = jax.jit(jax.value_and_grad(loss))
    dip, m, v, lr = put(90.0, cpu), 0.0, 0.0, 0.5
    for t in range(1, 201):                     # Adam, written out
        _, g = step(dip)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        dip = dip - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    assert abs(float(dip) - 72.0) < 0.1, f"starting from dip=90 the optimiser reached {float(dip)}"



# ---------------------------------------------------------------------------
# Stations on the extension of a fault edge or of the fault plane
# ---------------------------------------------------------------------------
# One of XI, ET, Q is (snapped to) exactly zero there.  Okada sets the
# arctangents arctan(XI*ET/(Q*R)) and arctan(.../(XI*...)) to zero where their
# denominator vanishes; the derivative there is finite but used to come out as
# zero, and the snapping itself used to discard the gradient of XI, ET and Q.
# A grid station hits these lines whenever the fault corners lie on the grid.
# The fault: strike north, dipping east, top edge from (0, 0) to (0, 10) at depth 2.
EDGE_FAULT = dict(x_fault=0.0, y_fault=0.0, depth=2.0, length=10.0, width=5.0,
                  strike=0.0, dip=50.0, rake=30.0, slip=1.0)
TAN_DIP = float(np.tan(np.radians(50.0)))
EDGE_CASES = [
    # id,                                 dip,  x,             y,    z
    ("xi=0-start-surface",                50.0, -3.7,          0.0,  None),
    ("xi=0-end-surface",                  50.0, -3.7,          10.0, None),
    ("q=0-vertical-surface",              90.0, 0.0,           15.0, None),
    ("xi=0-start-depth",                  50.0, -3.7,          0.0,  -3.0),
    ("xi=0-end-depth",                    50.0, -3.7,          10.0, -3.0),
    ("eta=0-top-edge-depth",              50.0, -TAN_DIP,      4.3,  -3.0),
    ("q=0-below-bottom-edge-depth",       50.0, 6.0 / TAN_DIP, 4.3,  -8.0),
    ("eta=0-vertical-depth",              90.0, -3.7,          4.3,  -2.0),
    ("q=0-vertical-depth",                90.0, 0.0,           15.0, -3.0),
]


@pytest.mark.parametrize("dip,x,y,z", [pytest.param(*c[1:], id=c[0]) for c in EDGE_CASES])
def test_gradient_on_the_extension_of_a_fault_edge(okada, dip, x, y, z):
    cpu = jax.devices("cpu")[0]
    coords = {"x": put([x], cpu), "y": put([y], cpu)}
    if z is not None:
        coords["z"] = put([z], cpu)
    params = {k: put(v, cpu) for k, v in dict(EDGE_FAULT, dip=dip).items()}
    vertical = dip == 90.0
    for name in PARAM_NAMES:
        if vertical and name == "dip":
            continue        # a finite difference would cross the snap band at dip=90
        ad = _ad_grad(okada, coords, params, name)
        fd = _fd_grad(okada, coords, params, name)
        if vertical:
            # Loose: the known near-vertical limitation, see
            # test_non_dip_gradients_on_a_vertical_fault.  The bug this guards
            # against is a gradient of zero, or wrong by tens of percent.
            assert abs(ad - fd) < 1e-2 * max(abs(fd), abs(ad), 1e-12), \
                f"d/d({name}): AD={ad!r} vs FD={fd!r}"
        else:
            _assert_close(ad, fd, f"d/d({name})")

@pytest.mark.parametrize("name", PARAM_NAMES)
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_every_source_parameter_can_be_vmapped(okada, name, with_z):
    """Batching over source parameters is what makes multi-fault models cheap."""
    cpu = jax.devices("cpu")[0]
    coords, params = make_coords(cpu, with_z=with_z), make_params(cpu)
    v0 = float(params[name])
    vals = put([v0, v0 * 1.01 + 0.1, v0 * 0.99 - 0.1], cpu)
    out = jax.vmap(lambda v: okada.compute(coords, dict(params, **{name: v}),
                                           compute_strain=False)[0])(vals)
    for i in range(3):
        one = okada.compute(coords, dict(params, **{name: vals[i]}), compute_strain=False)[0]
        assert bool(jnp.allclose(out[i], one, rtol=1e-12, atol=1e-15))


@pytest.mark.xfail(strict=True, reason="known limitation near a vertical fault "
                   "(README, Remark 4): gradients good to about 1e-3")
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_non_dip_gradients_on_a_vertical_fault(okada, with_z):
    """Near dip=90 the A-, B- and C-terms take their whole derivative from the
    inclined formula at a dip rotated DIP_GRAD_FLOOR away, so every parameter is
    off by about 1e-3 (surface) or 1e-4 (depth), up to 1% next to the plane of
    the fault.  Extrapolating from two rotations would bring this to about 1e-6
    but costs 10-30% on every evaluation, so it was not adopted."""
    cpu = jax.devices("cpu")[0]
    coords = make_coords(cpu, with_z=with_z)
    params = make_params(cpu, dip=90.0)
    for name in PARAM_NAMES:
        if name != "dip":
            _assert_close(_ad_grad(okada, coords, params, name),
                          _fd_grad(okada, coords, params, name), f"d/d({name}) at dip=90")


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_gradients_inside_the_surrogate_zone_continue_those_outside(okada, with_z):
    """Below |cos(dip)| = DIP_GRAD_FLOOR every derivative comes from the surrogate.
    The gradient is a smooth function of cos(dip), so a polynomial fitted to the
    gradients just outside the zone, where the inclined formulae are accurate,
    predicts those inside it.  A finite difference cannot check this: the values
    there lose digits to the same 1/cos(dip) cancellation.  For the same reason
    1e-6 < |cos(dip)| < 1e-4 is left out: the values themselves, and with them
    d/d(rake) and d/d(slip), are good to only about eps/cos(dip)**2 there.

    The tolerance is the known limitation (README, Remark 4), about 1e-3; this
    guards against it getting worse."""
    cpu = jax.devices("cpu")[0]
    coords = make_coords(cpu, with_z=with_z)
    grad = jax.jit(jax.grad(lambda p: _functional(okada, coords, p)))

    def gradients(cos_dip):
        g = grad(make_params(cpu, dip=float(np.degrees(np.arccos(cos_dip)))))
        return np.array([float(g[k]) for k in PARAM_NAMES])

    nodes = np.array([2e-3, 4e-3, 6e-3, 8e-3, 1e-2])
    coef = np.polyfit(nodes, np.stack([gradients(c) for c in nodes]), len(nodes) - 1)
    for cos_dip in (9e-4, 1e-4, 0.0):
        got, ref = gradients(cos_dip), np.polyval(coef, cos_dip)
        for name, a, b in zip(PARAM_NAMES, got, ref):
            err = abs(a - b) / max(abs(a), abs(b), 1e-12)
            assert err < 2e-3, f"d/d({name}) at cos(dip)={cos_dip}: {a!r} vs {b!r} (rel. err {err:.3e})"
