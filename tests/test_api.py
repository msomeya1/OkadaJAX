"""The public contract of OkadaWrapper: shapes, dtypes, devices, validation.

Most of these are cheap guards that would have caught the C-*, D-* and J-3
issues in DEBUG_NOTES.md.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import finite, make_coords, make_params, put



# ---------------------------------------------------------------------------
# shapes and containers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(7,), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_output_shape_follows_coordinate_shape(okada, device, shape, with_z):
    rng = np.random.default_rng(0)
    coords = {"x": put(rng.uniform(-50, 50, shape), device),
              "y": put(rng.uniform(-50, 50, shape), device)}
    if with_z:
        coords["z"] = put(-rng.uniform(1, 11, shape), device)
    for strain, n in ((False, 3), (True, 12)):
        out = okada.compute(coords, make_params(device), compute_strain=strain)
        assert len(out) == n and all(t.shape == shape for t in out)


def test_gradient_and_hessian_preserve_coordinate_shape(okada, device):
    coords, params = make_coords(device, with_z=True, n=4), make_params(device)
    for g in okada.gradient(coords, params, arg="x", compute_strain=False):
        assert g.shape == coords["x"].shape
    for h in okada.hessian(coords, params, arg1="x", arg2="y", compute_strain=False):
        assert h.shape == coords["x"].shape


@pytest.mark.parametrize("where,key", [("coords", "Z"), ("coords", "unused"),
                                       ("params", "widht"), ("params", "openning"),
                                       ("params", "dpeth")])
def test_unknown_keys_raise(okada, device, where, key):
    """'Z' silently selected the surface formulation, 'widht' a point source."""
    coords, params = make_coords(device), make_params(device)
    (coords if where == "coords" else params)[key] = put(1.0, device)
    with pytest.raises(ValueError, match="unrecognized"):
        okada.compute(coords, params, compute_strain=False)


def test_inputs_are_not_mutated(okada, device):
    coords, params = make_coords(device), make_params(device)
    before_c, before_p = dict(coords), dict(params)
    okada.compute(coords, params, compute_strain=True)
    assert coords.keys() == before_c.keys() and params.keys() == before_p.keys()
    assert all(coords[k] is before_c[k] for k in coords)
    assert all(params[k] is before_p[k] for k in params)


# ---------------------------------------------------------------------------
# device / dtype
# ---------------------------------------------------------------------------
def test_output_stays_on_the_input_device(okada, device):
    out = okada.compute(make_coords(device, with_z=True), make_params(device),
                        compute_strain=True)
    assert all(t.devices() == {device} for t in out)


def test_uniform_dtype_is_preserved(okada, device):
    out = okada.compute(make_coords(device, with_z=True), make_params(device),
                        compute_strain=True)
    assert all(t.dtype == jnp.float64 for t in out)


def test_float32_works_but_warns(okada, device):
    """float32 is JAX's default unless jax_enable_x64 is set.  It is not
    rejected, but it costs accuracy and must not happen silently."""
    coords = make_coords(device, dtype=jnp.float32, with_z=True)
    params = make_params(device, dtype=jnp.float32)
    with pytest.warns(UserWarning, match="float32"):
        out = okada.compute(coords, params, compute_strain=True)
    assert all(t.dtype == jnp.float32 for t in out)


@pytest.mark.parametrize("coord_dtype,param_dtype",
                         [(jnp.float32, jnp.float64), (jnp.float64, jnp.float32)])
def test_mixed_dtypes_are_promoted(okada, device, coord_dtype, param_dtype):
    out = okada.compute(make_coords(device, dtype=coord_dtype, with_z=True),
                        make_params(device, dtype=param_dtype), compute_strain=True)
    assert all(t.dtype == jnp.float64 for t in out)


def test_promotion_gives_the_float64_answer(okada, device):
    coords = make_coords(device, with_z=True)
    p32 = make_params(device, dtype=jnp.float32)
    mixed = okada.compute(coords, p32, compute_strain=True)
    ref = okada.compute(coords, {k: v.astype(jnp.float64) for k, v in p32.items()},
                        compute_strain=True)
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(mixed, ref))


def test_promotion_keeps_the_gradient(okada, device):
    coords, params = make_coords(device), make_params(device, dtype=jnp.float32)
    g = jax.grad(lambda s: jnp.sum(okada.compute(coords, dict(params, slip=s),
                                                 compute_strain=False)[0]))(params["slip"])
    assert g.dtype == jnp.float32 and bool(jnp.isfinite(g)) and float(g) != 0.0


def test_mixed_devices_are_rejected(okada):
    """D-3 needs no code: JAX refuses arrays committed to different devices."""
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("needs a GPU")
    gpu = gpus[1] if len(gpus) > 1 else gpus[0]
    with pytest.raises(ValueError, match="device"):
        okada.compute(make_coords(gpu), make_params(jax.devices("cpu")[0]),
                      compute_strain=False)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_missing_required_keys_raise(okada, device):
    """ValueError rather than AssertionError: `python -O` strips assertions."""
    coords, params = make_coords(device), make_params(device)
    with pytest.raises(ValueError, match="missing"):
        okada.compute({"x": coords["x"]}, params, compute_strain=False)
    with pytest.raises(ValueError, match="missing"):
        okada.compute(coords, {k: v for k, v in params.items() if k != "dip"},
                      compute_strain=False)


@pytest.mark.parametrize("method", ["compute", "gradient", "hessian"])
def test_all_three_methods_validate_identically(okada, device, method):
    params = {k: v for k, v in make_params(device).items() if k != "slip"}
    kw = {"gradient": dict(arg="depth"), "hessian": dict(arg1="depth", arg2="dip")}
    with pytest.raises(ValueError, match="missing"):
        getattr(okada, method)(make_coords(device), params, compute_strain=False,
                               **kw.get(method, {}))


def test_mismatched_coordinate_shapes_raise(okada, device):
    coords = make_coords(device)
    coords["y"] = coords["y"][:-1]
    with pytest.raises(ValueError, match="same shape"):
        okada.compute(coords, make_params(device), compute_strain=False)


def test_invalid_gradient_arg_raises(okada, device):
    with pytest.raises(ValueError):
        okada.gradient(make_coords(device), make_params(device),
                       arg="not_a_parameter", compute_strain=False)


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("origin", ["TopLeft", "centre", "", "top-left"])
def test_invalid_fault_origin_raises_valueerror(okada, device, with_z, origin):
    """The DC3D branch raised ValueError but the SRECTF branch fell through to
    `return out` with `out` unbound."""
    with pytest.raises(ValueError, match="fault_origin"):
        okada.compute(make_coords(device, with_z=with_z), make_params(device),
                      compute_strain=False, fault_origin=origin)


def test_invalid_fault_origin_is_caught_even_for_a_point_source(okada, device):
    with pytest.raises(ValueError, match="fault_origin"):
        okada.compute(make_coords(device), make_params(device, rect=False),
                      compute_strain=False, fault_origin="TopLeft")


@pytest.mark.parametrize("drop", ["length", "width"])
def test_partial_rectangle_parameters_raise(okada, device, drop):
    params = {k: v for k, v in make_params(device).items() if k != drop}
    with pytest.raises(ValueError, match="rectangular fault"):
        okada.compute(make_coords(device), params, compute_strain=False)


def test_a_point_source_still_needs_neither_length_nor_width(okada, device):
    out = okada.compute(make_coords(device), make_params(device, rect=False),
                        compute_strain=False)
    assert len(out) == 3 and finite(out)


def test_hessian_reports_the_actual_problem(okada, device):
    coords, params = make_coords(device), make_params(device)   # no "z"
    with pytest.raises(ValueError, match="not a key"):
        okada.hessian(coords, params, arg1="x", arg2="z", compute_strain=False)
    with pytest.raises(ValueError, match="both be"):
        okada.hessian(coords, params, arg1="x", arg2="depth", compute_strain=False)


@pytest.mark.parametrize("nu", [0.5, -1.0, 1.5, 1.0, 100.0])
def test_invalid_poisson_ratio_raises(okada, device, nu):
    with pytest.raises(ValueError, match="Poisson"):
        okada.compute(make_coords(device), make_params(device), compute_strain=True, nu=nu)


@pytest.mark.parametrize("nu", [0.0, 0.25, 0.3, 0.49])
def test_valid_poisson_ratios_are_accepted(okada, device, nu):
    assert finite(okada.compute(make_coords(device), make_params(device),
                                compute_strain=True, nu=nu))


def test_positive_z_is_flagged_rather_than_rejected(okada, device):
    """z > 0 is reported through IRET, not by raising: raising needs a
    data-dependent Python branch, which cannot be traced by jax.jit."""
    coords = make_coords(device, with_z=True)
    coords["z"] = jnp.abs(coords["z"])
    out, iret = okada.compute(coords, make_params(device), compute_strain=False,
                              return_iret=True)
    assert bool(jnp.all(iret == 2)) and all(bool(jnp.all(t == 0.0)) for t in out)


# ---------------------------------------------------------------------------
# Python numbers (C-6 is already fine in JAX, except for coordinates)
# ---------------------------------------------------------------------------
def test_scalar_python_floats_are_accepted(okada, device):
    coords = make_coords(device)
    params = {k: float(v) for k, v in make_params(device).items()}
    out = okada.compute(coords, params, compute_strain=False)
    assert all(t.devices() == {device} and t.dtype == coords["x"].dtype for t in out)


def _device_params():
    cpu = jax.devices("cpu")[0]
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        return [pytest.param(cpu, id="cpu")]
    gpu = gpus[1] if len(gpus) > 1 else gpus[0]
    # With a GPU present, arithmetic on Python floats alone (the trig in
    # setup()) would run on JAX's *default* device unless they are put where
    # the coordinates are -- which is what the CPU case checks.
    return [pytest.param(cpu, id="cpu"), pytest.param(gpu, id="gpu")]


@pytest.mark.parametrize("device", _device_params())
def test_python_floats_give_the_same_answer_as_arrays(okada, device):
    """Bit-identical, which also means computed on the same device: a Python
    float has no device, so it must be put where the coordinates are."""
    coords, arrays = make_coords(device, with_z=True), make_params(device)
    a = okada.compute(coords, arrays, compute_strain=True)
    b = okada.compute(coords, {k: float(v) for k, v in arrays.items()}, compute_strain=True)
    assert all(bool(jnp.array_equal(u, v)) for u, v in zip(a, b))


@pytest.mark.parametrize("method", ["compute", "gradient", "hessian"])
def test_all_three_methods_accept_python_floats(okada, device, method):
    params = {k: float(v) for k, v in make_params(device).items()}
    kw = {"gradient": dict(arg="x"), "hessian": dict(arg1="x", arg2="y")}
    assert finite(getattr(okada, method)(make_coords(device), params, compute_strain=False,
                                         **kw.get(method, {})))


def test_scalar_coordinates_are_accepted(okada, device):
    out = okada.compute({"x": 12.0, "y": -34.0}, make_params(device), compute_strain=False)
    assert all(t.ndim == 0 for t in out)


# ---------------------------------------------------------------------------
# C-5: the tensile and isotropic source components
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
def test_opening_is_accepted_and_changes_the_field(okada, device, with_z, rect):
    coords, params = make_coords(device, with_z=with_z), make_params(device, rect=rect)
    base = okada.compute(coords, params, compute_strain=True)
    opened = okada.compute(coords, dict(params, opening=put(2.0, device)), compute_strain=True)
    assert finite(opened)
    assert not bool(jnp.allclose(jnp.stack(base), jnp.stack(opened)))


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
@pytest.mark.parametrize("rect", [False, True], ids=["point", "rect"])
def test_opening_defaults_to_zero(okada, device, with_z, rect):
    coords, params = make_coords(device, with_z=with_z), make_params(device, rect=rect)
    a = okada.compute(coords, params, compute_strain=True)
    b = okada.compute(coords, dict(params, opening=put(0.0, device)), compute_strain=True)
    assert all(bool(jnp.array_equal(u, v)) for u, v in zip(a, b))


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_source_components_superpose(okada, device, with_z):
    coords = make_coords(device, with_z=with_z)
    shear = make_params(device, slip=3.0, rake=70.0)
    o = put(2.0, device)
    mixed = okada.compute(coords, dict(shear, opening=o), compute_strain=True)
    only_shear = okada.compute(coords, dict(shear, opening=put(0.0, device)), compute_strain=True)
    only_open = okada.compute(coords, dict(shear, slip=put(0.0, device), opening=o),
                              compute_strain=True)
    for m, a, b in zip(mixed, only_shear, only_open):
        assert bool(jnp.allclose(m, a + b, rtol=1e-10, atol=1e-14))


def test_opening_is_differentiable(okada, device):
    coords, params = make_coords(device, with_z=True), make_params(device)
    g = jax.grad(lambda o: jnp.sum(jnp.stack(okada.compute(coords, dict(params, opening=o),
                                                           compute_strain=False))))(put(1.5, device))
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def test_opening_agrees_between_the_1985_and_1992_paths(okada, device):
    coords = make_coords(device)
    params = dict(make_params(device), opening=put(2.0, device))
    a = okada.compute(coords, params, compute_strain=True)
    b = okada.compute(dict(coords, z=jnp.zeros_like(coords["x"])), params, compute_strain=True)
    assert all(bool(jnp.allclose(u, v, rtol=1e-9, atol=1e-12)) for u, v in zip(a, b))


def test_inflation_works_for_a_point_source_at_depth(okada, device):
    coords, params = make_coords(device, with_z=True), make_params(device, rect=False)
    f = lambda i: jnp.sum(jnp.stack(okada.compute(coords, dict(params, inflation=i),
                                                  compute_strain=True)))
    assert bool(jnp.isfinite(f(put(1.0, device))))
    assert float(jax.grad(f)(put(1.0, device))) != 0.0


def test_inflation_is_rejected_where_no_kernel_supports_it(okada, device):
    infl = put(1.0, device)
    with pytest.raises(ValueError, match="point source"):
        okada.compute(make_coords(device, with_z=True),
                      dict(make_params(device, rect=True), inflation=infl), compute_strain=False)
    with pytest.raises(ValueError, match="requires 'z'"):
        okada.compute(make_coords(device),
                      dict(make_params(device, rect=False), inflation=infl), compute_strain=False)
