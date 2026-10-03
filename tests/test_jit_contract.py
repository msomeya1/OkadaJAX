"""What JAX needs from the kernels, so that the fast path stays fast.

The JAX counterpart of OkadaTorch's test_performance_contract.py.  There the
contract was "no graph breaks, no host syncs"; here it is:

* everything can be wrapped in ``jax.jit`` -- no value-dependent Python
  branches (A-1), which also raise TracerBoolConversionError under NUTS;
* calling a kernel again with the same shapes compiles nothing new (J-1).
"""
import jax
import jax.numpy as jnp
import pytest
from jax import monitoring

from OkadaJAX import DC3D, DC3D0, SPOINT, SRECTF

from conftest import make_coords, make_params, put


_COMPILES = []
monitoring.register_event_duration_secs_listener(
    lambda name, *a, **kw: _COMPILES.append(name)
    if name == "/jax/core/compile/backend_compile_duration" else None)


def _compiles_on_repeat(fn, n=2):
    """XLA compilations triggered by `n` further calls, after one warm-up."""
    jax.block_until_ready(fn())
    _COMPILES.clear()
    for _ in range(n):
        jax.block_until_ready(fn())
    return len(_COMPILES)


T = lambda v: jnp.asarray(v, dtype=jnp.float64)
X, Y, Z = T([1.0, 7.0, -3.0]), T([2.0, -4.0, 5.0]), T([-2.0, -1.0, -3.0])
s = T(0.7071067811865476)
KERNELS = {
    "SPOINT": lambda: SPOINT(0.5, X, Y, T(3.0), s, s, 1.0, 0.5, 0.2),
    "SRECTF": lambda: SRECTF(0.5, X, Y, T(3.0), T(10.0), T(5.0), s, s, 1.0, 0.5, 0.2),
    "DC3D0": lambda: DC3D0(2 / 3, X, Y, Z, T(3.0), T(45.0), 1.0, 0.5, 0.2, 0.1)[0],
    "DC3D": lambda: DC3D(2 / 3, X, Y, Z, T(3.0), T(45.0), -2., 2., -1., 1., 1.0, 0.5, 0.2)[0],
}


@pytest.mark.parametrize("name", list(KERNELS))
def test_repeated_eager_calls_compile_nothing_new(name):
    assert _compiles_on_repeat(KERNELS[name]) == 0


@pytest.mark.parametrize("with_z,rect", [
    pytest.param(False, True, id="surface-rect"),
    pytest.param(False, False, id="surface-point"),
    pytest.param(True, True, id="depth-rect"),
    pytest.param(True, False, id="depth-point")])
def test_unjitted_value_and_grad_compiles_nothing_new(okada, with_z, rect):
    """The loop of notebook 6: an optimiser calling value_and_grad without jit.
    With J-1 this took 1.5 s per step and leaked until the process died."""
    cpu = jax.devices("cpu")[0]
    coords, params = make_coords(cpu, with_z=with_z), make_params(cpu, rect=rect)
    vg = jax.value_and_grad(lambda p: jnp.sum(jnp.stack(
        okada.compute(coords, p, compute_strain=False))))
    assert _compiles_on_repeat(lambda: vg(params)) == 0


@pytest.mark.parametrize("with_z,rect", [
    pytest.param(False, True, id="surface-rect"),
    pytest.param(False, False, id="surface-point"),
    pytest.param(True, True, id="depth-rect"),
    pytest.param(True, False, id="depth-point")])
def test_compute_can_be_jitted_and_traces_once(okada, with_z, rect):
    """jit must trace once and then serve new parameter values from the cache --
    which is what NumPyro's NUTS relies on."""
    cpu = jax.devices("cpu")[0]
    coords = make_coords(cpu, with_z=with_z)
    traces = []

    @jax.jit
    def f(p):
        traces.append(1)
        return jnp.stack(okada.compute(coords, p, compute_strain=True))

    a = f(make_params(cpu, rect=rect))
    b = f(make_params(cpu, rect=rect, dip=61.0, rake=0.0, slip=0.0))
    assert len(traces) == 1
    eager = jnp.stack(okada.compute(coords, make_params(cpu, rect=rect), compute_strain=True))
    assert bool(jnp.allclose(a, eager, rtol=1e-12, atol=1e-15))


@pytest.mark.parametrize("with_z", [False, True], ids=["surface", "depth"])
def test_jitted_gradient_matches_eager_gradient(okada, with_z):
    cpu = jax.devices("cpu")[0]
    coords, params = make_coords(cpu, with_z=with_z), make_params(cpu)
    f = lambda p: jnp.sum(jnp.stack(okada.compute(coords, p, compute_strain=False)))
    eager, jitted = jax.grad(f)(params), jax.jit(jax.grad(f))(params)
    for k in params:
        assert bool(jnp.allclose(eager[k], jitted[k], rtol=1e-10, atol=1e-15)), k
