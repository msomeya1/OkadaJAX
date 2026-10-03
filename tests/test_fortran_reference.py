"""Bit-for-bit faithfulness to the original Okada (1985, 1992) FORTRAN.

The reference outputs are frozen in ``golden/fortran_reference.npz`` -- the same
file the OkadaTorch suite uses -- so neither gfortran nor the FORTRAN sources are
needed (and the latter must not be distributed).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from OkadaJAX import DC3D, DC3D0, SPOINT, SRECTF

from conftest import put, rel_err

# max(|got - ref|) / max(|ref|, 1) must stay below this.  The observed worst
# case is 3.6e-12 (SRECTF, regular); 1e-9 leaves headroom for library and
# compiler differences without letting a real regression through.
TOL = 1e-9

CASES = [(name, tag) for name in ("SPOINT", "SRECTF", "DC3D0", "DC3D")
         for tag in ("regular", "degenerate")]
NARGS = {"SPOINT": 9, "SRECTF": 11, "DC3D0": 10, "DC3D": 13}


def _kernel(name, strain):
    """Each kernel compiled once and reused for every row."""
    if name == "SPOINT":
        return jax.jit(lambda *a: SPOINT(*a, compute_strain=strain))
    if name == "SRECTF":
        return jax.jit(lambda *a: SRECTF(*a, compute_strain=strain))
    if name == "DC3D0":
        return jax.jit(lambda *a: DC3D0(*a, compute_strain=strain, is_degree=True)[0])
    return jax.jit(lambda *a: DC3D(*a, compute_strain=strain, is_degree=True)[0])


def _iret_kernel(name):
    fn = DC3D0 if name == "DC3D0" else DC3D
    return jax.jit(lambda *a: fn(*a, compute_strain=False, is_degree=True)[1])


def _run(fn, name, row, device):
    a = [put(float(v), device) for v in row[:NARGS[name]]]
    out = fn(*a)
    return np.atleast_1d(np.asarray(out, dtype=float)) if hasattr(out, "shape") \
        else np.array([float(t) for t in out])


@pytest.mark.parametrize("name,tag", CASES, ids=[f"{n}-{t}" for n, t in CASES])
def test_matches_original_fortran(fortran_golden, name, tag):
    cpu = jax.devices("cpu")[0]
    key = f"{name}_{tag}"
    args, ref, iret = (fortran_golden[key + s] for s in ("_args", "_out", "_iret"))
    usable = (iret == 0) & np.isfinite(ref).all(axis=1)
    assert usable.sum() > 0.8 * len(args), "golden data looks degenerate"

    fn = _kernel(name, True)
    got = np.stack([_run(fn, name, args[i], cpu) for i in np.flatnonzero(usable)])
    # Check finiteness separately: every comparison against NaN is False, so a
    # NaN result would otherwise slip through the tolerance test unnoticed.
    bad = np.flatnonzero(~np.isfinite(got).all(axis=1))
    if bad.size:
        i = np.flatnonzero(usable)[bad[0]]
        pytest.fail(f"{key}: {bad.size} case(s) non-finite where the FORTRAN is finite; "
                    f"first at index {i}\n  args = {args[i]}\n  ref = {ref[i]}\n  got = {got[bad[0]]}")
    err = rel_err(got, ref[usable])
    assert err.max() < TOL, f"{key}: max rel.err {err.max():.3e}"


def _eager(name, strain):
    if name == "SPOINT":
        return lambda *a: SPOINT(*a, compute_strain=strain)
    if name == "SRECTF":
        return lambda *a: SRECTF(*a, compute_strain=strain)
    fn = DC3D0 if name == "DC3D0" else DC3D
    return lambda *a: fn(*a, compute_strain=strain, is_degree=True)[0]


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("name,tag", CASES, ids=[f"{n}-{t}" for n, t in CASES])
def test_displacement_only_matches_full_call(fortran_golden, name, tag, mode):
    """``compute_strain=False`` must compute the same three components.

    Exactly, when run eagerly.  Under jit the two settings are two different XLA
    programs, and fusion may round differently -- measured up to 1.3e-15 (DC3D)
    -- so there the check is to rounding only.
    """
    cpu = jax.devices("cpu")[0]
    make = _kernel if mode == "jit" else _eager
    full_fn, disp_fn = make(name, True), make(name, False)
    for row in fortran_golden[f"{name}_{tag}_args"][:40]:
        full, disp = _run(full_fn, name, row, cpu)[:3], _run(disp_fn, name, row, cpu)
        assert disp.size == 3
        ok = np.isfinite(full) & np.isfinite(disp)
        assert np.array_equal(np.isfinite(full), np.isfinite(disp)), \
            f"{name}: compute_strain=True/False disagree on which components are finite"
        if mode == "eager":
            assert np.array_equal(full[ok], disp[ok]), \
                f"{name}: compute_strain=True/False differ: {full} vs {disp}"
        else:
            assert rel_err(disp[ok], full[ok]).max(initial=0) < 1e-13, \
                f"{name}: compute_strain=True/False differ beyond rounding: {full} vs {disp}"


def test_iret_flags_agree_with_fortran(fortran_golden):
    """The 1992 routines must raise IRET on exactly the cases the original does."""
    cpu = jax.devices("cpu")[0]
    for name in ("DC3D0", "DC3D"):
        fn = _iret_kernel(name)
        for tag in ("regular", "degenerate"):
            key = f"{name}_{tag}"
            got = np.array([int(_run(fn, name, row, cpu)[0])
                            for row in fortran_golden[key + "_args"]])
            ref = fortran_golden[key + "_iret"]
            mism = np.flatnonzero(got != ref)
            assert mism.size == 0, (f"{key}: IRET differs on {mism.size} cases, first at "
                                    f"{mism[0]} (fortran={ref[mism[0]]}, jax={got[mism[0]]})")


@pytest.mark.parametrize("name", ["SPOINT", "SRECTF", "DC3D0", "DC3D"])
def test_gpu_agrees_with_cpu(fortran_golden, name):
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        pytest.skip("needs a GPU")
    gpu, cpu = (gpus[1] if len(gpus) > 1 else gpus[0]), jax.devices("cpu")[0]
    fn = _kernel(name, True)
    for row in fortran_golden[f"{name}_regular_args"][:50]:
        err = rel_err(_run(fn, name, row, gpu), _run(fn, name, row, cpu)).max()
        # measured worst case 1.8e-12 (DC3D): XLA's GPU transcendentals round
        # differently from the CPU ones -- far inside the FORTRAN tolerance
        assert err < 1e-11, f"{name}: GPU and CPU differ by {err:.3e}"
