# OkadaJAX

[![tests](https://github.com/msomeya1/OkadaJAX/actions/workflows/test.yml/badge.svg)](https://github.com/msomeya1/OkadaJAX/actions/workflows/test.yml)

`OkadaJAX` provides JAX implementations of FORTRAN subroutines that calculate displacements and strains (spatial derivatives of displacements) due to a point source or a rectangular fault (Okada 1985, 1992).
It is the JAX counterpart of [OkadaTorch](https://github.com/msomeya1/OkadaTorch) and shares its interface.

**Features**
- **The whole code is differentiable**: the gradient with respect to the input can be easily computed using automatic differentiation (AD), allowing for flexible gradient-based optimization.
- **No for-loop over observation stations and sources**: vectorization allows rapid calculation for multiple stations and sources.
- **Compatible with `jax.jit`, `jax.vmap` and `jax.grad`**, and so with libraries built on them such as Optax and NumPyro.



**References**
- Okada, Y. (1985). Surface deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 75(4), 1135-1154.
https://doi.org/10.1785/BSSA0750041135
- Okada, Y. (1992). Internal deformation due to shear and tensile faults in a half-space. Bulletin of the seismological society of America, 82(2), 1018-1040.
https://doi.org/10.1785/BSSA0820021018
- [Program to calculate deformation due to a fault model DC3D0 / DC3D](https://www.bosai.go.jp/information/dc3d_e.html) (NIED website) 


Programs published in this repository are different from the original programs published on the NIED website.




If you use `OkadaJAX` in your study, please consider citing the following preprint, which describes the method shared by `OkadaTorch` and `OkadaJAX`.
- Masayoshi Someya, Taisuke Yamada, Tomohisa Okazaki. OkadaTorch: A Differentiable Programming of Okada Model to Calculate Displacements and Strains from Fault Parameters, arXiv preprint (2025). https://arxiv.org/abs/2507.17126


If you find any bugs when using `OkadaJAX`, please let us know.


## Install

Execute:
```shell
git clone https://github.com/msomeya1/OkadaJAX.git
cd OkadaJAX
pip install .
```

To run the example notebooks as well, execute:
```shell
pip install ".[examples]"
```

This installs the CPU build of JAX. For a GPU, install JAX for your CUDA version first, following the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

Confirmed to work with JAX 0.6.0 (CPU and CUDA 12) on Python 3.13.



## Usage

Okada (1985, 1992) provides four subroutines:
- `SPOINT` (Okada 1985): Calculate displacements and strains at the surface ($z=0$) created by a point source.
- `SRECTF` (Okada 1985): Calculate displacements and strains at the surface ($z=0$) created by a rectangular fault.
- `DC3D0` (Okada 1992): Same as `SPOINT`, but under the surface ($z\leq0$).
- `DC3D` (Okada 1992): Same as `SRECTF`, but under the surface ($z\leq0$).


||At Surface (Okada 1985)|Under Surface (Okada 1992)|
|-|-|-|
|Point Source|`SPOINT`|`DC3D0`|
|Rectangular Fault|`SRECTF`|`DC3D`|


We have ported all of these subroutines into JAX, together with a convenient wrapper class, `OkadaWrapper`.

```python
import jax
jax.config.update("jax_enable_x64", True)   # see Remark 1
import jax.numpy as jnp
from OkadaJAX import OkadaWrapper

okada = OkadaWrapper()
X, Y = jnp.meshgrid(jnp.linspace(-100.0, 100.0, 51), jnp.linspace(-100.0, 100.0, 51))
coords = {"x": X, "y": Y}                    # add "z" (<= 0) for the Okada (1992) formulae
params = {"x_fault": 0.0, "y_fault": 0.0, "depth": 5.0, "length": 50.0, "width": 20.0,
          "strike": 30.0, "dip": 45.0, "rake": 90.0, "slip": 2.0}

ux, uy, uz = okada.compute(coords, params, compute_strain=False)
d_ux_d_dip = okada.gradient(coords, params, arg="dip", compute_strain=False)[0]
```

The functions and `OkadaWrapper` have the same interface as in `OkadaTorch`, with JAX arrays in place of tensors, so its documentation applies here too:
- `SPOINT` and `SRECTF`: [OkadaTorch docs/Okada1985.md](https://github.com/msomeya1/OkadaTorch/blob/main/docs/Okada1985.md)
- `DC3D0` and `DC3D`: [OkadaTorch docs/Okada1992.md](https://github.com/msomeya1/OkadaTorch/blob/main/docs/Okada1992.md)
- `OkadaWrapper`: [OkadaTorch docs/OkadaWrapper.md](https://github.com/msomeya1/OkadaTorch/blob/main/docs/OkadaWrapper.md)

The example notebooks `1_*.ipynb` to `7_*.ipynb` cover the same ground for `OkadaJAX`.





## Remark 1: float64 must be enabled

**JAX uses `float32` unless `float64` is enabled**, and without it a `float64` numpy array is silently truncated to `float32`. Enable it once, at the start of the program, before any array is created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

`OkadaJAX` does not do this for you, because it changes the behaviour of every JAX computation in the process. A warning is displayed if `OkadaJAX` runs in `float32` or lower precision; see Remark 4 for what that costs.

For Hamiltonian Monte Carlo the difference is not a matter of a few digits. In `7_OkadaWrapper_Bayesian_demo.ipynb`, NUTS run in `float32` adapted its step size down to about `1e-7` and hit the maximum tree depth (1023 leapfrog steps) on every sample: rounding errors in the gradient and in the energy prevented it from integrating a usable trajectory. 10,000 samples took two hours and gave an effective sample size of only 20 to 30. In `float64` the same run takes about five minutes, with a step size of about `6e-2`, roughly 20 leapfrog steps per sample, and an effective sample size in the thousands.

`OkadaWrapper` accepts plain Python numbers and numpy arrays as well as JAX arrays; they are cast to one floating dtype and placed on the same device as the rest of the inputs.


## Remark 2: Vectorization

Stations are vectorized directly: pass `x,y(,z)` as arrays of any shape (they must all have the same shape), 
and the returned displacements and strains have the same shape.

Sources are not vectorized directly: every source parameter must be a scalar. 
Multiple sources are handled by `jax.vmap`, and since the Okada solution 
is linear with respect to source, summation over the batch dimension provides 
the multi-source solution:

```python
keys = ["x_fault", "y_fault", "depth", "length", "width", "strike", "dip", "rake", "slip"]

def one_fault(values):
    return okada.compute(coords, dict(zip(keys, values)), compute_strain=False)[2]

uz = jax.vmap(one_fault)(faults).sum(0)     # faults: an (n_faults, 9) array
```

`3_OkadaWrapper_compute.ipynb` works this through with the slip distribution of the 2011 Tohoku-oki earthquake.


## Remark 3: Coordinate System and Notation



The coordinate system used in functions `SPOINT`, `SRECTF`, `DC3D0` and `DC3D` is defined so that 
- the x-axis is parallel to the strike direction of the fault, 
- the z-axis is vertically upward, 
- and the y-axis is determined so that the entire system is right-handed.

However, `OkadaWrapper` uses a Cartesian coordinate system in which east is x, north is y, and up is z.




Also, the original FORTRAN subroutines and their JAX implementations use uppercase variables (e.g., `UX`), while the OkadaWrapper uses lowercase variables (e.g., `ux`), but there is no particular difference between them (**except for the coordinate system difference noted above**). 
For example,
- `U1`, `UX`, and `ux` all represent the x component of the displacement.
- `U12`, `UXY`, and `uxy` all represent the x component of the displacement differentiated by y. 

> [!NOTE]
> `Uij` or `uij` means $\frac{\partial U_i}{\partial x_j}$ or $\frac{\partial u_i}{\partial x_j}$, respectively ($i,j=x,y,z$).
> In other words, the first index represents the component of displacement, and the second one represents which variable to differentiate.


## Remark 4: Precision

In `float32`, the displacements differ from `float64` by up to around 0.2% on a large fault and the gradients are good to three or four digits, which may be acceptable alongside neural network models. `float64` is recommended when precise gradients are required (e.g., gradient-based optimization), for small faults far from the stations, and for shallow faults (see below).


### Small faults far from the stations

The displacement due to a rectangular fault is a sum of four terms, one for each corner of the fault. When the fault is small compared with its distance to the stations, these terms nearly cancel, and the `float32` error grows as the fault gets smaller. The largest `float32` error, relative to the largest displacement, for a fault dipping 20° and stations on a 600 km × 600 km grid:

| Fault (length × width, depth) | Error |
|-|-|
| 100 km × 50 km, 5 km | 0.04% |
| 5 km × 5 km, 20 km | 0.1% |
| 1 km × 1 km, 30 km | 3% |

**Use `float64` for slip inversions with many small patches.**


### Nearly vertical faults

Within about 0.06° of vertical ($|\cos(\text{dip})| < 10^{-3}$), the formulae for an inclined fault lose their derivative to a cancellation between two terms in $1/\cos(\text{dip})$, and the formulae for a vertical fault have no dip dependence to differentiate. In this range the gradient is taken from the inclined-fault formulae at a dip rotated $10^{-3}$ rad away from vertical. **All gradients, not only the one with respect to `dip`, are then accurate to only about 0.1% at the surface and 0.01% at depth, and to about 1% right next to the plane of the fault.** The displacements and strains themselves are not affected by this, but they lose digits of their own close to vertical (around $10^{-4}$ at dip = 89.9999°), as in the original FORTRAN programs.

This rarely matters in practice: HMC and NUTS still sample the correct posterior (an inaccurate gradient only lowers the acceptance rate), and gradient-based optimization is affected only when the optimum lies in this range. Workarounds we examined:

- Keep `dip` out of this range, e.g., with an upper bound of 89.9° on the prior or the optimizer, if the fault need not be exactly vertical.
- Evaluate the inclined-fault formulae at two rotations, $\delta$ and $2\delta$, and use the gradient of $2S(\delta) - S(2\delta)$ (Richardson extrapolation). This makes the gradients accurate to about $10^{-6}$, but it was not adopted because it costs 10-30% more on every evaluation, whatever the dip.
- Take the derivatives other than the one with respect to `dip` from the formula that computes the value. They become exact at dip = 90°, but are less accurate than the above at, e.g., 89.9999°.


### Faults that reach the surface

When `depth = 0` (`fault_origin="topleft"`), i.e., the upper edge of the fault reaches the surface, the displacement at points on the fault trace is finite, but the gradient is not (typically `NaN`). To avoid issues with gradients, we set the output at such points to exactly zero and report them with `IRET = 1` (`okada.compute(..., return_iret=True)` returns `(out, iret)`).

In practice, this issue occurs within a finite-width zone around the trace. When computing displacements on a dense grid (e.g., seafloor displacement for tsunami simulations), some grid points may fall within this zone. The width of the zone depends on machine epsilon (and hence on the dtype). For `float64`, the width is on the order of nanometers, so this problem is practically negligible. For `float32`, however, the width is on the order of meters, meaning that some grid points may return zero displacement despite the true value being finite. Therefore, **`float64` is recommended for faults that may reach the surface.**



## Remark 5: Performance Hint

**Wrap the computation in [`jax.jit`](https://docs.jax.dev/en/latest/jit-compilation.html).** Without it, JAX dispatches the several hundred operations of the kernel one at a time; with it, they are compiled into a single program. For example, one step of the gradient-based optimization in `6_OkadaWrapper_optimization.ipynb` (625 stations) takes about 260 ms without `jax.jit` and about 2 ms with it:

```python
loss_and_grad = jax.jit(jax.value_and_grad(loss))
```

The first call compiles, which takes a few seconds; later calls with the same array shapes reuse the compiled program. NumPyro's NUTS compiles the model itself, so nothing extra is needed there.


## License

[MIT LICENSE](LICENSE). 

This covers the JAX implementation in this repository and not the original NIED programs.

## Version history

### 0.2.1

- **Gradient fixes**
  - Gradients were wrong at stations on the extension of a fault edge or of the fault plane, where one of the fault coordinates ($\xi$, $\eta$, $q$) is exactly zero. This happens for a whole row of grid stations when the fault corners lie on the grid. Two causes, both fixed:
    - Arctangents of the form $\arctan(a/b)$ are set to zero where $b = 0$, as in the original FORTRAN, but their derivative there is finite. It was returned as zero; it is now computed through $\arctan(u) = \mathrm{sign}(u)\,\pi/2 - \arctan(1/u)$, an idea taken from [geodef](https://github.com/ericlindsey/geodef).
    - Coordinates within rounding error of zero are snapped to exactly zero, and the snapping discarded their gradient. It now passes the gradient through.
- **Documentation**
  - Remark 4: `float32` errors for small faults far from the stations, and the known limitation of the gradients for nearly vertical faults.
  - The `float32` warning now mentions small faults.
- **Added tests** for the gradients at stations on the extension of a fault edge, and for the gradients of nearly vertical faults.

### 0.2.0

First working release, debugged and refactored together with `OkadaTorch` 0.2.0. Earlier development versions did not work properly.

- **Fixes specific to JAX**
  - `lax.cond` was handed functions defined anew on every call, so every call outside `jax.jit` compiled again and leaked the compiled code until the process ran out of memory mappings. Both formulae are now evaluated and one is selected with `jnp.where`.
  - `D / (ET2 + Q2)` returned `NaN` where the original FORTRAN is finite (a guard that used to hide the `0/0` had been removed).
- **Gradient issues** 
  - Removed the remaining `if DISLn != 0.0` / `if POTn != 0.0` from the Okada (1992) kernels. They dropped the gradient at `rake=0` or `slip=0`, and made those kernels impossible to `jax.jit` or `jax.vmap`.
  - Fixed `NaN` gradients caused by `jnp.where`, whose unselected branch is still evaluated: denominators are now made safe before the division.
  - Fixed incorrect gradients `d/d(dip)` near and at `dip = 90`, using a surrogate gradient from the inclined-fault formula evaluated at `dip ≈ 90`.
- **Singular station treatment** 
  - At singular stations, the output is exactly zero, and `compute(..., return_iret=True)` reports which ones they are. Dummy coordinates are assigned to those stations so that the gradients stay finite.
- **New features**
  - `params["opening"]` (tensile) and `params["inflation"]` (isotropic; point source with `z` only) are now supported in `OkadaWrapper`.
  - `jax.vmap` over every source parameter is supported.
- **Input validation** 
  - Raises `ValueError` instead of `AssertionError`.
  - Unrecognized keys are rejected rather than ignored.
  - Inputs are cast to one floating dtype, and Python numbers and numpy arrays are placed on the device of the other inputs.
  - A warning is displayed in `float32`.
- **Units** 
  - Tolerances are relative, so the same fault gives the same answer whether you work in meters or kilometers.
- **Added tests**, including a comparison with the original FORTRAN output and with `OkadaTorch`.
