# diff_rheo

A differentiable rheology solver for fitting constitutive models to shear
rheometer data using gradient-based optimisation in JAX.

`diff_rheo` is built around five ideas:

1. **Differentiable forward simulation** – constitutive models are implemented
   as JAX-compatible `equinox.Module` subclasses, so gradients flow through the
   ODE solver via automatic differentiation.
2. **Modular experiment protocol** – a `VirtualRheometer` combines a model,
   an experiment type (strain-rate or stress controlled), and an ODE solver into
   a reusable object.
3. **MAP fitting and variational inference** – supports both point-estimate
   (MAP/MSE) and full-distribution (ELBO) parameter fitting.
4. **Model selection** – Bayesian Information Criterion (BIC) helpers allow
   quantitative comparison of models with different numbers of parameters.
5. **Information geometry** – the Fisher Information Matrix yields Cramér-Rao
   parameter uncertainties, sloppy-direction analysis, and optimal experiment
   design (`diff_rheo._information`).

---

## Table of Contents

- [Installation](#installation)
- [How It Works](#how-it-works)
  - [Parameter Types](#parameter-types)
  - [Constitutive Models](#constitutive-models)
  - [Virtual Rheometer](#virtual-rheometer)
  - [Fitting](#fitting)
  - [Model Selection](#model-selection)
- [Usage Examples](#usage-examples)
  - [MAP Fitting](#map-fitting)
  - [Variational Inference](#variational-inference)
- [Architecture Reference](#architecture-reference)
- [Unfinished / Needs Improvement](#unfinished--needs-improvement)

---

## Installation

**Requirements:** Python ≥ 3.13.2, [uv](https://docs.astral.sh/uv/)

```bash
# Clone the repository
git clone <repo-url>
cd diff_rheo

# Install all dependencies (uv reads pyproject.toml + uv.lock)
uv sync

# Run scripts / notebooks inside the managed environment
uv run python my_script.py
uv run jupyter notebook
```

For development (includes test dependencies):

```bash
uv sync --group dev
```

Core dependencies are resolved automatically from `uv.lock`:

| Package | Purpose |
|---------|---------|
| `jax` / `jaxlib` | Automatic differentiation and JIT compilation |
| `equinox` | JAX-compatible pytree modules (models, parameters) |
| `diffrax` | Adaptive ODE integration with JAX |
| `optax` | Gradient-based optimisers (Adam, etc.) |
| `numpy` / `scipy` | Preprocessing and numerical utilities |
| `tqdm` | Training progress bars |
| `colorama` | Colour-coded results display |

---

## How It Works

### Parameter Types

Parameters are the building blocks of constitutive models.  The hierarchy is:

```
AbstractParameter
├── Parameter          – unconstrained float; optimised directly
├── StaticParameter    – fixed constant; excluded from gradients
├── LogParameter       – stores log(value); ensures positivity during optimisation
└── TanhParameter      – stores arctanh-transformed value; constrains to (min, max)

AbstractRandomParameter   (for variational inference)
├── GaussianParameter      – variational Gaussian q(θ) = N(μ, σ²) for a real-valued parameter
└── LogGaussianParameter   – variational log-normal q(θ) for a positive parameter
```

All parameter objects live as leaves of the Equinox pytree.  `LogParameter` is
the standard choice for physically positive quantities (viscosities, relaxation
times) because it makes the optimisation landscape smoother.

### Constitutive Models

All models are `equinox.Module` subclasses that inherit from
`AbstractConstitutiveModel`.  Two sub-hierarchies exist:

- **`AbstractGeneralizedNewtonianModel`** – algebraic models; no ODE required.
- **`AbstractViscoelasticModel`** – differential models; require an ODE solver.

#### Generalised Newtonian models

| Class | Parameters |
|-------|-----------|
| `Newtonian` | `viscosity` |
| `CarreauYasuda` | `zero_shear_viscosity`, `infinite_shear_viscosity`, `relaxation_time`, `power_index`, `yasuda_parameter` |
| `PowerLaw` | `consistency_index`, `power_index` |

#### Viscoelastic models

| Class | Parameters | Notes |
|-------|-----------|-------|
| `OldroydB` | 3 | Upper-Convected Maxwell + solvent |
| `GeneralizedOldroydB` | 3 | Parameterises the nonlinear F function |
| `Giesekus` | 4 | Quadratic stress term; shear-thinning |
| `LinearPTT` | 4 | Linear PTT stretch function |
| `ExponentialPTT` | 4 | Exponential PTT stretch function |
| `GeneralizedPTT` | 6 | Mittag-Leffler stretch function (fractional) |
| `FENECR` | 4 | Finite-extensibility with constant retardation |
| `FENEP` | 4 | Finite-extensibility with polymer pressure |
| `WhiteMetzner` | 4 | Rate-dependent viscosity and relaxation time |
| `XPomPom` | 5 | Extended Pom-Pom model for branched polymers |

All viscoelastic models work in **simple shear**, using a compact 6-vector
state `[σ11, σ22, σ33, σ12, σ13, σ23]` for the polymer extra-stress tensor.

### Virtual Rheometer

The `VirtualRheometer` combines model + experiment type + ODE solver:

```python
from diff_rheo import VirtualRheometer
from diff_rheo._solver import DiffraxSolver

rheometer = VirtualRheometer.setup(
    model,
    experiment_type="strain_rate_response",  # or "shear_stress_response"
    solver=DiffraxSolver(),
)
```

**Experiment types:**

- `"strain_rate_response"` – prescribe velocity gradient L(t), measure stress σ(t).
- `"shear_stress_response"` – prescribe shear stress σ₁₂(t), measure strain γ(t).

The rheometer is reused across training steps; the model parameters change but
the rheometer configuration (protocol, solver) stays fixed.

### Fitting

#### MAP Fitting (`fit_model_to_experimental_data`)

Minimises the mean-squared error between simulated and observed time-series
data using the Adam optimiser.

```python
from diff_rheo import fit_model_to_experimental_data, FittingConfig

config = FittingConfig(num_epochs=2000, learning_rate=1e-3, verbose=True)
fitted_model = fit_model_to_experimental_data(model, rheometer, data, config)
```

#### Variational Inference (`fit_variational_inference`)

Optimises the ELBO (Evidence Lower BOund):

    ELBO = E_q[log p(data | θ)] - D_KL(q(θ) ‖ p(θ))

where `q(θ)` is a factored Gaussian variational posterior.  The model must
use `GaussianParameter` or `LogGaussianParameter` attributes.

```python
from diff_rheo import fit_variational_inference, FittingConfig
import jax

config = FittingConfig(
    num_epochs=3000,
    learning_rate=5e-4,
    ensemble_size=16,
    key=jax.random.PRNGKey(0),
    verbose=True,
)
fitted_model = fit_variational_inference(model, rheometer, data, config)
```

### Model Selection

#### BIC from L2 loss (`calculate_bic_from_l2`)

Fast BIC estimate; assumes i.i.d. Gaussian residuals:

    BIC = k · log(N) - 2 · log L̂

where `k` is the number of trainable floating-point values and `log L̂` is
the maximised Gaussian log-likelihood.

```python
from diff_rheo import calculate_bic_from_l2

bic = calculate_bic_from_l2(fitted_model, rheometer, reference_data)
```

#### Ensemble-based BIC (`model_bic`)

Uses the ensemble log-likelihood from stochastic forward simulations.  More
accurate but slower; requires `GaussianParameter` / `LogGaussianParameter`
attributes.

---

## Usage Examples

### MAP Fitting

Fit an Oldroyd-B model to synthetic steady-state shear data:

```python
import jax
import jax.numpy as jnp
from diff_rheo import VirtualRheometer, fit_model_to_experimental_data
from diff_rheo._solver import DiffraxSolver
from diff_rheo._data_types import ShearStrainRateData, BatchedData, FittingConfig
from diff_rheo._forcing import VelocityGradient
from diff_rheo.models import OldroydB
from diff_rheo.parameters import LogParameter

# 1. Define the "true" model to generate synthetic data
true_model = OldroydB(
    polymer_viscosity=LogParameter(2.0),
    relaxation_time=LogParameter(1.0),
    solvent_viscosity=LogParameter(0.1),
)

# 2. Build the virtual rheometer
solver = DiffraxSolver()
rheometer = VirtualRheometer.setup(true_model, "strain_rate_response", solver)

# 3. Generate synthetic data (step shear)
shear_rate = 1.0
L = VelocityGradient.from_components(grad_u_12=lambda t: shear_rate)
time = jnp.linspace(0, 10, 200)
initial_condition = jnp.zeros(6)
sim = rheometer.run_experiment(true_model, L, time, initial_condition)
sigma_12 = sim.data[:, 0, 1]  # extract shear stress

data = ShearStrainRateData(
    time=time,
    data=sigma_12,
    forcing_data=jnp.full_like(time, shear_rate),
    initial_condition=initial_condition,
)
batched = BatchedData([data])

# 4. Define model to fit with perturbed initial parameters
fit_model = OldroydB(
    polymer_viscosity=LogParameter(1.0),   # wrong initial guess
    relaxation_time=LogParameter(0.5),
    solvent_viscosity=LogParameter(0.05),
)

# 5. Fit
config = FittingConfig(num_epochs=3000, learning_rate=1e-3, verbose=True)
fitted = fit_model_to_experimental_data(fit_model, rheometer, batched, config)

# 6. Display results
from diff_rheo import display_results
display_results(fitted, true_model)
```

### Variational Inference

Fit a Giesekus model with uncertainty quantification:

```python
import jax
import jax.numpy as jnp
from diff_rheo import VirtualRheometer, fit_variational_inference, display_results
from diff_rheo._solver import DiffraxSolver
from diff_rheo._data_types import ShearStrainRateData, BatchedData, FittingConfig
from diff_rheo.models import Giesekus
from diff_rheo.parameters import LogGaussianParameter, LogParameter

# Model with variational (stochastic) parameters
vi_model = Giesekus(
    polymer_viscosity=LogGaussianParameter(mean=0.7, log_std=-2.0),
    relaxation_time=LogGaussianParameter(mean=0.0, log_std=-2.0),
    solvent_viscosity=LogGaussianParameter(mean=-2.3, log_std=-2.0),
    mobility_parameter=LogGaussianParameter(mean=-2.3, log_std=-2.0),
    observation_noise=LogParameter(-4.0),
)

solver = DiffraxSolver()
rheometer = VirtualRheometer.setup(vi_model, "strain_rate_response", solver)

config = FittingConfig(
    num_epochs=4000,
    learning_rate=5e-4,
    ensemble_size=16,
    key=jax.random.PRNGKey(42),
    verbose=True,
)
fitted = fit_variational_inference(vi_model, rheometer, batched, config, direct=True)

# Parameter posteriors are accessible via fitted.parameter_values
# Returns (mean, std) tuples for stochastic parameters
params = fitted.parameter_values
```

---

## Architecture Reference

```
diff_rheo/
├── parameters.py          Parameter types (Parameter, LogParameter, GaussianParameter, …)
├── _data_types.py         Data containers (ExperimentalData, BatchedData, FittingConfig, …)
├── _forcing.py            Forcing functions (VelocityGradient, AppliedStress)
├── _solver.py             ODE solver wrappers (DiffraxSolver)
├── _protocols.py          Experiment protocols (strain-rate, stress-controlled)
├── _rheometer.py          VirtualRheometer (factory + run_experiment + run_ensemble)
├── _fitting.py            ModelFitter training loop
├── _core.py               Public API: loss functions, fit_*, model_bic, display_results
├── _utils.py              Internal utilities (tensor operations, Mittag-Leffler, AR noise)
└── models/
    ├── _constitutive_model.py   AbstractConstitutiveModel hierarchy
    ├── _generalized_newtonian.py  Newtonian, CarreauYasuda, PowerLaw
    ├── _viscoelastic.py         OldroydB, Giesekus, PTT variants, FENE, WhiteMetzner, XPomPom
    └── _rude.py                 RUDE data-driven TBNN model (experimental)
```

**Data flow for a single training step:**

```
ExperimentalData
    │
    ├─ get_forcing_function()   ──►  AbstractForcing (VelocityGradient / AppliedStress)
    │
    └─ BatchedData.fitting_schedule()
             │
             ▼
         ModelFitter._make_step()
             │
             ├─ eqx.filter_value_and_grad(loss_fn)
             │       │
             │       └─ loss_fn (data_fitting_loss / variational_inference_loss)
             │               │
             │               └─ VirtualRheometer.run_experiment() / run_ensemble()
             │                       │
             │                       └─ AbstractProtocol.run()
             │                               │
             │                               └─ DiffraxSolver.integrate()
             │                                       │
             │                                       └─ model.extra_stress_response_rhs()
             │
             └─ optax.adam.update() + eqx.apply_updates()
```

---

## Unfinished / Needs Improvement

### Known Limitations

1. **Simple shear only** – all viscoelastic protocols are implemented for
   simple shear flows only.  Extensional flows (uniaxial, biaxial, planar)
   are not supported.

2. **No GPU/multi-device support documentation** – JAX supports GPU/TPU out of
   the box, but the library has not been benchmarked or validated on
   accelerators.

3. **Test suite is minimal** – there are no automated tests for model
   correctness against known analytical solutions.  Model validation is
   entirely manual (via notebooks).

4. **RUDE model is incomplete** – the `RUDE` TBNN is not yet integrated with
   `VirtualRheometer` or `AbstractViscoelasticModel`.  It cannot currently be
   used as a drop-in constitutive model; only forward evaluation of the F
   function is supported.

5. **Shear stress response for GN models is unimplemented** – `VirtualRheometer.setup`
   raises `NotImplementedError` for `experiment_type="shear_stress_response"`
   with any generalized Newtonian model.

### Planned / In-Progress

6. **Curriculum learning** – `BatchedData.fitting_schedule` currently supports
   epoch-based data selection but the curriculum strategy is not fully
   documented or tested.

7. **Remaining OpenFOAM rheological models** – several models available in
   OpenFOAM (e.g., Leonov, Rolie-Poly, PTT-Phan-Thien-Tanner variants) have
   not been implemented.

8. **Optimisation of input forcing** – implemented in `diff_rheo._information`.
   `optimize_experiment` designs a forcing for parameter estimation (gradient
   ascent on the Expected Information Gain); `optimize_discriminating_experiment`
   designs one for model discrimination (T-optimality). Both single-tone
   (`OscillatoryShearDesign`) and multi-tone Fourier (`MultiToneShearDesign`)
   waveforms are supported, and the observable can include the first
   normal-stress difference N₁ (`shear_and_normal_stress_observable`). Free
   per-time-point waveforms remain future work.

9. **Batched/parallel experiments** – training currently loops over experiments
   in Python; vmapping over the batch dimension would be more efficient.

10. **Benchmark against commercial software** – no validation against ANSYS
    Fluent, OpenFOAM, or RheoTool has been performed.

11. **Autoregressive noise model** – `generate_autoregressive_noise` exists but
    is not connected to the fitting pipeline as a proper likelihood component.

12. **Variational inference stability** – the ELBO optimisation can be
    numerically unstable for certain model/data combinations.  Improved
    initialisation strategies and KL annealing schedules have not been explored.
