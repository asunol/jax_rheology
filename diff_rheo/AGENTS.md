# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Purpose

`diff_rheo` is a JAX-based differentiable rheology solver. Given experimental shear rheometer data (strain-rate or stress trajectories), it fits constitutive model parameters by integrating the model ODEs through `diffrax` and back-propagating through the solve with `equinox` + `optax`. The package supports both deterministic L2 fitting and variational inference for uncertainty quantification.

## Common commands

This project uses `uv` and targets Python ≥ 3.13.2.

```bash
# Install / sync the environment
uv sync

# Install the package in editable mode into an existing env
pip install -e .

# Run the full test suite
pytest tests/

# Run a single test file or test
pytest tests/test_solver.py
pytest tests/test_solver.py::test_name -v
```

CI (`.github/workflows/test_diff_rheo.yml`) runs `pytest tests/` across Python 3.10–3.13 on Linux/macOS/Windows. Note the CI matrix lists 3.10+ but `pyproject.toml` requires 3.13.2+; the local source-of-truth is `pyproject.toml`.

Cluster scripts in `scripts/` (`run_exp.sh`, `run_exp_gpu.sh`) submit SLURM array jobs that activate conda envs `diffrheo` / `diffrheogpu` and call `generate_random_data.py --model_idx $SLURM_ARRAY_TASK_ID`.

## Architecture

The system is organized around three orthogonal abstractions: **constitutive models**, **forcing**, and **protocols** (which describe what kind of experiment is being simulated). A `VirtualRheometer` composes a protocol with an ODE solver and is the object that fitting/inference loops differentiate through.

### Core pipeline

1. **Forcing** (`_forcing.py`) — `VelocityGradient` and `AppliedStress` wrap a callable that returns a 3×3 tensor at time `t`. Built via `from_components(grad_u_12=..., sigma_12=..., ...)`; missing components default to zero. `VelocityGradient.rate_of_strain` returns the symmetrized gradient.
2. **Constitutive model** (`models/`) — subclasses of `AbstractConstitutiveModel`. Two families:
   - `AbstractGeneralizedNewtonianModel` (`Newtonian`, `CarreauYasuda`, `PowerLaw`) implements `stress_response(t, velocity_gradient)` algebraically.
   - `AbstractViscoelasticModel` (`OldroydB`, `Giesekus`, `LinearPTT`/`ExponentialPTT`/`GeneralizedPTT`, `FENECR`, `FENEP`, `XPomPom`, `GeneralizedOldroydB`, `WhiteMetzner`) implements `extra_stress_response_rhs` (for strain-rate-driven experiments) and `shear_stress_experiment_rhs` (for stress-driven experiments). Total stress in the strain-rate protocol is `extra stress + 2 * solvent_viscosity * D`.
   - `RUDE` (`models/_rude.py`) is a tensor-basis NN that parameterises the non-linear `F` function plugged into `GeneralizedOldroydB`.
3. **Protocol** (`_protocols.py`) — small dispatch classes: `GeneralizedNewtonianStrainRateProtocol`, `ViscoelasticStrainRateProtocol`, `ViscoelasticShearStressProtocol`. They take a model + forcing and either evaluate algebraically (`vmap` over time) or hand the model's RHS to the solver. Strain-rate protocols return a 3×3 stress trajectory; the shear-stress protocol returns a 7-vector `[s11, s22, s33, s13, s23, u12, u12_integrated]`.
4. **Solver** (`_solver.py`) — `DiffraxSolver` wraps `dfx.diffeqsolve` with `Tsit5/Dopri5/Heun` and a PID step controller. Returns an `ODESolution` carrying `ys`, `ts`, success flag, stats, and the raw diffrax solution.
5. **Rheometer** (`_rheometer.py`) — `VirtualRheometer.setup(model, experiment_type, solver=...)` picks the right protocol given the model class and `experiment_type` ∈ `{"strain_rate_response", "shear_stress_response"}`. `run_experiment` is a `@eqx.filter_jit` of `protocol.run`. `run_ensemble(model, ..., key, size)` `vmap`s over keys, sampling each random parameter independently — this is the core machinery for variational inference.

### Parameters (`parameters.py`)

Every model attribute that the optimizer touches is wrapped in an `AbstractParameter`:

- `Parameter` — plain value.
- `LogParameter` — stores `log(x)`, exposes `x`. **`AbstractConstitutiveModel.__init__` auto-wraps any bare `float`/`jax.Array` kwarg in a `LogParameter`** — so `Newtonian(viscosity=1.5)` is equivalent to `Newtonian(viscosity=LogParameter(1.5))`. Pass `Parameter(1.5)` explicitly to avoid the log transform.
- `StaticParameter` — frozen (`trainable_count` skips it; `get_non_log_instance` keeps it as-is).
- `TanhParameter` — bounded in `(0, max_value)`.
- `GaussianParameter` / `LogGaussianParameter` — variational distributions over a parameter; `sample(key)` produces a deterministic `Parameter` snapshot. The KL-divergence term in `_core.kl_divergence` walks the pytree and finds these by `isinstance` — anything that should contribute to the variational ELBO must be one of these two types.

`model.get_instance(key)` returns a deterministic snapshot of the model: random parameters get sampled with `key` (or replaced by their non-random expectation if `key is None`); other parameters pass through unchanged. Both protocol implementations call `get_instance()` internally so model code can assume parameters are deterministic.

### Data and fitting

- `ExperimentalData` (`_data_types.py`) carries `(time, data, forcing_data, initial_condition)`. Subclasses `ShearStrainRateData` / `ShearStressData` know how to (a) interpolate the forcing trajectory into a `VelocityGradient` / `AppliedStress` via `dfx.backward_hermite_coefficients` + `CubicInterpolation`, and (b) extract the right slice from a simulation (`data[:, 0, 1]` for strain-rate, `data[:, -1]` for stress). `ShearStrainRateNormalStressData` is a strain-rate subclass whose observable is the two-channel `[σ₁₂, N₁]` (`data` shape `(T, 2)`, `N₁ = σ₁₁ − σ₂₂`) — measuring the normal stress breaks model degeneracies that no shear waveform can (Giesekus/PTT ≈ Oldroyd-B in σ₁₂ alone). The MSE loss and L2-BIC are shape-agnostic, so it drops straight into the fitting/selection pipeline.
- `BatchedData.from_data(*experiments)` wraps a list; `fitting_schedule(config, epoch)` is the iterator the trainer consumes (currently yields all experiments every epoch, but is the hook for curriculum/batching strategies).
- `FittingConfig` is a plain dataclass: `num_epochs`, `learning_rate`, `ensemble_size`, `verbose`, `key`, `schedule_lr`.
- `ModelFitter` (`_fitting.py`) is a thin training loop: `eqx.filter_value_and_grad(loss_fn, has_aux=True)` → `optimizer.update` → `eqx.apply_updates`. The loss function signature is `(model, rheometer, data_iterator, *, key=..., ensemble_size=...) -> (loss, aux)`.
- `_core.py` exposes the public losses:
  - `data_fitting_loss` — MSE between simulated and observed trajectories.
  - `variational_inference_loss` / `..._direct` — ELBO = KL(q‖N(0,1)) − E[log-likelihood]. The non-direct version computes the likelihood on `diff(trajectory)` (autoregressive-noise friendly); the `_direct` version computes it pointwise and adds a prior on `observation_noise`.
  - `model_bic` / `calculate_bic_from_l2` — BIC for model selection. The L2 version derives σ̂² = SSE/N analytically; the variational version uses `trajectory_log_likelihood`.
- `display_results(fit_model, ground_truth)` colorizes a parameter comparison table (used heavily in the `scripts/` analysis pipelines).

### Information geometry (`_information.py`)

Everything here is built on the **Fisher Information Matrix** `g = JᵀJ/σ²`, where `J = ∂(predicted trajectory)/∂(parameters)` is obtained by autodiff through the solver. Three capabilities, all from `g`:

- **Uncertainty** — `fisher_information(model, rheometer, data, noise=...)` returns a `FisherInformation` whose `.covariance()` / `.standard_errors()` / `.correlation()` give Cramér-Rao error bars and parameter correlations. `parameter_uncertainty(...)` is the `{name: (value, std)}` convenience wrapper.
- **Sloppy directions** — `sloppy_analysis(fisher)` returns a `SloppyAnalysis` with the FIM eigenspectrum, stiff/sloppy eigenvectors, condition number, and effective dimensionality.
- **Experiment design (parameter estimation)** — `optimize_experiment(model, rheometer, prior_data, ...)` gradient-ascends a Fisher-information criterion over a continuously-parameterised forcing waveform (`OscillatoryShearDesign` by default) to design the next experiment; `expected_information_gain(...)` scores one candidate. The `criterion` argument picks the objective: `"eig"` (D-optimality, default — total information), `"a_optimal"` (min summed posterior variance), `"e_optimal"` (lift the worst/sloppiest direction), or `"target"` (c-optimality — minimise the posterior variance of one named parameter or direction). To *un-sloppy a specific parameter* use `"target"`: the experiment optimal for the aggregate information is generally not the one optimal for a chosen parameter, so it must be targeted explicitly. Pass `target=` a parameter name or a sloppy eigenvector when `criterion="target"`.
- **Experiment design (model discrimination)** — `optimize_discriminating_experiment(reference_models, rival_models, ...)` designs a forcing that maximises the worst-case T-optimality separation between competing models (the cure for a model-selection "confusion matrix"); `discrimination_score(...)` is the underlying metric.

Forcing designs and observables: `OscillatoryShearDesign` parameterises a single-tone `A·sin(ω·t)`; `MultiToneShearDesign` is a multi-tone Fourier waveform `A·Σ sₖ·sin(ωₖ·t+φₖ)` (softmax-shared amplitude budget keeps the peak ≤ `A`, so the stiff ODEs stay solvable for any number of tones) — it is a strict superset of the single tone, so optimising over it bounds the *information maximum of pure shear*. `SplineShearDesign` drops the Fourier restriction entirely: `γ̇(t)` is a cubic spline through `n_knots` control points (`tanh`-squashed into the shear-rate envelope), the most general smooth pure-shear waveform — use it for the "is the sinusoid genuinely optimal?" sanity check by seeding it from a sinusoidal optimum (`design.to_unconstrained(A, ω)` / `seed_from_sine`). Because a free spline can become jagged, both `optimize_experiment` and `optimize_discriminating_experiment` take a `roughness_weight` that subtracts `SplineShearDesign.roughness_penalty` (a second-difference smoothness penalty) from the objective. Pass a non-`(A,ω)` design to `optimize_discriminating_experiment` via `init_z=design.default_z(...)`. The `extractor` argument selects the observable: `shear_stress_observable` (default, σ₁₂) or `shear_and_normal_stress_observable` (`[σ₁₂, N₁]`) — the latter resolves the shear-only degeneracies.

Conventions: the FIM is computed in *natural* (optimiser/log) coordinates by default — pass `coords="physical"` for physical units. `observation_noise` is excluded (the prediction does not depend on it). `mode="rev"` (default) works with any solver; `mode="fwd"` is cheaper but needs `DiffraxSolver(adjoint="direct")` for viscoelastic models. `optimize_experiment` uses forward-mode internally — use a `"direct"`-adjoint solver with viscoelastic models.

### Bayesian inference (`_bayesian.py`)

Where `_information.py` gives the Laplace/Cramér-Rao *approximation* to parameter uncertainty (one Jacobian, no sampling), `_bayesian.py` draws the **exact posterior** `p(θ|data)` with NumPyro's NUTS — the differentiable solver supplies `∇log p(θ|data)` by autodiff, so the integration is minimal. `numpyro` and `arviz` are imported lazily, so `import diff_rheo` still works without them; install via the `bayesian` extra (`pip install -e ".[bayesian]"`).

- `run_nuts(model_template, rheometer, data, ...)` → a `BayesianFit` (posterior draws + an `arviz.InferenceData`). The `model_template` fixes the model class and which parameters are free; priors supply the values. Build the solver with `DiffraxSolver(throw=False)` so a destabilising parameter draw yields a rejected sample rather than a raised error.
- `default_priors(model)` — weakly-informative priors keyed by parameter name: `LogNormal` for positive scale parameters, `Uniform` for bounded structural parameters (Giesekus `alpha`, PTT `epsilon`/`zeta`). The `Uniform` choice is deliberate — it keeps finite density at the nesting value 0 so the posterior is free to collapse a spurious component.
- `BayesianFit` — `.posterior(name)`, `.posterior_mean()`, `.posterior_mass_below(name, thr)` (the spurious-parameter test), `.posterior_model()`, `.waic()`, `.loo()` (PSIS-LOO with Pareto-k̂), `.divergence_count()`.
- `compare_models({label: BayesianFit}, ic="loo"|"waic")` — ranks competing fits by a posterior-predictive criterion. Unlike BIC's `k·log n` penalty, WAIC/PSIS-LOO integrate over the posterior and stay valid at nesting boundaries.

The likelihood is `obs ~ Normal(predicted_trajectory, σ)` with σ inferred; all experiments are concatenated into one `obs` site so the per-time-point log-likelihood feeds WAIC/LOO. `scripts/bayesian_overfit.py` applies this to the four Oldroyd-B/shear/low-noise datasets where BIC over-fits, comparing the BIC vs WAIC vs PSIS-LOO verdict.

### Adding a new constitutive model

1. Subclass `AbstractGeneralizedNewtonianModel` or `AbstractViscoelasticModel` in `models/`.
2. Declare every learnable quantity as a class-level `AbstractParameter` field (equinox needs the type annotations).
3. Implement the required `@abstractmethod` (`stress_response` for GN; `extra_stress_response_rhs` and `shear_stress_experiment_rhs` for VE). Decorate with `@eqx.filter_jit` if it does heavy compute.
4. Use the helpers in `_utils.py`: `_rate_of_strain_to_strain_rate`, `_flatten_symmetric_array`, `_vector_to_symmetric_matrix`, `_generalized_mittag_leffler_function`.
5. Re-export from `models/__init__.py`. The `VirtualRheometer.setup` dispatch is type-based, so subclassing the right abstract base is enough.

### Tests

`tests/conftest.py` provides standard fixtures: `rng_key`, `t_scalar`/`t_array`, pre-built `general_flow`/`applied_stress_flow`/`shear_stress_flow`, instantiated `newtonian_model`/`oldroydb_model`/`stochastic_oldroydb_model`, factories for parameterised variants, a `mock_solver` (`DiffraxSolver()`), and the canonical initial conditions (`jnp.zeros((3,3))` for strain-rate input, `jnp.zeros(7)` for stress input). Prefer reusing these over constructing fresh objects.
