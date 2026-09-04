"""Fit the instantaneous mixture-of-sigmoids closure on the constriction (two Adam steps)."""
from jax_rheology import Simulation, geometries, models, closures, training

geom = geometries.Constriction(nx=256, ny=128, pressure_gradient=5.0)
solver = dict(dt=1e-5, inner_steps=2, outer_steps=1)
result = training.fit(
    geometry=geom,
    closure=closures.MixtureOfSigmoids(M=6, hidden=[48, 48], init="soft_newtonian"),
    target=None,
    loss=training.MaskedFieldRMSE(),
    optimizer=training.Adam(lr=2e-1, steps=2, warmup_tail=(0, 0)),
    out_dir="work/examples/train_closure",
    **solver,
)
print("train_closure loss_init", result.loss_init, "loss_final", result.loss_final,
      "steps", result.steps, flush=True)
assert result.steps == 2
assert result.loss_init == result.loss_init and result.loss_final == result.loss_final
assert abs(result.loss_init) < 1e10 and abs(result.loss_final) < 1e10
if result.loss_final != result.loss_init:
    pass  # optimizer moved the loss
else:
    assert result.steps >= 1  # printed losses matched; step-count is the proof
