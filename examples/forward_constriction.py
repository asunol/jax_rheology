"""Carreau-Yasuda forward solve on the constriction, at reduced scale (two inner steps, one frame)."""
from jax_rheology import Simulation, geometries, models

sim = Simulation(
    geometry=geometries.Constriction(nx=256, ny=128, domain=((0, 8), (0, 4)),
                                     pressure_gradient=5.0),
    model=models.carreau_yasuda(eta_inf=0.02, eta_0=1.0, lam=5.0, n=0.5, a=2.0),
    dt=1e-5, inner_steps=2, outer_steps=1,
)
traj = sim.run()
print("forward_constriction shape", traj.shape, "dtype", traj.dtype, "finite", traj.finite)
print("forward_constriction vmax", float(abs(traj.array).max()))
assert traj.finite
traj.save("work/examples/forward_constriction")
