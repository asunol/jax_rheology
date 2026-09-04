"""Local extension beyond upstream nat_coms @ 92d9dad — paper-era
conformation-form FENE-P candidate (Bird 1980 conformation variables); see
_audit/DIFFRHEO_EQUIVALENCE.md for pointwise equivalence to
models._viscoelastic.FENEP.
"""
from __future__ import annotations

from typing import Union

import equinox as eqx
import jax
import jax.numpy as jnp

from .._data_types import ODESolution, SimulationData
from .._forcing import VelocityGradient
from .._protocols import AbstractProtocol
from .._solver import AbstractODESolver
from ..parameters import AbstractParameter
from ._constitutive_model import AbstractConstitutiveModel


def peterlin_f(tr_a: jax.Array, lsq: jax.Array) -> jax.Array:
    """Mirror log_conformation._fene_p_peterlin_f exactly."""
    denominator = lsq - tr_a
    floor = 1e-3 * lsq
    effective = floor + floor * jnp.logaddexp(
        0.0, (denominator - floor) / floor
    )
    return lsq / effective


class FENEPConformation(AbstractConstitutiveModel):
    """FENE-P with conformation state and package-compatible parameter names."""

    polymer_viscosity: AbstractParameter
    relaxation_time: AbstractParameter
    solvent_viscosity: AbstractParameter
    extension_length: AbstractParameter

    @eqx.filter_jit
    def conformation_rhs(
        self,
        t: Union[float, jax.Array],
        state: jax.Array,
        velocity_gradient: VelocityGradient,
    ) -> jax.Array:
        axx, axy, ayy, azz = state
        lam = self.relaxation_time.get_value()
        lsq = self.extension_length.get_value() ** 2
        f = peterlin_f(axx + ayy + azz, lsq)
        a = lsq / (lsq - 3.0)
        gd = velocity_gradient.gradient(t)[0, 1]
        return jnp.asarray(
            [
                2.0 * gd * axy - (f * axx - a) / lam,
                gd * ayy - f * axy / lam,
                -(f * ayy - a) / lam,
                -(f * azz - a) / lam,
            ],
            dtype=jnp.float64,
        )

    @eqx.filter_jit
    def total_stress(
        self,
        t: Union[float, jax.Array],
        state: jax.Array,
        velocity_gradient: VelocityGradient,
    ) -> jax.Array:
        axx, axy, ayy, azz = state
        lam = self.relaxation_time.get_value()
        gp = self.polymer_viscosity.get_value() / lam
        nu_s = self.solvent_viscosity.get_value()
        lsq = self.extension_length.get_value() ** 2
        f = peterlin_f(axx + ayy + azz, lsq)
        a = lsq / (lsq - 3.0)
        conformation = jnp.asarray(
            [[axx, axy, 0.0], [axy, ayy, 0.0], [0.0, 0.0, azz]]
        )
        polymer = gp * (f * conformation - a * jnp.eye(3))
        solvent = 2.0 * nu_s * velocity_gradient.rate_of_strain(t)
        return polymer + solvent

    def rest_conformation(self) -> jax.Array:
        return jnp.asarray([1.0, 0.0, 1.0, 1.0], dtype=jnp.float64)


class ConformationStrainRateProtocol(AbstractProtocol):
    """Integrate A from rest and expose total stress to standard data objects."""

    @eqx.filter_jit
    def run(
        self,
        model: FENEPConformation,
        velocity_gradient: VelocityGradient,
        time_range: jax.Array,
        initial_condition: jax.Array,
        solver: AbstractODESolver,
    ) -> SimulationData:
        del initial_condition
        instance = model.get_instance()
        conformation = solver.integrate(
            instance.conformation_rhs,
            instance.rest_conformation(),
            time_range,
            velocity_gradient,
        )
        total_stress = eqx.filter_vmap(
            instance.total_stress, in_axes=(0, 0, None)
        )(time_range, conformation.ys, velocity_gradient)
        solution = ODESolution(
            ys=total_stress,
            ts=time_range,
            result=conformation.result,
            stats=conformation.stats,
            raw_solution=conformation.raw_solution,
        )
        return SimulationData(
            forcing_function=velocity_gradient,
            solution=solution,
            experiment_type="strain_rate_response",
        )
