"""Training helpers for the instantaneous closure.

Re-exports the constriction geometry setup, the PIV observation operator, and
the :func:`~jax_rheology.training.fit.fit` interface that turns named closure
and optimiser specs into a training run.
"""

from jax_rheology.training.observation import (  # noqa: F401
    piv_downsample_THW,
    add_piv_noise_jax,
)
from jax_rheology.geometries.constricted_channel import (  # noqa: F401
    setup_channel_constriction,
)
from jax_rheology.training.fit import (  # noqa: F401
    fit,
    FitResult,
    RoiVelocityPressureLoss,
    MaskedFieldRMSE,
    SchemeAlternation,
    Adam,
    PIV,
)
