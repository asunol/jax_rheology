"""Generalized-Newtonian laws and the instantaneous TBNN closure.

Re-exports the public names of both. ``tbnn_memory`` is deliberately not
imported here: registering the memory closures is opt-in, so importing this
package does not pull in the viscoelastic model set.
"""

from jax_rheology.models.generalized_newtonian import *  # noqa: F401,F403
from jax_rheology.models.tbnn_instantaneous import *  # noqa: F401,F403
from jax_rheology.models.constructors import (  # noqa: F401
    GNFModel,
    MemoryModel,
    newtonian,
    power_law,
    carreau_yasuda,
    giesekus,
    fene_p,
)
