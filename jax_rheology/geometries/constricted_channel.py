"""Constricted channel: two semicircular wall obstacles forming a throat.

Builds the grid and the immersed-boundary particle set for the constriction
used by the generalized-Newtonian truth runs and the instantaneous training
runs.
"""
import jax.numpy as jnp
from jax_ib.base import particle_class as pc
from jax_ib.base import kinematics as ks

def setup_channel_constriction(domain):
    """Create two semicircular obstacles at the walls to form a constriction."""
    def param_rot_ellipse(geometry_param, theta):
        A = geometry_param[0]
        B = geometry_param[1] 
        phi = geometry_param[2]
        excc = jnp.sqrt(1-jnp.round((B/A)**2, 6))
        return B/jnp.sqrt(1-(excc*jnp.cos(theta-phi))**2)
    
    # Center of channel in x-direction
    center_x = (domain[0][1] + domain[0][0]) / 2  # x = 4.0 for default domain
    radius = 1.5  # Radius of semicircles
    
    # Bottom semicircle: center at (4.0, 0.0), creates obstacle from y=0 to y=1.5
    bottom_center_y = 0.0
    
    # Top semicircle: center at (4.0, 4.0), creates obstacle from y=2.5 to y=4.0
    top_center_y = domain[1][1]  # y = 4.0 for default domain
    
    # Create two circular particles positioned to act as semicircles
    particle_geometry_param = jnp.array([
        [radius, radius, 0.0],  # Bottom semicircle
        [radius, radius, 0.0]   # Top semicircle
    ])
    
    particle_center_position = jnp.array([
        [center_x, bottom_center_y],  # Bottom semicircle center
        [center_x, top_center_y]      # Top semicircle center
    ])
    
    displacement_param = jnp.array([
        [0.0, 0.0],  # Bottom semicircle
        [0.0, 0.0]   # Top semicircle
    ])
    
    rotation_param = jnp.array([
        [0.0, 0.0, 0.0, 0],  # Bottom semicircle
        [0.0, 0.0, 0.0, 0]   # Top semicircle
    ])
    
    mygrids = pc.Grid1d(100, domain=(0, 2*jnp.pi))
    
    particles = pc.particle(
        particle_center_position, particle_geometry_param,
        displacement_param, rotation_param, mygrids,
        param_rot_ellipse, ks.displacement, ks.rotation
    )
    
    return particles
