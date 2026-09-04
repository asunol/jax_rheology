"""Flatten and unflatten a parameter pytree to and from a single array.

The optimisers and the finite-difference gradient checks want one flat vector;
the solver wants the original pytree. Slice bounds are precomputed so the
round trip stays traceable under JIT.
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

def flatten_params(params):
    """Flatten PyTree parameters into a single array."""
    # Flatten the PyTree into leaves and the structure
    leaves, tree_def = jax.tree_util.tree_flatten(params)
    flat_leaves = [leaf.flatten() for leaf in leaves]
    params_flat = jnp.concatenate(flat_leaves)
    return params_flat, tree_def, [leaf.shape for leaf in leaves]  # FIXED: was incorrectly using "shapes" instead of "leaves"

def unflatten_params_old(params_flat, tree_def, shapes, starts=None, ends=None):
    """
    Unflatten parameters back into a PyTree structure.
    
    Args:
        params_flat: Flat parameter array
        tree_def: PyTree definition
        shapes: List of parameter shapes
        starts: List of start indices for each parameter block (optional)
        ends: List of end indices for each parameter block (optional)
    
    Returns:
        Reconstructed PyTree
    """
    # If starts/ends not provided, calculate them
    if starts is None or ends is None:
        starts = []
        ends = []
        idx = 0
        for shape in shapes:
            size = int(np.prod(shape))
            starts.append(idx)
            ends.append(idx + size)
            idx += size
    
    # Split the flat params back into the original shapes
    leaves = []
    for i, shape in enumerate(shapes):
        start_idx = int(starts[i])
        end_idx = int(ends[i])
        
        # Use standard slicing for safety
        param_slice = params_flat[start_idx:end_idx]
        
        # Reshape to original shape
        leaves.append(param_slice.reshape(shape))
    
    # Reconstruct the original PyTree structure
    return jax.tree_util.tree_unflatten(tree_def, leaves)

def unflatten_params(params_flat, tree_def, shapes, starts=None, ends=None):
    """
    Unflatten parameters back into a PyTree structure.
    
    Args:
        params_flat: Flat parameter array
        tree_def: PyTree definition
        shapes: List of parameter shapes
        starts: List of start indices for each parameter block (optional)
        ends: List of end indices for each parameter block (optional)
    
    Returns:
        Reconstructed PyTree
    """
    # Handle case where we don't need to unflatten (simple models like power law)
    if tree_def is None or shapes is None:
        return params_flat
    
    # Ensure params_flat is a JAX array
    params_flat = jnp.array(params_flat)
    
    # If starts/ends not provided, calculate them
    if starts is None or ends is None:
        starts = []
        ends = []
        idx = 0
        for shape in shapes:
            size = int(np.prod(shape))
            starts.append(idx)
            ends.append(idx + size)
            idx += size
    
    # Split the flat params back into the original shapes
    leaves = []
    for i, shape in enumerate(shapes):
        start_idx = int(starts[i])
        end_idx = int(ends[i])
        
        # Use standard slicing for safety
        param_slice = params_flat[start_idx:end_idx]
        
        # Reshape to original shape
        leaves.append(param_slice.reshape(shape))
    
    # Reconstruct the original PyTree structure
    return jax.tree_util.tree_unflatten(tree_def, leaves)