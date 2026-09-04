"""Pressure projection and the variable-viscosity velocity solves.

The incompressibility projection for each geometry's boundary conditions, and
the implicit velocity solves that a spatially varying viscosity requires.

Two families:

* Fast-diagonalization Poisson and Helmholtz solves, one per boundary
  configuration (all-Neumann for the periodic and closed cases, Dirichlet
  where a wall value is imposed, and the mixed contraction and cavity cases).
  These are direct and cheap, and are the default where the operator
  separates.
* Krylov solves (conjugate gradient and GMRES) for the variable-viscosity
  vector operator ``div(nu grad u)``, where the operator does not separate and
  fast diagonalization no longer applies. Viscosity is harmonically averaged
  to faces so the discrete operator stays symmetric.

All of it is differentiable: the solves are used inside the training loop, so
they are written against ``jax.scipy.sparse.linalg`` rather than SciPy.
"""

from typing import Callable, Optional
import scipy.linalg
import numpy as np
from jax_ib.base import array_utils
from jax_cfd.base import fast_diagonalization
import jax.numpy as jnp
from jax.scipy.sparse import linalg
from jax_cfd.base import pressure
from jax_ib.base import grids
from jax_rheology.core import boundaries as bnew
from jax_ib.base import boundaries
from jax_ib.base import finite_differences as fd
from jax_rheology.core import state as particle_class
from jax_ib.base import interpolation
from jax_ib.base import diffusion
from jax import lax
from jax import debug as jdebug  # NEW
# If you want to use GMRES for the adjoint solve (recommended, but optional):
from jax.scipy.sparse import linalg as jsla

# --- debug toggles ---
DEBUG_DOTTEST_ONCE = False  # will auto-disable after first run
# ----------------------

# ---- JIT-safe assertion helper ----
def _assert_all_finite(name, x):
    def _check(arr):
        import numpy as _np
        if not _np.all(_np.isfinite(_np.asarray(arr))):
            raise FloatingPointError(f"{name} contains NaN/Inf")
    jdebug.callback(_check, x)

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions


def _operator_velocity_bc(grid, bc_spec):
  """BC wrapped around the matrix-free *moving-wall* velocity operator.

  The boundary type is data-driven.
  When ``bc_spec is None`` this returns the historically hard-coded
  moving-wall operator BC -- ``HomogeneousBoundaryConditions`` with
  ``x``-periodic, ``y``-Dirichlet -- so every existing call site is
  **byte-for-byte unchanged**. When a :class:`jax_rheology.core.boundaries.BCSpec` is
  supplied, its :meth:`velocity_operator_bc` provides the per-axis type
  (e.g. the 4:1 contraction's ``x``: Dirichlet inlet / Neumann outlet).

  Homogeneous because nonzero BC values are RHS-lifted by the stepper
  (see ``equations_rheology.fully_implicit_rheology_stepper`` Step 4).
  """
  if bc_spec is not None:
    return bc_spec.velocity_operator_bc(grid)
  bc_types = ((boundaries.BCType.PERIODIC, boundaries.BCType.PERIODIC),
              (boundaries.BCType.DIRICHLET, boundaries.BCType.DIRICHLET))
  return boundaries.HomogeneousBoundaryConditions(bc_types)




# pressure.py
def linear_solve_implicit_with_bicgstab(
    matvec, b, *, tol, maxiter, M=None, MT=None,
    adjoint_mode: str = "normal_cg",  # "bicgstab" | "normal_cg" | "gmres"
    adjoint_tol: float = None,
    adjoint_maxiter: int = None,
):
    a_tol = tol if adjoint_tol is None else adjoint_tol
    a_max = maxiter if adjoint_maxiter is None else adjoint_maxiter

    def _solve(A, rhs):
        x, _ = linalg.bicgstab(A, rhs, tol=tol, maxiter=maxiter, M=M)
        return x

    def _transpose_solve(AT, ct):
        if adjoint_mode == "bicgstab":
            y, _ = linalg.bicgstab(AT, ct, tol=a_tol, maxiter=a_max, M=None)
            _assert_all_finite("[pressure] adjoint solve y", y)
            return y
        elif adjoint_mode == "normal_cg":
            rhs = AT(ct)
            def ATA_mv(x):
                return AT(matvec(x))
            y, _ = linalg.cg(ATA_mv, rhs, tol=a_tol, maxiter=a_max, M=None)
            return y
        elif adjoint_mode == "gmres":
            y, _ = jsla.gmres(AT, ct, tol=a_tol, maxiter=a_max, M=None)
            return y
        else:
            raise ValueError("adjoint_mode must be 'bicgstab' or 'normal_cg' or 'gmres'")

    return lax.custom_linear_solve(matvec, b, _solve, _transpose_solve, symmetric=False)



def laplacian_matrix_neumann(size: int, step: float) -> np.ndarray:
  """Create 1D Laplacian operator matrix, with homogeneous Neumann BC."""
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  matrix = scipy.linalg.toeplitz(column)
  matrix[0, 0] = matrix[-1, -1] = -1 / step**2
  #matrix = jnp.asarray(matrix)
  #matrix = lax.stop_gradient(matrix)
  # critical fix: return pure NumPy array - don't convert to JAX array
  # This was causing TracerArrayConversionError in fast_diagonalization
  return matrix


def laplacian_matrix_dirichlet(size: int, step: float) -> np.ndarray:
  """Create 1D Laplacian operator matrix with homogeneous Dirichlet BC."""
  column = np.zeros(size)
  column[0] = -2 / step ** 2
  column[1] = 1 / step ** 2
  matrix = scipy.linalg.toeplitz(column)
  # Dirichlet BCs are enforced as fixed values at the boundaries; the standard
  # second-difference stencil with -2 on the diagonal and 1 on off-diagonals is
  # appropriate for the interior unknowns.
  #matrix = jnp.asarray(matrix)
  #matrix = lax.stop_gradient(matrix)
  # critical fix: return pure NumPy array - don't convert to JAX array
  # This was causing TracerArrayConversionError in fast_diagonalization
  return matrix


def _rhs_transform(
    u: grids.GridArray,
    bc: boundaries.BoundaryConditions,
) -> Array:
  """Transform the RHS of pressure projection equation for stability.

  In case of poisson equation, the kernel is subtracted from RHS for stability.

  Args:
    u: a GridArray that solves grad^2x = u.
    bc: specifies boundary of x.

  Returns:
    u' s.t. u = u' + kernel of the laplacian.
  """
  u_data = u.data
  for axis in range(u.grid.ndim):
    if bc.types[axis][0] == boundaries.BCType.NEUMANN and bc.types[axis][
        1] == boundaries.BCType.NEUMANN:
      # if all sides are neumann, poisson solution has a kernel of constant
      # functions. We substact the mean to ensure consistency.
      u_data = u_data - jnp.mean(u_data)
  return u_data
  

def projection_and_update_pressure(
    All_variables: particle_class.All_Variables,
    solve: Callable = pressure.solve_fast_diag,
    bc_spec=None,
) -> GridVariableVector:
  """Apply pressure projection to make a velocity field divergence free.

  Rebuilds the per-step ``All_Variables`` carry with the new velocity
  and pressure, threading every other slot through unchanged. Using
  :func:`dataclasses.replace` instead of a positional reconstruction
  is important for the ``stepper_type='memory'`` path:
  a positional 6-argument
  ``All_Variables(particles, v, p, Drag, Step_count, MD_var)`` silently
  defaults ``memory_fields`` and ``memory_layout`` to ``None``, which
  changes the carry pytree structure between input and output of the
  ``jax.lax.scan`` body -- ``scan`` then raises
  ``TypeError: Scanned function carry input and carry output must
  have the same pytree structure`` on the first iteration.
  ``dataclasses.replace`` preserves both slots untouched.
  """
  import dataclasses

  v = All_variables.velocity
  old_pressure = All_variables.pressure
  grid = grids.consistent_grid(*v)
  # Pressure BC is data-driven. Default
  # (bc_spec=None) reproduces the legacy behavior exactly: derive the
  # pressure BC from the velocity field BC (periodic->periodic,
  # non-periodic->Neumann). The contraction needs an explicit spec
  # because its outlet pressure is Dirichlet (p=0), which the generic
  # velocity->pressure rule (Neumann everywhere non-periodic) cannot
  # express.
  if bc_spec is not None:
    pressure_bc = bc_spec.pressure_bc(grid)
  else:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)

  q0 = grids.GridArray(jnp.zeros(grid.shape), grid.cell_center, grid)
  q0 = grids.GridVariable(q0, pressure_bc)

  qsol = solve(v, q0)
  q = grids.GridVariable(qsol, pressure_bc)

  New_pressure_Array = grids.GridArray(qsol.data + old_pressure.data,
                                        qsol.offset, qsol.grid)
  New_pressure = grids.GridVariable(New_pressure_Array, pressure_bc)

  q_grad = fd.forward_difference(q)
  if boundaries.has_all_periodic_boundary_conditions(*v):
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc) for u, q_g in zip(v, q_grad))
  else:
    v_projected = tuple(
        grids.GridVariable(u.array - q_g, u.bc).impose_bc()
        for u, q_g in zip(v, q_grad))

  return dataclasses.replace(All_variables,
                              velocity=v_projected,
                              pressure=New_pressure)


def solve_fast_diag(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = None) -> GridArray:
  """Solve for pressure using the fast diagonalization approach."""
  del q0  # unused
  if not boundaries.has_all_periodic_boundary_conditions(*v):
    raise ValueError('solve_fast_diag() expects periodic velocity BC')
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = list(map(array_utils.laplacian_matrix, grid.shape, grid.step))
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=True, implementation=implementation)
  return grids.applied(pinv)(rhs)


def solve_helmholtz_fast_diag(
    rhs: GridArray,
    *,
    alpha: float,
    beta: float,
    implementation: Optional[str] = None
) -> GridArray:
  """Solves the Helmholtz equation (alpha - betagrad^2)x = rhs using fast diagonalization.
  
  Uses JAX-CFD's numerically stable pattern to avoid factor-of-2 errors.
  """
  grid = rhs.grid
  
  # JAX-CFD's transform function pattern (from diffusion.py)
  def func(x):  # x are the Laplacian eigenvalues
      # For equation (alpha - beta*grad^2)u = rhs
      # JAX-CFD solves: u + (alpha - beta*grad^2)^{-1}(beta*grad^2)u = (alpha - beta*grad^2)^{-1} * rhs
      # Transform: (alpha - beta*grad^2)^{-1} * beta * grad^2 
      dt_nu_x = beta * x  # beta*lam
      return dt_nu_x / (alpha - dt_nu_x)  # (betalam)/(alpha - betalam)
  
  # Get static Laplacian matrices (not dependent on traced values)
  laplacians = [
      array_utils.laplacian_matrix(s, d) for s, d in zip(grid.shape, grid.step)
  ]
  
  # Apply transform (JAX-CFD style)
  op = fast_diagonalization.transform(
      func,
      laplacians,
      rhs.dtype,
      hermitian=True,
      circulant=True,
      implementation=implementation
  )
  
  # JAX-CFD pattern: result = input/alpha + transform(input)
  correction = grids.applied(op)(rhs)
  result_data = rhs.data / alpha + correction.data
  
  return grids.GridArray(result_data, rhs.offset, rhs.grid)


def solve_fast_diag_moving_wall(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul') -> GridArray:
  """Solve for channel flow pressure using fast diagonalization."""
  del q0  # unused
  ndim = len(v)

  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  laplacians = [
      array_utils.laplacian_matrix(grid.shape[0], grid.step[0]),
      array_utils.laplacian_matrix_neumann(grid.shape[1], grid.step[1]),
  ]
  for d in range(2, ndim):
    laplacians += [array_utils.laplacian_matrix(grid.shape[d], grid.step[d])]
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs)


def solve_fast_diag_contraction(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul') -> GridArray:
  """Pressure-Poisson solve for the 4:1 contraction.

  Uses fast diagonalization with per-axis 1-D Laplacians built from the
  **contraction pressure BC** -- ``x``: Neumann inlet / Dirichlet outlet
  (``p=0``); ``y``: Neumann walls -- via
  :func:`array_utils.laplacian_matrix_w_boundaries`, which handles mixed
  per-face Neumann/Dirichlet at the cell-centered pressure offset. The
  Dirichlet outlet pins the pressure nullspace, so the operator is
  non-singular (no mean-subtraction needed, unlike the all-Neumann /
  periodic cases).

  The pressure BC is read off ``q0.bc`` (the projection builds ``q0`` with
  the contraction pressure BC from :meth:`BCSpec.pressure_bc`). This keeps
  the BC the single source of truth and mirrors the
  ``solve_fast_diag_Far_Field`` pattern.
  """
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  if q0 is not None:
    pressure_bc = q0.bc
  else:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  laplacians = array_utils.laplacian_matrix_w_boundaries(
      rhs.grid, rhs.offset, pressure_bc)
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs)


def solve_fast_diag_cavity(
    v: GridVariableVector,
    q0: Optional[GridVariable] = None,
    implementation: Optional[str] = 'matmul') -> GridArray:
  """Pressure-Poisson solve for the square lid-driven cavity.

  The cavity's velocity is Dirichlet on all four walls, so its pressure
  BC is **Neumann on all four faces** and the discrete pressure operator
  has a constant null space -- pressure is defined only up to an additive
  constant, and there is no Dirichlet face to pin it (contrast
  :func:`solve_fast_diag_contraction`, whose Dirichlet outlet pins the
  nullspace).

  Two things make the singular operator well-posed here:

    1. **Compatibility.** ``_rhs_transform`` subtracts the mean of the
       RHS on the all-Neumann layout so the projected system is
       solvable (``intgrad^2p = ointdp/dn = 0`` forces ``int RHS = 0``).
    2. **Uniqueness.** ``fast_diagonalization.pseudoinverse`` zeros the
       null (zero) eigenvalue rather than inverting it, which projects
       the constant mode out of the solution. This is why
       ``pseudoinverse`` (not ``inverse``) is mandatory for the cavity.

  The pressure BC is read off ``q0.bc`` (the projection builds ``q0``
  with the cavity pressure BC from :meth:`BCSpec.cavity`), keeping the
  BC the single source of truth -- same pattern as
  :func:`solve_fast_diag_contraction` and :func:`solve_fast_diag_Far_Field`.

  Dedicated rather than a reuse of ``solve_fast_diag_contraction`` so
  the all-Neumann mean-subtraction stays explicit. Numerically
  equivalent to that solver on the cavity BC (the pseudoinverse
  handles the nullspace either way); the explicit ``_rhs_transform``
  is belt-and-suspenders compatibility insurance. A new function
  rather than editing the contraction solver leaves every existing
  pressure path unchanged.
  """
  grid = grids.consistent_grid(*v)
  rhs = fd.divergence(v)
  if q0 is not None:
    pressure_bc = q0.bc
  else:
    pressure_bc = boundaries.get_pressure_bc_from_velocity(v)
  # _rhs_transform returns a bare array (mean-subtracted on the
  # all-Neumann layout); wrap it back into a GridArray so grids.applied
  # actually solves against the compatible RHS.
  rhs_transformed = grids.GridArray(
      _rhs_transform(rhs, pressure_bc), rhs.offset, rhs.grid)
  laplacians = array_utils.laplacian_matrix_w_boundaries(
      rhs.grid, rhs.offset, pressure_bc)
  pinv = fast_diagonalization.pseudoinverse(
      laplacians, rhs_transformed.dtype,
      hermitian=True, circulant=False, implementation=implementation)
  return grids.applied(pinv)(rhs_transformed)


def solve_helmholtz_fast_diag_moving_wall(
    rhs: GridArray,
    *,
    alpha: float,
    beta: float,
    implementation: Optional[str] = 'matmul'
) -> GridArray:
  """Solves the Helmholtz equation for channel flow using fast diagonalization.
  
  Uses JAX-CFD's numerically stable pattern to avoid factor-of-2 errors.
  """
  grid = rhs.grid
  ndim = grid.ndim
  
  # JAX-CFD's transform function pattern (same as periodic version)
  def func(x):  # x are the Laplacian eigenvalues
      dt_nu_x = beta * x  # beta*lam
      return dt_nu_x / (alpha - dt_nu_x)  # (betalam)/(alpha - betalam)
  
  # Assumes x-periodic, y-Dirichlet (no-slip walls) - get static matrices
  laplacian_periodic = array_utils.laplacian_matrix(grid.shape[0], grid.step[0])
  # For velocity Helmholtz solve in channel flow, enforce no-slip (Dirichlet) at walls
  laplacian_dirichlet = laplacian_matrix_dirichlet(grid.shape[1], grid.step[1])
  #laplacian_dirichlet = jnp.asarray(laplacian_dirichlet)
  #laplacian_dirichlet = lax.stop_gradient(laplacian_dirichlet)
  # Keep as pure NumPy array - don't convert to JAX array
  # This was causing TracerArrayConversionError in fast_diagonalization
  laplacians = [laplacian_periodic, laplacian_dirichlet]

  # Extend to 3D if necessary
  for d in range(2, ndim):
      laplacian_d = array_utils.laplacian_matrix(grid.shape[d], grid.step[d])
      laplacians.append(laplacian_d)

  # Apply transform (JAX-CFD style)
  op = fast_diagonalization.transform(
      func,
      laplacians,
      rhs.dtype,
      hermitian=True,
      circulant=False,  # Important: Not all matrices are circulant
      implementation=implementation
  )
  
  # JAX-CFD pattern: result = input/alpha + transform(input)
  correction = grids.applied(op)(rhs)
  result_data = rhs.data / alpha + correction.data
  
  return grids.GridArray(result_data, rhs.offset, rhs.grid)


def solve_helmholtz_cg_moving_wall(
    rhs: GridArray,
    *,
    alpha: float,
    beta: float,
    # Add common CG parameters for control, with sensible defaults
    maxiter: int = 100,
    tol: float = 1e-6,
    bc_spec=None,
) -> GridArray:
  """Solves the Helmholtz equation for channel flow using Conjugate Gradient.

  This iterative solver is more robust than the spectral (fast diagonalization)
  method for flows with complex forcing terms near wall boundaries, which is
  common in non-Newtonian simulations. It solves the equation:
  (alphaI - betagrad^2)x = rhs

  Args:
    rhs: The right-hand-side of the equation, as a GridArray.
    alpha: The coefficient of the identity term.
    beta: The coefficient of the Laplacian term.
    maxiter: Maximum number of iterations for the CG solver.
    tol: Tolerance for convergence of the CG solver.

  Returns:
    A GridArray containing the solution `x`.
  """
  grid = rhs.grid
  
  def helmholtz_operator_flat(x_flat: jnp.ndarray) -> jnp.ndarray:
    """Computes the Helmholtz operation A(x) = (alphaI - betagrad^2)x."""
    x_data = x_flat.reshape(grid.shape)
    x_array = GridArray(x_data, rhs.offset, grid)
    
    # Boundary type for the operator's ghost cells is data-driven via
    # ``bc_spec``; ``bc_spec=None`` reproduces the
    # legacy channel BC (x-periodic, y-Dirichlet) exactly. HOMOGENEOUS
    # because we solve for the correction, not absolute velocity values.
    bc = _operator_velocity_bc(grid, bc_spec)
    x_var = GridVariable(x_array, bc)
    
    # CRITICAL: Enforce boundary conditions to populate ghost cells for fd.laplacian
    # This is essential for iterative solvers - the operator must be self-contained
    x_var = x_var.impose_bc()
    
    # Use the library's laplacian function - it handles boundary conditions via shift operations
    laplacian_x = fd.laplacian(x_var)
    
    # Apply Helmholtz formula: (alpha - betagrad^2)x  
    result_data = alpha * x_var.data - beta * laplacian_x.data
    return result_data.ravel()

  # Use JAX's Conjugate Gradient solver.
  # We solve for x in Ax = b, where A is the operator and b is the RHS.
  solution_flat, _ = linalg.cg(
      helmholtz_operator_flat,
      rhs.data.ravel(),
      tol=tol,
      maxiter=maxiter
  )

  # Reshape the flattened solution back to the original grid shape
  solution_data = solution_flat.reshape(grid.shape)
  
  # Return the solution as a GridArray, matching the input type.
  return GridArray(solution_data, rhs.offset, grid)


# Variable-coefficient operator functions for Step 3: BE-IMEX implementation

def harmonic_mean_to_faces(nu_field: jnp.ndarray, grid: grids.Grid) -> tuple:
  """Interpolate cell-centered viscosity to face-centered using harmonic mean.
  
  The harmonic mean provides better numerical stability when viscosity has
  large spatial gradients, which is common in non-Newtonian flows.
  
  Args:
    nu_field: Cell-centered viscosity field as JAX array
    grid: Grid object containing spacing information
    
  Returns:
    Tuple of face-centered viscosity arrays (nu_x_faces, nu_y_faces, ...)
    where nu_x_faces[i,j] is viscosity at face between cells (i-1,j) and (i,j)
  """
  # Handle 2D case explicitly (most common)
  if grid.ndim == 2:
    # X-faces: harmonic mean between adjacent cells in x-direction
    # nu_x[i,j] = 2 * nu[i-1,j] * nu[i,j] / (nu[i-1,j] + nu[i,j])
    nu_left = jnp.roll(nu_field, 1, axis=0)  # nu[i-1,j]
    nu_right = nu_field  # nu[i,j]
    # Use epsilon to avoid division by zero
    epsilon = 1e-12
    nu_x_faces = 2 * nu_left * nu_right / (nu_left + nu_right + epsilon)
    
    # Y-faces: harmonic mean between adjacent cells in y-direction  
    nu_bottom = jnp.roll(nu_field, 1, axis=1)  # nu[i,j-1]
    nu_top = nu_field  # nu[i,j]
    nu_y_faces = 2 * nu_bottom * nu_top / (nu_bottom + nu_top + epsilon)
    
    return (nu_x_faces, nu_y_faces)
  
  else:
    # General N-D case
    face_viscosities = []
    for axis in range(grid.ndim):
      nu_left = jnp.roll(nu_field, 1, axis=axis)
      nu_right = nu_field
      epsilon = 1e-12
      nu_faces = 2 * nu_left * nu_right / (nu_left + nu_right + epsilon)
      face_viscosities.append(nu_faces)
    return tuple(face_viscosities)


def div_nu_grad_scalar(u_scalar: GridVariable, nu_field: jnp.ndarray) -> GridArray:
  """Compute div (nugradu) for a scalar field.
  
  This implements the variable-coefficient diffusion operator using
  face-centered viscosity values for numerical stability.
  
  Args:
    u_scalar: Scalar field u as GridVariable with boundary conditions
    nu_field: Cell-centered viscosity field as JAX array
    
  Returns:
    Result of div (nugradu) as GridArray
  """
  grid = u_scalar.grid
  
  # Step 1: Interpolate viscosity to faces using harmonic mean
  nu_faces = harmonic_mean_to_faces(nu_field, grid)
  
  # Step 2: Compute gradients of u at faces
  u_grads = fd.gradient_tensor(u_scalar)
  
  # Step 3: Multiply face-centered viscosity with face-centered gradients
  # Create viscous flux components as GridVariables for proper divergence
  viscous_flux_components = []
  for axis, (nu_face, u_grad) in enumerate(zip(nu_faces, u_grads)):
    flux_data = nu_face * u_grad.data
    flux_array = GridArray(flux_data, u_grad.offset, grid)
    flux_var = GridVariable(flux_array, u_scalar.bc)
    viscous_flux_components.append(flux_var)
  
  # Step 4: Take divergence using existing fd.divergence function
  # This properly handles boundary conditions and uses established finite difference patterns
  div_result = fd.divergence(viscous_flux_components)
  
  return div_result


def div_nu_symgrad_vector(u_vector: GridVariableVector, nu_field: jnp.ndarray) -> GridArrayVector:
    """Computes a stable approximation of div (nu(gradu+graduT)) using BC-aware functions.
    
    This uses the expanded form of the divergence, div (nugradu) = nugrad^2u + gradnu.gradu, which is
    a standard, robust method for variable-coefficient diffusion that avoids the
    complexities of corner-cell data from the full stress tensor.
    """
    grid = grids.consistent_grid(*u_vector)
    
    # Create a GridVariable for the cell-centered viscosity.
    # Neumann BCs are a standard choice, implying no viscosity gradient at the wall.
    nu_variable = grids.GridVariable(
        grids.GridArray(nu_field, grid.cell_center, grid),
        boundaries.neumann_boundary_conditions(grid.ndim)
    )
    
    # Pre-compute the gradient of viscosity, gradnu. This is a vector of GridArrays.
    grad_nu = fd.gradient_tensor(nu_variable)

    viscous_terms = []
    for i, u in enumerate(u_vector):
        # 1. Compute the Laplacian term: nugrad^2u
        # Interpolate nu to the location of velocity component `u`.
        nu_on_u_face = interpolation.linear(nu_variable, u.offset)
        # `fd.laplacian` is BC-aware and computes grad^2u correctly.
        laplacian_u = fd.laplacian(u)
        laplacian_term = nu_on_u_face.data * laplacian_u.data

        # 2. Compute the gradient cross-term: gradnu . gradu_i
        # `fd.gradient_tensor(u)` computes the gradient of the scalar velocity component `u`.
        grad_u_component = fd.gradient_tensor(u)
        
        # Interpolate each component of gradnu to the location of the corresponding
        # component of gradu so they can be multiplied.
        grad_nu_dot_grad_u = 0.0
        for j in range(grid.ndim):
            # grad_nu[j] is the j-th component of gradnu.
            # grad_u_component[j] is the j-th component of gradu_i.
            grad_nu_j_interp = interpolation.linear(grad_nu[j], grad_u_component[j].offset)
            grad_nu_dot_grad_u += grad_nu_j_interp.data * grad_u_component[j].data

        # 3. Sum the terms to get the final viscous force.
        total_viscous_force = laplacian_term + grad_nu_dot_grad_u
        # Convert to GridArray for compatibility with CG solver
        viscous_grid_array = grids.GridArray(total_viscous_force, u.offset, grid)
        viscous_terms.append(viscous_grid_array)
        
    return tuple(viscous_terms)


def solve_varvisc_cg_moving_wall_vector(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-12,
    preconditioner_fn: Optional[Callable] = None,
    bc_spec=None,
) -> GridArrayVector:
  """Solve (I - dt/rho * div [nu(gradu+gradu^T)]) u* = rhs for wall boundary conditions.
  
  This is the core variable-coefficient solver for the BE-IMEX scheme.
  It uses Conjugate Gradient iteration with a matrix-free operator.
  
  Args:
    rhs_vector: Right-hand side vector field as GridArrayVector
    nu_field: Cell-centered viscosity field as JAX array
    dt: Time step size
    density: Fluid density (default 1.0)
    maxiter: Maximum CG iterations
    tol: CG tolerance
    preconditioner_fn: Optional preconditioner function for CG solver
    
  Returns:
    Solution u* as GridArrayVector
  """
  grid = grids.consistent_grid(*rhs_vector)
  ndim = len(rhs_vector)
  
  # Coefficient for the viscous term
  visc_coeff = dt / density
  
  def variable_viscosity_operator_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
    """Matrix-free operator: (I - dt/rho * div [nu(gradu+gradu^T)]) u"""
    # Reshape flat vector to grid components
    u_components = []
    start_idx = 0
    for i in range(ndim):
      end_idx = start_idx + rhs_vector[i].data.size
      u_data = u_flat[start_idx:end_idx].reshape(grid.shape)
      u_array = GridArray(u_data, rhs_vector[i].offset, grid)
      
      # Operator ghost-cell BC is data-driven via ``bc_spec``;
      # bc_spec=None reproduces the legacy x-periodic, y-Dirichlet channel.
      bc = _operator_velocity_bc(grid, bc_spec)
      u_var = GridVariable(u_array, bc).impose_bc()
      u_components.append(u_var)
      start_idx = end_idx
    
    u_vector = tuple(u_components)
    
    # Apply variable-coefficient viscous operator
    visc_term = div_nu_symgrad_vector(u_vector, nu_field)
    
    # Compute (I - dt/rho * div [nu(gradu+gradu^T)]) u = u - visc_coeff * visc_term
    result_components = []
    for i, (u_comp, visc_comp) in enumerate(zip(u_vector, visc_term)):
      # CRITICAL FIX: Interpolate the cell-centered viscous force to the face
      # locations where the velocity component lives. A simple average is sufficient.
      # Convert GridArray to GridVariable for interpolation
      visc_var = GridVariable(visc_comp, u_comp.bc)
      visc_force_on_face = interpolation.linear(visc_var, u_comp.offset)
      
      result_data = u_comp.data - visc_coeff * visc_force_on_face.data
      result_components.append(result_data.ravel())
    
    return jnp.concatenate(result_components)
  
  # Flatten RHS for CG solver
  rhs_flat = jnp.concatenate([comp.data.ravel() for comp in rhs_vector])
  
  # Solve using Conjugate Gradient
  solution_flat, _ = linalg.cg(
      variable_viscosity_operator_flat,
      rhs_flat,
      tol=tol,
      maxiter=maxiter,
      M=preconditioner_fn
  )
  
  # Reshape solution back to grid components
  solution_components = []
  start_idx = 0
  for i in range(ndim):
    end_idx = start_idx + rhs_vector[i].data.size
    u_data = solution_flat[start_idx:end_idx].reshape(grid.shape)
    solution_components.append(GridArray(u_data, rhs_vector[i].offset, grid))
    start_idx = end_idx
  
  return tuple(solution_components)


def solve_varvisc_cg_periodic_vector(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-12,
    preconditioner_fn: Optional[Callable] = None,
) -> GridArrayVector:
  """Solve (I - dt/rho * div [nu(gradu+gradu^T)]) u* = rhs for periodic boundary conditions.
  
  This is identical to the wall version except for boundary condition handling.
  
  Args:
    rhs_vector: Right-hand side vector field as GridArrayVector
    nu_field: Cell-centered viscosity field as JAX array
    dt: Time step size  
    density: Fluid density (default 1.0)
    maxiter: Maximum CG iterations
    tol: CG tolerance
    preconditioner_fn: Optional preconditioner function for CG solver
    
  Returns:
    Solution u* as GridArrayVector
  """
  grid = grids.consistent_grid(*rhs_vector)
  ndim = len(rhs_vector)
  
  # Coefficient for the viscous term
  visc_coeff = dt / density
  
  def variable_viscosity_operator_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
    """Matrix-free operator: (I - dt/rho * div [nu(gradu+gradu^T)]) u"""
    # Reshape flat vector to grid components
    u_components = []
    start_idx = 0
    for i in range(ndim):
      end_idx = start_idx + rhs_vector[i].data.size
      u_data = u_flat[start_idx:end_idx].reshape(grid.shape)
      u_array = GridArray(u_data, rhs_vector[i].offset, grid)
      
      # CRITICAL FIX: Use proper periodic boundary conditions instead of HomogeneousBoundaryConditions
      # HomogeneousBoundaryConditions enforces zero values at boundaries, which kills the flow!
      # For periodic domains, we need actual periodic boundary conditions that handle wrap-around.
      bc = bnew.create_bc(grid)
      u_var = GridVariable(u_array, bc).impose_bc()
      u_components.append(u_var)
      start_idx = end_idx
    
    u_vector = tuple(u_components)
    
    # Apply variable-coefficient viscous operator
    visc_term = div_nu_symgrad_vector(u_vector, nu_field)
    
    # Compute (I - dt/rho * div [nu(gradu+gradu^T)]) u = u - visc_coeff * visc_term
    result_components = []
    for i, (u_comp, visc_comp) in enumerate(zip(u_vector, visc_term)):
      # CRITICAL FIX: Interpolate the cell-centered viscous force to the face
      # locations where the velocity component lives. A simple average is sufficient.
      # Convert GridArray to GridVariable for interpolation
      visc_var = GridVariable(visc_comp, u_comp.bc)
      visc_force_on_face = interpolation.linear(visc_var, u_comp.offset)
      
      result_data = u_comp.data - visc_coeff * visc_force_on_face.data
      result_components.append(result_data.ravel())
    
    return jnp.concatenate(result_components)
  
  # Flatten RHS for CG solver
  rhs_flat = jnp.concatenate([comp.data.ravel() for comp in rhs_vector])
  
  # Solve using Conjugate Gradient
  solution_flat, _ = linalg.cg(
      variable_viscosity_operator_flat,
      rhs_flat,
      tol=tol,
      maxiter=maxiter,
      M=preconditioner_fn
  )
  
  # Reshape solution back to grid components
  solution_components = []
  start_idx = 0
  for i in range(ndim):
    end_idx = start_idx + rhs_vector[i].data.size
    u_data = solution_flat[start_idx:end_idx].reshape(grid.shape)
    solution_components.append(GridArray(u_data, rhs_vector[i].offset, grid))
    start_idx = end_idx
  
  return tuple(solution_components)


def solve_varvisc_gmres_moving_wall_vector(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-12,
    preconditioner_fn: Optional[Callable] = None,
    bc_spec=None,
) -> GridArrayVector:
  """Solve (I - dt/rho * div [nu(gradu+gradu^T)]) u* = rhs for wall boundary conditions using GMRES.
  
  This is the GMRES version of the variable-coefficient solver for the BE-IMEX scheme.
  GMRES is mathematically correct for non-symmetric operators, unlike CG which requires
  symmetric positive-definite systems.
  
  Args:
    rhs_vector: Right-hand side vector field as GridArrayVector
    nu_field: Cell-centered viscosity field as JAX array
    dt: Time step size
    density: Fluid density (default 1.0)
    maxiter: Maximum GMRES iterations
    tol: GMRES tolerance
    preconditioner_fn: Optional preconditioner function for GMRES solver
    
  Returns:
    Solution u* as GridArrayVector
  """
  grid = grids.consistent_grid(*rhs_vector)
  ndim = len(rhs_vector)
  
  # Coefficient for the viscous term
  visc_coeff = dt / density
  
  def variable_viscosity_operator_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
    """Matrix-free operator: (I - dt/rho * div [nu(gradu+gradu^T)]) u"""
    # Reshape flat vector to grid components
    u_components = []
    start_idx = 0
    for i in range(ndim):
      end_idx = start_idx + rhs_vector[i].data.size
      u_data = u_flat[start_idx:end_idx].reshape(grid.shape)
      u_array = GridArray(u_data, rhs_vector[i].offset, grid)
      
      # Operator ghost-cell BC is data-driven via ``bc_spec``;
      # bc_spec=None reproduces the legacy x-periodic, y-Dirichlet channel.
      bc = _operator_velocity_bc(grid, bc_spec)
      u_var = GridVariable(u_array, bc).impose_bc()
      u_components.append(u_var)
      start_idx = end_idx
    
    u_vector = tuple(u_components)
    
    # Apply variable-coefficient viscous operator
    visc_term = div_nu_symgrad_vector(u_vector, nu_field)
    
    # Compute (I - dt/rho * div [nu(gradu+gradu^T)]) u = u - visc_coeff * visc_term
    result_components = []
    for i, (u_comp, visc_comp) in enumerate(zip(u_vector, visc_term)):
      # CRITICAL FIX: Interpolate the cell-centered viscous force to the face
      # locations where the velocity component lives. A simple average is sufficient.
      # Convert GridArray to GridVariable for interpolation
      visc_var = GridVariable(visc_comp, u_comp.bc)
      visc_force_on_face = interpolation.linear(visc_var, u_comp.offset)
      
      result_data = u_comp.data - visc_coeff * visc_force_on_face.data
      result_components.append(result_data.ravel())
    
    return jnp.concatenate(result_components)
  
  # Flatten RHS for GMRES solver
  rhs_flat = jnp.concatenate([comp.data.ravel() for comp in rhs_vector])
  
  # Solve using Generalized Minimal Residual Method (GMRES)
  solution_flat, _ = linalg.gmres(
      variable_viscosity_operator_flat,
      rhs_flat,
      tol=tol,
      maxiter=maxiter,
      M=preconditioner_fn
  )
  
  # Reshape solution back to grid components
  solution_components = []
  start_idx = 0
  for i in range(ndim):
    end_idx = start_idx + rhs_vector[i].data.size
    u_data = solution_flat[start_idx:end_idx].reshape(grid.shape)
    solution_components.append(GridArray(u_data, rhs_vector[i].offset, grid))
    start_idx = end_idx
  
  return tuple(solution_components)


def solve_varvisc_gmres_periodic_vector(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-12,
    preconditioner_fn: Optional[Callable] = None,
) -> GridArrayVector:
  """Solve (I - dt/rho * div [nu(gradu+gradu^T)]) u* = rhs for periodic boundary conditions using GMRES.
  
  This is the GMRES version for periodic boundary conditions, identical to the wall version
  except for boundary condition handling.
  
  Args:
    rhs_vector: Right-hand side vector field as GridArrayVector
    nu_field: Cell-centered viscosity field as JAX array
    dt: Time step size  
    density: Fluid density (default 1.0)
    maxiter: Maximum GMRES iterations
    tol: GMRES tolerance
    preconditioner_fn: Optional preconditioner function for GMRES solver
    
  Returns:
    Solution u* as GridArrayVector
  """
  grid = grids.consistent_grid(*rhs_vector)
  ndim = len(rhs_vector)
  
  # Coefficient for the viscous term
  visc_coeff = dt / density
  
  def variable_viscosity_operator_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
    """Matrix-free operator: (I - dt/rho * div [nu(gradu+gradu^T)]) u"""
    # Reshape flat vector to grid components
    u_components = []
    start_idx = 0
    for i in range(ndim):
      end_idx = start_idx + rhs_vector[i].data.size
      u_data = u_flat[start_idx:end_idx].reshape(grid.shape)
      u_array = GridArray(u_data, rhs_vector[i].offset, grid)
      
      # CRITICAL FIX: Use proper periodic boundary conditions instead of HomogeneousBoundaryConditions
      # HomogeneousBoundaryConditions enforces zero values at boundaries, which kills the flow!
      # For periodic domains, we need actual periodic boundary conditions that handle wrap-around.
      bc = bnew.create_bc(grid)
      u_var = GridVariable(u_array, bc).impose_bc()
      u_components.append(u_var)
      start_idx = end_idx
    
    u_vector = tuple(u_components)
    
    # Apply variable-coefficient viscous operator
    visc_term = div_nu_symgrad_vector(u_vector, nu_field)
    
    # Compute (I - dt/rho * div [nu(gradu+gradu^T)]) u = u - visc_coeff * visc_term
    result_components = []
    for i, (u_comp, visc_comp) in enumerate(zip(u_vector, visc_term)):
      # CRITICAL FIX: Interpolate the cell-centered viscous force to the face
      # locations where the velocity component lives. A simple average is sufficient.
      # Convert GridArray to GridVariable for interpolation
      visc_var = GridVariable(visc_comp, u_comp.bc)
      visc_force_on_face = interpolation.linear(visc_var, u_comp.offset)
      
      result_data = u_comp.data - visc_coeff * visc_force_on_face.data
      result_components.append(result_data.ravel())
    
    return jnp.concatenate(result_components)
  
  # Flatten RHS for GMRES solver
  rhs_flat = jnp.concatenate([comp.data.ravel() for comp in rhs_vector])
  
  # Solve using Generalized Minimal Residual Method (GMRES)
  solution_flat, _ = linalg.gmres(
      variable_viscosity_operator_flat,
      rhs_flat,
      tol=tol,
      maxiter=maxiter,
      M=preconditioner_fn
  )
  
  # Reshape solution back to grid components
  solution_components = []
  start_idx = 0
  for i in range(ndim):
    end_idx = start_idx + rhs_vector[i].data.size
    u_data = solution_flat[start_idx:end_idx].reshape(grid.shape)
    solution_components.append(GridArray(u_data, rhs_vector[i].offset, grid))
    start_idx = end_idx
  
  return tuple(solution_components)


def create_jacobi_preconditioner_fn(
    grid: grids.Grid,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0
) -> Callable:
    """
    Creates a JAX-native Jacobi preconditioner function.

    This function approximates the diagonal of the operator A = (I - dt/rho * L(nu)),
    where L is the variable-viscosity Laplacian-like operator. The preconditioner
    then applies the inverse of this diagonal.

    Args:
        grid: The Grid object.
        nu_field: The cell-centered kinematic viscosity field.
        dt: The time step.
        density: The fluid density.

    Returns:
        A function that takes a flattened residual vector `r` and
        returns the preconditioned vector `M^-^1r`.
    """
    nu_field = lax.stop_gradient(nu_field)
    visc_coeff = dt / density

    # Approximate the diagonal of the variable-coefficient Laplacian L(nu).
    # For a 2D finite difference scheme, the diagonal entry at a point is influenced
    # by the viscosity on its four faces. A common heuristic is to use the
    # sum of coefficients, which for a centered scheme is:
    # diag(L) ~= -2 * nu * (1/dx^2 + 1/dy^2)
    # This is a robust approximation for the diagonal's magnitude.
    laplacian_diag_approx = -2 * nu_field * sum(
        1 / grid.step[axis]**2 for axis in range(grid.ndim)
    )

    # The diagonal of the full operator matrix A = (I - visc_coeff * L)
    # Note: diag(L) is negative, so this term is positive.
    A_diag = 1.0 - visc_coeff * laplacian_diag_approx

    # (safety) clamp tiny values before invert
    eps = 1e-10
    A_diag = jnp.where(jnp.abs(A_diag) < eps, jnp.sign(A_diag) * eps, A_diag)

    # The preconditioner applies the inverse of the diagonal.
    # We pre-compute the inverse to use multiplication, which is more stable.
    M_inv_diag = 1.0 / A_diag

    # The CG solver operates on a flat vector containing all velocity components (ux, uy).
    # We need to build a corresponding flat preconditioner diagonal.
    # Since the operator is similar for both components, we can tile the diagonal.
    # This assumes ux and uy live on grids of the same shape, which is typical.
    num_components = grid.ndim
    component_diagonals = [M_inv_diag.ravel()] * num_components
    M_inv_flat = jnp.concatenate(component_diagonals)

    # This is the actual function that will be passed to the CG solver.
    def jacobi_fn(r_flat):
        """Applies the Jacobi preconditioner: M^-^1r."""
        return M_inv_flat * r_flat

    return jacobi_fn


def create_final_jacobi_preconditioner_fn(
    v: GridVariableVector,
    nu_field: jnp.ndarray,  # Cell-centered kinematic viscosity
    dt: float,
    density: float = 1.0
) -> Callable:
    """
    Creates a JAX-native Jacobi preconditioner correctly matched to the
    nugrad^2u + gradnu.gradu operator discretization on a staggered grid.
    
    This version correctly handles the fact that your div_nu_symgrad_vector 
    function computes nugrad^2u + gradnu.gradu, where only the nugrad^2u term contributes
    to the diagonal. The critical fix is interpolating viscosity to the
    correct face locations where velocity components actually live.
    
    Args:
        v: Velocity field (GridVariableVector) to get grid info and offsets
        nu_field: Cell-centered kinematic viscosity field
        dt: Time step
        density: Fluid density
        
    Returns:
        A function that applies the corrected Jacobi preconditioner: M^-^1r
    """
    nu_field = lax.stop_gradient(nu_field)
    grid = grids.consistent_grid(*v)
    visc_coeff = dt / density

    # Step 1: Interpolate cell-centered viscosity to the face locations
    # where ux and uy actually live. THIS IS THE CRITICAL FIX.
    nu_var = grids.GridVariable(
        grids.GridArray(nu_field, grid.cell_center, grid),
        boundaries.neumann_boundary_conditions(grid.ndim)
    )
    nu_on_ux_loc = interpolation.linear(nu_var, v[0].offset).data
    nu_on_uy_loc = interpolation.linear(nu_var, v[1].offset).data

    # Step 2: Compute the diagonal of the Laplacian operator (grad^2)
    # This is a constant determined by grid spacing.
    dx, dy = grid.step[0], grid.step[1]
    laplacian_diag_geom = -2.0 * (1/dx**2 + 1/dy**2)

    # Step 3: Compute the diagonal of the full operator A = (I - visc_coeff * nugrad^2)
    # for each velocity component using the correctly located viscosity.
    A_diag_ux = 1.0 - visc_coeff * nu_on_ux_loc * laplacian_diag_geom
    A_diag_uy = 1.0 - visc_coeff * nu_on_uy_loc * laplacian_diag_geom

    # (safety) clamp tiny values before invert
    eps = 1e-10
    A_diag_ux = jnp.where(jnp.abs(A_diag_ux) < eps, eps, A_diag_ux)
    A_diag_uy = jnp.where(jnp.abs(A_diag_uy) < eps, eps, A_diag_uy)

    # Step 4: Pre-compute the inverse for the preconditioner function.
    M_inv_diag_ux = 1.0 / A_diag_ux
    M_inv_diag_uy = 1.0 / A_diag_uy

    # Step 5: Flatten for the CG solver.
    M_inv_flat = jnp.concatenate([
        M_inv_diag_ux.ravel(),
        M_inv_diag_uy.ravel()
    ])

    def jacobi_fn(r_flat):
        """Applies the Jacobi preconditioner: M^-^1r."""
        return M_inv_flat * r_flat

    return jacobi_fn


def solve_varvisc_bicgstab_moving_wall_vector(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-7,
    preconditioner_fn: Optional[Callable] = None,
    bc_spec=None,
) -> GridArrayVector:
  """
  Solves the variable-viscosity system using BiConjugate Gradient Stabilized (BiCGSTAB).
  This solver is efficient for non-symmetric systems that are "almost" symmetric.
  """
  grid = grids.consistent_grid(*rhs_vector)
  ndim = len(rhs_vector)
  visc_coeff = dt / density
  
  # The matrix-free operator function is identical for all solvers
  def variable_viscosity_operator_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
    u_components = []
    start_idx = 0
    for i in range(ndim):
      end_idx = start_idx + rhs_vector[i].data.size
      u_data = u_flat[start_idx:end_idx].reshape(grid.shape)
      u_array = grids.GridArray(u_data, rhs_vector[i].offset, grid)
      # Operator ghost-cell BC is data-driven via ``bc_spec``;
      # bc_spec=None reproduces the legacy x-periodic, y-Dirichlet channel.
      bc = _operator_velocity_bc(grid, bc_spec)
      u_var = grids.GridVariable(u_array, bc).impose_bc()
      u_components.append(u_var)
      start_idx = end_idx
    
    u_vector = tuple(u_components)
    visc_term = div_nu_symgrad_vector(u_vector, nu_field)
    
    result_components = []
    for i, (u_comp, visc_comp) in enumerate(zip(u_vector, visc_term)):
      visc_var = grids.GridVariable(visc_comp, u_comp.bc)
      visc_force_on_face = interpolation.linear(visc_var, u_comp.offset)
      result_data = u_comp.data - visc_coeff * visc_force_on_face.data
      result_components.append(result_data.ravel())
    
    return jnp.concatenate(result_components)
  
  rhs_flat = jnp.concatenate([comp.data.ravel() for comp in rhs_vector])
  
  # --- This is the only line that changes ---
  # Call the BiCGSTAB solver with implicit differentiation
  solution_flat = linear_solve_implicit_with_bicgstab(
      matvec=variable_viscosity_operator_flat,
      b=rhs_flat,
      tol=tol, maxiter=maxiter,
      M=preconditioner_fn, MT=None,
      adjoint_mode="normal_cg",        
      adjoint_tol=tol*1e-1, adjoint_maxiter=5*maxiter
  )
    
  # The rest of the function is identical to the others
  solution_components = []
  start_idx = 0
  for i in range(ndim):
    end_idx = start_idx + rhs_vector[i].data.size
    u_data = solution_flat[start_idx:end_idx].reshape(grid.shape)
    solution_components.append(grids.GridArray(u_data, rhs_vector[i].offset, grid))
    start_idx = end_idx
  
  return tuple(solution_components)


def solve_varvisc_bicgstab_moving_wall_vector_per_component(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-7,
    preconditioner_fn: Optional[Callable] = None,
    bc_spec=None,
) -> GridArrayVector:
  """Per-component variant of the moving-wall BiCGSTAB solver.

  Built specifically for the **constant-viscosity** branch of the
  variable-coefficient implicit-viscous solve (i.e. Oldroyd-B
  variant-A with a single solvent viscosity nu_s, and pure Newtonian
  callers). For constant nu the operator inside
  :func:`div_nu_symgrad_vector` reduces to ``nu . grad^2 u_i`` on each
  component -- independent of every other component, since gradnu = 0
  zeroes the only cross-coupling term -- and the joint
  ``(u_x, u_y)`` flat solve done by
  :func:`solve_varvisc_bicgstab_moving_wall_vector` therefore has
  the same solution as two scalar BiCGSTAB solves.

  Why this exists: the joint flat-vector solver was observed to
  break down (NaN sol[1]) when the rhs for one velocity component
  was many orders of magnitude smaller than the other -- exactly the
  case in low-Wi Couette + Oldroyd-B, where the polymer-stress
  divergence delivers a tiny but nonzero u_y rate while u_x carries
  the full wall-driven viscous lift. The pathology is the standard
  BiCGSTAB-on-block-unbalanced-rhs failure: the ``omega = (s, t)/(t, t)``
  scalar in the BiCGSTAB step divides by a near-zero ``(t, t)`` once
  the joint residual is below the absolute tolerance ``tol . ||b||``
  and propagates NaN through every subsequent iterate. Splitting
  per-component gives each subsolve a well-balanced rhs and removes
  the imbalance.

  For variable nu this function is **not** correct -- the gradnu . gradu
  term re-couples components -- and callers must keep using
  :func:`solve_varvisc_bicgstab_moving_wall_vector`. The dispatch
  lives in :func:`equations_rheology.fully_implicit_rheology_stepper`,
  which routes ``model_type='newtonian'`` (the only constant-nu
  model in the registry, used by every variant-A
  ``memory_be_imex_stepper`` caller) through this path and leaves
  every other model on the joint solver.

  A Helmholtz IMEX backend is a deferred alternative, not wired here.
  """
  grid = grids.consistent_grid(*rhs_vector)
  visc_coeff = dt / density

  # All velocity components share the same homogeneous-BC wrapping
  # the joint solver uses; the wall ghost contribution from v's
  # actual BC has already been lifted onto rhs_vector by
  # ``fully_implicit_rheology_stepper``'s Step 4. The operator ghost-cell
  # BC is data-driven via ``bc_spec``; bc_spec=None reproduces the
  # legacy x-periodic, y-Dirichlet channel exactly.
  homo_bc = _operator_velocity_bc(grid, bc_spec)

  # The cell-centered viscosity GridVariable is reused across both
  # component solves; build it once and capture by closure.
  nu_variable = grids.GridVariable(
      grids.GridArray(nu_field, grid.cell_center, grid),
      boundaries.neumann_boundary_conditions(grid.ndim))
  grad_nu = fd.gradient_tensor(nu_variable)

  solution_components = []
  for i, rhs_comp in enumerate(rhs_vector):
    offset = rhs_comp.offset

    # Default-bind ``offset`` so the closure captures the loop value
    # rather than the late-binding Python name.
    def _component_matvec(u_flat, offset=offset):
      u_data = u_flat.reshape(grid.shape)
      u_array = grids.GridArray(u_data, offset, grid)
      u_var = grids.GridVariable(u_array, homo_bc).impose_bc()

      # Single-component viscous force: nu . grad^2u_i + gradnu . gradu_i.
      # Same formula div_nu_symgrad_vector uses per component; we
      # just don't bother packaging the whole vector.
      nu_on_face = interpolation.linear(nu_variable, offset)
      laplacian_u = fd.laplacian(u_var)
      laplacian_term = nu_on_face.data * laplacian_u.data

      grad_u = fd.gradient_tensor(u_var)
      grad_nu_dot_grad_u = 0.0
      for j in range(grid.ndim):
        grad_nu_j_interp = interpolation.linear(grad_nu[j], grad_u[j].offset)
        grad_nu_dot_grad_u = (grad_nu_j_interp.data * grad_u[j].data
                                + grad_nu_dot_grad_u)
      visc_force = laplacian_term + grad_nu_dot_grad_u

      visc_var = grids.GridVariable(
          grids.GridArray(visc_force, offset, grid), homo_bc)
      visc_force_on_face = interpolation.linear(visc_var, offset)
      return (u_data - visc_coeff * visc_force_on_face.data).ravel()

    rhs_flat = rhs_comp.data.ravel()
    solution_flat = linear_solve_implicit_with_bicgstab(
        matvec=_component_matvec,
        b=rhs_flat,
        tol=tol, maxiter=maxiter,
        M=preconditioner_fn, MT=None,
        adjoint_mode="normal_cg",
        adjoint_tol=tol * 1e-1, adjoint_maxiter=5 * maxiter,
    )
    solution_components.append(
        grids.GridArray(solution_flat.reshape(grid.shape), offset, grid))

  return tuple(solution_components)


def solve_varvisc_bicgstab_periodic_vector(
    rhs_vector: GridArrayVector,
    nu_field: jnp.ndarray,
    dt: float,
    density: float = 1.0,
    maxiter: int = 200,
    tol: float = 1e-12,
    preconditioner_fn: Optional[Callable] = None,
) -> GridArrayVector:
  """
  Solves the variable-viscosity system using BiConjugate Gradient Stabilized (BiCGSTAB) for periodic boundary conditions.
  This solver is efficient for non-symmetric systems that are "almost" symmetric.
  """
  grid = grids.consistent_grid(*rhs_vector)
  ndim = len(rhs_vector)
  visc_coeff = dt / density
  
  # The matrix-free operator function is identical for all solvers
  def variable_viscosity_operator_flat(u_flat: jnp.ndarray) -> jnp.ndarray:
    u_components = []
    start_idx = 0
    for i in range(ndim):
      end_idx = start_idx + rhs_vector[i].data.size
      u_data = u_flat[start_idx:end_idx].reshape(grid.shape)
      u_array = grids.GridArray(u_data, rhs_vector[i].offset, grid)
      
      # CRITICAL FIX: Use proper periodic boundary conditions instead of HomogeneousBoundaryConditions
      # HomogeneousBoundaryConditions enforces zero values at boundaries, which kills the flow!
      # For periodic domains, we need actual periodic boundary conditions that handle wrap-around.
      bc = bnew.create_bc(grid)
      u_var = grids.GridVariable(u_array, bc).impose_bc()
      u_components.append(u_var)
      start_idx = end_idx
    
    u_vector = tuple(u_components)
    visc_term = div_nu_symgrad_vector(u_vector, nu_field)
    
    result_components = []
    for i, (u_comp, visc_comp) in enumerate(zip(u_vector, visc_term)):
      visc_var = grids.GridVariable(visc_comp, u_comp.bc)
      visc_force_on_face = interpolation.linear(visc_var, u_comp.offset)
      result_data = u_comp.data - visc_coeff * visc_force_on_face.data
      result_components.append(result_data.ravel())
    
    return jnp.concatenate(result_components)
  
  rhs_flat = jnp.concatenate([comp.data.ravel() for comp in rhs_vector])
  
  # Call the BiCGSTAB solver with implicit differentiation. Adjoint is
  # solved with normal_cg (matches the moving-wall sibling above and
  # the convention in tbnn_gradient_debug_constriction_new_piv).
  solution_flat = linear_solve_implicit_with_bicgstab(
      matvec=variable_viscosity_operator_flat,
      b=rhs_flat,
      tol=tol,
      maxiter=maxiter,
      M=preconditioner_fn,
      MT=None,
      adjoint_mode="normal_cg",
      adjoint_tol=tol * 1e-1,
      adjoint_maxiter=5 * maxiter,
  )
  
  # The rest of the function is identical to the others
  solution_components = []
  start_idx = 0
  for i in range(ndim):
    end_idx = start_idx + rhs_vector[i].data.size
    u_data = solution_flat[start_idx:end_idx].reshape(grid.shape)
    solution_components.append(grids.GridArray(u_data, rhs_vector[i].offset, grid))
    start_idx = end_idx
  
  return tuple(solution_components)


def create_helmholtz_preconditioner_fn(
    grid: grids.Grid,
    nu0: float,
    dt: float,
    helmholtz_solver: Callable,
    component_size: int  # Accept component_size as a static argument
) -> Callable:
    """
    Creates a robust, staggered-grid-aware preconditioner using a Helmholtz solve.
    This version is fully JIT-compatible and differentiable.
    
    Args:
        grid: The Grid object containing spacing information
        nu0: The constant viscosity for the Helmholtz operator (typically max(nu(x)))
        dt: The time step
        helmholtz_solver: The Helmholtz solver function (e.g., solve_helmholtz_fast_diag_moving_wall)
        component_size: Static size of each velocity component (computed outside JIT path)
        
    Returns:
        A function that takes a flattened residual vector `r` and
        returns the preconditioned vector `M^-^1r`.
    """
    nu0 = lax.stop_gradient(nu0)
    def helmholtz_preconditioner(r_flat):
        """
        Takes a flat residual vector, correctly reshapes it, applies the
        Helmholtz solve, and flattens the result.
        """
        # Unpack the flat residual using the provided static component_size.
        r_ux_flat = r_flat[:component_size]
        r_uy_flat = r_flat[component_size:]

        # Reshape each component according to the main grid shape.
        r_ux_reshaped = r_ux_flat.reshape(grid.shape)
        r_uy_reshaped = r_uy_flat.reshape(grid.shape)

        # Create GridArray objects for the solver using the correct offsets.
        r_ux_array = grids.GridArray(r_ux_reshaped, grid.cell_faces[0], grid)
        r_uy_array = grids.GridArray(r_uy_reshaped, grid.cell_faces[1], grid)

        # Apply the Preconditioner
        preconditioned_ux = helmholtz_solver(r_ux_array, alpha=1.0, beta=nu0 * dt)
        preconditioned_uy = helmholtz_solver(r_uy_array, alpha=1.0, beta=nu0 * dt)

        # Flatten and concatenate the results.
        return jnp.concatenate([
            preconditioned_ux.data.ravel(),
            preconditioned_uy.data.ravel()
        ])

    return helmholtz_preconditioner
