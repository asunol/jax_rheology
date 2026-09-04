"""PIV observation operator: window-average a velocity field and add noise.

Models what particle image velocimetry would see: a Hann-windowed separable
average down to interrogation windows, then correlated, spatially varying
noise scaled by the local velocity and its gradient. Applied to the predicted
field during training so the loss compares like with like.
"""
import jax
import jax.numpy as jnp
from jax import vmap

# =============================================================================
# PIV/Noise helpers (inlined from piv_res_noise_terse.py, adapted to (T,H,W))
# =============================================================================

def _create_hann_kernel_1d(W: int) -> jnp.ndarray:
    """1D Hann kernel normalized to sum = 1."""
    W = int(W)
    if W <= 1:
        return jnp.ones((1,), dtype=jnp.float32)
    x = jnp.arange(W, dtype=jnp.float32)
    w = 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * x / jnp.maximum(W - 1, 1)))
    w = w / jnp.maximum(jnp.sum(w), 1e-6)
    return w

def _separable_conv_THW(field_THW: jnp.ndarray, w_x: jnp.ndarray, w_y: jnp.ndarray) -> jnp.ndarray:
    """
    Separable 2D smoothing on (T,H,W) using reflect padding.
    X == width (W) = NHWC axis=2, Y == height (H) = NHWC axis=1.
    """
    f4 = field_THW[..., None]  # (T,H,W,1)
    Wx = int(w_x.shape[0]); Wy = int(w_y.shape[0])

    # Convolve ALONG X (width): pad axis=2
    kx = w_x[None, :, None, None]     # [1, Wx, 1, 1]
    pad_x = Wx // 2
    f4 = jnp.pad(f4, ((0,0), (0,0), (pad_x,pad_x), (0,0)), mode='reflect')
    f4 = jax.lax.conv_general_dilated(
        f4, kx, window_strides=(1,1), padding='VALID',
        dimension_numbers=('NHWC','HWIO','NHWC')
    )

    # Convolve ALONG Y (height): pad axis=1
    ky = w_y[:, None, None, None]     # [Wy, 1, 1, 1]
    pad_y = Wy // 2
    f4 = jnp.pad(f4, ((0,0), (pad_y,pad_y), (0,0), (0,0)), mode='reflect')
    f4 = jax.lax.conv_general_dilated(
        f4, ky, window_strides=(1,1), padding='VALID',
        dimension_numbers=('NHWC','HWIO','NHWC')
    )
    return f4[..., 0]

def piv_downsample_THW(u_THW: jnp.ndarray,
                       v_THW: jnp.ndarray,
                       W_x: int, W_y: int, s_x: int, s_y: int,
                       x_min: float, y_min: float, Lx: float, Ly: float,
                       kernel: str = 'hann'):
    """
    Downsample (T,H,W) velocity fields using Hann-windowed, separable smoothing
    and sampling at interrogation window centers.
    
    Args:
        u_THW, v_THW: Velocity fields, shape (T,H,W) or (H,W)
            H = height (rows), W = width (cols)
        W_x: Window size for width (X-direction/cols)
        W_y: Window size for height (Y-direction/rows)
        s_x: Stride for width (X-direction)
        s_y: Stride for height (Y-direction)
    
    Returns:
        u_ds, v_ds: Downsampled fields (T, H_ds, W_ds)
        x_c, y_c: Physical coordinates of vector centers
    """
    has_time = (u_THW.ndim == 3)
    if not has_time:
        u_THW = u_THW[None, ...]
        v_THW = v_THW[None, ...]
    T, H, W = u_THW.shape

    w_x = _create_hann_kernel_1d(W_x) if kernel == 'hann' else jnp.ones((W_x,), jnp.float32) / float(max(W_x,1))
    w_y = _create_hann_kernel_1d(W_y) if kernel == 'hann' else jnp.ones((W_y,), jnp.float32) / float(max(W_y,1))

    u_f = _separable_conv_THW(u_THW, w_x, w_y)   # (T,H,W)
    v_f = _separable_conv_THW(v_THW, w_x, w_y)

    # Centers along each axis; note W_x applies to width (W), W_y to height (H)
    x_idx = jnp.arange(0, W - W_x + 1, s_x, dtype=jnp.int32) + (W_x // 2)  # width/columns
    y_idx = jnp.arange(0, H - W_y + 1, s_y, dtype=jnp.int32) + (W_y // 2)  # height/rows

    # Sample at centers: (T, H_ds, W_ds)
    u_ds = u_f[:, y_idx, :][:, :, x_idx]
    v_ds = v_f[:, y_idx, :][:, :, x_idx]

    # Sanity: expected vector-grid size (Nx, Ny)
    exp_nx = (W - W_x) // s_x + 1
    exp_ny = (H - W_y) // s_y + 1
    assert (u_ds.shape[-1] == exp_nx) and (u_ds.shape[-2] == exp_ny), (
        f"PIV grid mismatch. Got (Ny,Nx)=({u_ds.shape[-2]},{u_ds.shape[-1]}), "
        f"expected ({exp_ny},{exp_nx}). "
        f"Did W_x/W_y or strides get swapped?"
    )

    if not has_time:
        u_ds = u_ds[0]
        v_ds = v_ds[0]

    # physical centers (align with your grid.domain)
    dx = jnp.float32(Lx / jnp.maximum(W, 1))
    dy = jnp.float32(Ly / jnp.maximum(H, 1))
    x_c = jnp.asarray(x_min, jnp.float32) + dx * x_idx.astype(jnp.float32)
    y_c = jnp.asarray(y_min, jnp.float32) + dy * y_idx.astype(jnp.float32)
    return u_ds, v_ds, x_c, y_c

# --- realistic PIV noise (same spatial noise for all frames) ---

def _gaussian1d_kernel_jax(sigma: float) -> jnp.ndarray:
    """Gaussian 1D kernel with width ~6sigma, normalized."""
    sigma = jnp.asarray(sigma, jnp.float32)
    n = jnp.maximum(1, (jnp.ceil(6.0 * sigma)).astype(jnp.int32))
    n = n + (n % 2 == 0)  # odd
    half = (n // 2).astype(jnp.int32)
    x = jnp.linspace(-half, half, int(n), dtype=jnp.float32)
    w = jnp.exp(-0.5 * (x / jnp.maximum(sigma, 1e-6)) ** 2)
    w = w / jnp.maximum(jnp.sum(w), 1e-6)
    return w

def _central_grad_norm(u: jnp.ndarray, v: jnp.ndarray, dx_vec: float, dy_vec: float) -> jnp.ndarray:
    """Normalized gradient magnitude of speed on the vector grid (per-frame)."""
    s = jnp.sqrt(u**2 + v**2)

    def grad2d(a):
        a_pad = jnp.pad(a, ((1,1),(1,1)), mode='edge')
        axp = a_pad[2:,1:-1]; axm = a_pad[:-2,1:-1]
        ayp = a_pad[1:-1,2:]; aym = a_pad[1:-1,:-2]
        ds_dx = (axp - axm) / (2.0 * dx_vec)
        ds_dy = (ayp - aym) / (2.0 * dy_vec)
        return jnp.sqrt(ds_dx**2 + ds_dy**2)

    G = vmap(grad2d)(s)  # (T,Ny,Nx)
    med = jnp.median(G, axis=(1,2), keepdims=True)
    return G / (med + 1e-6)

def add_piv_noise_jax(u_ds: jnp.ndarray, v_ds: jnp.ndarray, *,
                      W_x: int, W_y: int, s_x: int, s_y: int,
                      Lx: float, Ly: float,
                      key,
                      corr_frac: float = 0.5,
                      sigma_base: float = None,
                      beta_grad: float = 0.7,
                      use_bias: bool = True,
                      full_grid_shape: tuple = (256,128)):
    """
    Add PIV-like noise to downsampled fields. Uses one fixed spatial noise map (same for all frames).
    u_ds,v_ds: (T, H_ds, W_ds)   H_ds ~ Ny, W_ds ~ Nx
    full_grid_shape: (W_full, H_full) is original sim resolution used for physical spacing.
    """
    T, Hds, Wds = u_ds.shape

    # Spatial correlation sigmas in vector-grid units (how many vectors to correlate over)
    sig_x_vec = corr_frac * 0.5 * (W_x / float(max(s_x, 1)))
    sig_y_vec = corr_frac * 0.5 * (W_y / float(max(s_y, 1)))
    wx = _gaussian1d_kernel_jax(sig_x_vec)  # along x (W_ds)
    wy = _gaussian1d_kernel_jax(sig_y_vec)  # along y (H_ds)

    # ONE noise pattern (not time-varying)
    key, ku, kv = jax.random.split(key, 3)
    # Note orientation (H_ds, W_ds)
    Zu_single = jax.random.normal(ku, (Hds, Wds), dtype=jnp.float32)
    Zv_single = jax.random.normal(kv, (Hds, Wds), dtype=jnp.float32)

    # Correlate spatially via separable conv
    Zu = _separable_conv_THW(Zu_single[None, ...], wx, wy)[0]
    Zv = _separable_conv_THW(Zv_single[None, ...], wx, wy)[0]
    Zu = Zu / (jnp.std(Zu) + 1e-6)
    Zv = Zv / (jnp.std(Zv) + 1e-6)
    Zu = Zu[None, ...].repeat(T, axis=0)  # broadcast to all frames
    Zv = Zv[None, ...].repeat(T, axis=0)

    # Heteroscedastic noise scale via normalized |grad| of speed (vector-grid spacings)
    W_full, H_full = full_grid_shape
    dx_full = Lx / float(max(W_full, 1))
    dy_full = Ly / float(max(H_full, 1))
    dx_vec = dx_full * s_x
    dy_vec = dy_full * s_y
    Ghat = _central_grad_norm(u_ds, v_ds, dx_vec, dy_vec)  # (T,H_ds,W_ds)

    speed = jnp.sqrt(u_ds**2 + v_ds**2)
    if sigma_base is None:
        u95 = jnp.asarray(jnp.percentile(speed, 95.0), jnp.float32)
        sigma_base = 0.01 * (u95 + 1e-6)  # default = 1% of U95

    Sigma = sigma_base * (1.0 + beta_grad * Ghat)
    Sigma = jnp.clip(Sigma, a_min=0.0, a_max=5.0 * sigma_base)

    # Optional constant low-frequency bias (same for all frames)
    if use_bias:
        key, kb = jax.random.split(key)
        Bu = jax.random.normal(kb, (Hds, Wds), dtype=jnp.float32)
        Bv = jax.random.normal(kb, (Hds, Wds), dtype=jnp.float32)
        Bu = _separable_conv_THW(Bu[None, ...], wx, wy)[0]
        Bv = _separable_conv_THW(Bv[None, ...], wx, wy)[0]
        Bu = Bu / (jnp.std(Bu) + 1e-6)
        Bv = Bv / (jnp.std(Bv) + 1e-6)
        sigma_bias = 0.1 * sigma_base
        u_ds = u_ds + sigma_bias * Bu[None, :, :]
        v_ds = v_ds + sigma_bias * Bv[None, :, :]

    # Add dynamic noise (same spatial pattern each frame; scaled by Sigma)
    u_noisy = u_ds + Sigma * Zu
    v_noisy = v_ds + Sigma * Zv
    return u_noisy, v_noisy, key
