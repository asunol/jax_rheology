"""
Shared loader for the metal-crosslinked hydrogel LAOS data (rude/gel/data).

The experiment is a *stress-controlled* large-amplitude oscillatory shear (LAOS)
test at omega = 1 rad/s.  Four amplitudes were recorded:

    gel_1rads_1.csv -> sigma0 = 1 kPa
    gel_1rads_2.csv -> sigma0 = 2 kPa
    gel_1rads_3.csv -> sigma0 = 4 kPa
    gel_1rads_4.csv -> sigma0 = 3 kPa

CSV columns (0-indexed), matching rude/gel/rude.jl:
    col 2  = Normal stress N1            (Pa)
    col 3  = Step time                   (s)
    col 4  = Strain  gamma               (-)
    col 5  = Stress  sigma12 (applied)   (Pa)   -- clean sinusoid, the control variable

Following the RUDE paper we non-dimensionalise with the single-mode Maxwell
linear-response fit:
    G_m   = 37585 Pa     (modulus)
    tau_m = 0.557 s      (relaxation time)
so time' = t / tau_m, stress' = sigma / G_m.  In these units the linear-response
relaxation time is 1 and polymer viscosity eta_p' = G_m*tau_m / (G_m*tau_m) = 1.

This module builds two views of each experiment:

* Reverse / stress-controlled  (``ShearStressData``): forcing = applied
  sigma12'(t'), observable = measured strain gamma(t').  This is the physically
  honest experiment and exercises ``ViscoelasticShearStressProtocol``.
* Forward / strain-rate         (``ShearStrainRateData``): forcing = strain rate
  gamma_dot'(t') reconstructed from the measured strain by finite difference
  (exactly as rude.jl does), observable = sigma12'(t').
"""

from pathlib import Path
import numpy as np
import jax.numpy as jnp

import diff_rheo as dr

# --- linear-response normalisation constants (from rude/gel/rude.jl) ----------
G_M = 37585.0     # Pa
TAU_M = 0.557     # s
ETA_VISC = G_M * TAU_M   # Pa.s  (viscosity normalisation, = eta_p of the Maxwell mode)
OMEGA = 1.0       # rad/s (physical drive frequency)

# amplitude (kPa) of each file index 1..4
AMP_KPA = {1: 1.0, 2: 2.0, 3: 4.0, 4: 3.0}

DATA_DIR = Path(__file__).resolve().parents[2] / "rude" / "gel" / "data"

# column indices (0-indexed)
COL_N1, COL_TIME, COL_STRAIN, COL_SIGMA12 = 2, 3, 4, 5


def _load_raw(idx: int) -> np.ndarray:
    """Load one CSV file as a float array."""
    path = DATA_DIR / f"gel_{int(OMEGA)}rads_{idx}.csv"
    return np.loadtxt(path, delimiter=",")


def load_experiment(idx: int, stride: int = 25):
    """Load and normalise one amplitude.

    Parameters
    ----------
    idx : int
        File index 1..4.
    stride : int
        Sub-sampling stride (paper uses 5).  Larger = fewer points = faster fit.

    Returns
    -------
    dict with normalised numpy arrays:
        t   : time / tau_m
        gamma : strain (dimensionless)
        sigma : sigma12 / G_m
        N1  : (N1 - N1[0]) / G_m
        gammadot : d(gamma)/d(t')  (reconstructed, midpoint finite difference,
                   resampled onto t via linear interp -- as in rude.jl)
        amp_kpa : nominal amplitude
    """
    raw = _load_raw(idx)
    raw = raw[::stride]

    t = raw[:, COL_TIME] / TAU_M
    gamma = raw[:, COL_STRAIN]
    sigma = raw[:, COL_SIGMA12] / G_M
    N1 = (raw[:, COL_N1] - raw[0, COL_N1]) / G_M

    # strain rate by midpoint finite difference (rude.jl convention), then
    # linearly resample onto the sample times t so it aligns with sigma/gamma.
    dgamma = np.diff(gamma)
    dt = np.diff(t)
    gammadot_mid = dgamma / dt
    t_mid = 0.5 * (t[1:] + t[:-1])
    gammadot = np.interp(t, t_mid, gammadot_mid)

    return {
        "t": t,
        "gamma": gamma,
        "sigma": sigma,
        "N1": N1,
        "gammadot": gammadot,
        "amp_kpa": AMP_KPA[idx],
        "idx": idx,
    }


def reverse_experiment(idx: int, stride: int = 25) -> dr.ShearStressData:
    """Stress-controlled view: impose sigma12'(t'), predict strain gamma(t').

    initial_condition is the length-7 state [s11,s22,s33,s13,s23,gammadot,gamma]
    started from rest (zeros) -- consistent with sigma12(0) ~ 0, gamma(0) ~ 0.
    """
    d = load_experiment(idx, stride)
    return dr.ShearStressData(
        time=jnp.asarray(d["t"]),
        data=jnp.asarray(d["gamma"]),
        forcing_data=jnp.asarray(d["sigma"]),
        initial_condition=jnp.zeros(7),
    )


def forward_experiment(idx: int, stride: int = 25) -> dr.ShearStrainRateData:
    """Strain-rate view (paper repro): impose gamma_dot'(t'), predict sigma12'(t')."""
    d = load_experiment(idx, stride)
    return dr.ShearStrainRateData(
        time=jnp.asarray(d["t"]),
        data=jnp.asarray(d["sigma"]),
        forcing_data=jnp.asarray(d["gammadot"]),
        initial_condition=jnp.zeros((3, 3)),
    )


def reverse_batch(indices=(1, 2, 3, 4), stride: int = 25) -> dr.BatchedData:
    return dr.BatchedData.from_data(*[reverse_experiment(i, stride) for i in indices])


def forward_batch(indices=(1, 2, 3, 4), stride: int = 25) -> dr.BatchedData:
    return dr.BatchedData.from_data(*[forward_experiment(i, stride) for i in indices])


def _overview_figure(out_path: Path):
    """Plot the four LAOS experiments: waveforms + Lissajous curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for col, idx in enumerate([1, 2, 3, 4]):
        d = load_experiment(idx, stride=5)
        amp = d["amp_kpa"]
        ax = axes[0, col]
        ax.plot(d["t"], d["sigma"], lw=0.8, label=r"$\sigma_{12}/G_m$")
        ax.plot(d["t"], d["gamma"], lw=0.8, label=r"$\gamma$")
        ax.set_title(f"file {idx}: $\\sigma_0$={amp:.0f} kPa")
        ax.set_xlabel(r"$t/\tau_m$")
        if col == 0:
            ax.legend(fontsize=8)
        ax2 = axes[1, col]
        ax2.plot(d["gamma"], d["sigma"], lw=0.7)
        ax2.set_xlabel(r"$\gamma$")
        ax2.set_ylabel(r"$\sigma_{12}/G_m$")
        ax2.set_title("Lissajous")
    fig.suptitle("Gel LAOS data (stress-controlled, omega=1 rad/s) -- normalised", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # quick sanity print
    for i in (1, 2, 3, 4):
        d = load_experiment(i, stride=25)
        print(f"file {i}: amp={d['amp_kpa']:.0f}kPa  n={len(d['t'])}  "
              f"t'=[{d['t'][0]:.2f},{d['t'][-1]:.2f}]  "
              f"sigma'_max={np.max(np.abs(d['sigma'])):.4f}  "
              f"gamma_range=[{d['gamma'].min():.3f},{d['gamma'].max():.3f}]  "
              f"gammadot'_max={np.max(np.abs(d['gammadot'])):.3f}")
    _overview_figure(out_dir / "gel_data_overview.png")
