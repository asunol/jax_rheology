# Gel LAOS analysis

Model selection on large-amplitude oscillatory shear (LAOS) data for a metal-crosslinked hydrogel. BIC
selects a White–Metzner model, in which viscosity and relaxation time vary with shear rate, over the other
candidates in the `diff_rheo` model library. These results appear in the paper's Supplementary Information.

Measurements are stress-controlled at ω = 1 rad/s and amplitudes of 1, 2, 3, and 4 kPa. Fits use {1, 2, 4}
kPa and hold out 3 kPa. Stress is non-dimensionalised by the SAOS single-mode Maxwell fit
**G_m = 37585 Pa, τ_m = 0.557 s**.

## What ships in this release

`gel_composites_revised.py` redraws three supplementary figures from the committed fit caches under
`results/paper_figures/`. It runs no fits:

| script output | paper figure | content |
|---|---|---|
| `rev_fig2_bic.png` | Supplementary Fig. S18 | BIC ranking of the candidate models relative to the best (White–Metzner) |
| `rev_fig1_fits.png` | Supplementary Fig. S19 | (a) in-sample overlay of data, Oldroyd-B, White–Metzner, and RUDE (σ₁₂, N₁, Lissajous, 3 kPa held out); (b) White–Metzner vs RUDE trained on {1, 2} kPa and extrapolated to 4 kPa |
| `rev_fig3_saos.png` | Supplementary Fig. S20 | SAOS moduli for the several ways of parameterising the backbone |

```bash
cd diff_rheo/scripts/gel
python gel_composites_revised.py     # -> results/paper_figures_revised/rev_fig{1,2,3}*.png
```

Inputs it reads, all committed here: `results/paper_figures/curves.pkl` (forward fits for Oldroyd-B,
White–Metzner, and RUDE), `results/paper_figures/identifiability.pkl` (SAOS-fixed, LAOS-only, and joint
backbone fits), and `results/forward/results.json` (BIC values). `gel_data.py` provides the data loader and
the forward/reverse experiment builders, and defines the normalisation constants above.

`../_paper_style.py` supplies the shared plotting style.

## What does not ship

The fitting pipeline that produced the caches is not included in this release, so the figures can be
redrawn but not refit from raw data.

The raw hydrogel measurements are not redistributed here. They are the LAOS data of Lennon, McKinley, and
Swan, *Scientific machine learning of nonlinear rheology* (PNAS 120, 2023), published with the RUDE method;
obtain them from that paper's data-availability statement. `gel_data.py` expects them as
`gel_1rads_{1..4}.csv` under `diff_rheo/rude/gel/data/`.

The comparison RUDE closure is that paper's shipped network, likewise not redistributed.

## Conventions

- **Forward vs reverse.** *Forward* imposes the strain rate γ̇(t), finite-differenced from the measured
  strain, and predicts σ₁₂; this is the setup used in the paper, and γ̇ is noisy, so forward traces look
  wavy. *Reverse* imposes σ₁₂(t) and predicts strain; the γ̇ inversion divides by η_s, so it requires
  η_s > 0 and adds a solvent contribution to G″.
- **Held-out amplitude.** Fits use 1, 2, and 4 kPa and hold out 3 kPa. The comparison RUDE network was
  trained on the same three amplitudes and predicts 3 kPa, so the hold-out is matched for both models.
- **Normal stress.** N₁ and N₂ follow the sign convention checked by
  `diff_rheo/tests/test_normal_stress_signs.py`.
