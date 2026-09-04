#!/usr/bin/env python3
import argparse, os, json, math, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 12,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 14,
    'figure.titlesize': 16
})
DEFAULT_MODELS = ["Newtonian", "CarreauYasuda", "OldroydB", "Giesekus", "LinearPTT"]
IGNORE_PARAMS = {"observation_noise"}  # exclude from accuracy summaries

# ---------- robust file loading ----------
def load_results_file(path):
    """
    Supports:
      1) JSON object with 'results' (and possibly metadata fields)
      2) Plain JSON list of records
      3) JSONL where FIRST LINE is metadata (with 'parameter_lookup' etc.),
         and subsequent lines are per-sample records.
    Returns (records_list, metadata_dict)
    """
    with open(path, "r") as f:
        blob = f.read().strip()

    # Try whole-file JSON first
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            meta = {k: v for k, v in obj.items() if k != "results"}
            if "results" in obj and isinstance(obj["results"], list):
                return obj["results"], meta
            # If dict but no "results", treat as a single record (rare)
            return [obj], meta
        if isinstance(obj, list):
            return obj, {}
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL with first line = metadata
    records, meta = [], {}
    with open(path, "r") as f:
        first = f.readline().strip()
        try:
            m = json.loads(first)
            if isinstance(m, dict) and ("parameter_lookup" in m or "model_list" in m or "num_runs" in m):
                meta = m
            else:
                # first line is actually a record
                if isinstance(m, dict):
                    records.append(m)
                else:
                    # ignore non-dict first lines
                    pass
        except json.JSONDecodeError:
            pass
        # remaining lines are per-sample records
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                records.append(rec)

    return records, meta

def collect_records(directory, model_names):
    """Find results_l2_<MODEL>.json in directory; return all records + aggregated metadata."""
    all_recs = []
    metas = []
    for m in model_names:
        p = os.path.join(directory, f"results_l2_{m}.json")
        if os.path.exists(p):
            print(f"[load] {p}")
            recs, meta = load_results_file(p)
            all_recs.extend(recs)
            if meta:
                meta["_source_file"] = os.path.basename(p)
                metas.append(meta)
        else:
            print(f"[warn] not found: {p}")
    return all_recs, metas

# ---------- shaping data ----------
def to_long_df(records):
    """Two DataFrames:
       1) sample-level rows for confusion matrix
       2) long-form param rows for accuracy metrics/plots
    """
    samp_rows, param_rows = [], []
    for rec in records:
        true_model = rec.get("model_name")
        best_model = rec.get("best_model_name")
        if true_model is None or best_model is None:
            continue
        samp_rows.append({"true_model": true_model, "best_model": best_model})

        true_params = rec.get("model_parameters", {}) or {}
        est_params  = rec.get("best_params", {}) or {}
        keys = sorted(set(true_params.keys()) & set(est_params.keys()) - IGNORE_PARAMS)
        for k in keys:
            try:
                theta = float(true_params[k])
                theta_hat = float(est_params[k])
            except Exception:
                continue
            param_rows.append({
                "true_model": true_model,
                "best_model": best_model,
                "is_correct": (true_model == best_model),
                "param": k,
                "theta": theta,
                "theta_hat": theta_hat,
            })
    return pd.DataFrame(samp_rows), pd.DataFrame(param_rows)

# ---------- metrics (multiplicative / log-ratio; no bounds) ----------
def multiplicative_metrics(theta, theta_hat):
    theta = np.asarray(theta, float)
    theta_hat = np.asarray(theta_hat, float)
    eps = 1e-12
    F = np.maximum(theta_hat, eps) / np.maximum(theta, eps)  # factor = hat/true
    lnF = np.log(F)
    out = {}
    out["N"] = int(len(F))
    out["median_factor_abs"] = float(np.exp(np.median(np.abs(lnF))))  # “typical factor”
    q16, q50, q84 = np.percentile(lnF, [16, 50, 84])
    out["median_factor"] = float(np.exp(q50))     # geometric median (bias)
    out["factor_q16"] = float(np.exp(q16))
    out["factor_q84"] = float(np.exp(q84))
    for K in [1.1, 1.25, 1.5, 2.0]:
        within = np.mean((F >= 1.0/K) & (F <= K))
        out[f"pct_within_x{K}"] = float(100*within)
    # single %-like headline (median symmetric accuracy)
    out["MSA_percent"] = float(100*(math.exp(np.median(np.abs(lnF))) - 1.0))
    return out

def summarize(param_df, only_correct=True):
    df = param_df.copy()
    if only_correct:
        df = df[df["is_correct"]]
    rows = []
    for (m, p), g in df.groupby(["true_model","param"]):
        vals = {"true_model": m, "param": p}
        vals.update(multiplicative_metrics(g["theta"].values, g["theta_hat"].values))
        rows.append(vals)
    met = pd.DataFrame(rows).sort_values(["true_model","param"])
    overall = {}
    if len(df):
        overall.update(multiplicative_metrics(df["theta"].values, df["theta_hat"].values))
    return met, overall

MODEL_MARKERS = {
    "Newtonian": "o",
    "CarreauYasuda": "s",
    "OldroydB": "^",
    "Giesekus": "+",
    "LinearPTT": "D",
    "FENECR": "v",
}

LINEAR_PARAMS = {"alpha", "n", "zeta", "epsilon"}

PARAM_LATEX = {
    "viscosity": r"$\eta$",
    "solvent_viscosity": r"$\eta_s$",
    "polymer_viscosity": r"$\eta_p$",
    "relaxation_time": r"$\lambda$",
    "alpha": r"$\alpha$",
    "epsilon": r"$\varepsilon$",
    "zeta": r"$\zeta$",
    "n": r"$n$",
    "a": r"$a$",
    "time_constant": r"$K$",
    "b": r"$b$",
    "extensibility": r"$b$",
    "q": r"$q$",
    "Xs": r"$X_s$",
    "Xb": r"$X_b$",
    "observation_noise": r"$\sigma$",
}

def _param_to_latex(name):
    if name in PARAM_LATEX:
        return PARAM_LATEX[name]
    nice = name.replace("_", r"\_")
    return rf"$\mathrm{{{nice}}}$"

def _nice_xlim(x, is_log):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x): 
        return (0.9, 1.1) if not is_log else (1e-2, 1e2)
    if is_log:
        lo = max(1e-12, x.min())
        hi = x.max()
        if lo == hi:  # expand degenerate
            lo *= 0.8; hi *= 1.25
        return lo, hi
    else:
        lo, hi = x.min(), x.max()
        if lo == hi:
            pad = 0.05*max(1.0, abs(lo))
            lo -= pad; hi += pad
        return lo, hi

def to_sample_df(records):
    rows = []
    for r in records:
        tm = r.get("model_name")
        bm = r.get("best_model_name")
        if tm is None or bm is None:
            continue
        rows.append({
            "model_name": tm,
            "best_model_name": bm,
            "is_correct": (tm == bm),
            "model_parameters": r.get("model_parameters", {}) or {},
        })
    return pd.DataFrame(rows)
def build_config_from_metas(metas, fallback_models=DEFAULT_MODELS):
    """
    Combine metadata headers from multiple files.
    - model_list: take the first one we see; fallback to DEFAULT_MODELS.
    - parameter_lookup: merge; expand ranges when they differ; 'log' wins over 'uniform'.
    Returns: {'model_list': [...], 'parameter_lookup': {model: {param: [low, high, scale]}}}
    """
    model_list = None
    param_lookup = {}

    for m in metas:
        if model_list is None and "model_list" in m:
            model_list = list(m["model_list"])

        pl = m.get("parameter_lookup", {})
        for mdl, pmap in pl.items():
            if mdl not in param_lookup:
                param_lookup[mdl] = {}
            for pname, triple in pmap.items():
                lo2, hi2, sc2 = triple
                if pname not in param_lookup[mdl]:
                    param_lookup[mdl][pname] = [float(lo2), float(hi2), sc2]
                else:
                    lo1, hi1, sc1 = param_lookup[mdl][pname]
                    lo = min(lo1, float(lo2))
                    hi = max(hi1, float(hi2))
                    sc = "log" if (str(sc1).lower()=="log" or str(sc2).lower()=="log") else "uniform"
                    param_lookup[mdl][pname] = [lo, hi, sc]

    if model_list is None:
        model_list = list(fallback_models)
    return {"model_list": model_list, "parameter_lookup": param_lookup}


def plot_selected_param_space(df, config, output_path):
    """
    One figure with horizontal 'number-line' strips for:
      - Giesekus: relaxation_time
      - LinearPTT: polymer_viscosity
      - CarreauYasuda: n
    Uses ranges & scales from metadata config = {'model_list': [...], 'parameter_lookup': {...}}.
    Black = correctly identified (marker = TRUE model); Red = misidentified (marker = PREDICTED model).
    """
    model_list = config["model_list"]
    param_lookup = config["parameter_lookup"]

    markers = ['o', 's', '^', 'P', 'D', 'X']
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(model_list)}

    selection = [
        ("Giesekus",     "solvent_viscosity", "log", "Giesekus", "$\\eta_s$"),
        ("LinearPTT",    "polymer_viscosity", "log", "Linear PTT", "$\\eta_p$"),
        # ("CarreauYasuda","n", "linear", "Carreau Yasuda", "$n$"),
    ]

    nrows = len(selection)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 2 * nrows), squeeze=False)
    axes = axes.flatten()

    rng = np.random.default_rng(12345)  # deterministic jitter

    for i, (true_model, param_name, scale, model_name, param_name_latex) in enumerate(selection):
        ax = axes[i]

        if true_model not in param_lookup or param_name not in param_lookup[true_model]:
            ax.set_visible(False)
            continue

        low, high, _ = param_lookup[true_model][param_name]

        subset_df = df[df['model_name'] == true_model].reset_index(drop=True)
        if subset_df.empty:
            ax.set_visible(False)
            continue

        y_jitter = rng.uniform(-0.1, 0.1, size=len(subset_df))

        # misidentified: red, marker by predicted model
        incorrect_df = subset_df[~subset_df['is_correct']]
        for predicted_model, group in incorrect_df.groupby('best_model_name'):
            vals = group['model_parameters'].apply(lambda d: d.get(param_name, np.nan)).astype(float).values
            mask = np.isfinite(vals)
            if mask.any():
                ax.scatter(vals[mask], y_jitter[group.index][mask],
                           color='red',
                           marker=marker_map.get(predicted_model, 'o'),
                           zorder=3)

        # correctly identified: black, marker by TRUE model
        correct_df = subset_df[subset_df['is_correct']]
        if not correct_df.empty:
            vals = correct_df['model_parameters'].apply(lambda d: d.get(param_name, np.nan)).astype(float).values
            mask = np.isfinite(vals)
            if mask.any():
                ax.scatter(vals[mask], y_jitter[correct_df.index][mask],
                           color='black',
                           marker=marker_map.get(true_model, 'o'),
                           zorder=4)

        # styling from metadata
        ax.set_xlim(low, high)
        if str(scale).lower() == 'log':
            ax.set_xscale('log')
        
        # --- CHANGE 1: Update x-axis label ---
        # The model and parameter name are now combined in the x-axis label.
        ax.set_xlabel(f"{model_name} {param_name_latex}", fontsize=16)

        ax.get_yaxis().set_visible(False)
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.hlines(0, low, high, color='gray', linestyle='--', zorder=1)
        
        # --- CHANGE 2: Remove subplot title ---
        # The ax.set_title() line has been removed.

    # Legend: marker per predicted model
    legend_elements = [Line2D([0], [0], marker=m, color='gray', label=name,
                              markersize=8, linestyle='None')
                       for name, m in marker_map.items()]
    
    # --- CHANGE 3: Reposition the legend ---
    # The legend is placed outside the plot axes to prevent overlap.
    # 'bbox_to_anchor=(1.02, 1)' positions the legend's upper-left corner
    # slightly to the right of the figure's top-right corner.
    fig.legend(handles=legend_elements, title="Predicted Model",
               loc='upper left', bbox_to_anchor=(1.01, 1))

    # We use tight_layout to ensure everything fits cleanly without overlapping.
    plt.tight_layout()
    
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_all_param_spaces(df, config, out_dir):
    """
    One PNG per model saved as {out_dir}/{model}_param_space.png.
    Each figure has one horizontal number-line strip per parameter of that model.
    Black = correctly identified (marker = TRUE model); Red = misidentified (marker = PREDICTED model).
    """
    model_list = config["model_list"]
    param_lookup = config["parameter_lookup"]

    markers = ['o', 's', '^', 'P', 'D', 'X']
    marker_map = {m: markers[i % len(markers)] for i, m in enumerate(model_list)}

    rng = np.random.default_rng(12345)

    for true_model in model_list:
        if true_model not in param_lookup:
            continue

        model_params = {p: v for p, v in param_lookup[true_model].items()
                        if p not in IGNORE_PARAMS}
        param_names = sorted(model_params.keys())
        if not param_names:
            continue

        subset_df = df[df['model_name'] == true_model].reset_index(drop=True)
        if subset_df.empty:
            continue

        nrows = len(param_names)
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 2 * nrows), squeeze=False)
        axes = axes.flatten()

        y_jitter = rng.uniform(-0.1, 0.1, size=len(subset_df))

        for i, param_name in enumerate(param_names):
            ax = axes[i]
            low, high, scale = model_params[param_name]

            incorrect_df = subset_df[~subset_df['is_correct']]
            for predicted_model, group in incorrect_df.groupby('best_model_name'):
                vals = group['model_parameters'].apply(
                    lambda d: d.get(param_name, np.nan)).astype(float).values
                mask = np.isfinite(vals)
                if mask.any():
                    ax.scatter(vals[mask], y_jitter[group.index][mask],
                               color='red',
                               marker=marker_map.get(predicted_model, 'o'),
                               zorder=3)

            correct_df = subset_df[subset_df['is_correct']]
            if not correct_df.empty:
                vals = correct_df['model_parameters'].apply(
                    lambda d: d.get(param_name, np.nan)).astype(float).values
                mask = np.isfinite(vals)
                if mask.any():
                    ax.scatter(vals[mask], y_jitter[correct_df.index][mask],
                               color='black',
                               marker=marker_map.get(true_model, 'o'),
                               zorder=4)

            ax.set_xlim(low, high)
            if str(scale).lower() == 'log':
                ax.set_xscale('log')
            ax.set_xlabel(_param_to_latex(param_name), fontsize=16)
            ax.get_yaxis().set_visible(False)
            for spine in ['left', 'right', 'top']:
                ax.spines[spine].set_visible(False)
            ax.hlines(0, low, high, color='gray', linestyle='--', zorder=1)

        legend_elements = [Line2D([0], [0], marker=m, color='gray', label=name,
                                  markersize=8, linestyle='None')
                           for name, m in marker_map.items()]
        fig.legend(handles=legend_elements, title="Predicted Model",
                   loc='upper left', bbox_to_anchor=(1.01, 1))
        fig.suptitle(true_model, fontsize=16)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{true_model}_param_space.png")
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved: {out_path}")


# ---------- plots ----------
def plot_confusion(samp_df, out_png, model_order=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if samp_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Determine display order for both axes
    if model_order is None:
        # Try to honor DEFAULT_MODELS if present; otherwise fall back to present models
        try:
            from __main__ import DEFAULT_MODELS
            default = list(DEFAULT_MODELS)
        except Exception:
            default = []
        present = pd.unique(pd.concat([samp_df["true_model"], samp_df["best_model"]], ignore_index=True))
        model_order = [m for m in default if m in set(present)] + [m for m in present if m not in default]

    # Raw counts
    cm = (
        samp_df.groupby(["true_model", "best_model"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=model_order, columns=model_order, fill_value=0)
    )

    # Row-normalize (true model rows)
    row_sums = cm.sum(axis=1).replace(0, np.nan)
    cm_norm = cm.div(row_sums, axis=0).fillna(0)

    n = len(model_order)
    fig_size = (max(5, 0.9 * n), max(4.2, 0.9 * n))
    fig, ax = plt.subplots(figsize=fig_size, dpi=150)

    # origin='lower' makes the first row/col appear at the bottom-left
    im = ax.imshow(cm_norm.values, vmin=0, vmax=1, origin="lower", interpolation="nearest",cmap="Blues")

    ax.set_xticks(range(n)); ax.set_xticklabels(model_order, rotation=45, ha="right",fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(model_order,fontsize=10)

    # Annotate cells with %
    for i in range(n):
        for j in range(n):
            v = cm_norm.values[i, j]
            ax.text(
                j, i, f"{v*100:.0f}%", ha="center", va="center",
                fontsize=16, color=("white" if v > 0.5 else "black")
            )

    ax.set_xlabel("Identified model")
    ax.set_ylabel("True model")

    # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("row-normalized fraction")

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    return cm, cm_norm


def plot_dotwhisker(met_df, out_png):
    if met_df.empty: return
    models = list(met_df["true_model"].unique())
    params = list(met_df["param"].unique())
    nrows, ncols = len(models), len(params)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 2.4*nrows), dpi=150, squeeze=False)
    for i, m in enumerate(models):
        for j, p in enumerate(params):
            ax = axes[i,j]
            g = met_df[(met_df["true_model"]==m) & (met_df["param"]==p)]
            if len(g)==0:
                ax.axis("off"); continue
            med = g["median_factor"].values[0]
            lo  = g["factor_q16"].values[0]
            hi  = g["factor_q84"].values[0]
            ax.errorbar([0], [med], yerr=[[med-lo],[hi-med]], fmt="o", capsize=4)
            ax.axhline(1.0, lw=1, ls="--")
            for K, ls in [(1.25, ":"), (1.5, "-.")]:
                ax.axhline(K, lw=0.6, ls=ls); ax.axhline(1.0/K, lw=0.6, ls=ls)
            ax.set_yscale("log"); ax.set_xticks([])
            ax.set_title(f"{m} • {p}"); ax.set_ylabel("factor (hat/true)")
    fig.suptitle("Parameter Accuracy (median & 16–84% in factor space)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)

def plot_truth_vs_est(param_df, out_dir, only_correct=True):
    d = param_df.copy()
    if only_correct:
        d = d[d["is_correct"]]
    for p in sorted(d["param"].unique()):
        dp = d[d["param"]==p]
        if dp.empty: continue
        fig, ax = plt.subplots(figsize=(4.5,4.5), dpi=150)
        ax.scatter(dp["theta"], dp["theta_hat"], s=18, alpha=0.7)
        lo = max(1e-12, min(dp["theta"].min(), dp["theta_hat"].min()))
        hi = max(dp["theta"].max(), dp["theta_hat"].max())
        ax.plot([lo,hi],[lo,hi], lw=1.2, ls="--")         # 1:1
        for K, ls in [(1.25, ":"), (1.5, "-."), (2.0, (0,(3,2)))]:
            ax.plot([lo,hi],[lo*K,hi*K], ls=ls, lw=0.8)
            ax.plot([lo,hi],[lo/K,hi/K], ls=ls, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("True"); ax.set_ylabel("Estimate")
        ax.set_title(f"{p} (log–log)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"scatter_{p}.png"), bbox_inches="tight", dpi=300)

# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate and analyze diff_rheo identification/fit results.")
    ap.add_argument("--dir", required=True, help="Directory containing results_l2_<MODEL>.json files")
    ap.add_argument("--out", default="analysis_outputs", help="Output directory for figures/CSV")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated model names to search for")
    ap.add_argument("--include_incorrect", action="store_true",
                    help="Include incorrectly identified cases in parameter metrics/plots")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.out, exist_ok=True)

    records, metas = collect_records(args.dir, models)
    if not records:
        raise SystemExit("No records found. Check --dir and filenames (results_l2_<MODEL>.json).")

    # Save aggregated metadata so you retain the sampling ranges
    if metas:
        with open(os.path.join(args.out, "metadata_aggregate.json"), "w") as f:
            json.dump(metas, f, indent=2)

    samp_df, param_df = to_long_df(records)

    # confusion
    cm, cm_norm = plot_confusion(samp_df, os.path.join(args.out, "confusion.png"))
    if not cm.empty:
        cm.to_csv(os.path.join(args.out, "confusion_counts.csv"))
        cm_norm.to_csv(os.path.join(args.out, "confusion_row_norm.csv"))

    # metrics
    met_df, overall = summarize(param_df, only_correct=(not args.include_incorrect))
    met_df.to_csv(os.path.join(args.out, "parameter_metrics.csv"), index=False)

        # Build config from metadata headers (ranges & scales)
    config = build_config_from_metas(metas)
    # Row-wise DataFrame expected by the plot
    sample_df = to_sample_df(records)

    plot_selected_param_space(
        sample_df,
        config,
        output_path=os.path.join(args.out, "selected_param_space.png"),
    )

    plot_all_param_spaces(sample_df, config, out_dir=args.out)


    # # figures
    # plot_dotwhisker(met_df, os.path.join(args.out, "param_accuracy_grid.png"))
    # plot_truth_vs_est(param_df, args.out, only_correct=(not args.include_incorrect))

    # console summary
    if overall:
        print("\nOverall median symmetric accuracy (all positive params): "
              f"{overall['MSA_percent']:.1f}% (typical factor {overall['median_factor_abs']:.3g})")
    print(f"Saved outputs to: {args.out}")

if __name__ == "__main__":
    main()
