"""What does window overlap change about the interactions the model infers? (legnini23)

Consumes the overlap-ladder sweep (`ghyigj4d`, 4 steps x 3 seeds) and reports the zone-to-zone
interaction matrix at each step. The ladder holds the window WIDTH fixed at two SHH zones and
varies only the stride, so steps 1 and 2 differ in overlap and nothing else:

    step0  width 1               {SHH} {near} {medium} {far}              0 of 6 zone pairs
    step1  width 2, stride 2     {SHH,near} {medium,far}                  2 of 6
    step2  width 2, stride 1     step1 plus {SHH} {near,medium} {far}     3 of 6   <- overlap
    step3  whole sample          {SHH,near,medium,far}                    6 of 6

Steps 0 and 3 reproduce legnini23_coarse and legnini23/node_reg, so the ladder is anchored at both
ends by configurations already in the manuscript.

Three questions, in order:

  1. MEASURABLE   which zone pairs share a window at all. Pure geometry, so it is known before any
                  model is trained -- but it is the ceiling on everything below.
  2. RECOVERED    for the pairs a step CAN measure, does it recover the whole-sample estimate? This
                  is the question overlap has to answer: making an interaction visible is only
                  useful if the value is right.
  3. DILUTION     `compute_hierarchical_net_flow` averages over ALL windows of a sample, and a
                  window in which the pair does not co-occur contributes an exact zero. Every entry
                  is therefore deflated by that pair's co-occurrence fraction. Reported both raw
                  and corrected, because the raw numbers look like a monotone decay with window
                  size that is mostly bookkeeping.

--- the two-column step ------------------------------------------------------------------------

Step 2 is an overlapping tiling represented as two offset partitions. `prepare_geome_dataset` loops
over every column in `sample_key`, so training saw both; but `prepare_a2d_dataset` returns inside
that loop and `get_model_output` therefore only ever builds graphs from `sample_key[0]`. So this
script runs inference once per column, with that column temporarily placed first, and pools the
resulting windows per sample -- which is the correct aggregation for an overlapping tiling, and
means step 2's attention comes from two passes rather than one.

    python scripts/legnini_23/3d_overlap_ladder_flows.py
    python scripts/legnini_23/3d_overlap_ladder_flows.py --steps step1 step2 --seeds 44
    python scripts/legnini_23/3d_overlap_ladder_flows.py --plot-only
"""

from __future__ import annotations

import argparse
import gc
import itertools
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
INTERSCALE = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale")
REGISTRY = INTERSCALE / "config_files" / "registry.yaml"
SWEEP_YAML = INTERSCALE / "config_files" / "sweeps" / "overlap_ladder_legnini.yaml"
RESULTS = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results")
CKPT_DIR = RESULTS / "legnini23_revision" / "overlap_ladder"
OUT = REPO / "figures" / "legnini23" / "sliding_window_ablation"

SWEEP_ID = "ghyigj4d"
WANDB_ENTITY = "francesca-drummer"
WANDB_PROJECT = "InterScale_hyperparameter_sweep"
SWEEP_URL = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{SWEEP_ID}"

ZONES = ["SHH", "near", "medium", "far"]  # ordered by distance from the SHH source
ZONE_COL = "SHH_zone"
SAMPLE_COL = "sample"
STEPS = ["step0", "step1", "step2", "step3"]
SEEDS = [42, 43, 44]
STEP_LABEL = {
    "step0": "width 1",
    "step1": "width 2, stride 2",
    "step2": "width 2, stride 1",
    "step3": "whole sample",
}
STEP_OVERLAP = {"step0": "none", "step1": "none", "step2": "50%", "step3": "n/a"}


def _load_figure_style():
    """Import src/figure_style.py by path, WITHOUT putting src/ on sys.path.

    `sys.path.insert(0, REPO/"src")` would work, but src/ also contains `wandb.py`, which then
    shadows the real wandb package -- `wandb.Api` becomes an AttributeError in any script that
    also talks to the sweep API. Loading the one module by file location avoids the whole class of
    collision.
    """
    import importlib.util

    path = REPO / "src" / "figure_style.py"
    spec = importlib.util.spec_from_file_location("interscale_repro_figure_style", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_fs = _load_figure_style()
FigureStyle, GRID, INK, INK_2 = _fs.FigureStyle, _fs.GRID, _fs.INK, _fs.INK_2

STYLE = FigureStyle(REPO, font_size=9)
CONTINUOUS = STYLE.continuous_cmap
LEGEND_KW = STYLE.legend_kw
TITLE_KW = STYLE.title_kw
SUB_KW = {"fontsize": STYLE.general["title_fontsize"] - 3}
TITLE_PAD_SUB = 24
ZONE_COLORS = STYLE.zones("legnini23", ZONES)
UNMEASURABLE = "#e8e8e6"


# --------------------------------------------------------------------------------------------
# sweep bookkeeping
# --------------------------------------------------------------------------------------------


def fetch_sweep_runs(sweep_id=SWEEP_ID):
    """The ladder's runs, with the values each trial resolved to."""
    import wandb

    api = wandb.Api(timeout=60)
    rows = []
    for run in api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}").runs:
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        s = dict(run.summary)
        rows.append({
            "run_id": run.id, "run_name": run.name, "state": run.state,
            "step": cfg.get("arm"), "seed": cfg.get("optim.seed"),
            "checkpoint_prefix": s.get("checkpoint_prefix"),
            "resolved_sample_key": s.get("resolved_sample_key"),
            "resolved_max_seq_len": s.get("resolved_max_seq_len"),
            "val_concordance_corr": s.get("val_concordance_corr"),
            "val_r2": s.get("val_r2"), "val_pearson_corr": s.get("val_pearson_corr"),
            "test_concordance_corr": s.get("test_concordance_corr"),
            "test_r2": s.get("test_r2"), "epoch": s.get("epoch"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"sweep {sweep_id} has no runs yet")
    df["step_order"] = df["step"].map({s: i for i, s in enumerate(STEPS)})
    return df.sort_values(["step_order", "seed"]).reset_index(drop=True)


def build_cfg(step, seed):
    """Rebuild the config one trial trained, from the registry and the arms yaml."""
    import yaml
    from interscale.config.registry import resolve_config
    from interscale.config.sweep import ARM_PARAM, apply_sweep_config, build_sweep_config, load_arms

    with SWEEP_YAML.open() as f:
        yaml_config = yaml.safe_load(f)
    sweep_config, sweep_params = build_sweep_config(
        yaml_config, prediction_task="regression", model_type="CombinedModel")
    arms = load_arms(yaml_config, sweep_config)
    for name, spec in sweep_config["parameters"].items():
        if name not in (ARM_PARAM, "optim.seed"):
            assert len(spec["values"]) == 1, f"{name} is swept; cannot infer the trial's value"
    trial = {k: v["values"][0] for k, v in sweep_config["parameters"].items()}
    trial[ARM_PARAM] = step
    trial["optim.seed"] = int(seed)

    cfg = resolve_config("legnini23_overlap", "node_reg", registry_path=str(REGISTRY))
    cfg, _ = apply_sweep_config(cfg, "robustness", trial, model_type="CombinedModel",
                                sweep_params=sweep_params, arms=arms)
    cfg.defrost()
    cfg.wandb.use = False
    cfg.model.save = str(CKPT_DIR) + "/"
    cfg.freeze()
    return cfg


def load_model(cfg, adata):
    """Load the checkpoint for this trial.

    `split_key=None`: `_setup_anndata` asserts the split column holds both `train` and `val`, which
    is true here, but the field is unused on the inference path and skipping it keeps this callable
    on any subset.
    """
    import interscale

    interscale.model.CombinedModel._setup_anndata(
        adata=adata, prediction_task=cfg.dataset.prediction_task, layer_key=cfg.dataset.layer_key,
        sample_key_list=list(cfg.dataset.sample_key), prediction_obs=cfg.dataset.prediction_obs,
        split_key=None, view_registry=False)
    return interscale.model.CombinedModel.load(
        str(CKPT_DIR), adata, cfg, local_component=True, global_component=True)


# --------------------------------------------------------------------------------------------
# flows
# --------------------------------------------------------------------------------------------


def window_matrices(attn, zone_codes, n_zones):
    """(sums, counts) of attention per (sender zone, receiver zone) for one window.

    One bincount over a composite index rather than the reference's nested `.loc` loops. The
    diagonal is kept, because `compute_hierarchical_net_flow` includes it in agg[s, s].
    """
    n = attn.shape[0]
    code = (zone_codes[:, None].astype(np.int64) * n_zones + zone_codes[None, :].astype(np.int64))
    a = np.nan_to_num(attn, nan=0.0).astype(np.float64)
    size = n_zones * n_zones
    sums = np.bincount(code.ravel(), weights=a.ravel(), minlength=size).reshape(n_zones, n_zones)
    counts = np.bincount(code.ravel(), minlength=size).reshape(n_zones, n_zones)
    assert counts.sum() == n * n
    return sums, counts


def normalise(agg, compute_net=False):
    """The reference's per-window normalisation. compute_net=False matches the published matrices."""
    net = agg.T - agg if compute_net else agg.T.copy()
    if not compute_net:
        np.fill_diagonal(net, 0.0)
    mx = np.abs(net).max()
    return net / mx if mx > 0 else net


def flows_for_trial(cfg, model, adata_template, compute_net=False):
    """Per-window zone matrices for one trial, pooling every column of `sample_key`.

    Returns a list of dicts, one per window, each carrying the normalised matrix, the sample it
    belongs to, and which zones were present -- the last being what makes an entry measurable.
    """
    n_zones = len(ZONES)
    windows = []
    original_cfg = model._cfg
    try:
        windows = _flows_over_columns(cfg, model, adata_template, compute_net, n_zones)
    finally:
        model._cfg = original_cfg
    return windows


def _flows_over_columns(cfg, model, adata_template, compute_net, n_zones):
    """One inference pass per sample_key column, pooling the windows."""
    from interscale.tl import set_full_reproducibility

    windows = []
    for col in list(cfg.dataset.sample_key):
        # get_model_output builds graphs from sample_key[0] only, so each column takes its turn
        # there. A fresh AnnData per pass: get_model_output renames obs_names and writes obsm.
        cfg_col = cfg.clone()
        cfg_col.defrost()
        cfg_col.dataset.sample_key = [col]
        cfg_col.freeze()

        adata = adata_template.copy()
        adata.obs[col] = adata.obs[col].astype("category").cat.remove_unused_categories()
        zone_codes = pd.Categorical(adata.obs[ZONE_COL].astype(str), categories=ZONES).codes
        sample = adata.obs[SAMPLE_COL].astype(str).to_numpy()
        win_lab = adata.obs[col].astype(str).to_numpy()

        set_full_reproducibility(int(cfg.optim.seed))
        model._cfg = cfg_col          # get_model_output reads sample_key off the model's own cfg
        out = model.get_model_output(adata, prefix="")
        attn = out.obsm["_attn_matrix"]

        order = np.argsort(win_lab, kind="stable")
        sorted_lab = win_lab[order]
        bounds = np.flatnonzero(np.r_[True, sorted_lab[1:] != sorted_lab[:-1]])
        for idx in np.split(order, bounds[1:]):
            n = len(idx)
            assert n <= attn.shape[1], (
                f"window of {n} cells but attention is {attn.shape[1]} wide; max_seq_len too small")
            block = np.asarray(attn[idx][:, :n], dtype=np.float32)
            zc = zone_codes[idx]
            sums, counts = window_matrices(block, zc, n_zones)
            with np.errstate(invalid="ignore", divide="ignore"):
                agg = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
            present = np.zeros(n_zones, dtype=bool)
            present[np.unique(zc)] = True
            windows.append({
                "column": col, "window": win_lab[idx[0]],
                "sample": sample[idx[0]], "n_cells": n,
                "matrix": normalise(agg, compute_net), "present": present,
            })
        del adata, out, attn
        gc.collect()
    return windows


def aggregate(windows, n_zones=len(ZONES)):
    """Window -> sample -> all_samples, plus the co-occurrence bookkeeping.

    `cooccur` counts the windows in which both zones of a pair are present; it is what makes an
    entry measurable at all, and its fraction is the factor the hierarchical mean deflates by.
    """
    per_sample, cooc = {}, np.zeros((n_zones, n_zones), dtype=int)
    for w in windows:
        per_sample.setdefault(w["sample"], []).append(w["matrix"])
        cooc += np.outer(w["present"], w["present"]).astype(int)
    stacked = np.stack([np.nanmean(np.stack(v), axis=0) for v in per_sample.values()])
    n_win = len(windows)
    frac = cooc / n_win if n_win else np.zeros_like(cooc, dtype=float)
    return {
        "mean": np.nanmean(stacked, axis=0), "std": np.nanstd(stacked, axis=0),
        "n_samples": stacked.shape[0], "n_windows": n_win,
        "cooccur": cooc, "frac_cooccur": frac,
    }


def measurable_from_geometry(adata, columns):
    """Which zone pairs share a window, from `obs` alone -- no model involved."""
    out = set()
    for col in columns:
        for _, g in adata.obs.groupby(col, observed=True):
            zs = sorted(set(g[ZONE_COL].astype(str)), key=ZONES.index)
            out |= {frozenset(p) for p in itertools.combinations(zs, 2)}
    return out


def run(steps, seeds, runs, compute_net=False):
    rows = []
    finished = runs[runs["state"] == "finished"]
    for step in steps:
        for seed in seeds:
            row = finished[(finished["step"] == step) & (finished["seed"] == seed)]
            if row.empty:
                print(f"skipping {step} seed {seed}: no finished run")
                continue
            cfg = build_cfg(step, seed)
            from interscale.tl.utils import get_model_filename_prefix
            prefix = get_model_filename_prefix(cfg, True, True)
            recorded = row.iloc[0]["checkpoint_prefix"]
            if isinstance(recorded, str) and recorded and recorded != prefix:
                raise RuntimeError(f"rebuilt config mismatch: {prefix!r} vs run's {recorded!r}")
            print(f"\n===== {step} seed {seed} | {list(cfg.dataset.sample_key)} | "
                  f"max_seq_len {cfg.model.global_component.parameters.max_seq_len} =====", flush=True)

            from interscale.tl import remove_zero_expression_cells
            adata = remove_zero_expression_cells(sc.read_h5ad(cfg.dataset.h5ad_data))
            model = load_model(cfg, adata)
            windows = flows_for_trial(cfg, model, adata, compute_net=compute_net)
            res = aggregate(windows)
            geo = measurable_from_geometry(adata, list(cfg.dataset.sample_key))
            print(f"  {res['n_windows']} windows over {res['n_samples']} samples; "
                  f"{len(geo)}/6 zone pairs share a window")

            for i, s in enumerate(ZONES):
                for j, r in enumerate(ZONES):
                    rows.append({
                        "step": step, "step_label": STEP_LABEL[step],
                        "overlap": STEP_OVERLAP[step], "seed": seed,
                        "sender": s, "receiver": r,
                        "flow": float(res["mean"][i, j]), "flow_sd": float(res["std"][i, j]),
                        "cooccur": int(res["cooccur"][i, j]),
                        "frac_cooccur": float(res["frac_cooccur"][i, j]),
                        "measurable": (i != j) and (frozenset({s, r}) in geo),
                        "n_windows": res["n_windows"], "n_samples": res["n_samples"],
                    })
            del model, adata, windows
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    df = pd.DataFrame(rows)
    # Corrected for the dilution: an entry averaged over windows where the pair is absent carries a
    # factor equal to its co-occurrence fraction.
    df["flow_corrected"] = np.where(df["frac_cooccur"] > 0,
                                    df["flow"] / df["frac_cooccur"].replace(0, np.nan), np.nan)
    return df


# --------------------------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------------------------


def fig_ladder(df, path=None, value="flow_corrected"):
    steps = [s for s in STEPS if s in set(df["step"])]
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(3.9 * (len(steps) + 1), 4.9),
                             gridspec_kw={"width_ratios": [1] * len(steps) + [1.15]})

    for k, step in enumerate(steps):
        ax = axes[k]
        d = df[df["step"] == step].groupby(["sender", "receiver"], as_index=False).agg(
            v=(value, "mean"), measurable=("measurable", "first"), nw=("n_windows", "first"))
        m = d.pivot(index="sender", columns="receiver", values="v").reindex(ZONES, axis=0).reindex(ZONES, axis=1)
        ok = d.pivot(index="sender", columns="receiver", values="measurable").reindex(ZONES, axis=0).reindex(ZONES, axis=1)
        v = m.to_numpy(dtype=float)
        okv = ok.to_numpy(dtype=bool)
        diag = np.eye(len(ZONES), dtype=bool)

        ax.imshow(np.ma.masked_where(~okv, v), cmap=CONTINUOUS, vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(okv, np.ones_like(v)),
                  cmap=mpl.colors.ListedColormap([UNMEASURABLE]), vmin=0, vmax=1)
        for i in range(len(ZONES)):
            for j in range(len(ZONES)):
                if diag[i, j]:
                    ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=INK_2)
                elif not okv[i, j]:
                    ax.text(j, i, "n.m.", ha="center", va="center", fontsize=7.5, color="#8a8a86")
                else:
                    ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                            color="white" if v[i, j] < 0.55 else INK)
        ax.set_xticks(range(len(ZONES)), ZONES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(ZONES)), ZONES if k == 0 else [""] * len(ZONES), fontsize=8)
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if lab.get_text():
                lab.set_color(ZONE_COLORS[lab.get_text()])
        ax.set_title(step, **TITLE_KW, pad=TITLE_PAD_SUB)
        ax.text(0.5, 1.005, f"{STEP_LABEL[step]} · overlap {STEP_OVERLAP[step]}",
                transform=ax.transAxes, ha="center", va="bottom", color=INK_2, **SUB_KW)
        ax.set_xlabel("receiver zone", fontsize=8.5)
        if k == 0:
            ax.set_ylabel("sender zone", fontsize=8.5)
        ax.text(0.5, -0.30, f"{int(okv.sum())}/12 measurable · {int(d['nw'].iloc[0])} windows",
                transform=ax.transAxes, ha="center", fontsize=7.5, color=INK_2)

    # Recovery: for the pairs each step CAN measure, how close to the whole-sample estimate?
    ax = axes[-1]
    ref = (df[df["step"] == "step3"].groupby(["sender", "receiver"])[value].mean())
    for k, step in enumerate(steps):
        d = df[(df["step"] == step) & df["measurable"]]
        if d.empty:
            continue
        g = d.groupby(["sender", "receiver"]).agg(v=(value, "mean"), sd=(value, "std"))
        r = ref.reindex(g.index)
        ax.errorbar(np.full(len(g), k) + np.linspace(-0.16, 0.16, len(g)),
                    (g["v"] / r).to_numpy(), yerr=(g["sd"] / r).to_numpy(),
                    fmt="o", ms=6, capsize=3, lw=1,
                    color=STYLE.sequential(len(steps))[k], ecolor=INK_2, elinewidth=0.8)
    ax.axhline(1.0, color=INK_2, lw=1.2, ls=(0, (5, 3)))
    ax.set_xticks(range(len(steps)), [s.replace("step", "") for s in steps])
    ax.set_xlabel("ladder step", fontsize=8.5)
    ax.set_ylabel("value ÷ whole-sample value", fontsize=8.5)
    ax.set_title("Recovery", **TITLE_KW, pad=TITLE_PAD_SUB)
    ax.text(0.5, 1.005, "one point per measurable pair", transform=ax.transAxes,
            ha="center", va="bottom", color=INK_2, **SUB_KW)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    fig.suptitle("Overlap makes zone interactions measurable without changing their value",
                 **TITLE_KW, y=1.02)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--compute-net", action="store_true",
                    help="net flow (agg.T - agg) instead of raw incoming; the published legnini "
                         "matrices use raw incoming, which is the default here.")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        df = pd.read_csv(OUT / "overlap_ladder_flows.csv")
    else:
        runs = fetch_sweep_runs()
        runs.to_csv(OUT / "overlap_ladder_runs.csv", index=False)
        print(f"sweep {SWEEP_ID}: {len(runs)} runs ({SWEEP_URL})")
        print(runs[["run_name", "state", "step", "seed", "val_concordance_corr", "val_r2"]]
              .to_string(index=False))
        df = run(args.steps, args.seeds, runs, compute_net=args.compute_net)
        df.to_csv(OUT / "overlap_ladder_flows.csv", index=False)

    und = df[df["sender"] < df["receiver"]]
    print("\n--- 1. MEASURABLE: zone pairs that share a window ---")
    for step in [s for s in STEPS if s in set(df["step"])]:
        d = und[(und["step"] == step)].groupby(["sender", "receiver"])["measurable"].first()
        pairs = [f"{a}-{b}" for (a, b), ok in d.items() if ok]
        print(f"  {step} ({STEP_LABEL[step]:18s} overlap {STEP_OVERLAP[step]:4s}): "
              f"{len(pairs)}/6  {pairs or '(none)'}")

    print("\n--- 2. RECOVERED: measurable entries relative to the whole sample ---")
    ref = df[df["step"] == "step3"].groupby(["sender", "receiver"])["flow_corrected"].mean()
    for step in [s for s in STEPS if s in set(df["step"]) and s != "step3"]:
        d = df[(df["step"] == step) & df["measurable"]]
        if d.empty:
            print(f"  {step}: nothing measurable")
            continue
        g = d.groupby(["sender", "receiver"])["flow_corrected"].mean()
        ratio = (g / ref.reindex(g.index)).dropna()
        print(f"  {step}: n={len(ratio):2d}  mean ratio {ratio.mean():.3f}  "
              f"range {ratio.min():.3f}-{ratio.max():.3f}")

    print("\n--- 3. DILUTION: raw vs corrected magnitude ---")
    for step in [s for s in STEPS if s in set(df["step"])]:
        d = df[(df["step"] == step) & df["measurable"]]
        if d.empty:
            continue
        print(f"  {step}: median |raw| {d['flow'].abs().median():.4f}  "
              f"median |corrected| {d['flow_corrected'].abs().median():.4f}  "
              f"median co-occurrence {d['frac_cooccur'].median():.3f}")

    fig_ladder(df, OUT / "fig_overlap_ladder.png")
    print(f"\ndone. table and figure in {OUT}")


if __name__ == "__main__":
    main()
