"""InterScale vs GITIII: per-cell-type reconstruction MSE on CosMx, over 5 seeds.

Trains the lowest-val_mse configuration of sweep sio2fsjm at five seeds, scores every cell, and
plots the per-cell-type mean beside GITIII's published numbers.

The two error bars would mean different things, so only one is drawn:
  * InterScale  -- +/- 1 SD of the per-cell-type mean ACROSS THE 5 SEEDS, i.e. run-to-run
    reproducibility, which is what was asked for.
  * GITIII      -- a single deterministic run, no seed replicates. Its CSV carries an
    across-CELL std (spread between individual cells, n = 3k-184k), which is a different and
    much larger quantity; drawing it next to a 5-run SD would invite a false comparison, so it
    is reported in the printed table instead of on the bars.

MSE follows GITIII's definition exactly: mean over genes of (y_pred - y_true)^2 per cell,
averaged within the receiver cell's own type.

    sbatch ... celltype_reconstruction_mse_seeds.py                 # trains any missing seeds
    python  ... celltype_reconstruction_mse_seeds.py --plot-only    # reuse per-seed CSV
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from celltype_reconstruction_mse import (  # noqa: E402
    GITIII,
    OUT,
    SAVE_DIR,
    build_cfg,
    interscale_per_cell,
    train_or_load,
)

SEEDS = [40, 41, 42, 43, 44]
PER_SEED_CSV = OUT / "interscale_per_celltype_by_seed.csv"

C_IS = "#3E7884"
C_GI = "#BABABE"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

mpl.rcParams.update(
    {
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2, "text.color": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    }
)


def run_seed(seed: int) -> pd.DataFrame:
    """Train (or reuse) one seed and return its per-cell-type mean MSE."""
    cfg = build_cfg(seed)
    ckpt = SAVE_DIR / (
        f"dual_melton25_regr_node_{seed}_{cfg.model.local_component.name}_"
        f"{cfg.model.global_component.name}_model.pt"
    )
    print(f"\n===== seed {seed} ({'reusing checkpoint' if ckpt.exists() else 'training'}) =====", flush=True)

    model, adata = train_or_load(cfg, skip_train=ckpt.exists())
    per_cell, _ = interscale_per_cell(cfg, model, adata)
    per_cell = per_cell.dropna(subset=["mse_global"])

    g = per_cell.groupby("cell_type")
    out = pd.DataFrame(
        {
            "seed": seed,
            "mse_mean": g["mse_global"].mean(),
            "mse_std_cells": g["mse_global"].std(),
            "fvu": g["sse_global"].sum() / g["sst"].sum(),
            "n": g["mse_global"].size(),
        }
    ).reset_index()

    # Same lesson as the sweep: without an explicit release the next seed starts on a card and a
    # host already holding the previous one's model, 387k-cell AnnData and prediction frames.
    del model, adata, per_cell
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--plot-only", action="store_true", help="Reuse the per-seed CSV.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        by_seed = pd.read_csv(PER_SEED_CSV)
    else:
        by_seed = pd.concat([run_seed(s) for s in args.seeds], ignore_index=True)
        by_seed.to_csv(PER_SEED_CSV, index=False)
    print(f"\nper-seed rows: {len(by_seed)}  seeds: {sorted(by_seed['seed'].unique())}")

    agg = by_seed.groupby("cell_type").agg(
        mse=("mse_mean", "mean"), sd_runs=("mse_mean", "std"), n_seeds=("mse_mean", "size"),
        fvu=("fvu", "mean"),
    )

    gi = pd.read_csv(GITIII / "results" / "reconstruction_loss_per_celltype.csv").set_index("cell_type")
    gi = gi.rename(columns={"reconstruction_mse_mean": "mse", "reconstruction_mse_std": "sd_cells"})

    order = agg.sort_values("mse").index.tolist()
    order = [c for c in order if c in gi.index]

    table = pd.DataFrame(
        {
            "InterScale_mse": agg.reindex(order)["mse"],
            "InterScale_sd_across_5_runs": agg.reindex(order)["sd_runs"],
            "InterScale_fvu": agg.reindex(order)["fvu"],
            "GITIII_mse": gi.reindex(order)["mse"],
            "GITIII_sd_across_cells": gi.reindex(order)["sd_cells"],
            "n_cells": agg.reindex(order)["n_seeds"] * 0 + gi.reindex(order)["reconstruction_mse_count"],
        }
    )
    table.to_csv(OUT / "interscale_vs_gitiii_per_celltype.csv")
    print("\n" + table.to_string(float_format=lambda v: f"{v:.4f}"))

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.bar(x - w / 2, agg.reindex(order)["mse"], width=w - 0.03, color=C_IS, zorder=3,
           yerr=agg.reindex(order)["sd_runs"], capsize=4,
           error_kw=dict(lw=1.1, ecolor=INK_2), label="InterScale  (mean ± 1 SD over 5 seeds)")
    ax.bar(x + w / 2, gi.reindex(order)["mse"], width=w - 0.03, color=C_GI, zorder=3,
           label="GITIII  (single run)")

    for xi, v in zip(x, agg.reindex(order)["mse"], strict=True):
        ax.text(xi - w / 2, v * 0.5, f"{v:.3f}", ha="center", va="center", fontsize=7.2,
                color="white", rotation=90)
    for xi, v in zip(x, gi.reindex(order)["mse"], strict=True):
        ax.text(xi + w / 2, v * 0.5, f"{v:.3f}", ha="center", va="center", fontsize=7.2,
                color=INK, rotation=90)

    ax.set_xticks(x, order)
    ax.tick_params(axis="x", rotation=45)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.set_ylabel("Reconstruction MSE\n(mean over genes, per cell)")
    ax.set_xlabel("Cell type (cell_type_coarse)")
    # pad has to clear the two-line subtitle drawn below it, or the two collide.
    ax.set_title("Reconstruction loss per cell type — CosMx pancreas", fontsize=11, pad=54)
    ax.text(
        0.5, 1.015,
        "MSE is computed against each method's own target: GITIII predicts a centred cell-state "
        "residual, InterScale log1p expression.\nThe absolute values are therefore not "
        "interchangeable; the ordering across cell types is.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color=INK_2,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_png = OUT / "interscale_vs_gitiii_per_celltype.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
