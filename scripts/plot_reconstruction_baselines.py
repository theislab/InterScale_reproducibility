"""Figure: InterScale reconstruction against two model-free references.

Answers two questions the sweep results raise and the metrics alone obscure:

  1. The logged cosine similarity looks high (~0.90). Is it? A predictor that emits the SAME
     vector for every cell -- zero cell-specific information -- already scores 0.85. Uncentred
     cosine on non-negative expression data is dominated by the profile all cells share, so it
     cannot distinguish a model that learned something from one that learned the mean.
  2. Why does R2 climb steeply and then stall? R2 = 0 IS "predict the per-gene mean". The climb
     is the model finding the mean profile; the plateau is it finding nothing beyond that.

Inputs (produced by neighbour_mean_baseline.py and a wandb export):
    figures/reconstruction_baselines/baseline_{legnini23,melton25}.json
    figures/reconstruction_baselines/model_best.json

    python scripts/plot_reconstruction_baselines.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent.parent
FIGDIR = HERE / "figures" / "reconstruction_baselines"

# Reference categorical palette, slots 1-3 verbatim. Documented as passing all-pairs CVD and
# normal-vision separation in both modes; colour follows the predictor (the entity), and stays
# identical across every panel. Aqua is below 3:1 on a light surface, so the relief rule applies:
# every bar carries a visible direct label.
C_CONST = "#2a78d6"  # slot 1, blue   - constant predictor
C_MODEL = "#eb6834"  # slot 2, orange - InterScale
C_NBR = "#1baf7a"  # slot 3, aqua   - neighbour mean

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"

mpl.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

DATASETS = [("legnini23", "Legnini — 88 genes, 44k cells"), ("melton25", "CosMx — 979 genes, 387k cells")]
METRICS = [("pearson", "Pearson r"), ("ccc", "Concordance"), ("cosine", "Cosine sim.")]


def load():
    base = {d: json.loads((FIGDIR / f"baseline_{d}.json").read_text()) for d, _ in DATASETS}
    model = json.loads((FIGDIR / "model_best.json").read_text())
    return base, model


def bar_panel(ax, dataset, title, base, model):
    """Grouped bars: three predictors x three bounded metrics.

    R2 is deliberately NOT here. It is unbounded below (-3.09 for the CosMx model) and would
    compress every other bar to a sliver; it gets panel C, where its shape is the point.
    """
    nb = base[dataset]["baselines"]["neighbour mean"]
    const = base[dataset]["baselines"]["constant (train mean profile)"]
    mdl = model[dataset]["summary"]

    series = [
        ("constant", C_CONST, [const["pearson_mean"], const["ccc_mean"], const["cosine_mean"]]),
        ("InterScale (best of 60)", C_MODEL, [mdl["pearson"], mdl["ccc"], mdl["cosine"]]),
        ("neighbour mean", C_NBR, [nb["pearson_mean"], nb["ccc_mean"], nb["cosine_mean"]]),
    ]

    x = np.arange(len(METRICS))
    w = 0.26
    for i, (label, colour, vals) in enumerate(series):
        off = (i - 1) * w
        labelled = False  # the constant series' FIRST bar is the undefined one, so attaching the
        # legend label to xi == 0 would drop it from the legend entirely.
        for xi, v in zip(x, vals, strict=True):
            if v is None or not np.isfinite(v):
                # Pearson of a constant predictor is undefined, not zero: sd_pred = 0 puts a
                # zero in the denominator. Marked rather than silently drawn as a 0-height bar.
                ax.text(xi + off, 0.03, "undefined", ha="center", va="bottom", fontsize=7,
                        color=INK_2, rotation=90)
                continue
            ax.bar(xi + off, v, width=w - 0.03, color=colour, zorder=3,
                   label=None if labelled else label)
            labelled = True
            va, dy = ("bottom", 0.015) if v >= 0 else ("top", -0.015)
            shown = 0.0 if abs(v) < 0.005 else v
            ax.text(xi + off, v + dy, f"{shown:.2f}", ha="center", va=va, fontsize=7.5, color=INK)

    ax.axhline(0, color=INK_2, lw=0.8, zorder=2)
    ax.set_xticks(x, [m[1] for m in METRICS])
    ax.set_ylim(-0.15, 1.08)
    ax.set_ylabel("score (validation)")
    ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def trajectory_panel(ax, model, base, key, ylabel, title, ylim, annotate_start=None):
    """One metric over epochs for the best Legnini run, against the two reference levels."""
    h = model["legnini23"]["history"]
    nb = base["legnini23"]["baselines"]["neighbour mean"]
    const = base["legnini23"]["baselines"]["constant (train mean profile)"]

    ref_key = {"r2": ("r2_median", "cosine_mean"), "cosine": ("cosine_mean", "cosine_mean")}
    nb_val = nb[ref_key[key][0]] if key == "r2" else nb["cosine_mean"]
    const_val = const[ref_key[key][0]] if key == "r2" else const["cosine_mean"]

    def fmt(v):
        return f"{0.0 if abs(v) < 0.005 else v:.2f}"

    # Reference-line labels go on the LEFT, the model's end-of-training label on the RIGHT.
    # In panel D the model lands on 0.90 and the neighbour mean is also 0.90, so labels placed
    # at the same end would sit on top of each other.
    x_left = max(h["epoch"]) * 0.02
    ax.axhline(nb_val, color=C_NBR, lw=2, ls=(0, (5, 3)), zorder=2)
    ax.text(x_left, nb_val, f"neighbour mean {fmt(nb_val)}", ha="left", va="bottom",
            fontsize=7.5, color=C_NBR, zorder=6)
    ax.axhline(const_val, color=C_CONST, lw=2, ls=(0, (5, 3)), zorder=2)
    ax.text(x_left, const_val, f"constant {fmt(const_val)}", ha="left", va="top",
            fontsize=7.5, color=C_CONST, zorder=6)

    ax.plot(h["epoch"], h[key], color=C_MODEL, lw=2, zorder=4, solid_capstyle="round")
    ax.plot(h["epoch"][-1], h[key][-1], "o", ms=5, color=C_MODEL, mec="white", mew=1.5, zorder=5)
    ax.annotate(
        f"InterScale {fmt(h[key][-1])}",
        xy=(h["epoch"][-1], h[key][-1]),
        xytext=(-6, -16), textcoords="offset points",
        ha="right", va="top", fontsize=7.5, color=C_MODEL, zorder=6,
    )

    if annotate_start is not None:
        ax.annotate(
            f"epoch 0: {annotate_start:.1f} (off scale)",
            xy=(0, ylim[0]), xytext=(0.30, 0.06), textcoords="axes fraction",
            fontsize=7.5, color=INK_2,
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8),
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5, color=INK, pad=8)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def main():
    base, model = load()
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.4))

    bar_panel(axes[0, 0], "legnini23", "A  " + DATASETS[0][1], base, model)
    bar_panel(axes[0, 1], "melton25", "B  " + DATASETS[1][1], base, model)

    r2_start = model["legnini23"]["history"]["r2"][0]
    trajectory_panel(axes[1, 0], model, base, "r2", "validation R²",
                     "C  R² rises to the mean, then stalls (Legnini, best run)",
                     ylim=(-2.2, 0.45), annotate_start=r2_start)
    trajectory_panel(axes[1, 1], model, base, "cosine", "validation cosine similarity",
                     "D  Cosine saturates at the no-information level", ylim=(0.0, 1.02))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle(
        "Masked-cell expression reconstruction against model-free references",
        fontsize=11.5, y=0.985,
    )
    fig.text(
        0.5, 0.945,
        "A constant predictor carries no cell-specific information, yet scores 0.85 cosine — "
        "the metric cannot separate learning from predicting the mean.",
        ha="center", fontsize=8.5, color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.935))

    out = FIGDIR / "reconstruction_vs_baselines.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
