"""Can a given cell-type interaction be measured at all at a given window size?

This is the quantity that matters for reviewer comment 5, and it is not the same as the fraction of
individual cell pairs severed by a boundary. Losing 35% of Alpha-Beta cell pairs is statistical
attrition -- the remaining 65% still estimate an Alpha->Beta interaction perfectly well. What
removes an interaction is a window in which the two types never appear TOGETHER.

So the metric here is per cell-type pair (A, B):

    cooccur        number of windows containing at least one A and at least one B
    cooccur_min3   number of windows containing at least three of each -- a window with one A and
                   one B contributes a single-pair "mean", which is not an estimate
    close_<D>      number of windows containing an A-B pair separated by at most D px

Both datasets are covered in one figure, because the metric is what makes them comparable and they
sit in opposite regimes:

  melton25 (CosMx pancreas, geometric windows, unstructured tissue)
      NO cell-type pair ever fails to co-occur, at any of the six window sizes -- 0 of 72 ordered
      pairs. The window does not remove interactions here. What it does is undersample them: the
      worst pair (Beta-Mast) co-occurs in 37 of 3426 windows at 400 px and has three or more cells
      of each type in only TWO of them, against 46 of 100 windows at 3000 px.

  legnini23 (Molecular Cartography, zone-aligned windows, deliberately structured)
      Pairs fail outright. Every window of the `coarse` arm holds a single SHH zone, so no zone pair
      co-occurs anywhere and the whole matrix is unmeasurable; the two three-zone arms each lose one
      zone entirely.

--- a methodological consequence worth acting on -----------------------------------------------

`compute_hierarchical_net_flow` averages each entry over ALL windows of a sample, and a window in
which the pair does not co-occur contributes an exact zero. Every entry therefore carries a hidden
factor equal to that pair's co-occurrence fraction, which shrinks with the window and shrinks
hardest for rare or spatially segregated pairs. Measured here, the median |flow| runs 0.0041 at
400 px against 0.0681 at 3000 px -- a 17x spread that is mostly bookkeeping, not biology. Dividing
by the co-occurrence fraction removes most of it (0.047 against 0.108).

This is the same mechanism as the single-zone dilution in the legnini arms, and it means a small
entry in a published flow matrix should not be read as a weak interaction without checking how many
windows could have contributed to it.

Two things it does NOT fix, stated because they were tested: rescaling does not improve the
between-arm agreement of the melton entries (mean between-arm Spearman 0.275 raw, 0.266 rescaled),
and the per-pair between-seed spread does not track undersampling -- poorly sampled entries are
stably near zero rather than noisy. The melton cell-type flow matrix remains unreliable at these
model fits; only the co-occurrence geometry above is dependable.

    python scripts/melton_jimenez_25/5d_celltype_measurability.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
RESULTS = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results")
DATA = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data")
ATT = RESULTS / "melton25_revision" / "sliding_window" / "attention"
OUT = REPO / "figures" / "melton25" / "sliding_window_ablation"

# --------------------------------------------------------------------------------------------
# figure style -- everything visual comes from figures/config.yml via src/figure_style.py, so
# these figures match the rest of the manuscript rather than carrying their own palette.
# --------------------------------------------------------------------------------------------


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
CONTINUOUS = STYLE.continuous_cmap      # 'viridis'
DIVERGENT = STYLE.divergent_cmap        # 'bwr'
LEGEND_KW = STYLE.legend_kw
# Panel titles at config.yml's title_fontsize (14, bold) leave no room for the sentence-long
# titles these figures used to carry, so the explanation moved to the figure captions in
# reviewer_response/comment5_sliding_windows.md, which is where it belongs for publication.
TITLE_KW = STYLE.title_kw
SUB_KW = {"fontsize": STYLE.general["title_fontsize"] - 3}
# Clearance for a 14 pt bold title when a sub-line is drawn just above the axes at y=1.005.
# pad=6 puts the two on top of each other.
TITLE_PAD_SUB = 24



PX_UM_COSMX = 0.12028
PX_UM_RESOLVE = 0.138
ARMS = ["w400", "w800", "w1200", "w1600", "w2000", "w3000"]
# Cells of EACH type a window must contain before it counts toward an interaction. A window
# holding one cell of each contributes a "mean" over a single cell pair, which is not an estimate;
# at 10 of each it contributes 100 pairs. This is the primary criterion -- `cooccur_any` below
# keeps the >=1 version only as a reference, to show how much of the apparent co-occurrence rests
# on windows too thin to estimate anything from.
MIN_CELLS = 10

LEGNINI_ARM_COL = {
    "full_slide": "sample",
    "medium_snm": "sliding_window_high_SHH_near_medium",
    "medium_nmf": "sliding_window_high_near_medium_far",
    "coarse": "sliding_window_low",
}
ZONES = ["SHH", "near", "medium", "far"]
ZONE_COLORS = STYLE.zones("legnini23", ZONES)
CT_COLORS = STYLE.cell_types("melton25")

# Two-series contrast, both derived from config.yml rather than picked by hand: a mid tone of the
# continuous colormap for the typical case, and the warm end of the divergent colormap for the
# worst case, where red carries "this is the problem".
C_TYPICAL = mpl.colormaps[CONTINUOUS](0.45)
C_WORST = mpl.colormaps[DIVERGENT](0.92)




def melton_measurability(cutoffs_px=(200, 800, 1600)):
    """Per cell-type pair and window size: in how many windows are both types present?

    Reads `per_type` (cells of each type per window) and `counts` (A-B pair counts per distance bin
    per window) that 5a already wrote, so no pairwise recomputation is needed.
    """
    ref = np.load(ATT / "reference_pair_distances.npz", allow_pickle=True)
    cts = list(ref["cell_types"].astype(str))
    edges = ref["edges"]
    K = len(cts)

    rows = []
    for arm in ARMS:
        f = ATT / "windows" / f"{arm}.npz"
        if not f.exists():
            continue
        w = np.load(f, allow_pickle=True)
        pt = w["per_type"]
        counts = w["counts"]
        n_win = pt.shape[0]
        any_cell = pt > 0                 # reference only: "the type is present at all"
        present = pt >= MIN_CELLS         # primary criterion
        for i in range(K):
            for j in range(K):
                rec = {
                    "dataset": "melton25", "arm": arm, "window_px": int(arm[1:]),
                    "window_um": int(arm[1:]) * PX_UM_COSMX, "sender": cts[i], "receiver": cts[j],
                    "n_windows": n_win,
                    "cooccur": int((present[:, i] & present[:, j]).sum()),
                    "cooccur_any": int((any_cell[:, i] & any_cell[:, j]).sum()),
                }
                rec["frac_cooccur"] = rec["cooccur"] / n_win
                rec["frac_cooccur_any"] = rec["cooccur_any"] / n_win
                for c in cutoffs_px:
                    sel = edges[1:] <= c
                    rec[f"close_{c}"] = int((counts[:, i, j, sel].sum(axis=1) > 0).sum())
                rows.append(rec)
    return pd.DataFrame(rows), cts


def legnini_measurability():
    """Same metric on the zone-aligned legnini windows, where pairs do fail outright."""
    adata = sc.read_h5ad(DATA / "legnini23_pp.h5ad")
    rows = []
    for arm, col in LEGNINI_ARM_COL.items():
        obs = adata.obs[[col, "SHH_zone"]].dropna()
        tab = pd.crosstab(obs[col], obs["SHH_zone"]).reindex(columns=ZONES, fill_value=0)
        tab = tab[tab.sum(axis=1) > 0]
        pt = tab.to_numpy()
        n_win = pt.shape[0]
        any_cell, present = pt > 0, pt >= MIN_CELLS
        for i, s in enumerate(ZONES):
            for j, r in enumerate(ZONES):
                rows.append({
                    "dataset": "legnini23", "arm": arm, "sender": s, "receiver": r,
                    "n_windows": n_win,
                    "cooccur": int((present[:, i] & present[:, j]).sum()),
                    "cooccur_any": int((any_cell[:, i] & any_cell[:, j]).sum()),
                    "frac_cooccur": float((present[:, i] & present[:, j]).mean()),
                    "frac_cooccur_any": float((any_cell[:, i] & any_cell[:, j]).mean()),
                })
    return pd.DataFrame(rows)


def fig_measurability(mel, cts, leg, path=None):
    """Two regimes side by side: undersampled (melton) vs unmeasurable (legnini)."""
    mel_arms = ["w400", "w1600", "w3000"]
    leg_arms = ["full_slide", "medium_snm", "coarse"]
    fig = plt.figure(figsize=(16.5, 9.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.62, wspace=0.32)

    # --- top: melton co-occurrence matrices -------------------------------------------------
    for k, arm in enumerate(mel_arms):
        ax = fig.add_subplot(gs[0, k])
        d = mel[mel["arm"] == arm]
        m = (d.pivot(index="sender", columns="receiver", values="frac_cooccur")
               .reindex(index=cts, columns=cts))
        v = m.to_numpy()
        ax.imshow(v, cmap=CONTINUOUS, vmin=0, vmax=1)
        for i in range(len(cts)):
            for j in range(len(cts)):
                ax.text(j, i, f"{v[i, j]:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=5.2, color="white" if v[i, j] < 0.6 else INK)
        ax.set_xticks(range(len(cts)), cts, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(cts)), cts if k == 0 else [""] * len(cts), fontsize=6.5)
        # Tick labels in the config.yml cell-type colours, so a type is recognisable across figures.
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if lab.get_text():
                lab.set_color(CT_COLORS[lab.get_text()])
        n = d["n_windows"].iloc[0]
        ax.set_title(f"{int(arm[1:])*PX_UM_COSMX:.0f} µm window", **TITLE_KW, pad=TITLE_PAD_SUB)
        ax.text(0.5, 1.005, f"{n} windows · min {v[~np.eye(len(cts), dtype=bool)].min():.2f}",
                transform=ax.transAxes, ha="center", va="bottom", color=INK_2, **SUB_KW)
        if k == 0:
            ax.set_ylabel("melton25 — CosMx pancreas\ngeometric windows\n\nsender", fontsize=8.5)
        ax.set_xlabel("receiver", fontsize=8)

    # --- top right: the summary curve --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 3])
    off = mel[mel["sender"] != mel["receiver"]]
    g = off.groupby("window_um").agg(
        worst=("frac_cooccur", "min"), median=("frac_cooccur", "median"),
        worst_any=("frac_cooccur_any", "min")).reset_index()
    ax.plot(g["window_um"], g["median"], marker="o", ms=6, lw=2, color=C_TYPICAL,
            label="median pair")
    ax.plot(g["window_um"], g["worst"], marker="s", ms=6, lw=2, color=C_WORST,
            label="worst pair")
    ax.plot(g["window_um"], g["worst_any"], marker="^", ms=6, lw=2, color=C_WORST, ls=(0, (4, 2)),
            label="worst pair, ≥1 cell each")
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("window size (µm)")
    ax.set_ylabel("fraction of windows in which\nboth cell types are present")
    ax.set_title(f"≥{MIN_CELLS} cells of each", **TITLE_KW, pad=6)
    ax.legend(**{**LEGEND_KW, "prop": {"size": 8}})
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    # --- bottom: legnini, where it does reach zero -------------------------------------------
    for k, arm in enumerate(leg_arms):
        ax = fig.add_subplot(gs[1, k])
        d = leg[leg["arm"] == arm]
        m = (d.pivot(index="sender", columns="receiver", values="frac_cooccur")
               .reindex(index=ZONES, columns=ZONES))
        v = m.to_numpy()
        ax.imshow(v, cmap=CONTINUOUS, vmin=0, vmax=1)
        for i in range(len(ZONES)):
            for j in range(len(ZONES)):
                txt = "0" if v[i, j] == 0 else f"{v[i, j]:.2f}".lstrip("0")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if v[i, j] < 0.6 else INK,
                        fontweight="bold" if v[i, j] == 0 and i != j else "normal")
        ax.set_xticks(range(len(ZONES)), ZONES, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(ZONES)), ZONES if k == 0 else [""] * len(ZONES), fontsize=7.5)
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if lab.get_text():
                lab.set_color(ZONE_COLORS[lab.get_text()])
        offd = v[~np.eye(len(ZONES), dtype=bool)]
        ax.set_title(arm, **TITLE_KW, pad=TITLE_PAD_SUB)
        ax.text(0.5, 1.005, f"{d['n_windows'].iloc[0]} windows · "
                            f"{int((offd == 0).sum())}/12 unmeasurable",
                transform=ax.transAxes, ha="center", va="bottom", color=INK_2, **SUB_KW)
        if k == 0:
            ax.set_ylabel("legnini23 — SHH zones\nzone-aligned windows\n\nsender zone", fontsize=8.5)
        ax.set_xlabel("receiver zone", fontsize=8)

    ax = fig.add_subplot(gs[1, 3])
    ax.axis("off")
    ax.text(0, 0.98,
            "Reading the two rows\n\n"
            "Of the windows a configuration produces, in\n"
            f"what fraction do BOTH partners appear with at\n"
            f"least {MIN_CELLS} cells each? An interaction can only be\n"
            "estimated from windows that hold enough of\n"
            "both: one cell of each gives a 'mean' over a\n"
            "single cell pair.\n\n"
            "melton25 (top). At 48 µm, 32 of 72 ordered\n"
            "pairs are unmeasurable — a 40-cell window\n"
            f"cannot hold {MIN_CELLS} of two types at once. 2 of 72\n"
            "at 96 µm, and none from 144 µm upward.\n"
            "(At a >=1-cell criterion nothing ever fails, at\n"
            "any window size, which is why that criterion\n"
            "does not answer the question.)\n\n"
            "legnini23 (bottom). Pairs fail for a different\n"
            "reason: the windows were built from the same\n"
            "zone annotation the interactions are read on,\n"
            "so an excluded zone is absent outright, not\n"
            "merely thin. Every `coarse` window holds one\n"
            "zone, so all 12 zone pairs are unmeasurable.\n\n"
            "Consequence: because the hierarchical mean\n"
            "averages over all windows, a non-co-occurring\n"
            "window contributes an exact zero. Every entry\n"
            "is deflated by its co-occurrence fraction — so\n"
            "a small value is not evidence of a weak\n"
            "interaction until this is checked.",
            va="top", ha="left", fontsize=7.4, linespacing=1.5, transform=ax.transAxes)

    fig.suptitle("Is a cell-type interaction measurable at all?", **TITLE_KW, y=0.975)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-legnini", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mel, cts = melton_measurability()
    mel.to_csv(OUT / "celltype_measurability_melton.csv", index=False)
    off = mel[mel["sender"] != mel["receiver"]]
    print(f"criterion: a window counts only if it holds >= {MIN_CELLS} cells of EACH type\n")
    print(f"{'arm':7s} {'µm':>5s} {'windows':>8s} {'median frac':>12s} {'worst frac':>11s} "
          f"{'unmeasurable':>13s} {'(>=1 cell)':>11s}")
    for arm, g in off.groupby("arm", sort=False):
        print(f"{arm:7s} {g['window_um'].iloc[0]:5.0f} {g['n_windows'].iloc[0]:8d} "
              f"{g['frac_cooccur'].median():12.3f} {g['frac_cooccur'].min():11.3f} "
              f"{int((g['cooccur'] == 0).sum()):8d}/{len(g)} "
              f"{int((g['cooccur_any'] == 0).sum()):6d}/{len(g)}")

    leg = pd.DataFrame()
    if not args.skip_legnini:
        leg = legnini_measurability()
        leg.to_csv(OUT / "celltype_measurability_legnini.csv", index=False)
        print(f"\n{'arm':12s} {'windows':>8s} {'unmeasurable zone pairs':>24s}")
        for arm, g in leg.groupby("arm", sort=False):
            o = g[g["sender"] != g["receiver"]]
            print(f"{arm:12s} {g['n_windows'].iloc[0]:8d} "
                  f"{int((o['cooccur'] == 0).sum()):16d}/{len(o)}")
        fig_measurability(mel, cts, leg, OUT / "fig_celltype_measurability.png")

    print(f"\ndone. tables and figure in {OUT}")


if __name__ == "__main__":
    main()
