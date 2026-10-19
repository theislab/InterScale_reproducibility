"""Reviewer figure: what a sliding window makes structurally unmeasurable (legnini23).

The legnini sliding windows were built on purpose so that different arms contain different SHH
signalling zones -- `SHH_zone` is a binning of `dist_to_SHH` into rings, and the SAME variable is
used as the interaction axis of the flow matrix. That alignment is what makes this the cleanest
demonstration of the reviewer's concern: excluding a zone from the window provably sets a whole
row and column of the zone-to-zone matrix to exactly zero, for any seed and any amount of
training. Nothing statistical is involved.

Arms (from config_files/registry.yaml), with ring edges converted at the Resolve MC pitch of
0.138 um/px -- validated against the data: median cell spacing 11.4 um, cell diameter 15.6 um:

    ring        dist_to_SHH        zones a window spans
    SHH         0 - 27.6 um
    near        27.6 - 276 um
    medium      276 - 414 um
    far         > 414 um

    full_slide  whole sample, all 4 zones          17 windows, largest 4318 cells
    medium_snm  SHH + near + medium                31 windows, largest 3882
    medium_nmf  near + medium + far                34 windows, largest 4239
    coarse      one zone per window                64 windows, largest 2812

--- the dilution correction ---------------------------------------------------------------------

`compute_hierarchical_net_flow` normalises each window by its own max|net| and then averages over
windows. A window spanning a single zone has no cross-zone pair and contributes an all-zero matrix,
so an arm whose windows are partly single-zone has EVERY entry diluted by that fraction regardless
of what the model inferred. Measured on the data: 17/31 of medium_snm's windows and 16/34 of
medium_nmf's span more than one zone.

That matters for how the figure is read. Uncorrected, the arms look like a monotone decay
(0.69-0.91 -> 0.47-0.55 -> 0.375-0.449 -> 0), which invites "smaller windows weaken the inferred
flow". Rescaling by n_total/n_multizone recovers the full-slide value for all 12 visible pairs in
both medium arms, to within 6%. So the correct statement is the stronger one: the window decides
WHICH zone pairs are measurable and leaves the estimates of the measurable ones essentially
unchanged.

    python scripts/legnini_23/3b_sliding_window_figure.py
"""

from __future__ import annotations

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
DATA = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/legnini23_pp.h5ad")
OUT = REPO / "figures" / "legnini23" / "sliding_window_ablation"

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



PX_UM = 0.138  # Resolve Molecular Cartography; validated: cell spacing 11.4 um, diameter 15.6 um
RING_EDGES_PX = [0, 200, 2000, 3000, np.inf]
ZONES = ["SHH", "near", "medium", "far"]  # ordered by distance from the SHH source
# config.yml numbers ring_1 as furthest from the source and ring_4 as closest.
ZONE_COLORS = STYLE.zones("legnini23", ZONES)

ARM_COL = {
    "full_slide": "sample",
    "medium_snm": "sliding_window_high_SHH_near_medium",
    "medium_nmf": "sliding_window_high_near_medium_far",
    "coarse": "sliding_window_low",
}
ARM_TITLE = {
    "full_slide": "all 4 zones",
    "medium_snm": "SHH + near + medium",
    "medium_nmf": "near + medium + far",
    "coarse": "one zone per window",
}

# Zone-to-zone raw incoming flow (compute_net=False), all_samples mean, as produced by
# notebooks/legnini_23/3_sliding_window_ablation.ipynb. Rows are senders, columns receivers, in
# the notebook's own alphabetical order -- reindexed to distance order below.
NB_ORDER = ["SHH", "far", "medium", "near"]
FLOWS = {
    "full_slide": [[0, .741079, .865446, .914815], [.685825, 0, .765561, .753627],
                   [.804386, .768530, 0, .879190], [.875837, .781421, .907689, 0]],
    "medium_snm": [[0, 0, .492171, .548044], [0, 0, 0, 0],
                   [.47376, 0, 0, .495531], [.54033, 0, .513007, 0]],
    "medium_nmf": [[0, 0, 0, 0], [0, 0, .375219, .378032],
                   [0, .385453, 0, .439228], [0, .392609, .449229, 0]],
    "coarse": [[0] * 4 for _ in range(4)],
}



def load_flows():
    return {a: pd.DataFrame(v, index=NB_ORDER, columns=NB_ORDER).reindex(index=ZONES, columns=ZONES)
            for a, v in FLOWS.items()}


def window_composition(adata):
    """Per arm: how many windows there are, and how many span more than one zone.

    The multi-zone fraction is the dilution factor; the set of zones an arm's windows ever contain
    is what makes a row/column structurally zero.
    """
    out = {}
    for arm, col in ARM_COL.items():
        n_zones = adata.obs.groupby(col, observed=True)["SHH_zone"].nunique()
        sizes = adata.obs[col].value_counts()
        n_zones = n_zones[sizes.reindex(n_zones.index).fillna(0) > 0]
        # Zones that co-occur with any other zone in some window -- i.e. zones whose flows are
        # measurable at all for this arm.
        measurable = set()
        for w, k in n_zones.items():
            if k > 1:
                measurable |= set(adata.obs.loc[adata.obs[col] == w, "SHH_zone"].dropna().unique())
        out[arm] = {
            "n_windows": int(len(n_zones)),
            "n_multizone": int((n_zones > 1).sum()),
            "frac_multizone": float((n_zones > 1).mean()),
            "largest": int(sizes.max()),
            "measurable_zones": sorted(measurable, key=ZONES.index),
        }
    return out


def fig_legnini(flows, comp, path=None):
    arms = list(ARM_COL)
    fig, axes = plt.subplots(2, len(arms), figsize=(3.9 * len(arms), 9.0),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.55})

    vmax = max(np.nanmax(np.abs(flows[a].to_numpy())) for a in arms)

    for k, arm in enumerate(arms):
        m = flows[arm]
        c = comp[arm]
        measurable = set(c["measurable_zones"])
        # A cell is structurally unmeasurable when either partner's zone never shares a window
        # with another zone in this arm. Those are hard zeros, not small numbers.
        dead = np.zeros_like(m.to_numpy(), dtype=bool)
        for i, s in enumerate(ZONES):
            for j, r in enumerate(ZONES):
                if i == j:
                    continue
                if s not in measurable or r not in measurable:
                    dead[i, j] = True

        for row, corrected in enumerate([False, True]):
            ax = axes[row][k]
            vals = m.to_numpy().astype(float).copy()
            if corrected and c["frac_multizone"] > 0:
                vals = vals / c["frac_multizone"]
            diag = np.eye(len(ZONES), dtype=bool)
            masked = np.ma.masked_where(dead | diag, vals)

            # mpl.colormaps[...], not mpl.cm.get_cmap: the latter was removed in matplotlib 3.9.
            cmap = mpl.colormaps[CONTINUOUS]
            ax.imshow(masked, cmap=cmap, vmin=0, vmax=1.0 if corrected else vmax)
            # Structural zeros drawn as hatched grey, so they read as "cannot be measured"
            # rather than "measured and found to be zero".
            ax.imshow(np.ma.masked_where(~(dead | diag), np.ones_like(vals)),
                      cmap=mpl.colors.ListedColormap(["#e8e8e6"]), vmin=0, vmax=1)
            for i in range(len(ZONES)):
                for j in range(len(ZONES)):
                    if i == j:
                        ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=INK_2)
                    elif dead[i, j]:
                        ax.text(j, i, "n.m.", ha="center", va="center", fontsize=7, color="#8a8a86")
                    else:
                        v = vals[i, j]
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                                color="white" if v < 0.55 else INK)
            ax.set_xticks(range(len(ZONES)), ZONES, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(ZONES)), ZONES if k == 0 else [""] * len(ZONES), fontsize=8)
            # Tick labels carry the SHH ring colours from config.yml, so a zone is recognisable
            # here and in the spatial scatters of the setup notebook.
            for lab in ax.get_xticklabels():
                lab.set_color(ZONE_COLORS[lab.get_text()])
            for lab in ax.get_yticklabels():
                if lab.get_text():
                    lab.set_color(ZONE_COLORS[lab.get_text()])
            if row == 0:
                ax.set_title(arm, **TITLE_KW, pad=TITLE_PAD_SUB)
                ax.text(0.5, 1.005, ARM_TITLE[arm].replace("\n", " · "),
                        transform=ax.transAxes, ha="center", va="bottom", color=INK_2, **SUB_KW)
            if k == 0:
                ax.set_ylabel(("as computed" if not corrected else
                               "corrected for single-zone\nwindow dilution")
                              + "\n\nsender zone", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("receiver zone", fontsize=8.5)
            sub = (f"{c['n_multizone']}/{c['n_windows']} windows span >1 zone"
                   if row == 0 else
                   (f"× {1/c['frac_multizone']:.2f}" if c["frac_multizone"] > 0 else "undefined"))
            ax.text(0.5, -0.30, sub, transform=ax.transAxes,
                    ha="center", fontsize=7, color=INK_2)

    fig.suptitle("Excluding a signalling zone makes its interactions unmeasurable",
                 **TITLE_KW, y=0.985)
    fig.subplots_adjust(top=0.86, bottom=0.10)
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(DATA)
    print(f"{adata.n_obs} cells, {adata.n_vars} genes, {adata.obs['sample'].nunique()} samples")
    print("\nSHH ring edges:")
    for lo, hi, name in zip(RING_EDGES_PX[:-1], RING_EDGES_PX[1:], ZONES, strict=True):
        hi_s = "inf" if not np.isfinite(hi) else f"{hi*PX_UM:.0f} um"
        print(f"  {name:8s} {lo:6.0f}-{hi if np.isfinite(hi) else float('inf'):>7} px "
              f"= {lo*PX_UM:6.1f}-{hi_s}")
    print("\nzone sizes:", adata.obs["SHH_zone"].value_counts().reindex(ZONES).to_dict())

    comp = window_composition(adata)
    print(f"\n{'arm':12s} {'n_win':>6s} {'multi-zone':>11s} {'frac':>6s} {'largest':>8s}  measurable zones")
    for arm, c in comp.items():
        print(f"{arm:12s} {c['n_windows']:6d} {c['n_multizone']:11d} {c['frac_multizone']:6.3f} "
              f"{c['largest']:8d}  {c['measurable_zones']}")

    flows = load_flows()
    print("\nrescaled vs whole-sample, for the pairs each arm can measure:")
    full = flows["full_slide"]
    for arm in ["medium_snm", "medium_nmf"]:
        m, k = flows[arm], 1 / comp[arm]["frac_multizone"]
        r = [(m.loc[s, t] * k) / full.loc[s, t]
             for s in ZONES for t in ZONES
             if s != t and m.loc[s, t] != 0]
        print(f"  {arm:12s} n={len(r):2d}  mean ratio {np.mean(r):.3f}  "
              f"range {np.min(r):.3f}-{np.max(r):.3f}")

    fig_legnini(flows, comp, OUT / "fig_legnini_structural_zeros.png")
    pd.DataFrame(comp).T.to_csv(OUT / "window_composition.csv")
    print(f"\ndone. figure and table in {OUT}")


if __name__ == "__main__":
    main()
