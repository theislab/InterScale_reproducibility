"""Sensitivity of interaction coverage to window size AND overlap -- reviewer comment 5.

Answers the half of the reviewer's request the size-only sweep does not: how much of the boundary
loss is recovered by overlapping the windows instead of enlarging them.

Pure geometry. No model, no training, no seeds: whether two cells can ever share a window is a
property of the tiling alone, so this is exact and reproducible from coordinates.

    python scripts/melton_jimenez_25/5c_window_overlap_coverage.py
    python scripts/melton_jimenez_25/5c_window_overlap_coverage.py --n-fovs 12

--- why overlap and not a bigger window --------------------------------------------------------

The two knobs cost very different amounts:

  window size W   cells per window ~ W^2, and attention is quadratic in that, so peak attention
                  memory ~ W^4. Measured on this dataset, going from a 1600 px to a 3000 px window
                  takes max_seq_len 1218 -> 3436, i.e. ~8x the peak attention memory per graph.
                  That is a hard ceiling, not a wall-clock cost.
  overlap         max_seq_len is UNCHANGED -- the windows are the same size, there are just more of
                  them. Halving the stride gives ~4x the graphs: 4x wall-clock, linear, and no
                  change in peak memory.

So overlap buys coverage cheaply and window size buys it at quartic cost in memory. The reviewer's
suggestion to vary overlap is the efficient lever, which is what this script quantifies.

--- what squidpy returns -----------------------------------------------------------------------

`sq.tl.sliding_window(..., overlap=0)` writes ONE categorical column of window ids. With
`overlap > 0` it instead writes one BOOLEAN membership column per (library, window) -- 40 columns
for 2 FOVs at window 1600 / overlap 400. Those masks are what this script consumes directly, so
the numbers describe squidpy's real tiling rather than a reimplementation of it.

--- how coverage is computed -------------------------------------------------------------------

A pair of cells is *covered* when at least one window contains both. With overlap a pair may sit
in several windows, so the count has to be de-duplicated; summing per-window pair counts would
over-count exactly where overlap helps, which would fake the result this script is testing.

De-duplication is done with a packed bitset: each cell carries a bit per window, and a pair is
covered iff `bits_i & bits_j` is non-zero in any word. One bitwise AND per pair, exact, and cheap
enough to sweep 24 configurations. Distances are binned once per FOV and reused across
configurations, since they do not depend on the tiling.

Pairs are counted within `slide_fov` only. Windows are built per FOV, so two cells in different
FOVs cannot share a window at any size or overlap -- including them would add a constant loss to
every configuration and hide the effect being measured. Self-pairs are excluded.
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
import squidpy as sq

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
RESULTS = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results")
SLIDE_H5AD = RESULTS / "melton25_revision" / "sliding_window" / "attention" / "_slide_Run5211_S3.h5ad"
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



# CosMx SMI pixel pitch. obsm['spatial'] regresses on CenterX_global_px with slope 1.0000 and
# r2 1.0000 within every FOV (offset only), so it IS CosMx global pixels and inherits the
# instrument constant rather than needing to be inferred from cell sizes.
PX_UM = 0.12028

WINDOW_SIZES = [400, 800, 1200, 1600, 2000, 3000]
OVERLAP_FRACS = [0.0, 0.25, 0.5, 0.75]

# Same edges as 5a, so this table can be read beside coverage_by_distance.csv.
DIST_EDGES = np.array(
    [0, 20, 40, 60, 80, 110, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000, 3000, 4000, 6600],
    dtype=np.float32,
)
N_BINS = len(DIST_EDGES) - 1

# Cutoffs the response quotes, chosen to sit on bin edges. The regime labels are the scales over
# which each mode of signalling is generally described; the manuscript's own citations belong here.
FOCUS_PX = {200: "contact / juxtacrine", 400: "short-range paracrine",
            800: "paracrine", 1600: "morphogen-scale", 3000: "long-range"}

# max cells per window, measured over ALL splits -- this is what max_seq_len must cover, and it is
# what makes window size the expensive knob. Independent of overlap.
MAX_CELLS = {400: 89, 800: 330, 1200: 685, 1600: 1218, 2000: 1720, 3000: 3436}

# Ordered series, so the colours are sampled from config.yml's continuous colormap
# (light = small window, dark = large) rather than chosen by hand.
SIZE_COLORS = dict(zip(WINDOW_SIZES, STYLE.sequential(len(WINDOW_SIZES)), strict=True))



def pack_bits(masks):
    """Pack a list of boolean membership masks into one uint64 bitset per cell.

    Returns (n_cells, n_words). A pair shares a window iff the bitwise AND of their rows is
    non-zero in some word -- which is how overlapping windows get de-duplicated in O(1) per pair
    instead of O(n_windows).
    """
    n = len(masks[0])
    n_words = (len(masks) + 63) // 64
    bits = np.zeros((n, n_words), dtype=np.uint64)
    for k, m in enumerate(masks):
        bits[np.asarray(m, dtype=bool), k // 64] |= np.uint64(1) << np.uint64(k % 64)
    return bits


def window_masks(adata_fov, window_size, overlap):
    """Boolean membership masks for one FOV at one (window_size, overlap), from squidpy."""
    t = adata_fov.copy()
    before = set(t.obs.columns)
    key = "sw_tmp"
    sq.tl.sliding_window(t, library_key="slide_fov", window_size=int(window_size),
                         overlap=int(overlap), sliding_window_key=key)
    new = [c for c in t.obs.columns if c not in before]

    if overlap == 0:
        # One categorical column of window ids -> one mask per id.
        (col,) = new
        vals = t.obs[col].astype(str).to_numpy()
        return [vals == v for v in np.unique(vals)]
    # One boolean membership column per window.
    masks = []
    for c in new:
        v = t.obs[c]
        m = (v.astype(str).to_numpy() == "True") if v.dtype == object or str(v.dtype) == "category" \
            else np.asarray(v, dtype=bool)
        if m.any():
            masks.append(m)
    return masks


def bin_matrix(xy, chunk=1024):
    """Distance-bin index for every ordered pair of one FOV, as int8. Computed once per FOV."""
    n = len(xy)
    b = np.empty((n, n), dtype=np.int8)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = np.sqrt((xy[s:e, 0, None] - xy[None, :, 0]) ** 2
                    + (xy[s:e, 1, None] - xy[None, :, 1]) ** 2)
        b[s:e] = np.clip(np.digitize(d, DIST_EDGES, right=False) - 1, 0, N_BINS - 1).astype(np.int8)
    return b


def coverage_for_fov(xy, bins, masks, chunk=1024):
    """(covered_per_bin, total_per_bin) for one FOV and one tiling. Self-pairs excluded."""
    n = len(xy)
    bits = pack_bits(masks)
    covered = np.zeros(N_BINS, dtype=np.int64)
    total = np.zeros(N_BINS, dtype=np.int64)
    eye_col = np.arange(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        bb = bins[s:e].astype(np.int64)
        share = np.zeros((e - s, n), dtype=bool)
        for w in range(bits.shape[1]):
            share |= (bits[s:e, w, None] & bits[None, :, w]) != 0
        # Drop the diagonal: a cell trivially shares every one of its own windows with itself.
        keep = np.ones((e - s, n), dtype=bool)
        keep[np.arange(e - s), eye_col[s:e]] = False
        total += np.bincount(bb[keep], minlength=N_BINS)
        covered += np.bincount(bb[keep & share], minlength=N_BINS)
    return covered, total


def sweep(n_fovs, window_sizes, overlap_fracs):
    adata = sc.read_h5ad(SLIDE_H5AD)
    fovs = sorted(adata.obs["slide_fov"].astype(str).unique())[:n_fovs]
    print(f"{len(fovs)} FOVs: {fovs}")

    rows = []
    for fi, f in enumerate(fovs):
        sub = adata[adata.obs["slide_fov"].astype(str) == f].copy()
        xy = np.asarray(sub.obsm["spatial"], dtype=np.float32)[:, :2]
        print(f"[{fi+1}/{len(fovs)}] {f}: {sub.n_obs} cells", flush=True)
        bins = bin_matrix(xy)  # reused across every configuration
        for W in window_sizes:
            for frac in overlap_fracs:
                ov = int(round(frac * W))
                masks = window_masks(sub, W, ov)
                cov, tot = coverage_for_fov(xy, bins, masks)
                for b in range(N_BINS):
                    rows.append({
                        "slide_fov": f, "window_size": W, "overlap_frac": frac, "overlap_px": ov,
                        "n_windows": len(masks), "bin": b,
                        "d_lo_px": float(DIST_EDGES[b]), "d_hi_px": float(DIST_EDGES[b + 1]),
                        "pairs_covered": int(cov[b]), "pairs_possible": int(tot[b]),
                    })
        del bins
    return pd.DataFrame(rows)


def cumulative(df):
    """Coverage of pairs separated by <= each bin edge, pooled over FOVs."""
    out = []
    for (W, frac), g in df.groupby(["window_size", "overlap_frac"]):
        nwin = g.groupby("slide_fov")["n_windows"].first().mean()
        for e in sorted(g["d_hi_px"].unique()):
            s = g[g["d_hi_px"] <= e]
            poss = s["pairs_possible"].sum()
            out.append({
                "window_size": W, "overlap_frac": frac,
                "overlap_px": int(round(frac * W)),
                "windows_per_fov": nwin,
                "max_cells": MAX_CELLS.get(W, np.nan),
                "cutoff_px": e, "cutoff_um": e * PX_UM,
                "coverage": s["pairs_covered"].sum() / poss if poss else np.nan,
                "fraction_lost": 1 - s["pairs_covered"].sum() / poss if poss else np.nan,
            })
    return pd.DataFrame(out)


def fig_overlap(cum, path=None):
    """Fraction lost vs overlap, one panel per signalling regime."""
    focus = [200, 400, 800, 1600]
    fig, axes = plt.subplots(1, len(focus), figsize=(3.9 * len(focus), 4.6), sharey=True)
    for ax, e in zip(axes, focus, strict=True):
        d = cum[np.isclose(cum["cutoff_px"], e)]
        for W, g in d.groupby("window_size"):
            g = g.sort_values("overlap_frac")
            ax.plot(g["overlap_frac"] * 100, g["fraction_lost"], marker="o", ms=5, lw=1.8,
                    color=SIZE_COLORS[W], label=f"{W} px ({W*PX_UM:.0f} µm)")
        ax.set_title(f"≤ {e*PX_UM:.0f} µm", **TITLE_KW, pad=TITLE_PAD_SUB)
        ax.text(0.5, 1.005, FOCUS_PX[e], transform=ax.transAxes, ha="center", va="bottom",
                color=INK_2, **SUB_KW)
        ax.set_xlabel("window overlap (% of window size)")
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("fraction of interactions lost\nto window boundaries")
    axes[-1].legend(title="window size", bbox_to_anchor=(1.02, 1), loc="upper left", **LEGEND_KW)
    fig.suptitle("Overlap recovers boundary losses at constant max_seq_len", **TITLE_KW, y=1.05)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


def fig_cost(cum, path=None):
    """The trade-off: what each knob costs to reach a coverage target."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # A: fraction lost collapses onto the ratio of interaction distance to window size.
    ax = axes[0]
    d0 = cum[cum["overlap_frac"] == 0]
    for W, g in d0.groupby("window_size"):
        g = g[g["cutoff_px"] > 0]
        ax.plot(g["cutoff_px"] / W, g["fraction_lost"], marker="o", ms=4, lw=1.6,
                color=SIZE_COLORS[W], label=f"{W} px")
    ax.axhline(0.1, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(0.011, 0.11, "10% lost", fontsize=7.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("interaction distance / window size")
    ax.set_ylabel("fraction of interactions lost")
    ax.set_title("A  Loss is scale-free", **TITLE_KW, pad=6)
    ax.legend(ncols=2, title="window size", **LEGEND_KW)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    # B: cost of each knob at a fixed biological target.
    ax = axes[1]
    d = cum[np.isclose(cum["cutoff_px"], 800)]  # ~96 um, paracrine
    for W, g in d.groupby("window_size"):
        g = g.sort_values("overlap_frac")
        ax.plot(g["windows_per_fov"], g["fraction_lost"], marker="o", ms=6, lw=1.8,
                color=SIZE_COLORS[W], label=f"{W} px, max_seq_len {MAX_CELLS[W]}")
        for _, r in g.iterrows():
            if r["overlap_frac"] in (0.0, 0.75):
                ax.annotate(f"{int(r['overlap_frac']*100)}%",
                            (r["windows_per_fov"], r["fraction_lost"]),
                            textcoords="offset points", xytext=(0, 7), ha="center",
                            fontsize=6.5, color=INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("windows per FOV  (wall-clock; grows linearly with overlap)")
    ax.set_ylabel("fraction of ≤96 µm interactions lost")
    ax.set_title("B  Two ways to buy coverage", **TITLE_KW, pad=6)
    ax.legend(loc="upper right", **{**LEGEND_KW, "prop": {"size": 8}})
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


# --------------------------------------------------------------------------------------------
# per cell-type-pair loss -- the sender x receiver view, without a model in it
# --------------------------------------------------------------------------------------------

ATTN_IN = RESULTS / "melton25_revision" / "sliding_window" / "attention"


def celltype_pair_loss(cutoffs_px=(200, 800, 1600)):
    """Fraction of pairs of each (sender type, receiver type) combination that are severed.

    Reuses the (K, K, n_bins) pair-count arrays 5a already wrote, so this is the same quantity as
    the pooled coverage but resolved by cell-type pair. Model-free and seed-free by construction,
    which is the point: it gives the sender x receiver figure the interaction-detection version
    cannot, because the inferred flow matrix on this dataset is not reproducible across seeds.
    """
    ref = np.load(ATTN_IN / "reference_pair_distances.npz", allow_pickle=True)
    edges = ref["edges"]
    cts = list(ref["cell_types"].astype(str))
    K = len(cts)
    k = np.arange(K)
    ref_counts = ref["counts"].astype(np.int64)

    def upto(arr, cutoff):
        return arr[:, :, edges[1:] <= cutoff].sum(axis=2)

    rows = []
    cells = None
    for W in WINDOW_SIZES:
        f = ATTN_IN / "windows" / f"w{W}.npz"
        if not f.exists():
            continue
        w = np.load(f, allow_pickle=True)
        got = w["counts"].astype(np.int64).sum(axis=0)
        if cells is None:
            cells = w["per_type"].sum(axis=0)
            rc = ref_counts.copy()
            rc[k, k, 0] -= cells          # self-pairs out of the denominator
        g = got.copy()
        g[k, k, 0] -= cells               # ... and out of the numerator
        for cutoff in cutoffs_px:
            cg, cr = upto(g, cutoff), upto(rc, cutoff)
            for i in range(K):
                for j in range(K):
                    if cr[i, j] <= 0:
                        continue
                    rows.append({"window_size": W, "cutoff_px": cutoff,
                                 "cutoff_um": cutoff * PX_UM, "sender": cts[i],
                                 "receiver": cts[j], "pairs": int(cr[i, j]),
                                 "fraction_lost": 1 - cg[i, j] / cr[i, j]})
    return pd.DataFrame(rows), cts


def fig_celltype_loss(df, cts, windows=(400, 1600), cutoffs_px=(200, 800), path=None):
    """Sender x receiver fraction lost, for a couple of windows and interaction ranges."""
    combos = [(W, c) for W in windows for c in cutoffs_px]
    fig, axes = plt.subplots(1, len(combos), figsize=(4.1 * len(combos), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (W, c) in zip(axes, combos, strict=True):
        d = df[(df["window_size"] == W) & (df["cutoff_px"] == c)]
        m = (d.pivot(index="sender", columns="receiver", values="fraction_lost")
               .reindex(index=cts, columns=cts))
        v = m.to_numpy()
        im = ax.imshow(v, cmap=CONTINUOUS, vmin=0, vmax=1)
        for i in range(len(cts)):
            for j in range(len(cts)):
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v[i, j] < 0.55 else INK)   # viridis is dark at low values
        ax.set_xticks(range(len(cts)), cts, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cts)), cts if ax is axes[0] else [""] * len(cts), fontsize=7)
        ax.set_title(f"{W*PX_UM:.0f} µm window", **TITLE_KW, pad=TITLE_PAD_SUB)
        ax.text(0.5, 1.005, f"interactions ≤ {c*PX_UM:.0f} µm   ·   "
                            f"mean {np.nanmean(v):.3f} ± {np.nanstd(v):.3f}",
                transform=ax.transAxes, ha="center", va="bottom", color=INK_2, **SUB_KW)
        ax.set_xlabel("receiver cell type", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("sender cell type", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("fraction lost", fontsize=7.5)
    fig.suptitle("Boundary loss is close to uniform across cell-type pairs", **TITLE_KW, y=1.04)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


# --------------------------------------------------------------------------------------------
# the reviewer figure: fraction LOST, in micrometres, both readings side by side
# --------------------------------------------------------------------------------------------

# Length scales over which each mode of signalling is generally described. Shaded on the figure so
# a reader can see at a glance which regimes a given window covers. The manuscript's own citations
# belong against these bounds.
REGIMES = [
    (0, 20, "contact /\njuxtacrine", "#f2f4f6"),
    (20, 100, "short-range\nparacrine", "#e8eef4"),
    (100, 300, "morphogen\ngradients", "#dde7f0"),
]


def fig_boundary_loss(cov_csv=None, path=None):
    """Fraction of interactions severed by window boundaries, no overlap.

    Two panels, because "fraction lost" is ambiguous until the population is named and the two
    readings differ substantially:

      A  AT a separation      -- among pairs exactly this far apart. Shows each window's hard reach
                                 as a cliff: beyond it no tile can contain both cells.
      B  WITHIN a separation  -- among all pairs up to this far apart. This is the number that
                                 answers "what fraction of potential interactions is lost", and it
                                 is the one quoted in the response tables.

    Panel A was previously shown while panel B's numbers were quoted, which is why the figure and
    the tables did not appear to agree.
    """
    cov = pd.read_csv(cov_csv or (OUT / "coverage_by_distance.csv"))
    d = cov[cov["bin"] >= 0].copy()
    d["d_hi_um"] = d["d_hi_px"] * PX_UM
    d["d_mid_um"] = d["d_mid_px"] * PX_UM

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), sharey=True)

    for ax, cumulative_view in zip(axes, [False, True], strict=True):
        for lo, hi, label, colour in REGIMES:
            ax.axvspan(max(lo, 1.5), hi, color=colour, lw=0, zorder=0)
            ax.text(np.sqrt(max(lo, 1.5) * hi), 1.045, label, ha="center", va="bottom",
                    fontsize=7, color=INK_2, linespacing=1.15)

        for W, g in d.groupby("window_size"):
            g = g.sort_values("d_hi_px")
            if cumulative_view:
                x = g["d_hi_um"].to_numpy()
                y = 1 - (g["pairs_covered"].cumsum() / g["pairs_possible"].cumsum()).to_numpy()
            else:
                x = g["d_mid_um"].to_numpy()
                y = (1 - g["coverage"]).to_numpy()
            ax.plot(x, y, marker="o", ms=4, lw=1.9, color=SIZE_COLORS[int(W)],
                    label=f"{int(W)} px = {int(W)*PX_UM:.0f} µm")

        ax.axvline(110 * PX_UM, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
        # Rotated along the line and left of it: at y=0.03 it sat directly on the curves, which
        # all crowd the bottom-left of this plot.
        ax.text(110 * PX_UM * 0.93, 0.5, "local graph radius, 13 µm", fontsize=7,
                color=INK_2, rotation=90, ha="right", va="center")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("sender–receiver separation (µm)")
        ax.grid(color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("fraction of interactions lost\nto window boundaries")
    axes[0].set_title("A  At this separation", **TITLE_KW, pad=30)
    axes[1].set_title("B  Within this separation", **TITLE_KW, pad=30)
    axes[1].legend(title="window size", loc="lower right",
                   **{**LEGEND_KW, "prop": {"size": 9}, "title_fontproperties": {"size": 9, "weight": "bold"}})

    # Worked example, so a reader can check they are reading the axes the intended way.
    g = d[d["window_size"] == 1600].sort_values("d_hi_px")
    yb = 1 - (g["pairs_covered"].cumsum() / g["pairs_possible"].cumsum()).to_numpy()
    xi = int(np.argmin(np.abs(g["d_hi_um"].to_numpy() - 96.2)))
    axes[1].annotate(
        f"a 1600 px window loses {yb[xi]:.0%}\nof interactions within 96 µm",
        xy=(g["d_hi_um"].to_numpy()[xi], yb[xi]), xytext=(14, 0.80),
        fontsize=7.5, color=INK, ha="left",
        arrowprops=dict(arrowstyle="->", lw=0.9, color=INK_2, shrinkB=4),
    )

    fig.suptitle("Interactions severed by non-overlapping window boundaries", **TITLE_KW, y=1.07)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")
        return None
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-fovs", type=int, default=8,
                    help="FOVs to pool. Coverage is a within-FOV quantity, so a subset is "
                         "representative; validated against the all-FOV overlap=0 numbers.")
    ap.add_argument("--window-sizes", nargs="+", type=int, default=WINDOW_SIZES)
    ap.add_argument("--overlap-fracs", nargs="+", type=float, default=OVERLAP_FRACS)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        cum = pd.read_csv(OUT / "overlap_coverage_cumulative.csv")
    else:
        raw = sweep(args.n_fovs, args.window_sizes, args.overlap_fracs)
        raw.to_csv(OUT / "overlap_coverage_by_distance.csv", index=False)
        cum = cumulative(raw)
        cum.to_csv(OUT / "overlap_coverage_cumulative.csv", index=False)

    print("\nFRACTION OF INTERACTIONS LOST TO WINDOW BOUNDARIES")
    for e, label in FOCUS_PX.items():
        d = cum[np.isclose(cum["cutoff_px"], e)]
        if d.empty:
            continue
        piv = d.pivot_table(index="window_size", columns="overlap_frac", values="fraction_lost")
        piv.columns = [f"{int(c*100)}%" for c in piv.columns]
        print(f"\n  interactions <= {e*PX_UM:.0f} um  ({label})   [rows: window px, cols: overlap]")
        print(piv.round(3).to_string())

    print("\n\nwindows per FOV (wall-clock cost; max_seq_len is unchanged by overlap)")
    piv = cum.drop_duplicates(["window_size", "overlap_frac"]).pivot_table(
        index="window_size", columns="overlap_frac", values="windows_per_fov")
    piv.columns = [f"{int(c*100)}%" for c in piv.columns]
    piv["max_seq_len"] = [MAX_CELLS[w] for w in piv.index]
    print(piv.round(1).to_string())

    if (OUT / "coverage_by_distance.csv").exists():
        fig_boundary_loss(path=OUT / "fig_boundary_loss.png")
    fig_overlap(cum, OUT / "fig_overlap_sensitivity.png")
    fig_cost(cum, OUT / "fig_overlap_cost.png")

    if (ATTN_IN / "reference_pair_distances.npz").exists():
        ct, cts = celltype_pair_loss()
        ct.to_csv(OUT / "celltype_pair_loss.csv", index=False)
        for W in [400, 1600]:
            for c in [200, 800]:
                d = ct[(ct["window_size"] == W) & (ct["cutoff_px"] == c)]["fraction_lost"]
                if len(d):
                    print(f"  {W:5d} px window, <= {c*PX_UM:5.0f} um: fraction lost "
                          f"{d.mean():.3f} +/- {d.std():.3f}  (range {d.min():.3f}-{d.max():.3f})")
        fig_celltype_loss(ct, cts, path=OUT / "fig_celltype_pair_loss.png")
    else:
        print("skipping the cell-type panel: run 5a_sliding_window_attention.py first")
    print(f"\ndone. tables and figures in {OUT}")


if __name__ == "__main__":
    main()
