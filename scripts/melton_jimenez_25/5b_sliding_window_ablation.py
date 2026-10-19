"""What cell-cell interactions does a smaller sliding window stop seeing?

Consumes the products of `5a_sliding_window_attention.py` (six window sizes x three seeds,
extracted on the held-out slide Run5211_S3) and answers the question at three levels:

  1. STRUCTURAL   Which pairs can a windowing even expose? A pair of cells in different windows
                  never appears in the same attention block, at any strength, for any seed. This
                  is a property of the windowing alone -- no model involved -- and it is the hard
                  ceiling every other result sits under.
                    -> coverage_by_distance.csv, fig_coverage.png

  2. ATTRIBUTED   Where does the trained model actually put its attention, as a function of
                  sender-receiver distance? A window can expose a long-range pair and the model
                  can still ignore it.
                    -> attention_by_distance.csv, fig_attention_distance.png

  3. INFERRED     Which cell-type interactions come out as significant, and at which window size
                  do they first appear? This is the "what is missed" table: the permutation test
                  of notebooks/melton_jimenez_25/Interaction_analysis.ipynb, run once per arm.
                    -> permutation_stats.csv, detection_thresholds.csv,
                       fig_detection_threshold.png, fig_observed_vs_null.png

  plus the recovery curve against the widest arm (recovery.csv, fig_recovery.png) and the partial
  permutation degradation curves of the reference notebook, per arm (partial_permutation.csv,
  fig_partial_permutation.png).

  4. REPRODUCIBLE? The diagnostic that gates levels 2 and 3 above -- whether the cell-type flow
                   matrix is a property of the data or of the seed.
                     -> seed_stability.csv, fig_seed_stability.png

--- what came out, so the tables are read the right way round ----------------------------------

Levels 1 and 2 are solid and stand alone. Level 3 does not, on these models:

  * Coverage (level 1) is pure geometry: the w400 arm exposes 0.8% of same-FOV pairs and exactly
    ZERO beyond 600 px; w3000 exposes 36.7% and reaches 4000 px. No model, no seeds, no caveats.
  * Attention (level 2) is seed-stable to sd 0.0007-0.028, and its shape is the interesting part:
    mean relevance PER PAIR is nearly flat with distance (w3000 decays only 1.7x from 0-20 px to
    4000-6600 px). Since per-pair weight barely depends on distance, the total mass at each
    distance is set by how many pairs the windowing exposes there -- so the window size does not
    merely limit what is visible, it effectively determines the interaction range the model reports.
  * Level 3 fails its own diagnostic. 4 of 6 arms have a between-seed flow correlation below 0.5,
    the worst seed pair reaching r = -0.76, and the w3000 reference sits at -0.09 against ITSELF.
    The permutation test does not separate the arms either: 0 of 36 interactions are never
    significant and 24 of 36 are already significant at 400 px. Consistent with the sweep metrics
    -- every arm is at val R2 ~ 0 -- a model that does not fit the target has no reason to carry a
    reproducible interaction structure in its attention.

  So `recovery.csv` and `detection_thresholds.csv` are written and plotted, but they are reported
  against their own null: read fig_seed_stability.png first.

    python scripts/melton_jimenez_25/5b_sliding_window_ablation.py
    python scripts/melton_jimenez_25/5b_sliding_window_ablation.py --n-permutations 200
    python scripts/melton_jimenez_25/5b_sliding_window_ablation.py --plot-only
    python scripts/melton_jimenez_25/5b_sliding_window_ablation.py --validate

--- the flow definition ------------------------------------------------------------------------

Reproduced exactly from `interscale.evaluation.net_streams.compute_hierarchical_net_flow`, so the
numbers here are comparable with the published single-model analysis:

  per window   agg[s, r]  = mean attention over all (sender of type s, receiver of type r) pairs,
                            the diagonal included
               net        = agg.T - agg          (compute_net=True; information flows against
                                                  attention, hence the transpose)
                            or agg.T with a zero diagonal (compute_net=False)
               then divided by max|net| within the window
  per sample   mean over the windows of that slide_fov
  per condition mean and sd over the samples of that condition

`--validate` asserts this reimplementation against the real function on a subsample of windows.
It is a reimplementation for one reason: the reference makes one `adata[adata.obs[win] == w]`
subset and 81 pandas `.loc` calls per window, which for the w400 arm is 3,200 AnnData subsets and
260,000 `.loc` calls -- per seed, per permutation.

--- the diagonal -------------------------------------------------------------------------------

Every distance-resolved quantity here EXCLUDES each cell's own diagonal entry, and that is not a
detail. `obsm['_attn_matrix']` is not raw attention: it is the Chefer et al. relevance matrix that
`interscale.tl.self_attn_relevance.generate_relevance` builds as

    I = eye(n);   for each layer:   I += (grad * attn, clamped, head-averaged) @ I

so the diagonal STARTS at exactly 1 and the attention contributions accumulate on top of it. The
diagonal measuring ~99% of the total mass (row sums land at ~1.01) is therefore the identity term,
a construction constant -- not the model choosing to put its attention on each cell itself. The
interaction signal is the ~1% off the diagonal, and a distance profile that keeps the diagonal in
is a plot of `eye(n)`: everything real sits two orders of magnitude beneath it.

`drop_self_pairs` removes it exactly, using the per-type diagonal sums 5a stores alongside the
binned sums.

The flow matrices deliberately KEEP it, because `compute_hierarchical_net_flow` includes the
diagonal in agg[s, s] and the published numbers have to stay reproducible. It makes no difference
there: net = agg.T - agg is zero on the diagonal, and the raw-incoming branch zeroes it explicitly.

--- what the sweep metrics already say ---------------------------------------------------------

Reconstruction quality is flat across the arms (val CCC 0.014-0.018, val R2 within noise of 0 for
every arm), so the ablation cannot be read off the loss: a model that sees fewer interactions
reconstructs expression about as well. That is itself the finding that motivates looking at the
attention rather than the metric.
"""

from __future__ import annotations

import argparse
import time
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
RESULTS = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results")
IN = RESULTS / "melton25_revision" / "sliding_window" / "attention"
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
# Clearance for a 14 pt bold title when a sub-line is drawn just above the axes.
TITLE_PAD_SUB = 24


ARMS = ["w400", "w800", "w1200", "w1600", "w2000", "w3000"]
SEEDS = [42, 43, 44]
REFERENCE_ARM = "w3000"  # the widest windowing: the closest thing available to "sees everything"

# The ablation axis is an ordered quantity, so its colours are sampled from config.yml's
# continuous colormap (light = narrow window, dark = wide) instead of being chosen by hand.
ARM_COLORS = dict(zip(ARMS, STYLE.sequential(len(ARMS)), strict=True))
# Cell-type colours for axis labels, from figures/config.yml.
CT_COLORS = STYLE.cell_types("melton25")

# Two-series contrast, both derived from config.yml rather than picked by hand: a mid tone of the
# continuous colormap for the typical case, and the warm end of the divergent colormap for the
# worst case, where red carries "this is the problem".
C_TYPICAL = mpl.colormaps[CONTINUOUS](0.45)
C_WORST = mpl.colormaps[DIVERGENT](0.92)



def arm_window_size(arm: str) -> int:
    return int(arm[1:])


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------


def load_windows(arm: str):
    """Per-window pair counts and metadata for one arm. Model independent."""
    z = np.load(IN / "windows" / f"{arm}.npz", allow_pickle=True)
    return {
        "counts": z["counts"],  # (n_windows, K, K, n_bins)
        "per_type": z["per_type"],  # (n_windows, K)
        "window": z["window"].astype(str),
        "n_cells": z["n_cells"],
        "slide_fov": z["slide_fov"].astype(str),
        "condition": z["condition"].astype(str),
        "cell_types": list(z["cell_types"].astype(str)),
        "edges": z["edges"],
        "window_key": str(z["window_key"]),
    }


def load_flows(arm: str, seed: int):
    """Per-window summed attention for one (arm, seed)."""
    z = np.load(IN / "flows" / f"{arm}_seed{seed}.npz", allow_pickle=True)
    return z["sums"]  # (n_windows, K, K, n_bins)


def load_diag(arm: str, seed: int):
    """Per-window attention diagonal, summed per cell type. Shape (n_windows, K)."""
    z = np.load(IN / "flows" / f"{arm}_seed{seed}.npz", allow_pickle=True)
    return z["diag_sums"]


def drop_self_pairs(sums, counts, per_type, diag_sums):
    """Remove each cell's own diagonal entry from the binned sums and counts.

    This has to be done for anything read as a function of distance. The relevance matrix is
    initialised to `eye(n)` before the per-layer attention terms are accumulated onto it (see the
    module docstring), so its diagonal is ~1 by construction and comes to ~99% of the total mass.
    All of it sits at distance 0, in the first bin. Leaving it in makes every distance profile a
    plot of the identity matrix.

    Exact, not approximate: `per_type[w, s]` is how many diagonal entries type s contributed to
    bin 0 of `counts[w, s, s]`, and `diag_sums[w, s]` is what they contributed to `sums[w, s, s]`.

    Deliberately NOT applied to the flow matrices. `compute_hierarchical_net_flow` includes the
    diagonal in agg[s, s], so excluding it there would stop reproducing the published numbers --
    and it cancels anyway, since net = agg.T - agg is zero on the diagonal.
    """
    s = sums.astype(np.float64).copy()
    c = counts.copy()
    k = np.arange(s.shape[1])
    # The diagonal always lands in bin 0: edges[0] is 0 and digitize(0, edges) - 1 == 0.
    s[:, k, k, 0] -= diag_sums
    c[:, k, k, 0] -= per_type
    assert (c >= 0).all(), "subtracted more self-pairs than the bin held"

    # The result can go slightly negative from rounding alone: 5a stores `sums` as float32 while
    # `diag_sums` keeps the float64 accumulator, so the two disagree in the last float32 bit of a
    # quantity as large as the diagonal itself. The tolerance is therefore scaled to that
    # magnitude rather than absolute -- at 1e-6 it fired on w400 at -3.7e-6, which is exactly
    # float32 eps times a 44-wide diagonal sum, and would have fired ~1000x harder on w3000.
    #
    # A negative BEYOND that would mean the diagonal was double counted, which is worth stopping
    # for: it would silently deflate the first distance bin.
    tol = 8 * float(np.finfo(np.float32).eps) * max(1.0, float(diag_sums.max(initial=1.0)))
    assert s.min() > -tol, (
        f"negative attention after removing self-pairs: {s.min():.3e} < -{tol:.3e}. "
        f"The diagonal looks double counted rather than merely rounded."
    )
    return np.clip(s, 0.0, None), c


def available(arms, seeds):
    """The (arm, seed) pairs 5a actually wrote, so a partial extraction still analyses."""
    out = []
    for arm in arms:
        if not (IN / "windows" / f"{arm}.npz").exists():
            print(f"skipping {arm}: no windows/{arm}.npz")
            continue
        for seed in seeds:
            if (IN / "flows" / f"{arm}_seed{seed}.npz").exists():
                out.append((arm, seed))
            else:
                print(f"skipping {arm} seed {seed}: no flows file")
    if not out:
        raise SystemExit(f"nothing to analyse in {IN}; run 5a_sliding_window_attention.py first")
    return out


# --------------------------------------------------------------------------------------------
# 1. structural coverage
# --------------------------------------------------------------------------------------------


def coverage_table(arms) -> pd.DataFrame:
    """Fraction of same-FOV cell pairs, per distance bin, that share a window.

    The denominator is same-FOV pairs, not all pairs: squidpy builds windows within `slide_fov`,
    so two cells in different FOVs cannot share a window at ANY window size and counting them
    would drag every arm's coverage down by the same constant, hiding the effect.

    Windows partition the cells -- for w400, 9,604 windows of mean 40 cells cover all 386,727 --
    so a pair shares at most one window and this is a fraction, not a rate.
    """
    ref = np.load(IN / "reference_pair_distances.npz", allow_pickle=True)
    ref_counts = ref["counts"]  # (K, K, n_bins)
    edges = ref["edges"]
    cell_types = list(ref["cell_types"].astype(str))
    px_to_um = float(ref["pixel_to_um"])

    # Self-pairs are not interactions and are trivially "covered" -- a cell always shares a window
    # with itself -- so they would put a spurious 1.0 in the first bin for every arm. Both sides of
    # the fraction lose them: the reference histograms all n^2 pairs of an FOV, self-pairs included.
    # (Unrelated to the relevance matrix's diagonal: this is pure geometry, no model involved.)
    #
    # Done ONCE, outside the arm loop: the per-type cell totals of the slide are what has to be
    # subtracted, and those do not depend on the windowing. Subtracting inside the loop would take
    # them off the reference again for every arm.
    k = np.arange(ref_counts.shape[0])
    cells_per_type = load_windows(arms[0])["per_type"].sum(axis=0)
    ref_counts = ref_counts.copy()
    ref_counts[k, k, 0] -= cells_per_type
    assert (ref_counts >= 0).all(), "reference self-pair subtraction went negative"

    rows = []
    for arm in arms:
        w = load_windows(arm)
        assert (w["per_type"].sum(axis=0) == cells_per_type).all(), (
            f"arm {arm} covers a different cell population than {arms[0]}; the arms are not "
            f"comparable and the shared reference denominator does not apply"
        )
        got = w["counts"].sum(axis=0).copy()  # (K, K, n_bins) pairs that share a window
        got[k, k, 0] -= cells_per_type
        assert (got >= 0).all(), "arm self-pair subtraction went negative"
        for b in range(ref_counts.shape[-1]):
            denom_all = ref_counts[:, :, b].sum()
            rows.append(
                {
                    "arm": arm,
                    "window_size": arm_window_size(arm),
                    "bin": b,
                    "d_lo_px": float(edges[b]),
                    "d_hi_px": float(edges[b + 1]),
                    "d_mid_px": float((edges[b] + edges[b + 1]) / 2),
                    "d_mid_um": float((edges[b] + edges[b + 1]) / 2 * px_to_um),
                    "pairs_possible": int(denom_all),
                    "pairs_covered": int(got[:, :, b].sum()),
                    "coverage": float(got[:, :, b].sum() / denom_all) if denom_all else np.nan,
                }
            )
        # Per cell-type pair, for the "which interactions specifically" question.
        for i, s in enumerate(cell_types):
            for j, r in enumerate(cell_types):
                denom = ref_counts[i, j].sum()
                rows.append(
                    {
                        "arm": arm,
                        "window_size": arm_window_size(arm),
                        "bin": -1,  # -1 = pooled over all distances
                        "sender": s,
                        "receiver": r,
                        "pairs_possible": int(denom),
                        "pairs_covered": int(got[i, j].sum()),
                        "coverage": float(got[i, j].sum() / denom) if denom else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 2. attention as a function of distance
# --------------------------------------------------------------------------------------------


def attention_distance_table(pairs) -> pd.DataFrame:
    """Mean attention and share of total attention mass, per sender-receiver distance bin.

    Two quantities, because they answer different questions:

      mean_attention  the average weight on a pair at that distance. Attention rows are
                      normalised, so a wider window spreads a fixed budget over more pairs and
                      this necessarily falls -- it is comparable ACROSS distances within an arm,
                      not across arms at one distance.
      mass_share      the fraction of the arm's total attention mass sitting at that distance.
                      This IS comparable across arms: it says how the model spends the budget it
                      has, independent of how large that budget's denominator is.

    THE DIAGONAL IS EXCLUDED. The relevance matrix is initialised to `eye(n)`, so its diagonal is
    ~1 by construction and comes to ~99% of the raw mass, all of it in the first bin; with it in,
    both columns are a plot of the identity matrix. It is reported once per (arm, seed) instead, in
    the returned table's `bin = -1` row, where `mass_share` is its share of the raw total.
    """
    rows = []
    for arm, seed in pairs:
        w = load_windows(arm)
        raw_sums = load_flows(arm, seed)
        sums, counts = drop_self_pairs(raw_sums, w["counts"], w["per_type"], load_diag(arm, seed))
        edges = w["edges"]
        total_mass = sums.sum()
        raw_total = raw_sums.sum()

        rows.append(
            {
                "arm": arm, "window_size": arm_window_size(arm), "seed": seed, "bin": -1,
                "d_lo_px": np.nan, "d_hi_px": np.nan, "d_mid_px": np.nan,
                "n_pairs": int(w["per_type"].sum()),
                "mean_attention": float(load_diag(arm, seed).sum() / w["per_type"].sum()),
                "mass_share": float(1 - total_mass / raw_total) if raw_total else np.nan,
            }
        )
        for b in range(counts.shape[-1]):
            c = counts[:, :, :, b].sum()
            s = sums[:, :, :, b].sum()
            rows.append(
                {
                    "arm": arm,
                    "window_size": arm_window_size(arm),
                    "seed": seed,
                    "bin": b,
                    "d_lo_px": float(edges[b]),
                    "d_hi_px": float(edges[b + 1]),
                    "d_mid_px": float((edges[b] + edges[b + 1]) / 2),
                    "n_pairs": int(c),
                    "mean_attention": float(s / c) if c else np.nan,
                    "mass_share": float(s / total_mass) if total_mass else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 3. cell-type flow matrices -- the reference definition, vectorised
# --------------------------------------------------------------------------------------------


def window_flow_matrices(sums, counts, compute_net=True):
    """Per-window normalised flow matrices, matching `compute_hierarchical_net_flow`.

    Parameters
    ----------
    sums, counts : (n_windows, K, K, n_bins) arrays
        Summed attention and pair counts from 5a. Pooled over distance bins here -- the reference
        takes the mean over ALL pairs of a type combination.
    compute_net : bool
        True -> agg.T - agg. False -> agg.T with a zeroed diagonal.

    Returns
    -------
    (n_windows, K, K) array
        Each window divided by its own max|value|, exactly as the reference does. That per-window
        normalisation is why windows of wildly different cell counts can be averaged at all, and
        also why these numbers are relative within a window rather than absolute attention.
    """
    s = sums.sum(axis=-1)
    c = counts.sum(axis=-1)
    # Type combinations absent from a window: the reference leaves those entries at their 0.0
    # initialisation and `continue`s, so a 0 here is "not present", not "no attention".
    with np.errstate(invalid="ignore", divide="ignore"):
        agg = np.where(c > 0, s / np.maximum(c, 1), 0.0)

    net = np.swapaxes(agg, 1, 2) - agg if compute_net else np.swapaxes(agg, 1, 2).copy()
    if not compute_net:
        idx = np.arange(net.shape[1])
        net[:, idx, idx] = 0.0

    mx = np.abs(net).max(axis=(1, 2), keepdims=True)
    return np.where(mx > 0, net / np.maximum(mx, 1e-300), net)


def hierarchical_aggregate(flows, slide_fov, condition=None):
    """Window -> sample -> condition aggregation, matching the reference's three levels.

    Returns ``{condition: {"mean": (K, K), "std": (K, K), "n_samples": int}}``, with the single
    key ``all_samples`` when ``condition`` is None.
    """
    out = {}
    fovs = np.asarray(slide_fov)
    sample_means = {}
    for fov in np.unique(fovs):
        sample_means[fov] = np.nanmean(flows[fovs == fov], axis=0)

    if condition is None:
        stacked = np.stack(list(sample_means.values()))
        out["all_samples"] = {
            "mean": np.nanmean(stacked, axis=0),
            "std": np.nanstd(stacked, axis=0),
            "n_samples": stacked.shape[0],
        }
        return out

    conds = np.asarray(condition)
    # A sample's condition, taken from its first window -- a slide_fov is entirely one condition.
    fov_cond = {fov: conds[fovs == fov][0] for fov in np.unique(fovs)}
    for cond in np.unique(conds):
        members = [sample_means[f] for f, c in fov_cond.items() if c == cond]
        if not members:
            continue
        stacked = np.stack(members)
        out[cond] = {
            "mean": np.nanmean(stacked, axis=0),
            "std": np.nanstd(stacked, axis=0),
            "n_samples": stacked.shape[0],
        }
    return out


def flow_tables(pairs, compute_net=True):
    """Long-form flow matrices for every (arm, seed) and condition."""
    rows = []
    matrices = {}
    for arm, seed in pairs:
        w = load_windows(arm)
        flows = window_flow_matrices(load_flows(arm, seed), w["counts"], compute_net=compute_net)
        cts = w["cell_types"]
        for cond_key, res in (
            list(hierarchical_aggregate(flows, w["slide_fov"]).items())
            + list(hierarchical_aggregate(flows, w["slide_fov"], w["condition"]).items())
        ):
            matrices[(arm, seed, cond_key)] = res
            for i, s in enumerate(cts):
                for j, r in enumerate(cts):
                    rows.append(
                        {
                            "arm": arm, "window_size": arm_window_size(arm), "seed": seed,
                            "condition": cond_key, "sender": s, "receiver": r,
                            "flow": float(res["mean"][i, j]), "flow_sd": float(res["std"][i, j]),
                            "n_samples": res["n_samples"],
                        }
                    )
    return pd.DataFrame(rows), matrices


def recovery_table(flow_df) -> pd.DataFrame:
    """How much of the widest arm's flow structure each narrower arm recovers.

    Reported against a same-arm, different-seed ceiling. Without it a correlation of 0.7 with the
    reference arm is uninterpretable: if two seeds of the SAME arm only agree to 0.7, then 0.7 is
    agreement, not loss. The ceiling is the mean pairwise between-seed correlation within the
    reference arm.

    Computed on the OFF-DIAGONAL entries only. `net = agg.T - agg` is exactly zero on the diagonal
    for every arm, so leaving those 9 entries in would have every pair of arms agreeing perfectly
    on 9 of 81 values and pull every correlation toward 1 by a constant that says nothing about
    recovery. (The (s,r)/(r,s) duplication is harmless to a correlation -- an exact sign flip
    preserves r -- so it is left in, but note the independent count is 36, not 72.)
    """
    d = flow_df[(flow_df["condition"] == "all_samples") & (flow_df["sender"] != flow_df["receiver"])]
    piv = d.pivot_table(index=["sender", "receiver"], columns=["arm", "seed"], values="flow")
    if REFERENCE_ARM not in piv.columns.get_level_values(0):
        raise SystemExit(
            f"the reference arm {REFERENCE_ARM} was not extracted, so there is nothing to measure "
            f"recovery against. Extracted arms: {sorted(set(piv.columns.get_level_values(0)))}"
        )

    rows = []
    ref_seeds = [s for (a, s) in piv.columns if a == REFERENCE_ARM]
    ref_mean = piv[REFERENCE_ARM].mean(axis=1)

    ceiling = []
    for i, s1 in enumerate(ref_seeds):
        for s2 in ref_seeds[i + 1 :]:
            ceiling.append(piv[(REFERENCE_ARM, s1)].corr(piv[(REFERENCE_ARM, s2)]))
    ceiling = float(np.mean(ceiling)) if ceiling else np.nan

    for arm, seed in piv.columns:
        v = piv[(arm, seed)]
        rows.append(
            {
                "arm": arm,
                "window_size": arm_window_size(arm),
                "seed": seed,
                "pearson_vs_reference": float(v.corr(ref_mean)),
                "spearman_vs_reference": float(v.corr(ref_mean, method="spearman")),
                "cosine_vs_reference": float(
                    np.dot(v, ref_mean) / (np.linalg.norm(v) * np.linalg.norm(ref_mean))
                ),
                "seed_ceiling": ceiling,
                "reference_arm": REFERENCE_ARM,
            }
        )
    return pd.DataFrame(rows).sort_values(["window_size", "seed"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 4. permutation test, per arm
# --------------------------------------------------------------------------------------------


def _window_agg_from_labels(attn, labels, n_types):
    """(K, K) mean attention per sender/receiver type pair, for arbitrary labels.

    Costs n^2 + nK per call rather than the reference's 81 pandas `.loc` calls, which is what
    makes 100 permutations x 3,200 windows feasible: `attn[labels == s].sum(0)` over all s touches
    every element exactly once.
    """
    n = attn.shape[0]
    row_sums = np.zeros((n_types, n), dtype=np.float64)
    n_per = np.zeros(n_types, dtype=np.int64)
    for s in range(n_types):
        m = labels == s
        cnt = int(m.sum())
        n_per[s] = cnt
        if cnt:
            row_sums[s] = attn[m].sum(axis=0)

    g = np.zeros((n_types, n_types), dtype=np.float64)
    for r in range(n_types):
        m = labels == r
        if m.any():
            g[:, r] = row_sums[:, m].sum(axis=1)
    denom = np.outer(n_per, n_per)
    return np.where(denom > 0, g / np.maximum(denom, 1), 0.0), n_per


def _normalise_net(agg, compute_net=True):
    net = agg.T - agg if compute_net else agg.T.copy()
    if not compute_net:
        np.fill_diagonal(net, 0.0)
    mx = np.abs(net).max()
    return net / mx if mx > 0 else net


def permutation_test(arm, seed, n_permutations=100, condition_key=None, compute_net=True,
                     random_seed=42, fractions=None):
    """Shuffle cell type labels within each window and rebuild the flow hierarchy.

    Same null as `run_spatial_permutation_test` in Interaction_analysis.ipynb: permuting within a
    window preserves that window's cell type composition and its size, and destroys only the
    spatial arrangement inside it. So the test asks "is this flow explained by which cell types
    are present, or by where they are?"

    Note what that means for a small window. A 40-cell w400 window has little internal
    arrangement to destroy, so its observed and null flows are close by construction -- the loss
    of significance at small window sizes is the result, not an artefact of the test.

    `fractions`, when given, runs the partial-permutation variant instead: only that fraction of
    each window's cells is shuffled, giving the degradation curve of the reference notebook.
    """
    rng = np.random.default_rng(random_seed)
    w = load_windows(arm)
    cts = w["cell_types"]
    n_types = len(cts)

    path = IN / "attn" / f"{arm}_seed{seed}.h5"
    with h5py.File(path, "r") as h5:
        grp = h5["windows"]
        names = list(w["window"])
        blocks, labels = [], []
        for name in names:
            g = grp[name]
            blocks.append(np.asarray(g["attn"], dtype=np.float32))
            labels.append(np.asarray(g["type_codes"], dtype=np.int64))

    fovs = w["slide_fov"]
    conds = w["condition"] if condition_key else None

    def hierarchy(all_labels):
        flows = np.stack(
            [
                _normalise_net(_window_agg_from_labels(b, lab, n_types)[0], compute_net)
                for b, lab in zip(blocks, all_labels, strict=True)
            ]
        )
        # BOTH levels, always: the pooled `all_samples` and, when a condition column was given,
        # the per-condition split. `hierarchical_aggregate` returns one or the other, never both,
        # so asking only for the per-condition version left no `all_samples` row -- which is the
        # row `detection_thresholds` reads, so the detection table came out empty.
        out = hierarchical_aggregate(flows, fovs)
        if conds is not None:
            out.update(hierarchical_aggregate(flows, fovs, conds))
        return out

    observed = hierarchy(labels)
    conditions = list(observed)

    if fractions is not None:
        rows = []
        for f in fractions:
            for p in range(n_permutations if f > 0 else 1):
                perm = []
                for lab in labels:
                    lab = lab.copy()
                    k = int(len(lab) * f)
                    if k > 1:
                        pick = rng.choice(len(lab), k, replace=False)
                        lab[pick] = rng.permutation(lab[pick])
                    perm.append(lab)
                res = hierarchy(perm)
                for cond in conditions:
                    m = np.abs(res[cond]["mean"])
                    # Signal per cell type: total absolute flow it is involved in as sender or
                    # receiver, halved so a type is not counted twice.
                    signal = (m.sum(axis=0) + m.sum(axis=1)) / 2.0
                    for i, ct in enumerate(cts):
                        rows.append({"arm": arm, "window_size": arm_window_size(arm), "seed": seed,
                                     "condition": cond, "fraction": f, "perm": p,
                                     "cell_type": ct, "signal": float(signal[i])})
        return pd.DataFrame(rows)

    null = {c: np.zeros((n_permutations, n_types, n_types)) for c in conditions}
    for p in range(n_permutations):
        perm = [rng.permutation(lab) for lab in labels]
        res = hierarchy(perm)
        for c in conditions:
            null[c][p] = res[c]["mean"] if c in res else np.nan

    rows = []
    for c in conditions:
        obs = observed[c]["mean"]
        stack = null[c]
        nm = np.nanmean(stack, axis=0)
        ns = np.nanstd(stack, axis=0) + 1e-9
        z = (obs - nm) / ns
        # Two-tailed empirical p, +1/(n+1) so a p of exactly 0 is never reported: with 100
        # permutations the smallest resolvable p is 1/101.
        pv = (np.sum(np.abs(stack) >= np.abs(obs)[None], axis=0) + 1) / (n_permutations + 1)
        for i, s in enumerate(cts):
            for j, r in enumerate(cts):
                rows.append(
                    {
                        "arm": arm, "window_size": arm_window_size(arm), "seed": seed,
                        "condition": c, "sender": s, "receiver": r,
                        "observed": float(obs[i, j]), "null_mean": float(nm[i, j]),
                        "null_std": float(ns[i, j]), "z": float(z[i, j]), "p": float(pv[i, j]),
                        "n_permutations": n_permutations,
                    }
                )
    return pd.DataFrame(rows)


def detection_thresholds(perm_df, alpha=0.05, min_seeds=2) -> pd.DataFrame:
    """The smallest window size at which each cell-type interaction becomes significant.

    `min_seeds` of an arm's seeds must agree, which is what the seed replication in the sweep is
    for: with one seed per arm, a single lucky permutation p-value would set the threshold.

    A pair never significant at any arm gets `detected_at = NaN`, and one significant already at
    the narrowest arm gets the narrowest window size -- that is the "seen by everything" class, the
    local interactions the GCN would find on its own.

    Note the flow matrix is ANTISYMMETRIC when `compute_net=True` (net = agg.T - agg), so
    `observed[s, r] == -observed[r, s]` and the z-scores mirror. All 81 ordered pairs are returned
    because the heatmap needs a full matrix, but there are only 36 independent off-diagonal
    interactions plus a structurally zero diagonal -- count the upper triangle, not the rows.
    """
    d = perm_df[perm_df["condition"] == "all_samples"].copy()
    d["sig"] = d["p"] < alpha
    agg = (
        d.groupby(["sender", "receiver", "arm", "window_size"])
        .agg(n_sig=("sig", "sum"), n_seeds=("sig", "size"), z=("z", "mean"), p=("p", "median"),
             observed=("observed", "mean"))
        .reset_index()
    )
    # Capped at the seeds actually present, so a partial extraction (or a deliberate single-seed
    # run) reports what it found instead of silently declaring nothing detected: requiring 2 of 1
    # is unsatisfiable, and the whole table came back empty.
    agg["detected"] = agg["n_sig"] >= np.minimum(min_seeds, agg["n_seeds"])

    rows = []
    for (s, r), g in agg.groupby(["sender", "receiver"]):
        g = g.sort_values("window_size")
        hit = g[g["detected"]]
        rows.append(
            {
                "sender": s,
                "receiver": r,
                "detected_at": float(hit["window_size"].iloc[0]) if len(hit) else np.nan,
                "n_arms_detected": int(g["detected"].sum()),
                "z_narrowest": float(g["z"].iloc[0]),
                "z_widest": float(g["z"].iloc[-1]),
                "abs_z_gain": float(abs(g["z"].iloc[-1]) - abs(g["z"].iloc[0])),
                "flow_narrowest": float(g["observed"].iloc[0]),
                "flow_widest": float(g["observed"].iloc[-1]),
            }
        )
    return pd.DataFrame(rows).sort_values(["detected_at", "abs_z_gain"], ascending=[True, False])


# --------------------------------------------------------------------------------------------
# 5. is any of this reproducible? -- the diagnostic that gates every claim above
# --------------------------------------------------------------------------------------------


def seed_stability_table(flow_df, perm_df, alpha=0.05) -> pd.DataFrame:
    """Whether the cell-type interaction structure reproduces across seeds at all.

    This is not a nicety. Three quantities in this script are only interpretable if the answer is
    yes, and on these models it is largely NO:

      flow_r_between_seeds   mean pairwise Pearson r of the off-diagonal flow matrix between the
                             3 seeds of one arm. If this is ~0, the "recovery vs the widest arm"
                             curve is measuring noise against noise: you cannot ask how much of a
                             reference a narrower arm recovers when the reference does not
                             reproduce itself.
      sig_jaccard            mean pairwise Jaccard of the SETS of significant pairs between seeds.
                             Gates the detection-threshold table: a pair that is significant in a
                             different seed each time has no threshold.
      n_windows              the confound that makes `z` incomparable across arms. z is
                             (observed - null_mean) / null_std, and null_std shrinks with the
                             number of windows averaged over -- which runs from 3426 (w400) to 100
                             (w3000), a 34x range. A larger |z| at a small window size therefore
                             means "more windows", not "stronger evidence", and the raw z column of
                             detection_thresholds.csv must not be read as an effect size.

    Reported rather than corrected: with 3 seeds there is no honest way to rescue a per-pair
    threshold from label sets that agree at Jaccard ~0.4, and the right conclusion is that this
    question needs either more seeds or a model that fits the data.
    """
    rows = []
    d_flow = flow_df[(flow_df["condition"] == "all_samples")
                     & (flow_df["sender"] != flow_df["receiver"])]
    d_perm = perm_df[(perm_df["condition"] == "all_samples")
                     & (perm_df["sender"] < perm_df["receiver"])]

    for arm in [a for a in ARMS if a in set(d_flow["arm"])]:
        piv = d_flow[d_flow["arm"] == arm].pivot_table(
            index=["sender", "receiver"], columns="seed", values="flow"
        )
        seeds = list(piv.columns)
        r = [piv[s1].corr(piv[s2]) for i, s1 in enumerate(seeds) for s2 in seeds[i + 1 :]]

        g = d_perm[d_perm["arm"] == arm]
        sets = [
            set(map(tuple, gg[gg["p"] < alpha][["sender", "receiver"]].to_numpy()))
            for _, gg in g.groupby("seed")
        ]
        jac = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                union = sets[i] | sets[j]
                jac.append(len(sets[i] & sets[j]) / len(union) if union else np.nan)

        w = load_windows(arm)
        # |z| per seed, so the WITHIN-arm spread can be put beside the between-arm differences.
        z_by_seed = g.groupby("seed")["z"].apply(lambda z: z.abs().mean())
        rows.append({
            "arm": arm,
            "window_size": arm_window_size(arm),
            "n_windows": len(w["n_cells"]),
            "cells_per_window": float(w["n_cells"].mean()),
            "flow_r_between_seeds": float(np.nanmean(r)) if r else np.nan,
            "flow_r_min": float(np.nanmin(r)) if r else np.nan,
            "flow_magnitude": float(piv.abs().to_numpy().mean()),
            "n_sig_mean": float(np.mean([len(s) for s in sets])),
            "n_sig_min": int(min(len(s) for s in sets)) if sets else 0,
            "n_sig_max": int(max(len(s) for s in sets)) if sets else 0,
            "sig_jaccard": float(np.nanmean(jac)) if jac else np.nan,
            "mean_abs_z": float(z_by_seed.mean()),
            "abs_z_sd_between_seeds": float(z_by_seed.std()),
            "abs_z_min": float(z_by_seed.min()),
            "abs_z_max": float(z_by_seed.max()),
        })
    return pd.DataFrame(rows).sort_values("window_size").reset_index(drop=True)


def fig_seed_stability(stab_df, path=None):
    """Three panels saying: the geometry is solid, the per-pair interaction calls are not."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

    ax = axes[0]
    ax.plot(stab_df["window_size"], stab_df["flow_r_between_seeds"], marker="o", ms=7, lw=2,
            color=ARM_COLORS[ARMS[3]], label="between seeds, same arm")
    ax.fill_between(stab_df["window_size"], stab_df["flow_r_min"],
                    stab_df["flow_r_between_seeds"], color=ARM_COLORS[ARMS[3]], alpha=0.2, lw=0)
    ax.axhline(0, color=INK_2, lw=1.2, ls=(0, (5, 3)))
    ax.set_ylim(-1, 1)
    ax.set_xlabel("nominal window size (px)")
    ax.set_ylabel("Pearson r of the flow matrix")
    ax.set_title("A  Seed reproducibility", **TITLE_KW, pad=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    ax.plot(stab_df["window_size"], stab_df["sig_jaccard"], marker="o", ms=7, lw=2, color=ARM_COLORS[ARMS[3]])
    ax.set_ylim(0, 1)
    ax.set_xlabel("nominal window size (px)")
    ax.set_ylabel("Jaccard between seeds")
    ax.set_title("B  Same pairs called significant?", **TITLE_KW, pad=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    # Between-arm differences in |z| against the within-arm spread across seeds. Drawn this way
    # deliberately: an earlier version of this panel plotted mean |z| against the window count and
    # titled it "null_std shrinks with the window count", which the data does not show -- 2000 px
    # has the second-fewest windows and the HIGHEST |z|, 3000 px the fewest and nearly the lowest.
    # The window count is a real theoretical confound on z (null_std does shrink as more windows
    # are averaged, and the count spans 100-3426), but it is not what drives these numbers. Seed
    # variation is: the error bars overlap almost every arm with almost every other.
    ax = axes[2]
    ax.errorbar(stab_df["window_size"], stab_df["mean_abs_z"],
                yerr=stab_df["abs_z_sd_between_seeds"], marker="o", ms=7, lw=2,
                capsize=4, color=C_WORST, ecolor=INK_2, elinewidth=1)
    ax.set_xlabel("nominal window size (px)")
    ax.set_ylabel("mean |z| of the permutation test")
    ax.set_title("C  |z| does not order the arms", **TITLE_KW, pad=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return _finish(fig, path)


# --------------------------------------------------------------------------------------------
# validation against the reference implementation
# --------------------------------------------------------------------------------------------


def _load_extraction_module():
    """Import 5a as a module. Its name starts with a digit, so `import` will not do it."""
    import importlib.util

    path = Path(__file__).with_name("5a_sliding_window_attention.py")
    spec = importlib.util.spec_from_file_location("sliding_window_extraction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def storage_precision_gap(arm=REFERENCE_ARM, seed=SEEDS[0], n_windows=12):
    """How far apart the two stored products are, and why.

    5a bins the full-precision float32 attention into `flows/*.npz` but writes float16 into
    `attn/*.h5`, so the flow matrices (from the npz) and the permutation test's observed flow (from
    the h5) are the same quantity at two precisions. Both are internally consistent -- the
    permutation test computes observed AND null from the h5 -- so no result mixes them, but the two
    CSVs disagree by this much and it is better measured than assumed.
    """
    w = load_windows(arm)
    order = np.argsort(-w["n_cells"])[:n_windows]
    bin_window = _load_extraction_module().bin_window
    n_types = len(w["cell_types"])

    with h5py.File(IN / "attn" / f"{arm}_seed{seed}.h5", "r") as h5:
        binned = []
        for i in order:
            g = h5["windows"][w["window"][i]]
            binned.append(
                bin_window(
                    np.asarray(g["attn"], dtype=np.float32),
                    np.asarray(g["type_codes"], dtype=np.int64),
                    np.asarray(g["coords"], dtype=np.float32),
                    n_types,
                )
            )
    from_h5 = np.stack([s for s, _ in binned])
    from_npz = load_flows(arm, seed)[order]

    flow_h5 = window_flow_matrices(from_h5, w["counts"][order])
    flow_npz = window_flow_matrices(from_npz, w["counts"][order])
    gap = float(np.nanmax(np.abs(flow_h5 - flow_npz)))
    print(f"{arm} seed {seed}: max |flow(float16 h5) - flow(float32 npz)| = {gap:.3e} "
          f"over {n_windows} of the widest windows")
    return gap


def validate_against_reference(arm=REFERENCE_ARM, seed=SEEDS[0], n_windows=12, tol=1e-6):
    """Assert `window_flow_matrices` reproduces `compute_hierarchical_net_flow` exactly.

    Runs the real function on a handful of windows built into a small AnnData and compares. Both
    the net and the raw-incoming branches, because they normalise differently.

    The binned sums are recomputed here from the SAME float16 blocks the reference is fed, rather
    than read from `flows/*.npz`. That is the point of the test: 5a bins the full-precision float32
    attention into the npz but writes float16 into the h5, so those two products describe the same
    quantity at two precisions and differ by ~1e-4 on the widest arms. Comparing the npz against a
    float16-fed reference measures that storage gap, not the aggregation -- it is what made this
    assertion fail at 1.66e-4 on w2000 while passing at 4e-5 on w400, purely because w400's blocks
    are smaller and its weights larger. Feeding both sides identical inputs isolates the algorithm,
    which is the thing under test, and lets the tolerance be tight.
    """
    import anndata as ad
    from interscale.evaluation.net_streams import compute_hierarchical_net_flow

    w = load_windows(arm)
    cts = w["cell_types"]
    n_types = len(cts)

    # Widest windows first -- a 1-cell window agrees trivially and would prove nothing.
    order = np.argsort(-w["n_cells"])[:n_windows]
    with h5py.File(IN / "attn" / f"{arm}_seed{seed}.h5", "r") as h5:
        grp = h5["windows"]
        blocks, labels, names, coords = [], [], [], []
        for i in order:
            name = w["window"][i]
            g = grp[name]
            blocks.append(np.asarray(g["attn"], dtype=np.float32))
            labels.append(np.asarray(g["type_codes"], dtype=np.int64))
            coords.append(np.asarray(g["coords"], dtype=np.float32))
            names.append(name)

    # 5a's own binning, applied to the float16 blocks, so both sides of the comparison start from
    # byte-identical values. Imported rather than reimplemented: a second copy of the binning would
    # make this a test of the copy.
    bin_window = _load_extraction_module().bin_window
    binned = [bin_window(b, lab, xy, n_types) for b, lab, xy in zip(blocks, labels, coords, strict=True)]
    sums_sub = np.stack([s for s, _ in binned])
    counts_sub = np.stack([c for _, c in binned])

    max_n = max(b.shape[0] for b in blocks)
    n_cells = sum(b.shape[0] for b in blocks)
    attn = np.full((n_cells, max_n), np.nan, dtype=np.float32)
    obs = {"win": [], "fov": [], "ct": []}
    at = 0
    for name, b, lab, i in zip(names, blocks, labels, order, strict=True):
        n = b.shape[0]
        attn[at : at + n, :n] = b
        obs["win"] += [name] * n
        obs["fov"] += [w["slide_fov"][i]] * n
        obs["ct"] += [cts[c] for c in lab]
        at += n

    small = ad.AnnData(
        X=np.zeros((n_cells, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "win": pd.Categorical(obs["win"]),
                "fov": pd.Categorical(obs["fov"]),
                "ct": pd.Categorical(obs["ct"], categories=cts),
            }
        ),
    )
    small.obsm["_attn_matrix"] = attn

    for compute_net in (True, False):
        ref = compute_hierarchical_net_flow(
            small, window_key="win", sample_key="fov", cell_type_col="ct", compute_net=compute_net
        )["all_samples"]["mean"]
        mine = hierarchical_aggregate(
            window_flow_matrices(sums_sub, counts_sub, compute_net=compute_net),
            w["slide_fov"][order],
        )["all_samples"]["mean"]
        # The reference indexes by sorted(unique(present cell types)), which on a 12-window
        # subsample can be a subset of the full 9. Compare only the types it produced.
        idx = [cts.index(c) for c in ref.index]
        diff = np.abs(np.asarray(ref.values) - mine[np.ix_(idx, idx)])
        worst = float(np.nanmax(diff))
        print(f"  compute_net={compute_net}: max |reference - reimplementation| = {worst:.3e}")
        assert worst < tol, f"reimplementation disagrees with compute_hierarchical_net_flow ({worst:.3e})"
    print(f"validated on {n_windows} windows of {arm} seed {seed} ({n_cells} cells)")


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def _finish(fig, path):
    """Save and close when writing a file; hand the figure back when a notebook wants it inline."""
    if path is None:
        return fig
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return None


def fig_coverage(cov_df, path=None):
    d = cov_df[cov_df["bin"] >= 0]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    for arm, g in d.groupby("arm", sort=False):
        g = g.sort_values("d_mid_px")
        ax.plot(g["d_mid_px"], g["coverage"], marker="o", ms=4, lw=1.8,
                color=ARM_COLORS[arm], label=f"{arm_window_size(arm)} px")
    ax.set_xscale("log")
    ax.set_xlabel("sender-receiver distance (FOV-local px)")
    ax.set_ylabel("fraction of same-FOV pairs sharing a window")
    ax.set_title("A  What the windowing can expose", **TITLE_KW, pad=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    ax = axes[1]
    # Summed then divided, not a mean of per-bin coverages: the bins hold wildly different numbers
    # of pairs (152k in the first, 270M in the 2000-3000 px one), so averaging the fractions would
    # weight a nearly empty bin like a huge one.
    tot = (
        d.groupby(["arm", "window_size"])[["pairs_covered", "pairs_possible"]]
        .sum()
        .assign(coverage=lambda t: t["pairs_covered"] / t["pairs_possible"])
        .reset_index()
        .sort_values("window_size")
    )
    ax.plot(tot["window_size"], tot["coverage"], marker="o", ms=7, lw=2, color=ARM_COLORS[ARMS[3]])
    for _, r in tot.iterrows():
        ax.annotate(f"{r['coverage']:.3f}", (r["window_size"], r["coverage"]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("nominal window size (px)")
    ax.set_ylabel("fraction of all same-FOV pairs covered")
    ax.set_title("B  Pooled over distance", **TITLE_KW, pad=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)

    # Legend outside the axes, to the right of the whole figure: inside panel A it sat on top of
    # the 3000 px curve, and panel B has a single series so it needs no legend of its own.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="window size", loc="center left",
               bbox_to_anchor=(1.0, 0.5), **{**LEGEND_KW, "prop": {"size": 9}})
    fig.tight_layout()
    return _finish(fig, path)


def fig_attention_distance(att_df, path=None):
    # bin == -1 is the diagonal, which has no distance. Reported in the subtitle rather than
    # plotted: at ~0.99 of the raw mass it would compress every other point onto the axis, and it
    # is the relevance matrix's eye(n) initialisation rather than a model output anyway.
    self_share = att_df[att_df["bin"] == -1]["mass_share"]
    att_df = att_df[att_df["bin"] >= 0]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, (col, label, title) in zip(
        axes,
        [
            ("mass_share", "share of non-self attention mass",
             "A  Where attention is spent"),
            ("mean_attention", "mean attention weight per pair",
             "B  Weight per pair"),
        ],
        strict=True,
    ):
        for arm, g in att_df.groupby("arm", sort=False):
            # mean/sd across the 3 seeds at each distance bin.
            m = (
                g.groupby("d_mid_px")[col]
                .agg(mean="mean", std="std")
                .reset_index()
                .sort_values("d_mid_px")
            )
            ax.plot(m["d_mid_px"], m["mean"], marker="o", ms=4, lw=1.8,
                    color=ARM_COLORS[arm], label=f"{arm_window_size(arm)} px")
            ax.fill_between(m["d_mid_px"], m["mean"] - m["std"], m["mean"] + m["std"],
                            color=ARM_COLORS[arm], alpha=0.25, lw=0)
        ax.axvline(110, color=INK_2, lw=1, ls=(0, (4, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("sender-receiver distance (FOV-local px)")
        ax.set_ylabel(label)
        ax.set_title(title, **TITLE_KW, pad=6)
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].legend(title="window size", **{**LEGEND_KW, "prop": {"size": 9}})
    if len(self_share):
        fig.suptitle(
            f"diagonal excluded: the relevance matrix is initialised to eye(n), so its diagonal is "
            f"{self_share.mean():.1%} of the raw mass "
            f"({self_share.min():.1%}-{self_share.max():.1%} over {len(self_share)} models)",
            **SUB_KW, color=INK_2, y=1.03,
        )
    fig.tight_layout()
    return _finish(fig, path)


def fig_recovery(rec_df, path=None):
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    m = (
        rec_df.groupby("window_size")["pearson_vs_reference"]
        .agg(mean="mean", std="std")
        .reset_index()
        .sort_values("window_size")
    )
    ax.errorbar(m["window_size"], m["mean"], yerr=m["std"], marker="o", ms=7, lw=2,
                capsize=4, color=ARM_COLORS[ARMS[3]], ecolor=INK_2, elinewidth=1)
    ceiling = rec_df["seed_ceiling"].iloc[0]
    ax.axhline(ceiling, color=INK_2, lw=1.2, ls=(0, (5, 3)))
    ax.text(m["window_size"].max(), ceiling, f" seed-to-seed ceiling ({ceiling:.2f})",
            ha="right", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xlabel("nominal window size (px)")
    ax.set_ylabel(f"Pearson r of the flow matrix vs. {REFERENCE_ARM}")
    ax.set_title("Recovery of the widest arm", **TITLE_KW, pad=6)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return _finish(fig, path)


def fig_detection_threshold(det_df, cell_types, path=None):
    """The headline: for each ordered cell-type pair, the narrowest window that detects it."""
    mat = det_df.pivot(index="sender", columns="receiver", values="detected_at").reindex(
        index=cell_types, columns=cell_types
    )
    gain = det_df.pivot(index="sender", columns="receiver", values="abs_z_gain").reindex(
        index=cell_types, columns=cell_types
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    sizes = sorted(det_df["detected_at"].dropna().unique())
    cmap = mpl.colors.ListedColormap([ARM_COLORS[f"w{int(s)}"] for s in sizes]) if sizes else CONTINUOUS
    norm = mpl.colors.BoundaryNorm(
        [*(sizes + [max(sizes) + 1])], len(sizes)
    ) if sizes else None

    sns.heatmap(mat, ax=axes[0], cmap=cmap, norm=norm, annot=True, fmt=".0f",
                annot_kws={"fontsize": 7}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "narrowest window that detects (px)"})
    axes[0].set_title("A  Detection threshold", **TITLE_KW, pad=6)
    axes[0].set_xlabel("receiver cell type")
    axes[0].set_ylabel("sender cell type")

    lim = float(np.nanmax(np.abs(gain.values))) if np.isfinite(gain.values).any() else 1.0
    sns.heatmap(gain, ax=axes[1], cmap=DIVERGENT, center=0, vmin=-lim, vmax=lim,
                annot=True, fmt=".1f", annot_kws={"fontsize": 7},
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "|z| at widest - |z| at narrowest"})
    axes[1].set_title("B  Evidence added by the widest window", **TITLE_KW, pad=6)
    axes[1].set_xlabel("receiver cell type")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        for lab in ax.get_xticklabels():
            lab.set_ha("right")
    fig.tight_layout()
    return _finish(fig, path)


def fig_detection_threshold_merged(det_df, cell_types, path=None):
    """Both halves of fig_detection_threshold in ONE triangular cell-type x cell-type matrix.

    Below the diagonal: the narrowest window that detects the interaction.
    Above the diagonal: |z| at the widest window minus |z| at the narrowest.

    Merging is lossless because both quantities are symmetric in sender/receiver. The net flow is
    antisymmetric (net = agg.T - agg), so |z| and p are identical for A->B and B->A, and the
    directed 9x9 panels carried every number twice. What is plotted here is therefore 36
    independent interactions rather than 81 cells, and the axes are cell type rather than
    sender/receiver.

    The upper half is kept deliberately hard to over-read: z is NOT comparable across window sizes,
    because null_std shrinks with the number of windows averaged (rho = -1.000 against window
    count, from 0.0021 at 3426 windows to 0.0371 at 100), and the values themselves do not
    reproduce across seeds (between-seed r = -0.02, per-pair sd 3.97 against a range of -10.6 to
    2.1). It is shown as a diagnostic, not as evidence; the footnote on the figure says so.
    """
    n = len(cell_types)
    lower = (det_df.pivot(index="sender", columns="receiver", values="detected_at")
             .reindex(index=cell_types, columns=cell_types).to_numpy(dtype=float))
    upper = (det_df.pivot(index="sender", columns="receiver", values="abs_z_gain")
             .reindex(index=cell_types, columns=cell_types).to_numpy(dtype=float))

    # Both must be symmetric for the merge to be lossless; assert rather than assume.
    for name, m in (("detected_at", lower), ("abs_z_gain", upper)):
        finite = np.isfinite(m) & np.isfinite(m.T)
        assert np.allclose(m[finite], m.T[finite]), f"{name} is not symmetric; merging would lose data"

    iu = np.triu_indices(n, k=1)      # upper triangle -> evidence gain
    il = np.tril_indices(n, k=-1)     # lower triangle -> detection threshold
    up = np.full((n, n), np.nan); up[iu] = upper[iu]
    lo = np.full((n, n), np.nan); lo[il] = lower[il]

    fig, ax = plt.subplots(figsize=(8.6, 7.4))

    # Lower: discrete window sizes, in the ladder's own colours.
    sizes = sorted(np.unique(lo[np.isfinite(lo)]))
    cmap_lo = mpl.colors.ListedColormap([ARM_COLORS[f"w{int(v)}"] for v in sizes])
    norm_lo = mpl.colors.BoundaryNorm(list(sizes) + [max(sizes) + 1], len(sizes))
    im_lo = ax.imshow(np.ma.masked_invalid(lo), cmap=cmap_lo, norm=norm_lo)

    # Upper: divergent, symmetric about zero.
    lim = float(np.nanmax(np.abs(up))) if np.isfinite(up).any() else 1.0
    im_up = ax.imshow(np.ma.masked_invalid(up), cmap=DIVERGENT, vmin=-lim, vmax=lim)

    # Cells that are neither: the diagonal, and pairs never detected at any window size.
    blank = np.zeros((n, n), dtype=bool)
    blank[np.diag_indices(n)] = True
    blank[il] |= ~np.isfinite(lo[il])
    ax.imshow(np.ma.masked_where(~blank, np.ones((n, n))),
              cmap=mpl.colors.ListedColormap(["#e8e8e6"]), vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "–", ha="center", va="center", fontsize=10, color=INK_2)
            elif i > j:
                v = lo[i, j]
                ax.text(j, i, "n.d." if not np.isfinite(v) else f"{int(v)}",
                        ha="center", va="center", fontsize=8,
                        color="#8a8a86" if not np.isfinite(v) else (
                            "white" if v >= sizes[len(sizes) // 2 + 1] else INK))
            else:
                v = up[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.62 * lim else INK)

    ax.set_xticks(range(n), cell_types, rotation=45, ha="right")
    ax.set_yticks(range(n), cell_types)
    for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lab.set_color(CT_COLORS[lab.get_text()])
        lab.set_fontsize(9)
    ax.set_xlabel("cell type")
    ax.set_ylabel("cell type")
    ax.set_title("Detection threshold and evidence gain", **TITLE_KW, pad=TITLE_PAD_SUB)
    ax.text(0.5, 1.005, "lower: narrowest window that detects   ·   upper: |z| gain, widest − narrowest",
            transform=ax.transAxes, ha="center", va="bottom", color=INK_2, **SUB_KW)

    # Two colourbars on one axes: `fig.colorbar(ax=...)` negotiates their positions against
    # tight_layout and put the inner bar's label on top of the outer bar's colours. Explicit
    # axes via make_axes_locatable make the gap a number rather than an outcome -- `pad` for the
    # second is measured from the first, so 0.95 in is a real separation.
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    div = make_axes_locatable(ax)
    cax_lo = div.append_axes("right", size="4%", pad=0.20)
    cax_up = div.append_axes("right", size="4%", pad=0.95)

    bounds = list(sizes) + [max(sizes) + 1]
    cb_lo = fig.colorbar(im_lo, cax=cax_lo, boundaries=bounds,
                         ticks=[(bounds[i] + bounds[i + 1]) / 2 for i in range(len(sizes))])
    cb_lo.ax.set_yticklabels([f"{int(v)}" for v in sizes], fontsize=8)
    cb_lo.set_label("lower: narrowest window (px)", fontsize=9, labelpad=8)

    cb_up = fig.colorbar(im_up, cax=cax_up)
    cb_up.ax.tick_params(labelsize=8)
    cb_up.set_label("upper: |z| gain", fontsize=9, labelpad=8)

    fig.text(0.5, 0.015,
             "n.d. = never significant at any window size (p<0.05 in ≥2 of 3 seeds).  "
             "Both quantities are symmetric, so this is 36 independent interactions.\n"
             "The upper half is diagnostic only: z is not comparable across window sizes "
             "(null_std tracks the window count, ρ = −1.00) and does not reproduce across seeds "
             "(between-seed r = −0.02).",
             ha="center", fontsize=7.2, color=INK_2, linespacing=1.5)
    fig.subplots_adjust(left=0.14, right=0.80, top=0.90, bottom=0.20)
    return _finish(fig, path)


def fig_observed_vs_null(perm_df, cell_types, path=None, condition="all_samples"):
    """Observed flow beside its within-window null, one column per arm."""
    arms = [a for a in ARMS if a in set(perm_df["arm"])]
    fig, axes = plt.subplots(2, len(arms), figsize=(3.1 * len(arms), 6.8), squeeze=False)
    d = perm_df[perm_df["condition"] == condition]
    lim = float(np.nanmax(np.abs(d["observed"])))

    for k, arm in enumerate(arms):
        g = d[d["arm"] == arm].groupby(["sender", "receiver"], as_index=False)[
            ["observed", "null_mean", "p"]
        ].mean()
        obs = g.pivot(index="sender", columns="receiver", values="observed").reindex(cell_types, axis=0).reindex(cell_types, axis=1)
        nul = g.pivot(index="sender", columns="receiver", values="null_mean").reindex(cell_types, axis=0).reindex(cell_types, axis=1)
        pv = g.pivot(index="sender", columns="receiver", values="p").reindex(cell_types, axis=0).reindex(cell_types, axis=1)
        stars = pv.map(lambda v: "***" if v < 0.001 else ("**" if v < 0.01 else ("*" if v < 0.05 else "")))

        for row, (m, name) in enumerate([(obs, "observed"), (nul, "null")]):
            ax = axes[row][k]
            sns.heatmap(m, ax=ax, cmap=DIVERGENT, center=0, vmin=-lim, vmax=lim, cbar=False,
                        annot=stars if row == 0 else False, fmt="", annot_kws={"fontsize": 6})
            ax.set_title(f"{arm_window_size(arm)} px — {name}" if row == 0 else "", fontsize=8.5)
            ax.set_xlabel("receiver" if row == 1 else "")
            ax.set_ylabel("sender" if k == 0 else "")
            ax.set_xticklabels(ax.get_xticklabels() if row == 1 else [], rotation=45, ha="right", fontsize=6)
            ax.set_yticklabels(cell_types if k == 0 else [], rotation=0, fontsize=6)

    fig.suptitle("Observed net flow vs. permuted null", **TITLE_KW, y=1.02)
    fig.tight_layout()
    return _finish(fig, path)


def fig_partial_permutation(part_df, path=None, condition="all_samples"):
    d = part_df[part_df["condition"] == condition]
    arms = [a for a in ARMS if a in set(d["arm"])]
    fig, axes = plt.subplots(1, len(arms), figsize=(2.7 * len(arms), 4.0), sharey=True, squeeze=False)
    types = sorted(d["cell_type"].unique())
    # Cell-type colours come from figures/config.yml so a cell type is the same colour in every
    # figure of the manuscript.
    pal = STYLE.cell_types("melton25", types)

    for k, arm in enumerate(arms):
        ax = axes[0][k]
        g = d[d["arm"] == arm].groupby(["cell_type", "fraction"], as_index=False)["signal"].mean()
        for ct, gg in g.groupby("cell_type"):
            gg = gg.sort_values("fraction")
            ax.plot(gg["fraction"], gg["signal"], marker="o", ms=3.5, lw=1.5, color=pal[ct], label=ct)
        ax.set_title(f"{arm_window_size(arm)} px", **TITLE_KW, pad=6)
        ax.set_xlabel("fraction of cells permuted")
        if k == 0:
            ax.set_ylabel("aggregate absolute net flow")
        ax.grid(color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0][-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                       **{**LEGEND_KW, "prop": {"size": 8}})
    fig.suptitle("Signal degradation under partial spatial scrambling", **TITLE_KW, y=1.05)
    fig.tight_layout()
    return _finish(fig, path)


# --------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument("--partial-permutations", type=int, default=20,
                    help="Permutations per shuffle fraction for the degradation curves.")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.5, 0.8, 1.0])
    ap.add_argument("--raw-incoming", action="store_true",
                    help="compute_net=False: raw incoming attention instead of net flow.")
    ap.add_argument("--plot-only", action="store_true", help="Reuse the CSVs already written.")
    ap.add_argument("--skip-partial", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="Only check the vectorised flow against compute_hierarchical_net_flow.")
    ap.add_argument("--validate-arm", default=REFERENCE_ARM, choices=ARMS,
                    help="Arm to validate against (default: the reference arm).")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    compute_net = not args.raw_incoming

    if args.validate:
        validate_against_reference(arm=args.validate_arm, seed=args.seeds[0])
        storage_precision_gap(arm=args.validate_arm, seed=args.seeds[0])
        return

    pairs = available(args.arms, args.seeds)
    arms_present = sorted({a for a, _ in pairs}, key=arm_window_size)
    cell_types = load_windows(arms_present[0])["cell_types"]
    print(f"analysing {len(pairs)} (arm, seed) pairs: {arms_present}")

    if args.plot_only:
        cov = pd.read_csv(OUT / "coverage_by_distance.csv")
        att = pd.read_csv(OUT / "attention_by_distance.csv")
        rec = pd.read_csv(OUT / "recovery.csv")
        perm = pd.read_csv(OUT / "permutation_stats.csv")
        det = pd.read_csv(OUT / "detection_thresholds.csv")
        # Derived on the fly when absent, from the two CSVs it needs, so a run predating this
        # diagnostic does not have to repeat 18 x 100 permutations to gain it.
        if (OUT / "seed_stability.csv").exists():
            stab = pd.read_csv(OUT / "seed_stability.csv")
        else:
            print("seed_stability.csv missing; recomputing from flow_matrices.csv + permutation_stats.csv")
            stab = seed_stability_table(pd.read_csv(OUT / "flow_matrices.csv"), perm)
            stab.to_csv(OUT / "seed_stability.csv", index=False)
        part = (
            pd.read_csv(OUT / "partial_permutation.csv")
            if (OUT / "partial_permutation.csv").exists() else None
        )
    else:
        print("\n[1/5] structural coverage")
        cov = coverage_table(arms_present)
        cov.to_csv(OUT / "coverage_by_distance.csv", index=False)

        print("[2/5] attention vs distance")
        att = attention_distance_table(pairs)
        att.to_csv(OUT / "attention_by_distance.csv", index=False)

        print("[3/5] cell-type flow matrices and recovery")
        flow_df, _ = flow_tables(pairs, compute_net=compute_net)
        flow_df.to_csv(OUT / "flow_matrices.csv", index=False)
        rec = recovery_table(flow_df)
        rec.to_csv(OUT / "recovery.csv", index=False)

        print(f"[4/5] permutation test ({args.n_permutations} permutations per arm/seed)")
        perm_parts = []
        for arm, seed in pairs:
            t0 = time.time()
            perm_parts.append(
                permutation_test(arm, seed, n_permutations=args.n_permutations,
                                 condition_key="condition", compute_net=compute_net)
            )
            print(f"  {arm} seed {seed}: {time.time() - t0:.0f}s", flush=True)
        perm = pd.concat(perm_parts, ignore_index=True)
        perm.to_csv(OUT / "permutation_stats.csv", index=False)
        det = detection_thresholds(perm)
        det.to_csv(OUT / "detection_thresholds.csv", index=False)

        stab = seed_stability_table(flow_df, perm)
        stab.to_csv(OUT / "seed_stability.csv", index=False)

        part = None
        if not args.skip_partial:
            print(f"[5/5] partial permutation curves ({args.partial_permutations} per fraction)")
            part_parts = []
            for arm, seed in pairs:
                t0 = time.time()
                part_parts.append(
                    permutation_test(arm, seed, n_permutations=args.partial_permutations,
                                     compute_net=compute_net, fractions=args.fractions)
                )
                print(f"  {arm} seed {seed}: {time.time() - t0:.0f}s", flush=True)
            part = pd.concat(part_parts, ignore_index=True)
            part.to_csv(OUT / "partial_permutation.csv", index=False)

    print("\nfigures")
    fig_coverage(cov, OUT / "fig_coverage.png")
    fig_attention_distance(att, OUT / "fig_attention_distance.png")
    fig_recovery(rec, OUT / "fig_recovery.png")
    fig_detection_threshold(det, cell_types, OUT / "fig_detection_threshold.png")
    fig_detection_threshold_merged(det, cell_types, OUT / "fig_detection_threshold_merged.png")
    fig_observed_vs_null(perm, cell_types, OUT / "fig_observed_vs_null.png")
    fig_seed_stability(stab, OUT / "fig_seed_stability.png")
    if part is not None:
        fig_partial_permutation(part, OUT / "fig_partial_permutation.png")

    print("\n--- IS THIS REPRODUCIBLE? read before the detection table ---")
    print(stab[["window_size", "n_windows", "flow_r_between_seeds", "flow_r_min",
                "n_sig_mean", "sig_jaccard", "mean_abs_z"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    weak = stab[stab["flow_r_between_seeds"] < 0.5]
    if len(weak):
        print(
            f"\nWARNING: {len(weak)} of {len(stab)} arms have a between-seed flow correlation "
            f"below 0.5 (min {stab['flow_r_min'].min():.2f}). The flow matrix does not reproduce "
            f"across seeds for those arms, so `recovery.csv` compares noise against noise and the "
            f"per-pair thresholds below are not stable. `z` is additionally NOT comparable across "
            f"arms: null_std shrinks with the window count, which spans "
            f"{stab['n_windows'].min()}-{stab['n_windows'].max()}."
        )

    print("\n--- interactions by detection threshold ---")
    print(det.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # Counted on the upper triangle only. The net flow is antisymmetric, so every off-diagonal
    # interaction appears twice in `det` with opposite signs, and the 9 diagonal entries are
    # structurally zero -- reporting "162 of 81 pairs" style numbers off the raw rows would just
    # be double counting.
    undirected = det[det["sender"] < det["receiver"]]
    n_pairs = len(undirected)
    never = undirected["detected_at"].isna().sum()
    print(f"\n{n_pairs} independent cell-type interactions (upper triangle of the antisymmetric "
          f"net-flow matrix; the 9 diagonal entries are structurally zero)")
    print(f"  never significant at any window size: {never}")
    for size in sorted(undirected["detected_at"].dropna().unique()):
        n = int((undirected["detected_at"] == size).sum())
        print(f"  first detected at {int(size):5d} px: {n:3d} interactions")
    print(f"\ndone. tables and figures in {OUT}")


if __name__ == "__main__":
    main()
