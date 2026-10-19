"""Extract and store the attention masks of the sliding-window ablation sweep.

Six sliding-window sizes x three seeds, trained as one wandb grid sweep (arms in
`interscale/config_files/sweeps/sliding_window_melton25.yaml`). This script turns each trained
model into the data products `5b_sliding_window_ablation.py` and the notebook consume:

  attn/<arm>_seed<seed>.h5      raw per-window attention, float16, one dataset per window
  flows/<arm>_seed<seed>.npz    per-window attention SUMS binned by (sender type, receiver type,
                                sender-receiver distance)
  windows/<arm>.npz             the matching per-window pair COUNTS and window metadata. Model
                                independent -- it depends only on the windowing -- so it is
                                written once per arm rather than once per (arm, seed).
  reference_pair_distances.npz  every within-FOV cell pair of the analysis slide, binned the same
                                way. The denominator for "what fraction of the possible
                                interactions at distance d does this windowing even expose?"
  sweep_runs.csv                the sweep's runs and their metrics, as pulled from the wandb API

    python scripts/melton_jimenez_25/5a_sliding_window_attention.py            # all arms, all seeds
    python scripts/melton_jimenez_25/5a_sliding_window_attention.py --arms w400 w3000 --seeds 42
    python scripts/melton_jimenez_25/5a_sliding_window_attention.py --skip-existing

--- why one slide -------------------------------------------------------------------------------

Everything is extracted on Run5211_S3 only. `split` on this dataset is assigned per slide
(train=S1, val=S2, test=S3), so S3 is the held-out test slide for every arm -- the ablation is
about what the models generalise to seeing, not what they memorised. It carries both conditions
(ND 51,365 cells / T1D 100,230), and squidpy's windows are built within `slide_fov`, so no window
straddles the slide boundary and subsetting to it leaves every window intact.

It is also the only tractable choice. `get_model_output` allocates one
n_cells x max_seq_len frame and does a full-frame masked assignment per window; on the full
386,727 cells the w400 arm would do 9,604 such assignments against a 386,727-row frame.

--- a confound worth stating -------------------------------------------------------------------

`prepare_a2d_dataset` overwrites `spatial_neigbors_kwargs.library_key` with the window column, so
the LOCAL graph is also cut at window boundaries: a 40-cell w400 window loses a far larger
fraction of its radius-110 neighbours to the boundary than a 1362-cell w3000 window does. The
local component therefore does not see a constant neighbourhood across arms. This is inherent to
the design rather than introduced here -- training builds its graphs the same way -- but it means
a difference between arms is a difference in "what the windowing exposes", local and global
together, not in the transformer's range alone.
"""

from __future__ import annotations

import argparse
import gc
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")

REPO = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility")
INTERSCALE = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale")
REGISTRY = INTERSCALE / "config_files" / "registry.yaml"
SWEEP_YAML = INTERSCALE / "config_files" / "sweeps" / "sliding_window_melton25.yaml"

RESULTS = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results")
CKPT_DIR = RESULTS / "melton25_revision" / "sliding_window"
OUT = RESULTS / "melton25_revision" / "sliding_window" / "attention"

# The sweep this script reads back. Registered by
#   python src/interscale/main_sweep.py --dataset melton25_sw --task node_reg \
#       --model_type CombinedModel --sweep_goal robustness \
#       --sweep_cfg config_files/sweeps/sliding_window_melton25.yaml \
#       --metric val_concordance_corr --create_only
SWEEP_ID = "z63vmk20"
WANDB_ENTITY = "francesca-drummer"
WANDB_PROJECT = "InterScale_hyperparameter_sweep"
SWEEP_URL = f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{SWEEP_ID}"

ANALYSIS_SLIDE = "Run5211_S3"
CELL_TYPE_COL = "cell_type_coarse"
SLIDE_COL = "slide"
SAMPLE_COL = "slide_fov"  # NOT `fov`: fov ids repeat across the three slides
CONDITION_COL = "condition"

ARMS = ["w400", "w800", "w1200", "w1600", "w2000", "w3000"]
SEEDS = [42, 43, 44]

# Sender-receiver distance bins, in FOV-LOCAL pixels -- `obsm['spatial']` on this dataset is the
# per-FOV coordinate (0-5461 x 0-3642 px, one CosMx FOV), which is also what squidpy's
# sliding_window and the radius-110 neighbour graph were built on. `obsm['spatial_fov']` is the
# global one and must not be used here: windows are within-FOV, so a global coordinate would put
# cells from different FOVs at meaningful-looking distances.
#
# 110 is an edge on purpose -- it is the radius of the local graph, so the first four bins are
# "what the GCN already sees" and everything above is what the transformer has to supply.
# 6600 closes the last bin above the FOV diagonal, sqrt(5439^2 + 3616^2) ~ 6532 px.
DIST_EDGES = np.array(
    [0, 20, 40, 60, 80, 110, 150, 200, 300, 400, 600, 800, 1200, 1600, 2000, 3000, 4000, 6600],
    dtype=np.float32,
)
N_BINS = len(DIST_EDGES) - 1

# CosMx pixel pitch, for reporting distances in microns alongside pixels. Cross-checked against
# the data rather than taken on trust: at 0.12 um/px a w3000 window is 360 um across and holds
# 1362 cells, i.e. ~0.0105 cells/um^2 or ~9.8 um per cell, which is the right order for pancreas.
PIXEL_TO_UM = 0.12


# --------------------------------------------------------------------------------------------
# sweep bookkeeping
# --------------------------------------------------------------------------------------------


def fetch_sweep_runs(sweep_id: str = SWEEP_ID) -> pd.DataFrame:
    """Pull the sweep's runs and the values each trial resolved to.

    `arm` and `optim.seed` come from the run's own config, and the `resolved_*` fields from the
    summary keys `main_sweep.py` writes after applying the trial. Reading both lets the extraction
    below assert that the config it rebuilds is the config that was trained, rather than assuming
    the sweep yaml has not moved since.
    """
    import wandb

    api = wandb.Api()
    sweep = api.sweep(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{sweep_id}")

    rows = []
    for run in sweep.runs:
        cfg = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = dict(run.summary)
        rows.append(
            {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "arm": cfg.get("arm"),
                "seed": cfg.get("optim.seed"),
                "checkpoint_prefix": summary.get("checkpoint_prefix"),
                "resolved_dataset_name": summary.get("resolved_dataset_name"),
                "resolved_sample_key": summary.get("resolved_sample_key"),
                "resolved_max_seq_len": summary.get("resolved_max_seq_len"),
                "val_concordance_corr": summary.get("val_concordance_corr"),
                "val_r2": summary.get("val_r2"),
                "val_pearson_corr": summary.get("val_pearson_corr"),
                "val_mse": summary.get("val_mse"),
                "val_loss": summary.get("val_loss"),
                "test_concordance_corr": summary.get("test_concordance_corr"),
                "test_r2": summary.get("test_r2"),
                "test_pearson_corr": summary.get("test_pearson_corr"),
                "test_mse": summary.get("test_mse"),
                "epoch": summary.get("epoch"),
                "total_parameters": summary.get("total_parameters"),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"sweep {sweep_id} has no runs yet")
    # Sorted by the nominal window size the arm names encode, so every table and plot downstream
    # is ordered by the ablation axis rather than alphabetically (w1200 before w400).
    df["window_size"] = df["arm"].map(arm_window_size)
    return df.sort_values(["window_size", "seed"]).reset_index(drop=True)


def arm_window_size(arm: str | None) -> float:
    """`w1600` -> 1600. Used for ordering and for the x axis of every ablation plot."""
    if not isinstance(arm, str) or not arm.startswith("w"):
        return np.nan
    return float(arm[1:])


def build_cfg(arm: str, seed: int):
    """Rebuild the exact config the sweep trained for one (arm, seed).

    Re-resolved from the registry and the sweep yaml rather than reconstructed from the wandb
    config blob, so the checkpoint path is derived the same way `get_model_filename_prefix`
    derived it at training time. `verify_cfg_against_run` then checks the two agree.
    """
    import yaml
    from interscale.config.registry import resolve_config
    from interscale.config.sweep import ARM_PARAM, apply_sweep_config, build_sweep_config, load_arms

    with SWEEP_YAML.open() as f:
        yaml_config = yaml.safe_load(f)
    sweep_config, sweep_params = build_sweep_config(
        yaml_config, prediction_task="regression", model_type="CombinedModel"
    )
    arms = load_arms(yaml_config, sweep_config)

    # Every non-arm parameter of this sweep is single-valued except optim.seed, so taking the first
    # declared value reproduces the trial exactly. Asserted rather than assumed: a later edit that
    # adds a second value to one of them would otherwise silently rebuild the wrong config.
    for name, spec in sweep_config["parameters"].items():
        if name in (ARM_PARAM, "optim.seed"):
            continue
        assert len(spec["values"]) == 1, (
            f"sweep parameter {name} has {len(spec['values'])} values; this script cannot infer "
            f"which one a given run used. Read it from the run's wandb config instead."
        )

    trial = {k: v["values"][0] for k, v in sweep_config["parameters"].items()}
    trial[ARM_PARAM] = arm
    trial["optim.seed"] = int(seed)

    cfg = resolve_config("melton25_sw", "node_reg", registry_path=str(REGISTRY))
    cfg, _ = apply_sweep_config(
        cfg, "robustness", trial, model_type="CombinedModel", sweep_params=sweep_params, arms=arms
    )
    cfg.defrost()
    cfg.wandb.use = False  # loading a checkpoint must not open a wandb run
    cfg.model.save = str(CKPT_DIR) + "/"
    cfg.freeze()
    return cfg


def verify_cfg_against_run(cfg, run_row, local_component=True, global_component=True):
    """Fail loudly if the rebuilt config is not the one the run trained."""
    from interscale.tl.utils import get_model_filename_prefix

    prefix = get_model_filename_prefix(cfg, local_component, global_component)
    recorded = run_row.get("checkpoint_prefix")
    if isinstance(recorded, str) and recorded and recorded != prefix:
        raise RuntimeError(
            f"rebuilt config does not match run {run_row['run_id']}: checkpoint prefix "
            f"{prefix!r} vs the run's {recorded!r}. The sweep yaml or the registry has changed "
            f"since the sweep ran; read the config from wandb instead of rebuilding it."
        )
    recorded_seq = run_row.get("resolved_max_seq_len")
    if recorded_seq is not None and not pd.isna(recorded_seq):
        assert int(recorded_seq) == int(cfg.model.global_component.parameters.max_seq_len), (
            f"max_seq_len mismatch for {run_row['run_id']}: rebuilt "
            f"{cfg.model.global_component.parameters.max_seq_len} vs run {int(recorded_seq)}"
        )
    return prefix


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------


def load_analysis_adata(cfg, slide: str = ANALYSIS_SLIDE):
    """The analysis slide, filtered exactly as training filtered.

    `remove_zero_expression_cells` runs BEFORE the slide subset, so the cell set matches what the
    model was trained and validated on rather than being re-derived on a subset.

    Cached to disk: the result is identical for all 18 (arm, seed) pairs -- the arms differ only in
    which obs column they window by, and every window column is already in the file -- so
    re-deriving it would mean re-reading a 900 MB h5ad and re-summing 387k x 979 sparse rows
    eighteen times.
    """
    from interscale.tl import remove_zero_expression_cells

    win = cfg.dataset.sample_key[0]
    cache = OUT / f"_slide_{slide}.h5ad"

    if cache.exists():
        sub = sc.read_h5ad(cache)
    else:
        adata = sc.read_h5ad(cfg.dataset.h5ad_data)
        adata = remove_zero_expression_cells(adata)
        sub = adata[adata.obs[SLIDE_COL].astype(str) == slide].copy()
        del adata
        gc.collect()
        cache.parent.mkdir(parents=True, exist_ok=True)
        sub.write_h5ad(cache)
        print(f"cached analysis slide to {cache} ({cache.stat().st_size / 1e6:.0f} MB)")

    # Per arm, not cached: each arm windows by a different column, and the categories left over
    # from the other two slides would otherwise make ToCategoryIterator yield empty graphs.
    sub.obs[win] = sub.obs[win].astype("category").cat.remove_unused_categories()
    print(
        f"analysis slide {slide}: {sub.n_obs} cells, {sub.obs[win].nunique()} windows of {win}, "
        f"{sub.obs[SAMPLE_COL].nunique()} FOVs, conditions {sorted(sub.obs[CONDITION_COL].unique())}"
    )
    return sub


def load_model(cfg, adata):
    """Load the checkpoint this (arm, seed) wrote and mark it usable for inference.

    `split_key=None` skips registering the split field. It has to be skipped: `_setup_anndata`
    asserts the split column contains both `train` and `val`, and this AnnData is one held-out
    slide, so it is entirely `test`. Nothing on the inference path reads the registered field --
    only `prepare_geome_dataset` uses the split, it reads `cfg.dataset.split_key` straight off
    `obs`, and `get_model_output` goes through `prepare_a2d_dataset`, which does not split at all.
    """
    import interscale

    interscale.model.CombinedModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        split_key=None,
        view_registry=False,
    )
    model = interscale.model.CombinedModel.load(
        str(CKPT_DIR), adata, cfg, local_component=True, global_component=True
    )
    return model


# --------------------------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------------------------


def bin_window(attn, type_codes, coords, n_types, edges=DIST_EDGES):
    """Bin one window's attention by (sender type, receiver type, sender-receiver distance).

    Parameters
    ----------
    attn : (n, n) float array
        The window's attention block. Row i, column j is attention from cell i to cell j, the same
        orientation `compute_hierarchical_net_flow` reads out of `obsm['_attn_matrix']`. The
        diagonal is kept: that function's per-cell-type mean includes it, and dropping it here
        would make these sums disagree with it for the s == r entries.
    type_codes : (n,) int array
        Cell type as an index into the shared category order, so every arm and window uses the
        same axes.
    coords : (n, 2) float array
        FOV-local pixel coordinates.

    Returns
    -------
    sums, counts : (n_types, n_types, n_bins) arrays
        Summed attention, and the number of pairs contributing to each entry. The mean is
        sums/counts; keeping them apart is what lets windows be pooled without weighting a
        1-cell window like a 3436-cell one.

    Both are produced by a single bincount over a composite index rather than by nested loops.
    The nested-loop form in `compute_hierarchical_net_flow` costs one pandas `.loc` per
    (window, sender type, receiver type) -- 9604 x 81 calls for the w400 arm, on top of one
    AnnData subset per window.
    """
    n = attn.shape[0]
    # Squared distances, then one sqrt: (x-x')^2 + (y-y')^2 via broadcasting.
    d = np.sqrt(
        (coords[:, 0, None] - coords[None, :, 0]) ** 2 + (coords[:, 1, None] - coords[None, :, 1]) ** 2
    )
    # right=False, minus 1: edges[0] is 0 and a self-distance of exactly 0 must land in bin 0.
    # Distances at or beyond the last edge are clipped into the last bin rather than dropped, so
    # `counts` always sums to n^2 and the coverage denominator stays exact.
    b = np.clip(np.digitize(d, edges, right=False) - 1, 0, len(edges) - 2).astype(np.int64)

    pair = (type_codes[:, None].astype(np.int64) * n_types + type_codes[None, :].astype(np.int64)) * N_BINS
    code = (pair + b).ravel()
    size = n_types * n_types * N_BINS

    a = np.nan_to_num(attn, nan=0.0).astype(np.float64).ravel()
    sums = np.bincount(code, weights=a, minlength=size).reshape(n_types, n_types, N_BINS)
    counts = np.bincount(code, minlength=size).reshape(n_types, n_types, N_BINS)
    assert counts.sum() == n * n, f"binning lost pairs: {counts.sum()} != {n * n}"
    return sums.astype(np.float64), counts.astype(np.int64)


def reference_pair_distances(adata, n_types, out_path, chunk=2048):
    """Every within-FOV cell pair of the analysis slide, binned the same way.

    This is the denominator of the coverage question. The universe of interactions a windowing
    could possibly expose is bounded by the FOV: squidpy builds windows within `slide_fov`, so a
    pair in two different FOVs can never share a window at any window size. Pairs are counted
    per FOV and summed, never across FOVs.

    Windows are a partition -- for w400, 9604 windows of mean 40 cells sum to 386,727 cells, i.e.
    every cell is in exactly one window -- so a pair co-occurs in at most one window and
    coverage = (pairs sharing a window) / (pairs in the same FOV) needs no de-duplication.
    """
    codes = adata.obs[CELL_TYPE_COL].cat.codes.to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    fovs = adata.obs[SAMPLE_COL].astype(str).to_numpy()

    total = np.zeros((n_types, n_types, N_BINS), dtype=np.int64)
    for fov in np.unique(fovs):
        sel = np.flatnonzero(fovs == fov)
        c, xy = codes[sel], coords[sel]
        n = len(sel)
        # Chunked over rows: a 7627-cell FOV is 5.8e7 pairs, and materialising every FOV's full
        # distance matrix at once would be 233 MB per FOV for no reason.
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            d = np.sqrt(
                (xy[start:stop, 0, None] - xy[None, :, 0]) ** 2 + (xy[start:stop, 1, None] - xy[None, :, 1]) ** 2
            )
            b = np.clip(np.digitize(d, DIST_EDGES, right=False) - 1, 0, N_BINS - 1).astype(np.int64)
            pair = (c[start:stop, None].astype(np.int64) * n_types + c[None, :].astype(np.int64)) * N_BINS
            total += np.bincount(
                (pair + b).ravel(), minlength=n_types * n_types * N_BINS
            ).reshape(n_types, n_types, N_BINS)
        print(f"  reference pairs {fov}: {n} cells", flush=True)

    np.savez_compressed(
        out_path,
        counts=total,
        edges=DIST_EDGES,
        cell_types=np.array(list(adata.obs[CELL_TYPE_COL].cat.categories), dtype=object),
        pixel_to_um=PIXEL_TO_UM,
        slide=ANALYSIS_SLIDE,
    )
    print(f"wrote {out_path}  ({total.sum():,} within-FOV pairs)")
    return total


# --------------------------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------------------------


def extract_one(arm, seed, run_row, cell_type_categories, store_attention=True):
    """Run one (arm, seed) model over the analysis slide and write its attention products."""
    cfg = build_cfg(arm, seed)
    prefix = verify_cfg_against_run(cfg, run_row)
    win = cfg.dataset.sample_key[0]
    max_seq_len = int(cfg.model.global_component.parameters.max_seq_len)
    print(f"\n===== {arm} seed {seed} | {win} | max_seq_len {max_seq_len} | {prefix} =====", flush=True)

    adata = load_analysis_adata(cfg)
    # One shared category order across every arm and seed, so the (n_types, n_types, n_bins)
    # arrays from different arms can be stacked without re-aligning axes.
    adata.obs[CELL_TYPE_COL] = pd.Categorical(
        adata.obs[CELL_TYPE_COL].astype(str), categories=list(cell_type_categories)
    )
    n_types = len(cell_type_categories)

    model = load_model(cfg, adata)
    from interscale.tl import set_full_reproducibility

    set_full_reproducibility(int(cfg.optim.seed))
    adata = model.get_model_output(adata, prefix="")

    attn = adata.obsm["_attn_matrix"]
    assert attn.shape[1] == max_seq_len, f"attention is {attn.shape[1]} wide, expected {max_seq_len}"

    codes = adata.obs[CELL_TYPE_COL].cat.codes.to_numpy()
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)[:, :2]
    window_labels = adata.obs[win].astype(str).to_numpy()
    fov_labels = adata.obs[SAMPLE_COL].astype(str).to_numpy()
    cond_labels = adata.obs[CONDITION_COL].astype(str).to_numpy()

    # Group once. `adata[adata.obs[win] == w]` per window is what makes the reference
    # implementation unusable at 9604 windows.
    order = np.argsort(window_labels, kind="stable")
    sorted_labels = window_labels[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_labels[1:] != sorted_labels[:-1]])
    groups = np.split(order, boundaries[1:])
    names = sorted_labels[boundaries]

    (OUT / "attn").mkdir(parents=True, exist_ok=True)
    (OUT / "flows").mkdir(parents=True, exist_ok=True)
    (OUT / "windows").mkdir(parents=True, exist_ok=True)

    sums = np.zeros((len(groups), n_types, n_types, N_BINS), dtype=np.float32)
    counts = np.zeros((len(groups), n_types, n_types, N_BINS), dtype=np.int64)
    per_type = np.zeros((len(groups), n_types), dtype=np.int32)
    # The DIAGONAL of the relevance matrix, summed per cell type. Stored separately because it
    # dominates everything and carries almost no information.
    #
    # `_attn_matrix` is not raw attention: it is the Chefer et al. relevance matrix built by
    # `tl/self_attn_relevance.py::generate_relevance` as
    #     I = eye(n);  for each layer:  I += (grad * attn, clamped, head-averaged) @ I
    # so the diagonal starts at exactly 1 and the attention contributions accumulate on top. That
    # identity term is why the diagonal measures ~0.99 of the total mass (row sums come out at
    # ~1.01) -- it is a construction constant, NOT the model choosing to attend to each cell
    # itself. The interaction signal is the ~1% off the diagonal, and any distance profile that
    # leaves the diagonal in is a plot of `eye(n)`.
    #
    # It is subtracted downstream rather than dropped here, because the flow definition in
    # `compute_hierarchical_net_flow` INCLUDES the diagonal in agg[s, s] and the published
    # numbers have to stay reproducible. (It cancels in the net flow anyway: net = agg.T - agg
    # is zero on the diagonal.) With `per_type` giving the matching counts, `sums`/`counts` can
    # be corrected exactly for the distance analysis and left alone for the flows.
    diag_sums = np.zeros((len(groups), n_types), dtype=np.float64)
    meta = []

    attn_path = OUT / "attn" / f"{arm}_seed{seed}.h5"
    h5 = h5py.File(attn_path, "w") if store_attention else None
    try:
        if h5 is not None:
            h5.attrs.update(
                arm=arm, seed=int(seed), window_key=win, max_seq_len=max_seq_len,
                slide=ANALYSIS_SLIDE, sweep_id=SWEEP_ID, checkpoint_prefix=prefix,
            )
            grp = h5.create_group("windows")

        for wi, (name, idx) in enumerate(zip(names, groups, strict=True)):
            n = len(idx)
            assert n <= max_seq_len, (
                f"window {name} has {n} cells but max_seq_len is {max_seq_len}; the stored "
                f"attention is narrower than the window and every downstream flow is truncated"
            )
            block = np.asarray(attn[idx][:, :n], dtype=np.float32)
            s, c = bin_window(block, codes[idx], coords[idx], n_types)
            sums[wi] = s
            counts[wi] = c
            per_type[wi] = np.bincount(codes[idx], minlength=n_types)
            diag_sums[wi] = np.bincount(
                codes[idx], weights=np.nan_to_num(np.diagonal(block), nan=0.0), minlength=n_types
            )
            meta.append(
                {
                    "window": name,
                    "n_cells": n,
                    "slide_fov": fov_labels[idx[0]],
                    "condition": cond_labels[idx[0]],
                    "n_nonfinite": int((~np.isfinite(block)).sum()),
                }
            )
            if h5 is not None:
                # One group per window. `attn` is float16: that halves the store at a precision
                # far finer than anything downstream resolves -- attention weights are O(1e-3)
                # here and every use of them is a mean over thousands of pairs.
                #
                # Labels and coordinates travel WITH the block so the permutation test can relabel
                # a window without reopening the slide AnnData and re-deriving a mapping from
                # cell_index. That mapping is the one step where an off-by-one would silently
                # produce a plausible null distribution for the wrong cells.
                #
                # Datasets rather than HDF5 attributes: attributes are capped at 64 KB, and a
                # 3436-cell w3000 window's cell_index + type_codes + coords come to ~45 KB, close
                # enough that a wider window later would start failing.
                g = grp.create_group(name)
                g.create_dataset("attn", data=block.astype(np.float16), compression="gzip", compression_opts=4)
                g.create_dataset("cell_index", data=idx.astype(np.int32))
                g.create_dataset("type_codes", data=codes[idx].astype(np.int8))
                g.create_dataset("coords", data=coords[idx].astype(np.float32))
                g.attrs["slide_fov"] = str(fov_labels[idx[0]])
                g.attrs["condition"] = str(cond_labels[idx[0]])
                g.attrs["n_cells"] = int(n)
            if wi % 500 == 0:
                print(f"  window {wi}/{len(groups)}", flush=True)
    finally:
        if h5 is not None:
            h5.close()

    meta_df = pd.DataFrame(meta)
    np.savez_compressed(
        OUT / "flows" / f"{arm}_seed{seed}.npz",
        sums=sums,
        diag_sums=diag_sums,
        window=meta_df["window"].to_numpy().astype(object),
        arm=arm,
        seed=int(seed),
        edges=DIST_EDGES,
        cell_types=np.array(list(cell_type_categories), dtype=object),
    )
    # Written once per arm: counts and metadata depend on the windowing, not on the model, so the
    # three seeds of an arm would write three identical copies.
    win_path = OUT / "windows" / f"{arm}.npz"
    if not win_path.exists():
        np.savez_compressed(
            win_path,
            counts=counts,
            per_type=per_type,
            window=meta_df["window"].to_numpy().astype(object),
            n_cells=meta_df["n_cells"].to_numpy(),
            slide_fov=meta_df["slide_fov"].to_numpy().astype(object),
            condition=meta_df["condition"].to_numpy().astype(object),
            window_key=win,
            window_size=arm_window_size(arm),
            edges=DIST_EDGES,
            cell_types=np.array(list(cell_type_categories), dtype=object),
            pixel_to_um=PIXEL_TO_UM,
        )
        print(f"wrote {win_path}")

    nonfinite = int(meta_df["n_nonfinite"].sum())
    print(
        f"{arm} seed {seed}: {len(groups)} windows, {meta_df['n_cells'].sum()} cells, "
        f"attention {attn_path.stat().st_size / 1e6:.0f} MB"
        + (f"  WARNING {nonfinite} non-finite attention entries" if nonfinite else "")
    )

    del model, adata, attn, sums, counts
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return meta_df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--skip-existing", action="store_true", help="Leave (arm, seed) pairs already written.")
    ap.add_argument("--no-attention", action="store_true", help="Write only the binned products.")
    ap.add_argument("--reference-only", action="store_true", help="Only rebuild reference_pair_distances.npz.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    runs = fetch_sweep_runs()
    runs.to_csv(OUT / "sweep_runs.csv", index=False)
    print(f"sweep {SWEEP_ID}: {len(runs)} runs ({SWEEP_URL})")
    print(runs[["run_name", "state", "arm", "seed", "val_concordance_corr", "val_r2"]].to_string(index=False))

    finished = runs[runs["state"] == "finished"]
    if finished.empty:
        raise SystemExit("no finished runs in the sweep yet; nothing to extract")

    # The category order is fixed from the data once, not per arm, so every stored array shares
    # its axes. Taken from the first arm's config since they all read the same h5ad.
    cfg0 = build_cfg(args.arms[0], args.seeds[0])
    adata0 = load_analysis_adata(cfg0)
    cell_type_categories = sorted(adata0.obs[CELL_TYPE_COL].astype(str).unique())
    print(f"cell types ({len(cell_type_categories)}): {cell_type_categories}")

    ref_path = OUT / "reference_pair_distances.npz"
    if args.reference_only or not ref_path.exists():
        adata0.obs[CELL_TYPE_COL] = pd.Categorical(
            adata0.obs[CELL_TYPE_COL].astype(str), categories=cell_type_categories
        )
        reference_pair_distances(adata0, len(cell_type_categories), ref_path)
        if args.reference_only:
            return
    del adata0
    gc.collect()

    for arm in args.arms:
        for seed in args.seeds:
            row = finished[(finished["arm"] == arm) & (finished["seed"] == seed)]
            if row.empty:
                print(f"skipping {arm} seed {seed}: no finished run in the sweep")
                continue
            if args.skip_existing and (OUT / "flows" / f"{arm}_seed{seed}.npz").exists():
                print(f"skipping {arm} seed {seed}: already extracted")
                continue
            extract_one(
                arm,
                seed,
                row.iloc[0].to_dict(),
                cell_type_categories,
                store_attention=not args.no_attention,
            )

    print(f"\ndone. products in {OUT}")


if __name__ == "__main__":
    main()
