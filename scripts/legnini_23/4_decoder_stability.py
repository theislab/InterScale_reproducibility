# %% [markdown]
# # Ablation study for sliding windows
#
# This tutorials shows how to set up an multiple slide molecular cartography dataset used from [Legnini et al., 2023](https://doi.org/10.1038/s41592-023-01986-w). To follow along with this and the following tutorials, please execute the following steps first:
#
# - Set up InterScale environment (see instructions in ReadMe)
# - Download the sample data from the original publication from [Zenodo](https://doi.org/10.5281/zenodo.6143560) under the accession no. 6143561

# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import seaborn as sns
from scipy.sparse import csr_array, csr_matrix, issparse
import interscale

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

# %%
import os

# LRZ home
if os.path.exists("/dss/dsshome1/05/di93tig"):
    print('LRZ cluster')
    CLUSTER = 'LRZ'
    BASE_DIR_REPO = "/dss/dsshome1/05/di93tig/1_projects" 
    BASE_DIR_PROJECT = "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale"
elif os.path.exists("/home/icb/francesca.drummer/"):
    print('HPC cluster')
    CLUSTER = 'HPC'
    BASE_DIR_REPO = "/home/icb/francesca.drummer/1-Projects/A3_InterScale/InterScale_reproducibility/"
    BASE_DIR_PROJECT = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/"
else:
    print('unkown')
    CLUSTER = 'unknown'
DATA = "legnini23"

# %%
import yaml
with open(os.path.join(BASE_DIR_REPO, "figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
#CELL_TYPE_COLORS = config["palettes"][DATA]

slides = config["slide_examples"]["legnini23"]

# %% [markdown]
# <mark>TODO: Change data path</mark>

# %%
LRZ_LEGNINI23 = f'{BASE_DIR_PROJECT}/data/{DATA}.h5ad'  

# %% [markdown]
# ## Load data

# %%
adata = sc.read_h5ad(LRZ_LEGNINI23)
adata

# %%
print('Zero count cells: ', (adata.X.sum(1) ==0).sum())

# %%
# only needed when re-running on an adata that already carries the zones
adata.obs.drop(columns=["SHH_zone"], errors="ignore", inplace=True)

# %%
from interscale.tl import clustering

# The last edge is open-ended on purpose. create_spatial_cluster_distance bins
# dist_to_SHH with pd.cut, so any cell beyond the last edge becomes NaN in
# SHH_zone. With a closed edge at 6000 that silently dropped 685 cells of
# slide1_C2-2 (dist_to_SHH 6000-8034, the only slide reaching that far), and the
# NaN propagated into sliding_window_low -> scvi's CategoricalObsField assigns
# code -1 and register_fields dies with
#   ValueError: Making .obs["sliding_window_low"] categorical failed.
# np.inf puts that tail into 'far', where it belongs, and keeps 4 rings.
ring_edges  = [0, 200, 2000, 3000, np.inf]
ring_labels = ['SHH', 'near', 'medium', 'far']

clustering.create_spatial_cluster_distance(
    adata, 'SHH', 'sample',
    distance_bins=ring_edges,
    bin_labels=ring_labels,
    threshold=95,
)

# %%
sq.pl.spatial_scatter(
    adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])],
    spatial_key = 'spatial',
    library_key='sample', 
    #library_id = [839, 6],
    color="SHH_zone", #cell_type_coarse",
    shape= None,
    size = 10,
)

# %%
sq.pl.spatial_scatter(
    adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])],
    spatial_key = 'spatial',
    library_key='sample', 
    #library_id = [839, 6],
    color="dist_to_SHH", #cell_type_coarse",
    shape= None,
    size = 10,
)

# %% [markdown]
# Check that no NaN's exists in the data.

# %%
# dropna=False: a plain groupby().size() hides exactly the NaNs this checks for
print(adata.obs["SHH_zone"].value_counts(dropna=False))
assert adata.obs["SHH_zone"].isna().sum() == 0, (
    "cells outside every ring - widen ring_edges (the last edge must cover max(dist_to_SHH))"
)

# %% [markdown]
# ## Load trained models

# %%
import sys

project_root = Path(BASE_DIR_REPO)  # InterScale_reproducibility, for `from src.wandb import ...`
sys.path.insert(0, str(project_root))

# The interscale package is a SIBLING of InterScale_reproducibility, not a subdirectory of it, so
# `Path(BASE_DIR_REPO) / "interscale"` points at nothing here. It only resolved in
# 3_sliding_window_ablation.ipynb, where BASE_DIR_REPO is the A3_InterScale root.
INTERSCALE_REPO = project_root.parent / "interscale"
REGISTRY = INTERSCALE_REPO / "config_files" / "registry.yaml"

DATASET = "legnini23"

# ---------------------------------------------------------------------------------------------
# WHICH MASKING ARM TO LOAD.  Flip this and re-run from here; nothing below is arm-specific.
#
#   "cell"  whole cells are blanked and all G genes of them are scored  (mask_strategy "node")
#   "gene"  individual (cell, gene) entries are blanked and only those are scored
#
# The two sweeps below are matched: same five seeds, same 300 epochs, same architecture, and both
# trained with MASK_VALUE = -1. They differ only in masking granularity and its rate, so a
# difference in how stable the gene loadings are across seeds is attributable to that.
#
# The rate is not the same number in the two arms because it is not the same quantity: 0.30 of
# the CELLS vs 0.25 of every cell's GENES. 0.25 is where gene masking measurably wins
# (val_concordance_corr 0.1947 +/- 0.0041 against cell 0.30's 0.1634 +/- 0.0084, 3 seeds).
#
# NOTE these are NOT the published seed_stability/ models. Those were trained at the old 100-epoch
# default with MASK_VALUE = 0 and are left untouched in `.../legnini23_revision/seed_stability/`.
# 300 epochs because at 100 no trial on this dataset ever early-stops (optim.min_delta is 0.0, so
# only the n_epochs cap stops a run) and 100 -> 300 moves val_concordance_corr by +200% to +286%;
# seed spread is 2-18% at 100 and 1-4% at 300, so a stability analysis at 100 would largely be
# measuring undertraining.
# ---------------------------------------------------------------------------------------------
MASKING = "gene"  # "cell" | "gene"

ARMS = {
    "cell": {
        "task": "node_reg",
        "sweep_id": "ze40fzaz",  # config_files/sweeps/seed_cell_masking.yaml
        "models_subdir": "seed_stability_cell",
        # Nothing to override: node_reg's registry defaults are what the sweep trained with.
        "overrides": {},
    },
    "gene": {
        "task": "node_reg_genemask",
        "sweep_id": "dty3dlk3",  # config_files/sweeps/seed_gene_masking.yaml
        "models_subdir": "seed_stability_gene",
        # The sweep pinned the rate below the task file's 0.5; build_cfg has to reapply it or the
        # config recorded here is not the one that trained. (It changes no tensor shape, so
        # loading would succeed either way -- which is exactly why it would go unnoticed.)
        "overrides": {"dataset.mask_percentage": 0.25},
    },
}
assert MASKING in ARMS, f"MASKING must be one of {sorted(ARMS)}"
ARM = ARMS[MASKING]
TASK, SWEEP_ID = ARM["task"], ARM["sweep_id"]

MODELS_DIR = Path(BASE_DIR_PROJECT) / "results" / "legnini23_revision" / ARM["models_subdir"]
# Per-arm, so the two arms' per-gene CSVs can sit side by side instead of overwriting each other.
RESULTS_DIR = Path(BASE_DIR_PROJECT) / "results" / DATA / f"rec_seed_stablity_{MASKING}"

assert REGISTRY.exists(), f"registry not found: {REGISTRY}"
assert MODELS_DIR.exists(), (
    f"checkpoint dir not found: {MODELS_DIR}\n"
    f"Train it with:  SWEEP_ID={SWEEP_ID} sbatch scripts/run.sh {DATASET} {TASK} "
    f"--sweep_cfg config_files/sweeps/seed_{MASKING}_masking.yaml --sweep_goal robustness --count 5"
)
print(f"masking     {MASKING} (task {TASK}, sweep {SWEEP_ID})")
print(f"registry    {REGISTRY}\ncheckpoints {MODELS_DIR}\nresults     {RESULTS_DIR}")


# %%
import os
from pathlib import Path

import wandb

from interscale.config.registry import resolve_config
from interscale.evaluation import plot_global_directionality
from interscale.evaluation.net_streams import compute_hierarchical_net_flow
from interscale.tl.utils import set_full_reproducibility
from src.wandb import Wandb_evaluation, plot_f1_across_seeds

wandb.login()

# %% [markdown]
# ## Load WandB sweep

# %% [markdown]
# Load sweep from WandB (5 seeds, all the same model parameters)

# %%
# NOT `wandb = ...`: that name is the wandb module, imported two cells up, and rebinding it is
# what made `wandb.Api(timeout=60)` raise AttributeError further down.
#
# SWEEP_ID comes from the MASKING switch above. The first argument is only a label --
# Wandb_evaluation hardcodes entity/project to francesca-drummer/InterScale_hyperparameter_sweep.
sweep_eval = Wandb_evaluation(
    "InterScale",
    SWEEP_ID,
    local_component=True,
    global_component=True,
    sweep_goal="robustness",
    classes=[None],
)


# %%
sweep_df = sweep_eval.get_dataframe()
sweep_df.head()


# %% [markdown]
# ## Model inference
#
# Load each of the five seed models, run inference, and keep
#
# - `local_emb[seed]` / `global_emb[seed]` &mdash; the two embeddings, `(n_cells, n_embed)`
# - `models[seed]` &mdash; the loaded `CombinedModel`, for the gene-loading section below
# - `performance[seed]` &mdash; per-gene reconstruction scores, also written to
#   `results/legnini23/rec_seed_stablity/model_{seed}_performance.csv`
#
# The models are loaded from disk rather than from wandb artifacts: the sweep pinned `model.save`,
# so all five checkpoints already sit in `legnini23_revision/seed_stability/`.
#
# **The seed cannot be read off `run.config['optim']['seed']`.** `main_sweep.py` calls
# `wandb.init(config=cfg)` *before* `apply_sweep_config`, so the nested config wandb stored is the
# base config &mdash; seed 44 for all five runs, which is also why every run is named `..._44_...`.
# Only the flattened `run.config['optim.seed']` (what `sweep_df["seed"]` reads) carries the trial's
# actual value, and it is what `build_cfg` applies below.

# %%
def build_cfg(seed: int):
    """Registry config for (DATASET, TASK) pinned to one trial of the selected seed sweep.

    `config_files/sweeps/seed_{cell,gene}_masking.yaml` declare `optim.seed` over [40..44], a
    fixed `optim.n_epochs` and `model.save`, and -- for the gene arm only -- a fixed
    `dataset.mask_percentage`. Reapplying those on top of the resolved pair reproduces the config
    each trial trained with; everything else (radius 200, n_embed 16, dual_decoder) comes from the
    registry, unchanged.

    `optim.seed` is not cosmetic here: `get_model_filename_prefix` puts it in the checkpoint
    name, which is what makes the five models separable when `load()` picks the file. The same
    prefix also carries `dataset.name`, which the registry sets to `legnini23_genemask` for the
    gene arm -- so the two arms' checkpoints are distinguishable even before the directory split.
    """
    cfg = resolve_config(DATASET, TASK, registry_path=str(REGISTRY))
    cfg.defrost()
    cfg.optim.seed = int(seed)
    cfg.optim.n_epochs = 300
    cfg.model.save = str(MODELS_DIR) + os.sep
    for dotted, value in ARM["overrides"].items():
        node, *rest = dotted.split(".")
        target = cfg[node]
        for part in rest[:-1]:
            target = target[part]
        target[rest[-1]] = value
    cfg.freeze()
    return cfg


def per_gene_metrics(y_true: np.ndarray, y_pred: np.ndarray, min_cells: int = 3) -> dict:
    """Per-gene reconstruction scores over the cells finite in both truth and prediction.

    `get_model_output` leaves a cell's prediction as NaN across *all* genes when it was never
    assigned to a graph, so the finite mask is taken per cell rather than per gene -- that keeps
    every gene scored on the same cells, which is what makes the rows comparable to each other.

    Beyond the four requested scores this returns offset-corrected R2 and the first two moments
    of truth and prediction, because raw R2 on this model is dominated by a systematic mean
    offset and by variance shrinkage: the raw number alone (and especially its mean over genes,
    which low-variance genes drive to -1e2) describes the miscalibration rather than the
    covariation structure the decoder loadings are read from.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    t, p = y_true[ok], y_pred[ok]
    n_cells, n_genes = t.shape
    if n_cells < min_cells:
        raise ValueError(f"only {n_cells} cells are finite in both truth and prediction")

    resid = t - p
    t_mu, p_mu = t.mean(axis=0), p.mean(axis=0)
    t_c, p_c = t - t_mu, p - p_mu
    ss_tot = (t_c**2).sum(axis=0)

    # A gene that is constant over these cells has ss_tot == 0 (R2 undefined) or zero predicted
    # sd (r undefined); errstate keeps those from printing, and the NaNs below are deliberate.
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - (resid**2).sum(axis=0) / ss_tot
        # Same R2 after removing the constant offset between the two means.
        r2_offset_corrected = 1.0 - ((t_c - p_c) ** 2).sum(axis=0) / ss_tot
        pearson = (t_c * p_c).sum(axis=0) / np.sqrt(ss_tot * (p_c**2).sum(axis=0))
        # Uncentered, i.e. over the raw expression vectors -- the centered cosine IS Pearson, so
        # centering here would just duplicate the previous column.
        cosine_sim = (t * p).sum(axis=0) / (np.linalg.norm(t, axis=0) * np.linalg.norm(p, axis=0))

    constant = ss_tot == 0
    r2[constant] = np.nan
    r2_offset_corrected[constant] = np.nan

    return {
        "n_cells": np.full(n_genes, n_cells),
        "r2": r2,
        "mse": (resid**2).mean(axis=0),
        "pearson": pearson,
        "cosine_sim": cosine_sim,
        "r2_offset_corrected": r2_offset_corrected,
        "true_mean": t_mu,
        "true_std": t.std(axis=0),
        "pred_mean": p_mu,
        "pred_std": p.std(axis=0),
    }


# %%
from interscale.tl import remove_zero_expression_cells

# Training ran on the zero-expression-filtered adata (43762 -> 43114 cells here). Inference has to
# filter identically, or the per-sample graphs handed to the model are not the graphs it was
# trained on.
#
# remove_zero_expression_cells reads adata.X, and a cell further down repoints adata.X at
# log1p_norm -- where no cell sums to zero, so re-running this cell after that one would silently
# keep all 43762. layers['raw'] is the same matrix X holds on load (identical nnz, identical 648
# zero rows), so restoring it first makes this cell order-independent.
adata.X = adata.layers["raw"]
adata = remove_zero_expression_cells(adata)

SEEDS = sorted(sweep_df["seed"].astype(int))
print("seeds:", SEEDS)

cfg0 = build_cfg(SEEDS[0])
window_key = cfg0.dataset.sample_key[0]
assert cfg0.model.decoder.dual_decoder, (
    "cfg has no local decoder, so there is nothing to compare against the global one"
)

# Below max_seq_len, tl/padding.py random-subsamples each graph every call instead of erroring,
# so the predictions would come from a graph the model never saw.
sizes = adata.obs[window_key].value_counts()
sizes = sizes[sizes > 0]
assert sizes.max() <= cfg0.model.global_component.parameters.max_seq_len, (
    f"largest {window_key} graph is {sizes.max()} cells but max_seq_len is "
    f"{cfg0.model.global_component.parameters.max_seq_len}"
)
print(f"{len(sizes)} graphs over '{window_key}', largest {sizes.max()} cells "
      f"(max_seq_len {cfg0.model.global_component.parameters.max_seq_len})")

# Ground truth is the layer the decoders were trained to reconstruct, NOT .X -- this h5ad keeps
# counts in X (max ~120) against log1p-scale predictions (max ~4.8).
_y_true = adata.layers[cfg0.dataset.layer_key]
y_true = np.asarray(_y_true.todense() if issparse(_y_true) else _y_true, dtype=np.float64)

DECODERS = {"local": "_y_pred_local", "global": "_y_pred_global"}
# 'all' first so the csv leads with the number over every cell; the three real splits follow so
# the test rows stay comparable to the sweep's `test_r2` / `test_pearson_corr` columns.
SPLITS = ["all", "train", "val", "test"]

# %%
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

models = {}
local_emb = {}
global_emb = {}
performance = {}

for seed in SEEDS:
    cfg = build_cfg(seed)

    interscale.model.CombinedModel._setup_anndata(
        adata=adata,
        prediction_task=cfg.dataset.prediction_task,
        layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key,
        prediction_obs=cfg.dataset.prediction_obs,
        view_registry=False,
    )
    set_full_reproducibility(40)

    # model_name is left at None so the filename comes from cfg via get_model_filename_prefix,
    # i.e. dual_legnini23_regr_node_<seed>_GCN_self-attn-transformer_model.pt. load() is a
    # classmethod that builds the model itself; note it uses strict=False, so read the
    # missing/unexpected-key warnings it prints -- a silent mismatch is a garbage model.
    models[seed] = interscale.model.CombinedModel.load(
        str(MODELS_DIR),
        adata,
        cfg,
        local_component=True,
        global_component=True,
    )

    # prefix="" writes obsm _local_emb/_global_emb and layers _y_pred_local/_y_pred_global.
    # Every seed writes the SAME keys, so everything needed is copied out before the next
    # iteration overwrites it.
    adata = models[seed].get_model_output(adata, prefix=f"{seed}")
    adata.obsm.pop("_attn_matrix", None)  # 43114 x 4318 float32 (~750 MB), unused here

    print(adata)

    local_emb[seed] = np.asarray(adata.obsm[f"{seed}_local_emb"], dtype=np.float32)
    global_emb[seed] = np.asarray(adata.obsm[f"{seed}_global_emb"], dtype=np.float32)

    DECODERS = {"local": f"{seed}_y_pred_local", "global": f"{seed}_y_pred_global"}
    y_pred = {d: np.asarray(adata.layers[layer], dtype=np.float64) for d, layer in DECODERS.items()}

    rows = []
    for split in SPLITS:
        cells = slice(None) if split == "all" else (adata.obs[cfg.dataset.split_key] == split).to_numpy()
        for decoder in DECODERS:
            rows.append(
                pd.DataFrame(
                    {
                        "gene": adata.var_names,
                        "decoder": decoder,
                        "split": split,
                        **per_gene_metrics(y_true[cells], y_pred[decoder][cells]),
                    }
                )
            )

    performance[seed] = pd.concat(rows, ignore_index=True)
    csv_path = RESULTS_DIR / f"model_{seed}_performance.csv"
    performance[seed].to_csv(csv_path, index=False)

    # Medians, not means: a handful of low-variance genes drive mean R2 to -1e2 and make it
    # useless as a sanity check.
    test_median = (
        performance[seed]
        .query("split == 'test'")
        .groupby("decoder")[["r2", "mse", "pearson", "cosine_sim"]]
        .median()
    )
    print(
        f"\nseed {seed}: wrote {csv_path.name} ({len(performance[seed])} rows)\n"
        f"  per-gene medians on the test split:\n{test_median.to_string()}"
    )

print(f"\nembeddings: local {local_emb[SEEDS[0]].shape}, global {global_emb[SEEDS[0]].shape} per seed")

# %% [markdown]
# ## Gene loadings

# %%
from interscale.evaluation import (
    gene_loadings,
    calculate_gene_ranks,
    latent_rank_report,
    get_genes_dim,
    calculate_dim_importance,
)

# %%
adata

# %%
print(SEEDS, models)

# %%
SEED = SEEDS[-2]

# %%
df_local_list = []
df_global_list = []

for SEED in SEEDS:
    set_full_reproducibility(40)
    gene_loadings(
        adata,
        models[SEED],
        local_latent_key=f"{SEED}_global_emb",
        global_latent_key=f"{SEED}_global_emb",
        local_varm_key=f"{SEED}_local_std_gene_loadings",
        global_varm_key=f"{SEED}_global_std_gene_loadings",
        layer_key="log1p_norm",
    )

    result_local = calculate_dim_importance(
        adata,
        which = "local", 
        prefix = SEED,
        mode="full",
        use_ratio=True,
        cumulative_cutoff=0.60,
        spacing=2,
    )
    result_global = calculate_dim_importance(
        adata,
        which = "global",
        prefix = SEED,
        mode="full",
        use_ratio=True,
        cumulative_cutoff=0.60,
        spacing=2,
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    interscale.pl.dim_importance_elbow(result_local, title="Local Embedding Dim Importance Elbow", ax=axes[0])
    interscale.pl.dim_importance_elbow(result_global, title="Global Embedding Dim Importance Elbow", ax=axes[1])

    df_global, ax = get_genes_dim(
        adata,
        which="global",
        prefix = SEED,
        n_top=10,
        X_layer="log1p_norm",
        min_frac=0.2,
        rank_by="loading_x_sd",
        enforce_specificity=True,
        specificity_min=1.5,
        # residualize=True,
        plot=True,
        figsize=(3.3, 5),
    )

    df_local, ax = get_genes_dim(
        adata,
        which="local",
        prefix = SEED,
        n_top=10,
        X_layer="log1p_norm",
        min_frac=0.2,
        rank_by="loading_x_sd",
        enforce_specificity=True,
        specificity_min=1.5,
        # residualize=True,
        plot=True,
        figsize=(3.3, 5),
    )
    df_local_list.append(df_local)
    df_global_list.append(df_global)

# %%
import itertools, pandas as pd

for df_list in [df_local_list, df_global_list]:
    names = [f"df{i}" for i in range(len(df_list))]
    idx = [set(df.index) for df in df_list]
    overlap = pd.DataFrame(0, index=names, columns=names)
    for (i, a), (j, b) in itertools.combinations(enumerate(idx), 2):
        overlap.iloc[i, j] = overlap.iloc[j, i] = len(a & b)
    for i, a in enumerate(idx):
        overlap.iloc[i, i] = len(a)
    
    print(overlap)

    common = reduce(lambda a, b: a & b, (set(df.index) for df in df_list))
    print(len(common), sorted(common)[:10])

# %%
local_emb_list = []
for emb in local_dim:
    adata.obs[f'local_emb_{emb}'] = adata.obsm['_local_emb'][:, emb]
    local_emb_list.append(f'local_emb_{emb}')

global_emb_list = []
for emb in global_dim:
    adata.obs[f'global_emb_{emb}'] = adata.obsm['_global_emb'][:, emb]
    global_emb_list.append(f'global_emb_{emb}')

sq.pl.spatial_scatter(
    adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])],
    spatial_key = 'spatial',
    library_key='sample', 
    #library_id = [839, 6],
    color=local_emb_list, #cell_type_coarse",
    shape= None,
    ncols = 3,
    size = 10,
)

sq.pl.spatial_scatter(
    adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])],
    spatial_key = 'spatial',
    library_key='sample', 
    #library_id = [839, 6],
    color=global_emb_list, #cell_type_coarse",
    shape= None,
    ncols = 3,
    size = 10,
)

# %%
adata.X = adata.layers['log1p_norm']

# %% [markdown]
# ### Morans I
#
# Can I visualize the different global and local 

# %%
df_global.index.values

# %%
from liana.methods._liana_core import _score_interactions

# Subset your gene pairs
gene_pairs = [('CHAT',  'KLF13', 'ZNF609',   'PER3')]

scores = li.mt.spatial_score(
    adata,
    gene_pairs=gene_pairs,
    distance_threshold=100,  # can I put it to no distance threshold? 
    method='local_weighted_cosine',
)

# %%
adata.obs['local_emb_1'] = adata.obsm['_local_emb'][:, 1]

# %%
adata.obs['pos_global_emb'] = adata[:, 'FOXO6'].X.todense()  + adata[:, 'NEUROG2'].X.todense() + adata[:, 'HHIP'].X.todense()
adata.obs['neg_global_emb'] = adata[:, 'NEUROD1'].X.todense()  + adata[:, 'PAX7'].X.todense() 

# %%
sq.pl.spatial_scatter(
    adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])],
    spatial_key = 'spatial',
    library_key='sample', 
    #library_id = [839, 6],
    color=['pos_global_emb', 'neg_global_emb'], #cell_type_coarse",
    shape= None,
    ncols = 3,
    size = 10,
)

# %%
import numpy as np, scanpy as sc, pandas as pd
from scipy.sparse import issparse

def get(adata, g):
    x = adata[:, g].X
    return np.asarray(x.todense()).ravel() if issparse(x) else np.asarray(x).ravel()

# Global
pos = ['FOXO6', 'NEUROG2', 'HHIP'] 
neg = ['NEUROD1', 'PAX7'] 
# Local
# pos = ['AGO2', 'CUX2', 'FOS', 'PAX6', 'TUBB3']
# neg = ['GLI1'] 

# check they all exist first
missing = [g for g in pos + neg if g not in adata.var_names]
assert not missing, missing

# z-scale each gene so no single gene dominates the sum
def module(adata, genes):
    M = np.vstack([get(adata, g) for g in genes])
    M = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)
    return M.sum(0)

adata.obs['pos_global_emb'] = module(adata, pos)
adata.obs['neg_global_emb'] = module(adata, neg)

# %%
THRESHOLD = 2

p = adata.obs['pos_global_emb'] > THRESHOLD
n = adata.obs['neg_global_emb'] > THRESHOLD

state = np.full(adata.n_obs, 'other', dtype=object)
state[p] = 'pos'
state[n] = 'neg'
state[p & n] = 'both'          # decide explicitly; otherwise neg silently overwrites pos

adata.obs['emb_state'] = pd.Categorical(
    state, categories=['other', 'pos', 'neg', 'both']
)


# %%
from matplotlib.colors import to_hex

palette = {'other': 'lightgrey', 'pos': 'tab:blue', 'neg': 'tab:red', 'both': 'green'}

cats = adata.obs['emb_state'].cat.categories
adata.uns['emb_state_colors'] = [to_hex(palette[c]) for c in cats]

sq.pl.spatial_scatter(
    adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])],
    spatial_key='spatial',
    library_key='sample',
    color='emb_state',
    shape=None,
    ncols=3,
    size=10,
)

# %%
import squidpy as sq

sub = adata[adata.obs['sample'].isin([slides["Ctrl"], slides["SSH"]])].copy()

sq.gr.spatial_neighbors(sub, library_key='sample', coord_type='generic', n_neighs=6)
sq.gr.nhood_enrichment(sub, cluster_key='emb_state')
sq.pl.nhood_enrichment(sub, cluster_key='emb_state')

# %%
