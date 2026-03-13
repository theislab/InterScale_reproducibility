# %% [markdown]
# # CosMx Pancreas - Attention matrix evaluation
#
# Evaluate attention matrix from trained regression model.

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd

## plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

## InterScale imports
import InterScale as interscale
from InterScale.config import load_config
from InterScale.tl import prepare_geome_dataset, check_and_update_cfg
from InterScale.geome_dataloader import GraphAnnDataModule
from InterScale.evaluation.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
from InterScale.tl import prepare_a2d_dataset
from InterScale.evaluation import scale_cls_by_sample

from pathlib import Path
import torch

# %% [markdown]
# ## Set up paths
#
# Check which cluster by testing for distinctive directories then set:
#
# - `BASE_DIR_REPO`: path to github repo (InterScale code)
# - `BASE_DIR_PROJECT`: path to project folder, where results, models, etc are stored

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
    BASE_DIR_REPO = "/home/icb/francesca.drummer/1-Projects/"
    BASE_DIR_PROJECT = ""
else:
    print('unkown')
    CLUSTER = 'unknown'

DATA = "melton25"

# %%
# path on ICB or LRZ cluster to InterScale_reproducibility folder
CFG_CLASS = os.path.join(BASE_DIR_REPO, "GT-long-range-niches/src/config_files/Cosmx_pancreas/regr_InterScale.yaml")
RESULTS_DIR = os.path.join(BASE_DIR_PROJECT, f"results/{DATA}/")
FIGURE_DIR = os.path.join(BASE_DIR_REPO, f"InterScale_reproducibility/figures/{DATA}")

# %%
path = os.path.join("/home", "user", "documents", "/etc", "config.txt")
path

# %%
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from notebook location)
project_root = Path(f'{BASE_DIR_REPO}/InterScale_reproducibility')
sys.path.insert(0, str(project_root))

from src.utils import set_full_reproducibility
from src.wandb import Wandb_evaluation

# %% [markdown]
# ## Global parameters

# %% [markdown]
# Fix the seeds across all imports.

# %%
set_full_reproducibility()

# %% [markdown]
# ## Figure settings

# %%
import yaml
import matplotlib.pyplot as plt

config_path = os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]
CONDITION_COLORS = config["palettes"]["Condition_pancreas"]

# %%
from figures.scripts.plots import Plotting

plotting = Plotting(config_path)
plotting._setup_plotting_params()

# %%
CELL_TYPE_KEY = 'cell_type_coarse'

# %% [markdown]
# ## Load config and model

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, "data", f"{DATA}.h5ad"))
adata

# %% [markdown]
# ### WandB

# %%
import wandb
from yacs.config import CfgNode as CN

WANDB_ENTITY = "francesca-drummer"


def load_model(project: str, best_run_id: str, adata = None):
    """Load best model artifact from WandB according to metric."""
    
    # 1. Get best run and associated config
    api = wandb.Api()
    best_run = api.run(f"{WANDB_ENTITY}/{project}/{best_run_id}")
    config_dict = best_run.config
    
    cfg = CN(config_dict)
    # cfg.optim.accelerator = 'cpu'
    # cfg.model.decoder.dual_decoder = False
    # cfg.model.global_component.parameters.type_gex_embedding = None
    # cfg.freeze()  # Optional: make it immutable
    
    # 4. Download the model artifact
    artifact = list(best_run.logged_artifacts())[0]
    artifact_dir = artifact.download()
    print(f"Model artifact downloaded to: {artifact_dir}")

    if adata is None:
        adata = sc.read_h5ad(cfg.dataset.h5ad_data)
    
    # 5. Setup AnnData
    interscale.model.CombinedModel._setup_anndata(
        adata=adata, 
        prediction_task=cfg.dataset.prediction_task, 
        layer_key=cfg.dataset.layer_key, 
        sample_key_list=cfg.dataset.sample_key, 
        prediction_obs=cfg.dataset.prediction_obs, 
        group_key=cfg.dataset.group_label, 
        view_registry=False
    )
    
    # 6. Load model from the artifact directory
    combined_model = interscale.model.CombinedModel.load(
        artifact_dir,
        adata, 
        cfg= cfg,
        model_name = f"{artifact_dir}/model",
        local_component=True, 
        global_component=True, 
        wandb_save=False
    )
    
    print(f"Model loaded successfully from run: {best_run_id}")

    return combined_model, cfg, adata


# %%
PROJECT = "GTLongRange_CosmXPancreas"
InterScale_best = "pqolkd11"

# %%
combined_model, cfg, adata = load_model(PROJECT, InterScale_best, adata)

# %% [markdown]
# ### Local source

# %%
cfg = load_config(CFG_CLASS)

# %%
assert BASE_DIR_PROJECT in cfg.model.save 
assert BASE_DIR_PROJECT in cfg.dataset.h5ad_data

# %%
# assign cell type colors
color_list = [CELL_TYPE_COLORS[ct] for ct in adata.obs[CELL_TYPE_KEY].cat.categories]
adata.uns[f'{CELL_TYPE_KEY}_colors'] = color_list

# %%
model_name = os.path.join('melton25_regr_node_44_GCN_self-attn-transformer_')

# %%
'melton25_regr_node_44_GCN_self-attn-transformer_' == 'melton25_regr_node_44_GCN_self-attn-transformer_'

# %%
#interscale.model.LocalModel._setup_anndata(adata = adata, prediction_task = cfg.dataset.prediction_task, layer_key = cfg.dataset.layer_key, sample_key_list = cfg.dataset.sample_key, prediction_obs =  cfg.dataset.prediction_obs, group_key = cfg.dataset.group_label, view_registry = False)
#local_model = interscale.model.LocalModel.load(RESULTS_DIR, adata, cfg, local_component = True, global_component = False, wandb_save = True)
interscale.model.CombinedModel._setup_anndata(adata = adata, prediction_task = cfg.dataset.prediction_task, layer_key = cfg.dataset.layer_key, sample_key_list = cfg.dataset.sample_key, prediction_obs =  cfg.dataset.prediction_obs, group_key = cfg.dataset.group_label, view_registry = False)
combined_model = interscale.model.CombinedModel.load(RESULTS_DIR, adata, cfg, model_name = model_name, local_component = True, global_component = True, wandb_save = True)

# %% [markdown]
# ## Inference: get model output 

# %%
slide_ids = list(config["slide_examples"]["Melton_Jimenez"].values())

# %%
sub_adata = adata[adata.obs['slide_fov'].isin(slide_ids)]

# %%
result = combined_model.get_model_output(sub_adata, prefix = 'combined')

# %%
result.obs['condition']

# %%
result_complete = combined_model.get_model_output(adata, prefix = 'combined')

# %%
#### from InterScale.evaluation.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
gene_rank_analysis(result_complete[result_complete.obs['condition']=='T1D'],
                   layers_local_pred = 'combined_y_pred_local',
                   layers_global_pred = 'combined_y_pred_global',
                   top_n = 5,
                   plot_result = True,
                   return_top_genes = True)

# %% [markdown]
# ### Highly variable genes

# %%
assert result_complete.X[:10, :10].dtype == np.float32

# %%
sc.pp.highly_variable_genes(result_complete, flavor = "seurat_v3", n_top_genes = 500)

# %%
result_hvg = result_complete[:,result_complete.var['highly_variable']]

# %%
#### from InterScale.evaluation.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
gene_rank_analysis(result_hvg[result_hvg.obs['condition']=='T1D'],
                   layers_local_pred = 'combined_y_pred_local',
                   layers_global_pred = 'combined_y_pred_global',
                   top_n = 5,
                   plot_result = True,
                   return_top_genes = True)

# %%
gene_rank_analysis(result_hvg[result_hvg.obs['condition']=='ND'],
                   layers_local_pred = 'combined_y_pred_local',
                   layers_global_pred = 'combined_y_pred_global',
                   top_n = 5,
                   plot_result = True,
                   return_top_genes = True)

# %%
gene_rank_analysis(result_hvg[result_hvg.obs['condition']=='T1D'],
                   layers_local_pred = 'combined_y_pred_local',
                   layers_global_pred = 'combined_y_pred_global',
                   top_n = 5,
                   plot_result = True,
                   return_top_genes = True, 
                   score_metric = "cosine")

# %% [markdown]
# ### Condition-based gene-rank analysis

# %%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from anndata import AnnData
from typing import Literal, Optional, List, Tuple
from matplotlib.lines import Line2D

def _compute_gene_scores(adata: AnnData, layers_pred: str,
                         score_metric: Literal["r2", "cosine"] = "r2") -> pd.DataFrame:
    """Compute per-gene scores (R² or cosine) between adata.X and a prediction layer."""
    y_true = adata.X.toarray().astype(float)
    y_pred = adata.layers[layers_pred]
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    y_pred = y_pred.astype(float)

    scores = []
    for i in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, i]) & ~np.isnan(y_pred[:, i])
        a, b = y_true[mask, i], y_pred[mask, i]
        n = np.sum(mask)

        if score_metric == "r2":
            score = r2_score(a, b) if n > 1 else np.nan
        else:  # cosine
            norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
            score = np.dot(a, b) / (norm_a * norm_b) if (n > 0 and norm_a > 0 and norm_b > 0) else np.nan
        scores.append(score)

    ranks = rankdata(scores, method="average")
    return pd.DataFrame({
        "gene": adata.var_names,
        "score": scores,
        "rank": ranks,
    })


def gene_rank_condition_comparison(
    adata: AnnData,
    library_key: str,
    library_id: Optional[str | List[str]] = None,
    layers_local_pred: str = "layers_local",
    layers_global_pred: str = "layers_global",
    top_n: int = 5,
    score_metric: Literal["r2", "cosine"] = "r2",
    color_dict: Optional[dict[str, str]] = None,
    save_dir: Optional[str] = None,
    post_fix: Optional[str] = None,
    return_top_genes: bool = False,
) -> Optional[dict]:
    """Compare gene prediction ranks between two conditions for local and global decoders.

    For each decoder (local / global), genes are scored independently in
    the two selected library IDs. The resulting ranks are plotted against each
    other so you can see which genes are better predicted under one condition
    vs. the other.

    Parameters
    ----------
    adata : AnnData
        Must contain the prediction layers and ``adata.obs[library_key]``.
    library_key : str
        Column in ``adata.obs`` that stores condition / library labels.
    library_id : str or list of str, optional
        Exactly two values from ``adata.obs[library_key]`` to compare.
        If ``None``, the first two unique values in the column are used.
        A single string is interpreted as a one-element list; in that case
        the second unique value is inferred automatically.
    layers_local_pred, layers_global_pred : str
        Layer names for local and global predictions.
    top_n : int
        Number of top genes to highlight per category.
    score_metric : 'r2' | 'cosine'
        Metric used for scoring.
    color_dict : dict, optional
        Mapping of library ID to color string, e.g.
        ``{'ND': '#999999', 'T1D': '#0b559f'}``. If ``None``, defaults
        to blue / red.
    save_dir : str, optional
        Directory for saving figures.
    post_fix : str, optional
        Suffix appended to saved filenames.
    return_top_genes : bool
        If True, return a dict of DataFrames with top genes.

    Returns
    -------
    dict or None
        If ``return_top_genes``, returns a dict with keys
        ``'local'`` and ``'global'``, each containing
        ``(top_id_a, top_id_b, top_consensus)`` DataFrames.
    """
    # --- validate inputs ---
    assert library_key in adata.obs.columns, (
        f"'{library_key}' not found in adata.obs"
    )
    for layer in [layers_local_pred, layers_global_pred]:
        assert layer in adata.layers, f"'{layer}' not found in adata.layers"

    # --- resolve library_id to exactly two values ---
    unique_ids = adata.obs[library_key].unique().tolist()

    if library_id is None:
        assert len(unique_ids) >= 2, (
            f"Need at least 2 unique values in '{library_key}', found {len(unique_ids)}. "
            "Pass library_id explicitly."
        )
        id_a, id_b = unique_ids[0], unique_ids[1]
    else:
        if isinstance(library_id, str):
            library_id = [library_id]
        for lid in library_id:
            assert lid in unique_ids, (
                f"'{lid}' not found in adata.obs['{library_key}']. "
                f"Available: {unique_ids}"
            )
        if len(library_id) == 1:
            remaining = [v for v in unique_ids if v != library_id[0]]
            assert len(remaining) >= 1, (
                f"Only one unique value in '{library_key}'; cannot infer second ID."
            )
            id_a, id_b = library_id[0], remaining[0]
        elif len(library_id) == 2:
            id_a, id_b = library_id[0], library_id[1]
        else:
            raise ValueError(
                f"library_id must contain exactly 1 or 2 values, got {len(library_id)}."
            )

    mask_a = adata.obs[library_key] == id_a
    mask_b = adata.obs[library_key] == id_b
    assert mask_a.sum() > 0, f"No cells found for library_id '{id_a}'"
    assert mask_b.sum() > 0, f"No cells found for library_id '{id_b}'"

    adata_a = adata[mask_a].copy()
    adata_b = adata[mask_b].copy()

    # --- compute scores per condition & decoder ---
    results = {}
    for layer_name, layer_label in [
        (layers_local_pred, "Local"),
        (layers_global_pred, "Global"),
    ]:
        df_a = _compute_gene_scores(adata_a, layer_name, score_metric)
        df_b = _compute_gene_scores(adata_b, layer_name, score_metric)

        merged = pd.merge(
            df_a[["gene", "rank"]].rename(columns={"rank": f"Rank {id_a}"}),
            df_b[["gene", "rank"]].rename(columns={"rank": f"Rank {id_b}"}),
            on="gene",
            how="inner",
        )

        # Rank difference: positive → better in id_b
        merged["Rank Difference"] = (
            merged[f"Rank {id_a}"] - merged[f"Rank {id_b}"]
        )
        merged["Avg Rank"] = (
            merged[f"Rank {id_a}"] + merged[f"Rank {id_b}"]
        ) / 2

        top_a = merged.nlargest(top_n, "Rank Difference")   # better in id_a
        top_b = merged.nsmallest(top_n, "Rank Difference")   # better in id_b
        top_consensus = merged.nlargest(top_n, "Avg Rank")   # best overall

        # --- resolve colors ---
        default_colors = {id_a: "blue", id_b: "red"}
        colors = color_dict if color_dict is not None else default_colors
        color_a = colors[id_a]
        color_b = colors[id_b]

        # --- plot ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            merged[f"Rank {id_a}"],
            merged[f"Rank {id_b}"],
            alpha=0.5, color="gray",
        )

        for df_top, color in [
            (top_a, color_a),
            (top_b, color_b),
        ]:
            ax.scatter(
                df_top[f"Rank {id_a}"],
                df_top[f"Rank {id_b}"],
                color=color, zorder=3,
            )
            for _, row in df_top.iterrows():
                ax.annotate(
                    row["gene"],
                    xy=(row[f"Rank {id_a}"], row[f"Rank {id_b}"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=9, color=color,
                )

        # diagonal
        lo = merged[[f"Rank {id_a}", f"Rank {id_b}"]].values.min()
        hi = merged[[f"Rank {id_a}", f"Rank {id_b}"]].values.max()
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4)

        ax.set_xlabel(f"Rank — {id_a}")
        ax.set_ylabel(f"Rank — {id_b}")
        ax.set_title(
            f"{layer_label} Decoder: {id_a} vs {id_b} ({score_metric})"
        )

        # legend with only condition colors, placed outside
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_a,
                   markersize=8, label=id_a),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color_b,
                   markersize=8, label=id_b),
        ]
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1),
                  loc="upper left", borderaxespad=0, frameon=False)

        if save_dir is not None:
            name = (
                f"gene_rank_{layer_label.lower()}_{id_a}_vs_{id_b}_{score_metric}"
                + (f"_{post_fix}" if post_fix else "")
                + ".png"
            )
            path = os.path.join(save_dir, name)
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {path}")

        plt.show()

        results[layer_label.lower()] = (top_a, top_b, top_consensus)

    if return_top_genes:
        return results


# %%
gene_rank_condition_comparison(
    result_hvg,
    library_key="condition",
    layers_local_pred = 'combined_y_pred_local',
    layers_global_pred = 'combined_y_pred_global',
    color_dict = CONDITION_COLORS
)

# %% [markdown]
# ### GEX prediction
#
# Check if genes are well predicted for local and global embedding.

# %%
result

# %%
sq.pl.spatial_scatter(result_complete, 
                      color = ['TYK2', 'SERPINA3', 'TTR'],
                      library_key = 'slide_fov',
                      library_id = slide_ids,
                       ncols=3,
                    shape = None)

# %%
sq.pl.spatial_scatter(result, 
                      layer = 'combined_y_pred_local',
                      color = ['TYK2', 'SERPINA3', 'TTR', 'cell_type_coarse'],
                      library_key = 'slide_fov',
                    shape = None)

# %%
sq.pl.spatial_scatter(result, 
                      layer = 'combined_y_pred_global',
                      color = ['TYK2', 'SERPINA3', 'TTR', 'cell_type_coarse'],
                      library_key = 'slide_fov',
                    shape = None)


# %%
def umap_embeddings(embedding_gnn, 
                    embedding_trans, 
                    leiden_res: float = 0.1):
    """

    
    Input:
    ------
        embedding_gnn: numpy.array [N, E_gnn]
        embedding_trans: numpy.array [N, E_trans]

    Return:
    -------
        leiden_gnn: numpy.array [N]
            Leiden cluster assignments for each cell based on GNN output
        leiden_trans: numpy.array [N]
            Leiden cluster assignments for each cell based on Transformer output
    """
    scaler = StandardScaler()
    H_GNN_normalized = scaler.fit_transform(embedding_gnn)
    H_T_normalized = scaler.fit_transform(embedding_trans)
    
    # Apply UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    H_GNN_umap = umap_model.fit_transform(H_GNN_normalized)
    H_T_umap = umap_model.fit_transform(H_T_normalized)

    adata_gnn = AnnData(X=H_GNN_umap)
    
    # Build a KNN graph and perform Leiden clustering
    sc.pp.neighbors(adata_gnn, n_neighbors=40, use_rep='X')
    sc.tl.leiden(adata_gnn, resolution=leiden_res)

    leiden_gnn = np.array(adata_gnn.obs['leiden'])
    
    adata_trans = AnnData(X=H_T_umap)
    
    # Build a KNN graph and perform Leiden clustering
    sc.pp.neighbors(adata_trans, n_neighbors=40, use_rep='X')
    sc.tl.leiden(adata_trans, resolution=leiden_res)
    
    # Extract Leiden clusters
    leiden_trans = np.array(adata_trans.obs['leiden'])
    print(f"Nr. of clusters: local = {len(np.unique(leiden_gnn))}, global = {len(np.unique(leiden_trans))}")

    del adata_gnn, adata_trans
    
    return leiden_gnn, leiden_trans


# %%
result


# %%
def normalized_attention(attention_matrix, clamp = 0.05):
    np.fill_diagonal(attention_matrix.values, 0)
    
    # Clamp and scale attention matrix
    scores = torch.tensor(attention_matrix.values)
    if clamp:
        q05, q95 = torch.quantile(scores, clamp), torch.quantile(scores, 1-clamp)
        scores = np.clip(scores, a_min=q05, a_max=q95)
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores)
    return scores

def normalized_class_attention(adata, attn_matrix_key, class_key, clamp: int = 0.05):
    """
    Given an attention matrix of size NxN with K classes it returns a normalized attention matrix KxK.
    Each element in the normalized attention matrix can be interpreted as class k_i paying attention to class k_j, where i and j are elements of the K classes.

    Parameters
    ----------
        attention_matrix: AnnData
        attn_matrix_key: str
            Key in .obsm pointing to saved attention matrix during inference
        class_key: str
            Key pointing to class in .obs which is used to normalize over
    Returns
    -------
        attn_norm: 
            KxK, where 
    """
    scores = normalized_attention(attention_matrix, clamp)
    attention_matrix = pd.DataFrame(scores, index = attention_matrix.index, columns = attention_matrix.columns)
    
    # Create an empty KxK DataFrame to store the summed and normalized attention values
    class_names = np.unique(attention_matrix.columns)
    K = len(class_names)
    attn_norm = pd.DataFrame(np.zeros((K, K)), index=class_names, columns=class_names)

    # Iterate over each unique cell type combination
    for i, class_i in enumerate(class_names):
        for j, class_j in enumerate(class_names):
            # Find the indices in the original CxC DataFrame that correspond to the given cell types
            indices_i = (attention_matrix.index == class_i)
            indices_j = (attention_matrix.columns == class_j)
            norm_value = attention_matrix.loc[indices_i, indices_j].sum() / len(np.argwhere(indices_i==True))
            summed_value = norm_value.sum() / len(np.argwhere(indices_j==True))
            attn_norm.at[class_i, class_j] = summed_value

    return attn_norm


# %%
def normalized_class_attention(
    adata, 
    attn_matrix_key, 
    class_key, 
    clamp: float = 0.05,
    key_added: str = None,
    copy: bool = False
):
    """
    Given an attention matrix of size NxN with K classes it returns a normalized attention matrix KxK.
    Each element in the normalized attention matrix can be interpreted as class k_i paying attention 
    to class k_j, where i and j are elements of the K classes.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object containing attention matrix and cell type annotations
    attn_matrix_key : str
        Key in .obsm pointing to saved attention matrix during inference
    class_key : str
        Key pointing to class in .obs which is used to normalize over
    clamp : float, optional
        Quantile value for clamping (default: 0.05)
    key_added : str, optional
        If provided, save normalized attention matrix to adata.uns[key_added].
        If None, return the matrix instead.
    copy : bool, optional
        If True, return a copy of adata with the result added. Only relevant if key_added is not None.
        (default: False)
    
    Returns
    -------
    If key_added is None:
        attn_norm : pd.DataFrame
            KxK DataFrame where entry (i,j) represents mean normalized attention from class i to class j
    If key_added is not None and copy is False:
        None (modifies adata in place)
    If key_added is not None and copy is True:
        adata : AnnData
            Copy of adata with normalized attention added to .uns
    """
    adata = adata.copy() if copy else adata
    
    # Extract attention matrix from adata
    attention_matrix = adata.obsm[attn_matrix_key]
    
    # Convert to DataFrame with cell indices
    attention_df = pd.DataFrame(
        attention_matrix, 
        index=adata.obs_names, 
        #columns=adata.obs_names
    )
    
    # Normalize the attention matrix (clamp outliers and scale to [0,1])
    #scores = normalized_attention(attention_df, clamp)
    attention_df = pd.DataFrame(attention_df, index=attention_df.index, columns=attention_df.columns)
    
    # Get cell type labels for each cell
    cell_types = adata.obs[class_key]
    
    # Get unique cell types
    class_names = cell_types.unique()
    K = len(class_names)
    
    # Create empty KxK DataFrame to store normalized attention values
    attn_norm = pd.DataFrame(np.zeros((K, K)), index=class_names, columns=class_names)
    
    # Compute mean attention for each cell type pair
    for class_i in class_names:
        for class_j in class_names:
            # Find cells belonging to each type
            cells_i = cell_types == class_i
            cells_j = cell_types == class_j
            
            # Extract submatrix: attention from cells of type i to cells of type j
            submatrix = attention_df.loc[cells_i, cells_j]
            
            # Compute mean attention (averaged over all source-target cell pairs)
            attn_norm.at[class_i, class_j] = submatrix.values.mean()
    
    # Save to adata or return
    if key_added is not None:
        adata.uns[key_added] = attn_norm
        return adata if copy else None
    else:
        return attn_norm


# %%
result.obsm['combined_attn_matrix']

# %%
attn_norm = normalized_class_attention(result, 'combined_attn_matrix', 'cell_type_coarse', copy = True)

# %%
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def normalized_attention(attention_matrix, clamp=0.05):
    # Ensure it's a numeric numpy array
    if isinstance(attention_matrix, pd.DataFrame):
        scores = attention_matrix.values.astype(np.float32)
    else:
        scores = np.array(attention_matrix, dtype=np.float32)
    
    # Set diagonal to 0
    np.fill_diagonal(scores, 0)
    
    # Clamp and scale attention matrix
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    if clamp:
        q05 = torch.quantile(scores_tensor, clamp)
        q95 = torch.quantile(scores_tensor, 1-clamp)
        scores = np.clip(scores, a_min=q05.item(), a_max=q95.item())
    
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores)
    return scores

# Get unique cell types
class_key = 'cell_type_coarse'
class_names = result.obs[class_key].unique()
K = len(class_names)

# Store normalized attention matrices for each sliding window
window_attn_matrices = []

for sliding_wind in np.unique(result.obs['sliding_window_square']):
    sub_result = result[result.obs['sliding_window_square'] == sliding_wind]
    n_cells = sub_result.shape[0]
    print(f"Window {sliding_wind}: Nr of cells: {n_cells}")
    
    # Extract the NxN attention matrix for this window
    attn_matrix = sub_result.obsm['combined_attn_matrix'][:, :n_cells]
    
    # Convert to DataFrame (ensure numeric)
    attn_df = pd.DataFrame(
        attn_matrix.astype(np.float32),
        index=sub_result.obs_names,
        columns=sub_result.obs_names
    )
    
    # Apply normalization (clamp outliers and scale to [0,1])
    attn_normalized = normalized_attention(attn_df, clamp=0.05)
    
    # Get cell types for cells in this window
    cell_types = sub_result.obs[class_key].values
    
    # Create KxK normalized attention matrix for this window
    attn_norm_window = pd.DataFrame(
        np.zeros((K, K)), 
        index=class_names, 
        columns=class_names
    )
    
    # Compute mean attention for each cell type pair
    for class_i in class_names:
        for class_j in class_names:
            # Find cells of each type
            cells_i = cell_types == class_i
            cells_j = cell_types == class_j
            
            # Skip if either cell type is not present
            if not cells_i.any() or not cells_j.any():
                continue
            
            # Get indices
            i_indices = np.where(cells_i)[0]
            j_indices = np.where(cells_j)[0]
            
            # Extract submatrix: attention from cells of type i to cells of type j
            submatrix = attn_normalized[np.ix_(i_indices, j_indices)]
            
            # Compute mean attention
            attn_norm_window.at[class_i, class_j] = submatrix.mean()
    
    window_attn_matrices.append(attn_norm_window)
    print(f"  Normalized to {K}x{K} class attention matrix")

# Average across all windows
attn_norm_avg = pd.DataFrame(
    np.zeros((K, K)), 
    index=class_names, 
    columns=class_names
)

for class_i in class_names:
    for class_j in class_names:
        # Collect values from all windows (excluding zero entries)
        values = [m.at[class_i, class_j] for m in window_attn_matrices 
                 if m.at[class_i, class_j] > 0]
        
        if len(values) > 0:
            attn_norm_avg.at[class_i, class_j] = np.mean(values)
        else:
            attn_norm_avg.at[class_i, class_j] = 0

# Scale final matrix to [0, 1]
min_val = attn_norm_avg.values.min()
max_val = attn_norm_avg.values.max()

if max_val > min_val:  # Avoid division by zero
    attn_norm_avg = (attn_norm_avg - min_val) / (max_val - min_val)
else:
    attn_norm_avg = attn_norm_avg * 0  # All zeros if no variation

print("\nFinal averaged normalized attention matrix (scaled to [0,1]):")
print(attn_norm_avg)

print("\nFinal averaged normalized attention matrix:")
print(attn_norm_avg)

# %%
attn_norm = pd.DataFrame(result.obsm['combined_attn_matrix'])

 # %%
 # Identify columns (and rows) with all NaN values
nan_cols = attn_norm.columns[attn_norm.isna().all(axis=0)].tolist()
nan_rows = attn_norm.index[attn_norm.isna().all(axis=1)].tolist()

assert attn_norm.shape[1] - len(nan_cols) == attn_norm.shape[0]

if nan_cols or nan_rows:
    print(f"Warning: Found cell types with all NaN values:")
    if nan_cols:
        print(f"  Columns (target): {len(nan_cols)}")
    if nan_rows:
        print(f"  Rows (source): {nan_rows}")

# %%
attn_norm.shape[0] - len(nan_cols)


# %%
result.obs['condition']

# %% [markdown]
# ## Can we interpret the CLS token on regression trained model?
#
# First, scale the CLS token per sliding window so that they are comparable despite their different number of cells. 

# %%
result_t1d = result[result.obs['condition'] == 'T1D']
scale_cls_by_sample(result_t1d, "slide_fov")
result_nd = result[result.obs['condition'] == 'ND']
scale_cls_by_sample(result_nd, "slide_fov")

# %%

# %%
data_t1d = result_t1d.obs.groupby('cell_type_coarse').agg(
    mean_cls_horizontal=('combined_cls_horizontal_scaled', 'mean'),
    mean_cls_vertical=('combined_cls_vertical_scaled', 'mean')
)

data_nd = result_nd.obs.groupby('cell_type_coarse').agg(
    mean_cls_horizontal=('combined_cls_horizontal_scaled', 'mean'),
    mean_cls_vertical=('combined_cls_vertical_scaled', 'mean')
)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(5, 6))

# Get min/max for shared color scale
vmin = min(data_t1d.values.min(), data_nd.values.min())
vmax = max(data_t1d.values.max(), data_nd.values.max())

# Plot with shared color scale
sns.heatmap(data_t1d, annot=True, linewidth=.5, fmt='.2f', 
            ax=axes[0], vmin=vmin, vmax=vmax, cbar=False, cmap = PALETTE)
axes[0].set_title('T1D')

sns.heatmap(data_nd, annot=True, linewidth=.5, fmt='.2f', 
            ax=axes[1], vmin=vmin, vmax=vmax, cmap = PALETTE)
axes[1].set_title('ND')

plt.tight_layout()
plt.show()

# %%

# %%

# %%
slide_nr = np.unique(result_nd.obs['slide_fov'])[0]
sq.pl.spatial_scatter(result_nd, 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'cell_type_coarse'],
                    cmap = PALETTE,
                    shape= None,
                      ncols = 3,
                      save = os.path.join(FIGURE_DIR, f"spatial_cls_t1d_{slide_nr}.png")
)

# %%
# assign cell type colors
color_list = [CELL_TYPE_COLORS[ct] for ct in result_nd.obs[CELL_TYPE_KEY].cat.categories]
result_nd.uns[f'{CELL_TYPE_KEY}_colors'] = color_list

# %%
slide_nr = np.unique(result_t1d.obs['slide_fov'])[0]
sq.pl.spatial_scatter(result_t1d, 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'cell_type_coarse'],
                   cmap = PALETTE,
                    shape= None,
                      ncols = 3,
                      save = os.path.join(FIGURE_DIR, f"spatial_cls_t1d_{slide_nr}.png")
)

# %%
