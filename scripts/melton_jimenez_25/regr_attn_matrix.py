# %% [markdown]
# # CosMx Pancreas - Attention matrix evaluation
#
# Evaluate attention matrix from trained regression model.

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
#from InterScale.eval.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
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

# %%
# path on ICB or LRZ cluster to InterScale_reproducibility folder
CFG_CLASS = os.path.join(BASE_DIR_REPO, "GT-long-range-niches/src/config_files/Cosmx_pancreas/regr_InterScale.yaml")
RESULTS_DIR = os.path.join(BASE_DIR_PROJECT, "results/melton25/")
FIGURE_DIR = os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/melton25")

# %%
path = os.path.join("/home", "user", "documents", "/etc", "config.txt")
path

# %% [markdown]
# ## Figure settings

# %%
import yaml
import matplotlib.pyplot as plt

with open(os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"]["Melton_Jimenez"]

# %%
CELL_TYPE_KEY = 'cell_type_coarse'

# %% [markdown]
# ## Load config and model

# %%
cfg = load_config(CFG_CLASS)

# %%
assert BASE_DIR_PROJECT in cfg.model.save 
assert BASE_DIR_PROJECT in cfg.dataset.h5ad_data

# %%
adata = sc.read_h5ad(cfg.dataset.h5ad_data)
adata

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
sq.pl.spatial_scatter(result_nd, 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'cell_type_coarse'],
                    cmap = PALETTE,
                    shape= None,
                      ncols = 3,
)

# %%
# assign cell type colors
color_list = [CELL_TYPE_COLORS[ct] for ct in result_nd.obs[CELL_TYPE_KEY].cat.categories]
result_nd.uns[f'{CELL_TYPE_KEY}_colors'] = color_list

# %%
sq.pl.spatial_scatter(result_t1d, 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'cell_type_coarse'],
                   cmap = PALETTE,
                    shape= None,
                      ncols = 3,
)
