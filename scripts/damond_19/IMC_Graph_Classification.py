# %% [markdown]
# # IMC Pancreas - Graph classificiation evaluation

# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
## Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import InterScale as interscale
from InterScale.config import load_config
from InterScale.tl import prepare_geome_dataset, check_and_update_cfg
from InterScale.geome_dataloader import GraphAnnDataModule
#from InterScale.eval.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
from InterScale.tl import prepare_a2d_dataset
from InterScale.evaluation import scale_cls_by_sample

from pathlib import Path
import torch
import os

# %%
HPC = "/home/icb/francesca.drummer/1-Projects/"
LRZ = "/dss/dsshome1/05/di93tig/1_projects/" 

CFG_CLASS = f"{ICB}GT-long-range-niches/src/config_files/damond19/damond19_class_graph_Combined_condition.yaml"

RESULTS_DIR = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results/damond19/"

# %%
import yaml
import matplotlib.pyplot as plt

with open(os.path.join(HPC, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"]["Damond"]

# %% [markdown]
# ## Load pretrained models

# %%
cfg = load_config(CFG_CLASS)

# %%
adata = sc.read_h5ad(cfg.dataset.h5ad_data)
adata

# %%
adata.obs['group']

# %%
adata.obsm['spatial']

# %%
# import torch

# model = torch.load('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results/cosmx_pancreas/pancreas_regr_node_GCN__model.pt')
# state_dict = model['state_dict']

# %% [markdown]
# Want to load from: '/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/results/melton25/melton25_clas_graph_44_GCN_self-attn-transformer_model.ckpt'

# %%
#interscale.model.LocalModel._setup_anndata(adata = adata, prediction_task = cfg.dataset.prediction_task, layer_key = cfg.dataset.layer_key, sample_key_list = cfg.dataset.sample_key, prediction_obs =  cfg.dataset.prediction_obs, group_key = cfg.dataset.group_label, view_registry = False)
#local_model = interscale.model.LocalModel.load(RESULTS_DIR, adata, cfg, local_component = True, global_component = False, wandb_save = True)
interscale.model.CombinedModel._setup_anndata(adata = adata, prediction_task = cfg.dataset.prediction_task, layer_key = cfg.dataset.layer_key, sample_key_list = cfg.dataset.sample_key, prediction_obs =  cfg.dataset.prediction_obs, group_key = cfg.dataset.group_label, view_registry = False)
combined_model = interscale.model.CombinedModel.load(RESULTS_DIR, adata, cfg, local_component = True, global_component = True, wandb_save = True)

# %%
combined_model

# %% [markdown]
# ## Inference 

# %%
np.unique(adata.obs['stage'])

# %%
result_long = combined_model.get_model_output(adata[adata.obs['stage'] == 'Long-duration'], prefix = 'combined')

# %%
result_long.write(os.path.join(RESULTS_DIR, 'result_long.h5ad'))

# %%
result_nd = combined_model.get_model_output(adata[adata.obs['stage'] == 'Non-diabetic'], prefix = 'combined')

# %%
result_nd.write(os.path.join(RESULTS_DIR, 'result_nd.h5ad'))

# %%
result_onset = combined_model.get_model_output(adata[adata.obs['stage'] == 'Onset'], prefix = 'combined')

# %%
result_onset.write(os.path.join(RESULTS_DIR, 'result_onset.h5ad'))

# %% [markdown]
# ## Evaluation score

# %%
from InterScale.evaluation import calculate_pr_auc

# %% [markdown]
# ## Graph Label analysis
#
# First, we scale the CLS token and then calculate the average mean expression of the CLS (horizontal and vertical) token per cell type.

# %%
from InterScale.evaluation.graph_classification import scale_cls_by_sample
scale_cls_by_sample(result_long, "sliding_window_assignment")
scale_cls_by_sample(result_nd, "sliding_window_assignment")
scale_cls_by_sample(result_onset, "sliding_window_assignment")

# %%
np.unique(adata.obs['CellType'])

# %%
fig = plot_adata_grouped_heatmaps(
        adata_dict={'ND': result_nd, 'Onset': result_onset, 'Long': result_long},
        group_by='CellCat',
        #value_cols=['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled'],
        labels=['No Diabetes', 'Onset', 'Long'],
    agg_func='mean',
        figsize=(10, 8),
        save_path=f"{HPC}/InterScale_reproducibility/figures/damond19/graph_class/cls_CellCat.png"
    )

# %%
fig = plot_adata_grouped_heatmaps(
        adata_dict={'ND': result_nd, 'Onset': result_onset, 'Long': result_long},
        group_by='CellType',
        #value_cols=['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled'],
        labels=['No Diabetes', 'Onset', 'Long'],
    agg_func='mean',
        figsize=(10, 8),
        save_path=f"{HPC}/InterScale_reproducibility/figures/damond19/graph_class/cls_CellType.png"
    )

# %%
# assign cell type colors
color_list = [CELL_TYPE_COLORS[ct] for ct in adata.obs['CellType'].cat.categories]
result_long.uns['CellType_colors'] = color_list
result_onset.uns['CellType_colors'] = color_list
result_nd.uns['CellType_colors'] = color_list

# %%
image_nr_long = np.unique(result_long.obs['ImageNumber'][result_long.obs['split'] == 'test'])
image_subset = list(np.random.choice(image_nr_long, 2))
print(image_subset)
sq.pl.spatial_scatter(result_long[result_long.obs['split'] == 'test'], 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'CellType'],
                      library_key = 'ImageNumber',
                      library_id = image_subset,
                    cmap = PALETTE,
                    shape= None,
                      ncols = 3,
                      size = 50,
                      save = f"{HPC}/InterScale_reproducibility/figures/damond19/graph_class/"
)

# %%
image_nr_onset = np.unique(result_onset.obs['ImageNumber'][result_onset.obs['split'] == 'test'])
image_subset_onset = list(np.random.choice(image_nr_onset, 2))
print(image_subset_onset)
sq.pl.spatial_scatter(result_onset[result_onset.obs['split'] == 'test'], 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'CellType'],
                      library_key = 'ImageNumber',
                      library_id = image_subset_onset,
                    cmap = 'viridis_r',
                    shape= None,
                      ncols = 3,
                      size = 50,
                      save = f"{HPC}/InterScale_reproducibility/figures/legnini23/graph_class/"
)

# %%
image_nr_nd = np.unique(result_nd.obs['ImageNumber'][result_nd.obs['split'] == 'test'])
image_subset_nd = list(np.random.choice(image_nr_nd, 2))
print(image_subset_nd)
sq.pl.spatial_scatter(result_nd[result_nd.obs['split'] == 'test'], 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'CellType'],
                      library_key = 'ImageNumber',
                      library_id = image_subset_nd,
                    cmap = 'viridis_r',
                    shape= None,
                      ncols = 3,
                      size = 50,
                      save = f"{HPC}/InterScale_reproducibility/figures/legnini23/graph_class/"
)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Union


def plot_adata_grouped_heatmaps(
    adata_dict: Dict[str, 'AnnData'],
    group_by: str,
    value_cols: List[str] = None,
    labels: List[str] = None,
    split_key: Optional[str] = 'split',
    split_value: Optional[str] = 'test',
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    cmap: str = 'viridis_r',
    annot: bool = True,
    fmt: str = '.2f',
    linewidth: float = 0.5,
    shared_colorbar: bool = True,
    agg_func: str = 'mean',
    **kwargs
) -> plt.Figure:
    """
    Create side-by-side heatmaps from multiple AnnData objects with shared color scale.
    
    Parameters
    ----------
    adata_dict : Dict[str, AnnData]
        Dictionary with keys as condition names and values as AnnData objects.
        Order of keys determines subplot order.
    group_by : str
        Column name in adata.obs to group by (e.g., 'CellType', 'cluster').
    value_cols : List[str], optional
        List of column names to aggregate. If None, defaults to:
        ['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled']
    labels : List[str], optional
        Subplot titles. If None (default), automatically uses keys from adata_dict.
    split_key : str, optional
        Column name for filtering (e.g., 'split'). Set to None to skip filtering.
    split_value : str, optional
        Value to filter on (e.g., 'test'). Only used if split_key is not None.
    figsize : tuple, default (10, 6)
        Figure size (width, height).
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    cmap : str, default 'viridis_r'
        Colormap name.
    annot : bool, default True
        Whether to annotate heatmap cells with values.
    fmt : str, default '.2f'
        Format string for annotations.
    linewidth : float, default 0.5
        Width of lines dividing cells.
    shared_colorbar : bool, default True
        Whether to use a shared colorbar (only on last subplot).
    agg_func : str, default 'mean'
        Aggregation function ('mean', 'median', 'sum', etc.).
    **kwargs
        Additional keyword arguments passed to sns.heatmap.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> # Basic usage - subplot titles automatically use dict keys ('ND', 'Onset', 'Long')
    >>> adata_dict = {
    ...     'ND': result_nd,
    ...     'Onset': result_onset,
    ...     'Long': result_long
    ... }
    >>> fig = plot_adata_grouped_heatmaps(
    ...     adata_dict=adata_dict,
    ...     group_by='CellType',
    ...     save_path='output/cls_heatmap.png'
    ... )
    
    >>> # Custom columns and no filtering - titles use dict keys
    >>> fig = plot_adata_grouped_heatmaps(
    ...     adata_dict={'Control': adata1, 'Treatment': adata2},
    ...     group_by='cell_type',
    ...     value_cols=['gene_A', 'gene_B', 'gene_C'],
    ...     split_key=None,
    ...     figsize=(12, 5)
    ... )
    
    >>> # Only use custom labels if you want titles different from dict keys
    >>> fig = plot_adata_grouped_heatmaps(
    ...     adata_dict={'cond1': adata1, 'cond2': adata2},
    ...     group_by='CellType',
    ...     labels=['Control Group', 'Treatment Group']  # Optional override
    ... )
    """
    # Set default value columns if not provided
    if value_cols is None:
        value_cols = ['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled']
    
    # Use dict keys as labels if not provided (default behavior)
    if labels is None:
        labels = list(adata_dict.keys())
    else:
        # Validate if custom labels are provided
        if len(labels) != len(adata_dict):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of AnnData objects ({len(adata_dict)})")
    
    # Process each AnnData object and compute aggregated data
    processed_data = {}
    for key, adata in adata_dict.items():
        # Filter by split if specified
        if split_key is not None and split_value is not None:
            if split_key not in adata.obs.columns:
                raise KeyError(f"Column '{split_key}' not found in adata.obs for '{key}'")
            adata_filtered = adata[adata.obs[split_key] == split_value]
        else:
            adata_filtered = adata
        
        # Check if group_by column exists
        if group_by not in adata_filtered.obs.columns:
            raise KeyError(f"Column '{group_by}' not found in adata.obs for '{key}'")
        
        # Check if value columns exist
        for col in value_cols:
            if col not in adata_filtered.obs.columns:
                raise KeyError(f"Column '{col}' not found in adata.obs for '{key}'")
        
        # Create aggregation dictionary
        agg_dict = {f'mean_{col}': (col, agg_func) for col in value_cols}
        
        # Group and aggregate
        processed_data[key] = adata_filtered.obs.groupby(group_by).agg(**agg_dict)
    
    # Calculate shared color scale
    all_values = np.concatenate([df.values.flatten() for df in processed_data.values()])
    vmin = all_values.min()
    vmax = all_values.max()
    
    # Create figure with subplots
    n_plots = len(processed_data)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    # Handle case of single subplot
    if n_plots == 1:
        axes = [axes]
    
    # Plot each heatmap
    for idx, (key, label) in enumerate(zip(adata_dict.keys(), labels)):
        data = processed_data[key]
        
        # Determine if this is the last plot (for colorbar)
        show_cbar = shared_colorbar and (idx == n_plots - 1)
        
        sns.heatmap(
            data,
            annot=annot,
            linewidth=linewidth,
            fmt=fmt,
            ax=axes[idx],
            vmin=vmin,
            vmax=vmax,
            cbar=show_cbar,
            cmap=cmap,
            **kwargs
        )
        axes[idx].set_title(label)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

# %%
