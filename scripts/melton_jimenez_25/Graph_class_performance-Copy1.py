# %% [markdown]
# # Graph classification sweep performance 
#
# Plots showing the performance for the graph classification on the CosMx Pancrease data with no diabetes and Type 1 Diabetes conditions.
#
# - InterScale (GCN + Transformer)
# - GCN
# - PCATransformer

# %%
LRZ = "/dss/dsshome1/05/di93tig/1_projects/"
HPC = "/home/icb/francesca.drummer/1-Projects/"
DATA = "melton25"

# %%
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from notebook location)
project_root = Path(f'{HPC}InterScale_reproducibility')
sys.path.insert(0, str(project_root))

# %%
import wandb
import pandas as pd
wandb.login()

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# ste sys path correctly
#from src.wandb import load_result_as_df, compute_mean_and_std, summary_df, plot_robustness
from src.wandb import Wandb_evaluation, plot_f1_across_seeds

# %%
import yaml
import matplotlib.pyplot as plt

with open(os.path.join(HPC, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

# %% [markdown]
# ## Load WandB IDs

# %%
GNN_sweep = "6qip4qz4"
InterScale_sweep = "8a7ycqw7"
PCATrans_sweep = "8v6uc6k4"

# %% [markdown]
# ## Graph Classification performance

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ['ND', 'T1D']

# %% [markdown]
# Q: Why does PCATrans not have test_f1 scores?

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, SWEEP_GOAL, CLASSES)
PCATrans_wandb = Wandb_evaluation("PCATransformer", PCATrans_sweep, SWEEP_GOAL, CLASSES)
InterScale_wandb = Wandb_evaluation("InterScale", InterScale_sweep, SWEEP_GOAL, CLASSES)

# %%
GNN_wandb.get_dataframe().head()

# %%
PCATrans_wandb.get_mean_and_std()

# %%
GNN_wandb.plot_robustness(metric="test_acc", save_path = f"{HPC}/InterScale_reproducibility/figures/melton25/graph_class")

# %%
PCATrans_wandb.plot_robustness(metric="test_acc", save_path = f"{HPC}/InterScale_reproducibility/figures/melton25/graph_class")

# %%
InterScale_wandb.plot_robustness(metric="test_acc", save_path = f"{HPC}/InterScale_reproducibility/figures/melton25/graph_class")

# %%
wandb_evals = [GNN_wandb, PCATrans_wandb, InterScale_wandb]  # Your list of dataframes

g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius=150,
    pct_mask_nodes=0.0,
    config_path='config.yml',
    height=6,
    aspect=0.9
)
plt.show()
#
# # Save with high DPI from config
# # g.savefig('f1_scores.png', bbox_inches='tight')
#
# # Print statistics:
# print(stats)

# %% [markdown]
# # Interpretation of CLS token
#
# Given the sweep, we obtain the best model and check the CLS token.

# %%
from InterScale.evaluation.graph_classification import scale_cls_by_sample#, plot_adata_grouped_heatmaps
import squidpy as sq

# %%
InterScale_wandb = Wandb_evaluation("InterScale", InterScale_sweep, SWEEP_GOAL, CLASSES)

# %%
import InterScale as interscale

combined_model, model_config, adata = InterScale_wandb.load_model("test_f1/class_T1D")

# %%
adata

# %%
result_t1d = combined_model.get_model_output(adata[adata.obs['condition'] == 'T1D'], prefix = 'combined')
result_nd = combined_model.get_model_output(adata[adata.obs['condition'] == 'ND'], prefix = 'combined')

# %%
scale_cls_by_sample(result_t1d, 'sliding_window_square')
scale_cls_by_sample(result_nd, "sliding_window_square")

# %%
GROUP = 'cell_type_coarse'
fig = plot_adata_grouped_heatmaps(
        adata_dict={'ND': result_nd, 'T1D': result_t1d},
        group_by=GROUP,
        #value_cols=['combined_cls_horizontal_scaled', 'combined_cls_vertical_scaled'],
    agg_func='mean',
        figsize=(10, 8),
        save_path=f"{HPC}/InterScale_reproducibility/figures/{DATA}/graph_class/cls_{GROUP}.png"
    )

# %%
# assign cell type colors
color_list = [CELL_TYPE_COLORS[ct] for ct in adata.obs[GROUP].cat.categories]
result_t1d.uns[f'{GROUP}_colors'] = color_list
result_nd.uns[f'{GROUP}_colors'] = color_list

# %%
image_nr_long = np.unique(result_t1d.obs['slide_fov'][result_t1d.obs['split'] == 'test'])
image_subset = list(np.random.choice(image_nr_long, 2))
sq.pl.spatial_scatter(result_t1d[result_t1d.obs['split'] == 'test'], 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', GROUP],
                      library_key = 'slide_fov',
                      library_id = image_subset,
                    cmap = PALETTE,
                    shape= None,
                      ncols = 3,
                      size = 50,
                      save = f"{HPC}/InterScale_reproducibility/figures/{DATA}/graph_class/"
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
