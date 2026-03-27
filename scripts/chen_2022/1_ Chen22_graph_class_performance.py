# %% [markdown]
# # Graph classification sweep performance - Region label prediction
#
# Chen et al, 2022 data with condition labes (Mid-AD and Control)
#
# **Part 1: Hyperparam sweep**
#
# Plots showing the performance for the region label classification.
#
# **Part 2: Graph label prediction performance**
#
# - InterScale (GCN + Transformer)
# - GCN
# - PCATransformer
#
#

# %%
import os

# LRZ home
if os.path.exists("/dss/dsshome1/05/di93tig"):
    print('LRZ cluster')
    CLUSTER = 'LRZ'
    BASE_DIR_REPO = "/dss/dsshome1/05/di93tig/1_projects/InterScale_reproducibility" 
    BASE_DIR_PROJECT = "/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale"
elif os.path.exists("/home/icb/francesca.drummer/"):
    print('HPC cluster')
    CLUSTER = 'HPC'
    BASE_DIR_REPO = "/home/icb/francesca.drummer/1-Projects/"
    BASE_DIR_PROJECT = ""
else:
    print('unkown')
    CLUSTER = 'unknown'
DATA = "chen22"

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from notebook location)
project_root = Path(f'{BASE_DIR_REPO}')
print(project_root)
sys.path.insert(0, str(project_root))

# %%
import wandb
wandb.login()

# %%
import pandas as pd
# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
# InterScale imports
from InterScale.config import load_config, config_from_wandb_run

# utility function imports
from src.wandb import Wandb_evaluation, plot_f1_across_seeds, plot_class_f1_comparison, set_plot_configs
from src.plots import Plotting
from src.utils import set_full_reproducibility

# %%
import yaml
import matplotlib.pyplot as plt
import scanpy as sc

config_path = os.path.join(BASE_DIR_REPO, "figures/config.yml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

from figures.scripts.plots import Plotting

plotting = Plotting(config_path)
plotting._setup_plotting_params()

# %% [markdown]
# ## Hyperparameter sweep
#
# Evaluation of node-classification (Cell type) hyperparameter sweep.

# %%
InterScale_wandb = "3pfj292b"
SWEEP_GOAL = "graph"
CLASSES = ["control", "Mid-AD"]

# %%
InterScale_sweep = Wandb_evaluation("InterScale", InterScale_wandb, True, True, SWEEP_GOAL, CLASSES)

# %%
df = InterScale_sweep.filter_runs(exclude_parameters = {})
df.head()

# %%
df[df['test_f1_micro'] > 0.55]

# %%
df[(df['test_f1_class_control'] > 0.55) & (df['test_f1_class_Mid-AD'] > 0.55)].sort_values('test_f1_macro', ascending=False)

# %%
InterScale_sweep.export_config_to_yaml(best_run_id="waj87gc6", save_path=f"{BASE_DIR_REPO}/chen22_best_config.yaml")

# %% [markdown]
# ## 1. Robustness sweep
#
# Run robustness sweep with config.yaml from best model. 

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, "data/melton25.h5ad"))

# %% [markdown]
# ### Load WandB IDs

# %%
GNN_sweep = "39pecmnt"
InterScale_sweep = "3pfj292b"
PCATrans_sweep = "tohmr60x"

# %% [markdown]
# ## Graph Classification performance

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ["control", "Mid-AD"]

# %% [markdown]
# Q: Why does PCATrans not have test_f1 scores?

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, True, False, SWEEP_GOAL, CLASSES)
PCATrans_wandb = Wandb_evaluation("PCATransformer", PCATrans_sweep, False, True, SWEEP_GOAL, CLASSES)
InterScale_wandb = Wandb_evaluation("InterScale", InterScale_sweep, True, True, SWEEP_GOAL, CLASSES)

# %%
GNN_wandb.get_mean_and_std()

# %%
InterScale_wandb

# %%
GNN_wandb.plot_robustness(metric="test_acc", save_path = f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
InterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/figures/{DATA}/graph_class")

# %%
PCATrans_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %% [markdown]
# ## F1 score performance

# %%
#wandb_evals = [GNN_wandb, PCATrans_wandb, DualInterScale_wandb]  # Your list of dataframes, PCATrans_wandb, InterScale_wandb
wandb_evals = [GNN_wandb, InterScale_wandb, PCATrans_wandb] 


# %%

# %%
def plot_f1_across_seeds(wandb_evaluations, 
                         radius, 
                         pct_mask_nodes, 
                         BASE_DIR_REPO, 
                         height=4, 
                         aspect=0.7,
                         save_path=None,
                         dropna=True):
    """
    Plot mean and standard deviation of per-class F1 scores across seeds.
    Uses seaborn catplot with one facet per class and one bar per model.
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
        List of Wandb_evaluation instances, one per model
    radius : float or int or None
        Radius value to filter by
    pct_mask_nodes : float or int or None
        Percentage of masked nodes to filter by
    config_path : str, optional
        Path to config.yml file (default: 'config.yml')
    height : float, optional
        Height of each facet in inches (default: 4)
    aspect : float, optional
        Aspect ratio of each facet (default: 0.7)
    save_path: str
        Path name from repo to save the plot.
    dropna : bool, optional
        Whether to drop rows with NaN in 'radius' or 'pct_mask_nodes' (default: True).
        Set to False if radius or pct_mask_nodes is None.
    
    Returns:
    --------
    g : seaborn FacetGrid
        The catplot object
    stats_df : pd.DataFrame
        DataFrame with mean and std for each class and model
    plot_data : pd.DataFrame
        Long-form data used for plotting
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)

    # Process each Wandb_evaluation instance
    all_data = []
    model_names = []
    
    for wandb_eval in wandb_evaluations:
        model_name = wandb_eval.model
        classes = wandb_eval.classes
        
        model_names.append(model_name)
        
        df = wandb_eval.df.dropna(subset=["radius", "pct_mask_nodes"]) if dropna else wandb_eval.df

        # Filter by specified parameters
        if radius is None:
            mask_radius = df['radius'].isna()
        else:
            mask_radius = df['radius'] == radius

        if pct_mask_nodes is None:
            mask_pct = df['pct_mask_nodes'].isna()
        else:
            mask_pct = df['pct_mask_nodes'] == pct_mask_nodes

        df_filtered = df[mask_radius & mask_pct].copy()
        
        if df_filtered.empty:
            print(f"Warning: No data found for {model_name} with radius={radius}, pct_mask_nodes={pct_mask_nodes}")
            continue
        
        # Identify F1 score columns
        f1_cols = [col for col in df_filtered.columns if col.startswith('test_f1_class_')]
        
        if not f1_cols:
            raise ValueError(f"No 'test_f1_class_*' columns found in {model_name}")
        
        # Add model label
        df_filtered['model'] = model_name
        
        # Collect the data
        all_data.append(df_filtered)
    
    if not all_data:
        raise ValueError("No data found matching the specified parameters")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Identify F1 columns
    f1_cols = [col for col in combined_df.columns if col.startswith('test_f1_class_')]
    
    # Reshape to long format for seaborn
    plot_data = combined_df.melt(
        id_vars=['model'],
        value_vars=f1_cols,
        var_name='class',
        value_name='f1_score'
    )
    
    # Clean up class names
    plot_data['class'] = plot_data['class'].str.replace('test_f1/class_', '')
    
    # Calculate statistics for reference
    stats_df = plot_data.groupby(['model', 'class'])['f1_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reset_index()

    print(stats_df)
    
    # Create color palette based on model names from config
    palette = []
    for model_name in model_names:
        if model_name in model_palette:
            palette.append(model_palette[model_name])
        else:
            print(f"Warning: Model '{model_name}' not found in config palette. Using default color.")
            palette.append(None)
    
    # If all models are in config, use the palette; otherwise let seaborn handle it
    use_palette = palette if all(c is not None for c in palette) else None
    
    # Create the catplot
    g = sns.catplot(
        data=plot_data,
        kind="bar",
        x="class",
        y="f1_score",
        hue="model",
        height=height,
        aspect=aspect,
        errorbar="sd",
        capsize=0.1,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.8,
        palette=use_palette,
        legend=False
    )
    
    # Customize the plot with config settings
    g.set_axis_labels(
        "Model", 
        "Test F1 Score", 
        fontsize=general_config['legend_fontsize'],
        fontweight=general_config['legend_fontweight']
    )
    g.set_titles(
        "Class: {col_name}", 
        fontsize=general_config['title_fontsize'],
        fontweight=general_config['title_fontweight']
    )
    
    # Set y-axis limits and grid
    for ax in g.axes.flat:
        ax.set_ylim([0, 1.2])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Rotate x-axis labels if needed
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    
    # Overall title
    g.fig.suptitle(
        f'F1 Scores Across Seeds (radius={radius}, pct_mask={pct_mask_nodes})',
        fontsize=general_config['title_fontsize'],
        fontweight=general_config['title_fontweight'],
        y=1.02
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            os.path.join(BASE_DIR_REPO, 'figures/', f'{save_path}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    return g, stats_df, plot_data


# %% [markdown]
# ## Class robustness

# %%
r = None
pct_mask = 0.5

g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius=r,
    pct_mask_nodes=pct_mask,
    BASE_DIR_REPO=BASE_DIR_REPO, 
    height=10,
    aspect=0.9,
    save_path=f'{DATA}/graph_class/f1_comparison_models_{r}_{pct_mask}',
    dropna = False
)
plt.show()
#
# # Save with high DPI from config
# # g.savefig('f1_scores.png', bbox_inches='tight')
#
# # Print statistics:
# print(stats)

# %% [markdown]
# ## Overall performance
#
# How do the models perform overall? PLot Macro F1, Micro F1 & Accuracy.

# %%
def plot_overall_metric_comparison(wandb_evaluations, metric, radius=None, pct_mask_nodes=None, 
                                   BASE_DIR_REPO=None, save_path=None, figsize=(8, 5), dropna=True):
    """
    Plot overall performance for a specific metric across models (aggregated across seeds).
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
    metric : str
        Column name of the metric to plot (e.g., 'test_f1_macro', 'test_accuracy', 'test_loss')
    radius, pct_mask_nodes : filter values (optional, use None to match NaN rows)
    BASE_DIR_REPO : str
    save_path : str, optional
    figsize : tuple
    dropna : bool, optional
        Whether to drop rows with NaN in 'radius' or 'pct_mask_nodes' (default: True).
        Set to False if radius or pct_mask_nodes is None.
    
    Returns:
    --------
    fig, ax, stats_df
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)
    
    all_data = []
    
    for wandb_eval in wandb_evaluations:
        model_name = wandb_eval.model
        df = wandb_eval.df.dropna(subset=["radius", "pct_mask_nodes"]) if dropna else wandb_eval.df.copy()

        # Filter by radius
        if radius is None:
            df = df[df['radius'].isna()]
        else:
            df = df[df['radius'] == radius]

        # Filter by pct_mask_nodes
        if pct_mask_nodes is None:
            df = df[df['pct_mask_nodes'].isna()]
        else:
            df = df[df['pct_mask_nodes'] == pct_mask_nodes]
        
        if df.empty:
            print(f"Warning: No data for {model_name} with specified filters")
            continue
        
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found for {model_name}")
            continue
        
        plot_df = df[['seed', metric]].copy() if 'seed' in df.columns else df[[metric]].copy()
        plot_df['model'] = model_name
        all_data.append(plot_df)
    
    if not all_data:
        raise ValueError("No valid data found for any model")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Statistics
    stats_df = combined_df.groupby('model')[metric].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reset_index()
    print(stats_df)
    
    # Build palette
    palette = {m: model_palette[m] for m in combined_df['model'].unique() if m in model_palette}
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = stats_df['model'].tolist()
    means = stats_df['mean'].tolist()
    stds = stats_df['std'].tolist()
    colors = [palette.get(m, None) for m in models]
    
    x = np.arange(len(models))
    bar_width = 0.6
    ax.set_xlim(-0.4, len(models) - 0.4)
    
    ax.bar(x, means, yerr=stds, width=bar_width, color=colors,
           edgecolor='black', linewidth=1.0, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    metric_label = metric.replace('_', ' ').replace('test ', '').title()
    ax.set_xlabel('Model', fontsize=general_config['legend_fontsize'], 
                  fontweight=general_config['legend_fontweight'])
    ax.set_ylabel(metric_label, fontsize=general_config['legend_fontsize'], 
                  fontweight=general_config['legend_fontweight'])
    ax.set_title(f'{metric_label} Across Seeds (radius={radius}, pct_mask={pct_mask_nodes})',
                 fontsize=general_config['title_fontsize'], 
                 fontweight=general_config['title_fontweight'])
    
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(
            os.path.join(BASE_DIR_REPO, 'figures/', f'{save_path}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    return fig, ax, stats_df


# %%
import numpy as np

fig, ax, stats = plot_overall_metric_comparison(
    wandb_evals,
    metric='test_acc',  # or 'test_accuracy', 'test_balanced_accuracy', etc.
    radius=None,
    pct_mask_nodes=0.5,
    BASE_DIR_REPO=BASE_DIR_REPO,
    save_path=f'{DATA}/graph_class/overall_f1_comparison_models_{r}_{pct_mask}',
    figsize=(3, 5), 
   dropna=False
)

# %%
