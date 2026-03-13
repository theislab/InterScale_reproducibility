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
project_root = Path(f'{BASE_DIR_REPO}/InterScale_reproducibility')
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
from InterScale.config import load_config_from_yaml, config_from_wandb_run

# utility function imports
from src.wandb import Wandb_evaluation, plot_f1_across_seeds, plot_class_f1_comparison, set_plot_configs
from src.plots import Plotting
from src.utils import set_full_reproducibility

# %%
import yaml
import matplotlib.pyplot as plt
import scanpy as sc

config_path = os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml")

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

# %%
df = Dual_wandb.filter_runs(exclude_parameters = {"pct_mask_nodes":[0.0]})
df.head()

# %%
df

# %%
df[df['test_acc'] > 0.55]

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, "data/melton25.h5ad"))

# %%
Dual_wandb.export_config_to_yaml(best_run_id="jaokfets", save_path=f"{BASE_DIR_REPO}/InterScale_reproducibility/best_config.yaml")

# %% [markdown]
# ## 1. Robustness sweep
#
# Run robustness sweep with config.yaml from best model. 

# %% [markdown]
# ### Load WandB IDs

# %%
GNN_sweep = "8zpa634k"
DualInterScale_sweep = "mg6bc24j"
PCATrans_sweep = "0vn9y1n5"

# %% [markdown]
# ## Graph Classification performance

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6', 'White Matter']

# %% [markdown]
# Q: Why does PCATrans not have test_f1 scores?

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, True, False, SWEEP_GOAL, CLASSES)
PCATrans_wandb = Wandb_evaluation("PCATransformer", PCATrans_sweep, False, True, SWEEP_GOAL, CLASSES)
DualInterScale_wandb = Wandb_evaluation("DualInterScale", DualInterScale_sweep, True, True, SWEEP_GOAL, CLASSES)

# %%
GNN_wandb.get_dataframe().head()

# %%
GNN_wandb.plot_robustness(metric="test_acc", save_path = f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %%
DualInterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %%
PCATrans_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %%
DualInterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %% [markdown]
# ## F1 score performance

# %%
wandb_evals = [GNN_wandb, PCATrans_wandb, DualInterScale_wandb]  # Your list of dataframes, PCATrans_wandb, InterScale_wandb

# %% [markdown]
# ## Class robustness

# %%
g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius="None",
    pct_mask_nodes=0.5,
    BASE_DIR_REPO=BASE_DIR_REPO, 
    height=10,
    aspect=0.9,
    save_path=f'{DATA}/node_class/f1_comparison_models_radius'
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
                                   BASE_DIR_REPO=None, save_path=None, figsize=(8, 5)):
    """
    Plot overall performance for a specific metric across models (aggregated across seeds).
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
    metric : str
        Column name of the metric to plot (e.g., 'test_f1_macro', 'test_accuracy', 'test_loss')
    radius, pct_mask_nodes : filter values (optional)
    BASE_DIR_REPO : str
    save_path : str, optional
    figsize : tuple
    
    Returns:
    --------
    fig, ax, stats_df
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)
    
    all_data = []
    
    for wandb_eval in wandb_evaluations:
        df = wandb_eval.df.copy()
        model_name = wandb_eval.model
        
        if radius is not None:
            df = df[df['radius'] == radius]
        if pct_mask_nodes is not None:
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
    
    # Replace the sns.barplot block with:
    models = stats_df['model'].tolist()
    means = stats_df['mean'].tolist()
    stds = stats_df['std'].tolist()
    colors = [palette.get(m, None) for m in models]
    
    x = np.arange(len(models))
    bar_width = 0.5
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
            os.path.join(BASE_DIR_REPO, 'InterScale_reproducibility/figures/', f'{save_path}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    return fig, ax, stats_df


# %%
import numpy as np

fig, ax, stats = plot_overall_metric_comparison(
    wandb_evals,
    metric='test_acc',  # or 'test_accuracy', 'test_balanced_accuracy', etc.
    radius="None",
    pct_mask_nodes=0.1,
    BASE_DIR_REPO=BASE_DIR_REPO,
    save_path=f'{DATA}/overall_f1_macro_comparison',
    figsize=(5, 5)
)

# %%
