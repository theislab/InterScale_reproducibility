# %% [markdown]
# # Node classification sweep performance 
#
# Plots showing the performance for the graph classification on the CosMx Pancrease data with no diabetes and Type 1 Diabetes conditions.
#
# - InterScale (GCN + Transformer)
# - GCN
# - PCATransformer

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
from src.wandb import Wandb_evaluation, plot_f1_across_seeds, plot_class_f1_comparison, set_plot_configs

# %%
import yaml
import matplotlib.pyplot as plt
import os

with open(os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

# %% [markdown]
# ## Load WandB IDs

# %%
GNN_sweep = "l0jetvrz"
InterScale_sweep = "yqvyut7l"
PCATrans_sweep = "8govw06t"

# %% [markdown]
# ## Graph Classification performance

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ['Alpha', 'Acinar', 'Beta', 'Ductal', 'Endocrine', 'Immune', 'Mast', 'Beta', 'Endothelial']

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
GNN_wandb.plot_robustness(metric="test_acc", save_path = f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %%
PCATrans_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %%
InterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %% [markdown]
# ## F1 score performance

# %%
wandb_evals = [GNN_wandb, PCATrans_wandb, InterScale_wandb]  # Your list of dataframes, PCATrans_wandb, InterScale_wandb

# %%

# %%

g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius=60,
    pct_mask_nodes=0.25,
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
# ## Class robustness

# %%
def plot_class_f1_robustness(wandb_evaluations, class_idx, pct_mask_nodes=None, BASE_DIR_REPO=None, y_max='auto',
                              save_path=None, figsize=(8, 6)):
    """
    Plot class-specific F1 score robustness across radius, comparing multiple models.
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
        List of Wandb_evaluation instances, one per model
    class_idx : str or int
        Class index to plot F1 for
    pct_mask_nodes : float or int, optional
        Percentage of masked nodes to filter by (if None, uses all data)
    BASE_DIR_REPO : str, optional
        Base directory for config files
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (default: (8, 6))
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    stats_df : pd.DataFrame with mean and std for each radius and model
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)
    
    all_data = []
    
    for wandb_eval in wandb_evaluations:
        df = wandb_eval.df.copy()
        model_name = wandb_eval.model
        
        # Apply filter if specified
        if pct_mask_nodes is not None:
            df = df[df['pct_mask_nodes'] == pct_mask_nodes]
        
        if df.empty:
            print(f"Warning: No data for {model_name} with specified filters")
            continue
        
        # Get F1 column for specified class
        f1_col = f'test_f1_class_{class_idx}'
        
        if f1_col not in df.columns:
            print(f"Warning: {f1_col} not found for {model_name}")
            continue
        
        plot_df = df[['radius', 'seed', f1_col]].copy() if 'seed' in df.columns else df[['radius', f1_col]].copy()
        plot_df = plot_df.rename(columns={f1_col: 'f1_score'})
        plot_df['model'] = model_name
        all_data.append(plot_df)
    
    if not all_data:
        raise ValueError("No valid data found for any model")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate statistics
    stats_df = combined_df.groupby(['model', 'radius'])['f1_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.lineplot(
        data=combined_df,
        x='radius',
        y='f1_score',
        hue='model',
        marker='o',
        errorbar='sd',
        palette=model_palette,
        ax=ax
    )
    
    ax.set_xlabel('Radius', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    
    title = f'Class {class_idx} F1 Robustness Across Models'
    if pct_mask_nodes is not None:
        title += f' (pct_mask={pct_mask_nodes})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set y-axis limits
    if y_max == 'auto':
        y_max_val = combined_df['f1_score'].max() + 0.2
    else:
        y_max_val = y_max
    ax.set_ylim(0, y_max_val)
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(
            os.path.join(save_path, f'f1_robustness_class_{class_idx}_{pct_mask_nodes}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    print("\nStatistics Summary:")
    print(stats_df.to_string(index=False))
    
    return fig, ax, stats_df


# %%
for class_id in CLASSES: 
    fig, ax, stats = plot_class_f1_robustness(
        wandb_evals,
        class_idx=class_id,
        pct_mask_nodes=0.1,
        BASE_DIR_REPO=BASE_DIR_REPO, 
        save_path=f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class"
    )

# %%

# %%
fig, ax, stats = plot_class_f1_robustness(
        wandb_evals,
        class_indices=CLASSES,
        pct_mask_nodes=0.1,
        BASE_DIR_REPO=BASE_DIR_REPO, 
        save_path=f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class",
        n_cols = 3)

# %%
