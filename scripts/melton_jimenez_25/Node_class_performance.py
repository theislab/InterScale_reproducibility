# %% [markdown]
# # Node classification sweep performance - Cell type prediction
#
# CosMx Pancrease data with cell types.
#
# **Part 1: Hyperparmeter sweep**
#
# Run hyperparameter sweep with script `hyperparameter.yaml` and config file: 
#
# **Part 2: Robustness sweep**
#
# Plots showing the performance for the graph classification on the CosMx Pancrease data with no diabetes and Type 1 Diabetes conditions.
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
from InterScale.config import load_config_from_yaml, config_from_wandb_run

# %%
import yaml
import matplotlib.pyplot as plt
import scanpy as sc

with open(os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

# %% [markdown]
# ## Hyperparameter sweep
#
# Evaluation of node-classification (Cell type) hyperparameter sweep.

# %%
CLASSES = ['Alpha', 'Acinar', 'Beta', 'Ductal', 'Endocrine', 'Immune', 'Mast', 'Beta', 'Endothelial']

# %% [markdown]
# ### Load WandB IDs

# %%
GNN_sweep = "tixesx9b"
Dual_sweep = "d9lf50og"

# %%
SWEEP_GOAL = 'hyperparameter'

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, True, False, SWEEP_GOAL, CLASSES)
Dual_wandb = Wandb_evaluation("InterScale", Dual_sweep, True, True, SWEEP_GOAL, CLASSES)

# %%
GNN_wandb.get_dataframe().head()

# %%
GNN_wandb.plot_parameter_space(metric = 'test_acc', relevant_params = GNN_wandb.hyperparameters, exclude_parameters = {"pct_mask_nodes":[0.0]}, save_path = f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %%
relevant_params = Dual_wandb.hyperparameters + Dual_wandb.local_component_params + Dual_wandb.global_component_params
Dual_wandb.plot_parameter_space(metric = 'test_acc', relevant_params = relevant_params, exclude_parameters = {"pct_mask_nodes":[0.0]})

# %% [markdown]
# ### Best model selection

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
# ## 2. Robustness sweep
#
# Run robustness sweep with config.yaml from best model. 

# %% [markdown]
# ### Load WandB IDs

# %%
GNN_sweep = "l0jetvrz"
InterScale_sweep = "yqvyut7l"
DualInterScale_sweep = "1nm8ndul"
PCATrans_sweep = "8govw06t"

# %% [markdown]
# ## Graph Classification performance

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ['Alpha', 'Acinar', 'Beta', 'Ductal', 'Endocrine', 'Immune', 'Mast', 'Beta', 'Endothelial']

# %% [markdown]
# Q: Why does PCATrans not have test_f1 scores?

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, True, False, SWEEP_GOAL, CLASSES)
PCATrans_wandb = Wandb_evaluation("PCATransformer", PCATrans_sweep, False, True, SWEEP_GOAL, CLASSES)
InterScale_wandb = Wandb_evaluation("InterScale", InterScale_sweep, True, True, SWEEP_GOAL, CLASSES)
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
InterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class")

# %% [markdown]
# ## F1 score performance

# %%
wandb_evals = [GNN_wandb, PCATrans_wandb, DualInterScale_wandb]  # Your list of dataframes, PCATrans_wandb, InterScale_wandb

# %% [markdown]
# ## Class robustness

# %%
g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius=60,
    pct_mask_nodes=0.0,
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

# %%
from src.wandb import plot_class_f1_robustness

# %%
for class_id in CLASSES: 
    fig, ax, stats = plot_class_f1_robustness(
        wandb_evals,
        class_idx=class_id,
        pct_mask_nodes=0.0,
        BASE_DIR_REPO=BASE_DIR_REPO, 
        save_path=f"{BASE_DIR_REPO}/InterScale_reproducibility/figures/{DATA}/node_class"
    )

# %%
