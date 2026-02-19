# %% [markdown]
# # Node classification sweep performance 
#
# Plots showing the performance for the node classification (cell type) on the CosMx Pancrease data with no diabetes and Type 1 Diabetes conditions.
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
import os

with open(os.path.join(HPC, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

# %% [markdown]
# ## Load WandB IDs

# %%
GNN_sweep = "nnd5tq6w"
InterScale_sweep = "ppierz1l"
PCATrans_sweep = ""

# %% [markdown]
# ## Graph Classification performance

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ['Alpha', 'Acinar', 'Beta', 'Ductal', 'Endocrine', 'Immune', 'Mast', 'Beta', 'Endothelial']

# %% [markdown]
# Q: Why does PCATrans not have test_f1 scores?

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, SWEEP_GOAL, CLASSES)
# PCATrans_wandb = Wandb_evaluation("PCATransformer", PCATrans_sweep, SWEEP_GOAL, CLASSES)
# InterScale_wandb = Wandb_evaluation("InterScale", InterScale_sweep, SWEEP_GOAL, CLASSES)

# %%
GNN_wandb.get_dataframe().head()

# %%
PCATrans_wandb.get_mean_and_std()

# %%
GNN_wandb.plot_robustness(metric="test_f1_class_Acinar", save_path = f"{HPC}/InterScale_reproducibility/figures/melton25/graph_class")

# %%
PCATrans_wandb.plot_robustness(metric="test_acc", save_path = f"{HPC}/InterScale_reproducibility/figures/melton25/graph_class")

# %%
InterScale_wandb.plot_robustness(metric="test_acc", save_path = f"{HPC}/InterScale_reproducibility/figures/melton25/graph_class")

# %%
wandb_evals = [GNN_wandb]  # Your list of dataframes

g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius=60,
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
