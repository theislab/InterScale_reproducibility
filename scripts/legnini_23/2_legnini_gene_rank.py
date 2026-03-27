# %% [markdown]
# # Legnini - Gene rank analysis
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
from InterScale.evaluation.gene_rank_analysis import gene_rank_analysis, gene_rank_condition_comparison
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

DATA = "legnini23"

# %%
# path on ICB or LRZ cluster to InterScale_reproducibility folder
CFG_CLASS = os.path.join(BASE_DIR_REPO, f"GT-long-range-niches/src/config_files/{DATA}/legnini23_regr_node_DualCombined_genes.yaml")
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
CONDITION_COLORS = config["palettes"][DATA]["condition"]

# %%
from figures.scripts.plots import Plotting

plotting = Plotting(config_path)
plotting._setup_plotting_params()

# %%
CONDITION_KEY = 'condition'

# %% [markdown]
# ## Load config and model

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, "data", f"{DATA}.h5ad"))
adata

# %%
np.unique(adata.obs['condition'])

# %% [markdown]
# ### Local source

# %%
cfg = load_config(CFG_CLASS)

# %%
assert BASE_DIR_PROJECT in cfg.model.save 
assert BASE_DIR_PROJECT in cfg.dataset.h5ad_data

# %%
# assign cell type colors
color_list = [CONDITION_COLORS[ct] for ct in adata.obs[CONDITION_KEY].cat.categories]
adata.uns[f'{CONDITION_COLORS}_colors'] = color_list

# %%
#interscale.model.LocalModel._setup_anndata(adata = adata, prediction_task = cfg.dataset.prediction_task, layer_key = cfg.dataset.layer_key, sample_key_list = cfg.dataset.sample_key, prediction_obs =  cfg.dataset.prediction_obs, group_key = cfg.dataset.group_label, view_registry = False)
#local_model = interscale.model.LocalModel.load(RESULTS_DIR, adata, cfg, local_component = True, global_component = False, wandb_save = True)
interscale.model.CombinedModel._setup_anndata(adata = adata, prediction_task = cfg.dataset.prediction_task, layer_key = cfg.dataset.layer_key, sample_key_list = cfg.dataset.sample_key, prediction_obs =  cfg.dataset.prediction_obs, group_key = cfg.dataset.group_label, view_registry = False)
combined_model = interscale.model.CombinedModel.load(RESULTS_DIR, adata, cfg, local_component = True, global_component = True, wandb_save = True)

# %% [markdown]
# ## Inference: get model output 

# %%
slide_ids = list(config["slide_examples"][DATA].values())

# %%
sub_adata = adata[adata.obs['sample'].isin(slide_ids)]

# %%
result = combined_model.get_model_output(sub_adata, prefix = 'combined')

# %%
result

# %%
result_complete = combined_model.get_model_output(adata, prefix = 'combined')

# %%
assert result_complete.X[:10, :10].dtype == np.float32

# %% [markdown]
# ### Condition-based gene-rank analysis

# %%
gene_rank_analysis(result,
                   layers_local_pred = 'combined_y_pred_local',
                   layers_global_pred = 'combined_y_pred_global',
                  color_dict=CONDITION_COLORS,
                  save_dir=FIGURE_DIR)

# %%
gene_rank_condition_comparison(result_complete,
                               library_key = "condition",
                               color_dict = CONDITION_COLORS,
                   layers_local_pred = 'combined_y_pred_local',
                   layers_global_pred = 'combined_y_pred_global')

# %%
sq.pl.spatial_scatter(result_complete, 
                      color = ['SIX6', 'BMP4'],
                      library_key = 'sample',
                      library_id = slide_ids,
                       ncols=3,
                      cmap="viridis",
                    shape = None)

# %% [markdown]
# ## Can we interpret the CLS token on regression trained model?
#
# First, scale the CLS token per sliding window so that they are comparable despite their different number of cells. 

# %%
result_ctrl = result[result.obs['condition'] == 'Ctrl']
scale_cls_by_sample(result_ctrl, "sample")
result_shh = result[result.obs['condition'] == 'SHH']
scale_cls_by_sample(result_shh, "sample")

# %%
library_key = "cell_type_coarse"

data_shh = result_shh.obs.groupby(library_key).agg(
    mean_cls_horizontal=('combined_cls_horizontal_scaled', 'mean'),
    mean_cls_vertical=('combined_cls_vertical_scaled', 'mean')
)

data_ctrl = result_ctrl.obs.groupby(library_key).agg(
    mean_cls_horizontal=('combined_cls_horizontal_scaled', 'mean'),
    mean_cls_vertical=('combined_cls_vertical_scaled', 'mean')
)

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(6, 6))

# Get min/max for shared color scale
vmin = min(data_shh.values.min(), data_ctrl.values.min())
vmax = max(data_shh.values.max(), data_ctrl.values.max())

# Plot with shared color scale
sns.heatmap(data_t1d, annot=True, linewidth=.5, fmt='.2f', 
            ax=axes[0], vmin=vmin, vmax=vmax, cbar=False, cmap = PALETTE)
axes[0].set_title('T1D')

sns.heatmap(data_nd, annot=True, linewidth=.5, fmt='.2f', 
            ax=axes[1], vmin=vmin, vmax=vmax, cmap = PALETTE)
axes[1].set_title('ND')

plt.tight_layout()
fig.savefig(os.path.join(FIGURE_DIR, f"heatmap_{library_key}.png"), dpi=300, bbox_inches='tight')
plt.show()

# %%
slide_nr = np.unique(result_nd.obs['sample'])[0]
sq.pl.spatial_scatter(result_nd, 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled'],
                    cmap = PALETTE,
                    shape= None,
                      ncols = 3,
                      save = os.path.join(FIGURE_DIR, f"spatial_cls_nd_{slide_nr}.png")
)

# %%
slide_nr = np.unique(result_t1d.obs['sample'])[0]
sq.pl.spatial_scatter(result_t1d, 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled'],
                   cmap = PALETTE,
                    shape= None,
                      ncols = 3,
                      save = os.path.join(FIGURE_DIR, f"spatial_cls_t1d_{slide_nr}.png")
)

# %%
