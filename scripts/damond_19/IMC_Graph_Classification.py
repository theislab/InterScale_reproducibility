# %% [markdown]
# # IMC Pancreas - Graph classificiation evaluation

# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd


import InterScale as interscale
from InterScale.config import load_config
from InterScale.tl import prepare_geome_dataset, check_and_update_cfg
from InterScale.geome_dataloader import GraphAnnDataModule
#from InterScale.eval.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
from InterScale.tl import prepare_a2d_dataset
from InterScale.evaluation import scale_cls_by_sample

from pathlib import Path
import torch

# %%
ICB = "/home/icb/francesca.drummer/1-Projects/"
LRZ = "/dss/dsshome1/05/di93tig/1_projects/" 

CFG_CLASS = f"{ICB}GT-long-range-niches/src/config_files/damond19/damond19_class_graph_Combined_condition.yaml"

RESULTS_DIR = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/results/damond19/"

# %% [markdown]
# ## Load pretrained models

# %%
cfg = load_config(CFG_CLASS)

# %%
adata = sc.read_h5ad(cfg.dataset.h5ad_data)
adata

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
result_t1d = combined_model.get_model_output(adata[adata.obs['condition'] == 'T1D'], prefix = 'combined')

# %%
result_nd = combined_model.get_model_output(adata[adata.obs['condition'] == 'ND'], prefix = 'combined')

# %%
result_nd.obsm['combined_y_pred']

# %%
result_nd

# %%
from sklearn.metrics import roc_auc_score, roc_curve

# %% [markdown]
# ## Evaluation score

# %%
from InterScale.evaluation import calculate_pr_auc

# %%
calculate_pr_auc(result_t1d)

# %%
result_nd

# %% [markdown]
# ## Graph Label analysis
#
# First, we scale the CLS token and then calculate the average mean expression of the CLS (horizontal and vertical) token per cell type.

# %%
from InterScale.evaluation.graph_classification import scale_cls_by_sample
scale_cls_by_sample(result_nd, "slide_fov")
scale_cls_by_sample(result_t1d, "slide_fov")

# %%
data = result_t1d.obs.groupby('cell_type_coarse').agg(
    mean_cls_horizontal=('combined_cls_horizontal_scaled', 'mean'),
    mean_cls_vertical=('combined_cls_vertical_scaled', 'mean')
)

plt.figure(figsize=(8, 6))  # Reduce width (first number) to make columns thinner
sns.heatmap(data, annot=True, linewidth=.5)
plt.show()

# %%
import seaborn as sns

data = result_nd.obs.groupby('cell_type_coarse').agg(
    mean_cls_horizontal=('combined_cls_horizontal_scaled', 'mean'),
    mean_cls_vertical=('combined_cls_vertical_scaled', 'mean')
)
sns.heatmap(data, annot=True)

# %%
canpy.pl.heatmap(result_nd, var_names, groupby,

# %%
np.unique(result_t1d[result_t1d.obs['split'] == 'test'].obs['slide_fov'])

# %%
sq.pl.spatial_scatter(result_t1d[result_t1d.obs['split'] == 'test'], 
                      color = ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'cell_type_coarse'],
                      library_key = 'slide_fov',
                      library_id = ['3_12', '3_13'],
                    cmap = 'viridis_r',
                    shape= None,
                      ncols = 3,
)

# %%
np.unique(result_nd[result_nd.obs['split'] == 'test'].obs['slide_fov'])

# %%
sq.pl.spatial_scatter(result_nd[result_nd.obs['split'] == 'test'], 
                      color =  ['combined_cls_vertical_scaled', 'combined_cls_horizontal_scaled', 'cell_type_coarse'],
                      library_key = 'slide_fov',
                      library_id = ['3_15', '3_17'],
                    cmap = 'viridis_r',
                    shape= None,
                      ncols = 3,
)

# %%
