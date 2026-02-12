# %% [markdown]
# # Setup and download data
#
# This tutorials shows how to set up an multiple slide Alzheimer Disease Visium dataset from [Chen et al., 2022](https://pubmed.ncbi.nlm.nih.gov/36544231/). To follow along with this and the following tutorials, please execute the following steps first:
#
# - Set up InterScale environment (see instructions in ReadMe)
# - Load raw data from GEO accession (GSE220442) with [SpatialData loader](https://spatialdata.scverse.org/projects/io/en/latest/generated/spatialdata_io.visium.html).

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
import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import issparse
import seaborn as sns

from pathlib import Path

# %%
from pathlib import Path
import sys

# set INTERSCALE_DIR to the current working directory
INTERSCALE_DIR = Path.cwd()
print(INTERSCALE_DIR)
project_root = Path(f'{BASE_DIR_REPO}/InterScale_reproducibility')
print(project_root)
sys.path.insert(0, str(project_root))

# %%
from src.sliding_window import sliding_window

# %% [markdown]
# <mark>TODO: Change data path</mark>

# %%
CHEN_22 = os.path.join(BASE_DIR_PROJECT, 'data/chen_adata.h5ad')

# %% [markdown]
# ## Load data

# %%
adata = sc.read_h5ad(CHEN_22)
adata

# %%
library_ids = list(adata.uns['spatial'].keys())
titles = [
    f"{pid} - {adata.obs.loc[adata.obs['patient_id'] == pid, 'stage'].iloc[0]}"
    for pid in library_ids
]

sq.pl.spatial_scatter(
    adata,
    spatial_key='spatial',
    library_key='patient_id', 
    color="Layer",
    size=10,
    img=False,
    title=titles
)

# %% [markdown]
# ## 1. Normalization
#
# The data needs to be normalized for InterScale. Check if the data is already normalized: 

# %%
print(f'Min count: {adata.X.min()}, Max count: {adata.X.max()}')

# %% [markdown]
# The data is not yet normalized, so normalize and transform with log1p.

# %%
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
print(f'Min count: {adata.layers["log1p_norm"].min()}, Max count: {adata.layers["log1p_norm"].max()}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
p1 = sns.histplot(adata.X.sum(1), bins=100, kde=False, ax=axes[0])
axes[0].set_title("Total counts")
p2 = sns.histplot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[1])
axes[1].set_title("Shifted logarithm")
plt.show()

# %%
adata.layers['raw_counts'] = adata.X

# %%
# highly variable gene selection
sc.pp.highly_variable_genes(
    adata, flavor="seurat_v3", layer="raw_counts", n_top_genes=5000, subset=True
)

# %%
adata

# %% [markdown]
# ### Is there a reason for the bi-modal behaviour?

# %%
# Add a column marking which mode each cell belongs to
cutoff = 1500  # adjust based on your plot - looks like the valley is around 1500
adata.obs['count_mode'] = np.where(
    adata.layers["log1p_norm"].sum(1) < cutoff, 
    'low', 
    'high'
)

# Check the split
print(adata.obs['count_mode'].value_counts())

# %%
import pandas as pd

# List your categorical columns of interest
cat_cols = ['patient_id', 'stage', 'Layer']  # adjust to your .obs columns

fig, axes = plt.subplots(1, len(cat_cols), figsize=(4*len(cat_cols), 4))

for ax, col in zip(axes, cat_cols):
    # Cross-tabulation with percentages
    ct = pd.crosstab(adata.obs[col], adata.obs['count_mode'], normalize='index') * 100
    ct.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{col}')
    ax.set_ylabel('% of cells')
    ax.legend(title='Mode')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Calculate spatial connectivity matrix
#
# Use [`squidpy.gr.spatial_neighbors()`](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html)) to calculate the spatial connectivity. For Visium it is important to set `coord_type='grid'` and indicate the IDs per slide (e.i. `patient_id`)

# %%
np.unique(adata.obs['patient_id'])

# %%
sq.gr.spatial_neighbors(
    adata,
    coord_type = "grid",
    library_key = "patient_id"
)

# %%
sq.pl.spatial_scatter(
    adata,
    library_key='patient_id',
    library_id = '2-5',
    connectivity_key="spatial_connectivities",
    img=False,
    color=["Layer"]
)

# %% [markdown]
# ## 3. Optional: Calculate sliding windows
#
# Sliding windows are necessary in case the tissue slide contains more than 4k cells. First, check how many cells are at minimum or maximum in your dataset.

# %%
tissue_cell_number = adata.obs.groupby('patient_id').size()
print(f"Nr cells per sliding window: Min: {tissue_cell_number.min()}, Max: {tissue_cell_number.max()}, Avg: {tissue_cell_number.mean()}")

# %%
tissue_cell_number.plot(kind='bar', color='steelblue', edgecolor='black')

# %% [markdown]
# Instead of creating sliding windows we train per Brain Layer.

# %%
layer_cell_number = adata.obs.groupby(['patient_id', 'Layer']).size()
print(f"Nr cells per sliding window: Min: {tissue_cell_number.min()}, Max: {tissue_cell_number.max()}, Avg: {tissue_cell_number.mean()}")

# %%
layer_cell_number.plot(kind='bar', color='steelblue', edgecolor='black')

# %%
# remove noise
nr_cells_before = len(adata.obs_names)
adata = adata[adata.obs['Layer'] != 'Noise']
nr_cells_now = len(adata.obs_names)
print(f'Cells removed: {nr_cells_before-nr_cells_now}')

# %% [markdown]
# Create combined `patient_id` and layer label to use as `sample_id`. 

# %%
adata.obs['patientID_layer'] = adata.obs['patient_id'].astype(str) + '_' + adata.obs['Layer'].astype(str)
print(np.unique(adata.obs['patientID_layer']))

# %%
# Plot to see if it still makes sense
sq.pl.spatial_scatter(
    adata,
    spatial_key='spatial',
    library_key='patient_id', 
    color="Layer",
    size=10,
    img=False,
    title=titles
)

# %% [markdown]
# In addition to having the layers as `sample_id` to use as batch label we now also calculate sliding windows to split the data. 

# %%
sliding_window(adata, 
                   library_key = 'patient_id', 
                    partial_windows = "split",
                  window_size = 100, 
                  max_nr_cells = 2000, 
                   sliding_window_key = f"sliding_window_assignment"
                  )

# %%
# Plot to see if it still makes sense
sq.pl.spatial_scatter(
    adata,
    spatial_key='spatial',
    library_key='patient_id', 
    color="sliding_window_assignment",
    size=10,
    img=False,
    title=titles
)

# %%
layer_cell_number = adata.obs.groupby(['sliding_window_assignment']).size()
print(f"Nr cells per sliding window: Min: {tissue_cell_number.min()}, Max: {tissue_cell_number.max()}, Avg: {tissue_cell_number.mean()}")

# %% [markdown]
# We observe that there are a slides with only a few cells but 2 slides with more than 4k cells. For this case we have the option to 1) increase the context length to max = 4875 or 2) calculate sliding windows for all slides that have more than 3k cells. 

# %% [markdown]
# With the sliding windows the maximum number of cells per sliding window is `2099`. 
#
# <mark>TODO: Adjust max_seq_len in config file!</mark>
# Given this we adjust the `max_seq_len` from the model.global_component to `2099`. The default is `2000`.

# %% [markdown]
# ## 4. Split data into train and val set
#
# Training the model requires a `split` assignment for each donor/patient/sliding window that you wanna train on. 

# %%
import numpy as np
from sklearn.model_selection import train_test_split

# %%
split_map = {
    '1-1': 'train', '2-3': 'train',
    '18-64': 'val', '2-8': 'val',
    '2-5': 'test', 'T4857': 'test'
}
adata.obs['split'] = adata.obs['patient_id'].map(split_map)

# %%
split_map_1 = {
    '1-1': 'val', '2-3': 'train',
    '18-64': 'train', '2-8': 'val',
    '2-5': 'test', 'T4857': 'test'
}
adata.obs['split_1'] = adata.obs['patient_id'].map(split_map_1)

# %% [markdown]
# ## Save adata object
#
# Save the prepared adata object such that it can be loaded for the model training. 

# %%
adata.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/chen_adata_pp.h5ad')

# %% [markdown]
# ## Prepare config file
#
# Duplicate the minum requirement config file from `/GT-long-range-niches/src/config_files/InterScale_example.yaml` and add the necessary specification for the Visum data:
#
# ```
# model:
#   local_component:
#     name: GCN
#   global_component:
#     name: self-attn-transformer
#         max_seq_len: 2099
#   save: /path/to/save/model/
# dataset:
#   h5ad_data: chen_adata_pp.h5ad
#   name: Visium_chen_22
#   sample_key: ['patient_id']
#   spatial_neigbors_kwargs:
#     coord_type: grid
#     library_key: patient_id
# ```
#
#
# Save the config file as `.yaml` and proceed to training (either interactively in jupyter notebook or by running a script).

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, 'data/chen_adata_pp.h5ad'))
adata

# %%
print(adata.X[:10, :10])

# %%
sc.pp.pca(adata, svd_solver="arpack", use_highly_variable=False)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["patientID", "Layer", "stage", "nCount_Spatial"], cmap="viridis")

# %% [markdown]
# ### Highly-variable genes

# %%
sc.pp.highly_variable_genes(adata, flavor = 'seurat_v3')

# %%
sc.pp.pca(adata, svd_solver="arpack", key_added = "hvg_PCA", use_highly_variable=True)

# %%
sc.pp.neighbors(adata, use_rep = "hvg_PCA", key_added = "hvg_NN")
sc.tl.umap(adata, key_added="hvg_umap", neighbors_key='hvg_NN')

# %%
# Patch for compatibility
import matplotlib
import matplotlib.cm as cm

if not hasattr(matplotlib.colormaps, "get_cmap"):
    matplotlib.colormaps.get_cmap = cm.get_cmap

# %%
sc.pl.umap(adata, color=["patientID", "Layer", "nCount_Spatial"], cmap="viridis")

# %%
adata.var['highly_variable']

# %%
adata_hvg = adata[:,adata.var['highly_variable']]

# %%
adata_hvg

# %%
# save hvg adata object
adata_hvg.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/chen_adata_pp_hvg.h5ad')

# %% [markdown]
# Both highly variable and no highly variable gene selection look seperated by patientIDs. Need to integrate to remove batch effects.
#
# ### scVI Integration

# %%
sc.pp.filter_genes(adata_hvg, min_cells=1)

# %%
adata_hvg.X = adata_hvg.layers["raw_counts"].copy()
sc.pp.normalize_total(adata_hvg)
sc.pp.log1p(adata_hvg)
adata_hvg.layers["logcounts"] = adata_hvg.X.copy()

# %%
adata_scvi = adata_hvg.copy()

# %%
import scvi
scvi.model.SCVI.setup_anndata(adata_scvi, layer="raw_counts", batch_key="patientID")
adata_scvi

# %%
model_scvi = scvi.model.SCVI(adata_scvi)
model_scvi

# %%
max_epochs_scvi = np.min([round((20000 / adata_scvi.n_obs) * 400), 400])
max_epochs_scvi

# %%
model_scvi.train()

# %%
adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()

# %%
adata_scvi.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/chen_adata_pp_scvi.h5ad')

# %%
sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
sc.tl.umap(adata_scvi)
adata_scvi

# %%
sc.pl.umap(adata_scvi, color=["Layer", "patientID", "stage"], wspace=1)

# %% [markdown]
# Patient ID is integrated but Layers still seperated. What if I put both layers and patient ID as scVI 

# %% [markdown]
# ## Data vizualization

# %%
layer_order = [
    "Layer 1", "Layer 2", "Layer 3",
    "Layer 4", "Layer 5", "Layer 6",
    "White Matter"
]

# Create summary dataframe
df = (
    adata.obs
    .groupby(["patientID", "Layer"])["nCount_Spatial"]
    .sum()
    .reset_index()
)

# Make layer a categorical with correct order
df["Layer"] = pd.Categorical(df["Layer"], categories=layer_order, ordered=True)

# Sort values to ensure correct plotting order
df = df.sort_values(["patientID", "Layer"])

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df,
    x="Layer",
    y="nCount_Spatial",
    hue="patientID",
    marker="o"
)

plt.xticks(rotation=45)
plt.xlabel("Layer")
plt.ylabel("Total Spatial Counts")
plt.title("Number of Counts per Layer per Patient")
plt.tight_layout()
plt.show()

# %%
