# %% [markdown]
# # Setup and download data
#
# This tutorials shows how to set up an multiple slide Alzheimer Disease Visium dataset from [Chen et al., 2022](https://pubmed.ncbi.nlm.nih.gov/36544231/). To follow along with this and the following tutorials, please execute the following steps first:
#
# - Set up InterScale environment (see instructions in ReadMe)
# - Load raw data from GEO accession (GSE220442) with [SpatialData loader](https://spatialdata.scverse.org/projects/io/en/latest/generated/spatialdata_io.visium.html).

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

# set INTERSCALE_DIR to the current working directory
INTERSCALE_DIR = Path.cwd()
print(INTERSCALE_DIR)

# %% [markdown]
# <mark>TODO: Change data path</mark>

# %%
CHEN_22 = '/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/chen_adata.h5ad'  

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
# Add a column marking which mode each cell belongs to
cutoff = 1500  # adjust based on your plot - looks like the valley is around 1500
adata.obs['count_mode'] = np.where(
    adata.layers["log1p_norm"].sum(1) < cutoff, 
    'low', 
    'high'
)

# Check the split
print(adata.obs['count_mode'].value_counts())

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

# %% [markdown]
# ## Save adata object
#
# Save the prepared adata object such that it can be loaded for the model training. 

# %%
adata_breast.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/chen_adata_pp.h5ad'  )

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
