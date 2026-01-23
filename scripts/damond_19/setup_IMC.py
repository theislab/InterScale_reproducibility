# %% [markdown]
# # Setup and download - IMC Type 1 Diabetes Pancreas data
#
# This tutorials shows how to set up an multiple slide spatial proteomics (IMC) dataset used from [Damond et al., 2019](https://doi.org/10.1016/j.cmet.2018.11.014). To follow along with this and the following tutorials, please execute the following steps first:
#
# - Set up InterScale environment (see instructions in ReadMe)
# - Download the sample data from the original publication from [Zenodo](https://zenodo.org/records/13907274)

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
LRZ_IMC_PANCREAS = '/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/spatial_proteomics_pancreas_t1d_IMC.h5ad'  
HPC_IMC_PANCREAS = '/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/imc_pancreas.h5ad'

# %% [markdown]
# ## Load data

# %%
adata_pancreas = sc.read_h5ad(HPC_IMC_PANCREAS)
adata_pancreas

# %%
print('Nr. of donors: ', len(np.unique(adata_pancreas.obs['case'])))

# %% [markdown]
# ## 1. Normalization
#
# The data needs to be normalized for InterScale. Check if the data is already normalized: 

# %%
print(f'Min count: {adata_pancreas.X.min()}, Max count: {adata_pancreas.X.max()}')

# %%
for layer_name in ['MeanIntensity', 'MedianIntensity']:
    print(f'Min count: {adata_pancreas.layers[layer_name].min()}, Max count: {adata_pancreas.layers[layer_name].max()}')

# %% [markdown]
# The data is not yet normalized, so normalize like described in the publication: "raw IMC counts were 99th-percentile-normalized and scaled from 0 to 1 (scaled counts)."

# %%
percentile_99 = np.percentile(adata_pancreas.X, 99, axis=0)
pct_norm_99 = adata_pancreas.X / percentile_99
adata_pancreas.layers["pct-norm-99"] = np.clip(pct_norm_99, 0, 1)

# %%
print(f'Min count: {adata_pancreas.layers["pct-norm-99"].min()}, Max count: {adata_pancreas.layers["pct-norm-99"].max()}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
p1 = sns.histplot(adata_pancreas.X.sum(1), bins=100, kde=False, ax=axes[0])
axes[0].set_title("Total counts")
p2 = sns.histplot(adata_pancreas.layers["pct-norm-99"].sum(1), bins=100, kde=False, ax=axes[1])
axes[1].set_title("Norm to target sum 1")
plt.show()

# %% [markdown]
# ## 2. Calculate spatial connectivity matrix
#
# Use [`squidpy.gr.spatial_neighbors()`](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html)) to calculate the spatial connectivity. For image-based ST it is important to set `coord_type='generic'`. In Squidpy, you have the option between k-nearest neighbors, delaunay and radius based neighborhood. For InterScale, we use a `radius`-based neighborhood to capture density information. Find the radius for which the number of connected neighbors is approximately 10-30, depending on tissue density. 

# %%
print('Nr. of images', len(np.unique(adata_pancreas.obs['ImageNumber'])))

# %%
adata_pancreas.obs['ImageNumber'] = pd.Categorical(adata_pancreas.obs['ImageNumber'])

# %%
sq.gr.spatial_neighbors(
    adata_pancreas,
    coord_type = "generic",
    library_key = "ImageNumber",
    radius = 30,
)

# %%
# Calculate nr. of neighbors per cells
conn = adata_pancreas.obsp['spatial_connectivities']
# Print average number of connections per node
avg_connections = conn.nnz / conn.shape[0]  # total connections / number of nodes
print(f"Average number of connections per node: {avg_connections:.2f}")

# %%
sq.pl.spatial_scatter(
            adata_pancreas,
            color = ['CellCat','CellType'],
            spatial_key = 'spatial',
            library_key='image_name',
            library_id=['M25'],
            shape= None,
            size=50
)

# %%
sq.pl.spatial_scatter(
            adata_pancreas,
            color = ['CellCat','CellType'],
            spatial_key = 'spatial',
            library_key='image_name',
            library_id=['M25'],
            shape= None,
            connectivity_key="spatial_connectivities",
            size=50
)

# %% [markdown]
# ## 3. Optional: Calculate sliding windows
#
# Sliding windows are necessary in case the tissue slide contains more than 4k cells. First, check how many cells are at minimum or maximum in your dataset.

# %%
tissue_cell_number = adata_pancreas.obs.groupby('ImageNumber').size()
print(f"Nr cells per sliding window: Min: {tissue_cell_number.min()}, Max: {tissue_cell_number.max()}, Avg: {tissue_cell_number.mean()}")

# %% [markdown]
# Select images that have more than some MAX number of cells.

# %%
MAX_CELLS = 2500
tissue_cell_number = adata_pancreas.obs.groupby('ImageNumber').size()
batches_high_nr_cells = tissue_cell_number[tissue_cell_number > MAX_CELLS]
print(batches_high_nr_cells)

# %% [markdown]
# We observe that there are a slides with only a few cells but 2 slides with more than 4k cells. For this case we have the option to 1) increase the context length to max = 4875 or 2) calculate sliding windows for all slides that have more than 3k cells. 

# %%
# Use the index of batches_high_nr_cells to get the image numbers
sub_adata = adata_pancreas[adata_pancreas.obs['ImageNumber'].isin(batches_high_nr_cells.index)]
sub_adata

# %%
print(f"Nr cells per for large images: Min: {batches_high_nr_cells.min()}, Max: {batches_high_nr_cells.max()}, Avg: {batches_high_nr_cells.mean()}")

# %%
sq.tl.sliding_window(
    adata=sub_adata,
    library_key="ImageNumber",  # to stratify by sample
    window_size=600,
    overlap=0,
    copy=False,  # we modify in place
)

# %%
window_size = sub_adata.obs.groupby('sliding_window_assignment').size()
print(f"Nr cells per sliding window: Min: {window_size.min()}, Max: {window_size.max()}, Avg: {window_size.mean()}")

# %%
sub_adata.obs["ImageNumber"]

# %% [markdown]
# With the sliding windows the maximum number of cells per sliding window is `2418`. 
#
# <mark>TODO: Adjust max_seq_len in config file!</mark>
# Given this we adjust the `max_seq_len` from the model.global_component to `2418`. The default is `2000`.

# %%
del sub_adata.uns['sliding_window_assignment_colors']

# %%
sq.pl.spatial_scatter(
    sub_adata[sub_adata.obs['ImageNumber'].isin([839, 6])],
    spatial_key = 'spatial',
    library_key='ImageNumber', 
    #library_id = [839, 6],
    color="sliding_window_assignment", #cell_type_coarse",
    shape= None,
    size = 10,
)

# %% [markdown]
# Add sliding window assignments to full dataset.

# %%
max_image_number = adata_pancreas.obs['ImageNumber'].astype(int).max()

# Renumber sliding_window_assignments in sub_adata starting from max_image_number + 1
unique_assignments = sub_adata.obs['sliding_window_assignment'].unique()
assignment_mapping = {old: new for old, new in zip(
    sorted(unique_assignments), 
    range(max_image_number + 1, max_image_number + 1 + len(unique_assignments))
)}

# Create a renumbered version
sub_adata.obs['sliding_window_assignment_renumbered'] = sub_adata.obs['sliding_window_assignment'].map(assignment_mapping)

# Initialize with ImageNumber
adata_pancreas.obs['sliding_window_assignment'] = adata_pancreas.obs['ImageNumber'].astype(int).copy()
mask = adata_pancreas.obs['ImageNumber'].isin(batches_high_nr_cells.index)
adata_pancreas.obs.loc[mask, 'sliding_window_assignment'] = sub_adata.obs['sliding_window_assignment_renumbered'].values

# %%
window_size = adata_pancreas.obs.groupby('sliding_window_assignment').size()
print(f"Nr cells per sliding window: Min: {window_size.min()}, Max: {window_size.max()}, Avg: {window_size.mean()}")

# %% [markdown]
# ## 4. Split data into train and val set
#
# Training the model requires a `split` assignment for each donor/patient/sliding window that you wanna train on. 

# %%
from sklearn.model_selection import train_test_split

# %%
df = adata_pancreas.obs[['case', 'stage']]
value_counts = pd.DataFrame(df.values, columns=df.columns).value_counts()
print(value_counts)

# %%
adata_pancreas.obs['split'] = 'train'
# assign each one [Long-duration, ND, Onset]
adata_pancreas.obs['split'][adata_pancreas.obs['case'].isin([6418, 6126, 6228])] = 'val'
adata_pancreas.obs['split'][adata_pancreas.obs['case'].isin([6089, 6386, 6380])] = 'test'

# %%
df = adata_pancreas.obs[['split', 'case', 'stage']]
value_counts = pd.DataFrame(df.values, columns=df.columns).value_counts()
print(value_counts)

# %% [markdown]
# ## Save adata object
#
# Save the prepared adata object such that it can be loaded for the model training. 

# %%

# %%
#adata_pancreas.write('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/imc_pancreas_pp.h5ad')

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
#         max_seq_len: 2418
#   save: /path/to/save/model/
# dataset:
#   h5ad_data: visium_tme_pp.h5ad
#   name: Visium_breast_cancer_TME
#   sample_key: ['batch']
#   spatial_neigbors_kwargs:
#     coord_type: grid
#     library_key: batch
# ```
#
#
# Save the config file as `.yaml` and proceed to training (either interactively in jupyter notebook or by running a script).

# %%

# %% [markdown]
# ## scVI load 

# %%
adata_pancreas = sc.read_h5ad('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/imc_pancreas_pp.h5ad')

# %%
adata_pancreas.uns['scvi']['connectivities_key'] = 'scvi_connectivities'
adata_pancreas.uns['scvi']['distances_key'] = 'scvi_distances'
adata_pancreas.uns['scvi']['params']['method'] = 'method'

sc.tl.umap(adata_pancreas, min_dist=0.3, neighbors_key='scvi')

# %%
adata_pancreas.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/imc_pancreas_pp_umap.h5ad')

# %%
adata_pancreas

# %%
import matplotlib
if not hasattr(matplotlib.colormaps, 'get_cmap'):
    matplotlib.colormaps.get_cmap = lambda name: matplotlib.colormaps[name]

# %%
sc.pl.umap(
    adata_pancreas,
    frameon=False,
    ncols=1,
    neighbors_key = 'scvi'
)

# %%
sc.pl.embedding(adata_pancreas, basis='X_scVI', frameon=False, ncols=1)

# %%
adata_pancreas

# %%
