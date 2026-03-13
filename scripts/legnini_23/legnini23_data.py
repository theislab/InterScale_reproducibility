# %%
import scanpy as sc
import squidpy as sq
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_array, csr_matrix, issparse

from graph_transformer_long_range_niches.pp import sliding_window, split_adata
from graph_transformer_long_range_niches._paths import CFG_FILES, RESULTS

import warnings
warnings.filterwarnings('ignore')

# %%
LEGNINI23 = "/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/legnini23.h5ad"

# %% [markdown]
# # Load Molecular Cartography Data

# %%
adata = sc.read_h5ad(LEGNINI23)
adata

# %%
print('Zero count cells: ', (adata.X.sum(1) ==0).sum())

# %% [markdown]
# ## Normalize
#
# Ideally, counts should be normalized between 0 to 3. 

# %%
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
# log1p transform
adata.layers["raw"] = adata.X
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
# Freeman-Tukey square root transform
assert issparse(adata.X)
sqrt_X = adata.X.sqrt()
# Create a new sparse matrix for X + 1
X_plus_1 = adata.X + csr_matrix(np.ones(adata.X.shape))
# Calculate the square root of (X + 1)
sqrt_X_plus_1 = X_plus_1.sqrt()
adata.layers['norm_ftsqrt'] = sqrt_X + sqrt_X_plus_1

# %%
# shifted Logarithm
scales_counts = sc.pp.normalize_total(adata, target_sum=10000, inplace=False)
# log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
p0 = sns.histplot(adata.layers["raw"].sum(1), bins=100, kde=False, ax=axes[0])
axes[0].set_title("Total Counts")
p1 = sns.histplot(adata.layers["norm_ftsqrt"].sum(1), bins=100, kde=False, ax=axes[1])
axes[1].set_title("Freeman-Tukey")
p2 = sns.histplot(adata.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[2])
axes[2].set_title("Log1p Norm")
plt.show()

# %% [markdown]
# - Why does each cell have exactly 279 counts?
# - Why are there cells with zero gene counts?

# %%
print('Raw - Min: ', {adata.layers['raw'].min()}, ', Max: ', {adata.layers['raw'].max()})
print('Log1pNorm - Min: ', {adata.layers['log1p_norm'].min()}, ', Max: ', {adata.layers['log1p_norm'].max()})
print('NormTRSqrt - Min: ', {adata.layers['norm_ftsqrt'].min()}, ', Max: ', {adata.layers['norm_ftsqrt'].max()})

# %%
adata.X = adata.layers['norm_ftsqrt']
sq.pl.spatial_scatter(
        adata,
        library_key = 'sample',
        #library_id = shh_slides,
        color = ['SHH'],
        cmap = 'viridis_r',
        size = 10,
        shape= None)

# %% [markdown]
# ## Neighbor graph (radius based)
#
# Check the average connectivity for different radius values.

# %%
for radi in [0, 200, 300]:
    print('Radius ', radi)
    sq.gr.spatial_neighbors(
        adata,
        radius=200,
        coord_type='generic',
        library_key = 'sample'
    )
    conn = adata.obsp['spatial_connectivities']
    # Print average number of connections per node
    avg_connections = conn.nnz / conn.shape[0]  # total connections / number of nodes
    print(f"Average number of connections per node: {avg_connections:.2f}")

# %%
# adata.obsm['spatial'] = adata.obsm['coordinates']
# del adata.obsm['coordinates']

# %%
# np.isnan(adata.obsm["spatial"]).sum()

# %% [markdown]
# Remove all cells that have entry NaN in .obsm['coordinates'].
# If NaN is in .obsm it leads to error when creating graph. Why is there NaN in obsm?
# [NaN value when importing Visium datasete](https://github.com/scverse/squidpy/issues/797)

# %%
# Create a boolean mask for rows without NaN coordinates
valid_coords = ~np.isnan(adata.obsm["spatial"]).any(axis=1)

# Filter the AnnData object
adata = adata[valid_coords].copy()

# Verify the removal of NaN values
print(f"NaN values remaining: {np.isnan(adata.obsm['spatial']).sum()}")

# %% [markdown]
# Make sure that obs_names are unique and convertable to string. 

# %%
adata.obs_names_make_unique

# %%
adata.obs['obs_names'] = adata.obs_names

# %%
adata.obs_names = np.array(range(adata.n_obs))

# %%
print(conn[1,:10])

# %%
import scipy.sparse as sp

def is_symmetric(A, tol=1e-10):
    if not sp.issparse(A):
        raise ValueError("Input must be a sparse matrix")
    
    # Check if A is square
    if A.shape[0] != A.shape[1]:
        return False
    
    # Compute the difference between A and its transpose
    diff = (A - A.T).tocoo()  # Convert to COOrdinate format for efficiency
    return np.all(np.abs(diff.data) < tol)


# %%
print(is_symmetric(conn))  # Should return True

# %%
from scipy.sparse.csgraph import connected_components

n_components, labels = connected_components(csgraph=conn)

# %%
adata.write(LEGNINI23)

# %% [markdown]
# ## GEX 

# %% [markdown]
# ## Cell type annotation

# %%
adata.var_names

# %%
sc.pp.neighbors(adata, n_pcs=10)
sc.tl.umap(adata)

# %%
sc.tl.leiden(adata, resolution=0.5)

# %%
np.unique(adata.obs['leiden'])

# %%
del adata.uns['leiden_colors']

# %%
sc.pl.umap(
    adata,
    color=['leiden'],
    size = 10,
)

# %%
sq.pl.spatial_scatter(
    adata,
    library_key = 'sample',
    color=['leiden'],
    size = 10,
    shape= None
)

# %%
sq.pl.spatial_scatter(
    adata,
    library_key = 'sample',
    color=['DBX2'],
    size = 10,
    shape= None
)

# %%
sc.tl.rank_genes_groups(adata, groupby='leiden')

# %%
sc.pl.rank_genes_groups(adata)

# %% [markdown]
# ## Sliding window

# %%
adata.obs_names = adata.obs_names.str.replace('-', '', regex=False)

# %%
adata.obs_names

# %%
adata.obs.groupby('sample').size()

# %%
adata

# %%
sliding_window(adata, library_key = 'sample', window_size=2100, overlap=0, spatial_key='coordinates')

# %%
adata

# %%
split_adata(adata, split_obs='sample', val_size=0.2, test_size=0.1, seed = 44, stratify_groups='condition')
adata.obs.groupby(['split', 'condition']).size()

# %%
sq.gr.spatial_neighbors(adata, 
                        library_key = 'sample', 
                        spatial_key = 'coordinates')

# %%
adata.var_names_make_unique()

# %%
sq.gr.spatial_neighbors(adata, spatial_key='coordinates')

# %%
