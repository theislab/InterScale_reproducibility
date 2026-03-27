# %% [markdown]
# # CosMx Pancreas data visualizations
#
# - Cell type composition

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
import yaml
import matplotlib.pyplot as plt
import os

with open(os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

# %%
import scanpy as sc
import squidpy as sq

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Figure setting

# %%
import yaml
import matplotlib.pyplot as plt

with open(os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"]["melton25"]

# %% [markdown]
# ## Load data

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, 'data', 'melton25.h5ad'))
adata

# %% [markdown]
# ## Cell type composition

# %%
# Step 1: Group observations by 'condition' and count each 'CellTypes_max' entry
counts = adata.obs.groupby(['condition', 'cell_type_coarse']).size().unstack(fill_value=0)

# Step 2: Calculate relative abundance (proportion) by normalizing counts within each condition
relative_abundance = counts.div(counts.sum(axis=1), axis=0)

#colors = adata.uns['Niche_label_colors']

# Step 3: Plot stacked bar plot
relative_abundance.plot(kind='bar', stacked=True, color=CELL_TYPE_COLORS, figsize=(4, 5),width=0.9)

# Add labels and title
plt.xlabel('Condition')
plt.ylabel('Relative Abundance')
#plt.title('Relative Abundance of Cell Types by Condition')
plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f"{project_root}/figures/{DATA}/data_visualization/cell_type_relative_abundance_per_condition.png", format="png", bbox_inches="tight", dpi=1200)
plt.show()

# %% [markdown]
# ## Connectivity on spatial slide
#
# Analyze the influence of the radius on the connectivity in the spatial slide.
#
# For selected radii:
# - Plot nr. avg. connected cells (cell type specific)
# - Plot nr. avg. connected cells on spatial slide

# %%
radii = [60, 110, 150]

# %%
color_list = [CELL_TYPE_COLORS[ct] for ct in adata.obs["cell_type_coarse"].cat.categories]
adata.uns[f'{"cell_type_coarse"}_colors'] = color_list

# %%
for radius in radii:
    sq.gr.spatial_neighbors(
        adata,
        coord_type = "generic",
        library_key = "slide_fov",
        radius = radius,
    )
    sq.pl.spatial_scatter(
        adata,
        library_key='slide_fov',
        library_id = '1_12',
        img=False,
        shape= None,
        color=["cell_type_coarse"],
        connectivity_key = "spatial_connectivities",
        figsize=(10,10),
        save = f"{project_root}/figures/{DATA}/data_visualization/spatial_conn_{radius}.png",
        dpi = 300
    )

# %%
library_id = "1_12"
sq.pl.spatial_scatter(
    adata,
    library_key='slide_fov',
    library_id = library_id,
    img=False,
    shape= None,
    color=["cell_type_coarse"],
    figsize=(10,10),
    save = f"{project_root}/figures/{DATA}/data_visualization/{library_id}.png",
    dpi = 300,
    size = 50
)

# %%
for radius in radii:
    sq.gr.spatial_neighbors(
        adata,
        coord_type="generic",
        library_key="slide_fov",
        radius=radius,
    )
    
    # Number of neighbors per cell (row-wise non-zero counts)
    conn = adata.obsp["spatial_connectivities"]
    n_neighbors_per_cell = np.diff(conn.indptr)  # works for CSR matrix
    avg_neighbors = n_neighbors_per_cell.mean()
    print(f"Radius {radius}: avg connected nodes = {avg_neighbors:.2f}")

# %%
# Collect per-cell data across radii
records = []
for radius in radii:
    sq.gr.spatial_neighbors(adata, coord_type="generic", library_key="slide_fov", radius=radius)
    adata.obs["n_neighbors"] = np.diff(adata.obsp["spatial_connectivities"].tocsr().indptr)
    df = adata.obs[["cell_type_coarse", "n_neighbors"]].copy()
    df["radius"] = radius
    records.append(df)

df_long = pd.concat(records)

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
sns.boxplot(
    data=df_long,
    x="cell_type_coarse",
    y="n_neighbors",
    hue="radius",
    ax=ax,
    palette="YlOrRd",
    flierprops={"marker": ".", "markersize": 2}  # shrink outlier dots
)
ax.set_xlabel("Cell type")
ax.set_ylabel("N neighbors")
ax.set_title("Neighbor connectivity per cell type across radii")
ax.tick_params(axis="x", rotation=45)
ax.legend(title="Radius", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

# %%
