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

# %%
