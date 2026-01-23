# %% [markdown]
# # Setup and download data
#
# This tutorials shows how to set up an multiple slide Visium dataset used in [Withnell & Secrier, 2024](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03428-y). To follow along with this and the following tutorials, please execute the following steps first:
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
VISIUM_BREAST = '/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/visium_tme.h5ad'  

# %% [markdown]
# ## Load data

# %%
adata_breast = sc.read_h5ad(VISIUM_BREAST)
adata_breast

# %% [markdown]
# ## 1. Normalization
#
# The data needs to be normalized for InterScale. Check if the data is already normalized: 

# %%
print(f'Min count: {adata_breast.X.min()}, Max count: {adata_breast.X.max()}')

# %% [markdown]
# The data is not yet normalized, so normalize and transform with log1p.

# %%
scales_counts = sc.pp.normalize_total(adata_breast, target_sum=None, inplace=False)
adata_breast.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

# %%
print(f'Min count: {adata_breast.layers["log1p_norm"].min()}, Max count: {adata_breast.layers["log1p_norm"].max()}')

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
p1 = sns.histplot(adata_breast.X.sum(1), bins=100, kde=False, ax=axes[0])
axes[0].set_title("Total counts")
p2 = sns.histplot(adata_breast.layers["log1p_norm"].sum(1), bins=100, kde=False, ax=axes[1])
axes[1].set_title("Shifted logarithm")
plt.show()

# %% [markdown]
# ## 2. Calculate spatial connectivity matrix
#
# Use [`squidpy.gr.spatial_neighbors()`](https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html)) to calculate the spatial connectivity. For Visium it is important to set `coord_type='grid'` and indicate the IDs per slide (e.i. `batch_id`)

# %%
np.unique(adata_breast.obs['batch'])

# %%
sq.gr.spatial_neighbors(
    adata_breast,
    coord_type = "grid",
    library_key = "batch"
)

# %%
sq.pl.spatial_scatter(
    adata_breast,
    library_key='batch',
    library_id = '1',
    connectivity_key="spatial_connectivities",
    img=False,
    color=["tumour_cells"]
)

# %% [markdown]
# ## 3. Optional: Calculate sliding windows
#
# Sliding windows are necessary in case the tissue slide contains more than 4k cells. First, check how many cells are at minimum or maximum in your dataset.

# %%
tissue_cell_number = adata_breast.obs.groupby('batch').size()
print(f"Nr cells per sliding window: Min: {tissue_cell_number.min()}, Max: {tissue_cell_number.max()}, Avg: {tissue_cell_number.mean()}")

# %%
tissue_cell_number.plot(kind='bar', color='steelblue', edgecolor='black')

# %% [markdown]
# We observe that there are a slides with only a few cells but 2 slides with more than 4k cells. For this case we have the option to 1) increase the context length to max = 4875 or 2) calculate sliding windows for all slides that have more than 3k cells. 

# %%
batches_over_4k = tissue_cell_number[tissue_cell_number > MAX_CELLS]
print(batches_over_4k)

# %%
del adata_breast.obs['sliding_window_assignment']

# %%
sq.tl.sliding_window(
    adata=adata_breast,
    library_key="batch",  # to stratify by sample
    window_size=10000,
    overlap=0,
    copy=False,  # we modify in place
)

# %%
window_size = adata_breast.obs.groupby('sliding_window_assignment').size()
print(f"Nr cells per sliding window: Min: {window_size.min()}, Max: {window_size.max()}, Avg: {window_size.mean()}")

# %%
sq.pl.spatial_scatter(
    adata_breast,
    spatial_key = 'spatial',
    library_key='batch', 
    color="sliding_window_assignment", #cell_type_coarse",
    size = 10,
)

# %% [markdown]
# With the sliding windows the maximum number of cells per sliding window is `2418`. 
#
# <mark>TODO: Adjust max_seq_len in config file!</mark>
# Given this we adjust the `max_seq_len` from the model.global_component to `2418`. The default is `2000`.

# %% [markdown]
# ## 4. Split data into train and val set
#
# Training the model requires a `split` assignment for each donor/patient/sliding window that you wanna train on. 

# %%
import numpy as np
from sklearn.model_selection import train_test_split

# %%
df = pd.DataFrame({
    'sliding_window_assignment': adata_breast.obs['sliding_window_assignment'],
    'batch': adata_breast.obs['batch']
})

# Get unique batches
unique_batches = adata_breast.obs['batch'].unique()

# Split batches (not cells) into train/val/test
train_batches, temp_batches = train_test_split(
    unique_batches,
    test_size=0.3,
    random_state=42
)

val_batches, test_batches = train_test_split(
    temp_batches,
    test_size=0.5,
    random_state=42
)

# Assign cells based on their batch membership
adata_breast.obs['split'] = 'train'
adata_breast.obs.loc[adata_breast.obs['batch'].isin(val_batches), 'split'] = 'val'
adata_breast.obs.loc[adata_breast.obs['batch'].isin(test_batches), 'split'] = 'test'

# Verify: each batch should appear in only one split
print("Batches per split:")
print(adata_breast.obs.groupby('split')['batch'].unique())

print("\nCells per split:")
print(adata_breast.obs['split'].value_counts())
print(adata_breast.obs.groupby(['split', 'batch']).size())

# %%
adata_breast.obs['split']

# %% [markdown]
# ## Save adata object
#
# Save the prepared adata object such that it can be loaded for the model training. 

# %%
adata_breast.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/visium_tme_pp.h5ad'  )

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
# ## Explore cell type labels

# %%
adata_breast = sc.read_h5ad('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/visium_tme_pp.h5ad'  )
adata_breast

# %%

# %%
adata_breast.obs.columns

# %%
sq.pl.spatial_scatter(
    adata_breast,
    library_key='batch',
    library_id = '1',
    connectivity_key="spatial_connectivities",
    img=False,
    color=["EMT_cold", "EMT_hot", "EPI_cold", "EPI_hot"]
)

# %% [markdown]
# Check that there are no cells which are EMT hot and EMT cold.
#
# - EMT hot: statistically significant spatial cluster where tumor cells undergoing epithelial-to-mesenchymal transition are concentrated/aggregared beyond what would be expected
# - EMT cold: basically the opposite

# %%
# Check specific pairs quickly
obs = adata_breast.obs

# EMT_hot AND EMT_cold (contradictory?)
print("EMT_hot ∩ EMT_cold:", np.sum((obs['EMT_hot'] > 0) & (obs['EMT_cold'] > 0)))

# EMT_hot AND EPI_hot (transitional?)
print("EMT_hot ∩ EPI_hot:", np.sum((obs['EMT_hot'] > 0) & (obs['EPI_hot'] > 0)))

# EMT_hot AND EPI_cold (expected co-occurrence)
print("EMT_hot ∩ EPI_cold:", np.sum((obs['EMT_hot'] > 0) & (obs['EPI_cold'] > 0)))

# EPI_hot AND EMT_cold (expected co-occurrence)
print("EPI_hot ∩ EMT_cold:", np.sum((obs['EPI_hot'] > 0) & (obs['EMT_cold'] > 0)))

# %% [markdown]
# EMT tumour spots can be linked with two cellular hallmarks: 
#
# - angiogenesis
# - hypoxia

# %%
obs = adata_breast.obs

print("="*60)
print("EMT_hot overlaps:")
print("="*60)
print(f"EMT_hot ∩ hypoxia_hot: {np.sum((obs['EMT_hot'] > 0) & (obs['hypoxia_hallmarks_hot'] > 0))}")
print(f"EMT_hot ∩ hypoxia_cold: {np.sum((obs['EMT_hot'] > 0) & (obs['hypoxia_hallmarks_cold'] > 0))}")
print(f"EMT_hot ∩ angio_hot: {np.sum((obs['EMT_hot'] > 0) & (obs['angio_hallmarks_hot'] > 0))}")
print(f"EMT_hot ∩ angio_cold: {np.sum((obs['EMT_hot'] > 0) & (obs['angio_hallmarks_cold'] > 0))}")

print("\n" + "="*60)
print("EMT_cold overlaps:")
print("="*60)
print(f"EMT_cold ∩ hypoxia_hot: {np.sum((obs['EMT_cold'] > 0) & (obs['hypoxia_hallmarks_hot'] > 0))}")
print(f"EMT_cold ∩ hypoxia_cold: {np.sum((obs['EMT_cold'] > 0) & (obs['hypoxia_hallmarks_cold'] > 0))}")
print(f"EMT_cold ∩ angio_hot: {np.sum((obs['EMT_cold'] > 0) & (obs['angio_hallmarks_hot'] > 0))}")
print(f"EMT_cold ∩ angio_cold: {np.sum((obs['EMT_cold'] > 0) & (obs['angio_hallmarks_cold'] > 0))}")

print("\n" + "="*60)
print("EPI_hot overlaps:")
print("="*60)
print(f"EPI_hot ∩ hypoxia_hot: {np.sum((obs['EPI_hot'] > 0) & (obs['hypoxia_hallmarks_hot'] > 0))}")
print(f"EPI_hot ∩ hypoxia_cold: {np.sum((obs['EPI_hot'] > 0) & (obs['hypoxia_hallmarks_cold'] > 0))}")
print(f"EPI_hot ∩ angio_hot: {np.sum((obs['EPI_hot'] > 0) & (obs['angio_hallmarks_hot'] > 0))}")
print(f"EPI_hot ∩ angio_cold: {np.sum((obs['EPI_hot'] > 0) & (obs['angio_hallmarks_cold'] > 0))}")

print("\n" + "="*60)
print("EPI_cold overlaps:")
print("="*60)
print(f"EPI_cold ∩ hypoxia_hot: {np.sum((obs['EPI_cold'] > 0) & (obs['hypoxia_hallmarks_hot'] > 0))}")
print(f"EPI_cold ∩ hypoxia_cold: {np.sum((obs['EPI_cold'] > 0) & (obs['hypoxia_hallmarks_cold'] > 0))}")
print(f"EPI_cold ∩ angio_hot: {np.sum((obs['EPI_cold'] > 0) & (obs['angio_hallmarks_hot'] > 0))}")
print(f"EPI_cold ∩ angio_cold: {np.sum((obs['EPI_cold'] > 0) & (obs['angio_hallmarks_cold'] > 0))}")

print("\n" + "="*60)
print("Key expected relationships from paper:")
print("="*60)
# EMT_hot should co-occur with hypoxia_hot and angio_hot
# EPI_hot should NOT co-occur with hypoxia_hot and angio_hot
emt_with_hypoxia = np.sum((obs['EMT_hot'] > 0) & (obs['hypoxia_hallmarks_hot'] > 0))
epi_with_hypoxia = np.sum((obs['EPI_hot'] > 0) & (obs['hypoxia_hallmarks_hot'] > 0))
print(f"EMT_hot with hypoxia_hot: {emt_with_hypoxia} vs EPI_hot with hypoxia_hot: {epi_with_hypoxia}")

emt_with_angio = np.sum((obs['EMT_hot'] > 0) & (obs['angio_hallmarks_hot'] > 0))
epi_with_angio = np.sum((obs['EPI_hot'] > 0) & (obs['angio_hallmarks_hot'] > 0))
print(f"EMT_hot with angio_hot: {emt_with_angio} vs EPI_hot with angio_hot: {epi_with_angio}")

# %% [markdown]
# ## Create categorical variable for node/cell
#
# Given the two tumour types (EMT and EPI) with the hot and cold features we create a categorical node variable that summarizes the tumour spots and hallmarks (angiogenesis and hypoxia). 

# %%
import numpy as np
import pandas as pd

def create_combined_category(adata, 
                              threshold=0,
                              handle_nan='none'):
    """
    Create a categorical variable capturing all combinations of
    EMT, EPI, hypoxia, and angiogenesis hot/cold status.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object
    threshold : float
        Threshold for considering a spot as hot/cold (default: > 0)
    handle_nan : str
        How to handle NaN: 'none' (treat as no signal), 'exclude' (separate category)
    
    Returns
    -------
    adata : AnnData
        AnnData with new categorical columns
    category_mapping : pd.DataFrame
        DataFrame mapping category numbers to feature combinations
    """
    adata = adata.copy()
    obs = adata.obs
    
    # Define columns
    columns = {
        'EMT': ('EMT_hot', 'EMT_cold'),
        'EPI': ('EPI_hot', 'EPI_cold'),
        'Hypoxia': ('hypoxia_hallmarks_hot', 'hypoxia_hallmarks_cold'),
        'Angio': ('angio_hallmarks_hot', 'angio_hallmarks_cold')
    }
    
    # Function to assign status for each feature
    def get_status(hot_col, cold_col, threshold, handle_nan):
        hot_vals = obs[hot_col].copy() if hot_col in obs.columns else pd.Series(0, index=obs.index)
        cold_vals = obs[cold_col].copy() if cold_col in obs.columns else pd.Series(0, index=obs.index)
        
        # Handle NaN
        hot_nan = hot_vals.isna()
        cold_nan = cold_vals.isna()
        
        if handle_nan == 'none':
            hot_vals = hot_vals.fillna(0)
            cold_vals = cold_vals.fillna(0)
        
        # Assign status
        status = np.full(len(obs), 'none', dtype=object)
        status[hot_vals > threshold] = 'hot'
        status[cold_vals > threshold] = 'cold'
        
        # If both hot and cold > threshold, prioritize hot
        both = (hot_vals > threshold) & (cold_vals > threshold)
        status[both] = 'hot'  # or could be 'both' if you want to track this
        
        if handle_nan == 'exclude':
            status[hot_nan | cold_nan] = 'nan'
        
        return status
    
    # Get status for each feature
    statuses = {}
    for feature, (hot_col, cold_col) in columns.items():
        statuses[feature] = get_status(hot_col, cold_col, threshold, handle_nan)
        adata.obs[f'{feature}_status'] = statuses[feature]
    
    # Create combined string label
    combined_labels = []
    for i in range(len(obs)):
        parts = []
        for feature in ['EMT', 'EPI', 'Hypoxia', 'Angio']:
            status = statuses[feature][i]
            if status == 'hot':
                parts.append(f'{feature}+')
            elif status == 'cold':
                parts.append(f'{feature}-')
            # 'none' is not added to keep labels shorter
        
        if len(parts) == 0:
            label = 'None'
        else:
            label = '_'.join(parts)
        combined_labels.append(label)
    
    adata.obs['combined_category_str'] = pd.Categorical(combined_labels)
    
    # Create numeric encoding
    unique_categories = sorted(adata.obs['combined_category_str'].unique())
    category_to_num = {cat: i for i, cat in enumerate(unique_categories)}
    adata.obs['combined_category_num'] = adata.obs['combined_category_str'].map(category_to_num)
    
    # Create mapping DataFrame
    category_mapping = pd.DataFrame({
        'category_num': range(len(unique_categories)),
        'category_str': unique_categories,
        'count': [sum(adata.obs['combined_category_str'] == cat) for cat in unique_categories],
    })
    category_mapping['percentage'] = (category_mapping['count'] / len(adata) * 100).round(2)
    
    # Add individual feature columns to mapping
    for cat in unique_categories:
        for feature in ['EMT', 'EPI', 'Hypoxia', 'Angio']:
            if f'{feature}+' in cat:
                category_mapping.loc[category_mapping['category_str'] == cat, feature] = 'hot'
            elif f'{feature}-' in cat:
                category_mapping.loc[category_mapping['category_str'] == cat, feature] = 'cold'
            else:
                category_mapping.loc[category_mapping['category_str'] == cat, feature] = 'none'
    
    print("Category Mapping:")
    print("=" * 100)
    print(category_mapping.to_string(index=False))
    print(f"\nTotal categories: {len(unique_categories)}")
    print(f"Total spots: {len(adata)}")
    
    return adata, category_mapping


def create_detailed_category(adata, threshold=0):
    """
    Create a more detailed categorical encoding with explicit status for each feature.
    
    Returns a DataFrame with separate columns for each feature status,
    plus combined numeric and string categories.
    """
    adata = adata.copy()
    obs = adata.obs
    
    # Define columns
    feature_cols = {
        'EMT': ('EMT_hot', 'EMT_cold'),
        'EPI': ('EPI_hot', 'EPI_cold'),
        'Hypoxia': ('hypoxia_hallmarks_hot', 'hypoxia_hallmarks_cold'),
        'Angio': ('angio_hallmarks_hot', 'angio_hallmarks_cold')
    }
    
    # Status encoding: 0 = none, 1 = cold, 2 = hot
    status_encoding = {'none': 0, 'cold': 1, 'hot': 2}
    
    for feature, (hot_col, cold_col) in feature_cols.items():
        hot_vals = obs[hot_col].fillna(0) if hot_col in obs.columns else pd.Series(0, index=obs.index)
        cold_vals = obs[cold_col].fillna(0) if cold_col in obs.columns else pd.Series(0, index=obs.index)
        
        # Assign status
        status = np.full(len(obs), 'none', dtype=object)
        status[cold_vals > threshold] = 'cold'
        status[hot_vals > threshold] = 'hot'  # hot takes priority
        
        adata.obs[f'{feature}_status'] = status
        adata.obs[f'{feature}_status_num'] = [status_encoding[s] for s in status]
    
    # Create combined numeric code using base-3 encoding
    # EMT * 27 + EPI * 9 + Hypoxia * 3 + Angio * 1
    adata.obs['combined_code'] = (
        adata.obs['EMT_status_num'] * 27 +
        adata.obs['EPI_status_num'] * 9 +
        adata.obs['Hypoxia_status_num'] * 3 +
        adata.obs['Angio_status_num'] * 1
    )
    
    # Create string label
    def make_label(row):
        parts = []
        for feature in ['EMT', 'EPI', 'Hypoxia', 'Angio']:
            status = row[f'{feature}_status']
            if status == 'hot':
                parts.append(f'{feature}+')
            elif status == 'cold':
                parts.append(f'{feature}-')
        return '_'.join(parts) if parts else 'None'
    
    adata.obs['combined_label'] = adata.obs.apply(make_label, axis=1)
    adata.obs['combined_label'] = pd.Categorical(adata.obs['combined_label'])
    
    # Create sequential category number (0, 1, 2, ...)
    unique_labels = sorted(adata.obs['combined_label'].unique())
    label_to_cat = {label: i for i, label in enumerate(unique_labels)}
    adata.obs['category_id'] = adata.obs['combined_label'].map(label_to_cat)
    
    # Build mapping table
    mapping_data = []
    for cat_id, label in enumerate(unique_labels):
        mask = adata.obs['combined_label'] == label
        count = mask.sum()
        
        row = {
            'category_id': cat_id,
            'label': label,
            'count': count,
            'percentage': round(count / len(adata) * 100, 2)
        }
        
        # Parse individual statuses
        for feature in ['EMT', 'EPI', 'Hypoxia', 'Angio']:
            if f'{feature}+' in label:
                row[feature] = 'hot'
            elif f'{feature}-' in label:
                row[feature] = 'cold'
            else:
                row[feature] = 'none'
        
        mapping_data.append(row)
    
    mapping_df = pd.DataFrame(mapping_data)
    
    # Reorder columns
    mapping_df = mapping_df[['category_id', 'label', 'EMT', 'EPI', 'Hypoxia', 'Angio', 'count', 'percentage']]
    
    print("="*100)
    print("CATEGORY MAPPING")
    print("="*100)
    print(mapping_df.to_string(index=False))
    print(f"\nTotal unique categories: {len(unique_labels)}")
    print(f"Total spots: {len(adata)}")
    
    return adata, mapping_df


def create_simplified_category(adata, threshold=0):
    """
    Create a simplified categorical encoding focusing on biologically meaningful groups.
    
    Categories:
    0: None - no hot/cold designation
    1: EMT_only - only EMT hot
    2: EPI_only - only EPI hot  
    3: EMT_Hypoxic - EMT hot + hypoxia hot
    4: EMT_Angiogenic - EMT hot + angio hot
    5: EMT_Hypoxic_Angiogenic - EMT hot + hypoxia hot + angio hot
    6: EPI_Proliferative - EPI hot (+ potentially others)
    7: Stromal_Vascular - hypoxia/angio hot without EMT/EPI
    8: Cold_regions - any cold designation
    9: Mixed - other combinations
    """
    adata = adata.copy()
    obs = adata.obs
    
    # Get masks
    def get_mask(col):
        if col in obs.columns:
            return np.array(obs[col].fillna(0) > threshold)
        return np.zeros(len(obs), dtype=bool)
    
    emt_hot = get_mask('EMT_hot')
    emt_cold = get_mask('EMT_cold')
    epi_hot = get_mask('EPI_hot')
    epi_cold = get_mask('EPI_cold')
    hypoxia_hot = get_mask('hypoxia_hallmarks_hot')
    hypoxia_cold = get_mask('hypoxia_hallmarks_cold')
    angio_hot = get_mask('angio_hallmarks_hot')
    angio_cold = get_mask('angio_hallmarks_cold')
    
    # Any cold
    any_cold = emt_cold | epi_cold | hypoxia_cold | angio_cold
    any_hot = emt_hot | epi_hot | hypoxia_hot | angio_hot
    
    # Initialize
    labels = np.full(len(obs), 'None', dtype=object)
    
    # Assign in order of specificity (most specific last)
    
    # Cold regions (if only cold, no hot)
    labels[any_cold & ~any_hot] = 'Cold_regions'
    
    # Stromal/Vascular (hypoxia/angio without EMT/EPI)
    labels[(hypoxia_hot | angio_hot) & ~emt_hot & ~epi_hot] = 'Stromal_Vascular'
    
    # EPI only
    labels[epi_hot & ~emt_hot & ~hypoxia_hot & ~angio_hot] = 'EPI_only'
    
    # EPI with hallmarks
    labels[epi_hot & ~emt_hot] = 'EPI_Proliferative'
    
    # EMT only
    labels[emt_hot & ~hypoxia_hot & ~angio_hot] = 'EMT_only'
    
    # EMT + Angiogenic
    labels[emt_hot & angio_hot & ~hypoxia_hot] = 'EMT_Angiogenic'
    
    # EMT + Hypoxic
    labels[emt_hot & hypoxia_hot & ~angio_hot] = 'EMT_Hypoxic'
    
    # EMT + Hypoxic + Angiogenic (triple)
    labels[emt_hot & hypoxia_hot & angio_hot] = 'EMT_Hypoxic_Angiogenic'
    
    # Mixed (EMT + EPI both hot - unusual)
    labels[emt_hot & epi_hot] = 'Mixed_EMT_EPI'
    
    adata.obs['simplified_category'] = pd.Categorical(labels)
    
    # Create numeric encoding
    category_order = [
        'None',
        'Cold_regions', 
        'Stromal_Vascular',
        'EPI_only',
        'EPI_Proliferative',
        'EMT_only',
        'EMT_Angiogenic',
        'EMT_Hypoxic',
        'EMT_Hypoxic_Angiogenic',
        'Mixed_EMT_EPI'
    ]
    
    # Only include categories that exist
    existing_categories = [c for c in category_order if c in labels]
    cat_to_num = {cat: i for i, cat in enumerate(existing_categories)}
    adata.obs['simplified_category_num'] = adata.obs['simplified_category'].map(cat_to_num)
    
    # Create mapping
    mapping_data = []
    for cat_num, cat_name in enumerate(existing_categories):
        count = (labels == cat_name).sum()
        mapping_data.append({
            'category_num': cat_num,
            'category_name': cat_name,
            'description': get_category_description(cat_name),
            'count': count,
            'percentage': round(count / len(adata) * 100, 2)
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    
    print("="*100)
    print("SIMPLIFIED CATEGORY MAPPING")
    print("="*100)
    print(mapping_df.to_string(index=False))
    
    return adata, mapping_df


def get_category_description(cat_name):
    """Return description for each category."""
    descriptions = {
        'None': 'No hot/cold signal above threshold',
        'Cold_regions': 'Only cold designations (depleted regions)',
        'Stromal_Vascular': 'Hypoxia/Angio hot without EMT/EPI (stromal)',
        'EPI_only': 'Epithelial hot only (tumor core)',
        'EPI_Proliferative': 'Epithelial hot region (proliferative)',
        'EMT_only': 'EMT hot without hypoxia/angio',
        'EMT_Angiogenic': 'EMT hot + Angiogenesis hot',
        'EMT_Hypoxic': 'EMT hot + Hypoxia hot',
        'EMT_Hypoxic_Angiogenic': 'EMT + Hypoxia + Angiogenesis (invasive front)',
        'Mixed_EMT_EPI': 'Both EMT and EPI hot (transitional/unusual)'
    }
    return descriptions.get(cat_name, 'Unknown')


def visualize_categories(adata, category_col='simplified_category', figsize=(12, 5)):
    """Visualize category distribution."""
    import matplotlib.pyplot as plt
    
    counts = adata.obs[category_col].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Category Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Category Proportions')
    
    plt.tight_layout()
    plt.show()


def export_category_mapping(mapping_df, filename='category_mapping.csv'):
    """Export mapping to CSV file."""
    mapping_df.to_csv(filename, index=False)
    print(f"Mapping saved to {filename}")


# %%
# Option 1: Full detailed encoding (all combinations)
adata_breast, mapping_detailed = create_detailed_category(adata_breast, threshold=0)

# Option 2: Simplified biologically meaningful categories
adata_breast, mapping_simplified = create_simplified_category(adata_breast, threshold=0)

# Option 3: Combined category with string labels
adata_breast, mapping_combined = create_combined_category(adata_breast, threshold=0)

# Visualize
visualize_categories(adata_breast, category_col='simplified_category')

# Export mapping
export_category_mapping(mapping_detailed, 'detailed_mapping.csv')

# Access labels for your model
labels_numeric = adata_breast.obs['category_id'].values  # detailed
labels_simplified = adata_breast.obs['simplified_category_num'].values  # simplified
labels_string = adata_breast.obs['combined_label'].values  # string version

# %%
adata_breast.write('/dss/dssfs03/tumdss/pn36po/pn36po-dss-0002/di93tig/Projects/A3_InterScale/data/visium_tme_pp_node_labels.h5ad'  )

# %%
print(adata_breast.obs.groupby(['split', 'simplified_category_num']).size())

# %%
