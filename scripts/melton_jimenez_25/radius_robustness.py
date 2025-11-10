# %% [markdown]
# # Radius robustness
#
# Plots showing the radius robustness of 
#
# - InterScale (GCN + Transformer)
# - GCN
# - PCATransformer
#
# For classification: i) graph label, ii) node label and regression. 

# %%
# ! pwd

# %%
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from notebook location)
project_root = Path('/dss/dsshome1/05/di93tig/1_projects/InterScale_reproducibility')
sys.path.insert(0, str(project_root))

# %%
import wandb
import pandas as pd
wandb.login()

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# ste sys path correctly
from src.wandb import load_result_as_df, compute_mean_and_std, summary_df, plot_robustness

# %%
custom_palette = {
    "PCATrans": "#BABABE", 
    "NeighTrans": "#CA6702",  
    "GCN": "#88C8B2",  
    "GCNTrans": "#005F73",
}

# %% [markdown]
# ## Load WandB paths

# %%
InterScale_condition_robustness = "tq6v6g2a"
# Pancreas_GNNTrans_condition_robustness = "ta15q33w"
# Pancreas_PCATrans_condition_robustness = "lw9pqd9t"
# Pancreas_NeighTrans_condition_robustness = "yebvf809"

# %% [markdown]
# ## Robustness plots
#
# Parameters: 
#
# - dataset.pct_mask_modes:
# - dataset.spatial_neigbors_kwargs.radius:
# - dataset.split_key:
# - model.decoder.type:
#
# Hypothesis: 
#
# - Some split_key predictions work better than others because potentially patients are more similar

# %%
SWEEP_GOAL = 'robustness'


# %%
def load_result_as_df(sweep_id, sweep_goal: str, classes: list):
    """
    sweep_id: str - ID from WandB run
    sweep_goal: robustenss, parameter
    calsses: list of class names
    """
    api = wandb.Api()
    entity, project = "francesca-drummer", "InterScale_hyperparameter_sweep"  

    # Get all runs associated with the sweep
    sweep_runs = api.sweep(f"{entity}/{project}/{sweep_id}").runs

    data = []
    for run in sweep_runs:
        if run.state == 'finished':
            prediction_task = run.config['dataset']['prediction_task']
        
            run_data = {
                "id": run.id,
                "name": run.name,
                "seed": run.config.get("optim.seed", None),
                "state": run.state,  # finished, running, failed
                "pct_mask_nodes": run.config.get("dataset.pct_mask_nodes", None),
                "radius": run.config.get("dataset.spatial_neigbors_kwargs.radius", None),
                "decoder_type": run.config.get("model.decoder.type", None),
                "runtime_seconds": run.summary.get("_runtime", None),
                "total_parameters": run.summary.get("total_parameters", None),
            }
            if 'regression' in prediction_task:
                run_data.update({
                    "test_r2": run.summary.get("test_r2", None),
                    "test_pearson_corr": run.summary.get("test_pearson_corr", None),
                })
            elif 'classification' in prediction_task:
                num_classes = run.config['dataset']['num_classes']
                run_data.update({
                    "test_acc": run.summary.get("test_accuracy", None),
                    "test_f1_micro/avg": run.summary.get("test_f1_micro/avg", None)
                })
                for class_idx in classes:
                    run_data[f'test_f1/class_{class_idx}'] = run.summary.get(f"test_f1_{class_idx}", None)
    
            if 'graph' in prediction_task:
                run_data.update({
                    "split_key": run.config.get("dataset.split_key", None),
                })
            if sweep_goal == 'parameter':
                if 'gnn' in run_data['name']:
                    run_data.update({
                        "gnn_num_layers": run.config.get("gnn.num_layers", None),
                        "gnn_hidden_dim": run.config.get("gnn.hidden_dim", None),
                        "embed_dim": run.config.get("gnn.embed_dim", None),
                    })
                if 'transformer' in run_data['name']:
                    run_data.update({
                        "trans_n_heads": run.config.get("transformer.n_heads", None),
                        "trans_num_layers": run.config.get("transformer.num_layers", None),
                        "trans_dim_feedforward": run.config.get("transformer.dim_feedforward", None),
                    })
            
            data.append(run_data)
        
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df


# %%
# df_GNN = load_result_as_df(Pancreas_GNN_condition_robustness, SWEEP_GOAL)
# df_GNNTrans = load_result_as_df(Pancreas_GNNTrans_condition_robustness, SWEEP_GOAL)
# df_PCATrans = load_result_as_df(Pancreas_PCATrans_condition_robustness, SWEEP_GOAL)
# df_NeighTrans = load_result_as_df(Pancreas_NeighTrans_condition_robustness, SWEEP_GOAL)
df_InterScale = load_result_as_df(InterScale_condition_robustness, SWEEP_GOAL, ['ND', 'T1D'])

# %%
df_InterScale.head()

# %%
plot_robustness(df_InterScale, metric="test_acc")

# %%
plot_robustness(df_GNNTrans, metric="test_acc")

# %%
plot_robustness(df_NeighTrans, metric="test_acc")

# %%
plot_robustness(df_PCATrans, metric="test_acc")


# %%
def plot_f1_barplot(df_list, model_names, split_scatter=False):
    """
    Plots a barplot of F1 scores for different models, averaged across split_keys, 
    with optional scatter plot for individual split values.

    Parameters:
    -----------
    df_list : list of pandas.DataFrame
        A list of DataFrames, each containing F1 scores with column names in the format 
        'test_f1/class{class_idx}' and a 'split_key' column. Each DataFrame corresponds to one model.
    
    model_names : list of str
        A list of model names, corresponding to the DataFrames in df_list.

    split_scatter : bool, optional (default: False)
        If True, plots individual F1 scores for each split_key, colored by split_key.

    Returns:
    --------
    None
        Displays a bar plot of mean F1 scores for each model, aggregated by class index, 
        and optional scatter plot for split-specific F1 scores.
    """
    # Assign model names to each DataFrame
    for df, model in zip(df_list, model_names):
        df["model"] = model  # Explicitly add model name

    df_combined = pd.concat(df_list, ignore_index=True)

    # Melt the dataframe to have 'test_f1' and 'class' as columns
    df_melted = pd.melt(df_combined, 
                    id_vars=['split_key', 'model'],  # Keep the 'split_key' column
                    value_vars=[col for col in df.columns if col.startswith('test_f1/class_')],  # Select columns starting with 'test_f1/class_'
                    var_name='class',  # Name for the new column containing the class information
                    value_name='test_f1'  # Name for the new column containing the F1 score values
                   )
    custom_palette = {
        "PCATrans": "#BABABE", 
        "NeighTrans": "#CA6702",  
        "GNN": "#88C8B2",  
        "GNNTrans": "#005F73",
    }
    sns.barplot(df_melted, x="class", y="test_f1", hue="model", palette=custom_palette)
    
    if split_scatter:
        # Stripplot: Individual test_f1 values, colored by split_key
        sns.stripplot(df_melted, x="class", y="test_f1", hue="split_key", dodge=True, 
                      edgecolor="black", linewidth=0.6, marker="o", palette="Set3")        

    
    plt.xlabel("Class")
    plt.ylabel("Test F1-score")
    plt.title("Mean F1-score per Class with Individual Split Values")
    plt.legend(
        title="Model", 
        bbox_to_anchor=(1.05, 1),  # move it outside
        loc='upper left', 
        borderaxespad=0.
    )
    plt.ylim((0,1))
    plt.show()

# %%
radius = 30
pct_mask_nodes = 0.1
decoder_type = 'nonlinear'

class_dict = {'test_f1/class_0': 'ND', 
              'test_f1/class_1': 'T1D',}

df_list = [df_GNN.copy(), df_PCATrans.copy(), df_GNNTrans.copy(), df_NeighTrans.copy()]

for idx in range(len(df_list)):    
    
    # Apply filters to the whole DataFrame
    df_list[idx] = df_list[idx][
        (df_list[idx]['radius'] == radius) &
        (df_list[idx]['pct_mask_nodes'] == pct_mask_nodes) &
        (df_list[idx]['decoder_type'] == decoder_type)
    ]

plot_f1_barplot(df_list, ['GNN', 'PCATrans', 'GNNTrans', 'NeighTrans'], False)

# %%
radius = 50
pct_mask_nodes = 0.1
decoder_type = 'linear'

class_dict = {'test_f1/class_0': 'ND', 
              'test_f1/class_1': 'T1D',}

df_list = [df_GNN.copy(), df_PCATrans.copy(), df_GNNTrans.copy(), df_NeighTrans.copy()]

## get sweep results for parameters
best_value = []
for df in [df_GNN.copy(), df_PCATrans.copy(), df_GNNTrans.copy(), df_NeighTrans.copy()]:
    best = df.iloc[df['test_acc'].idxmax()]
    best_value.append([best['radius'], best['pct_mask_nodes'], best['decoder_type'], best['runtime_seconds'], best['test_acc']])

print(best_value)

for idx in range(len(df_list)):    
    
    # Apply filters to the whole DataFrame
    df_list[idx] = df_list[idx][
        (df_list[idx]['radius'] == best_value[idx][0]) &
        (df_list[idx]['pct_mask_nodes'] == best_value[idx][1]) &
        (df_list[idx]['decoder_type'] == best_value[idx][2])
    ]

plot_f1_barplot(df_list, ['GNN', 'PCATrans', 'GNNTrans', 'NeighTrans'], False)

# %%
df_melted = pd.melt(df_gnn, 
                    id_vars=['split_key', 'model'],  # Keep the 'split_key' column
                    value_vars=[col for col in df.columns if col.startswith('test_f1/class_')],  # Select columns starting with 'test_f1/class_'
                    var_name='class',  # Name for the new column containing the class information
                    value_name='test_f1'  # Name for the new column containing the F1 score values
                   )

# %%
