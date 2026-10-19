import wandb
import pandas as pd
import numpy as np

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

import os

import yaml
import scanpy as sc

import interscale 
from yacs.config import CfgNode as CN

from interscale.config import load_config #, config_from_wandb_run


def _mask_cfg(run, key, legacy_key=None):
    """Read a dataset masking setting off a wandb run, whether or not the sweep varied it.

    wandb stores two views of the config. A sweep parameter appears as a FLAT dotted key
    (``dataset.mask_percentage``) carrying that trial's value. Everything else appears in the
    NESTED dict logged by ``main_sweep.py``'s ``wandb.init(config=cfg)``, which runs after the
    registry pair is resolved but before the trial is applied -- so a value that comes from a task
    file rather than from the sweep (``node_reg_genemask``'s ``mask_strategy: gene``, say) is only
    ever in the nested copy.

    Checking the flat key alone reported every gene-masking run as ``mask_strategy: node``, and
    left the rate ``None`` for any sweep that did not pin it. ``legacy_key`` covers sweeps
    registered before ``pct_mask_nodes`` was split into strategy + percentage.

    KNOWN GAP: an ARM sweep (``mask_granularity_legnini.yaml``) varies these through its ``arms:``
    block, which is expanded inside the trial and never round-trips through wandb -- only the arm
    NAME does, as ``run.config["arm"]``. Both views therefore show the arm-less base config for
    every trial, so these two columns are the same for all of them and are NOT the trial's real
    setting. Group an arm sweep by ``arm`` and read the rate off the yaml.
    """
    nested = run.config.get("dataset") or {}
    for candidate in (key, legacy_key):
        if candidate is None:
            continue
        if (flat := run.config.get(f"dataset.{candidate}")) is not None:
            return flat
        if (value := nested.get(candidate)) is not None:
            return value
    return None


class Wandb_evaluation():
    
    def __init__(self, model, sweep_id, local_component: bool, global_component: bool, sweep_goal: str, classes: list):
        """
        model: str
            Model name, e.i. InterScale, GCN, ...
        sweep_id: str 
            ID from WandB run
        local_component: bool
            Whether local component is used or not
        global_component: bool
            Whether global component is used or not
        sweep_goal: robustenss, parameter
        calsses: list of class names
        """
        self.model = model
        self.sweep_id = sweep_id
        self.sweep_goal = sweep_goal
        self.classes = classes
        
        api = wandb.Api()
        self.entity, self.project = "francesca-drummer", "InterScale_hyperparameter_sweep"  
    
        # Get all runs associated with the sweep
        sweep_runs = api.sweep(f"{self.entity}/{self.project}/{self.sweep_id}").runs

        data = []
        for run in sweep_runs:
            if run.state == 'finished':
                self.prediction_task = run.config['dataset']['prediction_task']
                self.prediction_level = run.config['dataset']['prediction_level']
                self.model_name = ""
                if local_component:
                    lc_name = run.config['model']['local_component']['name']
                    self.model_name += f'{lc_name}_'
                if global_component:
                    gc_name = run.config['model']['global_component']['name']
                    self.model_name += f'{gc_name}_'
                    
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "seed": run.config.get("optim.seed", None),
                    "state": run.state,  # finished, running, failed
                    # Kept under its original name: ~20 groupby/filter/plot sites in this file and
                    # in other notebooks key on the "pct_mask_nodes" COLUMN, so renaming it would
                    # break them. The interscale config key it reads was split into
                    # dataset.mask_strategy + dataset.mask_percentage, hence the fallback -- new
                    # sweeps carry the new key, sweeps registered before the split the old one.
                    #
                    # Read it together with mask_strategy: under "gene" this is a fraction of each
                    # cell's GENES, not a fraction of the cells, so the two are not comparable at
                    # equal values.
                    # `mask_percentage` is the honest name: mask_strategy says what a masked
                    # UNIT is, so this is a fraction of the cells under "node" and a fraction of
                    # every cell's genes under "gene". The two are NOT comparable at equal values.
                    "mask_percentage": _mask_cfg(run, "mask_percentage", "pct_mask_nodes"),
                    "mask_strategy": _mask_cfg(run, "mask_strategy"),
                    # Deprecated alias, same values. Kept only because ~20 groupby/filter/plot
                    # sites in this file and in other notebooks key on this column name. Do not
                    # read it as "percent of nodes" -- for a gene-masked run it is percent of
                    # genes, which is exactly the misreading the name invites. Prefer
                    # `mask_percentage`, and read it next to `mask_strategy`.
                    "pct_mask_nodes": _mask_cfg(run, "mask_percentage", "pct_mask_nodes"),
                    "radius": run.config.get("dataset.spatial_neigbors_kwargs.radius", None),
                    "decoder_type": run.config.get("model.decoder.type", None),
                    "runtime_seconds": run.summary.get("_runtime", None),
                    "total_parameters": run.summary.get("total_parameters", None),
                }
                if 'regression' in self.prediction_task:
                    run_data.update({
                        "test_r2": run.summary.get("test_r2", None),
                        "test_pearson_corr": run.summary.get("test_pearson_corr", None),
                    })
                elif 'classification' in self.prediction_task:
                    num_classes = run.config['dataset']['num_classes']
                    run_data.update({
                        "test_acc": run.summary.get("test_accuracy", None),
                        "test_f1_micro": run.summary.get("test_f1_micro", None),
                        "test_f1_macro": run.summary.get("test_f1_macro", None)
                    })
                    for class_idx in classes:
                        run_data[f'test_f1_class_{class_idx}'] = run.summary.get(f"test_f1_{class_idx}", None)
        
                if 'graph' in self.prediction_task:
                    run_data.update({
                        "split_key": run.config.get("dataset.split_key", None),
                    })
                if sweep_goal == 'hyperparameter':
                    self.hyperparameters = ["n_embed", "lr", "lr_warmup", "wd", "batch_size", "pct_mask_nodes"]
                    run_data.update({
                        "n_embed": run.config.get("model.n_embed", None),
                        "lr": run.config.get("optim.lr", None),
                        "wd": run.config.get("optim.wd", None),
                        "lr_warmup": run.config.get("optim.lr_warmup", None),
                        "batch_size": run.config.get("dataset.batch_size", None),
                        "seed": run.config['optim']['seed'], #overwrite because not variable
                        "radius": run.config['dataset']['spatial_neigbors_kwargs']['radius'], #overwrite because not variable
                    }),
                    if local_component:
                        self.local_component_params = ["LC_num_layers", "LC_hidden_dim", "LC_dropout"]
                        run_data.update({
                            "LC_num_layers": run.config.get("model.local_component.parameters.num_layers", None),
                            "LC_hidden_dim": run.config.get("model.local_component.parameters.hidden_dim", None),
                            "LC_dropout": run.config.get("model.local_component.parameters.dropout_local", None),
                        })
                    if global_component:
                        self.global_component_params = ["GC_n_heads", "GC_num_layers", "GC_dim_feedforward", "GC_hidden_dim", "GC_dropout"]
                        run_data.update({
                            "GC_n_heads": run.config.get("model.global_component.parameters.n_heads", None),
                            "GC_num_layers": run.config.get("model.global_component.parameters.num_layers", None),
                            "GC_dim_feedforward": run.config.get("model.global_component.parameters.dim_feedforward", None),
                            "GC_hidden_dim": run.config.get("model.global_component.parameters.hidden_dim", None),
                            "GC_dropout": run.config.get("model.global_component.parameters.dropout_global", None),
                        })
                
                data.append(run_data)
        self.df = pd.DataFrame(data)

    def filter_runs(self, df=None, exclude_parameters=None):
        """
        Filter out runs based on specific parameter values.
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            DataFrame to filter. If None, uses self.df
        exclude_parameters : dict, optional
            Dictionary where keys are column names and values are lists of values to exclude.
            Example: {"radius": [0], "pct_mask_nodes": [0.5, 0.8]}
        
        Returns:
        --------
        pandas.DataFrame
            Filtered DataFrame with specified parameter values removed.
        """
        if df is None:
            df = self.df.copy()
        
        if exclude_parameters is None:
            return df
        
        mask = pd.Series(True, index=df.index)
        for param, values in exclude_parameters.items():
            if param in df.columns:
                mask &= ~df[param].isin(values)
            else:
                print(f"Warning: parameter '{param}' not found in DataFrame columns")
        
        return df[mask]
    
    def get_dataframe(self):
        return self.df
    
    def get_mean_and_std(self):
        # Compute mean and standard deviation
        if self.prediction_task == 'regression':
            return self.df.groupby(["pct_mask_nodes", "radius"],  dropna=False).agg(
                mean_test_r2=("test_r2", "mean"),
                std_test_r2=("test_r2", "std"),
                mean_test_pearson=("test_pearson_corr", "mean"),
                std_test_pearson=("test_pearson_corr", "std"),
                mean_run_time=("runtime_seconds", "mean"),
                std_run_time=("runtime_seconds", "std"),
            ).reset_index()
        elif self.prediction_task == 'classification':
            return self.df.groupby(["pct_mask_nodes", "radius"],  dropna=False).agg(
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "std"),
                mean_test_f1_class_0=(f"test_f1_class_{self.classes[0]}", "mean"),
                std_test_f1_class_0=(f"test_f1_class_{self.classes[0]}", "std"),
                mean_test_f1_class_1=(f"test_f1_class_{self.classes[1]}", "mean"),
                std_test_f1_class_1=(f"test_f1_class_{self.classes[1]}", "std"),
            ).reset_index()
        else:
            raise ValueError(f"sweep_goal must be 'classification' or 'regression', got '{self.sweep_goal}'")


    def get_best_run_id(self, metric: str = 'test_acc', maximize: bool = True):
        """
        Get the run ID with the best performance for a given metric.
        
        Parameters:
        -----------
        metric : str
            Column name in self.df to evaluate (e.g., 'test_acc', 'test_r2', 'test_f1_micro/avg')
        maximize : bool, optional (default=True)
            If True, returns the run with the highest metric value.
            If False, returns the run with the lowest metric value (useful for loss metrics).
        
        Returns:
        --------
        str
            The run ID of the best performing run.
        
        Raises:
        -------
        ValueError
            If the metric is not found in the DataFrame columns.
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric '{metric}' not found in DataFrame. Available columns: {list(self.df.columns)}")
        
        # Drop rows with NaN values for the metric
        valid_df = self.df.dropna(subset=[metric])
        
        if len(valid_df) == 0:
            raise ValueError(f"No valid (non-NaN) values found for metric '{metric}'")
        
        if maximize:
            best_idx = valid_df[metric].idxmax()
        else:
            best_idx = valid_df[metric].idxmin()
        
        best_run_id = valid_df.loc[best_idx, 'id']
        best_value = valid_df.loc[best_idx, metric]
        
        print(f"Best run for {metric}: {best_run_id} (value: {best_value:.4f})")
        
        return best_run_id

    
    def summary_df(self, df, metric, decoder_type = "linear"):
        """
        metric: str - column in df
        """
        # Filter data for linear and non-linear decoder types
        decoder_df = df[df["decoder_type"] == "linear"]
    
        # Group by radius and pct_mask_modes, then compute mean & std across seeds for linear
        decoder_summary_df = decoder_df.groupby(["radius", "pct_mask_nodes"]).agg(
            mean_test_r2=(metric, "mean"),
            std_test_r2=(metric, "std"),
            mean_run_time=("runtime_seconds", "mean"),
            std_run_time=("runtime_seconds", "std"),
        ).reset_index()
    
        # Display the tables for linear and non-linear decoders
        print(f"{decoder_type} Decoder Summary ({metric}:")
        print(decoder_summary_df)
        return decoder_summary_df
    
    def plot_robustness(self, metric="test_r2", save_path = None, dropna=True):
        """
        Plots the robustness of a model's performance across different radii and 
        percentages of masked nodes.
    
        Parameters:
        -----------
        df : pandas.DataFrame
            A DataFrame containing columns 'radius', 'pct_mask_nodes', and the specified metric.
        metric : str, optional (default="test_r2")
            The metric to plot on the y-axis. Can be "test_r2" for model performance 
            or "runtime_seconds" for computational cost.
    
        Returns:
        --------
        None
            Displays a line plot showing how the specified metric changes with radius 
            and percentage of masked nodes.
    
        Notes:
        ------
        - If metric is "test_r2", the y-axis is limited to [0, 1] and labeled "Mean Test R² Score".
        - If metric is "runtime_seconds", the y-axis is limited to [0, 1500] and labeled "Mean runtime in seconds".
        - The standard deviation is shown as a shaded region.
        """
        df = self.df.dropna(subset=["radius", "pct_mask_nodes"]) if dropna else self.df
    
        plt.figure(figsize=(4, 6))
        sns.lineplot(
            data=df,
            x="radius",
            y=metric,
            hue="pct_mask_nodes",
            marker="o",
            palette="coolwarm",
            errorbar=("sd")
        )
        
        # Formatting
        plt.xlabel("Radius")
        plt.legend(title="Pct Mask Nodes")
        if metric == "runtime_seconds":
            plt.ylim(0, 5000)  # Set y-axis range
            plt.ylabel("Mean runtime in seconds")
        else:
            plt.ylim(0, 1.2)  # Set y-axis range
            plt.ylabel(f"Mean {metric} Score")
        plt.grid(True)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'{self.model_name}_{self.prediction_level}_{self.prediction_task}_radius_vs_{metric}.jpg'), dpi=1200)
        
        # Show the plot
        plt.show()
    
    def plot_parameter_space(self, 
                             metric: str = 'test_r2', 
                             relevant_params = [], 
                             save_path: str = None,
                             exclude_parameters: None | dict = None
        ):
        """Plot the total number of model parameters as a function of performance.

        relevant_params: List
            If List is not empty, plot the relevant model or optim parameters. For example, select any of the self.hyperparameter lists saved in the initialization.
        """
        assert metric in self.df.columns.values

        plot_df = self.filter_runs(exclude_parameters=exclude_parameters)
    
        # Apply LOWESS smoothing
        smoothed = lowess(plot_df[metric], plot_df['total_parameters'], frac=0.4)  # frac controls smoothness
        # Create scatterplot with trend line
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='total_parameters', y=metric, data=plot_df, label='Data Points')
        
        # Plot smoothed trend
        plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOWESS Curve')
        
        # Labels and title
        plt.xlabel("Total Parameters")
        plt.ylabel(f"{metric}")
        plt.title(f"Trend of {metric} with Increasing Parameters")
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'{self.model_name}_{self.prediction_level}_{self.prediction_task}_nr_params_vs_{metric}.jpg'), dpi=1200)
        
        plt.show()

        # Plot relevant parameters in subplots
        if len(relevant_params) > 0:
            n_params = len(relevant_params)
            n_cols = min(4, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols  # ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.atleast_1d(axes).flatten()  # ensure axes is always a flat array
            
            for i, param in enumerate(relevant_params):
                sns.regplot(x=param, y=metric, data=plot_df, ax=axes[i], scatter_kws={'alpha': 0.6})
                axes[i].set_xlabel(param)
                axes[i].set_ylabel(metric)
                axes[i].set_title(f'{param} vs {metric}')
            
            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            if save_path is not None:
                fig.savefig(os.path.join(save_path, f"{self.model_name}_{self.prediction_level}_{self.prediction_task}_relevantParams_vs_{metric}.jpg"), dpi=1200)
            
            plt.show()
    
        if save_path is not None:
            print(f'Saved figures to {save_path}')

    def load_model(self, best_run_id: str, adata = None):
        """Load best model artifact from WandB according to metric."""
        
        # 1. Get best run and associated config
        api = wandb.Api()
        best_run = api.run(f"{self.entity}/{self.project}/{best_run_id}")
        config_dict = best_run.config
        
        cfg = CN(config_dict)
        cfg.optim.accelerator = 'cpu'
        cfg.model.decoder.dual_decoder = False
        cfg.model.global_component.parameters.type_gex_embedding = None
        cfg.freeze()  # Optional: make it immutable
        
        # 4. Download the model artifact
        artifact = list(best_run.logged_artifacts())[0]
        artifact_dir = artifact.download()
        print(f"Model artifact downloaded to: {artifact_dir}")

        if adata is None:
            adata = sc.read_h5ad(cfg.dataset.h5ad_data)
        
        # 5. Setup AnnData
        interscale.model.CombinedModel._setup_anndata(
            adata=adata, 
            prediction_task=cfg.dataset.prediction_task, 
            layer_key=cfg.dataset.layer_key, 
            sample_key_list=cfg.dataset.sample_key, 
            prediction_obs=cfg.dataset.prediction_obs, 
            group_key=cfg.dataset.group_label, 
            view_registry=False
        )
        
        # 6. Load model from the artifact directory
        combined_model = interscale.model.CombinedModel.load(
            artifact_dir,
            adata, 
            cfg= cfg,
            model_name = f"{artifact_dir}/model",
            local_component=True, 
            global_component=True, 
            wandb_save=False
        )
        
        print(f"Model loaded successfully from run: {best_run_id}")

        return combined_model, cfg, adata

    def export_config_to_yaml(self, best_run_id: str, save_path="best_config.yaml"):
        """Export the best run's config to a YAML file for training with the best hyperparameters.

        Uses all InterScale config variables (wandb, model, optim, dataset,
        local/global component parameters) and writes a single YAML that can be
        loaded with load_config_from_yaml() for training.

        Parameters
        ----------
        metric : str, optional
            Metric to select the best run (e.g. 'test_acc', 'test_r2'). Best = max.
        save_path : str, optional
            Path for the output YAML file.

        Returns
        -------
        str
            Path to the saved YAML file.
        """
        api = wandb.Api()
        best_run = api.run(f"{self.entity}/{self.project}/{best_run_id}")
        cfg = config_from_wandb_run(best_run, save_yaml_path=save_path)
        print(f"Config saved to {save_path}")
        return save_path

def set_plot_configs(BASE_DIR_REPO):
    # Load config
    with open(os.path.join(BASE_DIR_REPO, "figures/config.yml"), "r") as f:
        config = yaml.safe_load(f)
    
    general_config = config['plot_configs']['general']
    model_palette = config['palettes']['Models']
    
    # Apply general plot settings
    plt.rcParams['figure.dpi'] = general_config['dpi']
    plt.rcParams['savefig.dpi'] = general_config['dpi_save']
    plt.rcParams['font.family'] = general_config['font_family']
    plt.rcParams['font.size'] = 10

    return general_config, model_palette
    

def plot_f1_across_seeds(wandb_evaluations, 
                         radius, 
                         pct_mask_nodes, 
                         BASE_DIR_REPO, 
                         height=4, 
                         aspect=0.7,
                        save_path = None):
    """
    Plot mean and standard deviation of per-class F1 scores across seeds.
    Uses seaborn catplot with one facet per class and one bar per model.
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
        List of Wandb_evaluation instances, one per model
    radius : float or int
        Radius value to filter by
    pct_mask_nodes : float or int
        Percentage of masked nodes to filter by
    config_path : str, optional
        Path to config.yml file (default: 'config.yml')
    height : float, optional
        Height of each facet in inches (default: 4)
    aspect : float, optional
        Aspect ratio of each facet (default: 0.7)
    save_path: str
        Path name from repo to save the plot. 
    
    Returns:
    --------
    g : seaborn FacetGrid
        The catplot object
    stats_df : pd.DataFrame
        DataFrame with mean and std for each class and model
    plot_data : pd.DataFrame
        Long-form data used for plotting
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)

    # Process each Wandb_evaluation instance
    all_data = []
    model_names = []
    
    for wandb_eval in wandb_evaluations:
        model_name = wandb_eval.model
        df = wandb_eval.df
        classes = wandb_eval.classes
        
        model_names.append(model_name)
        
        # Filter by specified parameters
        df_filtered = df[
            (df['radius'] == radius) &
            (df['pct_mask_nodes'] == pct_mask_nodes)
        ].copy()
        
        if df_filtered.empty:
            print(f"Warning: No data found for {model_name} with radius={radius}, pct_mask_nodes={pct_mask_nodes}")
            continue
        
        # Identify F1 score columns
        f1_cols = [col for col in df_filtered.columns if col.startswith('test_f1_class_')]
        
        if not f1_cols:
            raise ValueError(f"No 'test_f1_class_*' columns found in {model_name}")
        
        # Add model label
        df_filtered['model'] = model_name
        
        # Collect the data
        all_data.append(df_filtered)
    
    if not all_data:
        raise ValueError("No data found matching the specified parameters")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Identify F1 columns
    f1_cols = [col for col in combined_df.columns if col.startswith('test_f1_class_')]
    
    # Reshape to long format for seaborn
    plot_data = combined_df.melt(
        id_vars=['model'],
        value_vars=f1_cols,
        var_name='class',
        value_name='f1_score'
    )
    
    # Clean up class names
    plot_data['class'] = plot_data['class'].str.replace('test_f1/class_', '')
    
    # Calculate statistics for reference
    stats_df = plot_data.groupby(['model', 'class'])['f1_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reset_index()

    print(stats_df)
    
    # Create color palette based on model names from config
    palette = []
    for model_name in model_names:
        if model_name in model_palette:
            palette.append(model_palette[model_name])
        else:
            print(f"Warning: Model '{model_name}' not found in config palette. Using default color.")
            palette.append(None)
    
    # If all models are in config, use the palette; otherwise let seaborn handle it
    use_palette = palette if all(c is not None for c in palette) else None
    
    # Create the catplot
    g = sns.catplot(
        data=plot_data,
        kind="bar",
        x="class",
        y="f1_score",
        hue="model",
        height=height,
        aspect=aspect,
        errorbar="sd",  # Standard deviation error bars
        capsize=0.1,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.8,
        palette=use_palette,
        legend=False
    )
    
    # Customize the plot with config settings
    g.set_axis_labels(
        "Model", 
        "Test F1 Score", 
        fontsize=general_config['legend_fontsize'],
        fontweight=general_config['legend_fontweight']
    )
    g.set_titles(
        "Class: {col_name}", 
        fontsize=general_config['title_fontsize'],
        fontweight=general_config['title_fontweight']
    )
    
    # Set y-axis limits and grid
    for ax in g.axes.flat:
        ax.set_ylim([0, 1.2])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    # Rotate x-axis labels if needed
    for ax in g.axes.flat:
        ax.tick_params(axis='x', rotation=45)
        for label in ax.get_xticklabels():
            label.set_ha('right')
    
    # Overall title
    g.fig.suptitle(
        f'F1 Scores Across Seeds (radius={radius}, pct_mask={pct_mask_nodes})',
        fontsize=general_config['title_fontsize'],
        fontweight=general_config['title_fontweight'],
        y=1.02
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            os.path.join(BASE_DIR_REPO, 'figures/', f'{save_path}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    return g, stats_df, plot_data

def plot_class_f1_comparison(wandb_evaluations, radius=None, pct_mask_nodes=None, BASE_DIR_REPO=None,
                              save_path=None, figsize=(10, 6)):
    """
    Plot class-specific F1 scores as grouped barplot comparing multiple models across seeds.
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
        List of Wandb_evaluation instances, one per model
    radius : float or int, optional
        Radius value to filter by (if None, uses all data)
    pct_mask_nodes : float or int, optional
        Percentage of masked nodes to filter by (if None, uses all data)
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (default: (10, 6))
    palette : dict or list, optional
        Color palette for models
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    stats_df : pd.DataFrame with mean and std for each class and model
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)
    
    all_data = []
    
    for wandb_eval in wandb_evaluations:
        df = wandb_eval.df.copy()
        classes = wandb_eval.classes
        model_name = wandb_eval.model
        
        # Apply filters if specified
        if radius is not None:
            df = df[df['radius'] == radius]
        if pct_mask_nodes is not None:
            df = df[df['pct_mask_nodes'] == pct_mask_nodes]
        
        if df.empty:
            print(f"Warning: No data for {model_name} with specified filters")
            continue
        
        # Get F1 columns
        f1_cols = [f'test_f1_class_{c}' for c in classes]
        existing_cols = [c for c in f1_cols if c in df.columns]
        
        if not existing_cols:
            print(f"Warning: No F1 columns found for {model_name}")
            continue
        
        # Reshape to long format
        plot_df = df.melt(
            id_vars=['seed'] if 'seed' in df.columns else [],
            value_vars=existing_cols,
            var_name='class',
            value_name='f1_score'
        )
        plot_df['class'] = plot_df['class'].str.replace('test_f1_class_', '')
        plot_df['model'] = model_name
        all_data.append(plot_df)
    
    if not all_data:
        raise ValueError("No valid data found for any model")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate statistics
    stats_df = combined_df.groupby(['model', 'class'])['f1_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(
        data=combined_df,
        x='class',
        y='f1_score',
        hue='model',
        errorbar='sd',
        capsize=0.1,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.8,
        palette=model_palette,
        ax=ax
    )
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    
    title = 'Class F1 Scores Comparison Across Seeds'
    if radius is not None or pct_mask_nodes is not None:
        filters = []
        if radius is not None:
            filters.append(f'radius={radius}')
        if pct_mask_nodes is not None:
            filters.append(f'pct_mask={pct_mask_nodes}')
        title += f' ({", ".join(filters)})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(title='Model', loc='upper right', framealpha=0.9)
    
    # Rotate x labels if many classes
    if len(combined_df['class'].unique()) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(
            os.path.join(BASE_DIR_REPO, '{}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    print("\nStatistics Summary:")
    print(stats_df.to_string(index=False))
    
    return fig, ax, stats_df

def plot_class_f1_robustness(wandb_evaluations, class_idx, pct_mask_nodes=None, BASE_DIR_REPO=None, y_max='auto',
                              save_path=None, figsize=(8, 6)):
    """
    Plot class-specific F1 score robustness across radius, comparing multiple models.
    
    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
        List of Wandb_evaluation instances, one per model
    class_idx : str or int
        Class index to plot F1 for
    pct_mask_nodes : float or int, optional
        Percentage of masked nodes to filter by (if None, uses all data)
    BASE_DIR_REPO : str, optional
        Base directory for config files
    save_path : str, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size (default: (8, 6))
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    stats_df : pd.DataFrame with mean and std for each radius and model
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)
    
    all_data = []
    
    for wandb_eval in wandb_evaluations:
        df = wandb_eval.df.copy()
        model_name = wandb_eval.model
        
        # Apply filter if specified
        if pct_mask_nodes is not None:
            df = df[df['pct_mask_nodes'] == pct_mask_nodes]
        
        if df.empty:
            print(f"Warning: No data for {model_name} with specified filters")
            continue
        
        # Get F1 column for specified class
        f1_col = f'test_f1_class_{class_idx}'
        
        if f1_col not in df.columns:
            print(f"Warning: {f1_col} not found for {model_name}")
            continue
        
        plot_df = df[['radius', 'seed', f1_col]].copy() if 'seed' in df.columns else df[['radius', f1_col]].copy()
        plot_df = plot_df.rename(columns={f1_col: 'f1_score'})
        plot_df['model'] = model_name
        all_data.append(plot_df)
    
    if not all_data:
        raise ValueError("No valid data found for any model")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate statistics
    stats_df = combined_df.groupby(['model', 'radius'])['f1_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reset_index()
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.lineplot(
        data=combined_df,
        x='radius',
        y='f1_score',
        hue='model',
        marker='o',
        errorbar='sd',
        palette=model_palette,
        ax=ax
    )
    
    ax.set_xlabel('Radius', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    
    title = f'Class {class_idx} F1 Robustness Across Models'
    if pct_mask_nodes is not None:
        title += f' (pct_mask={pct_mask_nodes})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Set y-axis limits
    if y_max == 'auto':
        y_max_val = combined_df['f1_score'].max() + 0.2
    else:
        y_max_val = y_max
    ax.set_ylim(0, y_max_val)
    
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(title='Model', loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(
            os.path.join(save_path, f'f1_robustness_class_{class_idx}_{pct_mask_nodes}.jpg'),
            dpi=300, bbox_inches='tight'
        )
    
    plt.show()
    
    print("\nStatistics Summary:")
    print(stats_df.to_string(index=False))
    
    return fig, ax, stats_df

def plot_mse_per_celltype(csv_path,
                          BASE_DIR_REPO,
                          models=('InterScale', 'GITIII'),
                          sort_by='InterScale',
                          figsize=(10, 5),
                          gitiii_color="#4D4D4D",
                          annotate_n_cells=True,
                          save_path=None):
    """
    Plot per-cell-type reconstruction MSE (mean +/- sd) for InterScale and GITIII
    as a grouped barplot, in the same style as `plot_f1_across_seeds`.

    Parameters:
    -----------
    csv_path : str
        Path to the per-cell-type comparison csv. Expected columns:
        'cell_type', '<model>_mse' and one sd column per model
        ('InterScale_sd_across_5_runs', 'GITIII_sd_across_cells'),
        optionally 'n_cells'.
    BASE_DIR_REPO : str
        Repository base dir, used to load `figures/config.yml`.
    models : tuple of str, optional
        Models to plot, in bar order (default: ('InterScale', 'GITIII')).
    sort_by : str or None, optional
        Model whose MSE defines the cell-type ordering (default: 'InterScale').
        If None, the csv order is kept.
    figsize : tuple, optional
        Figure size (default: (10, 5)).
    gitiii_color : str, optional
        Colour for GITIII bars (default: dark grey "#4D4D4D"). InterScale takes
        its colour from `palettes.Models` in config.yml.
    annotate_n_cells : bool, optional
        Annotate each cell type with its number of cells (if 'n_cells' present).
    save_path : str, optional
        Path name relative to `<BASE_DIR_REPO>/figures/` to save the plot as jpg.

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    plot_data : pd.DataFrame
        Long-form data used for plotting (cell_type, model, mse, sd, n_cells).
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)

    df = pd.read_csv(csv_path)

    # Resolve the mse / sd column for each model (sd column names differ per model)
    records = []
    for model in models:
        mse_col = f'{model}_mse'
        if mse_col not in df.columns:
            raise ValueError(f"No column '{mse_col}' found in {csv_path}")

        sd_candidates = [
            col for col in df.columns
            if col.startswith(f'{model}_sd') or col == f'{model}_std'
        ]
        if not sd_candidates:
            print(f"Warning: no sd column found for '{model}', plotting without error bars")
            sd_col = None
        else:
            sd_col = sd_candidates[0]

        sub = pd.DataFrame({
            'cell_type': df['cell_type'],
            'model': model,
            'mse': df[mse_col],
            'sd': df[sd_col] if sd_col is not None else np.nan,
            'sd_source': sd_col if sd_col is not None else 'none',
        })
        if 'n_cells' in df.columns:
            sub['n_cells'] = df['n_cells']
        records.append(sub)

    plot_data = pd.concat(records, ignore_index=True)

    # Cell-type ordering
    if sort_by is not None and f'{sort_by}_mse' in df.columns:
        order = df.sort_values(f'{sort_by}_mse')['cell_type'].tolist()
    else:
        order = df['cell_type'].tolist()

    # Colours: InterScale from config palette, GITIII dark grey
    colors = {}
    for model in models:
        if model == 'GITIII':
            colors[model] = gitiii_color
        elif model in model_palette:
            colors[model] = model_palette[model]
        else:
            print(f"Warning: Model '{model}' not found in config palette. Using grey.")
            colors[model] = "#999999"

    # Grouped barplot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(order))
    n_models = len(models)
    width = 0.8 / n_models

    for i, model in enumerate(models):
        sub = plot_data[plot_data['model'] == model].set_index('cell_type').loc[order]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(
            x + offset,
            sub['mse'].values,
            width=width,
            yerr=sub['sd'].values,
            capsize=3,
            label=model,
            color=colors[model],
            edgecolor='black',
            linewidth=1.0,
            alpha=0.8,
            error_kw={'elinewidth': 1.0, 'ecolor': 'black'},
        )

    xticklabels = list(order)
    if annotate_n_cells and 'n_cells' in plot_data.columns:
        n_cells = plot_data.drop_duplicates('cell_type').set_index('cell_type')['n_cells']
        xticklabels = [f'{ct}\n(n={int(n_cells[ct]):,})' for ct in order]

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right')
    ax.set_xlabel(
        'Cell type',
        fontsize=general_config['legend_fontsize'],
        fontweight=general_config['legend_fontweight'],
    )
    ax.set_ylabel(
        'Reconstruction MSE',
        fontsize=general_config['legend_fontsize'],
        fontweight=general_config['legend_fontweight'],
    )
    ax.set_title(
        'Reconstruction error per cell type',
        fontsize=general_config['title_fontsize'],
        fontweight=general_config['title_fontweight'],
    )

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            os.path.join(BASE_DIR_REPO, 'figures/', f'{save_path}.jpg'),
            dpi=300, bbox_inches='tight'
        )

    plt.show()

    # Note: sd columns can have different meanings per model (see 'sd_source')
    print(plot_data.groupby('model')['sd_source'].first().to_string())

    return fig, ax, plot_data


# Mapping from the model names used in the benchmark result csvs to the
# palette keys in figures/config.yml, plus display labels for the x-axis.
MODEL_ALIASES = {
    'Interscale_Dual': 'DualInterScale',
    'InterScale_Dual': 'DualInterScale',
    'Interscale_Global_PCA': 'PCATransformer',
    'InterScale_Global_PCA': 'PCATransformer',
}

MODEL_LABELS = {
    'Interscale_Dual': 'InterScale (Dual)',
    'InterScale_Dual': 'InterScale (Dual)',
    'Interscale_Global_PCA': 'InterScale (PCA)',
    'InterScale_Global_PCA': 'InterScale (PCA)',
    'Steamboat': 'Steamboat',
    'Amici': 'AMICI',
    'AMICI': 'AMICI',
    'GITIII': 'GITIII',
}

# Baselines are plotted in greys, InterScale variants take their config colour.
BASELINE_COLORS = {
    'GITIII': "#4D4D4D",
    'Steamboat': "#4D4D4D",
    'Amici': "#8C8C8C",
    'AMICI': "#8C8C8C",
}

DATASET_LABELS = {
    'meltonjimenez_25': 'Melton & Jimenez 2025',
    'melton25': 'Melton & Jimenez 2025',
    'legnini_23': 'Legnini 2023',
    'legnini23': 'Legnini 2023',
}



def plot_cosine_sim_comparison(csv_path,
                               dataset,
                               BASE_DIR_REPO,
                               models=None,
                               figsize=None,
                               annotate_seed=False,
                               bar_width_in=0.38,
                               bar_gap_in=0.04,
                               height=4.5,
                               save_path=None):
    """
    Plot the cosine similarity per model for one dataset, in the same style as
    `plot_f1_across_seeds` / `plot_mse_per_celltype`.

    For models run with several seeds (the InterScale variants) only the *best*
    run is shown, i.e. the seed with the highest cosine similarity. Models run
    once (Steamboat, AMICI) contribute their single value. Every bar therefore
    corresponds to exactly one run and no error bars are drawn.

    Parameters:
    -----------
    csv_path : str
        Path to the combined benchmark csv. Expected columns: 'model',
        'dataset', 'cosine_sim', optionally 'seed'.
    dataset : str
        Dataset to plot, as it appears in the 'dataset' column
        (e.g. 'meltonjimenez_25' or 'legnini_23').
    BASE_DIR_REPO : str
        Repository base dir, used to load `figures/config.yml`.
    models : list of str, optional
        Models to plot, in bar order. Defaults to all models present for this
        dataset, InterScale variants first and baselines after.
    figsize : tuple, optional
        Figure size. By default the width is derived from `bar_width_in`,
        `bar_gap_in` and the number of models, so the bars keep the same
        physical width no matter how many models are plotted.
    annotate_seed : bool, optional
        Append the seed of the selected run to the x tick label, for models
        that were run with more than one seed (default: False -- the seeds are
        in the returned dataframe and would make the tick labels long).
    bar_width_in : float, optional
        Physical width of a single bar in inches (default: 0.38).
    bar_gap_in : float, optional
        Physical gap between neighbouring bars in inches (default: 0.04, i.e.
        the bars almost touch).
    height : float, optional
        Figure height in inches (default: 4.5). Ignored if `figsize` is given.
    save_path : str, optional
        Path name relative to `<BASE_DIR_REPO>/figures/` to save the plot as jpg.

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    best_df : pd.DataFrame
        The selected best run per model, with the number of runs it was picked
        from ('n_runs') for reference.
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)

    df = pd.read_csv(csv_path)

    for col in ('model', 'dataset', 'cosine_sim'):
        if col not in df.columns:
            raise ValueError(f"No column '{col}' found in {csv_path}")

    df_ds = df[df['dataset'] == dataset].copy()
    if df_ds.empty:
        raise ValueError(
            f"No rows for dataset='{dataset}' in {csv_path}. "
            f"Available: {sorted(df['dataset'].unique())}"
        )

    # Keep only the best run per model (highest cosine similarity)
    n_runs = df_ds.groupby('model')['cosine_sim'].size().rename('n_runs')
    best_df = (
        df_ds.loc[df_ds.groupby('model')['cosine_sim'].idxmax()]
        .join(n_runs, on='model')
        .reset_index(drop=True)
    )

    # Default order: InterScale variants first, then baselines
    if models is None:
        present = best_df['model'].tolist()
        interscale = [m for m in present if m.lower().startswith('interscale')]
        baselines = [m for m in present if m not in interscale]
        models = sorted(interscale) + sorted(baselines)
    else:
        missing = [m for m in models if m not in set(best_df['model'])]
        if missing:
            print(f"Warning: no data for {missing} in dataset '{dataset}', skipping")
            models = [m for m in models if m not in missing]

    best_df = best_df.set_index('model').loc[models].reset_index()

    # Colours: baselines grey, InterScale variants from the config palette
    colors = []
    for model in models:
        if model in BASELINE_COLORS:
            colors.append(BASELINE_COLORS[model])
            continue
        key = MODEL_ALIASES.get(model, model)
        if key in model_palette:
            colors.append(model_palette[key])
        else:
            print(f"Warning: Model '{model}' (palette key '{key}') not found in "
                  f"config palette. Using grey.")
            colors.append("#999999")

    # Size the figure so that each bar is `bar_width_in` inches wide, rather
    # than padding the x-axis (which would only add empty space around the
    # bars). MARGIN_IN covers the y-axis label, ticks and a little breathing
    # room on the right.
    MARGIN_IN = 1.1
    n = len(models)
    if figsize is None:
        fig_width = MARGIN_IN + n * (bar_width_in + bar_gap_in) + 2 * bar_gap_in
        figsize = (fig_width, height)

    fig, ax = plt.subplots(figsize=figsize)

    # Narrow bars placed close together, so they almost touch. Work in inches
    # and let the x-limits span the full bar block plus one gap on each side.
    x = np.arange(n) * (bar_width_in + bar_gap_in)
    ax.bar(
        x,
        best_df['cosine_sim'].values,
        width=bar_width_in,
        color=colors,
        edgecolor='black',
        linewidth=1.0,
        alpha=0.8,
    )
    ax.set_xlim(
        x[0] - bar_width_in / 2 - bar_gap_in,
        x[-1] + bar_width_in / 2 + bar_gap_in,
    )

    xticklabels = []
    for _, row in best_df.iterrows():
        label = MODEL_LABELS.get(row['model'], row['model'].replace('_', ' '))
        seed = row.get('seed', np.nan)
        if annotate_seed and row['n_runs'] > 1 and pd.notna(seed):
            label = f'{label}, seed {int(seed)}'
        xticklabels.append(label)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=90, ha='center', va='top')
    ax.set_ylabel(
        'Cosine similarity',
        fontsize=general_config['legend_fontsize'],
        fontweight=general_config['legend_fontweight'],
    )
    ax.set_title(
        DATASET_LABELS.get(dataset, dataset),
        fontsize=general_config['title_fontsize'],
        fontweight=general_config['title_fontweight'],
    )

    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            os.path.join(BASE_DIR_REPO, 'figures/', f'{save_path}.jpg'),
            dpi=300, bbox_inches='tight'
        )

    plt.show()

    cols = [c for c in ('model', 'cosine_sim', 'seed', 'n_runs', 'epochs') if c in best_df.columns]
    print(best_df[cols].to_string(index=False))

    return fig, ax, best_df
