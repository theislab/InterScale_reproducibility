import wandb
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

import os

import yaml
import scanpy as sc

import interscale 
from yacs.config import CfgNode as CN

from interscale.config import load_config


class Wandb_evaluation():
    
    def __init__(self, model, sweep_id, local_component: bool, global_component: bool, sweep_goal: str, classes: list,
                 entity: str = "francesca-drummer", project: str = "InterScale_hyperparameter_sweep"):
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
        entity, project: str
            WandB entity/project the sweep belongs to
        """
        self.model = model
        self.sweep_id = sweep_id
        self.sweep_goal = sweep_goal
        self.classes = classes

        api = wandb.Api()
        self.entity, self.project = entity, project

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
                    "pct_mask_nodes": run.config.get("dataset.pct_mask_nodes", None),
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

    @classmethod
    def from_dataframe(cls, df, model, sweep_id, sweep_goal, classes,
                       model_name="", prediction_task=None, prediction_level=None):
        """
        Rebuild an evaluation from an already downloaded run table, without
        contacting WandB. Used by the figure scripts to regenerate plots offline
        from the cached CSVs (see scripts/graph_classification/).
        """
        obj = cls.__new__(cls)  # bypass __init__, which queries the WandB API
        obj.model = model
        obj.sweep_id = sweep_id
        obj.sweep_goal = sweep_goal
        obj.classes = classes
        obj.model_name = model_name
        obj.prediction_task = prediction_task
        obj.prediction_level = prediction_level
        obj.entity = obj.project = None
        obj.df = df
        return obj

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


# --------------------------------------------------------------------------- #
# statistical comparison across seeds
#
# Same approach as InterScale's "Evaluation comparison" (mirroring
# https://github.com/theislab/tissue): for each scenario (here, each x-group
# — a class or a filter panel) the best-performing model is the one with the
# highest mean score across repeated runs (seeds); every other model is then
# compared against it with a two-sided t-test, p < 0.05 counts as
# statistically significant.
# --------------------------------------------------------------------------- #
def _stars(p_value):
    """Two-sided-t-test p-value -> significance label. Every comparison gets a
    bracket, significant or not — 'ns' when p >= 0.05 — so a missing bracket
    never has to be read as "wasn't tested"."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def _best_vs_rest_ttest(groups: dict) -> tuple:
    """Two-sided t-test between the best-performing group and every other group.

    Parameters
    ----------
    groups : dict[str, array-like]
        Raw per-seed scores for each group (model).

    Returns
    -------
    best : str
        Key of the group with the highest mean.
    results : dict[str, dict]
        For every non-best group with >=2 seeds: {'p_value': float, 'significant': bool}.
    """
    groups = {k: np.asarray(v, dtype=float) for k, v in groups.items() if len(v) > 0}
    groups = {k: v[~np.isnan(v)] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    if len(groups) < 2:
        return None, {}

    means = {k: v.mean() for k, v in groups.items()}
    best = max(means, key=means.get)

    results = {}
    if len(groups[best]) >= 2:
        for k, v in groups.items():
            if k == best or len(v) < 2:
                continue
            _, p_value = scipy_stats.ttest_ind(groups[best], v, equal_var=False, nan_policy="omit")
            results[k] = {"p_value": p_value, "significant": bool(p_value < 0.05)}
    return best, results


def _ttest_long_table(ttest_by_group: dict, group_col: str) -> pd.DataFrame:
    """Flatten {group_key: (best, results)} into a long table for the stats CSV."""
    rows = []
    for group_key, (best, results) in ttest_by_group.items():
        if best is None:
            continue
        rows.append({group_col: group_key, "model": best, "best_model": best,
                     "p_value_vs_best": np.nan, "significant": False})
        for model, res in results.items():
            rows.append({group_col: group_key, "model": model, "best_model": best,
                         "p_value_vs_best": res["p_value"], "significant": res["significant"]})
    return pd.DataFrame(rows)


def _draw_significance_bracket(ax, x1, x2, y, label):
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, color="black", clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=9, clip_on=False)


def _annotate_ttest_grouped(ax, df, x, y, hue, order, hue_order) -> dict:
    """Annotate a dodged/grouped bar plot: within each x-group, draw a bracket
    from the best-performing hue level to every other hue level (two-sided
    t-test). Every non-best hue level gets a bracket, labeled with stars when
    p < 0.05 and 'ns' otherwise — comparisons are never silently omitted.

    Assumes one bar container per hue level in `hue_order` (as produced by
    seaborn's bar/catplot with `order=` and `hue_order=` set to the same
    values passed here), and expands the y-limits to fit the brackets.

    Returns {x_group: (best_model, {other_model: {'p_value', 'significant'}})}.
    """
    if len(ax.containers) < len(hue_order):
        return {}  # nothing was drawn on this axis (e.g. an empty facet)

    containers = dict(zip(hue_order, ax.containers))
    ymin, ymax = ax.get_ylim()
    step = (ymax - ymin) * 0.09
    top = ymax
    ttest_by_group = {}

    for xi_idx, xi in enumerate(order):
        groups = {h: df.loc[(df[x] == xi) & (df[hue] == h), y].dropna().to_numpy() for h in hue_order}
        groups = {h: v for h, v in groups.items() if len(v) > 0}
        best, results = _best_vs_rest_ttest(groups)
        ttest_by_group[xi] = (best, results)
        if not results:
            continue

        bar_x = {h: containers[h].patches[xi_idx].get_x() + containers[h].patches[xi_idx].get_width() / 2
                 for h in groups}
        bar_h = {h: containers[h].patches[xi_idx].get_height() for h in groups}

        for level, (model, res) in enumerate(results.items()):
            label = _stars(res["p_value"])  # always drawn — 'ns' when not significant
            bracket_y = max(bar_h[best], bar_h[model]) + step * 0.3 + level * step
            _draw_significance_bracket(ax, bar_x[best], bar_x[model], bracket_y, label)
            top = max(top, bracket_y + step)

    ax.set_ylim(ymin, top + step * 0.3 if top > ymax else ymax)
    return ttest_by_group


def plot_f1_across_seeds(wandb_evaluations,
                         radius,
                         pct_mask_nodes,
                         BASE_DIR_REPO,
                         height=4,
                         aspect=0.7,
                        save_path = None,
                        dropna=True,
                        ttest=False):
    """
    Plot mean and standard deviation of per-class F1 scores across seeds.
    Uses seaborn catplot with one facet per class and one bar per model.

    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
        List of Wandb_evaluation instances, one per model
    radius : float or int or None
        Radius value to filter by. None matches the runs that have no radius
        (NaN), as in the graph-level sweeps.
    pct_mask_nodes : float or int or None
        Percentage of masked nodes to filter by; None matches NaN.
    dropna : bool, optional
        Drop runs with NaN radius/pct_mask_nodes before filtering (default True).
        Set False when radius or pct_mask_nodes is None.
    config_path : str, optional
        Path to config.yml file (default: 'config.yml')
    height : float, optional
        Height of each facet in inches (default: 4)
    aspect : float, optional
        Aspect ratio of each facet (default: 0.7)
    save_path: str
        Path name from repo to save the plot.
    ttest : bool, optional (default: False)
        If True, identify the best-performing model per class (highest mean
        F1 across seeds) and annotate every other model with the result of a
        two-sided t-test against it (p < 0.05 = significant), same approach
        as InterScale's evaluation comparison. Adds 'best_model',
        'p_value_vs_best' and 'significant' columns to stats_df.

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
        df = wandb_eval.df.dropna(subset=["radius", "pct_mask_nodes"]) if dropna else wandb_eval.df
        classes = wandb_eval.classes

        model_names.append(model_name)

        # Filter by specified parameters; None means "the runs without this
        # parameter" (NaN), which is how the graph-level sweeps store radius.
        mask_radius = df['radius'].isna() if radius is None else df['radius'] == radius
        mask_pct = (df['pct_mask_nodes'].isna() if pct_mask_nodes is None
                    else df['pct_mask_nodes'] == pct_mask_nodes)
        df_filtered = df[mask_radius & mask_pct].copy()

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

    # Fix the class/model order explicitly so it matches the bar containers
    # `_annotate_ttest_grouped` reads back off the axis below.
    class_order = sorted(plot_data['class'].unique())

    # Create the catplot
    g = sns.catplot(
        data=plot_data,
        kind="bar",
        x="class",
        y="f1_score",
        hue="model",
        order=class_order,
        hue_order=model_names,
        height=height,
        aspect=aspect,
        errorbar="sd",  # Standard deviation error bars
        capsize=0.1,
        edgecolor="black",
        linewidth=1.0,
        alpha=0.8,
        palette=use_palette,
        legend=True,
    )
    # Legend outside the plot, to the right — not overlapping the bars.
    sns.move_legend(g, "center left", bbox_to_anchor=(1.02, 0.5), title="Model",
                     frameon=True)

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

    # Two-sided t-test of every model against the best-performing model per
    # class (see the module-level docstring above `_stars`).
    if ttest:
        ttest_by_group = {}
        for ax in g.axes.flat:
            ttest_by_group.update(_annotate_ttest_grouped(
                ax, plot_data, x="class", y="f1_score", hue="model",
                order=class_order, hue_order=model_names,
            ))
        ttest_df = _ttest_long_table(ttest_by_group, group_col="class")
        if len(ttest_df):
            stats_df = stats_df.merge(ttest_df, on=["class", "model"], how="left")

    # Overall title
    title = f'F1 Scores Across Seeds (radius={radius}, pct_mask={pct_mask_nodes})'
    if ttest:
        title += ' — vs. best model: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant'
    g.fig.suptitle(
        title,
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
                              save_path=None, figsize=(10, 6), ttest=False):
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
    ttest : bool, optional (default: False)
        If True, identify the best-performing model per class (highest mean
        F1 across seeds) and annotate every other model with the result of a
        two-sided t-test against it (p < 0.05 = significant), same approach
        as InterScale's evaluation comparison. Adds 'best_model',
        'p_value_vs_best' and 'significant' columns to stats_df.

    Returns:
    --------
    fig, ax : matplotlib figure and axes
    stats_df : pd.DataFrame with mean and std for each class and model
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)

    all_data = []
    model_names = []

    for wandb_eval in wandb_evaluations:
        model_names.append(wandb_eval.model)
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
    
    # Fix the class/model order explicitly so it matches the bar containers
    # `_annotate_ttest_grouped` reads back off the axis below.
    class_order = sorted(combined_df['class'].unique())

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=combined_df,
        x='class',
        y='f1_score',
        hue='model',
        order=class_order,
        hue_order=model_names,
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
    if ttest:
        title += ' — vs. best model: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    # Legend outside the plot, to the right — not overlapping the bars.
    ax.legend(title='Model', loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.9)

    # Two-sided t-test of every model against the best-performing model per
    # class (see the module-level docstring above `_stars`).
    if ttest:
        ttest_by_group = _annotate_ttest_grouped(
            ax, combined_df, x="class", y="f1_score", hue="model",
            order=class_order, hue_order=model_names,
        )
        ttest_df = _ttest_long_table(ttest_by_group, group_col="class")
        if len(ttest_df):
            stats_df = stats_df.merge(ttest_df, on=["class", "model"], how="left")

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


def plot_overall_metric_comparison(wandb_evaluations, metric, radius=None, pct_mask_nodes=None,
                                   BASE_DIR_REPO=None, save_path=None, figsize=(8, 5), dropna=True,
                                   ttest=False):
    """
    Plot overall performance for a specific metric across models (aggregated across seeds).

    Parameters:
    -----------
    wandb_evaluations : list of Wandb_evaluation
    metric : str
        Column name of the metric to plot (e.g., 'test_f1_macro', 'test_acc', 'test_loss')
    radius, pct_mask_nodes : filter values (optional, use None to match NaN rows)
    BASE_DIR_REPO : str
        Repository root, used to read figures/config.yml
    save_path : str, optional
        Full path of the file to write, including extension
    figsize : tuple
    dropna : bool, optional
        Whether to drop rows with NaN in 'radius' or 'pct_mask_nodes' (default: True).
        Set to False if radius or pct_mask_nodes is None.
    ttest : bool, optional (default: False)
        If True, identify the best-performing model (highest mean metric
        across seeds) and annotate every other model with the result of a
        two-sided t-test against it (p < 0.05 = significant), same approach
        as InterScale's evaluation comparison. Adds 'best_model',
        'p_value_vs_best' and 'significant' columns to stats_df.

    Returns:
    --------
    fig, ax, stats_df
    """
    general_config, model_palette = set_plot_configs(BASE_DIR_REPO)

    all_data = []
    model_order = []  # preserves the order models are passed in (config sweep order)

    for wandb_eval in wandb_evaluations:
        model_name = wandb_eval.model
        df = wandb_eval.df.dropna(subset=["radius", "pct_mask_nodes"]) if dropna else wandb_eval.df.copy()

        if radius is None:
            df = df[df['radius'].isna()]
        else:
            df = df[df['radius'] == radius]

        if pct_mask_nodes is None:
            df = df[df['pct_mask_nodes'].isna()]
        else:
            df = df[df['pct_mask_nodes'] == pct_mask_nodes]

        if df.empty:
            print(f"Warning: No data for {model_name} with specified filters")
            continue

        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found for {model_name}")
            continue

        plot_df = df[['seed', metric]].copy() if 'seed' in df.columns else df[[metric]].copy()
        plot_df['model'] = model_name
        all_data.append(plot_df)
        model_order.append(model_name)

    if not all_data:
        raise ValueError("No valid data found for any model")

    combined_df = pd.concat(all_data, ignore_index=True)

    # groupby() would sort models alphabetically; reindex to the order they
    # were passed in instead (e.g. GCN, PCATransformer, InterScale), matching
    # plot_f1_across_seeds / plot_class_f1_comparison.
    stats_df = combined_df.groupby('model')[metric].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('n_seeds', 'count')
    ]).reindex(model_order).reset_index()

    palette = {m: model_palette[m] for m in combined_df['model'].unique() if m in model_palette}

    fig, ax = plt.subplots(figsize=figsize)

    models = stats_df['model'].tolist()
    means = stats_df['mean'].tolist()
    stds = stats_df['std'].tolist()
    colors = [palette.get(m, None) for m in models]

    x = np.arange(len(models))
    bar_width = 0.6
    ax.set_xlim(-0.4, len(models) - 0.4)

    ax.bar(x, means, yerr=stds, width=bar_width, color=colors,
           edgecolor='black', linewidth=1.0, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    metric_label = metric.replace('_', ' ').replace('test ', '').title()
    ax.set_xlabel('Model', fontsize=general_config['legend_fontsize'],
                  fontweight=general_config['legend_fontweight'])
    ax.set_ylabel(metric_label, fontsize=general_config['legend_fontsize'],
                  fontweight=general_config['legend_fontweight'])
    title = f'{metric_label} Across Seeds (radius={radius}, pct_mask={pct_mask_nodes})'
    if ttest:
        title += ' — vs. best model: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant'
    ax.set_title(title, fontsize=general_config['title_fontsize'],
                 fontweight=general_config['title_fontweight'])

    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha('right')

    # Two-sided t-test of every model against the overall best-performing
    # model (see the module-level docstring above `_stars`).
    if ttest:
        groups = {m: combined_df.loc[combined_df['model'] == m, metric].dropna().to_numpy()
                  for m in models}
        best, results = _best_vs_rest_ttest(groups)
        if results:
            best_idx = models.index(best)
            for level, (model, res) in enumerate(results.items()):
                label = _stars(res["p_value"])  # always drawn — 'ns' when not significant
                i = models.index(model)
                bar_top = max(means[best_idx] + (stds[best_idx] or 0), means[i] + (stds[i] or 0))
                ymin, ymax = ax.get_ylim()
                step = (ymax - ymin) * 0.09
                bracket_y = bar_top + step * 0.3 + level * step
                if bracket_y + step > ymax:
                    ax.set_ylim(ymin, bracket_y + step)
                _draw_significance_bracket(ax, x[best_idx], x[i], bracket_y, label)

        ttest_df = _ttest_long_table({"overall": (best, results)}, group_col="_group")
        if len(ttest_df):
            stats_df = stats_df.merge(ttest_df.drop(columns="_group"), on="model", how="left")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax, stats_df