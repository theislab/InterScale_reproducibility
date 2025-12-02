import wandb
import pandas as pd

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

import os
from statsmodels.nonparametric.smoothers_lowess import lowess

class Wandb_evaluation():
    
    def __init__(self, sweep_id, sweep_goal: str, classes: list):
        """
        sweep_id: str - ID from WandB run
        sweep_goal: robustenss, parameter
        calsses: list of class names
        """
        self.sweep_id = sweep_id
        self.sweep_goal = sweep_goal
        self.classes = classes
        
        api = wandb.Api()
        entity, project = "francesca-drummer", "InterScale_hyperparameter_sweep"  
    
        # Get all runs associated with the sweep
        sweep_runs = api.sweep(f"{entity}/{project}/{sweep_id}").runs

    
        data = []
        for run in sweep_runs:
            if run.state == 'finished':
                self.prediction_task = run.config['dataset']['prediction_task']
                self.prediction_level = run.config['dataset']['prediction_level']
            
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
                        "test_f1_micro/avg": run.summary.get("test_f1_micro/avg", None)
                    })
                    for class_idx in classes:
                        run_data[f'test_f1/class_{class_idx}'] = run.summary.get(f"test_f1_{class_idx}", None)
        
                if 'graph' in self.prediction_task:
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
        self.df = pd.DataFrame(data)

    def get_dataframe(self):
        return self.df
    
    def get_mean_and_std(self):
        # Compute mean and standard deviation
        if self.prediction_task == 'regression':
            return self.df.groupby(["pct_mask_nodes", "radius"]).agg(
                mean_test_r2=("test_r2", "mean"),
                std_test_r2=("test_r2", "std"),
                mean_test_pearson=("test_pearson_corr", "mean"),
                std_test_pearson=("test_pearson_corr", "std"),
                mean_run_time=("runtime_seconds", "mean"),
                std_run_time=("runtime_seconds", "std"),
            ).reset_index()
        elif self.prediction_task == 'classification':
            return self.df.groupby(["pct_mask_nodes", "radius"]).agg(
                mean_test_acc=("test_acc", "mean"),
                std_test_acc=("test_acc", "std"),
                mean_test_f1_class_0=(f"test_f1/class_{self.classes[0]}", "mean"),
                std_test_f1_class_0=(f"test_f1/class_{self.classes[0]}", "std"),
                mean_test_f1_class_1=(f"test_f1/class_{self.classes[1]}", "mean"),
                std_test_f1_class_1=(f"test_f1/class_{self.classes[1]}", "std"),
            ).reset_index()
        else:
            raise ValueError(f"sweep_goal must be 'classification' or 'regression', got '{self.sweep_goal}'")

    
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
    
    def plot_robustness(self, metric="test_r2", save_path = None):
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
        plt.figure(figsize=(4, 6))
        sns.lineplot(
            data=self.df,
            x="radius",
            y=metric,
            hue="pct_mask_nodes",
            marker="o",
            palette="coolwarm",
            errorbar=("sd")  # Adds standard deviation as shaded region
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
            plt.savefig(os.path.join(save_path, f'performance_{self.prediction_level}_{self.prediction_task}_radius_vs_{metric}.jpg'), dpi=1200)
        
        # Show the plot
        plt.show()
    
    def plot_parameter_space(self, metric: str = 'test_r2', save_path: str = None):
    
        # Apply LOWESS smoothing
        smoothed = lowess(self.df[metric], self.df['total_parameters'], frac=0.4)  # frac controls smoothness
        # Create scatterplot with trend line
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='total_parameters', y=metric, data=self.df, label='Data Points')
        
        # Plot smoothed trend
        plt.plot(smoothed[:, 0], smoothed[:, 1], color='red', label='LOWESS Curve')
        
        # Labels and title
        plt.xlabel("Total Parameters")
        plt.ylabel(f"{metric}")
        plt.title(f"Trend of {metric} with Increasing Parameters")
        
        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'parameter_{self.prediction_level}_{self.prediction_task}_radius_vs_{metric}.jpg'), dpi=1200)
        
        plt.show()