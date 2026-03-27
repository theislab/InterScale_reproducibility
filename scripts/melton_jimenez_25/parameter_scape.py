# %% [markdown]
# # Melton et al., 2025: Hyperparamter sweep analysis - Cell type prediction
#
# Analyse how performance relates to hyperparameters..
#
# **Part 1: Hyperparmeter sweep**
#
# Run hyperparameter sweep with script `hyperparameter.yaml` and config file: 
#

# %%
import os

# LRZ home
if os.path.exists("/dss/dsshome1/05/di93tig"):
    print('LRZ cluster')
    CLUSTER = 'LRZ'
    BASE_DIR_REPO = "/dss/dsshome1/05/di93tig/1_projects/InterScale_reproducibility" 
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
import wandb
wandb.login()

# %%
import pandas as pd
# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from src.wandb import Wandb_evaluation, plot_f1_across_seeds, plot_class_f1_comparison, set_plot_configs
from InterScale.config import load_config, config_from_wandb_run

# %%
import yaml
import matplotlib.pyplot as plt
import scanpy as sc

with open(os.path.join(BASE_DIR_REPO, "InterScale_reproducibility/figures/config.yml"), "r") as f:
    config = yaml.safe_load(f)

PALETTE = config["palettes"]["continuous"]
CELL_TYPE_COLORS = config["palettes"][DATA]

# %% [markdown]
# ## Hyperparameter sweep
#
# Evaluation of node-classification (Cell type) hyperparameter sweep.

# %%
CLASSES = ['Alpha', 'Acinar', 'Beta', 'Ductal', 'Endocrine', 'Immune', 'Mast', 'Beta', 'Endothelial', 'Fibroblasts']

# %% [markdown]
# ### Load WandB IDs

# %%
GNN_sweep = "tixesx9b"
Dual_sweep = "d9lf50og"

# %%
SWEEP_GOAL = 'hyperparameter'

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, True, False, SWEEP_GOAL, CLASSES)
Dual_wandb = Wandb_evaluation("InterScale", Dual_sweep, True, True, SWEEP_GOAL, CLASSES)

# %%
Dual_wandb.get_dataframe().to_csv(f"{BASE_DIR_REPO}//DualInterScale.csv")

# %%
GNN_wandb.get_dataframe().head()

# %%
GNN_wandb.plot_parameter_space(metric = 'test_acc', relevant_params = GNN_wandb.hyperparameters, exclude_parameters = {"pct_mask_nodes":[0.0]}, save_path = f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
relevant_params = PCATrans_wandb.hyperparameters + PCATrans_wandb.global_component_params
PCATrans_wandb.plot_parameter_space(metric = 'test_acc', relevant_params = relevant_params, exclude_parameters = {"pct_mask_nodes":[0.0]})

# %%
relevant_params = Dual_wandb.hyperparameters + Dual_wandb.local_component_params + Dual_wandb.global_component_params
Dual_wandb.plot_parameter_space(metric = 'test_acc', relevant_params = relevant_params, exclude_parameters = {"pct_mask_nodes":[0.0]}, save_path = f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def plot_param_importance(
    df: pd.DataFrame,
    metric: str = "test_acc",
    lc_params: list[str] | None = None,
    gc_params: list[str] | None = None,
    shared_params: list[str] | None = None,
    exclude_params: list[str] | None = None,
    figsize: tuple = (5, 6),
    title: str | None = None,
    lc_color: str = "#A6CCBF",
    gc_color: str = "#D9A84B",
    shared_color: str = "#3E7884",
    top_n: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot hyperparameter importance as a horizontal bar chart in scanpy style,
    coloured by parameter group (LC / GC / SHARED), sorted by absolute
    correlation (highest at top).
 
    Parameters
    ----------
    df : pd.DataFrame
        Sweep results dataframe. Must contain `metric` and all HP columns.
    metric : str
        Column to correlate HPs against. Default 'test_acc'.
    lc_params : list[str] | None
        Local-component HP column names. Auto-detected (prefix 'LC_') if None.
    gc_params : list[str] | None
        Global-component HP column names. Auto-detected (prefix 'GC_') if None.
    shared_params : list[str] | None
        Shared HP column names. Auto-detected (everything else) if None.
    exclude_params : list[str] | None
        Columns to drop before computing correlations (e.g. fully-NaN cols).
    figsize : tuple
        Figure size passed to plt.subplots.
    title : str | None
        Plot title wrapped in double quotes, scanpy-style. Defaults to
        '"Hyperparameter importance" vs "{metric}"'.
    lc_color : str
        Bar colour for LC parameters. Default '#A6CCBF'.
    gc_color : str
        Bar colour for GC parameters. Default '#D9A84B'.
    shared_color : str
        Bar colour for shared parameters. Default '#E8857A'.
    top_n : int | None
        If set, show only the top N parameters by absolute correlation.
 
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects.
 
    Example
    -------
    >>> import pandas as pd
    >>> from plot_param_importance import plot_param_importance
    >>> df = pd.read_csv("DualInterScale.csv", index_col=0)
    >>> df = df[df["pct_mask_nodes"] != 0.0]
    >>> fig, ax = plot_param_importance(df, metric="test_acc")
    >>> fig.savefig("importance.png", dpi=150, bbox_inches="tight")
    """
 
    # ── Detect HP columns ─────────────────────────────────────────────────────
    non_hp_cols = {
        "id", "name", "seed", "state", "runtime_seconds", "total_parameters",
        "decoder_type", "radius",
        metric,
        *[c for c in df.columns if c.startswith("test_f1")],
    }
    if exclude_params:
        non_hp_cols.update(exclude_params)
 
    candidate_cols = [c for c in df.columns if c not in non_hp_cols]
 
    if lc_params is None:
        lc_params = [c for c in candidate_cols if c.startswith("LC_")]
    if gc_params is None:
        gc_params = [c for c in candidate_cols if c.startswith("GC_")]
    if shared_params is None:
        shared_params = [
            c for c in candidate_cols
            if c not in lc_params and c not in gc_params
        ]
 
    all_params = shared_params + lc_params + gc_params
 
    # ── Compute Pearson r ─────────────────────────────────────────────────────
    correlations = {}
    for hp in all_params:
        sub = df[[hp, metric]].dropna()
        if sub[hp].nunique() < 2 or len(sub) < 5:
            continue
        r = sub[hp].corr(sub[metric])
        if not np.isnan(r):
            correlations[hp] = r
 
    if not correlations:
        raise ValueError("No valid correlations could be computed.")
 
    # Sort descending by |r|; top_n slice; then reverse for barh (top = highest)
    corr_s = pd.Series(correlations).sort_values(key=abs, ascending=False)
    if top_n is not None:
        corr_s = corr_s.iloc[:top_n]
    corr_s = corr_s.iloc[::-1]   # reverse so highest ends up at the top of the plot
 
    # ── Helpers ───────────────────────────────────────────────────────────────
    def group_color(hp):
        if hp in lc_params:   return lc_color
        if hp in gc_params:   return gc_color
        return shared_color
 
    def group_label(hp):
        if hp in lc_params:   return "LC"
        if hp in gc_params:   return "GC"
        return "Shared"
 
    # ── Style (clean matplotlib / scanpy look) ────────────────────────────────
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        10,
        "axes.facecolor":   "white",
        "figure.facecolor": "white",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
        "xtick.bottom":     True,
        "ytick.left":       True,
        "axes.grid":        False,
    })
 
    fig, ax = plt.subplots(figsize=figsize)
 
    n = len(corr_s)
    y_pos = np.arange(n)
    bar_height = 0.65
 
    # ── Bars ──────────────────────────────────────────────────────────────────
    colors = [group_color(hp) for hp in corr_s.index]
    ax.barh(y_pos, corr_s.values, height=bar_height, color=colors, linewidth=0)
 
    # ── Zero line ─────────────────────────────────────────────────────────────
    ax.axvline(0, color="black", linewidth=0.8, zorder=3)
 
    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_s.index, fontsize=10)
    ax.set_ylim(-0.5, n - 0.5)
 
    x_max = corr_s.abs().max()
    ax.set_xlim(-x_max * 1.15, x_max * 1.15)
    ax.set_xlabel("Pearson r", fontsize=10, labelpad=6)
 
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="both", length=3, width=0.8)
    ax.tick_params(axis="y", which="both", left=False)
 
    # X ticks symmetric around 0
    tick_step = 0.1 if x_max <= 0.5 else 0.2
    x_ticks = np.concatenate([
        np.arange(0, -x_max * 1.1, -tick_step)[::-1],
        np.arange(tick_step, x_max * 1.1, tick_step),
    ])
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{v:.1f}" for v in x_ticks], fontsize=9)
 
    # ── Title (scanpy style: quoted group name) ───────────────────────────────
    if title is None:
        title = f'"Hyperparameter importance" vs "{metric}"'
    ax.set_title(title, fontsize=11, pad=10)
 
    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=shared_color, label="Shared"),
        mpatches.Patch(facecolor=lc_color,     label="LC"),
        mpatches.Patch(facecolor=gc_color,      label="GC"),
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=9,
        loc="lower right",
        handlelength=1.2,
        handleheight=1.0,
    )
 
    fig.tight_layout()
    return fig, ax
 
 
# ── Quick demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("DualInterScale.csv", index_col=0)
    df = df.drop(columns=["GC_hidden_dim", "GC_dim_feedforward"], errors="ignore")
    df = df[df["pct_mask_nodes"] != 0.0]
 
    fig, ax = plot_param_importance(df, metric="test_acc")
    fig.savefig(f"{BASE_DIR_REPO}/importance.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ### Best model selection

# %%
df = Dual_wandb.filter_runs(exclude_parameters = {"pct_mask_nodes":[0.0]})
df.head()

# %%
df

# %%
df[df['test_acc'] > 0.55]

# %%
adata = sc.read_h5ad(os.path.join(BASE_DIR_PROJECT, "data/melton25.h5ad"))

# %%
Dual_wandb.export_config_to_yaml(best_run_id="jaokfets", save_path=f"{BASE_DIR_REPO}/InterScale_reproducibility/best_config.yaml")

# %% [markdown]
# ### Load WandB IDs

# %%
GNN_sweep = "l0jetvrz"
InterScale_sweep = "yqvyut7l"
DualInterScale_sweep = "1nm8ndul"
PCATrans_sweep = "8govw06t"

# %%
SWEEP_GOAL = 'robustness'
CLASSES = ['Alpha', 'Acinar', 'Beta', 'Ductal', 'Endocrine', 'Immune', 'Mast', 'Beta', 'Endothelial', 'Fibroblasts']

# %% [markdown]
# Q: Why does PCATrans not have test_f1 scores?

# %%
GNN_wandb = Wandb_evaluation("GCN", GNN_sweep, True, False, SWEEP_GOAL, CLASSES)
PCATrans_wandb = Wandb_evaluation("PCATransformer", PCATrans_sweep, False, True, SWEEP_GOAL, CLASSES)
InterScale_wandb = Wandb_evaluation("InterScale", InterScale_sweep, True, True, SWEEP_GOAL, CLASSES)
DualInterScale_wandb = Wandb_evaluation("DualInterScale", DualInterScale_sweep, True, True, SWEEP_GOAL, CLASSES)

# %%
GNN_wandb.get_dataframe().head()

# %%
GNN_wandb.plot_robustness(metric="test_acc", save_path = f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
DualInterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
PCATrans_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
InterScale_wandb.plot_robustness(metric="test_acc", save_path =  f"{BASE_DIR_REPO}/figures/{DATA}/node_class")

# %%
wandb_evals = [GNN_wandb, PCATrans_wandb, DualInterScale_wandb]  # Your list of dataframes, PCATrans_wandb, InterScale_wandb

# %%
g, stats, plot_data = plot_f1_across_seeds(
    wandb_evaluations=wandb_evals,
    radius=60,
    pct_mask_nodes=0.0,
    BASE_DIR_REPO=BASE_DIR_REPO, 
    height=10,
    aspect=0.9,
    save_path=f'{DATA}/node_class/f1_comparison_models_radius'
)
plt.show()
#
# # Save with high DPI from config
# # g.savefig('f1_scores.png', bbox_inches='tight')
#
# # Print statistics:
# print(stats)

# %%
from src.wandb import plot_class_f1_robustness

# %%
for class_id in CLASSES: 
    fig, ax, stats = plot_class_f1_robustness(
        wandb_evals,
        class_idx=class_id,
        pct_mask_nodes=0.25,
        BASE_DIR_REPO=BASE_DIR_REPO, 
        #save_path=f"{BASE_DIR_REPO}/figures/{DATA}/node_class"
    )

# %%
