# Classification figures

Runnable replacement for the classification notebooks
(`notebooks/*/[…]_class_performance.ipynb`). One script, one config per
dataset **and prediction level**, regenerating every classification figure of
the paper from the corresponding W&B sweeps.

Node- and graph-level classification share all the code; only the config
differs, and `prediction_level` in it decides where the figures land.

```bash
# see what would be produced, without touching W&B
python scripts/classification/run.py --config chen22_graph --dry-run

# one config (downloads the sweeps, caches them, writes pdf + png)
export WANDB_API_KEY=...          # or `wandb login` once
python scripts/classification/run.py --config chen22_node    # -> figures/node_classification/chen22/
python scripts/classification/run.py --config chen22_graph   # -> figures/graph_classification/chen22/

# every config
python scripts/classification/run_all.py

# replot from the committed caches — no W&B, no cluster
python scripts/classification/run_all.py --cache-only
```

Useful flags: `--figures NAME [NAME …]` (subset), `--format {pdf,png,both}`,
`--refresh` (re-download), `--fig-dir DIR`, `--seed`, `-v`.

Requirements: `wandb`, `pandas`, `matplotlib`, `seaborn` — plus `interscale`
itself only for the CLS-heatmap figures. `--dry-run` needs nothing but `pyyaml`.

---

## Folder layout

```
scripts/classification/
├── README.md
├── run.py                      # the only entry point; --config picks the work
├── run_all.py                  # loops over every config, keeps going on failure
├── _common.py                  # shared helpers: CLI, paths, W&B loading, figure saving
├── model_parameters.csv        # every parameter interscale.config.load_config() accepts (see below)
└── configs/
    ├── chen22_node.yaml        # -> figures/node_classification/chen22/
    ├── chen22_graph.yaml       # -> figures/graph_classification/chen22/
    ├── damond19_graph.yaml
    ├── legnini23_graph.yaml
    └── melton25_{node,graph}.yaml
```

`model_parameters.csv` documents the InterScale training-config schema
(`interscale.config.get_cfg_defaults()` / `load_config()`, i.e. what a
`config_files/**/*.yaml` training config can set) — one row per parameter,
with its section (`wandb` / `model` / `local_component.parameters` /
`global_component.parameters` / `optim` / `dataset` /
`dataset.spatial_neigbors_kwargs`), default value, when it applies (e.g.
`local_component.name == GCN`, or `always`), and a description. This is
config-schema documentation, not sweep/run data — it does not need W&B
access and won't change unless `interscale/src/interscale/config/*.py`
changes.

Config names are `<dataset><year>_<level>`; one file per dataset × level, since
the two levels are different sweeps with different class labels.

Rules:

- **No new plotting code in this folder.** Plotting functions belong in
  `src/wandb.py` / `src/plots.py`. Functions defined inline in the notebooks
  (e.g. `plot_overall_metric_comparison`) move to `src/` during the migration.
- **No hardcoded values outside `configs/`.** Sweep IDs, class names, filters
  and figure lists live in the config; `run.py` is orchestration only.
- **No `# %%` cells, no `plt.show()`, no `wandb.login()` prompts.** These are
  scripts, not exported notebooks.

## Config (`configs/<dataset>_<level>.yaml`)

Everything dataset- and level-specific goes here:

```yaml
dataset: chen22                 # slug used in figure paths
prediction_level: graph         # node | graph — selects the output folder
sweep_goal: robustness
classes: [control, Mid-AD]

wandb:
  entity: francesca-drummer
  project: InterScale_hyperparameter_sweep
  sweeps:                       # model -> sweep id, plus which components it has
    GCN:            {id: 39pecmnt, local: true,  global: false}
    PCATransformer: {id: tohmr60x, local: false, global: true}
    InterScale:     {id: 3pfj292b, local: true,  global: true}

dropna: false                   # keep runs whose radius/pct_mask_nodes is NaN
filters:                        # the panel shown in the paper
  radius: null                  # null = the runs *without* a radius (NaN)
  pct_mask_nodes: 0.5

figures:                        # explicit list -> reviewable diff, deterministic output
  - {name: robustness_test_acc, fn: plot_robustness, metric: test_acc, per_model: true}
  - {name: f1_across_seeds,     fn: plot_f1_across_seeds}
  - {name: class_f1_comparison, fn: plot_class_f1_comparison}
  - {name: overall_test_acc,    fn: plot_overall_metric_comparison, metric: test_acc}
  - {name: cls_heatmap,         fn: plot_adata_grouped_heatmaps, group: <obs key>}
```

Per-figure `filters:` and `dropna:` override the config-wide ones.

**`prediction_level` is checked, not trusted.** Every W&B run reports its own
level; if it disagrees with the config the run aborts rather than filing
node results under `graph_classification/`. Graph-level sweeps have no radius,
hence `radius: null` + `dropna: false`; node-level sweeps store radius as the
string `"None"`, hence `radius: "None"` with the default `dropna: true`.

## What the runner does

1. **Resolves paths through [paths.py](../../paths.py)** (`ROOT`, `DATA_DIR`,
   `FIG_DIR`) — never through `BASE_DIR_REPO` / `BASE_DIR_PROJECT` cluster
   branches. Overridable via `INTERSCALE_DATA_DIR` / `INTERSCALE_FIG_DIR`
   so the same script runs locally, on LRZ and on the ICB HPC.
2. **Seeds** via `src.utils.set_full_reproducibility(seed)`, falling back to
   numpy/random when torch is absent, and applies the plotting defaults from
   `figures/config.yml`.
3. **Returns non-zero** if any figure fails, so `run_all.py` and CI notice.
4. **Logs** one line per file written; nothing interactive, every figure closed.
5. **Is idempotent**: rerunning overwrites the same files; no timestamps.

## Outputs

```
figures/{node,graph}_classification/<dataset>/
├── <figure_name>.pdf / .png            # names come from the config
├── <figure_name>.csv                   # the numbers behind the figure, when available
├── <figure_name>_t-test.pdf / .png     # same figure, t-test-annotated (see below)
├── <figure_name>_t-test.csv
└── _cache/<model>_<sweep_id>.{csv,json}
```

- Both `pdf` (paper) and `png` (quick viewing) by default, at `dpi_save` from
  [figures/config.yml](../../figures/config.yml).
- Colors, fonts and palettes come from `figures/config.yml` only.
- Legends are placed outside the axes, never overlapping the bars.

### Statistical comparison (`_t-test` figures)

For `f1_across_seeds`, `class_f1_comparison` and `overall_test_acc` (any figure
using `plot_f1_across_seeds` / `plot_class_f1_comparison` /
`plot_overall_metric_comparison`), the runner writes each figure twice:

- `<name>.{pdf,png,csv}` — unchanged, no significance annotation.
- `<name>_t-test.{pdf,png,csv}` — per scenario (class, or the whole panel for
  `overall_test_acc`), the best-performing model is the one with the highest
  mean score across seeds; every other model is compared against it with a
  **two-sided t-test** and gets a bracket, always — `*` p<0.05, `**` p<0.01,
  `***` p<0.001, `ns` when not significant, so a missing bracket is never
  mistaken for "not tested". The `_t-test.csv` adds `best_model`,
  `p_value_vs_best` and `significant` columns to the usual mean/std/n_seeds
  table. Models are always ordered as passed in `wandb.sweeps` (e.g. GCN,
  PCATransformer, InterScale), never resorted alphabetically. This mirrors
  the "Evaluation comparison"
  methodology used across the InterScale benchmarks (cf.
  [theislab/tissue](https://github.com/theislab/tissue)), implemented
  natively with `scipy.stats.ttest_ind` in `src/wandb.py` rather than the
  unmaintained `statannot` package.

## W&B access and caching

- Auth via `WANDB_API_KEY` (or a prior `wandb login`); the script fails with a
  clear message if it is missing rather than prompting.
- Every sweep pull is cached to `_cache/<model>_<sweep_id>.csv`, with the run's
  level and model name alongside in `.json`. With `--cache-only` all figures
  regenerate offline — this is what makes the repo reproducible for reviewers
  without W&B access, so the cache files are committed.

## Migration status

| Notebook | Config | Status |
| --- | --- | --- |
| `chen_2022/2_Chen22_node_class_performance.ipynb` | `chen22_node` | runs; output not yet compared |
| `chen_2022/1_ Chen22_graph_class_performance.ipynb` | `chen22_graph` | config written, never run |
| `damond_19/IMC_graph_class_performance.ipynb` | `damond19_graph` | config written, never run |
| `legnini_23/Graph_class_performance.ipynb` | `legnini23_graph` | config written, never run |
| `melton_jimenez_25/Melton_graph_class_performance.ipynb` | `melton25_graph` | config written, never run |
| `melton_jimenez_25/Node_class_performance.ipynb` | `melton25_node` | config written, never run |

Figure types and where they come from:

| Figure | Source | Status |
| --- | --- | --- |
| Robustness vs. masking/radius, per model | `Wandb_evaluation.plot_robustness` | ported |
| F1 across seeds, models compared | `src.wandb.plot_f1_across_seeds` | ported |
| Per-class F1 comparison | `src.wandb.plot_class_f1_comparison` | ported |
| Overall metric bar plot (acc / macro F1) | `src.wandb.plot_overall_metric_comparison` | ported |
| CLS-token grouped heatmaps | `interscale.evaluation.graph_classification.plot_adata_grouped_heatmaps` | no handler yet |

## Adding a dataset or level

1. Add `configs/<dataset>_<level>.yaml` with the sweep IDs and class names.
2. Add the dataset palette to `figures/config.yml` under `palettes:`.
3. `python scripts/classification/run.py --config <name> --dry-run`, then run it
   for real; commit the figures and the `_cache/` files.
4. Tick the row above.
