"""Shared machinery for the classification figure scripts.

The same code produces node- and graph-level classification figures; the config
decides which. ``prediction_level`` in the config selects the output folder
(``figures/node_classification/`` vs ``figures/graph_classification/``) and is
checked against what the W&B runs actually report.

All W&B access, caching, path resolution and figure saving happens here, so the
config files stay pure configuration.

Heavy imports (wandb, interscale, matplotlib) are deliberately deferred into the
functions that need them, so ``--dry-run`` works in any environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_DIR = HERE / "configs"
LEVELS = ("node", "graph")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

log = logging.getLogger("classification")


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def fig_root() -> Path:
    """Figure root; override with INTERSCALE_FIG_DIR to write outside the repo."""
    return Path(os.environ.get("INTERSCALE_FIG_DIR", REPO_ROOT / "figures"))


def data_root() -> Path:
    """Data root; override with INTERSCALE_DATA_DIR (h5ad files live on scratch)."""
    return Path(os.environ.get("INTERSCALE_DATA_DIR", REPO_ROOT / "data"))


def task_name(cfg: dict) -> str:
    """'node' -> 'node_classification', 'graph' -> 'graph_classification'."""
    return f"{cfg['prediction_level']}_classification"


def output_dir(cfg: dict) -> Path:
    """Figures are split by prediction level, then by dataset."""
    return fig_root() / task_name(cfg) / cfg["dataset"]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser(default_config: Path | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regenerate the classification figures for one config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=default_config,
                   required=default_config is None,
                   help="config YAML: a path, or a name from configs/ (e.g. chen22_graph)")
    p.add_argument("--fig-dir", type=Path, default=None,
                   help="figure root (default: $INTERSCALE_FIG_DIR or <repo>/figures)")
    p.add_argument("--format", choices=["pdf", "png", "both"], default="both",
                   help="output format(s)")
    p.add_argument("--figures", nargs="+", metavar="NAME", default=None,
                   help="only produce these figures (names from the config)")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--cache-only", action="store_true",
                   help="never contact W&B; regenerate from the cached run tables")
    p.add_argument("--refresh", action="store_true",
                   help="re-download the sweeps even if a cache exists")
    p.add_argument("--dry-run", action="store_true",
                   help="list what would be produced and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def resolve_config(spec: Path) -> Path:
    """Accept a path, or a bare config name such as `chen22_graph`."""
    if spec.exists():
        return spec
    for candidate in (CONFIG_DIR / spec.name, CONFIG_DIR / f"{spec.name}.yaml"):
        if candidate.exists():
            return candidate
    available = ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")))
    raise FileNotFoundError(f"no config '{spec}' (available: {available})")


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("dataset", "prediction_level", "classes", "wandb", "figures"):
        if key not in cfg:
            raise ValueError(f"{path}: missing required key '{key}'")
    if cfg["prediction_level"] not in LEVELS:
        raise ValueError(
            f"{path}: prediction_level must be one of {LEVELS}, "
            f"got {cfg['prediction_level']!r} — it selects the output folder."
        )
    return cfg


# --------------------------------------------------------------------------- #
# W&B loading + caching
# --------------------------------------------------------------------------- #
def _cache_paths(out_dir: Path, model: str, sweep_id: str) -> tuple[Path, Path]:
    cache = out_dir / "_cache"
    return cache / f"{model}_{sweep_id}.csv", cache / f"{model}_{sweep_id}.json"


def _check_wandb_auth() -> None:
    if os.environ.get("WANDB_API_KEY"):
        return
    if (Path.home() / ".netrc").exists():
        return
    raise RuntimeError(
        "No W&B credentials found. Export WANDB_API_KEY=... or run `wandb login` "
        "once, or use --cache-only to plot from the cached run tables."
    )


def load_evaluations(cfg: dict, out_dir: Path, *, cache_only: bool, refresh: bool) -> list:
    """Return one Wandb_evaluation per model, from cache when possible."""
    import pandas as pd

    from src.wandb import Wandb_evaluation

    wb = cfg["wandb"]
    classes = cfg["classes"]
    evals = []

    for model, spec in wb["sweeps"].items():
        csv_path, meta_path = _cache_paths(out_dir, model, spec["id"])

        if csv_path.exists() and meta_path.exists() and not refresh:
            log.info("%s: loading cached sweep %s", model, spec["id"])
            meta = json.loads(meta_path.read_text())
            # keep_default_na=False: some sweeps store radius as the *string*
            # "None", which pandas would otherwise read back as NaN and no
            # filter would ever match. Empty cells stay NaN via na_values.
            df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
            evals.append(Wandb_evaluation.from_dataframe(
                df, model=model, sweep_id=spec["id"],
                sweep_goal=cfg.get("sweep_goal"), classes=classes,
                model_name=meta.get("model_name", ""),
                prediction_task=meta.get("prediction_task"),
                prediction_level=meta.get("prediction_level"),
            ))
            continue

        if cache_only:
            raise FileNotFoundError(
                f"--cache-only, but no cache for {model} at {csv_path}. "
                "Run once with W&B access to populate it."
            )

        _check_wandb_auth()
        log.info("%s: downloading sweep %s from W&B", model, spec["id"])
        ev = Wandb_evaluation(
            model, spec["id"], spec["local"], spec["global"],
            cfg.get("sweep_goal"), classes,
            entity=wb["entity"], project=wb["project"],
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        ev.df.to_csv(csv_path, index=False)
        meta_path.write_text(json.dumps({
            "model_name": getattr(ev, "model_name", ""),
            "prediction_task": getattr(ev, "prediction_task", None),
            "prediction_level": getattr(ev, "prediction_level", None),
            "n_runs": len(ev.df),
        }, indent=2))
        evals.append(ev)

    _check_prediction_level(cfg, evals)
    return evals


def _check_prediction_level(cfg: dict, evals: list) -> None:
    """Fail if the sweeps are not the level the config claims.

    The level picks the output folder, so a mismatch would file node-level
    results under graph_classification (or vice versa) — worth stopping for.
    """
    expected = cfg["prediction_level"]
    wrong = [(ev.model, ev.sweep_id, ev.prediction_level) for ev in evals
             if ev.prediction_level and expected not in str(ev.prediction_level)]
    if wrong:
        detail = "; ".join(f"{m} (sweep {s}) is {lvl!r}" for m, s, lvl in wrong)
        raise ValueError(
            f"config says prediction_level={expected!r}, but W&B reports: {detail}. "
            f"Either these are the wrong sweep IDs, or the config belongs in "
            f"the other prediction_level."
        )


# --------------------------------------------------------------------------- #
# figure saving
# --------------------------------------------------------------------------- #
def _formats(fmt: str) -> list[str]:
    return ["pdf", "png"] if fmt == "both" else [fmt]


def save(fig, out_dir: Path, name: str, fmt: str, stats=None) -> list[Path]:
    """Write a figure (and its stats table) and close it."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in _formats(fmt):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    if stats is not None and len(stats):
        path = out_dir / f"{name}.csv"
        stats.to_csv(path, index=False)
        written.append(path)
    plt.close(fig)
    return written


def _seed_everything(seed: int) -> None:
    """src.utils.set_full_reproducibility if torch is around, else numpy/random only.

    Plotting cached run tables needs no torch, and requiring it would stop anyone
    without the training environment from regenerating the figures.
    """
    try:
        from src.utils import set_full_reproducibility
    except ImportError:
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        log.debug("torch not available — seeded numpy/random only")
    else:
        set_full_reproducibility(seed)


def save_table(df, out_dir: Path, name: str) -> list[Path]:
    """Write a stats table that has no figure of its own."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return [path]


def _figure_of(obj):
    """Accept a Figure, a seaborn FacetGrid, or a (fig, ...) tuple."""
    if hasattr(obj, "savefig") and hasattr(obj, "fig"):  # FacetGrid
        return obj.fig
    if hasattr(obj, "savefig"):
        return obj
    raise TypeError(f"cannot save object of type {type(obj)}")


# --------------------------------------------------------------------------- #
# figure handlers — one per `fn` value in the config
# --------------------------------------------------------------------------- #
def _fig_robustness(spec, evals, cfg, out_dir, args):
    """One robustness panel per model."""
    import matplotlib.pyplot as plt

    written = []
    for ev in evals:
        ev.plot_robustness(metric=spec.get("metric", "test_acc"),
                           save_path=None, dropna=_dropna(spec, cfg))
        written += save(plt.gcf(), out_dir, f"{spec['name']}_{ev.model}", args.format)
    return written


def _fig_f1_across_seeds(spec, evals, cfg, out_dir, args):
    from src.wandb import plot_f1_across_seeds

    flt = _filters(spec, cfg)
    g, stats, _ = plot_f1_across_seeds(
        wandb_evaluations=evals,
        radius=flt["radius"], pct_mask_nodes=flt["pct_mask_nodes"],
        BASE_DIR_REPO=str(REPO_ROOT),
        height=spec.get("height", 4), aspect=spec.get("aspect", 0.7),
        save_path=None, dropna=_dropna(spec, cfg),
    )
    return save(_figure_of(g), out_dir, spec["name"], args.format, stats)


def _fig_class_f1_comparison(spec, evals, cfg, out_dir, args):
    from src.wandb import plot_class_f1_comparison

    flt = _filters(spec, cfg)
    fig, _, stats = plot_class_f1_comparison(
        wandb_evaluations=evals,
        radius=flt["radius"], pct_mask_nodes=flt["pct_mask_nodes"],
        BASE_DIR_REPO=str(REPO_ROOT), save_path=None,
        figsize=tuple(spec.get("figsize", (10, 6))),
    )
    return save(_figure_of(fig), out_dir, spec["name"], args.format, stats)


def _fig_overall_metric(spec, evals, cfg, out_dir, args):
    from src.wandb import plot_overall_metric_comparison

    flt = _filters(spec, cfg)
    fig, _, stats = plot_overall_metric_comparison(
        evals, metric=spec.get("metric", "test_acc"),
        radius=flt["radius"], pct_mask_nodes=flt["pct_mask_nodes"],
        BASE_DIR_REPO=str(REPO_ROOT), save_path=None,
        figsize=tuple(spec.get("figsize", (8, 5))), dropna=_dropna(spec, cfg),
    )
    return save(_figure_of(fig), out_dir, spec["name"], args.format, stats)


def _table_pairwise_tests(spec, evals, cfg, out_dir, args):
    """Pairwise significance tests across seeds; writes a CSV, no figure."""
    from src.wandb import pairwise_model_tests

    flt = _filters(spec, cfg)
    tests = pairwise_model_tests(
        evals,
        metrics=spec.get("metrics"),
        per_class=spec.get("per_class", True),
        radius=flt["radius"], pct_mask_nodes=flt["pct_mask_nodes"],
        dropna=_dropna(spec, cfg),
        paired=spec.get("paired", True),
        alpha=spec.get("alpha", 0.05),
    )
    n_sig = int(tests["significant"].sum())
    log.info("%s: %d comparisons, %d significant after Holm correction (%s)",
             spec["name"], len(tests), n_sig, ", ".join(sorted(tests["test"].unique())))
    return save_table(tests, out_dir, spec["name"])


HANDLERS = {
    "plot_robustness": _fig_robustness,
    "plot_f1_across_seeds": _fig_f1_across_seeds,
    "plot_class_f1_comparison": _fig_class_f1_comparison,
    "plot_overall_metric_comparison": _fig_overall_metric,
    "pairwise_model_tests": _table_pairwise_tests,
}

# Handlers that produce a table instead of a figure (affects --dry-run output).
TABLE_FNS = {"pairwise_model_tests"}


def _filters(spec: dict, cfg: dict) -> dict:
    """Per-figure filters override the dataset-wide ones.

    A null (or missing) value means "the runs that have no such parameter",
    i.e. NaN — which is how the graph-level sweeps store radius.
    """
    flt = dict(cfg.get("filters") or {})
    flt.update(spec.get("filters") or {})
    return {"radius": flt.get("radius"), "pct_mask_nodes": flt.get("pct_mask_nodes")}


def _dropna(spec: dict, cfg: dict) -> bool:
    """Drop runs with NaN radius/pct_mask_nodes? Must be false when filtering on null."""
    if "dropna" in spec:
        return spec["dropna"]
    return cfg.get("dropna", True)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(default_config: Path | None = None, argv=None) -> int:
    args = build_parser(Path(default_config) if default_config else None).parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args.config = resolve_config(Path(args.config))
    cfg = load_config(args.config)
    if args.fig_dir:
        os.environ["INTERSCALE_FIG_DIR"] = str(args.fig_dir)
    out_dir = output_dir(cfg)

    specs = cfg["figures"]
    if args.figures:
        known = {s["name"] for s in specs}
        unknown = set(args.figures) - known
        if unknown:
            log.error("unknown figure(s): %s (available: %s)",
                      ", ".join(sorted(unknown)), ", ".join(sorted(known)))
            return 2
        specs = [s for s in specs if s["name"] in args.figures]

    if args.dry_run:
        print(f"config     : {args.config}")
        print(f"dataset    : {cfg['dataset']}")
        print(f"level      : {cfg['prediction_level']} -> {task_name(cfg)}/")
        print(f"output dir : {out_dir}")
        print(f"sweeps     : " + ", ".join(
            f"{m}={s['id']}" for m, s in cfg["wandb"]["sweeps"].items()))
        print("figures    :")
        for s in specs:
            fn = s["fn"]
            mark = " " if fn in HANDLERS else "!"
            suffix = "_<model> (one per model)" if s.get("per_model") else ""
            exts = "csv" if fn in TABLE_FNS else "/".join(_formats(args.format))
            print(f"  {mark} {s['name']}{suffix}.{exts}  [{fn}]")
        missing = [s["fn"] for s in specs if s["fn"] not in HANDLERS]
        if missing:
            print(f"\n! no handler yet for: {', '.join(sorted(set(missing)))}")
        return 0

    # Plot without a display and apply the repo-wide plotting defaults.
    import matplotlib
    matplotlib.use("Agg")

    _seed_everything(args.seed)

    from src.wandb import set_plot_configs
    set_plot_configs(str(REPO_ROOT))

    evals = load_evaluations(cfg, out_dir, cache_only=args.cache_only, refresh=args.refresh)

    failures = 0
    for spec in specs:
        handler = HANDLERS.get(spec["fn"])
        if handler is None:
            log.error("%s: no handler for fn=%s", spec["name"], spec["fn"])
            failures += 1
            continue
        try:
            for path in handler(spec, evals, cfg, out_dir, args):
                log.info("wrote %s", path.relative_to(fig_root().parent)
                         if fig_root().parent in path.parents else path)
        except Exception:
            log.exception("%s: failed", spec["name"])
            failures += 1

    if failures:
        log.error("%d of %d figures failed", failures, len(specs))
    else:
        log.info("all %d figures written to %s", len(specs), out_dir)
    return 1 if failures else 0
