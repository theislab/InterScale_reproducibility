"""Shared machinery for the decoder-loading stability notebooks.

Every function here was originally defined inline in
``notebooks/legnini_23/4_a_decoder_stability.ipynb``. It now lives in one module so that the
CosMx counterpart (``notebooks/Cosmx/4_decoder_stability_cosmx.ipynb``) runs *the same code*
rather than a copy that drifts: the two notebooks are meant to be a like-for-like comparison
of how stable the gene loadings are across seeds, and that claim is only as good as the
selection code being identical on both sides.

Import it with the repo root on ``sys.path``::

    import sys; sys.path.insert(0, BASE_DIR_REPO)
    from src.decoder_stability import build_cfg, run_seed_inference, collect_gene_sets

Do NOT put ``InterScale_reproducibility/src`` itself on ``sys.path`` -- ``src/wandb.py`` would
then shadow the real ``wandb`` package. The repo root is the level to add.

Layout
------
config          :func:`build_cfg`, :func:`target_layer`
inference       :func:`per_gene_metrics`, :func:`run_seed_inference`
gene selection  :func:`collect_gene_sets`
comparison      :func:`plot_membership`, :func:`intersection_by_k`,
                :func:`plot_intersection_by_k`, :func:`tail_prob_by_k`,
                :func:`plot_tail_prob_by_k`, :func:`overlap_matrix`,
                :func:`local_global_split`
"""

from __future__ import annotations

import itertools
import os
from functools import reduce
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import issparse

import interscale
from interscale.config.registry import resolve_config
from interscale.config.sweep import apply_sweep_config
from interscale.evaluation import calculate_dim_importance, gene_loadings, get_genes_dim
from interscale.tl.utils import set_full_reproducibility

# Re-exported so a notebook needs one import line, not two. `expected_intersection` is the
# closed-form null mean; `intersection_tail_probability` turns it into a p-value.
from src.utils import expected_intersection, intersection_tail_probability  # noqa: F401

__all__ = [
    "build_cfg",
    "target_layer",
    "per_gene_metrics",
    "run_seed_inference",
    "collect_gene_sets",
    "plot_membership",
    "intersection_by_k",
    "plot_intersection_by_k",
    "tail_prob_by_k",
    "plot_tail_prob_by_k",
    "overlap_matrix",
    "local_global_split",
    "expected_intersection",
    "intersection_tail_probability",
    "GROUP_COLORS",
    "SPLITS",
]

# Matches figures/config.yml's local/global pair. Kept as a module constant because four
# different plots below colour the same two groups and they must agree.
GROUP_COLORS = {"local": "#EE9B00", "global": "#27828E"}

# 'all' first so a per-gene csv leads with the number over every cell; the three real splits
# follow so the test rows stay comparable to a sweep's `test_r2` / `test_pearson_corr`.
SPLITS = ["all", "train", "val", "test"]


# ------------------------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------------------------


def build_cfg(
    dataset: str,
    task: str,
    registry_path: str | Path,
    seed: int,
    models_dir: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
    sweep_overrides: Mapping[str, Any] | None = None,
    model_type: str = "CombinedModel",
):
    """Registry config for ``(dataset, task)`` pinned to one trial of a seed sweep.

    A seed sweep's yaml declares ``optim.seed``, ``model.save`` and whatever else it holds
    fixed. Reapplying those on top of the resolved pair reproduces the config each trial
    trained with; everything else comes from the registry, unchanged.

    ``optim.seed`` is not cosmetic: :func:`interscale.tl.utils.get_model_filename_prefix` puts
    it in the checkpoint name, which is what makes the seeds separable when ``load()`` picks
    the file. The same prefix carries ``dataset.name``, which the registry sets to
    ``<dataset>_genemask`` for a gene-masking arm -- so the two arms' checkpoints are
    distinguishable even before the directory split.

    Parameters
    ----------
    dataset, task
        A registered pair, e.g. ``("melton25", "node_reg_genemask")``.
    registry_path
        ``interscale/config_files/registry.yaml``.
    seed
        The trial's ``optim.seed``.
    models_dir
        The sweep's ``model.save``. A trailing separator is appended here because
        ``train/_training.py`` builds the checkpoint path by string concatenation.
    overrides
        Dotted keys written directly onto the config (``{"optim.n_epochs": 300}``). Use this
        for plain scalars a sweep pinned.
    sweep_overrides
        Dotted keys routed through :func:`interscale.config.sweep.apply_sweep_config` instead.
        Use this for a whole hyperparameter block that a *hyperparameter* sweep selected --
        it is the same code path the training run took, so component keys the model type lacks
        are dropped exactly as they were then, rather than raising here.
    model_type
        Only consulted by ``apply_sweep_config``, to decide which component keys to drop.

    Notes
    -----
    **The seed cannot be read off** ``run.config['optim']['seed']``. ``main_sweep.py`` calls
    ``wandb.init(config=cfg)`` *before* ``apply_sweep_config``, so the nested config wandb
    stored is the base config -- seed 44 for every run of the sweep, which is also why every
    run is named ``..._44_...``. Only the flattened ``run.config['optim.seed']`` (what
    ``sweep_df["seed"]`` reads) carries the trial's actual value.
    """
    cfg = resolve_config(dataset, task, registry_path=str(registry_path))

    if sweep_overrides:
        cfg, _ = apply_sweep_config(
            cfg,
            "hyperparmeter",  # sic -- misspelled in config/sweep.py::SWEEP_GOALS
            dict(sweep_overrides),
            model_type=model_type,
            sweep_params=list(sweep_overrides),
        )

    cfg.defrost()
    cfg.optim.seed = int(seed)
    cfg.model.save = str(models_dir).rstrip(os.sep) + os.sep
    for dotted, value in (overrides or {}).items():
        node, *rest = str(dotted).split(".")
        target = cfg[node]
        for part in rest[:-1]:
            target = target[part]
        target[rest[-1]] = value
    cfg.freeze()
    return cfg


def target_layer(adata, cfg) -> str:
    """Name of the layer holding what the decoders reconstruct, materialising it if needed.

    ``cfg.dataset.layer_key`` is the ground truth for this, but it is ``None`` on melton25 --
    that dataset keeps the reconstruction target in ``.X`` and raw counts in ``layers``, the
    opposite of legnini23. :func:`gene_loadings` and :func:`get_genes_dim` both take a layer
    *name* and raise on ``None``, so ``.X`` is exposed under an explicit key here instead of
    branching at every call site.
    """
    if cfg.dataset.layer_key:
        return cfg.dataset.layer_key
    if "_target" not in adata.layers:
        adata.layers["_target"] = adata.X.copy()
    return "_target"


# ------------------------------------------------------------------------------------------
# inference
# ------------------------------------------------------------------------------------------


def per_gene_metrics(y_true: np.ndarray, y_pred: np.ndarray, min_cells: int = 3) -> dict:
    """Per-gene reconstruction scores over the cells finite in both truth and prediction.

    ``get_model_output`` leaves a cell's prediction as NaN across *all* genes when it was never
    assigned to a graph, so the finite mask is taken per cell rather than per gene -- that keeps
    every gene scored on the same cells, which is what makes the rows comparable to each other.

    Beyond the four requested scores this returns offset-corrected R2 and the first two moments
    of truth and prediction, because raw R2 on this model is dominated by a systematic mean
    offset and by variance shrinkage: the raw number alone (and especially its mean over genes,
    which low-variance genes drive to -1e2) describes the miscalibration rather than the
    covariation structure the decoder loadings are read from.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true).all(axis=1) & np.isfinite(y_pred).all(axis=1)
    t, p = y_true[ok], y_pred[ok]
    n_cells, n_genes = t.shape
    if n_cells < min_cells:
        raise ValueError(f"only {n_cells} cells are finite in both truth and prediction")

    resid = t - p
    t_mu, p_mu = t.mean(axis=0), p.mean(axis=0)
    t_c, p_c = t - t_mu, p - p_mu
    ss_tot = (t_c**2).sum(axis=0)

    # A gene that is constant over these cells has ss_tot == 0 (R2 undefined) or zero predicted
    # sd (r undefined); errstate keeps those from printing, and the NaNs below are deliberate.
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - (resid**2).sum(axis=0) / ss_tot
        # Same R2 after removing the constant offset between the two means.
        r2_offset_corrected = 1.0 - ((t_c - p_c) ** 2).sum(axis=0) / ss_tot
        pearson = (t_c * p_c).sum(axis=0) / np.sqrt(ss_tot * (p_c**2).sum(axis=0))
        # Uncentered, i.e. over the raw expression vectors -- the centered cosine IS Pearson, so
        # centering here would just duplicate the previous column.
        cosine_sim = (t * p).sum(axis=0) / (np.linalg.norm(t, axis=0) * np.linalg.norm(p, axis=0))

    constant = ss_tot == 0
    r2[constant] = np.nan
    r2_offset_corrected[constant] = np.nan

    return {
        "n_cells": np.full(n_genes, n_cells),
        "r2": r2,
        "mse": (resid**2).mean(axis=0),
        "pearson": pearson,
        "cosine_sim": cosine_sim,
        "r2_offset_corrected": r2_offset_corrected,
        "true_mean": t_mu,
        "true_std": t.std(axis=0),
        "pred_mean": p_mu,
        "pred_std": p.std(axis=0),
    }


def run_seed_inference(
    adata,
    seeds: Sequence[int],
    cfg_for,
    models_dir: str | Path,
    results_dir: str | Path,
    *,
    y_true: np.ndarray | None = None,
    layer_key: str | None = None,
    repro_seed: int | None = None,
    drop_attn: bool = True,
    verbose: bool = True,
):
    """Load every seed model, run inference, and score the reconstruction per gene.

    Returns ``(adata, models, local_emb, global_emb, performance)``. ``adata`` is returned
    because ``get_model_output`` may hand back a new object.

    Everything is keyed by seed and written with a ``f"{seed}_"`` prefix into ``adata``:
    ``obsm[f"{seed}_local_emb"]`` / ``obsm[f"{seed}_global_emb"]`` and
    ``layers[f"{seed}_y_pred_local"]`` / ``layers[f"{seed}_y_pred_global"]``. The prefix is
    what makes :func:`collect_gene_sets` able to revisit any seed afterwards -- with
    ``prefix=""`` every seed writes the same keys and only the last one survives the loop.

    Parameters
    ----------
    cfg_for
        ``seed -> cfg`` callable, normally ``functools.partial`` over :func:`build_cfg`.
    y_true
        Dense ground truth, ``(n_cells, n_genes)``. Computed from ``layer_key`` when omitted.
    layer_key
        Layer holding the reconstruction target; see :func:`target_layer`. Only used to derive
        ``y_true``, and only when that is not passed in.
    repro_seed
        Seed handed to :func:`set_full_reproducibility` before each load. ``None`` uses the
        model's own seed. Note the legnini notebook pinned this to 40 for every model, which
        is fine -- nothing stochastic happens between here and the forward pass -- but it must
        be *stated* rather than left to whichever value was last in scope.
    drop_attn
        Pop ``obsm["_attn_matrix"]`` after each seed. It is (n_cells x max_seq_len) float32 --
        ~750 MB on legnini23, ~2.5 GB on melton25 -- and unused by anything downstream here.
    """
    models_dir, results_dir = Path(models_dir), Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if y_true is None:
        if layer_key is None:
            raise ValueError("pass either y_true or layer_key")
        _t = adata.layers[layer_key]
        y_true = np.asarray(_t.todense() if issparse(_t) else _t, dtype=np.float64)

    models, local_emb, global_emb, performance = {}, {}, {}, {}

    for seed in seeds:
        cfg = cfg_for(seed)

        interscale.model.CombinedModel._setup_anndata(
            adata=adata,
            prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key,
            sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs,
            view_registry=False,
        )
        set_full_reproducibility(cfg.optim.seed if repro_seed is None else repro_seed)

        # model_name is left at None so the filename comes from cfg via
        # get_model_filename_prefix, i.e.
        # dual_<dataset>_regr_node_<seed>_GCN_self-attn-transformer_model.pt. load() is a
        # classmethod that builds the model itself; note it uses strict=False, so read the
        # missing/unexpected-key warnings it prints -- a silent mismatch is a garbage model,
        # which is exactly what happens if a hyperparameter block the sweep pinned (melton25's
        # n_embed=128) is not reapplied in build_cfg.
        models[seed] = interscale.model.CombinedModel.load(
            str(models_dir), adata, cfg, local_component=True, global_component=True
        )

        adata = models[seed].get_model_output(adata, prefix=f"{seed}")
        if drop_attn:
            adata.obsm.pop("_attn_matrix", None)

        local_emb[seed] = np.asarray(adata.obsm[f"{seed}_local_emb"], dtype=np.float32)
        global_emb[seed] = np.asarray(adata.obsm[f"{seed}_global_emb"], dtype=np.float32)

        decoders = {"local": f"{seed}_y_pred_local", "global": f"{seed}_y_pred_global"}
        y_pred = {d: np.asarray(adata.layers[k], dtype=np.float64) for d, k in decoders.items()}

        rows = []
        for split in SPLITS:
            cells = (
                slice(None)
                if split == "all"
                else (adata.obs[cfg.dataset.split_key] == split).to_numpy()
            )
            for decoder in decoders:
                rows.append(
                    pd.DataFrame(
                        {
                            "gene": adata.var_names,
                            "decoder": decoder,
                            "split": split,
                            **per_gene_metrics(y_true[cells], y_pred[decoder][cells]),
                        }
                    )
                )

        performance[seed] = pd.concat(rows, ignore_index=True)
        csv_path = results_dir / f"model_{seed}_performance.csv"
        performance[seed].to_csv(csv_path, index=False)
        del y_pred

        if verbose:
            # Medians, not means: a handful of low-variance genes drive mean R2 to -1e2 and
            # make it useless as a sanity check.
            test_median = (
                performance[seed]
                .query("split == 'test'")
                .groupby("decoder")[["r2", "mse", "pearson", "cosine_sim"]]
                .median()
            )
            print(
                f"\nseed {seed}: wrote {csv_path.name} ({len(performance[seed])} rows)\n"
                f"  per-gene medians on the test split:\n{test_median.to_string()}"
            )

    if verbose and seeds:
        s0 = seeds[0]
        print(
            f"\nembeddings: local {local_emb[s0].shape}, global {global_emb[s0].shape} per seed"
        )
    return adata, models, local_emb, global_emb, performance


# ------------------------------------------------------------------------------------------
# gene selection
# ------------------------------------------------------------------------------------------


def collect_gene_sets(
    adata,
    models: Mapping[int, Any],
    seeds: Sequence[int],
    layer_key: str,
    n_embed: int | None = None,
    *,
    dims: str | Sequence[int] | None = "all",
    n_top: int = 3,
    rank_by: str = "loading_x_sd",
    min_frac: float = 0.0,
    enforce_specificity: bool = False,
    specificity_min: float = 1.5,
    cumulative_cutoff: float | None = None,
    spacing: int = 2,
    repro_seed: int = 40,
    allow_empty: bool = False,
    standardize_local_by: str = "local",
    plot: bool = True,
    figsize: tuple = (3.3, 5),
    verbose: bool = True,
):
    """Top-loading gene set per seed, for the local and the global decoder.

    Returns ``(df_local_list, df_global_list, dims_used)``, one DataFrame per seed in ``seeds``
    order indexed by gene -- exactly what :func:`plot_membership` and :func:`intersection_by_k`
    consume -- plus ``dims_used[which][seed]``, the dimensions each set was read off.

    Requires that :func:`run_seed_inference` has already written the ``f"{seed}_"``-prefixed
    embeddings into ``adata``: :func:`gene_loadings` reads the latent from ``adata``, not from
    the model, so the prefix is what lets this revisit a seed other than the last one inferred.

    Parameters
    ----------
    n_embed
        Latent width. Only needed when ``dims="all"``.
    dims
        ``"all"`` scores every one of the ``n_embed`` dimensions -- the legnini23 setting.
        ``None`` instead uses the dimension selection :func:`calculate_dim_importance` stored,
        which is what ``cumulative_cutoff`` is for. An explicit sequence pins the dimensions.

        **This choice dominates the size of the gene sets, and therefore every overlap
        statistic downstream.** At ``n_embed=128`` and ``n_top=3``, ``"all"`` can return up to
        384 genes out of a 979-gene panel -- 39% of it -- so seeds would agree largely because
        each set is nearly half the panel, and the published 25-gene list would be "recovered"
        almost by construction. The published CosMx figure reads its loadings off a handful of
        dimensions (7 local, 5 global), i.e. ``dims=None`` with a cumulative cutoff. Use that
        to compare against it, and always read the observed overlap against the matching
        ``expected_intersection``, which scales with the set sizes.
    cumulative_cutoff
        Fraction of cumulative dimension importance to keep. ``None`` keeps everything, which
        only makes sense together with ``dims="all"``.
    allow_empty
        With ``enforce_specificity=True``, :func:`get_genes_dim` *raises* "No genes passed
        specificity filter" when no dimension has any gene ``specificity_min`` times more
        specific to it than to the others -- the signature of noise loadings. By default that
        propagates, because on a model that is supposed to work it is a real error.

        Set it True for an arm that may legitimately have no program (melton25's gene-masking
        arm), and the offending ``(seed, component)`` yields an EMPTY DataFrame instead, named
        in a warning. Failures are handled per seed and per component on purpose: one seed
        raising must not discard the other four, which is what an all-or-nothing catch at the
        call site does -- it reports zeros everywhere and hides that some seeds did produce
        sets. Read the returned sizes, do not assume uniformity.

        This never fabricates genes. Lowering ``specificity_min`` until something comes back
        would return the top of a noise ranking in the same format as a real program, which is
        worse than an empty result.

    standardize_local_by
        Which embedding's per-dimension std standardizes the LOCAL loadings -- see the warning
        below. ``"local"`` (the default) is correct. ``"global"`` exists only to reproduce the
        published ``Gene_loadings_analysis.ipynb`` bit-for-bit, and is not a defensible
        analysis setting on its own.

    .. warning::
       ``local_latent_key`` is ``f"{seed}_local_emb"``, not ``_global_emb``.
       ``DualDecoderCombinedModule.predict_local`` decodes the LOCAL embedding through the
       local decoder, so standardizing those loadings by the global embedding's std rescales
       every local dimension by the wrong constant -- which changes the dimension ranking and
       therefore which genes get selected. The legnini notebook passed the global key on both
       sides; that is fixed here (and reachable again only through
       ``standardize_local_by="global"``), so its local gene sets will not reproduce
       cell-for-cell under the default.
    """
    if standardize_local_by not in ("local", "global"):
        raise ValueError(
            f'standardize_local_by must be "local" or "global", not {standardize_local_by!r}'
        )
    if dims == "all":
        if n_embed is None:
            raise ValueError('dims="all" needs n_embed')
        dims_arg = list(range(int(n_embed)))
    else:
        dims_arg = None if dims is None else list(dims)
    if dims_arg is None and cumulative_cutoff is None and n_top is None:
        raise ValueError(
            "dims=None reads the stored dimension selection, so give calculate_dim_importance "
            "something to select with -- a cumulative_cutoff (the published CosMx figure used "
            "0.60) or an n_top."
        )

    df_local_list, df_global_list = [], []
    dims_used = {"local": {}, "global": {}}
    empty: list[tuple[int, str]] = []

    for seed in seeds:
        set_full_reproducibility(repro_seed)
        gene_loadings(
            adata,
            models[seed],
            local_latent_key=f"{seed}_{standardize_local_by}_emb",
            global_latent_key=f"{seed}_global_emb",
            local_varm_key=f"{seed}_local_std_gene_loadings",
            global_varm_key=f"{seed}_global_std_gene_loadings",
            layer_key=layer_key,
        )

        importance = {}
        for which in ("local", "global"):
            importance[which] = calculate_dim_importance(
                adata,
                which=which,
                    prefix=seed,
                mode="full",
                use_ratio=True,
                cumulative_cutoff=cumulative_cutoff,
                spacing=spacing,
            )

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, which in zip(axes, ("local", "global")):
                interscale.pl.dim_importance_elbow(
                    importance[which],
                    title=f"seed {seed}: {which} dim importance elbow",
                    ax=ax,
                )
            plt.show()

        dfs = {}
        for which in ("local", "global"):
            if dims_arg is not None:
                dims_used[which][seed] = dims_arg
            else:
                # calculate_dim_importance returns dims_left=None when nothing selected, and
                # get_genes_dim(dims=None) would then silently fall back to every dimension --
                # the near-whole-panel selection this branch exists to avoid.
                left = importance[which]["dims_left"]
                if left is None:
                    raise ValueError(
                        f"no dimension selection stored for {which} (seed {seed}): "
                        f"cumulative_cutoff={cumulative_cutoff}, n_top={n_top}"
                    )
                dims_used[which][seed] = list(left)
            try:
                # get_genes_dim returns `df if not plot else (df, ax)`, so the shape of its
                # return depends on `plot`. Unpacking a 2-tuple unconditionally worked only
                # because every caller so far passed plot=True.
                res = get_genes_dim(
                    adata,
                    which=which,
                    dims=dims_arg,
                    prefix=seed,
                    n_top=n_top,
                    X_layer=layer_key,
                    min_frac=min_frac,
                    rank_by=rank_by,
                    enforce_specificity=enforce_specificity,
                    specificity_min=specificity_min,
                    plot=plot,
                    figsize=figsize,
                )
                dfs[which] = res[0] if isinstance(res, tuple) else res
            except ValueError as exc:
                if not (allow_empty and "specificity" in str(exc).lower()):
                    raise
                dfs[which] = pd.DataFrame(index=pd.Index([], name="gene"))
                empty.append((seed, which))
                print(f"  seed {seed} {which}: NO PROGRAM -- {exc}")

        df_local_list.append(dfs["local"])
        df_global_list.append(dfs["global"])
        if verbose:
            for which in ("local", "global"):
                d = dims_used[which][seed]
                print(f"seed {seed} {which:6s}: {len(dfs[which]):3d} genes off "
                      f"{len(d)} dims {sorted(d)[:12]}{' ...' if len(d) > 12 else ''}")

    if empty:
        print(
            f"\n{len(empty)} of {2 * len(seeds)} (seed, component) selections produced NO "
            f"program: {sorted(empty)}\nThose are empty sets, not zero-overlap results -- the "
            f"loadings carry no dimension-specific signal. Check the arm's own "
            f"val_concordance_corr before reading anything below."
        )
    return df_local_list, df_global_list, dims_used


# ------------------------------------------------------------------------------------------
# cross-seed comparison
# ------------------------------------------------------------------------------------------


def plot_membership(
    true_genes: Iterable[str],
    dfs: Sequence[pd.DataFrame],
    labels: Sequence | None = None,
    ax=None,
    truth_label: str = "truth",
    only_truth: bool = False,
):
    """Which seeds recovered which of the published genes.

    Rows are genes, columns are the published set followed by one column per seed. Genes are
    ordered by how many seeds recovered them, then by whether they are in the published set --
    so the top-left block is "reproducible AND published" and the bottom-right is seed noise.

    ``only_truth=True`` restricts the rows to the published genes, which is the recall view;
    the default keeps the union, which also shows what the model found that the paper did not.
    """
    truth = set(map(str, true_genes))
    sets = {
        (str(labels[i]) if labels is not None else str(i)): set(map(str, df.index))
        for i, df in enumerate(dfs)
    }
    genes = set.union(truth, *sets.values())
    if only_truth:
        genes &= truth
    genes = sorted(genes, key=lambda g: (-sum(g in s for s in sets.values()), g not in truth, g))
    M = pd.DataFrame({s: [g in sets[s] for g in genes] for s in sets}, index=genes)
    M.insert(0, truth_label, [g in truth for g in genes])

    ax = sns.heatmap(
        M.astype(int), cmap="Greens", cbar=False, ax=ax, vmin=0, vmax=1,
        linewidths=0.4, linecolor="white",
    )
    ax.axvline(1, color="black", lw=2)
    runs = M.drop(columns=truth_label)
    n_missed = int((~runs.any(axis=1) & M[truth_label]).sum())
    scope = "true genes" if only_truth else "in the union"
    ax.set_title(
        f"{len(genes)} {scope}, {int(runs.all(axis=1).sum())} in all "
        f"{runs.shape[1]} seeds, {n_missed}/{len(truth)} true genes never recovered",
        fontsize=10,
    )
    ax.set_xlabel("seed")
    ax.tick_params(axis="y", labelsize=6)
    return M


def intersection_by_k(dfs: Sequence[pd.DataFrame], pool: int, ks=None) -> pd.DataFrame:
    """Observed and chance-expected intersection size over all k-subsets of the seed sets.

    For each k, every one of the ``C(n, k)`` subsets of seeds is intersected and the sizes are
    summarised. The comparison that matters is ``obs_mean`` against ``exp_mean``, not against
    zero: :func:`expected_intersection` is the size a random selection of the same set sizes
    from the same pool would already produce, and on an 88-gene panel with 25-gene sets that is
    not negligible at k=2.

    ``pool`` is the number of genes the selection could have drawn from -- the panel size when
    ``min_frac`` is 0, and the number of genes passing the filter otherwise. Getting it wrong
    moves only the dashed null, but it moves it a lot.

    Columns: ``k``, ``n_subsets``, ``obs_mean``, ``obs_sd``, ``obs_min``, ``obs_max``,
    ``exp_mean``, ``fold_mean``.
    """
    sets = [set(map(str, df.index)) for df in dfs]
    sizes = np.array([len(s) for s in sets], dtype=int)
    n = len(sets)
    ks = range(2, n + 1) if ks is None else ks

    rows = []
    for k in ks:
        obs, exp = [], []
        for c in combinations(range(n), k):
            obs.append(len(set.intersection(*(sets[i] for i in c))))
            exp.append(expected_intersection(sizes[list(c)], pool))
        obs = np.asarray(obs, dtype=float)
        exp = np.asarray(exp, dtype=float)
        rows.append(
            {
                "k": k,
                "n_subsets": len(obs),
                "obs_mean": obs.mean(),
                "obs_sd": obs.std(ddof=1) if len(obs) > 1 else 0.0,
                "obs_min": obs.min(),
                "obs_max": obs.max(),
                "exp_mean": exp.mean(),
                "fold_mean": float(obs.mean() / exp.mean()) if exp.mean() > 0 else np.inf,
            }
        )
    return pd.DataFrame(rows)


def plot_intersection_by_k(groups, pool, ax=None, log=False, colors=None):
    """Intersection size vs. number of seeds, observed against its random expectation.

    ``groups``: ``{"local": [df, ...], "global": [df, ...]}``.
    ``pool``: scalar, or ``{"local": P_local, "global": P_global}`` when the two components
    were filtered to different gene pools.
    """
    colors = GROUP_COLORS if colors is None else colors
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))

    out = {}
    for name, dfs in groups.items():
        P = pool if np.isscalar(pool) else pool[name]
        col = colors.get(name, "#666666") if isinstance(colors, dict) else colors[len(out)]
        t = intersection_by_k(dfs, P)
        out[name] = t

        lo = np.clip(t.obs_mean - t.obs_sd, 0, None)
        ax.plot(t.k, t.obs_mean, "-o", color=col, lw=2, ms=5, label=f"{name} observed", zorder=3)
        ax.fill_between(t.k, lo, t.obs_mean + t.obs_sd, color=col, alpha=0.2, lw=0, zorder=2)
        ax.plot(t.k, t.exp_mean, "--", color=col, lw=1.5,
                label=f"{name} expected (P={P})", zorder=3)

    ax.set_xticks(sorted({k for t in out.values() for k in t.k}))
    ax.set_xlabel("number of seeds in the intersection (k)")
    ax.set_ylabel("|∩| (genes)")
    ax.set_title(
        "Intersection size vs. number of seeds\n"
        "solid = observed (mean ± SD over subsets), dashed = random expectation",
        fontsize=9,
    )
    if log:
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return pd.concat(out, names=["group"]).reset_index(level=0)


def tail_prob_by_k(dfs, pool, ks=None, alternative="greater", n_iter=20_000, seed=0):
    """Tail probability over all k-subsets of the seed sets, k = 2..n."""
    sets = [set(map(str, df.index)) for df in dfs]
    sizes = np.array([len(s) for s in sets], dtype=int)
    n = len(sets)
    ks = range(2, n + 1) if ks is None else ks

    rows = []
    for k in ks:
        recs = []
        for c in combinations(range(n), k):
            obs = len(set.intersection(*(sets[i] for i in c)))
            recs.append(
                intersection_tail_probability(
                    sizes[list(c)], pool, obs,
                    alternative=alternative, n_iter=n_iter, seed=seed,
                )
            )
        p = np.array([r["p_value"] for r in recs], float)
        fc = np.array([r["fold_change"] for r in recs], float)
        rows.append({
            "k": k, "n_subsets": len(recs),
            "p_mean": p.mean(), "p_median": np.median(p),
            "p_sd": p.std(ddof=1) if len(p) > 1 else 0.0,
            "p_min": p.min(), "p_max": p.max(),
            "obs_mean": np.mean([r["observed"] for r in recs]),
            "exp_mean": np.mean([r["expected"] for r in recs]),
            "fold_mean": np.mean(fc[np.isfinite(fc)]) if np.isfinite(fc).any() else np.inf,
            "method": recs[0]["method"], "n_iter": recs[0]["n_iter"],
        })
    return pd.DataFrame(rows)


def plot_tail_prob_by_k(groups, pool, ax=None, alpha=0.05, colors=None,
                        alternative="greater", n_iter=20_000, seed=0, show_points=True):
    """p-value of the observed intersection under random selection, vs. number of seeds."""
    colors = GROUP_COLORS if colors is None else colors
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4))

    out, floors = {}, set()
    for name, dfs in groups.items():
        P = pool if np.isscalar(pool) else pool[name]
        col = colors.get(name, "#666666")
        t = tail_prob_by_k(dfs, P, alternative=alternative, n_iter=n_iter, seed=seed)
        out[name] = t
        floors.update(1 / (1 + r) for r in t.n_iter[t.method == "monte-carlo"])

        lo = np.clip(t.p_mean - t.p_sd, 1e-12, None)
        ax.plot(t.k, t.p_mean, "-o", color=col, lw=2, ms=5, label=f"{name} observed", zorder=3)
        ax.fill_between(t.k, lo, t.p_mean + t.p_sd, color=col, alpha=0.2, lw=0, zorder=2)
        if show_points:
            ax.scatter(np.repeat(t.k, 2), np.r_[t.p_min, t.p_max].reshape(2, -1).T.ravel(),
                       color=col, s=10, marker="_", zorder=4)

    ax.axhline(alpha, color="black", ls="--", lw=1.2, zorder=3, label=f"α = {alpha}")
    if floors:
        f = max(floors)
        ax.axhline(f, color="grey", ls=":", lw=1.0, zorder=3, label=f"MC floor (p < {f:.0e})")

    ax.set_yscale("log")
    ax.set_ylim(top=1.5)
    ax.set_xticks(sorted({k for t in out.values() for k in t.k}))
    ax.set_xlabel("number of seeds in the intersection (k)")
    ax.set_ylabel(f"p ({alternative}) under random selection")
    ax.set_title("Tail probability vs. number of seeds\n"
                 "solid = mean over k-subsets, band = ± SD, ticks = min/max", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return pd.concat(out, names=["group"]).reset_index(level=0)


def overlap_matrix(dfs: Sequence[pd.DataFrame], labels: Sequence | None = None):
    """Pairwise overlap counts between the seed gene sets; diagonal is each set's own size.

    Returns ``(overlap, common)`` -- the matrix, and the genes present in *every* seed.
    """
    names = [str(l) for l in labels] if labels is not None else [f"df{i}" for i in range(len(dfs))]
    idx = [set(map(str, df.index)) for df in dfs]
    overlap = pd.DataFrame(0, index=names, columns=names)
    for (i, a), (j, b) in itertools.combinations(enumerate(idx), 2):
        overlap.iloc[i, j] = overlap.iloc[j, i] = len(a & b)
    for i, a in enumerate(idx):
        overlap.iloc[i, i] = len(a)
    common = reduce(lambda a, b: a & b, idx) if idx else set()
    return overlap, common


def local_global_split(df_local: pd.DataFrame, df_global: pd.DataFrame) -> dict:
    """Genes selected by both decoders, and by only one of them, for a single seed."""
    local_genes, global_genes = pd.Index(df_local.index), pd.Index(df_global.index)
    return {
        "shared": global_genes.intersection(local_genes).tolist(),
        "unique_local": local_genes.difference(global_genes).tolist(),
        "unique_global": global_genes.difference(local_genes).tolist(),
    }
