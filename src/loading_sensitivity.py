"""Single-model sensitivity scores for the InterScale gene loadings, and a way to judge them.

The problem
-----------
``notebooks/legnini_23/4_a_1_decoder_stability.ipynb`` shows that the published gene->program
rule -- rank genes inside a latent dimension by ``|standardized loading| * sd(X)`` and keep the
top K -- returns a different gene set on every training seed. A handful of genes appear in all
five seeds, most appear in one or two. This module asks the constructive version of that
question:

    Using ONE model and ONE seed, can we score genes so that the high-scoring ones are the ones
    the other seeds would also have found?

Every score here is computable from a single trained model. The validation is leave-one-seed-out:
score genes with seed *s* alone, then ask how well that ranking predicts recurrence across the
*other four* seeds (:func:`evaluate`). A score is only interesting if it beats both the selection
statistic it is meant to prune and the trivial expression controls.

.. important::
   **The published selection is a function of the frozen decoder weights alone. No perturbation
   of the DATA can move it.**

   With ``S[f,e] = W[f,e] * sd(Z_e) / sd(X_f)`` and ``rank_by="loading_x_sd"`` multiplying by
   ``sd(X_f)`` from the *same* layer that ``gene_loadings`` standardized with, the two
   ``sd(X_f)`` cancel exactly and the ranking statistic is ``W[f,e] * sd(Z_e)``. ``sd(Z_e)`` is
   one number per dimension, so it cannot reorder genes inside a dimension. The within-dimension
   top-K is therefore the top-K of ``|W[:, e]|``, full stop.

   Consequences, all verified by :func:`prove_selection_is_data_independent`:

   * Permuting a gene's expression across cells changes nothing -- not even before the
     cancellation, since a permutation preserves ``sd(X_f)`` exactly.
   * Bootstrapping or subsampling cells changes nothing.
   * Masking, adding noise, dropping cell types: nothing, as long as the ranking layer and the
     standardization layer agree.

   The only channel through which data reaches the gene sets is *which dimensions* get selected
   (``calculate_dim_importance`` reads ``Z``). That is why the perturbation scores in this module
   are not framed as perturbations of the selection but as a measurement of something the
   selection ignores -- whether the ENCODER routes a gene into a dimension at all.

Layout
------
inputs        :func:`load_seed_arrays`, :func:`build_loading_adata`, :func:`load_perturbation`
diagnostics   :func:`prove_selection_is_data_independent`, :func:`dropout_report`
selection     :func:`RULES`, :func:`selected_sets`, :func:`recurrence`
scores        :func:`score_bank`
validation    :func:`evaluate`, :func:`summarise`
plots         :func:`plot_score_comparison`, :func:`plot_recurrence_vs_score`
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from interscale.evaluation import get_genes_dim

__all__ = [
    "SEEDS",
    "RULES",
    "load_seed_arrays",
    "build_loading_adata",
    "load_perturbation",
    "prove_selection_is_data_independent",
    "dropout_report",
    "selected_sets",
    "recurrence",
    "score_bank",
    "evaluate",
    "summarise",
    "plot_score_comparison",
    "plot_recurrence_vs_score",
]

SEEDS = [40, 41, 42, 43, 44]

# The two selection rules this repo actually uses, so a score is judged against the rule it would
# be applied to rather than against a convenient one.
#
#   all3    notebooks/legnini_23/4_a_1_decoder_stability.ipynb, "Gene loadings" section:
#           every dimension, top 3 genes each, no filtering.
#   left20  scripts/legnini_23/4a_seed_stability_extract.py: only the dimensions
#           calculate_dim_importance kept at cumulative_cutoff 0.60, top 20 each, with the
#           expression and specificity filters on.
RULES = {
    "all3": dict(dims="all", n_top=3, min_frac=0.0, enforce_specificity=False),
    "left20": dict(dims="left", n_top=20, min_frac=0.1, enforce_specificity=True),
}

# Higher recurrence than this counts as "the other seeds found it too". 3 of 4 held-out seeds is
# a majority without demanding unanimity, which on these sets is a handful of genes.
RECURRENT_MIN = 3


# ------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------


def load_seed_arrays(analysis_dir: str | Path, seeds: Sequence[int] = SEEDS):
    """Per-seed arrays and metadata written by ``4a_seed_stability_extract.py``.

    Returns ``(arrays, meta)``, both keyed by seed. ``arrays[seed]`` holds ``S_local``,
    ``S_global`` (standardized loadings, genes x dims), ``W_local``/``W_global`` (raw decoder
    weights), ``Z_local``/``Z_global`` (embeddings), ``importance_*``, ``dims_*`` and the
    per-gene reconstruction scores.

    Using these rather than re-running inference is not only faster: it guarantees the scores are
    computed on exactly the arrays the published cross-seed comparison was computed on.
    """
    analysis_dir = Path(analysis_dir)
    arrays, meta = {}, {}
    for s in seeds:
        arrays[s] = dict(np.load(analysis_dir / f"seed{s}_arrays.npz", allow_pickle=True))
        meta[s] = json.loads((analysis_dir / f"seed{s}_meta.json").read_text())
    names = [tuple(map(str, arrays[s]["var_names"])) for s in seeds]
    if len(set(names)) != 1:
        raise ValueError("seeds disagree on var_names; they are not the same gene panel")
    return arrays, meta


def build_loading_adata(arrays: Mapping[int, dict], X: np.ndarray, layer_key: str = "log1p_norm"):
    """A minimal AnnData carrying every seed's loadings, so interscale's own selection can re-run.

    The point is that :func:`selected_sets` calls :func:`interscale.evaluation.get_genes_dim`
    itself rather than reimplementing the top-K rule. A reimplementation would be one more thing
    that can silently disagree with the figure it is supposed to explain; this cannot, and the
    agreement is checked against ``meta[seed]["program_*"]`` in
    ``scripts/legnini_23/4h_sensitivity_benchmark.py``.

    Parameters
    ----------
    arrays
        As returned by :func:`load_seed_arrays`.
    X
        Dense ``(n_cells, n_genes)`` expression on the layer the decoders reconstruct, filtered
        exactly as training was (``remove_zero_expression_cells``).
    """
    seeds = list(arrays)
    genes = [str(g) for g in arrays[seeds[0]]["var_names"]]
    if X.shape[1] != len(genes):
        raise ValueError(f"X has {X.shape[1]} genes, the arrays have {len(genes)}")

    # .X is left at zeros on purpose: every call below passes X_layer=layer_key explicitly, so a
    # meaningful .X would only be a second copy of the same matrix that nothing reads.
    adata = ad.AnnData(
        X=np.zeros(X.shape, dtype=np.float32),
        var=pd.DataFrame(index=pd.Index(genes)),
    )
    adata.layers[layer_key] = np.asarray(X, dtype=np.float32)
    for s in seeds:
        for which in ("local", "global"):
            adata.varm[f"{s}_{which}_std_gene_loadings"] = np.asarray(arrays[s][f"S_{which}"], float)
            adata.obsm[f"{s}_{which}_emb"] = np.asarray(arrays[s][f"Z_{which}"], np.float32)
            imp = np.asarray(arrays[s][f"importance_{which}"], float)
            adata.uns[f"{s}_{which}_dim_importance"] = {
                "importance": imp,
                "dims_sorted": np.argsort(-imp),
                "dims_left": np.asarray(arrays[s][f"dims_{which}"], int),
                "which": which,
            }
    return adata


def load_perturbation(analysis_dir: str | Path, seeds: Sequence[int] = SEEDS):
    """Per-gene input-perturbation shifts written by ``4g_perturbation_sensitivity.py``.

    Returns ``{seed: {"dZ_mask_local": (G, E), ...}}``, or an empty dict with a warning if the
    npz files are not there -- the rest of the module works without them, it just cannot score
    the perturbation family.
    """
    analysis_dir = Path(analysis_dir)
    out = {}
    for s in seeds:
        p = analysis_dir / f"seed{s}_perturbation.npz"
        if p.exists():
            out[s] = dict(np.load(p, allow_pickle=True))
    missing = [s for s in seeds if s not in out]
    if missing:
        warnings.warn(
            f"no perturbation npz for seeds {missing} in {analysis_dir}; run "
            "scripts/legnini_23/4g_perturbation_sensitivity.py to add the dZ_* scores",
            stacklevel=2,
        )
    return out


# ------------------------------------------------------------------------------------------
# diagnostics
# ------------------------------------------------------------------------------------------


def prove_selection_is_data_independent(arrays, X, seed, which, n_resample: int = 50, rng=None):
    """Show numerically that no resampling of the cells can change the within-dimension top-K.

    Three checks, all on the ranking statistic ``|S[f,e]| * sd(X_f)`` that ``get_genes_dim`` uses
    with ``rank_by="loading_x_sd"``:

    ``max_rel_diff``
        How close that statistic is to ``|W[f,e]| * sd(Z_e)``, i.e. how completely the two
        ``sd(X_f)`` factors cancel. Order 1e-4, the difference being that ``gene_loadings``
        computes ``sd(X)`` with ``ddof=1`` through ``fast_array_utils`` while ``get_genes_dim``
        recomputes it separately.
    ``rankings_identical``
        Whether ranking genes by the statistic and by ``|W[:, e]|`` gives the same order in every
        dimension. It does.
    ``n_selection_changes``
        Over ``n_resample`` 50% cell subsamples, how many times the top-K set of any dimension
        changed after re-standardizing the loadings and re-ranking on the subsample. It is zero,
        which is the practical statement: bootstrap stability selection on this rule is a
        constant function and returns 1.0 for every selected gene and 0.0 for every other.

    Returns a dict with those three fields plus ``n_top`` and ``n_resample``.

    Notes
    -----
    A within-gene permutation is not even listed: permuting a column of ``X`` leaves ``sd(X_f)``
    identical, so ``S`` is unchanged before any cancellation argument is needed. This is the same
    degeneracy ``src/loading_stability.py`` documents for the permutation null, arrived at from
    the selection side rather than the p-value side.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    S = np.asarray(arrays[seed][f"S_{which}"], float)
    W = np.asarray(arrays[seed][f"W_{which}"], float)
    Z = np.asarray(arrays[seed][f"Z_{which}"], float)
    X = np.asarray(X, float)
    n_cells, n_dims = X.shape[0], S.shape[1]

    stat = np.abs(S * X.std(0, ddof=1)[:, None])
    weight_only = np.abs(W * Z.std(0, ddof=1)[None, :])
    max_rel = float(np.abs(stat - weight_only).max() / np.abs(weight_only).max())
    identical = all(
        (np.argsort(-stat[:, e]) == np.argsort(-weight_only[:, e])).all() for e in range(n_dims)
    )

    changes, K = 0, 3
    ref = [set(np.argsort(-stat[:, e])[:K]) for e in range(n_dims)]
    for _ in range(n_resample):
        idx = rng.choice(n_cells, n_cells // 2, replace=False)
        sd_b, zsd_b = X[idx].std(0, ddof=1), Z[idx].std(0, ddof=1)
        # Re-standardize on the subsample and re-rank on the subsample, i.e. redo everything
        # gene_loadings + get_genes_dim would do if handed only these cells.
        stat_b = np.abs(W * zsd_b[None, :] / np.maximum(sd_b, 1e-8)[:, None] * sd_b[:, None])
        changes += sum(set(np.argsort(-stat_b[:, e])[:K]) != ref[e] for e in range(n_dims))
    return {
        "seed": seed,
        "which": which,
        "n_top": K,
        "n_resample": n_resample,
        "max_rel_diff": max_rel,
        "rankings_identical": bool(identical),
        "n_selection_changes": int(changes),
    }


def dropout_report(arrays, perturbation, seeds: Sequence[int] = SEEDS) -> pd.DataFrame:
    """Compare the published embeddings against dropout-free ones, per seed and component.

    ``BaseModel.load()`` does not call ``.eval()`` and neither does ``get_model_output``, so every
    embedding in ``seed*_arrays.npz`` -- and therefore every ``sd(Z_e)``, every dimension
    importance, and every dimension selection built on them -- was produced with dropout active
    (0.1 on both components). ``4g_perturbation_sensitivity.py`` runs the same forward pass in
    eval mode, so the two can be compared directly.

    Returns one row per (seed, component) with the mean per-dimension Pearson r between the two
    embeddings and the ratio of their per-dimension standard deviations. A ratio below 1 means
    dropout inflated ``sd(Z_e)``, which scales the standardized loadings of that dimension.

    This does not invalidate the gene sets -- ``sd(Z_e)`` is a per-dimension constant and cannot
    reorder genes (see :func:`prove_selection_is_data_independent`) -- but it does feed
    ``calculate_dim_importance``, which decides *which* dimensions the ``left20`` rule reads.
    """
    rows = []
    for s in seeds:
        if s not in perturbation:
            continue
        for which in ("local", "global"):
            a = np.asarray(perturbation[s][f"Z_{which}"], float)  # eval mode
            b = np.asarray(arrays[s][f"Z_{which}"], float)  # as published, dropout on
            r = float(np.mean([np.corrcoef(a[:, e], b[:, e])[0, 1] for e in range(a.shape[1])]))
            rows.append(
                {
                    "seed": s,
                    "which": which,
                    "mean_dim_r": r,
                    "sd_ratio_eval_over_published": float(np.mean(a.std(0) / b.std(0))),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------
# selection and the prediction target
# ------------------------------------------------------------------------------------------


def _dims_for(rule: str, arrays, seed: int, which: str, n_embed: int) -> list[int]:
    return (
        list(range(n_embed))
        if RULES[rule]["dims"] == "all"
        else [int(d) for d in arrays[seed][f"dims_{which}"]]
    )


def selected_sets(adata, rule: str, which: str, seeds: Sequence[int] = SEEDS,
                  layer_key: str = "log1p_norm") -> dict[int, set]:
    """The gene set each seed's model selects under ``rule``, via ``get_genes_dim`` itself."""
    r = RULES[rule]
    n_embed = adata.varm[f"{seeds[0]}_{which}_std_gene_loadings"].shape[1]
    dims = list(range(n_embed)) if r["dims"] == "all" else None
    out = {}
    for s in seeds:
        with warnings.catch_warnings():
            # get_genes_dim warns when explicit dims differ from the stored dims_left. For the
            # "all" rule that is the intent, not a mistake.
            warnings.simplefilter("ignore", UserWarning)
            df = get_genes_dim(
                adata, which=which, prefix=s, dims=dims, n_top=r["n_top"], X_layer=layer_key,
                min_frac=r["min_frac"], rank_by="loading_x_sd",
                enforce_specificity=r["enforce_specificity"], specificity_min=1.5, plot=False,
            )
        out[s] = set(map(str, df.index))
    return out


def recurrence(sets: Mapping[int, set], genes: Sequence[str], holdout: int | None = None):
    """How many seeds selected each gene; excluding ``holdout`` when given.

    ``holdout`` is what makes the validation honest. A score computed on seed *s* is judged
    against how often the OTHER seeds selected the gene, so the seed the score came from
    contributes nothing to its own target.
    """
    others = [s for s in sets if s != holdout]
    return np.array([sum(str(g) in sets[s] for s in others) for g in genes], dtype=int)


# ------------------------------------------------------------------------------------------
# score bank
# ------------------------------------------------------------------------------------------


def score_bank(arrays, X, seed: int, which: str, rule: str, *, perturbation=None,
               n_jitter: int = 200, n_perm: int = 200, rng=None) -> dict[str, np.ndarray]:
    """Every candidate single-seed sensitivity score, as ``{name: (n_genes,) array}``.

    All are oriented so that larger means "more likely to recur". They fall into four families.

    **The selection statistic and trivial controls.** ``loading_x_sd`` is the quantity the rule
    itself ranks on -- any score that does not beat it is not worth having. ``loading`` is the
    same without the ``sd(X)`` factor, ``expr_sd`` and ``expr_frac`` are the gene's expression
    spread and detection rate, and ``recon_pearson`` is how well the decoder reconstructs the
    gene at all. The expression controls are the ones that matter: if they win, cross-seed
    recurrence is mostly "the decoder reliably puts weight on variable genes" and no sensitivity
    score is required.

    **Geometry of the top-K cut** (functions of the frozen weights, no data). ``margin_rel`` is
    how far above the K-th place cutoff the gene sits in its best dimension, i.e. distance to the
    decision boundary the rule draws. ``within_dim_z`` is the same idea standardized by how
    spread that dimension's loadings are. ``n_dims_top`` counts how many dimensions select the
    gene, and ``sum_loading_x_sd`` is its total loading mass over the scored dimensions.

    **Weight-space stability selection** (``jitter_freq_*``). The Meinshausen-Buhlmann recipe,
    applied where it can actually do something: perturb the *loadings* with Gaussian noise at
    ``eta`` times the dimension's own loading spread and count how often the gene still makes the
    top K. Data resampling is the usual way to run stability selection and is provably inert here
    (:func:`prove_selection_is_data_independent`), so the resampling has to happen in the space
    that differs between seeds -- the decoder weights.

    **Data-driven association.** ``emp_r`` is ``|Pearson r(Z_e, X_f)|``, the empirical
    counterpart of the decoder loading that ``src/loading_stability.py`` derives, and
    ``emp_perm_z`` is its permutation z-score -- the score
    ``4_a_1_decoder_stability.ipynb``'s ``perm_q_corr`` cell computes. ``coherence`` is that
    notebook's leave-one-out coherence prune, turned into a ranking instead of a threshold.
    ``loo_module_corr`` is the decoupleR/MOFAcell-style reading: build the dimension's weighted
    module score over its top-K genes, and score each gene by how much the module's correlation
    with its own latent dimension drops when that gene is left out.

    **Input perturbation** (only when ``perturbation`` is given). ``dZ_mask`` and ``dZ_permute``
    are the relative L2 shift of the latent dimension when the gene's input column is replaced by
    the mask value or shuffled across cells -- how strongly the ENCODER depends on the gene.
    ``dZ_*_per_input`` divides by the norm of the input change, so a gene does not score highly
    merely because perturbing it moves the input a long way; that is the sensitivity proper, and
    the version to read when comparing against ``expr_sd``. ``dZ_*_x_loading`` multiplies the raw
    shift by the decoder loading, i.e. requires the gene to be both read in by the encoder and
    written out by the decoder on the same dimension.

    Parameters
    ----------
    perturbation
        ``{seed: npz}`` from :func:`load_perturbation`. Omit to skip the ``dZ_*`` family.
    n_jitter, n_perm
        Draws for the weight-jitter and cell-permutation nulls.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    r = RULES[rule]
    K = int(r["n_top"])
    S = np.asarray(arrays[seed][f"S_{which}"], float)
    Z = np.asarray(arrays[seed][f"Z_{which}"], float)
    X = np.asarray(X, float)
    n_cells, n_genes = X.shape
    D = _dims_for(rule, arrays, seed, which, S.shape[1])

    sd = X.std(0, ddof=1)
    stat = np.abs(S[:, D] * sd[:, None])  # (G, |D|) the ranking statistic itself
    out: dict[str, np.ndarray] = {}

    # --- the rule's own statistic, and controls -------------------------------------------
    out["loading_x_sd"] = stat.max(1)
    out["loading"] = np.abs(S[:, D]).max(1)
    out["sum_loading_x_sd"] = stat.sum(1)
    out["expr_sd"] = sd.copy()
    out["expr_frac"] = (X > 0).mean(0)
    out["recon_pearson"] = np.asarray(arrays[seed][f"pearson_{which}"], float)

    # --- geometry of the cut ---------------------------------------------------------------
    cut = np.sort(stat, axis=0)[::-1][K - 1]  # K-th largest value in each dimension
    out["margin_rel"] = ((stat - cut) / (cut + 1e-12)).max(1)
    out["n_dims_top"] = (stat >= cut).sum(1).astype(float)
    out["within_dim_z"] = ((stat - stat.mean(0)) / (stat.std(0) + 1e-12)).max(1)

    # --- weight-space stability selection ---------------------------------------------------
    for eta in (0.25, 0.5):
        hit = np.zeros(n_genes)
        scale = eta * stat.std(0)
        for _ in range(n_jitter):
            s2 = np.abs(stat + rng.normal(0.0, scale, size=stat.shape))
            hit += (s2 >= np.sort(s2, axis=0)[::-1][K - 1]).any(1)
        out[f"jitter_freq_{eta}"] = hit / n_jitter

    # --- empirical loading and its permutation null -----------------------------------------
    Xz = (X - X.mean(0)) / (sd + 1e-12)
    Zd = Z[:, D]
    Zz = (Zd - Zd.mean(0)) / (Zd.std(0) + 1e-12)
    R = Xz.T @ Zz / n_cells
    out["emp_r"] = np.abs(R).max(1)
    s1 = np.zeros_like(R)
    s2 = np.zeros_like(R)
    for _ in range(n_perm):
        # Permuting the cell order of Z against unpermuted X destroys the pairing and keeps both
        # marginals; the same null src/loading_stability.py states.
        Rp = Xz.T @ Zz[rng.permutation(n_cells)] / n_cells
        s1 += Rp
        s2 += Rp**2
    mu = s1 / n_perm
    sg = np.sqrt(np.maximum(s2 / n_perm - mu**2, 0.0)) + 1e-12
    out["emp_perm_z"] = (np.abs(R - mu) / sg).max(1)

    # --- module-level scores over each dimension's own top-K --------------------------------
    loo = np.full((n_genes, len(D)), -np.inf)
    coh = np.full((n_genes, len(D)), -np.inf)
    for j in range(len(D)):
        top = np.argsort(-stat[:, j])[:K]
        w = (S[top, j] * sd[top])  # signed, same statistic
        z = Zz[:, j]
        if len(top) < 2:
            continue
        full = abs(np.corrcoef(Xz[:, top] @ w / (np.abs(w).sum() + 1e-12), z)[0, 1])
        for i, g in enumerate(top):
            keep = np.ones(len(top), bool)
            keep[i] = False
            wk = w[keep]
            module = Xz[:, top[keep]] @ wk / (np.abs(wk).sum() + 1e-12)
            if module.std() < 1e-9:
                continue
            loo[g, j] = full - abs(np.corrcoef(module, z)[0, 1])
            # sign(loading) orients the gene by the direction its own loading predicts, so a
            # correctly anti-correlated negative-loading gene scores high rather than being
            # dropped -- the notebook's coherence_prune convention.
            coh[g, j] = np.sign(w[i]) * np.corrcoef(Xz[:, g], module)[0, 1]
    out["loo_module_corr"] = np.where(np.isfinite(loo.max(1)), loo.max(1), -1.0)
    out["coherence"] = np.where(np.isfinite(coh.max(1)), coh.max(1), -1.0)

    # A dimension-level quality score handed down to its genes: how well the weighted module
    # score of the dimension's own top-K tracks the latent dimension it was read off. A gene sits
    # in a dimension the loadings actually describe, or in one where they do not.
    dim_q = np.zeros(len(D))
    for j in range(len(D)):
        top = np.argsort(-stat[:, j])[:K]
        w = S[top, j] * sd[top]
        module = Xz[:, top] @ w / (np.abs(w).sum() + 1e-12)
        dim_q[j] = 0.0 if module.std() < 1e-9 else abs(np.corrcoef(module, Zz[:, j])[0, 1])
    in_top = stat >= cut
    out["dim_module_corr"] = np.where(in_top, dim_q[None, :], 0.0).max(1)

    # A deliberately unclever combination of the two families: the gene's rank on the rule's own
    # statistic averaged with its rank on plain expression spread. It is here because the two
    # carry different information -- one is the frozen decoder, the other is the data -- not
    # because it was tuned on the benchmark below.
    def _rank(v):
        return pd.Series(v).rank().to_numpy() / len(v)

    out["ens_loading_expr"] = 0.5 * (_rank(out["loading_x_sd"]) + _rank(out["expr_sd"]))

    # The control turned into a question. If plain expression spread predicts recurrence as well
    # as the model's own statistic does, then "top-loading gene" is largely a restatement of
    # "high-variance gene", and the interesting quantity is what the loading says ON TOP of that.
    # This is the residual of the loading rank after regressing out the expression-spread rank;
    # near-chance performance here is the honest finding that there is no seed-stable signal
    # beyond expression variance, not a failure of the score.
    rl, re_ = _rank(out["loading_x_sd"]), _rank(out["expr_sd"])
    beta = np.polyfit(re_, rl, 1)
    out["resid_loading_vs_expr"] = rl - np.polyval(beta, re_)

    # --- input perturbation ------------------------------------------------------------------
    if perturbation and seed in perturbation:
        p = perturbation[seed]
        for tag in ("mask", "permute"):
            key = f"dZ_{tag}_{which}"
            if key not in p:
                continue
            dZ = np.asarray(p[key], float)[:, D]
            out[f"dZ_{tag}"] = dZ.max(1)
            out[f"dZ_{tag}_x_loading"] = (dZ * stat).max(1)
            # Raw dZ is confounded by how big the perturbation itself was: masking a highly
            # expressed gene moves the input further than masking a quiet one, and permuting a
            # high-variance gene moves it further than permuting a flat one -- so raw dZ would
            # partly re-measure `expr_sd`. Dividing by the norm of the input change turns it into
            # a finite-difference directional derivative, which is the sensitivity proper.
            #   permute:  E||x_perm - x||^2 = 2 n var(x_f)  ->  sqrt(2n) * sd_f
            #   mask:     ||x_f - MASK_VALUE|| with MASK_VALUE = -1
            step = (np.sqrt(2 * n_cells) * sd if tag == "permute"
                    else np.linalg.norm(X + 1.0, axis=0))
            out[f"dZ_{tag}_per_input"] = (dZ / (step[:, None] + 1e-12)).max(1)

    return out


# ------------------------------------------------------------------------------------------
# validation
# ------------------------------------------------------------------------------------------


def evaluate(adata, arrays, X, rule: str, which: str, *, perturbation=None,
             seeds: Sequence[int] = SEEDS, layer_key: str = "log1p_norm",
             recurrent_min: int = RECURRENT_MIN, rng=None) -> pd.DataFrame:
    """Leave-one-seed-out: how well does each score predict what the other seeds selected?

    For every held-out seed the scores are computed from that seed alone and compared against
    ``recurrence`` over the remaining four. Two scopes are reported, because they answer
    different questions:

    ``all_genes``
        Over the whole panel. This is "can the score find the reproducible genes from scratch".
    ``selected_only``
        Over the genes this seed's rule actually selected. This is the operational question --
        given one model's gene list, which entries should be trusted -- and it is the harder one,
        because the selection has already removed the easy negatives.

    Metrics per (seed, score, scope): ``auroc`` and ``ap`` for "selected by at least
    ``recurrent_min`` of the four held-out seeds", and ``spearman`` against the recurrence count
    itself. ``auroc`` is ``NaN`` when a scope has no positives or no negatives.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    genes = np.asarray([str(g) for g in adata.var_names])
    sets = selected_sets(adata, rule, which, seeds=seeds, layer_key=layer_key)

    rows = []
    for s in seeds:
        y = recurrence(sets, genes, holdout=s)
        in_sel = np.array([g in sets[s] for g in genes])
        bank = score_bank(arrays, X, s, which, rule, perturbation=perturbation, rng=rng)
        for name, v in bank.items():
            v = np.asarray(v, float)
            for scope, mask in (("all_genes", np.ones(len(genes), bool)), ("selected_only", in_sel)):
                yy, vv = y[mask], v[mask]
                pos = yy >= recurrent_min
                degenerate = pos.sum() in (0, len(yy)) or np.ptp(vv) == 0
                rows.append(
                    {
                        "rule": rule,
                        "which": which,
                        "seed": s,
                        "score": name,
                        "scope": scope,
                        "n": int(mask.sum()),
                        "n_recurrent": int(pos.sum()),
                        "auroc": np.nan if degenerate else roc_auc_score(pos, vv),
                        "ap": np.nan if degenerate else average_precision_score(pos, vv),
                        "spearman": np.nan if np.ptp(vv) == 0 else spearmanr(vv, yy).statistic,
                    }
                )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Mean +/- SD of ``metric`` over seeds, one row per score and one column per setting."""
    piv = (
        results.groupby(["rule", "which", "scope", "score"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    piv["setting"] = piv["rule"] + " / " + piv["which"] + " / " + piv["scope"]
    wide = piv.pivot(index="score", columns="setting", values="mean")
    return wide.assign(overall=wide.mean(axis=1)).sort_values("overall", ascending=False)


# ------------------------------------------------------------------------------------------
# plots
# ------------------------------------------------------------------------------------------


def plot_score_comparison(results: pd.DataFrame, metric: str = "auroc", scope: str = "selected_only",
                          ax=None, baseline: str = "loading_x_sd", figsize=(7.5, 7.5)):
    """One panel: every score's ``metric``, mean +/- SD over the five held-out seeds.

    Points are the four (rule, component) settings; the bar is their mean. The dashed line marks
    the selection statistic itself -- a score left of it is not adding information -- and the
    dotted line marks chance.
    """
    sub = results.query("scope == @scope").copy()
    sub["setting"] = sub["rule"] + " / " + sub["which"]
    per = sub.groupby(["score", "setting"])[metric].mean().reset_index()
    order = per.groupby("score")[metric].mean().sort_values().index
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    y = {g: i for i, g in enumerate(order)}
    ax.barh(list(y.values()), [per.query("score == @g")[metric].mean() for g in order],
            color="#cfd8dc", height=0.7, zorder=1)
    markers = {"all3 / local": "o", "all3 / global": "s",
               "left20 / local": "^", "left20 / global": "D"}
    colors = {"local": "#EE9B00", "global": "#27828E"}
    for setting, grp in per.groupby("setting"):
        ax.scatter([grp.set_index("score").loc[g, metric] for g in order],
                   [y[g] for g in order], s=26, zorder=3, label=setting,
                   marker=markers.get(setting, "o"),
                   color=colors["local" if "local" in setting else "global"],
                   edgecolor="white", linewidth=0.5)
    base = per.query("score == @baseline")[metric].mean()
    ax.axvline(base, color="black", ls="--", lw=1.2, zorder=2, label=f"{baseline} (the rule itself)")
    if metric in ("auroc",):
        ax.axvline(0.5, color="grey", ls=":", lw=1.0, zorder=2, label="chance")
    ax.set_yticks(list(y.values()))
    ax.set_yticklabels(list(order))
    ax.set_xlabel(f"{metric} for 'selected by >= {RECURRENT_MIN} of the 4 held-out seeds'")
    ax.set_title(f"Single-seed scores vs. cross-seed recurrence ({scope})", fontsize=10)
    # Outside the axes: with ~20 scores the bars reach the right edge and an inside legend
    # covers the very rows a reader is comparing.
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.01)
    plt.tight_layout()
    return ax


def plot_recurrence_vs_score(adata, arrays, X, score: str, rule: str, which: str, *,
                             perturbation=None, seeds: Sequence[int] = SEEDS,
                             layer_key: str = "log1p_norm", ax=None, annotate: int = 8):
    """The winning score against held-out recurrence, pooled over the five leave-one-out folds."""
    genes = np.asarray([str(g) for g in adata.var_names])
    sets = selected_sets(adata, rule, which, seeds=seeds, layer_key=layer_key)
    rng = np.random.default_rng(0)
    xs, ys, lab = [], [], []
    for s in seeds:
        y = recurrence(sets, genes, holdout=s)
        in_sel = np.array([g in sets[s] for g in genes])
        v = score_bank(arrays, X, s, which, rule, perturbation=perturbation, rng=rng)[score]
        xs.append(np.asarray(v)[in_sel])
        ys.append(y[in_sel])
        lab.append(genes[in_sel])
    xs, ys, lab = np.concatenate(xs), np.concatenate(ys), np.concatenate(lab)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))
    jitter = np.random.default_rng(1).normal(0, 0.07, size=len(ys))
    ax.scatter(xs, ys + jitter, s=16, alpha=0.6,
               color="#EE9B00" if which == "local" else "#27828E", edgecolor="none")
    for i in np.argsort(-xs)[:annotate]:
        ax.annotate(lab[i], (xs[i], ys[i] + jitter[i]), fontsize=6,
                    xytext=(3, 2), textcoords="offset points")
    rho = spearmanr(xs, ys).statistic
    ax.set_xlabel(score)
    ax.set_ylabel("seeds (of 4 held out) that also selected the gene")
    ax.set_title(f"{rule} / {which}: {score} vs held-out recurrence (rho = {rho:.2f})", fontsize=9)
    ax.set_yticks(range(5))
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return ax
