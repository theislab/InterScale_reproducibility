"""Permutation test and cross-seed consensus filtering for InterScale gene loadings.

Why this module exists
----------------------
The published gene->program assignment is a **top-K rule**: rank genes by
``|standardized loading|`` within each selected dimension and keep the K largest
(``interscale.evaluation.get_genes_dim`` with ``rank_by="loading_x_sd"``). Across five
training seeds that rule is not stable -- see ``notebooks/legnini_23/4_a_decoder_stability.ipynb``
and ``notebooks/Cosmx/4_decoder_stability_cosmx.ipynb``. This module replaces the cutoff with a
significance test plus a cross-seed frequency requirement, **without retraining anything**: it
reads the per-seed arrays ``scripts/legnini_23/4a_seed_stability_extract.py`` already wrote.

Nothing here is imported by existing code, and nothing here modifies an existing file. The
decoder-based standardized loading is *read* from those arrays (where
``interscale.evaluation.gene_loadings`` put it), never recomputed.

.. important::
   **The permutation null cannot be applied to the decoder-based standardized loading.**

   That loading is ``S[f,e] = W[e,f] * std(Z[:,e]) / std(X[:,f])``. A permutation of the cell
   order of ``Z[:,e]`` leaves ``std(Z[:,e])`` unchanged -- standard deviation does not depend on
   row order -- and touches neither ``W`` nor ``std(X[:,f])``. So every permutation reproduces
   ``S[f,e]`` exactly and the empirical p-value is 1.0 for every gene and dimension, by
   construction rather than by absence of signal.
   :func:`demonstrate_null_is_degenerate` shows this on real arrays.

   ``S`` is a deterministic function of frozen weights; it has no sampling distribution over
   cell orderings for a permutation to explore. What *is* testable is the quantity ``S`` is a
   claim about: whether dimension ``e``'s variation across cells is associated with gene ``f``'s
   measured expression. :func:`standardized_empirical_loading` is that quantity, obtained by
   substituting the decoder weight with its data-estimated counterpart in the *same*
   standardization:

       W_hat[f,e] = cov(X[:,f], Z[:,e]) / var(Z[:,e])          (univariate OLS slope)
       L[f,e]     = W_hat[f,e] * std(Z[:,e]) / std(X[:,f])
                  = cov(X[:,f], Z[:,e]) / (std(Z[:,e]) std(X[:,f]))
                  = Pearson r( Z[:,e], X[:,f] )

   Same formula, same standardization, empirical slope instead of the frozen one -- and the
   result is a correlation, which the permutation null does move.
   :func:`compare_decoder_and_empirical` reports how well the two agree per dimension, so the
   substitution is checked rather than assumed.

The null hypothesis
-------------------
Every test in this module tests one null, stated once here and repeated in each function's
docstring:

    **H0: gene f's expression across cells is independent of latent dimension e's value across
    cells** -- i.e. the gene<->dimension association is arbitrary.

The null is realised by permuting the cell order of ``Z[:,e]`` alone, against unpermuted
``X[:,f]``. That destroys the pairing while preserving both marginal distributions exactly (the
same cell values, reassigned to cells), so it controls for each gene's expression distribution
and each dimension's activation distribution.

What it does **not** claim:

* It is not a test that dimension ``e`` *causes* gene ``f``, nor that the decoder uses ``f`` to
  reconstruct ``e``.
* It is not a test of the decoder weight ``W[e,f]``; see the note above.
* It does not test that dimension ``e`` is reproducible across seeds. That is what the
  dimension matching (:func:`match_dimensions`) and the frequency rule
  (:func:`consensus_frequency`) are for.
* With n in the tens or hundreds of thousands of cells the null distribution of ``r`` has
  sd ``~1/sqrt(n)``, so **very small correlations are significant**. ``q < alpha`` is therefore
  a weak filter on these datasets, not a strict one; read :func:`significance_report`'s
  saturation columns before treating it as selective.

Layout
------
loading             :func:`standardized_empirical_loading`, :func:`compare_decoder_and_empirical`
step 1 permutation  :func:`benjamini_hochberg`, :func:`permutation_test_loadings`,
                    :func:`demonstrate_null_is_degenerate`, :func:`significance_report`
step 2 per seed     :func:`load_seed_arrays`, :func:`load_expression`,
                    :func:`significant_sets`, :func:`match_dimensions`,
                    :func:`match_all_to_reference`, :func:`matching_report`
step 3 consensus    :func:`consensus_frequency`, :func:`consensus_genes`
step 5 api          :func:`get_genes_dim_consensus`
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

__all__ = [
    "COMPONENTS",
    "SeedArrays",
    "standardized_empirical_loading",
    "standardized_partial_loading",
    "compare_decoder_and_empirical",
    "embedding_rank_report",
    "demonstrate_loading_gauge_freedom",
    "benjamini_hochberg",
    "permutation_test_loadings",
    "demonstrate_null_is_degenerate",
    "significance_report",
    "load_seed_arrays",
    "load_expression",
    "significant_sets",
    "size_matched_sets",
    "select_genes_permutation",
    "agreement_with_other_runs",
    "match_dimensions",
    "match_all_to_reference",
    "matching_report",
    "consensus_frequency",
    "consensus_genes",
    "get_genes_dim_consensus",
]

COMPONENTS = ("local", "global")

FDR_SCOPES = ("per_dimension", "global")

STATISTICS = ("marginal", "partial")


# ==========================================================================================
# the loading being tested
# ==========================================================================================


def _center(A: np.ndarray) -> np.ndarray:
    return A - A.mean(axis=0, keepdims=True)


def _residual_on(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Residual of the columns of ``A`` after least-squares projection onto those of ``B``.

    Solved through the normal equations with :func:`numpy.linalg.lstsq`, so a rank-deficient
    ``B`` -- correlated latent dimensions are the norm here -- degrades to the minimum-norm
    solution instead of raising.
    """
    if B.shape[1] == 0:
        return A - A.mean(axis=0, keepdims=True)
    coef = np.linalg.lstsq(B.T @ B, B.T @ A, rcond=None)[0]
    return A - B @ coef


def standardized_partial_loading(
    Z: np.ndarray, X: np.ndarray, *, dims: Sequence[int] | None = None, eps: float = 1e-12
) -> np.ndarray:
    """Standardized loading with the **multivariate** OLS slope, ``(n_genes, n_dims_requested)``.

    The same standardization as :func:`standardized_empirical_loading` and as
    ``interscale.evaluation.gene_loadings``, but the empirical slope is dimension ``e``'s
    coefficient in a regression of gene ``f`` on **all** dimensions of the component at once --
    which is the model the linear decoder actually implements (``X_hat = W Z + b``)::

        r_e            = Z[:, e] residualized on the other dimensions
        W_hat[f, e]    = cov(r_e, X[:, f]) / var(r_e)
        L_partial[f,e] = W_hat[f, e] * std(Z[:, e]) / std(X[:, f])

    Use this rather than the marginal version whenever the latent dimensions are correlated --
    which they are, and which is why ``calculate_dim_importance`` scores them with
    ``mode="full"``. A marginal loading attributes the whole of a shared signal to every
    dimension that shares it; measured on legnini23 the marginal loading barely tracks the
    decoder's own loading at all (median |Spearman| 0.06) while this one does.
    :func:`compare_decoder_and_empirical` is how to check that on any given run rather than
    taking it on trust.
    """
    Z = np.asarray(Z, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"Z has {Z.shape[0]} cells, X has {X.shape[0]}")
    dims = list(range(Z.shape[1])) if dims is None else [int(d) for d in dims]

    Zc = _center(Z)
    Xc = _center(X)
    sz = Zc.std(axis=0, ddof=1)
    sx = Xc.std(axis=0, ddof=1)
    out = np.zeros((X.shape[1], len(dims)))
    for j, d in enumerate(dims):
        others = [k for k in range(Zc.shape[1]) if k != d]
        r = _residual_on(Zc[:, [d]], Zc[:, others])[:, 0]
        vr = float(r @ r)
        if vr <= eps or sz[d] <= eps:
            continue
        slope = (r @ Xc) / vr
        out[:, j] = np.where(sx > eps, slope * sz[d] / np.where(sx > eps, sx, 1.0), 0.0)
    return out


def standardized_empirical_loading(
    Z: np.ndarray, X: np.ndarray, *, dims: Sequence[int] | None = None, eps: float = 1e-12
) -> np.ndarray:
    """Data-side standardized loading, ``(n_genes, n_dims_requested)``.

    ``W_hat[f,e] * std(Z[:,e]) / std(X[:,f])`` with ``W_hat`` the univariate OLS slope of gene
    ``f`` on dimension ``e``, which reduces to ``Pearson r(Z[:,e], X[:,f])``. This is the
    permutation-testable counterpart of the decoder loading that
    ``interscale.evaluation.gene_loadings`` stores; see the module docstring for why the decoder
    version cannot be permutation-tested.

    A constant dimension or a constant gene gives 0 rather than ``nan``: 0 is the honest
    statement that no association is measurable, and it fails every significance threshold,
    whereas ``nan`` would silently drop out of a comparison.
    """
    Z = np.asarray(Z, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"Z has {Z.shape[0]} cells, X has {X.shape[0]}")
    dims = list(range(Z.shape[1])) if dims is None else [int(d) for d in dims]

    Zc = _center(Z[:, dims])
    Xc = _center(X)
    n = Z.shape[0]
    sz = Zc.std(axis=0, ddof=1)
    sx = Xc.std(axis=0, ddof=1)
    cov = (Zc.T @ Xc) / (n - 1)  # (n_dims, n_genes)
    denom = np.outer(sz, sx)
    out = np.where(denom > eps, cov / np.where(denom > eps, denom, 1.0), 0.0)
    return out.T  # (n_genes, n_dims)


def compare_decoder_and_empirical(
    S: np.ndarray, L: np.ndarray, dims: Sequence[int], *, gene_names=None
) -> pd.DataFrame:
    """Per-dimension agreement between the decoder loading and the empirical one.

    The substitution in :func:`standardized_empirical_loading` is only defensible if the two
    rank genes similarly. This reports Pearson and Spearman of ``S[:, e]`` against ``L[:, e]``
    per dimension, and the sign agreement, so the notebook can show it instead of asserting it.

    A dimension where the two disagree is not necessarily an error: ``S`` mixes dimensions
    through ``W`` while ``L`` is marginal, so correlated latent dimensions pull them apart.
    Low agreement is a reason to distrust *that dimension's* gene assignment either way.
    """
    from scipy.stats import spearmanr

    S = np.asarray(S, dtype=float)
    L = np.asarray(L, dtype=float)
    rows = []
    for j, d in enumerate(dims):
        s, l = S[:, int(d)], L[:, j]
        ok = np.isfinite(s) & np.isfinite(l)
        if ok.sum() < 3 or s[ok].std() == 0 or l[ok].std() == 0:
            rows.append(
                {
                    "dim": int(d),
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "abs_spearman": np.nan,
                    "sign_agreement": np.nan,
                    "n_genes": int(ok.sum()),
                }
            )
            continue
        r = float(np.corrcoef(s[ok], l[ok])[0, 1])
        rho = float(spearmanr(s[ok], l[ok]).statistic)
        rows.append(
            {
                "dim": int(d),
                "pearson": r,
                "spearman": rho,
                "abs_spearman": abs(rho),
                "sign_agreement": float((np.sign(s[ok]) == np.sign(l[ok])).mean()),
                "n_genes": int(ok.sum()),
            }
        )
    out = pd.DataFrame(rows)
    if gene_names is not None:
        out.attrs["n_genes_total"] = len(gene_names)
    return out


# ==========================================================================================
# identifiability of the loading itself
# ==========================================================================================


def embedding_rank_report(Z: np.ndarray, *, tol: float = 1e-6) -> dict:
    """Is the embedding full rank, and does it look like LayerNorm output?

    Both InterScale components turn out to be rank-deficient by exactly one, which is what
    :func:`demonstrate_loading_gauge_freedom` then shows the consequence of. The local component
    is unmistakably LayerNorm output -- every row sums to 0 and has norm exactly ``sqrt(D)`` --
    and the global component is close to it.

    ``tol`` is on the ratio of smallest to largest singular value of the *centred* matrix, at
    1e-6 rather than machine epsilon because the arrays are saved as float32: an exact linear
    dependency shows up at ~1e-7, not at 1e-16.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    Zc = _center(Z)
    sv = np.linalg.svd(Zc, compute_uv=False)
    row_sum = Z.sum(axis=1)
    row_norm = np.linalg.norm(Z, axis=1)
    ratio = float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0
    return {
        "n_dims": int(Z.shape[1]),
        "rank": int((sv > sv[0] * tol).sum()),
        "full_rank": bool((sv > sv[0] * tol).sum() == Z.shape[1]),
        "sv_ratio": ratio,
        "max_abs_row_sum": float(np.abs(row_sum).max()),
        "mean_row_norm": float(row_norm.mean()),
        "sd_row_norm": float(row_norm.std()),
        "row_norm_is_sqrt_d": bool(
            np.isclose(row_norm.mean(), np.sqrt(Z.shape[1]), rtol=1e-3) and row_norm.std() < 1e-3
        ),
        "looks_layer_normed": bool(
            np.abs(row_sum).max() < 1e-3 * max(row_norm.mean(), 1e-12)
            and row_norm.std() < 1e-3 * max(row_norm.mean(), 1e-12)
        ),
        "tol": float(tol),
    }


def demonstrate_loading_gauge_freedom(
    W: np.ndarray,
    Z: np.ndarray,
    x_std: np.ndarray,
    *,
    dims: Sequence[int] | None = None,
    n_top: int = 20,
    n_cells: int = 5000,
    scale: float = 1.0,
    random_state: int = 0,
) -> dict:
    """Why the top-K loading rule cannot be stable across seeds, shown rather than argued.

    The embedding is rank-deficient by one (:func:`embedding_rank_report`), so there is a unit
    vector ``w`` with ``w . (Z[i, :] - mean(Z)) = 0`` for every cell ``i``. For the local
    component ``w`` is ``1/sqrt(D)`` exactly, because it is LayerNorm output and every row sums
    to zero; for the global component ``w`` is some other direction, which is why this reads it
    off the SVD instead of assuming it.

    Then for **any** per-gene constant ``a_f``, with ``W' = W + a (x) w``::

        W'[f, :] . Z[i, :] = W[f, :] . Z[i, :] + a_f * (w . Z[i, :])
                           = W[f, :] . Z[i, :] + a_f * (w . mean(Z))

    -- the prediction for gene ``f`` moves by the same constant in every cell, which the decoder's
    own bias ``b_f`` absorbs exactly. So ``(W, b)`` and ``(W + a (x) w, b - a (w . mean Z))`` are
    **the same function of the data**, cell for cell.

    But the standardized loading ``S[f, e] = W[f, e] std(Z_e) / std(X_f)`` is not the same, and
    neither is the top-K ranking within a dimension. **A gene's loading vector is determined only
    up to this gauge**, and which representative gradient descent lands on is arbitrary -- it
    differs per seed for reasons that have nothing to do with the data.

    This is a property of the model, not of the selection rule: no statistic computed from ``W``
    alone can be reproducible across seeds here. It is the mechanism behind the instability the
    reviewers asked about, and it is why :func:`permutation_test_loadings` tests a quantity that
    does not involve ``W`` at all.

    ``a_f`` is drawn with ``scale`` times each gene's own row-wise sd of ``W`` -- at the default
    ``scale=1`` a perturbation exactly the size of the loadings being perturbed, so the reported
    Jaccard is on a like-for-like footing rather than from an arbitrarily large nudge. ``a_f`` is
    unconstrained in principle, so sweep ``scale`` rather than reading one number as "the" effect.

    Returns
    -------
    dict with the gauge direction's diagnostics, ``max_rel_pattern_change`` (the reconstruction
    change after the bias absorbs the per-gene constant -- the number that should be ~1e-7), and
    ``per_dim`` / ``median_jaccard_top_k`` (how much the top-K set moves, which is the point).

    """
    W = np.asarray(W, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    x_std = np.asarray(x_std, dtype=np.float64)
    dims = list(range(W.shape[1])) if dims is None else [int(d) for d in dims]
    rng = np.random.default_rng(random_state)

    Zc = _center(Z)
    _, sv, Vt = np.linalg.svd(Zc, full_matrices=False)
    w = Vt[-1]  # the null (or near-null) direction of the centred embedding
    proj = Zc @ w  # exactly 0 if the dependency is exact

    a = rng.normal(0.0, float(scale) * W.std(axis=1, ddof=1), size=W.shape[0])
    W2 = W + np.outer(a, w)

    idx = rng.choice(Z.shape[0], size=min(int(n_cells), Z.shape[0]), replace=False)
    P1 = Z[idx] @ W.T
    P2 = Z[idx] @ W2.T
    pred_scale = max(float(P1.std()), 1e-12)
    # The bias absorbs a per-gene constant, so what has to be unchanged is the PATTERN across
    # cells, i.e. the difference after removing each gene's own mean shift.
    resid = (P2 - P1) - (P2 - P1).mean(axis=0, keepdims=True)

    sz = Z.std(axis=0, ddof=1)
    S1 = W * sz[None, :] / x_std[:, None]
    S2 = W2 * sz[None, :] / x_std[:, None]

    def top(col, k):
        return set(np.argsort(-np.abs(col), kind="stable")[:k].tolist())

    rows = []
    for d in dims:
        A, Bs = top(S1[:, d], n_top), top(S2[:, d], n_top)
        rows.append(
            {
                "dim": int(d),
                "n_shared": len(A & Bs),
                "jaccard": len(A & Bs) / len(A | Bs) if A | Bs else np.nan,
            }
        )
    per_dim = pd.DataFrame(rows)
    return {
        "sv_ratio": float(sv[-1] / sv[0]) if sv[0] > 0 else 0.0,
        "gauge_direction_is_ones": bool(np.isclose(np.abs(w), 1 / np.sqrt(len(w)), rtol=1e-3).all()),
        "max_abs_projection_on_gauge": float(np.abs(proj).max()),
        "max_rel_prediction_change": float(np.abs(P2 - P1).max() / pred_scale),
        "max_rel_pattern_change": float(np.abs(resid).max() / pred_scale),
        "perturbation_sd_over_W_sd": float(a.std(ddof=1) / W.std(ddof=1)),
        "n_top": int(n_top),
        "scale": float(scale),
        "per_dim": per_dim,
        "median_jaccard_top_k": float(per_dim["jaccard"].median()),
        "median_shared_top_k": float(per_dim["n_shared"].median()),
    }


# ==========================================================================================
# step 1 -- permutation test on one trained model
# ==========================================================================================


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values), same order as the input.

    Implemented here rather than pulled from statsmodels so the module runs with numpy/scipy/
    pandas only; the notebook asserts it against
    ``statsmodels.stats.multitest.multipletests(method="fdr_bh")`` on real p-values.

    ``nan`` p-values are excluded from the family (they are not tests) and returned as ``nan``.
    """
    p = np.asarray(p, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv, kind="stable")
    ranked = pv[order] * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downwards, then clip at 1
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(ranked, 1.0)
    q[ok] = adj
    return q


def permutation_test_loadings(
    Z: np.ndarray,
    X: np.ndarray,
    *,
    dims: Sequence[int] | None = None,
    gene_names: Sequence[str] | None = None,
    B: int = 1000,
    random_state: int = 0,
    statistic: Literal["marginal", "partial"] = "marginal",
    fdr_scope: Literal["per_dimension", "global"] = "per_dimension",
    decoder_loadings: np.ndarray | None = None,
    perm_chunk: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """Permutation test of every (gene, dimension) association, for ONE trained model.

    **H0: gene f's expression is independent of latent dimension e across cells** -- the
    gene<->dimension association is arbitrary. The null is realised by permuting the cell order
    of ``Z[:, e]`` alone and rescoring against unpermuted ``X``, which preserves both marginal
    distributions exactly. It is *not* a test of the decoder weight ``W[e, f]``, which no cell
    permutation can move (see the module docstring).

    The statistic is a standardized loading with an *empirical* slope in place of the frozen
    decoder weight -- ``statistic="marginal"`` (the default,
    :func:`standardized_empirical_loading`) or ``"partial"``
    (:func:`standardized_partial_loading`, **not identifiable on these embeddings** -- see
    :func:`embedding_rank_report`). The p-value is two-sided on the magnitude::

        p = (1 + #{ |L_null| >= |L_obs| }) / (B + 1)

    the standard conservative estimator: it never returns 0, which ``B`` draws cannot support.
    The smallest attainable p is ``1/(B+1)``, so at the default ``B=1000`` many genes tie at
    ``p = 9.99e-4``; ``n_at_p_floor`` in :func:`significance_report` counts them.

    Vectorization: for one dimension, all ``B`` permutations are scored against all genes with a
    single ``(B, n_cells) @ (n_cells, n_genes)`` matmul. Every factor of the loading except the
    covariance is invariant to a permutation of ``Z[:, e]`` -- ``std(X[:, f])``, ``std(Z[:, e])``
    and ``var(r_e)`` all are -- so only the covariance term moves, and the p-value does not
    depend on the rest at all. No forward pass is involved.

    Under ``statistic="partial"`` the permuted quantity is ``r_e`` (dimension ``e`` residualized
    on the other dimensions of the component) and it is scored against ``X`` residualized on the
    same -- the ter Braak / Kennedy scheme. The observed covariance is identical either way
    (``r_e`` is already orthogonal to those dimensions), but residualizing ``X`` too stops the
    null from absorbing their signal once the pairing is broken. With strongly collinear
    dimensions this test is mildly liberal; ``"marginal"`` is exact but tests a different
    quantity.

    Parameters
    ----------
    Z
        ``(n_cells, n_dims)`` embedding of one component, as saved in ``adata.obsm``.
    X
        ``(n_cells, n_genes)`` expression on the layer the decoders reconstruct, same cells and
        same order as ``Z``. Dense; :func:`load_expression` produces it.
    dims
        Dimensions to test. ``None`` tests every column of ``Z``. Restricting to the
        cumulative-importance selection is what makes the CosMx run affordable.
    B
        Permutations per dimension.
    statistic
        ``"marginal"`` (default) tests ``corr(Z[:, e], X[:, f])`` -- the literal reading of
        "shuffle this dimension and rescore". It attributes a signal shared by several correlated
        dimensions to all of them, but it is **identifiable**, which on these embeddings the
        alternative is not.

        ``"partial"`` tests dimension ``e``'s coefficient with the other dimensions held fixed --
        the quantity a decoder weight *is*. On both InterScale datasets the embeddings are
        rank-deficient by exactly one (the local component is LayerNorm output: every row sums to
        0 and has norm ``sqrt(D)``), so *no* dimension has a unique contribution given the others,
        ``r_e`` is numerically zero and this statistic is noise. Measured on legnini23 seed 40 it
        returns **0 significant genes out of 88 on every dimension**. It is kept for embeddings
        that are full rank; :func:`embedding_rank_report` says whether yours is, and
        :func:`permutation_test_loadings` warns when it is not.
    random_state
        Each dimension gets its own child stream (``SeedSequence(random_state).spawn``), so a
        result is reproducible and does not depend on which dimensions were requested or in
        what order.
    fdr_scope
        ``"per_dimension"`` corrects across genes within each dimension -- each dimension is its
        own hypothesis family, which is the natural unit when the question is "which genes
        belong to *this* program". ``"global"`` corrects across all (gene, dim) pairs at once,
        which is stricter and appropriate if the dimensions themselves are being screened.
    decoder_loadings
        Optional ``(n_genes, n_dims_total)`` array of the decoder-based standardized loadings
        (what ``gene_loadings`` stores). Copied into the ``loading_decoder`` column for
        side-by-side reading; never used in the test.
    perm_chunk
        Permutations scored per matmul. Caps the temporary at
        ``perm_chunk * n_cells`` float32 (155 MB at the default on melton25).

    Returns
    -------
    DataFrame, one row per (gene, dim): ``gene``, ``dim``, ``loading`` (the observed empirical
    standardized loading), ``abs_loading``, ``loading_decoder`` (if given), ``n_ge`` (null draws
    at least as extreme), ``B``, ``statistic``, ``p``, ``q``, ``fdr_scope``.

    """
    if fdr_scope not in FDR_SCOPES:
        raise ValueError(f"fdr_scope must be one of {FDR_SCOPES}, got {fdr_scope!r}")
    if statistic not in STATISTICS:
        raise ValueError(f"statistic must be one of {STATISTICS}, got {statistic!r}")
    if int(B) < 1:
        raise ValueError(f"B must be >= 1, got {B}")

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z[:, None]
    X = np.asarray(X)
    if Z.shape[0] != X.shape[0]:
        raise ValueError(f"Z has {Z.shape[0]} cells, X has {X.shape[0]}")
    n, n_genes = X.shape
    dims = list(range(Z.shape[1])) if dims is None else [int(d) for d in dims]
    if not dims:
        raise ValueError(
            "dims is empty, so there is nothing to test. A caller that derived dims from a "
            "cross-seed dimension matching should skip the seed instead -- 'no dimension "
            "matched' is a result about the matching, not an empty test."
        )
    if any(d < 0 or d >= Z.shape[1] for d in dims):
        raise ValueError(f"dims out of range for Z with {Z.shape[1]} dimensions")
    if decoder_loadings is not None:
        S_chk = np.asarray(decoder_loadings)
        if S_chk.ndim != 2 or S_chk.shape[0] != n_genes:
            raise ValueError(f"decoder_loadings must be (n_genes, n_dims) with n_genes={n_genes}, got {S_chk.shape}")
    if gene_names is None:
        gene_names = [str(i) for i in range(n_genes)]
    gene_names = np.asarray([str(g) for g in gene_names])
    if len(gene_names) != n_genes:
        raise ValueError(f"{len(gene_names)} gene names for {n_genes} columns of X")

    # Centre and scale once. float32 for the matmul: the statistic is a correlation, so the
    # float32 error (~1e-6 relative) is orders below the null's own sd (1/sqrt(n) >= 1e-3 here),
    # and it halves both the memory and the BLAS time. The mean and sd are accumulated in
    # float64 -- summing 4e5 float32 values loses about three digits, which the sd would carry
    # straight into every p-value.
    mu = np.asarray(X, dtype=np.float32).mean(axis=0, dtype=np.float64)
    Xc = np.asarray(X, dtype=np.float32) - mu.astype(np.float32)
    sx = np.sqrt(np.einsum("ij,ij->j", Xc, Xc, dtype=np.float64) / (n - 1))
    sx_safe = np.where(sx > 0, sx, np.inf)  # a constant gene gets loading 0, p 1

    L_obs = (
        standardized_empirical_loading(Z, X, dims=dims)
        if statistic == "marginal"
        else standardized_partial_loading(Z, X, dims=dims)
    )  # (n_genes, n_dims)
    Zc_all = None
    if statistic == "partial":
        Zc_all = _center(np.asarray(Z, dtype=np.float64))
        rank = embedding_rank_report(Zc_all)
        if not rank["full_rank"]:
            warnings.warn(
                f"statistic='partial' on a rank-deficient embedding: rank {rank['rank']} of "
                f"{rank['n_dims']} dimensions (smallest/largest singular value "
                f"{rank['sv_ratio']:.1e}). No dimension has a contribution that is unique given "
                "the others, so the residualized dimension is numerical noise and every p-value "
                "below is meaningless. Use statistic='marginal'.",
                UserWarning,
                stacklevel=2,
            )

    seeds = np.random.SeedSequence(random_state).spawn(len(dims))
    rows = []
    for j, (d, ss) in enumerate(zip(dims, seeds, strict=True)):
        rng = np.random.default_rng(ss)
        obs = np.abs(L_obs[:, j])

        if statistic == "marginal":
            zc = np.asarray(Z[:, d], dtype=np.float64)
            zc = zc - zc.mean()
            target = Xc
        else:
            others = [k for k in range(Zc_all.shape[1]) if k != d]
            zc = _residual_on(Zc_all[:, [d]], Zc_all[:, others])[:, 0]
            target = _residual_on(Xc.astype(np.float64), Zc_all[:, others]).astype(np.float32)
        sz = zc.std(ddof=1)

        if not np.isfinite(sz) or sz == 0:
            # A constant dimension -- or, under "partial", one the others already explain
            # completely -- carries no measurable association, and no permutation changes that.
            n_ge = np.full(n_genes, B, dtype=np.int64)
            obs_on_null_scale = np.zeros(n_genes)
        else:
            zc32 = zc.astype(np.float32)
            # Any positive, permutation-invariant rescaling of |covariance| leaves the p-value
            # alone, so this only has to put null and observed on ONE common scale. The observed
            # value is recomputed through the identical expression rather than taken from L_obs,
            # which divides by var(r_e) for "partial" and by std(Z[:, e]) for "marginal" -- that
            # is what keeps the >= comparison meaningful for both statistics.
            scale = 1.0 / ((n - 1) * sz * sx_safe)
            obs_on_null_scale = np.abs((zc @ target.astype(np.float64)) * scale)
            n_ge = np.zeros(n_genes, dtype=np.int64)
            done = 0
            while done < B:
                b = min(int(perm_chunk), B - done)
                # Tile the centred vector and shuffle each ROW in place. Shuffling the floats
                # rather than an index array keeps the temporary at b x n float32 -- an int64
                # index array of the same shape is twice that, and on melton25 (n=386727) the
                # difference is 0.8 GB per chunk.
                P = np.tile(zc32, (b, 1))
                rng.permuted(P, axis=1, out=P)
                # Re-centre per row: a permutation preserves the mean exactly in exact
                # arithmetic, but not in float32 over 4e5 elements.
                P -= P.mean(axis=1, keepdims=True)
                null = (P @ target) * scale[None, :]  # (b, n_genes)
                # >=, with a relative tolerance: the observed value is computed in float64 and
                # the null in float32, so a null draw that IS the observed permutation (or a
                # tied one) must not be excluded by rounding.
                n_ge += (np.abs(null) >= obs_on_null_scale[None, :] * (1 - 1e-6) - 1e-12).sum(axis=0)
                done += b
        p = (1.0 + n_ge) / (B + 1.0)
        rows.append(
            pd.DataFrame(
                {
                    "gene": gene_names,
                    "dim": int(d),
                    "loading": L_obs[:, j],
                    "abs_loading": obs,
                    "n_ge": n_ge,
                    "B": int(B),
                    "statistic": statistic,
                    "p": p,
                }
            )
        )
        if decoder_loadings is not None:
            rows[-1]["loading_decoder"] = np.asarray(decoder_loadings, dtype=float)[:, int(d)]
        if verbose:
            print(f"    dim {d:3d}: {int((p <= 0.05).sum()):4d}/{n_genes} genes p<=0.05", flush=True)

    table = pd.concat(rows, ignore_index=True)

    if fdr_scope == "per_dimension":
        table["q"] = table.groupby("dim", sort=False)["p"].transform(lambda s: benjamini_hochberg(s.to_numpy()))
    else:
        table["q"] = benjamini_hochberg(table["p"].to_numpy())
    table["fdr_scope"] = fdr_scope
    return table


def demonstrate_null_is_degenerate(
    Z: np.ndarray, x_std: np.ndarray, W: np.ndarray, *, dim: int = 0, B: int = 20, random_state: int = 0
) -> pd.DataFrame:
    """Show that permuting cells leaves the DECODER standardized loading bit-identical.

    Evidence for the module docstring's central caveat, computed on the real arrays rather than
    argued: ``S[:, e] = W[e, :] * std(Z[:, e]) / std(X)`` is recomputed under ``B`` permutations
    of ``Z[:, e]``'s cell order and every value comes back unchanged, so the empirical p-value of
    the decoder loading is 1.0 for every gene. That is why :func:`permutation_test_loadings`
    tests an empirical loading instead.

    "Unchanged" is up to floating-point summation order: ``std`` visits the same values in a
    different order after a permutation, so ``max_rel_deviation`` sits at machine precision
    (~1e-16) rather than at identically zero. That is rounding, not signal -- it is some thirteen
    orders of magnitude below the ``1/sqrt(n)`` null sd of any statistic tested here.

    Returns a DataFrame with the absolute and relative deviation per permutation.
    """
    Z = np.asarray(Z, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    x_std = np.asarray(x_std, dtype=np.float64)
    z = Z[:, int(dim)]
    S_obs = W[:, int(dim)] * z.std(ddof=1) / x_std

    rng = np.random.default_rng(random_state)
    rows = []
    for b in range(B):
        zp = z[rng.permutation(z.size)]
        S_null = W[:, int(dim)] * zp.std(ddof=1) / x_std
        rows.append(
            {
                "permutation": b,
                "std_z_obs": float(z.std(ddof=1)),
                "std_z_permuted": float(zp.std(ddof=1)),
                "max_abs_deviation": float(np.max(np.abs(S_null - S_obs))),
                "max_rel_deviation": float(np.max(np.abs(S_null - S_obs) / np.maximum(np.abs(S_obs), 1e-300))),
                "n_genes_changed": int((S_null != S_obs).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["p_value_of_decoder_loading"] = 1.0
    out.attrs["dim"] = int(dim)
    return out


def significance_report(table: pd.DataFrame, *, alpha: float = 0.05, n_genes: int | None = None) -> pd.DataFrame:
    """Per-dimension summary of a :func:`permutation_test_loadings` table.

    ``frac_significant`` is the column to read first. On these datasets ``n`` is 4e4-4e5 cells,
    the permutation null of a correlation has sd ``~1/sqrt(n)``, and a large share of genes clear
    ``q < alpha`` on some dimension. ``n_at_p_floor`` counts genes pinned at ``1/(B+1)``, i.e.
    genes whose p-value is limited by ``B`` and not resolved by it; ``median_abs_loading_sig`` says
    how big the significant associations actually are.
    """
    rows = []
    for d, g in table.groupby("dim", sort=True):
        sig = g["q"] < alpha
        floor = np.isclose(g["p"], 1.0 / (g["B"].iloc[0] + 1.0))
        rows.append(
            {
                "dim": int(d),
                "n_genes": int(len(g)),
                "n_significant": int(sig.sum()),
                "frac_significant": float(sig.mean()),
                "n_at_p_floor": int(floor.sum()),
                "median_abs_loading_sig": float(g.loc[sig, "abs_loading"].median()) if sig.any() else np.nan,
                "max_abs_loading": float(g["abs_loading"].max()),
                "min_q": float(g["q"].min()),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["alpha"] = alpha
    if n_genes is not None:
        out.attrs["n_genes_total"] = int(n_genes)
    return out


# ==========================================================================================
# step 2 -- the five existing seed outputs
# ==========================================================================================


@dataclass
class SeedArrays:
    """One seed's frozen outputs, as ``4a_seed_stability_extract.py`` wrote them.

    No model is instantiated and nothing is retrained: the decoder weights ``W``, the
    ``gene_loadings`` standardized loadings ``S``, the embeddings ``Z``, the per-gene expression
    sd ``x_std`` and the cumulative-importance dimension selection ``dims`` all come straight off
    ``<models_dir>/analysis/seed<N>_arrays.npz``. ``program`` is the OLD top-K gene set recorded
    in the matching ``seed<N>_meta.json`` -- the "before" side of the comparison, so it is read
    rather than recomputed.
    """

    seed: int
    var_names: np.ndarray
    W: dict[str, np.ndarray]
    S: dict[str, np.ndarray]
    Z: dict[str, np.ndarray]
    dims: dict[str, np.ndarray]
    program: dict[str, list[str]]
    x_std: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_embed(self) -> int:
        """Latent width, read off the stored embedding rather than the config."""
        return int(self.Z["global"].shape[1])

    @property
    def n_obs(self) -> int:
        """Cells the model was run on -- already zero-expression filtered, so below the h5ad."""
        return int(self.Z["global"].shape[0])


def load_seed_arrays(arrays_dir: str | Path, seed: int, *, mmap: bool = False) -> SeedArrays:
    """Read one seed's ``seed<N>_arrays.npz`` + ``seed<N>_meta.json``. Read-only, no model.

    ``mmap`` is ignored for ``.npz`` (numpy cannot memory-map inside a zip); it is accepted so a
    caller can pass it uniformly and is documented as a no-op rather than silently pretending.
    """
    arrays_dir = Path(arrays_dir)
    npz_path = arrays_dir / f"seed{seed}_arrays.npz"
    meta_path = arrays_dir / f"seed{seed}_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"{npz_path} not found. Extract it first with:\n"
            f"  SPEC=<legnini23|melton25> sbatch scripts/legnini_23/run_seed_stability.sh"
        )
    z = np.load(npz_path, allow_pickle=True)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return SeedArrays(
        seed=int(seed),
        var_names=np.asarray([str(g) for g in z["var_names"]]),
        W={c: np.asarray(z[f"W_{c}"]) for c in COMPONENTS},
        S={c: np.asarray(z[f"S_{c}"], dtype=np.float64) for c in COMPONENTS},
        Z={c: np.asarray(z[f"Z_{c}"], dtype=np.float32) for c in COMPONENTS},
        dims={c: np.asarray(z[f"dims_{c}"], dtype=int) for c in COMPONENTS},
        program={c: [str(g) for g in meta.get(f"program_{c}", [])] for c in COMPONENTS},
        x_std=np.asarray(z["x_std"], dtype=np.float64),
        meta=meta,
    )


def _read_h5ad_matrix(group) -> np.ndarray | csr_matrix:
    enc = dict(group.attrs).get("encoding-type") if hasattr(group, "attrs") else None
    if enc in ("csr_matrix", "csc_matrix"):
        shape = tuple(int(v) for v in group.attrs["shape"])
        mat = csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape if enc == "csr_matrix" else shape[::-1],
        )
        return mat if enc == "csr_matrix" else mat.T.tocsr()
    return np.asarray(group[:])


def load_expression(
    h5ad_path: str | Path,
    *,
    layer: str | None = None,
    x_std_expected: np.ndarray | None = None,
    counts_key_for_filter: str | None = None,
    rtol: float = 1e-4,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``(n_cells, n_genes)`` expression on the layer the decoders reconstruct.

    Read with h5py rather than ``anndata.read_h5ad`` so only the one matrix is touched -- the
    melton25 file also holds a raw ``counts`` layer of the same size.

    **The cell filter is verified, not assumed.** Training dropped zero-expression cells
    (``interscale.tl.remove_zero_expression_cells``), so the saved ``Z`` has fewer rows than the
    h5ad. Rather than reimplement that filter and hope it matches, this tries the candidate row
    subsets and keeps the one whose per-gene sd reproduces the ``x_std`` stored alongside ``Z``.
    If none matches, it raises: a silently misaligned X would make every p-value below wrong in a
    way no downstream check would catch.

    Parameters
    ----------
    layer
        ``adata.layers`` key, or ``None`` for ``.X``. legnini23 reconstructs
        ``layers["log1p_norm"]``; melton25 reconstructs ``.X`` (its ``cfg.dataset.layer_key`` is
        ``None``) -- so this must never be defaulted per dataset.
    x_std_expected
        ``x_std`` from :class:`SeedArrays`. The check that the filter matched.
    counts_key_for_filter
        Layer holding the counts the zero-expression filter was applied to. ``None`` uses the
        same matrix as ``layer``.

    Returns
    -------
    ``(X, keep)`` -- the dense matrix and the boolean row mask that produced it.

    """
    import h5py

    h5ad_path = Path(h5ad_path)
    with h5py.File(h5ad_path, "r") as f:
        target = f["X"] if layer is None else f["layers"][layer]
        M = _read_h5ad_matrix(target)
        if counts_key_for_filter is None:
            C = M
        else:
            C = _read_h5ad_matrix(f["X"] if counts_key_for_filter == "X" else f["layers"][counts_key_for_filter])

    n = M.shape[0]
    row_sum = np.asarray(C.sum(axis=1)).ravel() if issparse(C) else np.asarray(C).sum(axis=1)
    candidates = {
        "all cells": np.ones(n, dtype=bool),
        "drop zero-expression cells": row_sum != 0,
    }

    def dense(mask):
        sub = M[mask] if not issparse(M) else M[mask]
        return np.asarray(sub.todense() if issparse(sub) else sub, dtype=np.float32)

    if x_std_expected is None:
        mask = candidates["drop zero-expression cells"]
        if verbose:
            print(f"  no x_std to check against -- using '{mask.sum()} of {n} cells' unverified")
        return dense(mask), mask

    x_std_expected = np.asarray(x_std_expected, dtype=np.float64)
    tried = {}
    for name, mask in candidates.items():
        X = dense(mask)
        sd = X.astype(np.float64).std(axis=0, ddof=1)
        rel = float(np.max(np.abs(sd - x_std_expected) / np.maximum(x_std_expected, 1e-12)))
        tried[name] = rel
        if rel <= rtol:
            if verbose:
                print(
                    f"  X: {name} -> {mask.sum()} of {n} cells, "
                    f"per-gene sd matches the stored x_std (max rel. dev. {rel:.2e})"
                )
            return X, mask
        del X

    raise ValueError(
        f"No candidate cell filter reproduces the stored x_std for {h5ad_path.name} "
        f"(layer={layer!r}). Max relative deviation per candidate: "
        + ", ".join(f"{k}: {v:.3g}" for k, v in tried.items())
        + ".\nThe expression matrix would not be row-aligned with the saved embedding, so every "
        "permutation p-value would be meaningless. Check `layer` and the training-time filter."
    )


def significant_sets(
    table: pd.DataFrame, *, alpha: float = 0.05, dims: Sequence[int] | None = None
) -> dict[int, set[str]]:
    """``{dim: {gene, ...}}`` for genes with ``q < alpha`` -- the replacement for the top-K rule.

    Tests the null of :func:`permutation_test_loadings`: a gene is kept for dimension ``e`` when
    its association with ``e`` is stronger than the shuffled-pairing null survives at FDR
    ``alpha``, whatever its rank within the dimension. No K.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    sub = table if dims is None else table[table["dim"].isin([int(d) for d in dims])]
    sig = sub[sub["q"] < alpha]
    out = {int(d): set(g["gene"]) for d, g in sig.groupby("dim", sort=True)}
    for d in sorted(sub["dim"].unique()):
        out.setdefault(int(d), set())
    return out


def size_matched_sets(
    table: pd.DataFrame,
    target_sizes: Mapping[int, int],
    *,
    alpha: float = 0.05,
    dims: Sequence[int] | None = None,
) -> dict[int, set[str]]:
    """The significant genes, cut back to a given size per dimension by ``|loading|``.

    Needed because ``q < alpha`` and top-K do not select comparable numbers of genes: on
    legnini23 the test calls ~85% of an 88-gene panel significant on every dimension, so a
    before/after comparison of set *reproducibility* would be confounded by set *size* -- two
    sets covering 85% of a panel must overlap heavily whatever the statistic. This keeps the
    significance filter but takes the top ``target_sizes[dim]`` of the survivors by
    ``|abs_loading|``, so the only thing that differs from the top-K rule is **which statistic
    does the ranking**.

    Genes that fail ``q < alpha`` are never included, even if a target size is not reached; a
    smaller set is the honest outcome there.
    """
    sig = significant_sets(table, alpha=alpha, dims=dims)
    sub = table if dims is None else table[table["dim"].isin([int(d) for d in dims])]
    out = {}
    for d, genes in sig.items():
        k = int(target_sizes.get(d, len(genes)))
        g = sub[(sub["dim"] == d) & (sub["gene"].isin(genes))]
        out[int(d)] = set(g.nlargest(k, "abs_loading")["gene"]) if k > 0 else set()
    return out


def select_genes_permutation(
    Z: np.ndarray,
    X: np.ndarray,
    *,
    dims: Sequence[int],
    gene_names: Sequence[str],
    B: int = 1000,
    alpha: float = 0.05,
    statistic: Literal["marginal", "partial"] = "marginal",
    fdr_scope: Literal["per_dimension", "global"] = "per_dimension",
    candidate_genes: Sequence[str] | None = None,
    max_per_dim: int | None = None,
    random_state: int = 0,
    verbose: bool = False,
) -> dict:
    """Gene selection from **one** trained model. No replicates required.

    This is the shipped, user-facing criterion: an end user of InterScale has a single trained
    model, so the selection rule has to work from one. Replicate seeds are the *yardstick* the
    rule is validated against (:func:`agreement_with_other_runs`), never an input to it.

    **H0: gene f's expression across cells is independent of latent dimension e's value across
    cells.** Realised by permuting the cell order of ``Z[:, e]`` alone; see
    :func:`permutation_test_loadings`, which this calls and which no forward pass is needed for.

    Parameters
    ----------
    dims
        The dimensions to select over -- normally this model's own
        ``calculate_dim_importance`` selection.
    candidate_genes
        Restrict to these before testing, e.g. the genes passing the ``min_frac`` filter the
        top-K path uses. FDR is then computed over the restricted family, which is the correct
        family: a gene that was never a candidate is not a test.
    max_per_dim
        Optional cap on genes kept per dimension, by ``|loading|`` among those with
        ``q < alpha``. **Recommended on these datasets.** With 4e4-4e5 cells the permutation
        null of a correlation has sd ``~1/sqrt(n)``, so ``q < alpha`` alone accepts 85-95% of a
        panel -- significance is real but not selective, and an unbounded set is not a program.
        ``None`` keeps every significant gene.

    Returns
    -------
    dict with ``genes`` (the union over dims -- the selection), ``per_dim`` (``{dim: {genes}}``),
    ``table`` (the full per-(gene, dim) test) and ``settings``.

    """
    gene_names = np.asarray([str(g) for g in gene_names])
    X = np.asarray(X)
    if candidate_genes is not None:
        keep = np.isin(gene_names, np.asarray([str(g) for g in candidate_genes]))
        if not keep.any():
            raise ValueError("candidate_genes excludes every gene")
        X, gene_names = X[:, keep], gene_names[keep]

    table = permutation_test_loadings(
        Z,
        X,
        dims=list(dims),
        gene_names=gene_names,
        B=B,
        random_state=random_state,
        statistic=statistic,
        fdr_scope=fdr_scope,
        verbose=verbose,
    )
    per_dim = significant_sets(table, alpha=alpha, dims=list(dims))
    if max_per_dim is not None:
        per_dim = size_matched_sets(table, dict.fromkeys(per_dim, int(max_per_dim)), alpha=alpha, dims=list(dims))
    genes = sorted(set().union(*per_dim.values())) if per_dim else []
    return {
        "genes": genes,
        "per_dim": per_dim,
        "table": table,
        "settings": {
            "B": B,
            "alpha": alpha,
            "statistic": statistic,
            "fdr_scope": fdr_scope,
            "max_per_dim": max_per_dim,
            "random_state": random_state,
            "n_candidates": int(len(gene_names)),
            "dims": [int(d) for d in dims],
        },
    }


def agreement_with_other_runs(
    selected: Sequence[str],
    other_sets: Mapping[int, Sequence[str]],
    *,
    pool: int,
    published: Sequence[str] | None = None,
    n_iter: int = 20_000,
    seed: int = 0,
) -> dict:
    """Do genes selected from ONE model also appear in other runs' top-N lists?

    The evaluation the single-model criterion is judged by, and the one that needs the replicate
    seeds -- which is why it is a separate function from the selection. ``selected`` comes from
    one model (:func:`select_genes_permutation`); ``other_sets`` are the *other* runs'
    independently produced selections, e.g. the ``program_local`` / ``program_global`` top-N
    lists ``4a_seed_stability_extract.py`` recorded per seed.

    No dimension matching is involved. Both sides are component-level gene sets, so the
    comparison never has to align latent dimensions across runs -- which matters because those
    dimensions largely do not correspond (81-94% of legnini23's global dimensions have no
    counterpart above ``|r| = 0.3`` between seed pairs).

    The number to read is ``mean_recall`` against ``chance_recall``. A gene drawn at random from
    the ``pool`` lands in another run's list of size ``m`` with probability ``m / pool``, so
    ``chance_recall`` is ``mean(m_j) / pool`` -- **not zero**. Overlap above zero is not evidence;
    overlap above chance is.

    Returns
    -------
    dict with ``per_gene`` (a DataFrame: gene, n_runs, frac_runs, runs, in_published) and the
    summary keys ``n_selected``, ``mean_recall``, ``chance_recall``, ``fold``,
    ``n_in_all_runs``, ``n_in_no_run``, plus ``pairwise`` (a DataFrame of the observed vs
    expected overlap and its p-value against each run, via
    :func:`src.utils.intersection_tail_probability`).

    """
    from src.utils import intersection_tail_probability

    sel = sorted({str(g) for g in selected})
    others = {int(k): {str(g) for g in v} for k, v in other_sets.items()}
    pool = int(pool)
    if pool <= 0:
        raise ValueError(f"pool must be positive, got {pool}")
    truth = {str(g) for g in published} if published is not None else set()

    rows = []
    for g in sel:
        hits = sorted(k for k, v in others.items() if g in v)
        rows.append(
            {
                "gene": g,
                "n_runs": len(hits),
                "n_runs_total": len(others),
                "frac_runs": len(hits) / len(others) if others else np.nan,
                "runs": ",".join(str(k) for k in hits),
                "in_published": g in truth,
            }
        )
    per_gene = pd.DataFrame(
        rows, columns=["gene", "n_runs", "n_runs_total", "frac_runs", "runs", "in_published"]
    ).sort_values(["n_runs", "gene"], ascending=[False, True], ignore_index=True)

    sizes = np.array([len(v) for v in others.values()], dtype=float)
    chance = float(sizes.mean() / pool) if len(sizes) else np.nan

    pw = []
    for k, v in sorted(others.items()):
        if not sel or not v or max(len(sel), len(v)) > pool:
            pw.append(
                {
                    "run": k,
                    "n_other": len(v),
                    "observed": len(set(sel) & v),
                    "expected": np.nan,
                    "fold": np.nan,
                    "p_value": np.nan,
                }
            )
            continue
        r = intersection_tail_probability(
            [len(sel), len(v)], pool, len(set(sel) & v), alternative="greater", n_iter=n_iter, seed=seed
        )
        pw.append(
            {
                "run": k,
                "n_other": len(v),
                "observed": r["observed"],
                "expected": r["expected"],
                "fold": r["fold_change"],
                "p_value": r["p_value"],
            }
        )
    pairwise = pd.DataFrame(pw)

    return {
        "per_gene": per_gene,
        "pairwise": pairwise,
        "n_selected": len(sel),
        "n_runs": len(others),
        "other_set_sizes": [int(x) for x in sizes],
        "pool": pool,
        "mean_recall": float(per_gene["frac_runs"].mean()) if len(per_gene) else np.nan,
        "chance_recall": chance,
        "fold": (float(per_gene["frac_runs"].mean() / chance) if len(per_gene) and chance and chance > 0 else np.nan),
        "n_in_all_runs": int((per_gene["n_runs"] == len(others)).sum()) if len(per_gene) else 0,
        "n_in_no_run": int((per_gene["n_runs"] == 0).sum()) if len(per_gene) else 0,
        "n_published_selected": int(per_gene["in_published"].sum()) if len(per_gene) else 0,
        "n_published_total": len(truth),
    }


def match_dimensions(Z_a: np.ndarray, Z_b: np.ndarray, *, min_r: float = 0.3) -> pd.DataFrame:
    """Align two runs' latent dimensions -- **the repo's existing matching, not a new one**.

    Delegates to ``scripts/legnini_23/4b_seed_stability_compare.match_dims``, which is what the
    published cross-seed comparison uses: Hungarian assignment on ``|corr(z_i^a, z_j^b)|`` over
    the embedding columns, with matches below ``min_r`` flagged ``kept=False`` rather than
    dropped silently. ``min_r=0.3`` is that script's default.

    Matching on the *embeddings* and not on the loading vectors is deliberate and is the reason
    to reuse it: a standardized loading carries a ``1 / std(x_gene)`` factor that is a property
    of the data and therefore byte-identical across seeds, so loading-vector similarity is
    inflated by an artefact shared by every pair of dimensions (see
    ``shuffled_pairing_rows``' docstring in that script). Embedding correlation has no such term.

    Returns a DataFrame with ``dim_a``, ``dim_b``, ``r``, ``abs_r``, ``sign``, ``kept``.
    """
    match_dims = _import_match_dims()
    recs = match_dims(np.asarray(Z_a), np.asarray(Z_b), float(min_r))
    return pd.DataFrame(recs)


def _import_match_dims():
    """Import ``match_dims`` from the existing compare script without executing its main().

    The script is ``scripts/legnini_23/4b_seed_stability_compare.py`` -- a hyphen-free but
    digit-leading module name, so it cannot be imported with a plain ``import`` statement.
    """
    import importlib.util

    here = Path(__file__).resolve().parent.parent  # repo root
    path = here / "scripts" / "legnini_23" / "4b_seed_stability_compare.py"
    if not path.exists():
        raise FileNotFoundError(f"existing matching implementation not found: {path}")
    spec = importlib.util.spec_from_file_location("_is_4b_compare", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.match_dims


def match_all_to_reference(
    seeds: Mapping[int, SeedArrays],
    component: str,
    *,
    reference: int | None = None,
    min_r: float = 0.3,
) -> tuple[dict[int, dict[int, int]], pd.DataFrame]:
    """Map every seed's dimensions onto one reference seed's, via :func:`match_dimensions`.

    Latent dimensions are not aligned across independently initialised runs -- the five legnini23
    seeds' ``dims_local`` selections share almost no index -- so gene sets are only comparable
    per program after matching.

    Returns ``(mapping, quality)`` where ``mapping[seed][ref_dim] = that seed's dim`` for kept
    matches only (a reference dimension with no acceptable match in some seed is simply absent
    for that seed, which :func:`consensus_frequency` records as "not testable" rather than as
    "not significant"), and ``quality`` is one row per (seed, ref_dim) with ``abs_r`` and
    ``kept``.
    """
    if component not in COMPONENTS:
        raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
    order = sorted(seeds)
    ref = order[0] if reference is None else int(reference)
    if ref not in seeds:
        raise KeyError(f"reference seed {ref} not among {order}")

    Z_ref = seeds[ref].Z[component]
    mapping: dict[int, dict[int, int]] = {}
    rows = []
    for s in order:
        if s == ref:
            n_dims = Z_ref.shape[1]
            mapping[s] = {d: d for d in range(n_dims)}
            rows += [
                {
                    "seed": s,
                    "ref_dim": d,
                    "dim": d,
                    "r": 1.0,
                    "abs_r": 1.0,
                    "sign": 1.0,
                    "kept": True,
                    "is_reference": True,
                }
                for d in range(n_dims)
            ]
            continue
        m = match_dimensions(Z_ref, seeds[s].Z[component], min_r=min_r)
        mapping[s] = {int(r.dim_a): int(r.dim_b) for r in m.itertuples() if r.kept}
        for r in m.itertuples():
            rows.append(
                {
                    "seed": s,
                    "ref_dim": int(r.dim_a),
                    "dim": int(r.dim_b),
                    "r": float(r.r),
                    "abs_r": float(r.abs_r),
                    "sign": float(r.sign),
                    "kept": bool(r.kept),
                    "is_reference": False,
                }
            )
    quality = pd.DataFrame(rows)
    quality.attrs.update({"reference": ref, "min_r": float(min_r), "component": component})
    return mapping, quality


def matching_report(quality: pd.DataFrame, *, min_r: float | None = None) -> pd.DataFrame:
    """Per-seed matching quality, with an explicit flag when the dimensions do not align.

    ``flag`` is set when over half of a seed's matches fall below ``min_r``. A flagged seed's
    programs are not comparable to the reference's dimension by dimension, and the honest
    response is to say so -- the consensus below then rests on fewer than five seeds for the
    affected programs, which :func:`consensus_frequency` reports in ``n_testable``.
    """
    min_r = float(quality.attrs.get("min_r", 0.3)) if min_r is None else float(min_r)
    rows = []
    for s, g in quality[~quality["is_reference"]].groupby("seed", sort=True):
        kept = g["kept"]
        rows.append(
            {
                "seed": int(s),
                "n_dims": int(len(g)),
                "n_kept": int(kept.sum()),
                "discard_rate": float(1.0 - kept.mean()),
                "median_abs_r": float(g["abs_r"].median()),
                "mean_abs_r_kept": float(g.loc[kept, "abs_r"].mean()) if kept.any() else np.nan,
                "min_abs_r_kept": float(g.loc[kept, "abs_r"].min()) if kept.any() else np.nan,
                "flag_poor_matching": bool(kept.mean() < 0.5),
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update(dict(quality.attrs))
    out.attrs["min_r"] = min_r
    return out


# ==========================================================================================
# step 3 -- consensus across seeds
# ==========================================================================================


def consensus_frequency(
    tables: Mapping[int, pd.DataFrame],
    mapping: Mapping[int, Mapping[int, int]],
    *,
    alpha: float = 0.05,
    ref_dims: Sequence[int] | None = None,
) -> pd.DataFrame:
    """How many seeds call each gene significant for each **matched** reference dimension.

    One row per (gene, ref_dim): ``n_seeds`` seeds in which the gene is significant for the
    dimension matched to ``ref_dim``, out of ``n_testable`` seeds in which that reference
    dimension had an acceptable match at all. Reporting both is the point: 2/3 and 2/5 are
    different claims, and collapsing them would credit a program for seeds whose dimensions
    never aligned.

    ``frequency`` is ``n_seeds / n_testable``; the ``>= k`` rule in :func:`consensus_genes` uses
    the count ``n_seeds``, which is what the task defines.
    """
    seeds = sorted(tables)
    sig = {s: significant_sets(tables[s], alpha=alpha) for s in seeds}

    if ref_dims is None:
        ref_dims = sorted({int(d) for s in seeds for d in mapping.get(s, {}) if mapping[s][d] in sig[s]})
    ref_dims = [int(d) for d in ref_dims]

    rows = []
    for rd in ref_dims:
        testable = [s for s in seeds if rd in mapping.get(s, {}) and mapping[s][rd] in sig[s]]
        if not testable:
            continue
        genes: set[str] = set()
        for s in testable:
            genes |= sig[s][mapping[s][rd]]
        for g in sorted(genes):
            hits = [s for s in testable if g in sig[s][mapping[s][rd]]]
            rows.append(
                {
                    "gene": g,
                    "ref_dim": rd,
                    "n_seeds": len(hits),
                    "n_testable": len(testable),
                    "frequency": len(hits) / len(testable),
                    "seeds": ",".join(str(s) for s in hits),
                }
            )
    out = pd.DataFrame(rows, columns=["gene", "ref_dim", "n_seeds", "n_testable", "frequency", "seeds"])
    out.attrs.update({"alpha": float(alpha), "n_seeds_total": len(seeds), "seeds": seeds})
    return out


def consensus_genes(
    freq: pd.DataFrame, *, consensus_k: int = 3, per_dimension: bool = False
) -> list[str] | dict[int, list[str]]:
    """Genes significant for the same matched dimension in at least ``consensus_k`` seeds.

    ``per_dimension=False`` (default) returns the component-level consensus set -- the union over
    reference dimensions, which is the object the old ``program_local`` / ``program_global`` lists
    are and therefore the one that goes into the before/after comparison and into any downstream
    gene-set use. ``per_dimension=True`` returns ``{ref_dim: [genes]}``, the gene->program
    assignment the reviewers asked about.

    ``consensus_k`` counts seeds, not fraction: with five seeds the default 3 is a strict
    majority. A gene reaching ``k`` on *any* dimension enters the component-level set.
    """
    k = int(consensus_k)
    if k < 1:
        raise ValueError(f"consensus_k must be >= 1, got {consensus_k}")
    if freq.empty:
        return {} if per_dimension else []
    hit = freq[freq["n_seeds"] >= k]
    if per_dimension:
        return {int(d): sorted(g["gene"]) for d, g in hit.groupby("ref_dim", sort=True)}
    return sorted(set(hit["gene"]))


# ==========================================================================================
# step 5 -- the API, with the existing entry point left untouched
# ==========================================================================================


def get_genes_dim_consensus(
    adata=None,
    which: str = "global",
    *,
    consensus: bool = False,
    seed_inputs: Mapping[int, SeedArrays] | None = None,
    expression: np.ndarray | None = None,
    dims: Sequence[int] | None = None,
    B: int = 1000,
    alpha: float = 0.05,
    statistic: Literal["marginal", "partial"] = "marginal",
    fdr_scope: Literal["per_dimension", "global"] = "per_dimension",
    consensus_k: int | None = None,
    random_state: int = 0,
    min_r: float = 0.3,
    reference: int | None = None,
    return_details: bool = False,
    verbose: bool = True,
    **get_genes_dim_kwargs: Any,
):
    """Gene selection per latent dimension, by top-K (default) or by permutation consensus.

    ``consensus=False`` is the existing behaviour, delegated **verbatim** to
    ``interscale.evaluation.get_genes_dim``: this wrapper forwards ``adata``, ``which`` and every
    extra keyword and returns exactly what that function returns. Nothing about the existing
    top-K path is reimplemented here, and the existing function is not modified -- so existing
    call sites are unaffected whether they call ``get_genes_dim`` directly or this.

    ``consensus=True`` replaces the top-K cutoff with the permutation test of
    :func:`permutation_test_loadings` (**H0: a gene's expression is independent of a latent
    dimension across cells**; the null permutes the cell order of that dimension only) followed
    by the ``>= consensus_k`` of ``N`` seeds frequency rule on dimension-matched programs.

    * ``len(seed_inputs) == 1`` -- **the shipped path**: single-model ``q < alpha``. An end user
      of InterScale has one trained model, so this is the case the criterion has to work in. It
      is not a degraded mode and does not warn. Whether its genes recur across runs is measured
      separately, by :func:`agreement_with_other_runs`, using replicates that only the *authors*
      have at evaluation time.
    * ``len(seed_inputs) > 1`` -- additionally applies the ``>= consensus_k``-of-N frequency rule
      over dimension-matched programs. Use it when replicates genuinely exist. Note it needs the
      latent dimensions to correspond across runs, which on the global component they largely do
      not (see :func:`matching_report`); the single-model path has no such requirement.

    Parameters
    ----------
    adata, which, **get_genes_dim_kwargs
        Passed straight through when ``consensus=False``. ``which`` also selects the component
        when ``consensus=True``.
    seed_inputs
        ``{seed: SeedArrays}`` from :func:`load_seed_arrays`. Required when ``consensus=True``.
        No model is loaded and nothing is retrained.
    expression
        Dense ``(n_cells, n_genes)`` on the reconstructed layer, row-aligned with the saved
        embeddings -- :func:`load_expression` verifies that alignment against ``x_std``.
        Required when ``consensus=True``.
    dims
        Reference-seed dimensions to test. ``None`` uses the reference seed's own
        cumulative-importance selection (``SeedArrays.dims[which]``), i.e. the same dimension
        set the top-K path uses.
    B, alpha, statistic, fdr_scope, random_state
        The permutation settings; see :func:`permutation_test_loadings`.
    consensus_k
        Seeds a gene must be significant in. ``None`` (default) resolves to 3 with several
        models and to 1 with one, so the same call works either way. Passing ``> 1`` with a
        single model is a contradiction and warns.
    min_r, reference
        Dimension-matching settings; see :func:`match_all_to_reference`.
    return_details
        Also return the intermediate tables.

    Returns
    -------
    ``consensus=False``
        Whatever ``interscale.evaluation.get_genes_dim`` returns.
    ``consensus=True``
        A DataFrame in ``get_genes_dim``'s shape -- genes down the index, reference dimensions
        across the columns (as strings), signed empirical standardized loadings from the
        reference seed as values, restricted to the consensus genes -- so a downstream consumer
        (GO enrichment, ``plot_membership``, a heatmap) takes it unchanged. ``df.attrs`` carries
        ``consensus_genes``, ``frequency`` (the full table), ``per_seed_sets``, ``matching`` and
        the settings. With ``return_details=True``, ``(df, details)``.

    """
    if not consensus:
        from interscale.evaluation import get_genes_dim

        # `dims` is a parameter of THIS function too, so it has to be forwarded explicitly or a
        # caller passing dims=[...] would silently reach get_genes_dim with dims=None -- which
        # then looks for a stored dimension selection and raises.
        kwargs = dict(get_genes_dim_kwargs)
        if dims is not None:
            kwargs["dims"] = list(dims)
        return get_genes_dim(adata, which, **kwargs)

    if which not in COMPONENTS:
        raise ValueError(f"which must be one of {COMPONENTS}, got {which!r}")
    if not seed_inputs:
        raise ValueError(
            "consensus=True needs seed_inputs={seed: SeedArrays}; build them with "
            "load_seed_arrays(arrays_dir, seed). Nothing is retrained -- they are read off disk."
        )
    if expression is None:
        raise ValueError(
            "consensus=True needs expression=(n_cells, n_genes) on the reconstructed layer, "
            "row-aligned with the saved embeddings. Use load_expression(), which verifies that "
            "alignment against the stored x_std."
        )
    if get_genes_dim_kwargs and verbose:
        warnings.warn(
            f"consensus=True ignores the top-K arguments {sorted(get_genes_dim_kwargs)}: the "
            "cutoff is q < alpha, not a rank. Drop them, or pass consensus=False to use them.",
            UserWarning,
            stacklevel=2,
        )

    seeds = {int(s): v for s, v in seed_inputs.items()}
    order = sorted(seeds)
    ref = order[0] if reference is None else int(reference)
    single = len(order) == 1
    if consensus_k is None:
        consensus_k = 1 if single else 3
    elif single and consensus_k > 1:
        warnings.warn(
            f"consensus_k={consensus_k} was requested but only one model was supplied, so no "
            "frequency rule can be applied; using single-model q < alpha. That is the intended "
            "production path -- pass consensus_k=None (or 1) to say so explicitly, and measure "
            "cross-run recurrence with agreement_with_other_runs() instead.",
            UserWarning,
            stacklevel=2,
        )
        consensus_k = 1

    X = np.asarray(expression)
    ref_dims = list(seeds[ref].dims[which]) if dims is None else [int(d) for d in dims]

    # Match FIRST, then test only the dimensions the matching actually needs. Testing every
    # dimension of every seed would be the same analysis at three times the compute: on melton25
    # that is 256 dimensions per seed at ~2.4 s each against 91. Matching reads the embeddings
    # only, so it costs nothing to do first.
    if single:
        mapping, quality = {ref: {d: d for d in ref_dims}}, pd.DataFrame()
    else:
        mapping, quality = match_all_to_reference(seeds, which, reference=ref, min_r=min_r)

    tables = {}
    for s in order:
        sa = seeds[s]
        if X.shape != (sa.Z[which].shape[0], len(sa.var_names)):
            raise ValueError(
                f"expression is {X.shape} but seed {s} has {sa.Z[which].shape[0]} cells and "
                f"{len(sa.var_names)} genes. The matrix must be row-aligned with the embeddings."
            )
        test_dims = sorted({mapping[s][d] for d in ref_dims if d in mapping.get(s, {})})
        if not test_dims:
            warnings.warn(
                f"seed {s}: none of the reference's {len(ref_dims)} dimensions matched above "
                f"min_r={min_r}, so this seed contributes nothing to the consensus.",
                UserWarning,
                stacklevel=2,
            )
            tables[s] = pd.DataFrame(
                columns=["gene", "dim", "loading", "abs_loading", "n_ge", "B", "statistic", "p", "q"]
            )
            continue
        if verbose:
            print(
                f"  seed {s}: permutation test on {len(test_dims)} {which} dims, B={B}, statistic={statistic}",
                flush=True,
            )
        tables[s] = permutation_test_loadings(
            sa.Z[which],
            X,
            dims=test_dims,
            gene_names=sa.var_names,
            B=B,
            random_state=random_state + s,
            statistic=statistic,
            fdr_scope=fdr_scope,
            decoder_loadings=sa.S[which],
        )

    sets = significant_sets(tables[ref], alpha=alpha, dims=ref_dims)
    if single:
        freq = pd.DataFrame(
            [
                {"gene": g, "ref_dim": d, "n_seeds": 1, "n_testable": 1, "frequency": 1.0, "seeds": str(ref)}
                for d, gs in sets.items()
                for g in sorted(gs)
            ],
            columns=["gene", "ref_dim", "n_seeds", "n_testable", "frequency", "seeds"],
        )
        genes = sorted(set(freq["gene"]))
    else:
        freq = consensus_frequency(tables, mapping, alpha=alpha, ref_dims=ref_dims)
        genes = consensus_genes(freq, consensus_k=consensus_k)

    # get_genes_dim's shape: genes x dims, string column names, signed scores.
    L_ref = (
        standardized_empirical_loading(seeds[ref].Z[which], X, dims=ref_dims)
        if statistic == "marginal"
        else standardized_partial_loading(seeds[ref].Z[which], X, dims=ref_dims)
    )
    L_ref = pd.DataFrame(L_ref, index=seeds[ref].var_names, columns=[str(d) for d in ref_dims])
    df = L_ref.loc[[g for g in L_ref.index if g in set(genes)]]
    # winner-block ordering, as get_genes_dim(order_genes_by="winner") produces
    if len(df):
        A = df.abs().to_numpy()
        winner = A.argmax(axis=1)
        df = df.iloc[np.lexsort((-A[np.arange(len(df)), winner], winner))]

    df.attrs.update(
        {
            "consensus": True,
            "component": which,
            "consensus_genes": list(df.index),
            "frequency": freq,
            "per_seed_sets": {s: significant_sets(t, alpha=alpha) for s, t in tables.items()},
            "matching": quality,
            "reference_seed": ref,
            "ref_dims": ref_dims,
            "settings": {
                "B": B,
                "alpha": alpha,
                "statistic": statistic,
                "fdr_scope": fdr_scope,
                "consensus_k": consensus_k,
                "random_state": random_state,
                "min_r": min_r,
                "n_seeds": len(order),
                "single_model_fallback": single,
            },
        }
    )
    if verbose:
        print(f"  {which}: {len(df)} consensus genes (>= {consensus_k} of {len(order)} seeds, alpha={alpha}, B={B})")
    if return_details:
        return df, {
            "tables": tables,
            "frequency": freq,
            "mapping": mapping,
            "matching": quality,
            "reference_sets": sets,
        }
    return df


# ==========================================================================================
# step 4 -- feeding the existing before/after plots
# ==========================================================================================


def sets_to_frames(sets_per_seed: Mapping[int, Sequence[str]], seeds: Sequence[int] | None = None):
    """``{seed: [genes]}`` -> the list of gene-indexed DataFrames the existing plots consume.

    ``src.decoder_stability.plot_membership`` / ``intersection_by_k`` / ``tail_prob_by_k`` all
    read only ``df.index``, so a one-column frame is enough and no loading values are implied.
    """
    seeds = sorted(sets_per_seed) if seeds is None else list(seeds)
    return [pd.DataFrame(index=pd.Index(sorted(set(map(str, sets_per_seed[s]))), name="gene")) for s in seeds]


def recovery_numbers(
    true_genes: Sequence[str], sets_per_seed: Mapping[int, Sequence[str]], *, label: str = ""
) -> dict:
    """The annotations already on the existing stability figure, as numbers.

    ``plot_membership``'s title reports "N in all K seeds" and "M/T true genes never recovered";
    this computes the same quantities without drawing, so the before/after comparison can be
    printed and tabulated. ``recovered_all_seeds`` and ``never_recovered`` are the two the task
    asks to report.
    """
    truth = set(map(str, true_genes))
    sets = {s: set(map(str, g)) for s, g in sets_per_seed.items()}
    n_seeds = len(sets)
    counts = {g: sum(g in s for s in sets.values()) for g in truth}
    union = set().union(*sets.values()) if sets else set()
    inter = set.intersection(*sets.values()) if sets else set()
    return {
        "label": label,
        "n_seeds": n_seeds,
        "n_true": len(truth),
        "set_sizes": [len(sets[s]) for s in sorted(sets)],
        "mean_set_size": float(np.mean([len(v) for v in sets.values()])) if sets else 0.0,
        "true_recovered_all_seeds": int(sum(c == n_seeds for c in counts.values())),
        "true_never_recovered": int(sum(c == 0 for c in counts.values())),
        "true_recovered_any": int(sum(c > 0 for c in counts.values())),
        "frac_true_recovered_all": float(np.mean([c == n_seeds for c in counts.values()])) if truth else np.nan,
        "frac_true_recovered_any": float(np.mean([c > 0 for c in counts.values()])) if truth else np.nan,
        "genes_in_all_seeds": len(inter),
        "genes_in_union": len(union),
        "per_gene_frequency": counts,
    }
