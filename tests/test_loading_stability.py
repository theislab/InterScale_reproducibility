"""Tests for ``src.loading_stability``.

The minimal test the task asks for is
:func:`test_strong_loading_is_significant_and_noise_gene_is_uniform`: on one synthetic model, a
gene with a large known association gets ``q ~ 0`` and a noise gene's p-value is uniform. The
rest pin the pieces that a wrong result would pass silently -- the BH implementation against
statsmodels, the p-value floor, the degeneracy of the null against the decoder loading, and the
fact that the empirical loading really is the same standardization as ``gene_loadings``.

Run from the repo root::

    python -m pytest tests/test_loading_stability.py -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loading_stability import (  # noqa: E402
    benjamini_hochberg,
    compare_decoder_and_empirical,
    consensus_frequency,
    consensus_genes,
    demonstrate_null_is_degenerate,
    permutation_test_loadings,
    recovery_numbers,
    sets_to_frames,
    significance_report,
    significant_sets,
    standardized_empirical_loading,
    standardized_partial_loading,
)

N_CELLS, N_GENES, N_DIMS = 4000, 30, 6
STRONG, MEDIUM = "STRONG", "MEDIUM"


@pytest.fixture(scope="module")
def toy():
    """One synthetic 'trained model': Z, X, W and the decoder standardized loadings.

    Gene 0 (STRONG) is driven hard by dimension 0, gene 1 (MEDIUM) weakly by dimension 1, and
    every other gene is independent noise -- so the test has a known answer on both ends.
    """
    rng = np.random.default_rng(0)
    Z = rng.normal(0, 1, size=(N_CELLS, N_DIMS)).astype(np.float32)
    X = rng.normal(0, 1, size=(N_CELLS, N_GENES))
    X[:, 0] += 3.0 * Z[:, 0]  # r ~ 0.95
    X[:, 1] += 0.15 * Z[:, 1]  # r ~ 0.15, real but small
    genes = np.array([STRONG, MEDIUM] + [f"NOISE{i}" for i in range(N_GENES - 2)])
    W = rng.normal(0, 0.2, size=(N_GENES, N_DIMS))
    W[0, 0] = 2.0
    x_std = X.std(axis=0, ddof=1)
    S = W * Z.std(axis=0, ddof=1)[None, :] / x_std[:, None]
    return {"Z": Z, "X": X, "genes": genes, "W": W, "x_std": x_std, "S": S}


# ------------------------------------------------------------------------------------------
# the statistic
# ------------------------------------------------------------------------------------------


def test_empirical_loading_is_the_same_standardization_as_gene_loadings(toy):
    """W_hat * std(Z)/std(X) with the OLS slope -- i.e. Pearson r, computed two ways."""
    Z, X = toy["Z"].astype(np.float64), toy["X"]
    L = standardized_empirical_loading(Z, X)
    for e in range(N_DIMS):
        slope = np.polyfit(Z[:, e], X[:, 0], 1)[0]
        by_formula = slope * Z[:, e].std(ddof=1) / X[:, 0].std(ddof=1)
        assert np.isclose(L[0, e], by_formula, rtol=1e-6, atol=1e-9)
        assert np.isclose(L[0, e], np.corrcoef(Z[:, e], X[:, 0])[0, 1], rtol=1e-6, atol=1e-9)


def test_constant_gene_and_constant_dim_give_zero_not_nan(toy):
    Z, X = toy["Z"].copy(), toy["X"].copy()
    Z[:, 2] = 1.0
    X[:, 5] = 7.0
    L = standardized_empirical_loading(Z, X)
    assert np.isfinite(L).all()
    assert (L[:, 2] == 0).all()
    assert (L[5, :] == 0).all()


def test_dims_subset_selects_columns_in_order(toy):
    L_all = standardized_empirical_loading(toy["Z"], toy["X"])
    L_sub = standardized_empirical_loading(toy["Z"], toy["X"], dims=[3, 0])
    assert np.allclose(L_sub[:, 0], L_all[:, 3])
    assert np.allclose(L_sub[:, 1], L_all[:, 0])


# ------------------------------------------------------------------------------------------
# Benjamini-Hochberg
# ------------------------------------------------------------------------------------------


def test_bh_matches_statsmodels():
    rng = np.random.default_rng(1)
    p = np.concatenate([rng.uniform(0, 1, 200), rng.uniform(0, 1e-4, 20)])
    sm = pytest.importorskip("statsmodels.stats.multitest")
    assert np.allclose(benjamini_hochberg(p), sm.multipletests(p, method="fdr_bh")[1])


def test_bh_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.3, 0.9, 1.0])
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q) >= -1e-12)
    assert q.max() <= 1.0 and (q >= p - 1e-12).all()


def test_bh_ignores_nan_and_keeps_input_order():
    p = np.array([0.01, np.nan, 0.5])
    q = benjamini_hochberg(p)
    assert np.isnan(q[1])
    # family size is 2, not 3: a nan is not a test
    assert np.isclose(q[0], 0.02) and np.isclose(q[2], 0.5)
    assert np.isnan(benjamini_hochberg(np.array([np.nan, np.nan]))).all()


# ------------------------------------------------------------------------------------------
# step 1 -- the minimal test the task asks for
# ------------------------------------------------------------------------------------------


def test_strong_loading_is_significant_and_noise_gene_is_uniform(toy):
    """The minimal test: known-large loading -> q ~ 0; noise genes -> p uniform.

    H0 is that a gene's expression is independent of the dimension across cells, so under H0 the
    p-value is uniform on the ``B+1`` attainable values. Both ends are checked, because a test
    that only fires on the planted gene could still be miscalibrated (and one that is uniform
    everywhere has no power).
    """
    t = permutation_test_loadings(
        toy["Z"], toy["X"], gene_names=toy["genes"], B=999, random_state=0
    )
    d0 = t[t["dim"] == 0].set_index("gene")

    # Planted: no null draw reaches it, so p sits at the floor 1/(B+1). q is that floor times
    # family size over rank -- 1e-3 * 30 / 1 = 0.03 here, NOT ~0. With 30 genes at B=999 no q
    # below 0.03 is attainable at all; that BH-vs-B interaction is why `q ~ 0` should be read as
    # "at the floor the design allows", which is what this asserts.
    assert d0.loc[STRONG, "n_ge"] == 0
    assert np.isclose(d0.loc[STRONG, "p"], 1 / 1000)
    assert d0.loc[STRONG, "q"] < 0.05
    assert np.isclose(d0.loc[STRONG, "q"], d0["q"].min())
    assert d0["abs_loading"].idxmax() == STRONG

    # noise genes on the dimension nothing was planted on: p uniform
    noise = t[(t["dim"] == 3) & (t["gene"].str.startswith("NOISE"))]["p"].to_numpy()
    assert len(noise) == N_GENES - 2
    ks = pytest.importorskip("scipy.stats").kstest(noise, "uniform")
    assert ks.pvalue > 0.01, f"noise p-values not uniform (KS p={ks.pvalue:.3g})"
    assert abs(noise.mean() - 0.5) < 0.12
    # and none of them survives FDR
    assert (t[(t["dim"] == 3) & (t["gene"].str.startswith("NOISE"))]["q"] < 0.05).sum() == 0


def test_medium_effect_is_found_and_is_dimension_specific(toy):
    t = permutation_test_loadings(
        toy["Z"], toy["X"], gene_names=toy["genes"], B=999, random_state=0
    )
    assert t[(t["dim"] == 1) & (t["gene"] == MEDIUM)]["q"].iloc[0] < 0.05
    # the same gene on the other dimensions is not significant
    other = t[(t["dim"] != 1) & (t["gene"] == MEDIUM)]
    assert (other["q"] < 0.05).sum() == 0


def test_p_value_floor_is_one_over_b_plus_one(toy):
    for B in (99, 499):
        t = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"], B=B,
                                      dims=[0], random_state=0)
        assert np.isclose(t["p"].min(), 1.0 / (B + 1))
        assert t["p"].min() > 0.0  # never exactly zero


def test_reproducible_and_independent_of_the_dims_requested(toy):
    kw = dict(gene_names=toy["genes"], B=199, random_state=7)
    a = permutation_test_loadings(toy["Z"], toy["X"], dims=[0, 1, 2], **kw)
    b = permutation_test_loadings(toy["Z"], toy["X"], dims=[0, 1, 2], **kw)
    pd.testing.assert_frame_equal(a, b)
    # a per-dimension child stream: dim 1's p-values do not depend on dim 0 being asked for
    c = permutation_test_loadings(toy["Z"], toy["X"], dims=[0, 1], **kw)
    m = a[a["dim"] == 1].reset_index(drop=True)
    n = c[c["dim"] == 1].reset_index(drop=True)
    pd.testing.assert_series_equal(m["p"], n["p"])
    # ... and a different random_state gives a different null
    d = permutation_test_loadings(toy["Z"], toy["X"], dims=[0, 1, 2],
                                   gene_names=toy["genes"], B=199, random_state=8)
    assert not np.array_equal(a["n_ge"], d["n_ge"])


def test_fdr_scope_changes_the_family_without_dominating_per_dimension(toy):
    """Neither scope uniformly dominates, so the test asserts what is actually true.

    BH's adjusted value is ``min_{k >= rank} p_k m / k``. Pooling the dimensions raises ``m`` but
    also raises each test's rank, and the two do not cancel in a fixed direction: a p-value that
    is rank 1 inside its own dimension (``q = p * n_genes``) can be rank 40 globally
    (``q = p * n_genes * n_dims / 40``), which is *smaller*. "Global is stricter" is therefore
    false in general -- what is true is that the two control FDR over different families.
    """
    # B=9999, not 999: pooling 6 dims x 30 genes makes the family 180 tests, and BH cannot put
    # anything below 1/(B+1) * 180 -- which at B=999 is 0.18, so under global scope NOTHING is
    # significant at 0.05 however strong the effect. B has to resolve the family it is corrected
    # over; that is the practical coupling between B and fdr_scope.
    kw = dict(gene_names=toy["genes"], B=9999, random_state=0)
    per = permutation_test_loadings(toy["Z"], toy["X"], fdr_scope="per_dimension", **kw)
    glob = permutation_test_loadings(toy["Z"], toy["X"], fdr_scope="global", **kw)
    for tab in (per, glob):
        assert ((tab["q"] >= 0) & (tab["q"] <= 1)).all()
        assert tab[(tab["dim"] == 0) & (tab["gene"] == STRONG)]["q"].iloc[0] < 0.05
    assert not np.allclose(per["q"], glob["q"])
    assert per["fdr_scope"].iloc[0] == "per_dimension"
    assert glob["fdr_scope"].iloc[0] == "global"

    under_resolved = permutation_test_loadings(toy["Z"], toy["X"], fdr_scope="global",
                                               gene_names=toy["genes"], B=999, random_state=0)
    assert (under_resolved["q"] < 0.05).sum() == 0


def test_decoder_loadings_are_carried_but_not_tested(toy):
    t = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"], B=99,
                                  dims=[0, 1], decoder_loadings=toy["S"], random_state=0)
    assert "loading_decoder" in t
    row = t[(t["dim"] == 0) & (t["gene"] == STRONG)].iloc[0]
    assert np.isclose(row["loading_decoder"], toy["S"][0, 0])
    # the tested statistic is NOT the decoder one
    assert not np.isclose(row["loading"], row["loading_decoder"])


def test_partial_loading_is_the_multivariate_slope(toy):
    """W_hat is the coefficient of dim e with the others held fixed, standardized identically."""
    Z, X = toy["Z"].astype(np.float64), toy["X"]
    L = standardized_partial_loading(Z, X)
    Zc = Z - Z.mean(axis=0)
    Xc = X - X.mean(axis=0)
    coef = np.linalg.lstsq(Zc, Xc, rcond=None)[0]  # (n_dims, n_genes), the joint fit
    for e in range(N_DIMS):
        expected = coef[e, 0] * Z[:, e].std(ddof=1) / X[:, 0].std(ddof=1)
        assert np.isclose(L[0, e], expected, rtol=1e-5, atol=1e-9)


def test_partial_recovers_the_planted_dimension_and_marginal_leaks(toy):
    """The reason `partial` is the default: with correlated dims, marginal loadings leak.

    Dimension 4 is made an exact copy of dimension 0 plus noise, so the STRONG gene's *marginal*
    loading on 4 is large purely by proxy while its *partial* loading is not.
    """
    Z = toy["Z"].astype(np.float64).copy()
    rng = np.random.default_rng(3)
    Z[:, 4] = Z[:, 0] + rng.normal(0, 0.2, len(Z))
    X = toy["X"]
    marg = standardized_empirical_loading(Z, X)
    part = standardized_partial_loading(Z, X)
    assert abs(marg[0, 4]) > 0.5 * abs(marg[0, 0])  # leaked
    assert abs(part[0, 4]) < 0.2 * abs(part[0, 0])  # not leaked


def test_partial_statistic_finds_the_planted_gene_and_stays_calibrated(toy):
    t = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"], B=999,
                                  random_state=0, statistic="partial")
    assert t["statistic"].unique().tolist() == ["partial"]
    d0 = t[t["dim"] == 0].set_index("gene")
    assert d0.loc[STRONG, "n_ge"] == 0 and d0.loc[STRONG, "q"] < 0.05
    noise = t[(t["dim"] == 3) & (t["gene"].str.startswith("NOISE"))]["p"].to_numpy()
    ks = pytest.importorskip("scipy.stats").kstest(noise, "uniform")
    assert ks.pvalue > 0.01, f"partial-statistic p-values not uniform (KS p={ks.pvalue:.3g})"
    assert (t[(t["dim"] == 3) & (t["gene"].str.startswith("NOISE"))]["q"] < 0.05).sum() == 0


def test_partial_and_marginal_agree_when_dimensions_are_orthogonal(toy):
    """With uncorrelated dims the two statistics coincide, so any difference is collinearity."""
    Q = np.linalg.qr(toy["Z"].astype(np.float64) - toy["Z"].mean(axis=0))[0] * np.sqrt(N_CELLS)
    marg = standardized_empirical_loading(Q, toy["X"])
    part = standardized_partial_loading(Q, toy["X"])
    assert np.allclose(marg, part, atol=1e-8)


def test_compare_decoder_and_empirical_recovers_a_known_linear_decoder():
    """On data generated BY a linear decoder, the partial loading recovers its loading.

    ``toy``'s ``W`` is unrelated to how its ``X`` was generated, so agreement there is noise and
    asserting anything about it would be asserting noise. Here ``X`` really is ``W_true @ Z``
    plus noise, which is the regime the substitution assumes -- and the partial loading has to
    recover it while the marginal one need not, since ``Z`` is left correlated on purpose.
    """
    rng = np.random.default_rng(5)
    n, n_dims, n_genes = 6000, 5, 40
    base = rng.normal(0, 1, size=(n, n_dims))
    Z = base + 0.7 * base[:, [0]]  # deliberately correlated dimensions
    W_true = rng.normal(0, 1, size=(n_genes, n_dims))
    X = Z @ W_true.T + rng.normal(0, 0.5, size=(n, n_genes))
    S_true = W_true * Z.std(axis=0, ddof=1)[None, :] / X.std(axis=0, ddof=1)[:, None]

    dims = list(range(n_dims))
    part = compare_decoder_and_empirical(S_true, standardized_partial_loading(Z, X), dims=dims)
    marg = compare_decoder_and_empirical(S_true, standardized_empirical_loading(Z, X), dims=dims)

    assert list(part["dim"]) == dims
    assert part["abs_spearman"].between(0, 1).all()
    assert part["sign_agreement"].between(0, 1).all()
    assert (part["abs_spearman"] > 0.95).all(), part.to_string()
    assert part["abs_spearman"].median() > marg["abs_spearman"].median()


@pytest.mark.parametrize(("kw", "match"), [
    ({"statistic": "cov"}, "statistic must be"),
    ({"fdr_scope": "bonferroni"}, "fdr_scope must be"),
    ({"B": 0}, "B must be"),
    ({"dims": [99]}, "dims out of range"),
    ({"gene_names": ["a", "b"]}, "gene names for"),
])
def test_permutation_test_rejects_bad_arguments(toy, kw, match):
    call = {"gene_names": toy["genes"], "B": 9, **kw}
    with pytest.raises(ValueError, match=match):
        permutation_test_loadings(toy["Z"], toy["X"], **call)


def test_mismatched_cells_raise(toy):
    with pytest.raises(ValueError, match="cells"):
        permutation_test_loadings(toy["Z"][:100], toy["X"], B=9)


# ------------------------------------------------------------------------------------------
# the degeneracy this module exists to work around
# ------------------------------------------------------------------------------------------


def test_permuting_cells_cannot_move_the_decoder_loading(toy):
    """The reason the statistic had to change: std(Z) is invariant to cell order."""
    out = demonstrate_null_is_degenerate(toy["Z"], toy["x_std"], toy["W"], dim=0, B=10)
    # Machine precision, not identically zero: `std` sums the same values in a different order
    # after a permutation. 1e-12 is still ~9 orders below the 1/sqrt(n) null sd of any statistic
    # this module tests, i.e. the decoder loading does not move.
    assert (out["max_rel_deviation"] < 1e-12).all()
    assert np.allclose(out["std_z_obs"], out["std_z_permuted"], rtol=1e-12)
    assert out.attrs["p_value_of_decoder_loading"] == 1.0


# ------------------------------------------------------------------------------------------
# steps 2-3 -- selection, consensus
# ------------------------------------------------------------------------------------------


def test_significant_sets_lists_every_dim_even_when_empty(toy):
    t = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"], B=999,
                                  random_state=0)
    sets = significant_sets(t, alpha=0.05)
    assert set(sets) == set(range(N_DIMS))
    assert STRONG in sets[0]
    assert sets[3] == set() or not sets[3] & {STRONG, MEDIUM}
    with pytest.raises(ValueError, match="alpha must be"):
        significant_sets(t, alpha=0)


def test_consensus_counts_seeds_and_reports_what_was_testable(toy):
    t = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"], B=999,
                                  random_state=0)
    # three "seeds" sharing one table, so every gene is 3/3 -- the arithmetic, not the biology
    tables = {40: t, 41: t, 42: t}
    mapping = {40: {0: 0}, 41: {0: 0}, 42: {0: 0}}
    freq = consensus_frequency(tables, mapping, alpha=0.05, ref_dims=[0])
    assert (freq["n_seeds"] == 3).all() and (freq["n_testable"] == 3).all()
    assert STRONG in consensus_genes(freq, consensus_k=3)
    assert consensus_genes(freq, consensus_k=4) == []
    assert 0 in consensus_genes(freq, consensus_k=3, per_dimension=True)

    # a seed whose dimension did not match is not counted as a failure to replicate
    mapping_missing = {40: {0: 0}, 41: {0: 0}, 42: {}}
    freq2 = consensus_frequency(tables, mapping_missing, alpha=0.05, ref_dims=[0])
    assert (freq2["n_testable"] == 2).all()
    assert (freq2["n_seeds"] == 2).all()


def test_consensus_genes_on_empty_frequency_table():
    empty = pd.DataFrame(columns=["gene", "ref_dim", "n_seeds", "n_testable", "frequency", "seeds"])
    assert consensus_genes(empty) == []
    assert consensus_genes(empty, per_dimension=True) == {}
    with pytest.raises(ValueError, match="consensus_k must be"):
        consensus_genes(empty, consensus_k=0)


def test_significance_report_flags_the_p_value_floor(toy):
    # B=999, not 99: at B=99 the smallest attainable p is 0.01 and BH over 30 genes floors q at
    # 0.3, so nothing can be significant however strong the association. That interaction is what
    # the `n_at_p_floor` column exists to make visible, and it is asserted at the end.
    t = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"], B=999,
                                  dims=[0, 3], random_state=0)
    report = significance_report(t, alpha=0.05).set_index("dim")
    assert report.loc[0, "n_at_p_floor"] >= 1
    assert report.loc[0, "n_significant"] >= 1
    assert report.loc[3, "frac_significant"] == 0.0

    under_powered = permutation_test_loadings(toy["Z"], toy["X"], gene_names=toy["genes"],
                                              B=99, dims=[0], random_state=0)
    assert (under_powered["q"] < 0.05).sum() == 0
    assert significance_report(under_powered)["n_at_p_floor"].iloc[0] >= 1


# ------------------------------------------------------------------------------------------
# step 4 -- what goes into the existing plots
# ------------------------------------------------------------------------------------------


def test_sets_to_frames_gives_gene_indexed_frames():
    frames = sets_to_frames({40: ["B", "A"], 41: ["A"]})
    assert [list(f.index) for f in frames] == [["A", "B"], ["A"]]
    assert all(f.index.name == "gene" for f in frames)


def test_recovery_numbers_reproduce_the_figure_annotations():
    truth = ["A", "B", "C", "D"]
    sets = {40: ["A", "B", "X"], 41: ["A", "B"], 42: ["A", "Y"]}
    r = recovery_numbers(truth, sets, label="test")
    assert r["n_seeds"] == 3 and r["n_true"] == 4
    assert r["true_recovered_all_seeds"] == 1  # A
    assert r["true_never_recovered"] == 2  # C, D
    assert r["true_recovered_any"] == 2  # A, B
    assert r["genes_in_all_seeds"] == 1
    assert r["genes_in_union"] == 4  # A, B, X, Y
    assert r["per_gene_frequency"]["B"] == 2
