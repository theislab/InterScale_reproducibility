import random
import torch
import numpy as np

def set_full_reproducibility(seed: int = 44) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def expected_intersection(sizes, pool: int) -> float:
    r"""Expected number of genes common to **all** the sets, under random selection.

    ``sizes`` are the observed set sizes and ``pool`` the number of genes any of them could have
    contained (see :func:`pool_mask`). A given gene enters set *i* with marginal probability
    ``a_i / pool``, and the sets are drawn independently of each other, so

    .. math:: \mathbb{E}\Big|\bigcap_i A_i\Big| = P \prod_i \frac{a_i}{P}

    This is **exact** -- expectation is linear, so the negative dependence between genes *within*
    one fixed-size set does not enter. It is the variance, and hence the tail probability, that
    the dependence affects; use :func:`intersection_tail_probability` for that.

    With two sets this reduces to the hypergeometric mean ``a*b/P`` used by
    :func:`pair_metrics`.

    Parameters
    ----------
    sizes: sequence of int
        Observed size of each set, e.g. ``[26, 20, 24, 23, 26]`` for five seeds.
    pool: int
        Number of genes the selection could have drawn from.

    Returns
    -------
    float
        Expected size of the intersection. ``nan`` if ``pool`` is 0.

    Examples
    --------
    >>> round(expected_intersection([25] * 5, 88), 3)   # the recommended read-out
    0.169
    """
    sizes = np.asarray(sizes, dtype=float)
    if not pool or sizes.size == 0:
        return float("nan")
    if np.any(sizes > pool):
        raise ValueError(f"a set of size {sizes.max():.0f} cannot come from a pool of {pool}")
    return float(pool * np.prod(sizes / pool))


# Lifted verbatim from interscale_drvi-decoder/scripts/gene_program_stability.py so the
# reproducibility notebooks do not have to import from a sibling repo's scripts/ directory.
# Keep the two in sync; the docstring's doctests are the contract.
def intersection_tail_probability(
    sizes,
    pool: int,
    observed: int,
    *,
    alternative: str = "greater",
    n_iter: int = 20_000,
    seed: int = 0,
) -> dict:
    """P(intersection at least as extreme as ``observed``) under random selection.

    The companion to :func:`expected_intersection`: that gives the null's mean, this gives a
    p-value, which is what a "the programs are/are not reproducible" claim actually needs. There
    is no closed form for more than two sets -- the per-gene indicators are negatively dependent
    within each fixed-size set -- so the null is sampled. With exactly two sets the intersection
    *is* hypergeometric and the exact survival function is used instead.

    Parameters
    ----------
    sizes: sequence of int
        Observed size of each set.
    pool: int
        Number of genes the selection could have drawn from.
    observed: int
        Observed size of the intersection.
    alternative: {"greater", "less", "two-sided"}
        ``"greater"`` tests reproducibility (the usual case). ``"less"`` tests whether the sets
        overlap *less* than chance, which is worth having: under the registered settings two of
        the eight (arm, component) combinations land below their own null (``linear`` local, 0
        genes against 0.56 expected).
    n_iter: int
        Monte-Carlo draws (ignored when there are two sets).
    seed: int
        Seed for the null draws, so the p-value is reproducible.

    Returns
    -------
    dict
        ``observed``, ``expected``, ``p_value``, ``method`` ("hypergeometric" or
        "monte-carlo"), ``n_iter``, the null's ``null_mean`` / ``null_sd`` /
        ``null_q95`` (its 95th percentile, i.e. the smallest intersection that would be
        surprising), and ``fold_change``.

    Notes
    -----
    The Monte-Carlo p-value is ``(1 + #{null >= observed}) / (1 + n_iter)``, the standard
    conservative estimator: it never returns exactly 0, which an estimate from a finite number of
    draws cannot support. So the smallest reportable value at the default ``n_iter`` is
    ``p < 5e-5``; raise ``n_iter`` to go lower.

    Examples
    --------
    >>> r = intersection_tail_probability([26, 20, 24, 23, 26], 88, 10)
    >>> r["p_value"] < 1e-4, r["method"]
    (True, 'monte-carlo')
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError("alternative must be 'greater', 'less' or 'two-sided'")
    sizes = np.asarray(sizes, dtype=int)
    if sizes.size < 2:
        raise ValueError("need at least two sets to have an intersection to test")
    if np.any(sizes > pool):
        raise ValueError(f"a set of size {sizes.max()} cannot come from a pool of {pool}")

    expected = expected_intersection(sizes, pool)
    out = {"observed": int(observed), "expected": expected, "n_sets": int(sizes.size),
           "sizes": sizes.tolist(), "pool": int(pool),
           "fold_change": float(observed / expected) if expected > 0 else np.inf}

    if sizes.size == 2:
        from scipy.stats import hypergeom

        rv = hypergeom(pool, int(sizes[0]), int(sizes[1]))
        p_ge = float(rv.sf(observed - 1))
        p_le = float(rv.cdf(observed))
        null_mean, null_sd = float(rv.mean()), float(rv.std())
        null_q95 = float(rv.ppf(0.95))
        method, n_used = "hypergeometric", 0
    else:
        rng = np.random.default_rng(seed)
        counts = np.ones((n_iter, pool), dtype=bool)
        for a in sizes:
            # exactly `a` genes per draw: rank a uniform row and keep the a smallest
            idx = np.argpartition(rng.random((n_iter, pool)), a - 1, axis=1)[:, :a]
            m = np.zeros((n_iter, pool), dtype=bool)
            np.put_along_axis(m, idx, True, axis=1)
            counts &= m
        null = counts.sum(axis=1)
        p_ge = float((1 + np.sum(null >= observed)) / (1 + n_iter))
        p_le = float((1 + np.sum(null <= observed)) / (1 + n_iter))
        null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))
        null_q95 = float(np.quantile(null, 0.95))
        method, n_used = "monte-carlo", int(n_iter)

    out.update({
        "p_value": {"greater": p_ge, "less": p_le,
                    "two-sided": min(1.0, 2 * min(p_ge, p_le))}[alternative],
        "alternative": alternative,
        "method": method,
        "n_iter": n_used,
        "null_mean": null_mean,
        "null_sd": null_sd,
        "null_q95": null_q95,
    })
    return out
