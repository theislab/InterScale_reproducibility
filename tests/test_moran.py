"""Tests for the scale-dependent Moran analysis (``src/tl/moran.py``, ``src/pl/moran_plots.py``).

Runs with pytest or directly: ``python tests/test_moran.py``.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import pl  # noqa: E402
from src.tl.moran import auc_from_profile, delta_from_profile, moran_profile, moran_scale, radii_from_knn  # noqa: E402

KS = (4, 8, 16, 32, 64)


def _synthetic(n_side: int = 40, n_sections: int = 2, seed: int = 0) -> ad.AnnData:
    """Square grids (one per section) with a tissue-scale gradient, small hotspots and noise."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.c_[xs.ravel(), ys.ravel()].astype(float)

    blocks, obs = [], []
    for s in range(n_sections):
        gradient = coords[:, 0] / n_side
        centers = rng.uniform(0, n_side, size=(40, 2))
        d2 = ((coords[:, None, :] - centers[None]) ** 2).sum(-1)
        hotspots = np.exp(-d2 / (2 * 1.0**2)).sum(1)
        noise = rng.normal(size=len(coords))
        blocks.append(np.c_[gradient, hotspots, noise])
        obs.append(pd.DataFrame({"section": f"s{s}"}, index=[f"s{s}_{i}" for i in range(len(coords))]))

    X = np.vstack(blocks)
    adata = ad.AnnData(X=X, obs=pd.concat(obs), var=pd.DataFrame(index=["gradient", "hotspots", "noise"]))
    adata.obs["section"] = adata.obs["section"].astype("category")
    # offset sections so a graph built across sections would be wrong, not silently fine
    adata.obsm["spatial"] = np.vstack([coords + 10 * s for s in range(n_sections)])
    adata.obsm["latent"] = X[:, ::-1].copy()
    return adata


def _moran_reference(coords: np.ndarray, x: np.ndarray, k: int, sections: np.ndarray) -> float:
    """Moran's I with a row-normalized directed kNN graph built within sections (global mean centering)."""
    n = len(x)
    rows, cols = [], []
    for s in np.unique(sections):
        idx = np.flatnonzero(sections == s)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(coords[idx])
        _, nbrs = nn.kneighbors(coords[idx])
        rows.append(np.repeat(idx, k))
        cols.append(idx[nbrs[:, 1:]].ravel())
    W = csr_matrix((np.full(n * k, 1.0 / k), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    z = x - x.mean()
    return float(z @ (W @ z) / (z @ z))


def test_profile_matches_reference():
    adata = _synthetic()
    prof = moran_profile(adata, ks=(4, 16), library_key="section")
    sections = adata.obs["section"].to_numpy()
    for _, row in prof.iterrows():
        x = adata[:, row["feature"]].X.ravel()
        ref = _moran_reference(adata.obsm["spatial"], x, int(row["k"]), sections)
        # ties in grid distances can pick different neighbors -> small tolerance
        assert abs(row["I"] - ref) < 0.02, (row.to_dict(), ref)


def test_scale_dependence_and_auc_ranking():
    adata = _synthetic()
    prof = moran_profile(adata, ks=KS, library_key="section")
    I = prof.pivot(index="feature", columns="k", values="I")

    assert (I.loc["gradient"] > 0.9).all()
    assert I.loc["hotspots", 4] > 0.3
    # the localized pattern loses most of its autocorrelation at large k, the gradient does not
    assert I.loc["hotspots", 64] / I.loc["hotspots", 4] < 0.5
    assert I.loc["gradient", 64] / I.loc["gradient", 4] > 0.9
    assert (I.loc["noise"].abs() < 0.05).all()

    for log2_k in (False, True):
        auc = auc_from_profile(prof, log2_k=log2_k).set_index("feature")["auc"]
        assert auc["gradient"] > auc["hotspots"] > auc["noise"]


def test_auc_trapezoid():
    prof = pd.DataFrame({"feature": "f", "k": [4, 8, 16], "I": [1.0, 0.5, 0.25]})
    res = auc_from_profile(prof).iloc[0]
    assert np.isclose(res["auc"], 0.75 * 4 + 0.375 * 8)
    assert np.isclose(res["auc_norm"], res["auc"] / 12)
    res_log = auc_from_profile(prof, log2_k=True).iloc[0]
    assert np.isclose(res_log["auc"], 0.75 + 0.375)


def test_obsm_and_per_library():
    adata = _synthetic()
    prof = moran_profile(adata, ks=(4, 8), library_key="section", obsm_key="latent", per_library=True)
    assert set(prof["feature"]) == {"latent_0", "latent_1", "latent_2"}
    assert set(prof["library"]) == {"s0", "s1"}
    assert len(prof) == 3 * 2 * 2
    # latent_2 is the gradient column (features reversed)
    assert (prof.loc[prof["feature"] == "latent_2", "I"] > 0.9).all()
    auc = auc_from_profile(prof)
    assert {"feature", "library", "auc", "auc_norm"} <= set(auc.columns) and len(auc) == 6


def test_radius_graph():
    adata = _synthetic()
    radii = radii_from_knn(adata, ks=KS, library_key="section")
    # unit grid: 4th neighbor at distance 1, radii increase with k
    assert np.isclose(radii[0], 1.0) and np.all(np.diff(radii) > 0)

    prof = moran_profile(adata, library_key="section", graph="radius", radii=radii)
    # default radii are matched to ks
    prof_default = moran_profile(adata, ks=KS, library_key="section", graph="radius")
    assert np.allclose(np.sort(prof_default["radius"].unique()), radii)
    assert {"feature", "radius", "I", "frac_isolated"} <= set(prof.columns)
    assert (prof["frac_isolated"] == 0).all()  # regular grid, every cell has neighbors
    I = prof.pivot(index="feature", columns="radius", values="I")
    assert (I.loc["gradient"] > 0.9).all()
    assert I.loc["hotspots"].iloc[-1] / I.loc["hotspots"].iloc[0] < 0.5
    assert (I.loc["noise"].abs() < 0.05).all()

    auc = auc_from_profile(prof).set_index("feature")["auc_norm"]
    assert auc["gradient"] > auc["hotspots"] > auc["noise"]

    # isolated cells are reported when the radius is below the grid spacing
    prof_small = moran_profile(adata, library_key="section", graph="radius", radii=[0.5, 1.0])
    assert prof_small.groupby("radius")["frac_isolated"].first().to_dict() == {0.5: 1.0, 1.0: 0.0}
    assert prof_small.loc[prof_small["radius"] == 0.5, "I"].isna().all()

    try:
        moran_profile(adata, radii=[1.0])  # radii without graph="radius"
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_moran_delta():
    prof = pd.DataFrame({"feature": "f", "k": [4, 8, 16, 32, 64, 128], "I": [0.8, 0.6, 0.4, 0.35, 0.3, 0.25]})
    d = delta_from_profile(prof, split=16).iloc[0]
    assert np.isclose(d["delta_local"], 0.4) and np.isclose(d["delta_global"], 0.15)
    assert np.isclose(d["delta_total"], 0.55)
    assert np.isclose(d["slope_local"], 0.4 / 2) and np.isclose(d["slope_global"], 0.15 / 3)
    assert np.isclose(d["rel_local"], 0.5) and np.isclose(d["rel_global"], 0.15 / 0.4)

    # synthetic: hotspots lose their autocorrelation locally, the gradient barely decays
    adata = _synthetic()
    d = delta_from_profile(moran_profile(adata, ks=KS, library_key="section"), split=16).set_index("feature")
    assert d.loc["hotspots", "slope_local"] > d.loc["gradient", "slope_local"]
    assert d.loc["hotspots", "rel_local"] > 0.3 > d.loc["gradient", "rel_local"]

    for bad in (4, 128, 10):
        try:
            delta_from_profile(prof, split=bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_moran_scale_uns_and_copy():
    adata = _synthetic()
    res = moran_scale(adata, ks=KS, library_key="section", per_library=True, copy=True)
    assert "moran_scale" not in adata.uns
    assert {"profile", "profile_per_library", "stats", "params"} == set(res)
    assert res["params"]["split"] == 16  # middle scale by default
    stats = res["stats"].set_index("feature")
    assert np.allclose(stats["local_index"], stats["slope_local"] - stats["slope_global"])
    # mean curve over libraries
    mean = res["profile_per_library"].groupby(["feature", "k"])["I"].mean()
    assert np.allclose(res["profile"].set_index(["feature", "k"])["I"].loc[mean.index], mean)

    moran_scale(adata, ks=KS, library_key="section", split=8, key_added="ms")
    assert adata.uns["ms"]["params"]["split"] == 8 and "profile_per_library" not in adata.uns["ms"]
    assert None not in adata.uns["ms"]["params"].values()  # None cannot be written to h5ad


def test_plots():
    adata = _synthetic()
    adata.obsm["latent"] = np.c_[adata.X, adata.X[:, 1] ** 2]  # 4 features: 2 local, 2 global for n_top=2
    moran_scale(adata, ks=KS, library_key="section", obsm_key="latent", per_library=True)

    local, glob = pl.top_local_global(adata, n_top=10)
    assert len(local) == len(glob) == 2 and not set(local) & set(glob)  # n_top capped at half the features
    li = adata.uns["moran_scale"]["stats"].set_index("feature")["local_index"]
    assert li[local[0]] == li.max() and li[glob[0]] == li.min()

    ax = pl.moran_auc(adata, n_top=1, show=False)
    heights = [p.get_height() for p in ax.patches]
    assert heights == sorted(heights, reverse=True)
    assert len({p.get_facecolor() for p in ax.patches}) == 3  # local, global, other
    ax = pl.moran_auc(adata, features=["latent_0", "latent_1"], n_top=1, max_labels=1, show=False)
    assert len(ax.patches) == 2

    for genes, n_lines in (("both", 4), ("local", 2), ("global", 2), (["latent_0"], 1)):
        ax = pl.moran_curves(adata, genes=genes, n_top=2, show=False)
        assert len(ax.get_lines()) == n_lines + 2  # + split and zero lines
    ax = pl.moran_curves(adata, genes=["latent_0", "latent_1"], palette={"latent_0": "#ff0000"}, sd=True, show=False)
    assert ax.get_legend() is not None and len(ax.texts) == 1  # palette -> legend, other gene labelled directly

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pl.moran_curves(adata, genes=["latent_0"] * (pl.moran_plots.MAX_CURVES + 1), show=False)
    assert any("curves" in str(x.message) for x in w)
    for bad in ({"genes": "all"}, {"genes": ["nope"]}):
        try:
            pl.moran_curves(adata, show=False, **bad)
            raise AssertionError("expected an error")
        except (ValueError, KeyError):
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASSED {name}")
