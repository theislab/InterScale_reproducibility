"""Spatial scale-dependent Moran analysis.

Moran's I of a feature (gene or latent dimension) is computed on a series of spatial kNN graphs with
increasing ``k`` (or radius graphs with increasing radius). The resulting profile ``I(k)`` describes how far
spatial autocorrelation persists: localized patterns decay quickly with ``k``, tissue-scale patterns stay high.
The profile is summarized by its area under the curve (trapezoidal rule over ``k`` or radius).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.neighbors import NearestNeighbors

import anndata as ad
import squidpy as sq

__all__ = ["DEFAULT_KS", "moran_profile", "moran_auc", "moran_delta", "radii_from_knn"]

DEFAULT_KS = (4, 8, 16, 32, 64)


def _feature_adata(
    adata: ad.AnnData,
    genes: Sequence[str] | None,
    layer: str | None,
    obsm_key: str | None,
    library_key: str | None,
    spatial_key: str,
) -> ad.AnnData:
    """Lightweight AnnData holding only the features, coordinates and library labels."""
    if obsm_key is not None:
        X = np.asarray(adata.obsm[obsm_key])
        var_names = [f"{obsm_key}_{i}" for i in range(X.shape[1])]
    else:
        genes = list(adata.var_names) if genes is None else list(genes)
        sub = adata[:, genes]
        X = sub.X if layer is None else sub.layers[layer]
        X = X.toarray() if issparse(X) else np.asarray(X)
        var_names = genes
    obs = adata.obs[[library_key]].copy() if library_key is not None else pd.DataFrame(index=adata.obs_names)
    fa = ad.AnnData(X=X.astype(np.float64), obs=obs, var=pd.DataFrame(index=var_names))
    fa.obsm["spatial"] = np.asarray(adata.obsm[spatial_key])
    return fa


def _spatial_graph(fa: ad.AnnData, scale: float, graph: str, library_key: str | None) -> None:
    # `spatial_neighbors_knn/radius` replace the deprecated `spatial_neighbors(coord_type="generic", ...)`.
    if graph == "knn":
        if hasattr(sq.gr, "spatial_neighbors_knn"):
            sq.gr.spatial_neighbors_knn(fa, n_neighs=int(scale), library_key=library_key)
        else:
            sq.gr.spatial_neighbors(fa, coord_type="generic", n_neighs=int(scale), library_key=library_key)
    elif graph == "radius":
        if hasattr(sq.gr, "spatial_neighbors_radius"):
            sq.gr.spatial_neighbors_radius(fa, radius=float(scale), library_key=library_key)
        else:
            sq.gr.spatial_neighbors(fa, coord_type="generic", radius=float(scale), library_key=library_key)
    else:
        raise ValueError(f"`graph` must be 'knn' or 'radius', got {graph!r}.")


def _moran(
    fa: ad.AnnData, scales: Sequence[float], graph: str, library_key: str | None, n_jobs: int | None
) -> pd.DataFrame:
    scale_col = "k" if graph == "knn" else "radius"
    rows = []
    for scale in scales:
        _spatial_graph(fa, scale, graph, library_key)
        if fa.obsp["spatial_connectivities"].nnz == 0:  # radius below every neighbor distance: I undefined
            rows.append(pd.DataFrame({"feature": fa.var_names, scale_col: scale, "I": np.nan, "frac_isolated": 1.0}))
            continue
        res = sq.gr.spatial_autocorr(
            fa,
            mode="moran",
            transformation=True,  # row-normalized spatial weights
            n_perms=None,
            copy=True,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )
        df = pd.DataFrame({"feature": res.index, scale_col: scale, "I": res["I"].to_numpy()})
        if graph == "radius":
            # cells without neighbors within the radius have zero spatial lag and shrink I towards 0
            deg = np.diff(fa.obsp["spatial_connectivities"].tocsr().indptr)
            df["frac_isolated"] = float(np.mean(deg == 0))
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def radii_from_knn(
    adata: ad.AnnData,
    ks: Sequence[int] = DEFAULT_KS,
    library_key: str | None = None,
    spatial_key: str = "spatial",
) -> np.ndarray:
    """Radii matching kNN neighborhood sizes: median distance to the ``k``-th nearest neighbor (within sections)."""
    coords = np.asarray(adata.obsm[spatial_key])
    libs = adata.obs[library_key].astype(str).to_numpy() if library_key is not None else np.zeros(adata.n_obs)
    ks = sorted(int(k) for k in ks)
    dists = []
    for lib in np.unique(libs):
        c = coords[libs == lib]
        d, _ = NearestNeighbors(n_neighbors=min(max(ks) + 1, len(c))).fit(c).kneighbors(c)
        dists.append(d[:, [min(k, d.shape[1] - 1) for k in ks]])
    return np.median(np.vstack(dists), axis=0)


def moran_profile(
    adata: ad.AnnData,
    ks: Sequence[int] = DEFAULT_KS,
    library_key: str | None = None,
    genes: Sequence[str] | None = None,
    layer: str | None = None,
    obsm_key: str | None = None,
    spatial_key: str = "spatial",
    per_library: bool = False,
    n_jobs: int | None = None,
    graph: Literal["knn", "radius"] = "knn",
    radii: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Moran's I of each feature on spatial kNN (or radius) graphs of increasing size.

    Parameters
    ----------
    adata
        Annotated data with spatial coordinates in ``adata.obsm[spatial_key]``.
    ks
        Neighborhood sizes of the kNN graphs. For ``graph="radius"`` without ``radii``, the radii are matched to
        these neighborhood sizes with :func:`radii_from_knn`.
    library_key
        Column in ``adata.obs`` with the tissue section / FOV. Graphs are built within each section.
    genes
        Genes to analyse (default: all). Ignored if ``obsm_key`` is given.
    layer
        Expression layer to use (default: ``adata.X``).
    obsm_key
        If given, analyse the columns of ``adata.obsm[obsm_key]`` (e.g. a local or global latent) instead of genes.
    spatial_key
        Key of the spatial coordinates in ``adata.obsm``.
    per_library
        If ``False`` (default), one Moran's I is computed on the block-diagonal graph over all sections.
        If ``True``, Moran's I is computed separately in each section.
    n_jobs
        Passed to :func:`squidpy.gr.spatial_autocorr`.
    graph
        ``"knn"`` (default): kNN graphs with ``k`` in ``ks``. ``"radius"``: radius graphs with radii ``radii``.
    radii
        Radii (in units of the spatial coordinates) for ``graph="radius"``. If ``None``, the median distance to the
        ``k``-th nearest neighbor for each ``k`` in ``ks`` is used, so both graph types cover comparable scales.

    Returns
    -------
    Long-format data frame with columns ``feature``, ``k`` (or ``radius`` and ``frac_isolated``), ``I``
    (and ``library`` if ``per_library``).
    """
    if per_library and library_key is None:
        raise ValueError("`per_library=True` requires `library_key`.")
    if graph == "knn":
        if radii is not None:
            raise ValueError("`radii` is only used with `graph='radius'`.")
        scales = sorted(int(k) for k in ks)
    elif graph == "radius":
        if radii is None:
            radii = radii_from_knn(adata, ks=ks, library_key=library_key, spatial_key=spatial_key)
        scales = sorted(float(r) for r in radii)
    else:
        raise ValueError(f"`graph` must be 'knn' or 'radius', got {graph!r}.")
    fa = _feature_adata(adata, genes, layer, obsm_key, library_key, spatial_key)

    if not per_library:
        return _moran(fa, scales, graph, library_key, n_jobs)

    out = []
    for lib in fa.obs[library_key].astype(str).unique():
        sub = fa[fa.obs[library_key].astype(str) == lib].copy()
        if graph == "knn" and sub.n_obs <= max(scales):
            raise ValueError(f"Library {lib!r} has {sub.n_obs} cells, fewer than k={max(scales)} + 1.")
        out.append(_moran(sub, scales, graph, None, n_jobs).assign(library=lib))
    return pd.concat(out, ignore_index=True)


def moran_auc(profile: pd.DataFrame, log2_k: bool = False) -> pd.DataFrame:
    """Area under the Moran curve ``I(k)`` per feature (trapezoidal integration over ``k`` or radius).

    Parameters
    ----------
    profile
        Output of :func:`moran_profile`.
    log2_k
        Integrate over ``log2`` of the scale instead of the scale itself, which weights every doubling of the
        neighborhood equally. With the raw scale (default, as in the Methods), the largest interval dominates the area.
        Radius AUCs are in coordinate units and therefore not on the same scale as kNN AUCs; compare ``auc_norm``.

    Returns
    -------
    Data frame with one row per feature (and library, if present) and column ``auc``,
    plus ``auc_norm`` (AUC divided by the integration range, i.e. the mean Moran's I over scales).
    """
    groups = ["feature"] + (["library"] if "library" in profile.columns else [])
    scale_col = "k" if "k" in profile.columns else "radius"

    def _auc(df: pd.DataFrame) -> pd.Series:
        df = df.sort_values(scale_col)
        x = df[scale_col].to_numpy(dtype=float)
        x = np.log2(x) if log2_k else x
        auc = np.trapezoid(df["I"].to_numpy(), x)
        return pd.Series({"auc": auc, "auc_norm": auc / (x[-1] - x[0])})

    return profile.groupby(groups, sort=False).apply(_auc, include_groups=False).reset_index()


def moran_delta(profile: pd.DataFrame, split: float) -> pd.DataFrame:
    """Split the decay of the Moran curve into a local (small scales) and a global (large scales) part.

    With scales ``s_min < split < s_max`` (``k`` or radius):

    - ``delta_local = I(s_min) - I(split)``, ``delta_global = I(split) - I(s_max)``
    - ``slope_local`` / ``slope_global``: the deltas divided by the number of doublings of the scale they span
      (``log2(split / s_min)`` and ``log2(s_max / split)``). The two ranges usually span different numbers of scales,
      so the slopes (ΔI per doubling) are the comparable quantities.
    - ``rel_local = delta_local / I(s_min)``, ``rel_global = delta_global / I(split)``: fraction of the
      autocorrelation lost in each range, independent of how strongly autocorrelated the feature is overall.

    Parameters
    ----------
    profile
        Output of :func:`moran_profile` (or a profile averaged over libraries). ``split`` must be one of its scales.
    split
        Scale separating local from global.

    Returns
    -------
    Data frame with one row per feature (and library, if present).
    """
    groups = ["feature"] + (["library"] if "library" in profile.columns else [])
    scale_col = "k" if "k" in profile.columns else "radius"
    scales = np.sort(profile[scale_col].unique())
    if not np.isclose(scales, split).any():
        raise ValueError(f"`split`={split} is not one of the scales {scales.tolist()}.")
    split = scales[np.isclose(scales, split)][0]
    s_min, s_max = scales[0], scales[-1]
    if not s_min < split < s_max:
        raise ValueError(f"`split` must lie strictly between {s_min} and {s_max}.")

    I = profile.pivot_table(index=groups, columns=scale_col, values="I", dropna=False)
    out = pd.DataFrame(
        {"I_min": I[s_min], "I_split": I[split], "I_max": I[s_max]}, index=I.index
    )
    out["delta_local"] = out["I_min"] - out["I_split"]
    out["delta_global"] = out["I_split"] - out["I_max"]
    out["delta_total"] = out["I_min"] - out["I_max"]
    out["slope_local"] = out["delta_local"] / np.log2(split / s_min)
    out["slope_global"] = out["delta_global"] / np.log2(s_max / split)
    out["rel_local"] = out["delta_local"] / out["I_min"]
    out["rel_global"] = out["delta_global"] / out["I_split"]
    return out.reset_index()
