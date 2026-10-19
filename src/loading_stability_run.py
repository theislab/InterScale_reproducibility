"""One-call Steps 1-4 for one dataset, writing every table to ``stability_outputs/<dataset>/``.

Separated from :mod:`src.loading_stability` (which is the library: pure numpy/pandas/scipy, no
plotting, no repo paths) so that the notebook is a narrative over saved tables rather than a
place where a 20-minute computation hides. Run it from the notebook or from a shell::

    python -c "from src.loading_stability_run import run; run('legnini23')"

Nothing here retrains or modifies anything: it reads the per-seed arrays
``scripts/legnini_23/4a_seed_stability_extract.py`` already wrote, plus the h5ad, and writes only
into ``stability_outputs/``.

The dataset specs mirror ``scripts/legnini_23/4a_seed_stability_extract.py``'s ``SPECS`` and the
two existing stability notebooks' constants; the published gene lists and the top-K settings are
copied from them so the "before" side of the comparison is the published one and not a re-tuning.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.loading_stability import (
    COMPONENTS,
    consensus_frequency,
    consensus_genes,
    demonstrate_loading_gauge_freedom,
    demonstrate_null_is_degenerate,
    embedding_rank_report,
    load_expression,
    load_seed_arrays,
    match_all_to_reference,
    matching_report,
    permutation_test_loadings,
    recovery_numbers,
    significance_report,
    significant_sets,
    size_matched_sets,
)

__all__ = [
    "SPECS",
    "REPO",
    "OUT_ROOT",
    "run",
    "load_run",
    "load_table",
    "evaluate_single_model_selection",
]

REPO = Path(__file__).resolve().parent.parent
LUSTRE = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer")
OUT_ROOT = REPO / "stability_outputs"

SEEDS = [40, 41, 42, 43, 44]

# ------------------------------------------------------------------------------------------
# dataset specs -- all values copied from the existing pipeline, not re-chosen here
# ------------------------------------------------------------------------------------------

SPECS: dict[str, dict[str, Any]] = {
    # scripts/legnini_23/4a_seed_stability_extract.py SPECS["legnini23"], and the published
    # gene lists from notebooks/legnini_23/4_a_decoder_stability.ipynb.
    "legnini23": {
        "label": "Legnini et al. 2023 (neural tube organoid, Molecular Cartography, 88 genes)",
        "arrays_dir": LUSTRE / "results/legnini23_revision/seed_stability/analysis",
        "h5ad": LUSTRE / "data/legnini23_pp.h5ad",
        # legnini23 reconstructs layers["log1p_norm"]; .X holds raw counts, which is also what
        # the training-time zero-expression filter was applied to.
        "layer": "log1p_norm",
        "counts_key_for_filter": "X",
        "true_genes": {
            "local": [
                "DBX2",
                "AGO2",
                "CUX2",
                "FOS",
                "GLI1",
                "PAX6",
                "TUBB3",
                "SATB2",
                "DBX1",
                "NKX6.1",
                "ISL1",
                "JUNB",
            ],
            "global": [
                "AC129492.1",
                "TBP",
                "VAMP2",
                "FOXO6",
                "NEUROD1",
                "NEUROG2",
                "OLIG2",
                "PAX7",
                "DBX1",
                "HHIP",
                "PER3",
                "POLR1B",
                "PROM1",
                "ZNF609",
            ],
        },
        # min_frac the top-K path used (4a passes 0.1), which also sets the pool for the
        # chance-overlap baselines in the existing plots.
        "min_frac": 0.1,
    },
    # scripts/legnini_23/4a_seed_stability_extract.py SPECS["melton25"], and the published gene
    # lists from notebooks/Cosmx/4_decoder_stability_cosmx.ipynb.
    "melton25": {
        "label": "Melton/Jimenez 2025 (human pancreas, CosMx, 979 genes)",
        "arrays_dir": LUSTRE / "results/melton25_revision/best_recon/analysis",
        "h5ad": LUSTRE / "data/melton25.h5ad",
        # melton25's cfg.dataset.layer_key is None: .X IS the reconstructed matrix and `layers`
        # holds raw counts only. The opposite of legnini23 -- nothing may assume a layer name.
        "layer": None,
        "counts_key_for_filter": None,
        "true_genes": {
            "local": [
                "BTG1",
                "CD44",
                "ETV5",
                "GLUL",
                "H2AZ1",
                "MT1X",
                "CYSTM1",
                "LDLR",
                "HLA-A",
                "TPM2",
                "APP",
                "SPRY2",
                "TUBB",
                "DST",
                "CUZD1",
                "PTGES3",
                "RARRES2",
                "SRC",
                "INS",
                "CXCL2",
                "REG1A",
                "H4C3",
                "GDF15",
                "NEAT1",
                "GPX1",
            ],
            "global": [
                "CDKN1A",
                "FABP5",
                "MIF",
                "HSP90AB1",
                "TNFRSF12A",
                "IL12RB1",
                "SPP1",
                "RPL22",
                "PDGFD",
                "GPX3",
                "MEG3",
                "BTG1",
                "IL32",
                "IGFBP7",
                "HLA-A",
                "CALM3",
                "TPM1",
                "IGF1R",
                "HLA-E",
                "GDF15",
                "TAGLN",
                "IL10RB",
                "NDRG1",
                "C1QC",
            ],
        },
        "min_frac": 0.1,
    },
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    return obj


def run(
    dataset: str,
    *,
    seeds: list[int] | None = None,
    B: int = 1000,
    alpha: float = 0.05,
    statistic: str = "marginal",
    fdr_scope: str = "per_dimension",
    consensus_k: int = 3,
    random_state: int = 0,
    min_r: float = 0.3,
    reference: int | None = None,
    out_root: Path | None = None,
    reuse_tables: bool = False,
    verbose: bool = True,
) -> dict:
    """Steps 1-4 for one dataset. Returns the results and writes them under ``out_root``.

    The permutation test's null, restated: **a gene's expression is independent of a latent
    dimension across cells**, realised by permuting that dimension's cell order alone.

    Every table is written as CSV so the notebook can render it without recomputing, and a
    ``summary.json`` records the settings alongside the numbers.

    ``reuse_tables=True`` reads the per-(component, seed) permutation tables back from a previous
    run instead of recomputing them, and skips loading the expression matrix (the cached per-gene
    detection rate is enough for everything downstream). That is what makes re-deriving the
    selection criteria cheap: on melton25 the permutation tests are ~20 minutes and everything
    after them is seconds. It does NOT re-check the settings, so only reuse with the ``B``,
    ``alpha``, ``statistic`` and ``fdr_scope`` the tables were computed with.
    """
    if dataset not in SPECS:
        raise KeyError(f"dataset must be one of {sorted(SPECS)}, got {dataset!r}")
    spec = SPECS[dataset]
    seeds_list = list(SEEDS if seeds is None else seeds)
    ref = seeds_list[0] if reference is None else int(reference)
    out = Path(out_root or OUT_ROOT) / dataset
    out.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    say = print if verbose else (lambda *a, **k: None)
    say(f"\n{'=' * 78}\n{dataset}: {spec['label']}\n{'=' * 78}")

    # ---- the five existing seed outputs, read off disk -----------------------------------
    sa = {s: load_seed_arrays(spec["arrays_dir"], s) for s in seeds_list}
    ref_arrays = sa[ref]
    say(
        f"seeds {seeds_list}, reference {ref}: {ref_arrays.n_obs} cells, "
        f"{len(ref_arrays.var_names)} genes, n_embed {ref_arrays.n_embed}"
    )

    frac_path = out / "gene_frac.csv"
    if reuse_tables and frac_path.exists():
        cached = pd.read_csv(frac_path)
        frac = cached.set_index("gene").loc[list(ref_arrays.var_names), "frac"].to_numpy()
        X, keep = None, None
        say(f"  reuse_tables: expression matrix NOT loaded; per-gene detection rate read from {frac_path.name}")
    else:
        X, keep = load_expression(
            spec["h5ad"],
            layer=spec["layer"],
            x_std_expected=ref_arrays.x_std,
            counts_key_for_filter=spec["counts_key_for_filter"],
            verbose=verbose,
        )
        frac = (X > 0).mean(axis=0)
        pd.DataFrame({"gene": ref_arrays.var_names, "frac": frac}).to_csv(frac_path, index=False)
    pool_topk = int((frac >= spec["min_frac"]).sum())
    pool_perm = int(len(ref_arrays.var_names))
    say(
        f"pool: top-K path filtered at min_frac={spec['min_frac']} -> {pool_topk} genes; "
        f"the permutation path applies no expression filter -> {pool_perm}"
    )

    # ---- identifiability of the loading itself -------------------------------------------
    rank_rows, gauge_rows, gauge_dim_rows = [], [], []
    for comp in COMPONENTS:
        r = embedding_rank_report(ref_arrays.Z[comp])
        rank_rows.append({"component": comp, **r})
        for scale in (0.25, 0.5, 1.0, 2.0):
            g = demonstrate_loading_gauge_freedom(
                ref_arrays.W[comp],
                ref_arrays.Z[comp],
                ref_arrays.x_std,
                dims=list(ref_arrays.dims[comp]),
                n_top=int(ref_arrays.meta.get("n_top", 20)),
                scale=scale,
                random_state=random_state,
            )
            per_dim = g.pop("per_dim")
            gauge_rows.append({"component": comp, **g})
            for row in per_dim.to_dict("records"):
                gauge_dim_rows.append({"component": comp, "scale": scale, **row})
    rank_df = pd.DataFrame(rank_rows)
    gauge_df = pd.DataFrame(gauge_rows)
    say("\nembedding rank (reference seed):")
    say(
        rank_df[
            ["component", "n_dims", "rank", "full_rank", "sv_ratio", "looks_layer_normed", "row_norm_is_sqrt_d"]
        ].to_string(index=False)
    )
    say("\nloading gauge freedom -- same model, different loadings:")
    say(
        gauge_df[
            ["component", "scale", "max_rel_pattern_change", "median_jaccard_top_k", "median_shared_top_k", "n_top"]
        ].to_string(index=False)
    )

    # the literal null, against the decoder loading it cannot move
    deg = pd.concat(
        {
            comp: demonstrate_null_is_degenerate(
                ref_arrays.Z[comp],
                ref_arrays.x_std,
                ref_arrays.W[comp],
                dim=int(ref_arrays.dims[comp][0]),
                B=20,
                random_state=random_state,
            )
            for comp in COMPONENTS
        },
        names=["component"],
    ).reset_index(level=0)
    say("\npermuting cells against the DECODER loading (the literal null):")
    say(
        f"  max relative deviation over 20 permutations x 2 components: "
        f"{deg['max_rel_deviation'].max():.2e}  -> p = 1.0 for every gene, by construction"
    )

    # ---- Step 1+2: the permutation test, per seed and component --------------------------
    matching = {}
    tables = {}
    for comp in COMPONENTS:
        mapping, quality = match_all_to_reference(sa, comp, reference=ref, min_r=min_r)
        matching[comp] = (mapping, quality)
        rep_match = matching_report(quality)
        say(
            f"\n[{comp}] dimension matching to seed {ref} (Hungarian on |corr(z_i, z_j)|, "
            f"min_r={min_r}) -- the repo's own scripts/legnini_23/4b implementation:"
        )
        say(rep_match.to_string(index=False))

        ref_dims = [int(d) for d in ref_arrays.dims[comp]]
        for s in seeds_list:
            test_dims = sorted({mapping[s][d] for d in ref_dims if d in mapping.get(s, {})})
            if not test_dims:
                # None of the reference's informative dimensions has a counterpart in this seed
                # above min_r. That is a result about the latent space, not a reason to test
                # arbitrary dimensions, so the seed contributes an empty set and says so.
                say(f"  seed {s}:   0 dims matched above min_r={min_r} -- contributes nothing")
                tables[(comp, s)] = pd.DataFrame(
                    columns=["gene", "dim", "loading", "abs_loading", "n_ge", "B", "statistic", "p", "q", "fdr_scope"]
                )
                continue
            cached_t = out / "permutation_tables" / f"{comp}_seed{s}.csv.gz"
            if reuse_tables and cached_t.exists():
                tables[(comp, s)] = pd.read_csv(cached_t)
                say(f"  seed {s}: {len(test_dims):3d} dims, reused {cached_t.name}")
                continue
            if reuse_tables:
                raise FileNotFoundError(f"reuse_tables=True but {cached_t} is missing -- run once without it first.")
            t0 = time.time()
            tables[(comp, s)] = permutation_test_loadings(
                sa[s].Z[comp],
                X,
                dims=test_dims,
                gene_names=sa[s].var_names,
                B=B,
                random_state=random_state + s,
                statistic=statistic,
                fdr_scope=fdr_scope,
                decoder_loadings=sa[s].S[comp],
            )
            say(
                f"  seed {s}: {len(test_dims):3d} dims, B={B} -> {time.time() - t0:6.1f}s, "
                f"{int((tables[(comp, s)]['q'] < alpha).sum()):6d} of "
                f"{len(tables[(comp, s)]):6d} (gene, dim) tests significant"
            )

    nonempty = {(c, s): t for (c, s), t in tables.items() if len(t)}
    sig_report = (
        pd.concat(
            {k: significance_report(t, alpha=alpha, n_genes=len(ref_arrays.var_names)) for k, t in nonempty.items()},
            names=["component", "seed"],
        ).reset_index(level=[0, 1])
        if nonempty
        else pd.DataFrame(columns=["component", "seed", "dim", "n_genes", "n_significant"])
    )

    # ---- Step 3: consensus ---------------------------------------------------------------
    freq_all, consensus = {}, {}
    for comp in COMPONENTS:
        mapping, _ = matching[comp]
        ref_dims = [int(d) for d in ref_arrays.dims[comp]]
        freq_all[comp] = consensus_frequency(
            {s: tables[(comp, s)] for s in seeds_list}, mapping, alpha=alpha, ref_dims=ref_dims
        )
        consensus[comp] = {
            "genes": consensus_genes(freq_all[comp], consensus_k=consensus_k),
            "per_dimension": consensus_genes(freq_all[comp], consensus_k=consensus_k, per_dimension=True),
        }
        say(
            f"\n[{comp}] consensus (>= {consensus_k} of {len(seeds_list)} seeds, alpha={alpha}): "
            f"{len(consensus[comp]['genes'])} genes over "
            f"{len(consensus[comp]['per_dimension'])} matched dimensions"
        )

    # ---- Step 4: the three criteria, as per-seed gene sets -------------------------------
    # The genes the top-K path could have drawn from: criterion (c) is restricted to these so
    # that it differs from (a) only in the ranking statistic, not in the candidate set.
    pool_genes = set(np.asarray(ref_arrays.var_names)[frac >= spec["min_frac"]].tolist())

    criteria: dict[str, dict[str, dict[int, list[str]]]] = {}
    for comp in COMPONENTS:
        ref_dims = [int(d) for d in ref_arrays.dims[comp]]
        # (a) the published top-K rule, read from what 4a recorded -- not recomputed
        old = {s: list(sa[s].program[comp]) for s in seeds_list}
        # (b) q < alpha, union over the seed's matched dimensions
        new = {
            s: sorted(set().union(*significant_sets(tables[(comp, s)], alpha=alpha).values()))
            if len(tables[(comp, s)])
            else []
            for s in seeds_list
        }
        # (c) q < alpha, then matched to the top-K rule on BOTH confounds, so that the only
        #     thing differing from (a) is which statistic does the ranking:
        #       * the candidate pool -- (a) pre-filters at min_frac, (b) does not;
        #       * the set size -- (b)'s sets cover most of the panel, and large sets must
        #         overlap heavily whatever ranks them.
        #     The per-dimension cut is grown until the UNION reaches (a)'s own size for that
        #     seed, because dimensions share genes: a flat n_top per dimension undershoots.
        matched = {}
        for s in seeds_list:
            t = tables[(comp, s)]
            if not len(t):
                matched[s] = []
                continue
            t = t[t["gene"].isin(pool_genes)]
            dims_s = sorted(t["dim"].unique())
            target = len(old[s])
            best: set[str] = set()
            for k in range(1, len(pool_genes) + 1):
                sets_c = size_matched_sets(t, dict.fromkeys(dims_s, k), alpha=alpha)
                union = set().union(*sets_c.values()) if sets_c else set()
                if len(union) <= len(best):
                    break  # no more significant genes to add
                best = union
                if len(union) >= target:
                    break
            matched[s] = sorted(best)
        criteria[comp] = {"a_topk": old, "b_permutation": new, "c_size_matched": matched}

    recov_rows = []
    for comp in COMPONENTS:
        for name, sets_per_seed in criteria[comp].items():
            r = recovery_numbers(spec["true_genes"][comp], sets_per_seed, label=name)
            r.pop("per_gene_frequency")
            recov_rows.append(
                {
                    "component": comp,
                    "criterion": name,
                    # (a) and (c) draw from the min_frac-filtered pool; only (b) sees the panel.
                    "pool": pool_perm if name == "b_permutation" else pool_topk,
                    **r,
                }
            )
    recovery = pd.DataFrame(recov_rows)

    # The list a reader actually wants: per criterion, how many of the N seeds selected each
    # gene, with published-set membership alongside. `recovery_numbers.csv` counts these;
    # this NAMES them, so "stable across seeds" never has to be re-derived from the per-seed
    # sets by hand.
    stable_rows = []
    for comp in COMPONENTS:
        truth = set(spec["true_genes"][comp])
        for name, sets_per_seed in criteria[comp].items():
            counts: dict[str, list[int]] = {}
            for sd, gs in sets_per_seed.items():
                for g in gs:
                    counts.setdefault(str(g), []).append(int(sd))
            for g, hits in sorted(counts.items(), key=lambda kv: (-len(kv[1]), kv[0])):
                stable_rows.append(
                    {
                        "component": comp,
                        "criterion": name,
                        "gene": g,
                        "n_seeds": len(hits),
                        "n_seeds_total": len(sets_per_seed),
                        "seeds": ",".join(str(x) for x in sorted(hits)),
                        "in_published_set": g in truth,
                    }
                )
    stable = pd.DataFrame(
        stable_rows,
        columns=["component", "criterion", "gene", "n_seeds", "n_seeds_total", "seeds", "in_published_set"],
    )
    say("\ngenes selected in ALL seeds, per criterion (the 'stable' list):")
    for comp in COMPONENTS:
        for name in criteria[comp]:
            sub = stable[(stable["component"] == comp) & (stable["criterion"] == name)]
            allseeds = sorted(sub.loc[sub["n_seeds"] == sub["n_seeds_total"], "gene"])
            say(
                f"  {comp:6s} {name:15s} {len(allseeds):3d} genes"
                + (f": {allseeds}" if 0 < len(allseeds) <= 12 else "")
            )
    say("\nrecovery of the published genes, per criterion:")
    say(
        recovery[
            [
                "component",
                "criterion",
                "set_sizes",
                "mean_set_size",
                "n_true",
                "true_recovered_all_seeds",
                "true_never_recovered",
                "genes_in_all_seeds",
                "genes_in_union",
            ]
        ].to_string(index=False)
    )

    # ---- write everything ----------------------------------------------------------------
    rank_df.to_csv(out / "embedding_rank.csv", index=False)
    gauge_df.to_csv(out / "gauge_freedom.csv", index=False)
    pd.DataFrame(gauge_dim_rows).to_csv(out / "gauge_freedom_per_dim.csv", index=False)
    deg.to_csv(out / "decoder_null_degenerate.csv", index=False)
    sig_report.to_csv(out / "significance_per_dim.csv", index=False)
    recovery.to_csv(out / "recovery_numbers.csv", index=False)
    stable.to_csv(out / "stable_genes.csv", index=False)
    for comp in COMPONENTS:
        matching[comp][1].to_csv(out / f"dim_matching_{comp}.csv", index=False)
        matching_report(matching[comp][1]).to_csv(out / f"dim_matching_report_{comp}.csv", index=False)
        freq_all[comp].to_csv(out / f"consensus_frequency_{comp}.csv", index=False)
        pd.DataFrame({"gene": consensus[comp]["genes"]}).to_csv(out / f"consensus_genes_{comp}.csv", index=False)
    # JSON, not only CSV: a criterion that selected NOTHING for a seed contributes no CSV rows,
    # and that seed would then silently vanish from a reload -- turning a five-seed comparison
    # into a two-seed one. An empty set is a result and has to survive the round trip.
    (out / "per_seed_gene_sets.json").write_text(json.dumps(_jsonable(criteria), indent=1))
    pd.DataFrame(
        {"component": c, "criterion": k, "seed": s, "gene": g}
        for c in COMPONENTS
        for k, v in criteria[c].items()
        for s, gs in v.items()
        for g in gs
    ).to_csv(out / "per_seed_gene_sets.csv", index=False)
    # The full (gene, dim) tables are large on melton25 (979 x ~45 x 5 x 2), so they go out one
    # parquet-shaped csv per (component, seed) rather than one giant frame.
    tdir = out / "permutation_tables"
    tdir.mkdir(exist_ok=True)
    if not reuse_tables:
        for (comp, s), t in tables.items():
            t.to_csv(tdir / f"{comp}_seed{s}.csv.gz", index=False, compression="gzip")

    summary = {
        "dataset": dataset,
        "label": spec["label"],
        "settings": {
            "seeds": seeds_list,
            "reference": ref,
            "B": B,
            "alpha": alpha,
            "statistic": statistic,
            "fdr_scope": fdr_scope,
            "consensus_k": consensus_k,
            "random_state": random_state,
            "min_r": min_r,
        },
        "n_cells": int(ref_arrays.n_obs),
        "n_genes": int(len(ref_arrays.var_names)),
        "n_embed": int(ref_arrays.n_embed),
        "pool_topk": pool_topk,
        "pool_permutation": pool_perm,
        "cells_kept_of_h5ad": [int(keep.sum()), int(keep.size)] if keep is not None else None,
        "reused_tables": bool(reuse_tables),
        "n_consensus": {c: len(consensus[c]["genes"]) for c in COMPONENTS},
        "consensus_genes": {c: consensus[c]["genes"] for c in COMPONENTS},
        "consensus_per_dimension": {c: _jsonable(consensus[c]["per_dimension"]) for c in COMPONENTS},
        "true_genes": spec["true_genes"],
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    say(f"\nwrote {len(list(out.rglob('*')))} files to {out}  ({summary['runtime_seconds']}s total)")

    return {
        "reuse_tables": reuse_tables,
        "spec": spec,
        "summary": summary,
        "seed_arrays": sa,
        "X": X,
        "rank": rank_df,
        "gauge": gauge_df,
        "degenerate": deg,
        "tables": tables,
        "significance": sig_report,
        "matching": {c: matching[c][1] for c in COMPONENTS},
        "mapping": {c: matching[c][0] for c in COMPONENTS},
        "frequency": freq_all,
        "consensus": consensus,
        "criteria": criteria,
        "recovery": recovery,
        "stable_genes": stable,
        "pool": {"topk": pool_topk, "permutation": pool_perm},
    }


def evaluate_single_model_selection(
    dataset: str,
    *,
    alpha: float = 0.05,
    max_per_dim: int | None = None,
    out_root: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Validate the SINGLE-MODEL criterion against the other runs' top-N lists.

    This is the experiment the shipped function is judged by, and the only place the replicate
    seeds are used::

        for each seed s:                      # 's' plays "the one model the user has"
            selected  = permutation filter on seed s alone      (no other seed involved)
            yardstick = the OTHER seeds' top-N lists            (what 4a recorded per seed)
            score     = fraction of the other runs whose list contains each selected gene

    Leave-one-out over all five choices of ``s``, so the answer does not depend on which model
    happened to be trained first. The baseline is the same score for seed ``s``'s **top-K** set,
    i.e. exactly the published rule evaluated the same way -- so the two differ only in how seed
    ``s``'s genes were chosen, and both are read against the same chance level.

    No dimension matching anywhere: both sides are component-level gene sets.

    Reuses the permutation tables :func:`run` cached, so this costs seconds.

    Returns a dict of DataFrames, also written to ``<out>/single_model_evaluation*.csv``.
    """
    from src.loading_stability import agreement_with_other_runs, size_matched_sets

    spec = SPECS[dataset]
    out = Path(out_root or OUT_ROOT) / dataset
    summary = json.loads((out / "summary.json").read_text())
    seeds_list = list(summary["settings"]["seeds"])
    pool = int(summary["pool_topk"])
    say = print if verbose else (lambda *a, **k: None)

    sa = {s: load_seed_arrays(spec["arrays_dir"], s) for s in seeds_list}
    frac = pd.read_csv(out / "gene_frac.csv").set_index("gene")["frac"]
    candidates = set(frac.index[frac >= spec["min_frac"]].astype(str))

    rows, per_gene_rows = [], []
    for comp in COMPONENTS:
        # The yardstick: each run's own top-N selection, as recorded -- not recomputed.
        topk = {s: set(sa[s].program[comp]) for s in seeds_list}
        for s in seeds_list:
            others = {o: topk[o] for o in seeds_list if o != s}
            t = load_table(dataset, comp, s, out_root=out_root)
            t = t[t["gene"].isin(candidates)]

            selections = {"a_topk": set(topk[s])}
            if len(t):
                sig = significant_sets(t, alpha=alpha)
                selections["b_permutation"] = set().union(*sig.values()) if sig else set()
                # size-matched to this seed's own top-K set, so the only difference is the
                # ranking statistic
                target = max(1, round(len(topk[s]) / max(len(sig), 1)))
                cut = size_matched_sets(t, dict.fromkeys(sig, target), alpha=alpha)
                selections["c_size_matched"] = set().union(*cut.values()) if cut else set()
            else:
                selections["b_permutation"] = set()
                selections["c_size_matched"] = set()

            published = set(spec["true_genes"][comp]) & candidates
            for name, sel in selections.items():
                with warnings.catch_warnings():
                    # scipy's hypergeom skewness divides by (M - N); a selection taking the WHOLE
                    # pool makes that 0. Only unused higher moments are affected.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a = agreement_with_other_runs(sel, others, pool=pool, published=spec["true_genes"][comp])
                # Second, INDEPENDENT yardstick: the published gene list. The other runs' top-N
                # lists are produced by the very rule under test, and they agree with each other
                # at or below chance -- so they cap this experiment near chance whatever the
                # selection does. The published list is biology, not another unstable run.
                hit = len(sel & published)
                rec_chance = len(sel) / pool if pool else np.nan
                rows.append(
                    {
                        "component": comp,
                        "held_out_model": s,
                        "criterion": name,
                        "n_selected": a["n_selected"],
                        "other_set_sizes": a["other_set_sizes"],
                        "mean_recall": a["mean_recall"],
                        "chance_recall": a["chance_recall"],
                        "fold": a["fold"],
                        "n_in_all_other_runs": a["n_in_all_runs"],
                        "n_in_no_other_run": a["n_in_no_run"],
                        "median_pairwise_p": float(a["pairwise"]["p_value"].median()),
                        "n_published_selected": hit,
                        "n_published_in_pool": len(published),
                        "published_recall": hit / len(published) if published else np.nan,
                        "published_recall_chance": rec_chance,
                        "published_recall_fold": (
                            (hit / len(published)) / rec_chance
                            if published and rec_chance and rec_chance > 0
                            else np.nan
                        ),
                        "published_precision": hit / len(sel) if sel else np.nan,
                    }
                )
                g = a["per_gene"].copy()
                g.insert(0, "criterion", name)
                g.insert(0, "held_out_model", s)
                g.insert(0, "component", comp)
                per_gene_rows.append(g)

    detail = pd.DataFrame(rows)
    per_gene = pd.concat(per_gene_rows, ignore_index=True) if per_gene_rows else pd.DataFrame()
    agg = (
        detail.groupby(["component", "criterion"])
        .agg(
            n_models=("held_out_model", "size"),
            mean_n_selected=("n_selected", "mean"),
            mean_recall=("mean_recall", "mean"),
            sd_recall=("mean_recall", "std"),
            chance_recall=("chance_recall", "mean"),
            fold=("fold", "mean"),
            mean_in_all_other=("n_in_all_other_runs", "mean"),
            mean_in_no_other=("n_in_no_other_run", "mean"),
            published_recall=("published_recall", "mean"),
            published_recall_chance=("published_recall_chance", "mean"),
            published_recall_fold=("published_recall_fold", "mean"),
        )
        .reset_index()
    )
    detail.to_csv(out / "single_model_evaluation_per_model.csv", index=False)
    agg.to_csv(out / "single_model_evaluation.csv", index=False)
    per_gene.to_csv(out / "single_model_evaluation_per_gene.csv", index=False)

    say(f"\n{dataset}: single-model selection scored against the OTHER runs' top-N lists")
    say(
        f"  (leave-one-out over {len(seeds_list)} models; pool {pool} genes; "
        f"chance recall = mean other-set size / pool)"
    )
    say(agg.round(3).to_string(index=False))
    say("\n  `fold` is against the OTHER RUNS' top-N lists, whose own mutual overlap is at or")
    say("  below chance -- so that column is capped near 1 by the yardstick, not the method.")
    say("  `published_recall_fold` is against the published gene list: an independent yardstick.")
    return {"per_model": detail, "aggregate": agg, "per_gene": per_gene, "pool": pool}


def load_table(dataset: str, component: str, seed: int, *, out_root: Path | None = None) -> pd.DataFrame:
    """One (component, seed) permutation table as :func:`run` wrote it.

    These hold ``loading`` (the tested empirical standardized loading) next to
    ``loading_decoder`` (what ``gene_loadings`` stored), so the agreement between the two can be
    read off without re-reading the expression matrix -- which on melton25 is a 1.5 GB decompress
    from Lustre.
    """
    p = Path(out_root or OUT_ROOT) / dataset / "permutation_tables" / f"{component}_seed{seed}.csv.gz"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found -- run({dataset!r}) first.")
    return pd.read_csv(p)


def load_run(dataset: str, *, out_root: Path | None = None) -> dict:
    """Read back what :func:`run` wrote, without recomputing. Raises if it was never run."""
    out = Path(out_root or OUT_ROOT) / dataset
    if not (out / "summary.json").exists():
        raise FileNotFoundError(f"{out / 'summary.json'} not found -- run(dataset) first (it writes everything here).")
    res: dict[str, Any] = {"summary": json.loads((out / "summary.json").read_text())}
    for name, path in [
        ("rank", "embedding_rank.csv"),
        ("gauge", "gauge_freedom.csv"),
        ("gauge_per_dim", "gauge_freedom_per_dim.csv"),
        ("degenerate", "decoder_null_degenerate.csv"),
        ("significance", "significance_per_dim.csv"),
        ("recovery", "recovery_numbers.csv"),
        ("stable_genes", "stable_genes.csv"),
        ("per_seed_gene_sets", "per_seed_gene_sets.csv"),
    ]:
        res[name] = pd.read_csv(out / path)
    res["matching"] = {c: pd.read_csv(out / f"dim_matching_{c}.csv") for c in COMPONENTS}
    res["frequency"] = {c: pd.read_csv(out / f"consensus_frequency_{c}.csv") for c in COMPONENTS}
    res["consensus"] = {
        c: {"genes": pd.read_csv(out / f"consensus_genes_{c}.csv")["gene"].astype(str).tolist()} for c in COMPONENTS
    }
    res["criteria"] = {
        c: {k: {int(s): list(gs) for s, gs in v.items()} for k, v in crit.items()}
        for c, crit in json.loads((out / "per_seed_gene_sets.json").read_text()).items()
    }
    res["pool"] = {"topk": res["summary"]["pool_topk"], "permutation": res["summary"]["pool_permutation"]}
    return res
