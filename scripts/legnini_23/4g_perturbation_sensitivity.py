"""Legnini - single-model *input perturbation* sensitivity of the latent dimensions.

Reviewer question this answers: can a sensitivity score built from ONE model and ONE seed tell
which genes would have come up again under a different seed? This script produces the one family
of candidate scores that cannot be computed from the cached seed arrays, because it needs the
encoder: perturb one gene's column in the model's input, re-run the forward pass, and record how
far every latent dimension moved.

    dZ[f, e] = || Z'[:, e] - Z[:, e] || / || Z[:, e] - mean ||        (relative L2 shift)

`4h_sensitivity_benchmark.py` reads the npz written here, turns dZ into per-gene scores, and
scores them against cross-seed recurrence. Nothing is retrained; the checkpoints are the seed
sweep's, loaded read-only.

.. important::
   **This perturbation provably cannot move the published gene sets, and that is the point of
   running it.** The Fig. 3E/3F rule ranks genes within a dimension by
   ``|S[f,e]| * sd(X_f)`` with ``S[f,e] = W[f,e] * sd(Z_e) / sd(X_f)``, so the two ``sd(X_f)``
   factors cancel and the ranking is exactly the ranking of ``|W[f,e]|`` -- frozen decoder
   weights, no data. Permuting a gene's column leaves ``sd(X_f)`` unchanged anyway, and the only
   surviving data-dependent factor, ``sd(Z_e)``, is a per-dimension constant that cannot reorder
   genes inside a dimension. `loading_sensitivity.prove_selection_is_data_independent` checks
   this numerically.

   So dZ is not a perturbation *of the selection*. It is a measurement of a different, genuinely
   data-dependent thing the selection ignores: whether the ENCODER actually routes gene f into
   dimension e. Whether that predicts cross-seed recurrence is the empirical question.

Two perturbations, both applied to every cell of one gene at a time:

``permute``
    Shuffle the gene's values across cells. Destroys the gene's pairing with everything
    (its neighbours, its cell) while preserving its marginal distribution exactly, so a large
    dZ cannot be explained by the gene's expression level or spread. Stochastic, hence
    ``--repeats``.
``mask``
    Overwrite with ``tl.masking.MASK_VALUE`` (-1). This is the corruption the model was trained
    to be robust to, so it is the in-distribution ablation -- the model has seen this exact
    input pattern ~30% of the time per cell during training, which ``permute`` has not.

Usage
-----
    python scripts/legnini_23/4g_perturbation_sensitivity.py                    # all 5 seeds
    python scripts/legnini_23/4g_perturbation_sensitivity.py --seeds 42 --repeats 1

Run it inside the interscale apptainer container (scripts/legnini_23/run_seed_stability.sh shows
the invocation); it is CPU-only and takes ~4 min per seed and perturbation on 8 cores.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import torch

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

LUSTRE = Path("/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer")
REGISTRY = Path("/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale/config_files/registry.yaml")

# The masking-granularity arms of the seed sweep; keys match `4a_seed_stability_extract.py`'s
# SPECS so the arrays this writes land next to the ones that script wrote.
ARMS = {
    "cell": dict(task="node_reg", subdir="seed_stability_cell",
                 overrides={"optim.n_epochs": 300}),
    "gene": dict(task="node_reg_genemask", subdir="seed_stability_gene",
                 overrides={"optim.n_epochs": 300, "dataset.mask_percentage": 0.25}),
}
SEEDS = [40, 41, 42, 43, 44]


def build_cfg(seed: int, arm: dict):
    """Registry config pinned to one seed's checkpoint, as the seed sweep trained it."""
    from interscale.config.registry import resolve_config
    from interscale.config.sweep import apply_sweep_config

    cfg = resolve_config("legnini23", arm["task"], registry_path=str(REGISTRY))
    if arm["overrides"]:
        cfg, _ = apply_sweep_config(
            cfg, "hyperparmeter", arm["overrides"], model_type="CombinedModel",
            sweep_params=list(arm["overrides"]),
        )
    cfg.defrost()
    cfg.optim.seed = int(seed)
    cfg.model.save = str(LUSTRE / "results" / "legnini23_revision" / arm["subdir"]) + os.sep
    cfg.freeze()
    return cfg


def load_graphs(adata, cfg):
    """The per-sample PyG graphs, built once and reused for every perturbation.

    `CombinedModel.get_model_output` rebuilds these on every call and then scatters the output
    through pandas `.loc`, which is why it is far too slow to call 88 times. The graphs depend
    only on the spatial coordinates and the config, not on the perturbed expression, so they are
    built once here and only `batch.x` is overwritten per gene.
    """
    from interscale.tl.geome_utils import prepare_a2d_dataset

    a2d = prepare_a2d_dataset(cfg)
    pyg, _ = list(a2d(adata))
    # batch.obs_names carries the 1-based row ids assigned below; this is the permutation that
    # maps concatenated per-graph output back onto adata's row order.
    order = np.concatenate([b.obs_names.numpy().astype(int) - 1 for b in pyg])
    assert np.array_equal(np.sort(order), np.arange(adata.n_obs)), "graphs do not partition the cells"
    return pyg, order


@torch.no_grad()
def embed(model, pyg, x_override=None):
    """Local and global embeddings for every cell, in graph-concatenation order.

    ``x_override`` is a callable ``batch -> x`` used to swap in a perturbed input. The global
    side deliberately does NOT go through `TransformerNodeEncoderHook.evaluate`: that method
    additionally runs `generate_relevance`, which back-propagates to build the attention
    relevance matrix. It is the expensive part of inference, it needs grad enabled, and nothing
    here uses its output. `common_step_local_to_global` + `forward` is the same embedding.
    """
    gm = model.module.global_module
    loc, glo = [], []
    for b in pyg:
        x = b.x if x_override is None else x_override(b)
        le = model.module.local_module.forward(x, b.edge_index)
        b.batch = torch.zeros(len(b.obs_names))
        tin, spm, _pad, am = gm.common_step_local_to_global(b, le, eval_step=True)
        tout, spm, _ = gm.forward(tin, spm, am, register_hook=False)
        loc.append(le)
        glo.append(tout[:-1].squeeze(1))  # drop the CLS token appended at the end
    return torch.cat(loc).numpy(), torch.cat(glo).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="cell", choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--repeats", type=int, default=3, help="permutation draws per gene")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import interscale
    from interscale.tl import remove_zero_expression_cells
    from interscale.tl.masking import MASK_VALUE
    from interscale.tl.utils import set_full_reproducibility

    arm = ARMS[args.arm]
    models_dir = LUSTRE / "results" / "legnini23_revision" / arm["subdir"]
    out_dir = models_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg0 = build_cfg(args.seeds[0], arm)
    adata = sc.read_h5ad(cfg0.dataset.h5ad_data)
    adata = remove_zero_expression_cells(adata)
    # get_model_output renumbers obs_names to 1..N and the graph builder carries those ids
    # through batch.obs_names; do the same so `order` below is meaningful.
    adata.obs_names = [str(i) for i in range(1, adata.n_obs + 1)]
    genes = np.asarray([str(g) for g in adata.var_names])
    G = len(genes)
    print(f"{adata.n_obs} cells x {G} genes, arm '{args.arm}'", flush=True)

    for seed in args.seeds:
        out = out_dir / f"seed{seed}_perturbation.npz"
        if out.exists() and not args.overwrite:
            print(f"seed {seed}: {out.name} exists, skipping (--overwrite to redo)", flush=True)
            continue

        cfg = build_cfg(seed, arm)
        interscale.model.CombinedModel._setup_anndata(
            adata=adata, prediction_task=cfg.dataset.prediction_task,
            layer_key=cfg.dataset.layer_key, sample_key_list=cfg.dataset.sample_key,
            prediction_obs=cfg.dataset.prediction_obs, view_registry=False,
        )
        set_full_reproducibility(40)  # matches the notebook's repro_seed; nothing stochastic follows
        model = interscale.model.CombinedModel.load(
            str(models_dir), adata, cfg, local_component=True, global_component=True
        )
        model.module.eval()

        pyg, order = load_graphs(adata, cfg)
        inv = np.argsort(order)  # graph order -> adata row order

        Zl0, Zg0 = embed(model, pyg)
        base = {"local": Zl0, "global": Zg0}

        # Agreement with the embedding `4a_seed_stability_extract.py` stored. It is NOT exact,
        # and the reason matters:
        #
        #   `BaseModel.load()` never calls `.eval()` and `get_model_output` does not either, so
        #   every published embedding was produced with the module still in TRAIN mode -- i.e.
        #   with dropout active (dropout_local 0.1, dropout_global 0.1). `model.module.eval()`
        #   above turns that off, which is required here: with dropout on, the gene-to-gene
        #   differences in dZ would be swamped by the noise the forward pass injects on its own.
        #
        # Measured on seed 42 (mean Pearson r over the 16 dimensions, dropout-off vs stored):
        # local 0.997, global 0.853, and std(Z_global) is ~15-20% smaller with dropout off.
        # The local side is effectively unaffected; the global side is not, because the noise
        # compounds through the transformer. That is a property of the stored arrays, not of
        # this script -- see `loading_sensitivity.dropout_report`.
        ref_npz = out_dir / f"seed{seed}_arrays.npz"
        if ref_npz.exists():
            ref = np.load(ref_npz, allow_pickle=True)
            for k in base:
                a, b_ = base[k][inv], np.asarray(ref[f"Z_{k}"], np.float64)
                r = float(np.mean([np.corrcoef(a[:, e], b_[:, e])[0, 1]
                                   for e in range(a.shape[1])]))
                print(f"  seed {seed} {k}: mean per-dim r vs stored (dropout-on) embedding "
                      f"{r:.4f}; sd ratio {np.mean(a.std(0) / b_.std(0)):.3f}", flush=True)
                assert r > 0.5, (
                    f"{k} embedding barely correlates with {ref_npz.name} (r={r:.3f}); that is "
                    "more than dropout can explain -- the forward path is wrong"
                )
        # Denominator of the relative shift: the dimension's own spread, so a dimension with a
        # large constant offset does not look insensitive purely because its norm is large.
        denom = {k: np.linalg.norm(v - v.mean(0), axis=0) + 1e-12 for k, v in base.items()}

        res = {f"dZ_{p}_{k}": np.zeros((G, base[k].shape[1]))
               for p in ("permute", "mask") for k in base}
        rng = np.random.default_rng(args.rng_seed + seed)

        for tag in ("mask", "permute"):
            reps = 1 if tag == "mask" else args.repeats
            t0 = time.time()
            for f in range(G):
                acc = {k: np.zeros(base[k].shape[1]) for k in base}
                for _ in range(reps):
                    if tag == "mask":
                        def ov(b, f=f):
                            x = b.x.clone()
                            x[:, f] = MASK_VALUE
                            return x
                    else:
                        # One permutation per graph: the graphs are separate samples, and a
                        # global permutation would additionally shuffle cells across slides.
                        perms = {i: torch.as_tensor(rng.permutation(b.x.shape[0]))
                                 for i, b in enumerate(pyg)}
                        pos = {id(b): i for i, b in enumerate(pyg)}

                        def ov(b, f=f, perms=perms, pos=pos):
                            x = b.x.clone()
                            x[:, f] = x[perms[pos[id(b)]], f]
                            return x
                    Zl, Zg = embed(model, pyg, ov)
                    for k, Z in (("local", Zl), ("global", Zg)):
                        acc[k] += np.linalg.norm(Z - base[k], axis=0) / denom[k]
                for k in base:
                    res[f"dZ_{tag}_{k}"][f] = acc[k] / reps
                if f % 20 == 0:
                    print(f"  seed {seed} {tag}: gene {f}/{G} ({time.time() - t0:.0f}s)", flush=True)
            print(f"  seed {seed} {tag}: done in {time.time() - t0:.0f}s", flush=True)

        np.savez_compressed(
            out, var_names=genes, order=order, inv=inv,
            Z_local=base["local"][inv], Z_global=base["global"][inv],
            repeats=args.repeats, **res,
        )
        print(f"seed {seed}: wrote {out}", flush=True)


if __name__ == "__main__":
    main()
