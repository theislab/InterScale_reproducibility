"""Capacity diagnostic: can InterScale overfit ONE graph?

The sweeps say the model barely beats predicting the mean. Two very different causes fit that,
and no amount of hyperparameter search separates them:

  * the model CANNOT represent the data (bottleneck too narrow, decoder too weak, optimisation
    broken) -- then it will fail even when asked to memorise a single graph, and every result
    downstream is about the model;
  * the model CAN fit data but not THIS task (predicting a cell from its neighbours does not
    generalise) -- then it will memorise one graph easily and the sweep result is about the task.

Three regimes on one graph, no train/val split -- fit is measured on the data being fitted, and
overfitting is the desired outcome:

  A  local autoencoder, MASKING OFF -- input is the full expression, target is the same
     expression, all nodes supervised, local decoder only. Pure capacity: can GCN -> n_embed ->
     Linear reproduce what it was given? A model that fails here cannot represent the data at all.
  B  full model autoencoder, MASKING OFF -- same, but through local -> transformer -> global
     decoder. Adds the global path to the capacity question.
  C  the REAL task on one graph, fixed mask -- masked cells zeroed in the input and supervised
     as targets, with the SAME mask every step so memorisation is possible. Separates "cannot
     fit the task" from "cannot generalise the task".

Each regime runs at n_embed 16 (the published config) and 128 (the sweep's best), because the
88-genes-through-16-dimensions bottleneck is the leading suspect.

    python scripts/overfit_single_graph.py --dataset legnini23
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import scanpy as sc
import torch
from torch_geometric.data import Batch

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*spatial_neighbors.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*n_neighs.*")

REGISTRY = "/home/icb/francesca.drummer/1-Projects/A3_InterScale/interscale/config_files/registry.yaml"


def fit_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
    """MSE, mean per-gene Pearson and R2 of a fit, computed on the fitted data itself."""
    p = y_pred.detach().float()
    t = y_true.detach().float()
    mse = float(((p - t) ** 2).mean())

    pc = p - p.mean(0, keepdim=True)
    tc = t - t.mean(0, keepdim=True)
    den = pc.norm(dim=0) * tc.norm(dim=0)
    r = torch.where(den > 0, (pc * tc).sum(0) / den, torch.full_like(den, float("nan")))

    ss_res = ((t - p) ** 2).sum(0)
    ss_tot = (tc**2).sum(0)
    r2 = torch.where(ss_tot > 0, 1 - ss_res / ss_tot, torch.full_like(ss_tot, float("nan")))
    return {"mse": mse, "pearson": float(np.nanmean(r.cpu().numpy())), "r2": float(np.nanmean(r2.cpu().numpy()))}


def build_module(dataset, n_embed, device):
    """A fresh model at the requested embedding width, plus the graphs for this dataset."""
    import interscale
    from interscale.config.registry import resolve_config
    from interscale.tl import prepare_geome_dataset, remove_zero_expression_cells

    cfg = resolve_config(dataset, "node_reg", registry_path=REGISTRY)
    cfg.defrost()
    cfg.model.n_embed = int(n_embed)
    cfg.dataset.mask_percentage = 0.1
    cfg.freeze()

    adata = remove_zero_expression_cells(sc.read_h5ad(cfg.dataset.h5ad_data))
    interscale.model.CombinedModel._setup_anndata(
        adata=adata, prediction_task=cfg.dataset.prediction_task, layer_key=cfg.dataset.layer_key,
        sample_key_list=cfg.dataset.sample_key, prediction_obs=cfg.dataset.prediction_obs, view_registry=False,
    )
    model = interscale.model.CombinedModel(adata, cfg=cfg)
    pyg, _ = prepare_geome_dataset(adata, cfg)
    return cfg, model.module.to(device), pyg


def pick_graph(pyg, device):
    """Largest TRAINING graph -- the hardest single-graph target, so a pass is not a small-n win.

    prepare_geome_dataset returns [train, val] or [train, val, test], each a list of Data, not a
    flat list of graphs.
    """
    train_graphs = pyg[0]
    data = max(train_graphs, key=lambda d: d.num_nodes)
    return Batch.from_data_list([data]).to(device)


def run_regime(module, batch, regime, steps, lr, device, log_every):
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    x = batch.x

    if regime == "B":
        # common_step_local_to_global reads batched_data.mask whenever masking is enabled and the
        # call is not an eval step. Masking-off still has to supply one, so: nothing masked.
        batch.mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=device)

    if regime == "C":
        # One fixed mask, drawn once and reused every step: with a fresh draw each step the
        # targets change and memorisation is impossible by construction.
        g = torch.Generator(device="cpu").manual_seed(0)
        mask = (torch.rand(batch.num_nodes, generator=g) < 0.1).to(device)
        if not mask.any():
            mask[0] = True
        batch.mask = mask

    module.train()
    history = []
    for step in range(steps + 1):
        opt.zero_grad()

        if regime == "A":
            emb = module.local_module.forward(x, batch.edge_index)
            emb = emb["embedding"] if isinstance(emb, dict) else emb
            y_pred = module.local_module.decoder.forward(emb)
            y_true = x
        elif regime == "B":
            _, global_embedding, src_padding_mask, pad_index_nodes, _, _ = module.forward(batch)
            y_pred = module.predict_global(global_embedding, src_padding_mask, "node")
            # predict_global returns one row per padded token; keep the real cells, in order.
            idx = torch.as_tensor(pad_index_nodes[0], device=y_pred.device)
            y_pred = y_pred[: len(idx)]
            y_true = x[idx]
        elif regime == "C":
            _, _, y_pred, y_true, _, _ = module._common_step(batch, "regression", "node")
        else:
            raise ValueError(regime)

        loss = torch.nn.functional.mse_loss(y_pred, y_true)
        if step < steps:
            loss.backward()
            opt.step()

        if step % log_every == 0 or step == steps:
            m = fit_metrics(y_pred, y_true)
            history.append((step, m))
            print(f"    step {step:5d}  mse {m['mse']:.4f}  pearson {m['pearson']:.4f}  R2 {m['r2']:.4f}",
                  flush=True)
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="legnini23")
    ap.add_argument("--n-embed", type=int, nargs="+", default=[16, 128])
    ap.add_argument("--regimes", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=250)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    titles = {
        "A": "A  local autoencoder, masking OFF (all nodes supervised, input = target)",
        "B": "B  full model autoencoder, masking OFF (local -> transformer -> global decoder)",
        "C": "C  real task, ONE fixed mask on one graph (memorisation possible)",
    }

    summary = []
    for n_embed in args.n_embed:
        cfg, module, pyg = build_module(args.dataset, n_embed, device)
        batch = pick_graph(pyg, device)
        n_genes = int(batch.x.shape[1])

        # Linear ceiling for this width: PCA-k reconstruction of the same matrix. A k-dimensional
        # LINEAR autoencoder cannot beat it, so it separates "the bottleneck is too narrow" (the
        # model approaches this number and it is low) from "optimisation is failing" (the model
        # sits well below a ceiling that is high).
        Xg = batch.x.detach().cpu().numpy().astype(np.float64)
        Xc = Xg - Xg.mean(0, keepdims=True)
        k = min(n_embed, Xc.shape[1])
        _, sv, _ = np.linalg.svd(Xc, full_matrices=False)
        pca_r2 = float((sv[:k] ** 2).sum() / (sv**2).sum())
        print(f"\n########## n_embed={n_embed}  graph: {batch.num_nodes} cells x {n_genes} genes "
              f"({args.dataset}) ##########")
        print(f"  PCA-{k} reconstruction R2 on this graph (linear ceiling for this width): {pca_r2:.4f}")
        summary.append((n_embed, f"PCA-{k}", {"mse": float("nan"), "pearson": float("nan"), "r2": pca_r2}))
        for regime in args.regimes:
            print(f"\n  {titles[regime]}")
            # A fresh module per regime: otherwise regime B starts from whatever A left behind.
            _, module, _ = build_module(args.dataset, n_embed, device)
            hist = run_regime(module, batch.clone(), regime, args.steps, args.lr, device, args.log_every)
            summary.append((n_embed, regime, hist[-1][1]))

    print("\n\n================ SUMMARY (fit on the data being fitted) ================")
    print(f"{'n_embed':>8} {'regime':>7} {'mse':>9} {'pearson':>9} {'R2':>9}")
    for n_embed, regime, m in summary:
        print(f"{n_embed:>8} {regime:>7} {m['mse']:>9.4f} {m['pearson']:>9.4f} {m['r2']:>9.4f}")
    print("\nReading it: R2 near 1 in A/B means the architecture can represent the data and the")
    print("sweep result is about the task, not capacity. R2 near 0 in A means it cannot.")


if __name__ == "__main__":
    main()
