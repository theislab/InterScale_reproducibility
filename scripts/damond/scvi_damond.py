import scanpy as sc
import squidpy as sq
import numpy as np
import pandas as pd


import InterScale as interscale
from InterScale.config import load_config
from InterScale.tl import prepare_geome_dataset, check_and_update_cfg
from InterScale.geome_dataloader import GraphAnnDataModule
#from InterScale.eval.gene_rank_analysis import predict_gene_r2, gene_rank_analysis
from InterScale.tl import prepare_a2d_dataset
from InterScale.evaluation import scale_cls_by_sample

from pathlib import Path
import torch
import scvi

def run_scvi_on_damond():
    adata_pancreas = sc.read_h5ad('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/imc_pancreas_pp.h5ad')
    scvi.model.SCVI.setup_anndata(adata_pancreas, layer='pct-norm-99', batch_key="case")
    model = scvi.model.SCVI(adata_pancreas, n_layers=2, n_latent=30, gene_likelihood="nb")
    model.train()
    SCVI_LATENT_KEY = "X_scVI"
    adata_pancreas.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()
    sc.pp.neighbors(adata_pancreas, use_rep=SCVI_LATENT_KEY, key_added='scvi')
    sc.tl.leiden(adata_pancreas, key_added='scvi', neighbors_key='scvi')
    adata_pancreas.write('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/imc_pancreas_pp.h5ad')
    sc.tl.umap(adata_pancreas, min_dist=0.3, neighbors_key='scvi')
    adata_pancreas.write('/lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/imc_pancreas_pp_umap.h5ad')
    print("Wrote scvi results to: /lustre/groups/ml01/projects/2024_spatial_long_range_GT_francesca.drummer/data/imc_pancreas_pp.h5ad")

if __name__ == "__main__":
    run_scvi_on_damond()
      