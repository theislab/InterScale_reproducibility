import argparse
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scanpy as sc
import torch
from scipy.sparse import issparse
from sklearn.preprocessing import normalize

# InterScale imports
import InterScale as interscale
from InterScale.config import load_config
from InterScale.geome_dataloader import GraphAnnDataModule
from InterScale.tl import prepare_geome_dataset, remove_zero_expression_cells


def set_full_reproducibility(seed: int = 42) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False


def get_ram_mb() -> float:
	"""Return current process RAM usage in MB."""
	process = psutil.Process()
	return process.memory_info().rss / (1024 * 1024)


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run InterScale single seed evaluation."
	)
	parser.add_argument(
		"--seed", type=int, required=True, help="Random seed for reproducibility"
	)
	parser.add_argument(
		"--config",
		type=str,
		default="config_files/melton_jimenez_25/InterScale_Cosmx_Nrec_globalPCA.yaml",
		help="Path to InterScale config YAML file",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="results_seeds",
		help="Directory where single-seed results CSVs will be stored",
	)
	parser.add_argument(
		"--working_dir",
		type=str,
		default="/beegfs/scratch/ric.cirillo/dimarco.federico/PHD_gnn/INTERACTION/InterScale_reproducibility",
		help="Base working directory",
	)
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="meltonjimenez_25",
		help="Base working directory",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	# Change working directory
	if os.path.exists(args.working_dir):
		os.chdir(args.working_dir)

	# Set seed
	set_full_reproducibility(args.seed)

	# Prepare output path
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	out_csv = output_dir / f"IGP_{args.dataset_name}_metrics_seed_{args.seed}.csv"

	# 1. Load configuration
	cfg = load_config(args.config)
	cfg.defrost()
	cfg.optim.seed = args.seed
	cfg.freeze()

	# 2. Load and process AnnData
	adata = sc.read_h5ad(cfg.dataset.h5ad_data)
	adata = remove_zero_expression_cells(adata)

	interscale.model.GlobalModel._setup_anndata(
		adata=adata,
		prediction_task=cfg.dataset.prediction_task,
		layer_key=cfg.dataset.layer_key,
		sample_key_list=cfg.dataset.sample_key,
		prediction_obs=cfg.dataset.prediction_obs,
		group_key=cfg.dataset.group_label,
	)

	adata.obsp["adjacency_matrix_connectivities"] = adata.obsp["spatial_connectivities"]
	pyg_data_list, _ = prepare_geome_dataset(adata, cfg)

	# 3. Setup DataModule
	dm = GraphAnnDataModule(
		datas=pyg_data_list,
		num_workers=10,
		batch_size=int(cfg.dataset.batch_size),
		pct_mask_nodes=cfg.dataset.pct_mask_nodes,
		learning_type="node",
	)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Running seed {args.seed} on device: {device}")

	model = interscale.model.GlobalModel(adata, cfg=cfg)

	# Pre-train memory tracking & cleanup
	gc.collect()
	if device.type == "cuda":
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats(device)
		torch.cuda.synchronize()

	ram_start_train = get_ram_mb()
	t_train_start = time.perf_counter()

	# 4. Model Training
	model.train(
		max_epochs=40,
		datamodule=dm,
		early_stopping=True,
		enable_progress_bar=True,
		wandb_use=False,
	)

	if device.type == "cuda":
		torch.cuda.synchronize()

	t_train_end = time.perf_counter()

	# 5. Metrics Calculation
	train_runtime_sec = t_train_end - t_train_start
	train_vram_peak_gb = (
		(torch.cuda.max_memory_allocated(device) / (1024**3))
		if device.type == "cuda"
		else 0.0
	)
	train_ram_peak_gb = max(ram_start_train, get_ram_mb()) / 1024.0
	
	result = model.get_model_output(adata)

	X_orig = result.X
	X_recon = result.layers["_y_pred_global"]

	X_orig_norm = normalize(X_orig, axis=1)
	X_recon_norm = normalize(X_recon, axis=1)

	if issparse(X_orig_norm):
		row_wise_similarities = np.array(
			X_orig_norm.multiply(X_recon_norm).sum(axis=1)
		).ravel()
	else:
		row_wise_similarities = np.sum(X_orig_norm * X_recon_norm, axis=1)

	mean_cosine_similarity = float(np.mean(row_wise_similarities))
	epochs=len(model.history_.history)-1
	# Save metrics to single CSV
	res_df = pd.DataFrame(
		[
			{
				"model": "Interscale_Global_PCA",
				"seed": args.seed,
				"dataset": args.dataset_name,
				"cell": adata.n_obs,
				"epochs": epochs,
				"train_runtime_sec": train_runtime_sec,
				"train_vram_peak_gb": train_vram_peak_gb,
				"train_ram_peak_gb": train_ram_peak_gb,
				"cosine_sim": mean_cosine_similarity,
			}
		]
	)

	res_df.to_csv(out_csv, index=False)
	print(f"Results for seed {args.seed} successfully saved to {out_csv}")


if __name__ == "__main__":
	main()
