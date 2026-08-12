## Quick orientation for AI coding helpers

This project implements the InterScale model (graph-transformer for spatial transcriptomics). Use these notes to be productive quickly — they are intentionally concise and reference code locations to inspect for details.

- **Big picture:** The Python package lives under `GT-long-range-niches/src/InterScale` (also mirrored in this repo under `src/InterScale`). Main subsystems:
  - `config/` – default YAML config fragments (dataset, model, optimizer, wandb, etc).
  - `train/` and `main.py` – training entrypoints; scripts call `python src/InterScale/main.py --cfg <yaml>`.
  - `model/`, `nn/`, `module/`, `tl/` – implementation of model components and layers.
  - `evaluation` / `eval` – evaluation utilities (e.g. gene-level analyses such as `gene_ranke_analysis`). Example file: `src/InterScale/evaluation/_gene_set_covariance.py`.

- **Config-first workflow:** Configuration is YAML-driven. Example configs live in `configs/` and `src/config_files/` (see `InterScale_example.yaml`). Typical command to train:

  `python src/InterScale/main.py --cfg "path/to/config.yaml" --model_type "CombinedModel"`

  - The YAML you pass overrides defaults in `InterScale/config/`. Look at `config/*` for parameter names.
  - Paths (adata, results) must be set in your YAML; defaults are not globally absolute.

- **Environment & dependencies:** The project uses PyTorch, PyTorch Lightning, and PyG (torch-geometric). README has GPU and `enroot` container setup; common install steps:

  - Create env: `mamba create -n GT_long_range python=3.11 && mamba activate GT_long_range`
  - Install: `pip install -e .` (then PyG-related wheels, `pytorch-lightning`, `wandb`, `geome` as needed)

- **Run scripts and CI-cues:** Helper scripts at repo root in `GT-long-range-niches/` include `run.sh`, `run_gpu.sh`, `run_cv.sh`, `run_lrz.sh` — inspect these for experiment invocation patterns (how configs are layered and how hyperparameter sweeps are launched).

- **Testing & small-data fixtures:** Tests live in `tests/`. Use `pytest` to run unit tests. Small fixtures live under `tests/_data` and `tests/conftests.py` defines fixtures to mimic datasets — inspect these when modifying data-loading code.

- **Patterns & conventions to follow:**
  - Keep config keys stable: prefer adding new keys to `config/*` and provide examples in `src/config_files/`.
  - Training code uses checkpointing (`.ckpt`) via PyTorch Lightning — load/save patterns follow Lightning conventions.
  - Evaluation functions expect models trained for regression (gene prediction) for some gene-level analyses.

- **Files to check for domain knowledge / examples:**
  - `GT-long-range-niches/README.md` — high-level workflows and environment commands.
  - `src/config_files/InterScale_example.yaml` — minimal config required to run experiments.
  - `src/InterScale/main.py` and `GT-long-range-niches/run_gpu.sh` — concrete invocation examples.
  - `src/InterScale/evaluation/_gene_set_covariance.py` — example of evaluation code and expected inputs.

- **When you modify code:**
  - Run the relevant unit tests in `tests/` that cover the area (check `tests/test_transformer_cls_embedding.py` and nearby tests).
  - For dataset changes, use the small fixtures under `tests/_data` and update `tests/conftests.py` where appropriate.

- **Notes for AI agents:**
  - Prefer small, targeted edits (update a function, add a test) rather than broad refactors.
  - If a change affects config keys, update `src/config_files/InterScale_example.yaml` and add a small test demonstrating the new key.
  - For performance-sensitive changes, mention testing on a small dataset (use `tests/_data`) before suggesting large-scale runs.

If anything here is unclear or you want more depth in a specific area (data pipeline, model components, or experiment scripts), tell me which area and I will expand these notes.
