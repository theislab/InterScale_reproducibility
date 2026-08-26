mkdir -p logs
mkdir -p results_seeds


SEEDS=(44 42 40)




DATASET=legnini_23

for CURRENT_SEED in "${SEEDS[@]}"; do
	echo "Running with Seed: $CURRENT_SEED"

	python run_dual.py \
		--seed $CURRENT_SEED \
		--config config_files/${DATASET}/InterScale_Legnini_Nrec_Modular.yaml \
		--dataset_name ${DATASET} \
		--output_dir results_seeds

		python run_Gpca.py \
		--seed $CURRENT_SEED \
		--config config_files/${DATASET}/InterScale_Legnini_Nrec_globalPCA.yaml \
		--dataset_name ${DATASET} \
		--output_dir results_seeds
	done