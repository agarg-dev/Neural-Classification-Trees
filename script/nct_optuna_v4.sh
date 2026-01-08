#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/optuna_v4/nct_package_output_tuning_v4_%j.log
#SBATCH --error=logs/optuna_v4/nct_package_error_tuning_v4_%j.log

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1

# Load conda module and activate environment
module load conda
conda activate nct_new
dataset="waterbird"

# Run Optuna hyperparameter tuning V4 (expanded search space)
# New hyperparameters: node_balanced_sampling, routing_strategy, model_selection_criterion
python ../src/nct_package_optuna_v4.py \
    --data_dir ../data \
    --n_trials 150 \
    --seed 42 \
    --study_name nct_hierarchical_tuning_${dataset}_v4 \
    --storage sqlite:///$(pwd)/wandb_logs/databases/optuna_nct_${dataset}_v4.db \
    --dataset $dataset
