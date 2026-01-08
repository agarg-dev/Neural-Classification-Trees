#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/optuna_v3/nct_package_output_tuning_v3_%j.log
#SBATCH --error=logs/optuna_v3/nct_package_error_tuning_v3_%j.log

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1

# Load conda module and activate environment
module load conda
conda activate nct_new
dataset="celeba"

# Run Optuna hyperparameter tuning V3 (optimizing pseudo-label WGA at iteration 2)
# Updated search space with new hyperparameters
python ../src/nct_package_optuna_v3.py \
    --data_dir ../data \
    --n_trials 300 \
    --seed 42 \
    --study_name nct_hierarchical_tuning_${dataset}_v3 \
    --storage sqlite:///$(pwd)/wandb_logs/databases/optuna_nct_${dataset}_v3.db \
    --dataset $dataset
