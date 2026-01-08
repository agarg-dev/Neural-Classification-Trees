#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/optuna_v1/nct_output_tuning_%j.log
#SBATCH --error=logs/optuna_v1/nct_error_tuning_%j.log

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1

# Load conda module and activate environment
module load conda
conda activate nct_new
dataset="waterbird"
# Run Optuna hyperparameter tuning (optimizing pseudo-label F1 score at iteration 2)
# Continuing from previous study - adding 100 more trials
python ../src/nct_package_optuna.py \
    --data_dir ../data \
    --n_trials 100 \
    --seed 42 \
    --study_name nct_hierarchical_tuning_${dataset}_wga_v2 \
    --storage sqlite:///$(pwd)/wandb_logs/databases/optuna_nct_${dataset}_v2.db \
    --dataset $dataset
