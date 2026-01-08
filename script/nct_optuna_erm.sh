#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=logs/erm_optuna/nct_erm_tuning_%j.log
#SBATCH --error=logs/erm_optuna/nct_erm_tuning_%j.log

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1

# Load conda module and activate environment
module load conda
conda activate nct_new

# Run Optuna hyperparameter tuning for ERM baseline (1 iteration)
# This establishes the best ERM performance baseline
python ../src/nct_package_optuna_erm.py \
    --data_dir ../data \
    --n_trials 100 \
    --seed 42 \
    --study_name nct_erm_baseline_f1 \
    --storage sqlite:///$(pwd)/wandb_logs/databases/optuna_erm.db \
    --fresh_study
