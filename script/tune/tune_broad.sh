#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --output=/home/agarg/projects/rrg-ebrahimi/agarg/Neural-Classification-Trees/outputs/logs/tune/broad/%j_output.log
#SBATCH --error=/home/agarg/projects/rrg-ebrahimi/agarg/Neural-Classification-Trees/outputs/logs/tune/broad/%j_error.log

# ============================================
# NCT Optuna Hyperparameter Tuning for Nibi Cluster
# ============================================

# Define paths
PROJECT_DIR="/home/agarg/projects/rrg-ebrahimi/agarg/Neural-Classification-Trees"
ENV_DIR="$HOME/envs/nct_env"

# Create directories
mkdir -p ${PROJECT_DIR}/outputs/logs/tune/broad
mkdir -p ${PROJECT_DIR}/outputs/optuna

# Set environment variables
export PYTHONUNBUFFERED=1

# Load modules and activate virtual environment
module purge
module load python/3.11.5
source ${ENV_DIR}/bin/activate

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Python: $(which python)"
echo "=========================================="

dataset="isic"

# Copy compressed data to local storage and extract it
echo "Copying ${dataset}.tar.gz to local storage..."
mkdir -p $SLURM_TMPDIR/data
cp ${PROJECT_DIR}/data/${dataset}.tar.gz $SLURM_TMPDIR/data/

echo "Extracting data..."
tar -xzf $SLURM_TMPDIR/data/${dataset}.tar.gz -C $SLURM_TMPDIR/data/
echo "Done."


# Run Optuna hyperparameter tuning
cd ${PROJECT_DIR}
python src/tune/tune_broad.py \
    --data_dir $SLURM_TMPDIR/data \
    --n_trials 100 \
    --seed 42 \
    --study_name nct_hierarchical_tuning_${dataset}_broad \
    --storage sqlite:///${PROJECT_DIR}/outputs/optuna/optuna_nct_${dataset}_broad.db \
    --dataset $dataset \
    --metric wga

echo "=========================================="
echo "Optuna Tuning Complete"
echo "Finished: $(date)"
echo "=========================================="
