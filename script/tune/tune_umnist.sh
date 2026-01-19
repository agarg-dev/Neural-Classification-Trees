#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --time=10:00:00
#SBATCH --account=rrg-ebrahimi
#SBATCH --output=/home/agarg/links/scratch/agarg/Neural-Classification-Trees/outputs/logs/tune/umnist_v2/%j_output.log
#SBATCH --error=/home/agarg/links/scratch/agarg/Neural-Classification-Trees/outputs/logs/tune/umnist_v2/%j_error.log

# ============================================
# NCT Optuna Hyperparameter Tuning for UMNIST
# ============================================

# Define paths
PROJECT_DIR="/home/agarg/links/scratch/agarg/Neural-Classification-Trees"
ENV_DIR="$HOME/envs/nct_env"

dataset="umnist"

# Create directories
mkdir -p ${PROJECT_DIR}/outputs/logs/tune/umnist_v2
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
echo "Dataset: ${dataset}"
echo "Started: $(date)"
echo "Python: $(which python)"
echo "=========================================="

# Copy compressed data to local storage and extract it
echo "Copying ${dataset}.tar.gz to local storage..."
mkdir -p $SLURM_TMPDIR/data
cp ${PROJECT_DIR}/data/${dataset}.tar.gz $SLURM_TMPDIR/data/

echo "Extracting data..."
tar -xzf $SLURM_TMPDIR/data/${dataset}.tar.gz -C $SLURM_TMPDIR/data/
echo "Done."

# Run Optuna hyperparameter tuning
cd ${PROJECT_DIR}
python src/tune/tune_umnist.py \
    --data_dir $SLURM_TMPDIR/data \
    --n_trials 200 \
    --seed 42 \
    --study_name nct_tune_umnist_v2 \
    --storage sqlite:///${PROJECT_DIR}/outputs/optuna/optuna_nct_umnist_v2.db \
    --metric dna

echo "=========================================="
echo "Optuna Tuning Complete"
echo "Finished: $(date)"
echo "=========================================="
