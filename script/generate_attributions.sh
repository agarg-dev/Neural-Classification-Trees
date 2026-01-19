#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --account=rrg-ebrahimi_gpu
#SBATCH --output=/lustre10/scratch/agarg/agarg/Neural-Classification-Trees/outputs/logs/attributions/%j_output.log
#SBATCH --error=/lustre10/scratch/agarg/agarg/Neural-Classification-Trees/outputs/logs/attributions/%j_error.log

# ============================================
# Attribution Generation Script for Rorqual Cluster
# ============================================

# Define paths
PROJECT_DIR="/lustre10/scratch/agarg/agarg/Neural-Classification-Trees"
ENV_DIR="$HOME/envs/nct_env"

# Create logs directory
mkdir -p ${PROJECT_DIR}/outputs/logs/attributions

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

# Change to project directory
cd ${PROJECT_DIR}

# Run ALL methods (Automatic Run ID generation)
python ${PROJECT_DIR}/src/generate_attributions.py \
    --checkpoint "${PROJECT_DIR}/outputs/checkpoints/final_model.pt" \
    --data_dir "${PROJECT_DIR}/data" \
    --attribution_dir "${PROJECT_DIR}/outputs/attributions" \
    --num_per_node 5 \
    --num_transitions 5 \
    --methods all_layer

echo "=========================================="
echo "All Attribution Methods Complete"
echo "Finished: $(date)"
echo "=========================================="
