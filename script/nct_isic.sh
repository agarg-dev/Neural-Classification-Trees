#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --partition=bigmem
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/isic/nct_isic_output_%j.log
#SBATCH --error=logs/isic/nct_isic_error_%j.log

# Create output directories
mkdir -p logs/isic
mkdir -p wandb_logs/runs/nct
mkdir -p wandb_logs/databases
mkdir -p wandb_logs/results
mkdir -p wandb_logs/optuna/nct
mkdir -p wandb_logs/optuna/erm
mkdir -p .cache/wandb
mkdir -p checkpoints
mkdir -p attributions

# Set environment variables
export PYTHONUNBUFFERED=1

# Load conda module and activate environment
module load conda
conda activate nct_new

# Wandb configuration (offline mode for ARC compute nodes)
export WANDB_API_KEY="da20a2dfa4a6db68c79624a43a2656d91b9dfae1"
export WANDB_MODE=offline
export WANDB_DIR=/home/ankur.garg1/NCT/script/wandb_logs/runs/nct
export WANDB_CACHE_DIR=/home/ankur.garg1/NCT/script/.cache/wandb

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Wandb Mode: offline"
echo "Wandb Logs: $WANDB_DIR"
echo "=========================================="

# Configuration
data_dir="/home/ankur.garg1/NCT/data"
preset="/home/ankur.garg1/NCT/script/presets/isic_george.yaml"
checkpoint_dir="/home/ankur.garg1/NCT/script/checkpoints"
attribution_dir="/home/ankur.garg1/NCT/script/attributions"

# Copy dataset to fast local storage (/tmp)
echo "Copying ISIC dataset to /tmp for faster I/O..."
mkdir -p /tmp/nct_data
cp -r ${data_dir}/isic /tmp/nct_data/
echo "Dataset copied to /tmp/nct_data/isic"
df -h /tmp

# Run training with local data
python /home/ankur.garg1/NCT/src/nct_package_isic.py \
    --preset $preset \
    --data_dir /tmp/nct_data \
    --checkpoint_dir $checkpoint_dir \
    --attribution_dir $attribution_dir \
    --generate_attributions

echo "=========================================="
echo "NCT Package Training Complete"
echo "Finished: $(date)"
echo ""
echo "Output locations:"
echo "  Checkpoints: /home/ankur.garg1/NCT/script/checkpoints/"
echo "  Attributions: /home/ankur.garg1/NCT/script/attributions/\${SLURM_JOB_ID}_*/"
echo ""
echo "To upload to wandb, run on login node:"
echo "  conda activate nct_new"
echo "  wandb sync /home/ankur.garg1/NCT/script/wandb_logs/runs/nct/offline-run-*"
echo "=========================================="
