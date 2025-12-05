#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=bigmem
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/nct_package_output_%j.log
#SBATCH --error=logs/nct_package_error_%j.log

# Create output directories
mkdir -p logs
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

# Hyperparameters
data_dir="/home/ankur.garg1/NCT/data"
num_iterations=3
epochs_per_iteration="100 100 100"
initial_base_lr=8e-06
initial_head_lr=8e-06
lr_decay_factor=2
batch_size=64
seed=42
weight_decay=1
regularization_type="l2"
lambda_l1=1e-5
aux_loss_weight=0.6
phase1_patience=10
phase2_patience=10
optimizer_type="adamw"
scheduler_type="plateau"
patience=3
phase1_epochs_ratio="1 0.5 0.5"
routing_strategy="argmax"
confidence_threshold=0.5
hierarchical_architecture="correct_label_routing"

# Output directories
checkpoint_dir="/home/ankur.garg1/NCT/script/checkpoints"
attribution_dir="/home/ankur.garg1/NCT/script/attributions"

# Run training
python /home/ankur.garg1/NCT/src/nct_package_main.py \
    --data_dir $data_dir \
    --num_iterations $num_iterations \
    --epochs_per_iteration $epochs_per_iteration \
    --initial_base_lr $initial_base_lr \
    --initial_head_lr $initial_head_lr \
    --lr_decay_factor $lr_decay_factor \
    --batch_size $batch_size \
    --seed $seed \
    --weight_decay $weight_decay \
    --regularization_type $regularization_type \
    --lambda_l1 $lambda_l1 \
    --aux_loss_weight $aux_loss_weight \
    --phase1_patience $phase1_patience \
    --phase2_patience $phase2_patience \
    --optimizer_type $optimizer_type \
    --scheduler_type $scheduler_type \
    --phase1_epochs_ratio $phase1_epochs_ratio \
    --patience $patience \
    --routing_strategy $routing_strategy \
    --confidence_threshold $confidence_threshold \
    --checkpoint_dir $checkpoint_dir \
    --attribution_dir $attribution_dir \
    --generate_attributions \
    --scheduler_eta_min 1e-6 \
    --hierarchical_architecture $hierarchical_architecture

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
