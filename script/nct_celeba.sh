#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=bigmem
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/nct_celeba_output_%j.log
#SBATCH --error=logs/nct_celeba_error_%j.log

# Create output directories
mkdir -p logs
mkdir -p wandb_logs/runs/celeba
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
export WANDB_DIR=/home/ankur.garg1/NCT/script/wandb_logs/runs/celeba
export WANDB_CACHE_DIR=/home/ankur.garg1/NCT/script/.cache/wandb

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Dataset: CelebA (Blond Hair)"
echo "Wandb Mode: offline"
echo "Wandb Logs: $WANDB_DIR"
echo "=========================================="

data_dir="/home/ankur.garg1/NCT/data"
num_iterations=3
epochs_per_iteration="50 100 100"  # JTT uses 50 epochs
initial_base_lr=1e-5  # JTT baseline lr
initial_head_lr=1e-5  # Higher lr for classification heads
lr_decay_factor=1.5   # Gentle decay between iterations
batch_size=128         # JTT baseline batch size
seed=42
weight_decay=0.1      # JTT baseline weight decay
regularization_type="l2"
lambda_l1=0.0
aux_loss_weight=0.5   # Moderate auxiliary loss for hierarchical consistency
phase1_patience=8
phase2_patience=8
optimizer_type="adamw"  # Better than SGD for fine-tuning
scheduler_type="plateau"
patience=5
phase1_epochs_ratio="1 0.5 0.5"  # More head-only training in iteration 1
routing_strategy="argmax"
confidence_threshold=0.7
hierarchical_architecture="correct_label_routing"

# Output directories
checkpoint_dir="/home/ankur.garg1/NCT/script/checkpoints"
attribution_dir="/home/ankur.garg1/NCT/script/attributions"

# Run training
python /home/ankur.garg1/NCT/src/nct_package_celeba.py \
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
echo "NCT CelebA Training Complete"
echo "Finished: $(date)"
echo ""
echo "Output locations:"
echo "  Checkpoints: /home/ankur.garg1/NCT/script/checkpoints/"
echo "  Attributions: /home/ankur.garg1/NCT/script/attributions/\${SLURM_JOB_ID}_*/"
echo ""
echo "To upload to wandb, run on login node:"
echo "  conda activate nct_new"
echo "  wandb sync /home/ankur.garg1/NCT/script/wandb_logs/runs/celeba/offline-run-*"
echo "=========================================="
