#!/bin/bash
# Run this after salloc allocation

cd /home/ankur.garg1/NCT

# Load conda and activate environment
module load conda
conda activate nct_new

# Set environment variables
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR=/home/ankur.garg1/NCT/script/wandb_logs/runs/nct
export WANDB_CACHE_DIR=/home/ankur.garg1/NCT/script/.cache/wandb

# Run training
python src/nct_package_main.py \
    --data_dir ./data \
    --num_iterations 3 \
    --epochs_per_iteration 100 100 100 \
    --initial_base_lr 8e-06 \
    --initial_head_lr 8e-06 \
    --lr_decay_factor 2 \
    --batch_size 64 \
    --seed 42 \
    --hierarchical_architecture correct_label_routing \
    --checkpoint_dir ./script/checkpoints \
    --attribution_dir ./script/attributions \
    --generate_attributions