#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/nct_package_output_%j.log
#SBATCH --error=logs/nct_package_error_%j.log

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONUNBUFFERED=1

# Load conda module and activate environment
module load conda
conda activate nct_new


data_dir="../data"
num_iterations=3
epochs_per_iteration="50 50 50"
initial_base_lr=1e-5
initial_head_lr=1e-5
lr_decay_factor=2
batch_size=64
seed=42
weight_decay=5e-2
phase1_patience=10
phase2_patience=10
optimizer_type="adamw"
scheduler_type="plateau"
patience=3
phase1_epochs_ratio=0.5

python ../src/nct_package_main.py \
    --data_dir $data_dir \
    --num_iterations $num_iterations \
    --epochs_per_iteration $epochs_per_iteration \
    --initial_base_lr $initial_base_lr \
    --initial_head_lr $initial_head_lr \
    --lr_decay_factor $lr_decay_factor \
    --batch_size $batch_size \
    --seed $seed \
    --weight_decay $weight_decay \
    --phase1_patience $phase1_patience \
    --phase2_patience $phase2_patience \
    --optimizer_type $optimizer_type \
    --scheduler_type $scheduler_type \
    --phase1_epochs_ratio $phase1_epochs_ratio \
    --patience $patience


echo "NCT Package Training Complete"

