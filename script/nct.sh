#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

mkdir -p logs
export PYTHONUNBUFFERED=1
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


python ../src/nct_waterbird.py \
    --data_dir $data_dir \
    --num_iterations $num_iterations \
    --epochs_per_iteration $epochs_per_iteration \
    --initial_base_lr $initial_base_lr \
    --initial_head_lr $initial_head_lr \
    --lr_decay_factor $lr_decay_factor \
    --batch_size $batch_size \
    --seed $seed \
    --weight_decay $weight_decay





# #!/bin/bash
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=32gb
# #SBATCH --partition=gpu-v100
# #SBATCH --gres=gpu:1
# #SBATCH --time=3:00:00
# #SBATCH --output=logs/output_%j.log
# #SBATCH --error=logs/error_%j.log

# mkdir -p logs
# export PYTHONUNBUFFERED=1
# module load conda
# conda activate nct

# path="../data/titanic/train.csv"
# num_iterations=3
# epochs_per_iteration="100 100 100"
# initial_base_lr=1e-3
# initial_head_lr=1e-3
# lr_decay_factor=2
# within_iteration_lr_decay=2
# batch_size=64
# difficult_sample_weight=1.0
# seed=42

# python ../src/nct_waterbird.py \
#     --data_dir $path \
#     --num_iterations $num_iterations \
#     --epochs_per_iteration $epochs_per_iteration \
#     --initial_base_lr $initial_base_lr \
#     --initial_head_lr $initial_head_lr \
#     --lr_decay_factor $lr_decay_factor \
#     --batch_size $batch_size \
#     --difficult_sample_weight $difficult_sample_weight \
#     --seed $seed