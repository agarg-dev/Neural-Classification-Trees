#!/bin/bash
# Run after: salloc --nodes=1 --cpus-per-task=16 --mem=256G --gres=gpu:1 --time=6:00:00 --account=rrg-ebrahimi_gpu

# Dataset: waterbird, isic, or celeba
dataset="isic"

# Paths
PROJECT_DIR="/lustre10/scratch/agarg/agarg/Neural-Classification-Trees"
ENV_DIR="$HOME/envs/nct_env"

# Create directories
mkdir -p ${PROJECT_DIR}/outputs/logs/train
mkdir -p ${PROJECT_DIR}/outputs/wandb/runs/${dataset}
mkdir -p ${PROJECT_DIR}/outputs/checkpoints
mkdir -p ${PROJECT_DIR}/outputs/attributions

# Environment
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export WANDB_DIR=${PROJECT_DIR}/outputs/wandb/runs/${dataset}

module purge
module load python/3.11.5
source ${ENV_DIR}/bin/activate

# Copy compressed data to local storage and extract it
echo "Copying ${dataset}.tar.gz to local storage..."
mkdir -p $SLURM_TMPDIR/data
cp ${PROJECT_DIR}/data/${dataset}.tar.gz $SLURM_TMPDIR/data/

echo "Extracting data..."
tar -xzf $SLURM_TMPDIR/data/${dataset}.tar.gz -C $SLURM_TMPDIR/data/
echo "Done."

# or do below if using full directory directly
# Copy data to local storage
# cp -r ${PROJECT_DIR}/data/${dataset} $SLURM_TMPDIR/data/

# Disable terminal focus reporting to prevent stray characters in output
printf '\e[?1004l'

# Run training
cd ${PROJECT_DIR}
python src/train.py \
    --dataset ${dataset} \
    --preset script/configs/${dataset}/${dataset}_nct_optuna.yaml \
    --data_dir $SLURM_TMPDIR/data \
    --checkpoint_dir outputs/checkpoints \
    --attribution_dir outputs/attributions \
    --generate_attributions
