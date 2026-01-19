#!/bin/bash
#SBATCH --account=rrg-ebrahimi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=5:00:00
#SBATCH --output=/home/agarg/links/scratch/agarg/Neural-Classification-Trees/outputs/logs/train/%x_%j_output.log
#SBATCH --error=/home/agarg/links/scratch/agarg/Neural-Classification-Trees/outputs/logs/train/%x_%j_error.log


## # 10GB slice (1/7 of H100)
## #SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1

# 20GB slice (2/7 of H100)
# ##SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1

# 40GB slice (3/7 of H100)
# ##SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1


# Dataset: waterbird, isic, celeba, umnist, cmnist
dataset="waterbird"

seed=${1:-42}

# Paths
PROJECT_DIR="/home/agarg/links/scratch/agarg/Neural-Classification-Trees"
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


# Print info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: $dataset"
echo "Seed: $seed"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "=========================================="

# or do below if using full directory directly
# # Copy data to fast local storage
# echo "Copying ${dataset} dataset to SLURM_TMPDIR..."
# mkdir -p $SLURM_TMPDIR/data
# cp -r ${PROJECT_DIR}/data/${dataset} $SLURM_TMPDIR/data/
# echo "Done."


# Copy compressed data to local storage and extract it
echo "Copying ${dataset}.tar.gz to local storage..."
mkdir -p $SLURM_TMPDIR/data
cp ${PROJECT_DIR}/data/${dataset}.tar.gz $SLURM_TMPDIR/data/

echo "Extracting data..."
tar -xzf $SLURM_TMPDIR/data/${dataset}.tar.gz -C $SLURM_TMPDIR/data/
echo "Done."

# Run training
cd ${PROJECT_DIR}
python src/train.py \
    --dataset ${dataset} \
    --preset script/configs/${dataset}/${dataset}_nct_optuna.yaml \
    --data_dir $SLURM_TMPDIR/data \
    --checkpoint_dir outputs/checkpoints \
    --attribution_dir outputs/attributions \
    --generate_attributions \
    --seed ${seed}

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
