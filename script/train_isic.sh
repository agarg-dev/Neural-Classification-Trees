#!/bin/bash
#SBATCH --account=rrg-ebrahimi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=5:00:00
#SBATCH --output=/home/agarg/links/scratch/agarg/Neural-Classification-Trees/outputs/logs/train/isic/%j_output.log
#SBATCH --error=/home/agarg/links/scratch/agarg/Neural-Classification-Trees/outputs/logs/train/isic/%j_error.log


dataset="isic"

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


# Copy compressed data to local storage and extract it
echo "Copying ${dataset}.tar.gz to local storage..."
mkdir -p $SLURM_TMPDIR/data
cp ${PROJECT_DIR}/data/${dataset}.tar.gz $SLURM_TMPDIR/data/

echo "Extracting data..."
tar -xzf $SLURM_TMPDIR/data/${dataset}.tar.gz -C $SLURM_TMPDIR/data/
echo "Done."

preset=${dataset}_nct_optuna.yaml

#for ablation
# preset=${dataset}_aux_loss_ablation.yaml
# preset=${dataset}_nct_no_weight_iter1.yaml

# Run training
cd ${PROJECT_DIR}
python src/train.py \
    --dataset ${dataset} \
    --preset script/configs/${dataset}/${preset} \
    --data_dir $SLURM_TMPDIR/data \
    --checkpoint_dir outputs/checkpoints \
    --attribution_dir outputs/attributions \
    --generate_attributions \
    --seed ${seed}

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
