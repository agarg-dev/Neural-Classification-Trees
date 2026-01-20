#!/bin/bash
# ============================================
# Sync wandb offline logs to cloud
# Run this on Nibi login node after training job completes
# ============================================

echo "=========================================="
echo "Syncing Wandb Offline Logs"
echo "=========================================="

# Define paths
PROJECT_DIR="/home/agarg/projects/rrg-ebrahimi/agarg/Neural-Classification-Trees"
ENV_DIR="$HOME/envs/nct_env"

# Load modules and activate virtual environment
module purge
module load python/3.11.5
source ${ENV_DIR}/bin/activate

# Change to project directory
cd ${PROJECT_DIR}

synced_count=0

# Sync Waterbird runs
if ls outputs/wandb/runs/waterbird/offline-run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing Waterbird runs..."
    for run_dir in ./outputs/wandb/runs/waterbird/offline-run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-waterbirds "$run_dir"
        ((synced_count++))
    done
fi

# Sync CelebA runs
if ls outputs/wandb/runs/celeba/offline-run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing CelebA runs..."
    for run_dir in ./outputs/wandb/runs/celeba/offline-run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-celeba "$run_dir"
        ((synced_count++))
    done
fi

# Sync ISIC runs
if ls outputs/wandb/runs/isic/offline-run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing ISIC runs..."
    for run_dir in ./outputs/wandb/runs/isic/offline-run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-isic "$run_dir"
        ((synced_count++))
    done
fi

echo ""
echo "=========================================="
if [ $synced_count -eq 0 ]; then
    echo "No offline runs found."
    echo "Make sure your training job completed successfully."
else
    echo "Sync Complete! ($synced_count runs synced)"
    echo ""
    echo "View your runs at:"
    echo "  - Waterbird runs: https://wandb.ai/ankurgarg-university-of-calgary-in-alberta/nct-waterbirds"
    echo "  - CelebA runs: https://wandb.ai/ankurgarg-university-of-calgary-in-alberta/nct-celeba"
    echo "  - ISIC runs: https://wandb.ai/ankurgarg-university-of-calgary-in-alberta/nct-isic"
fi
echo "=========================================="
