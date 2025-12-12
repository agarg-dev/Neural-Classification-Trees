#!/bin/bash
# Sync wandb offline logs to cloud
# Run this on ARC login node after training job completes

echo "=========================================="
echo "Syncing Wandb Offline Logs"
echo "=========================================="

# Activate conda environment
source ~/.bashrc
module load conda
conda activate nct_new

# Change to script directory
cd /home/ankur.garg1/NCT/script

synced_count=0

# Sync normal NCT training runs
if ls wandb_logs/runs/nct/offline-run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing normal NCT training runs..."
    for run_dir in ./wandb_logs/runs/nct/offline-run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-waterbirds "$run_dir"
        ((synced_count++))
    done
fi

# Sync NCT optuna runs
if ls wandb_logs/optuna/nct/offline-run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing NCT optuna runs..."
    for run_dir in ./wandb_logs/optuna/nct/offline-run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-optuna "$run_dir"
        ((synced_count++))
    done
fi

# Sync ERM optuna runs
if ls wandb_logs/optuna/erm/offline-run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing ERM optuna runs..."
    for run_dir in ./wandb_logs/optuna/erm/offline-run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-optuna "$run_dir"
        ((synced_count++))
    done
fi

# Sync legacy wandb directory (if exists)
if ls wandb/run-* 1> /dev/null 2>&1; then
    echo ""
    echo "Syncing legacy wandb runs..."
    for run_dir in ./wandb/run-*; do
        echo "  Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-waterbirds "$run_dir"
        ((synced_count++))
    done
fi

echo ""
echo "=========================================="
if [ $synced_count -eq 0 ]; then
    echo "No offline runs found."
    echo "Make sure your training job completed successfully."
else
    echo "✓ Sync Complete! ($synced_count runs synced)"
    echo ""
    echo "View your runs at:"
    echo "  • Normal runs: https://wandb.ai/ankurgarg-university-of-calgary-in-alberta/nct-waterbirds"
    echo "  • Optuna runs: https://wandb.ai/ankurgarg-university-of-calgary-in-alberta/nct-optuna"
fi
echo "=========================================="
