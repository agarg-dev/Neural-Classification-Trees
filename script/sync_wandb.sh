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

# Check if there are offline runs to sync
if ls wandb_logs/wandb/offline-run-* 1> /dev/null 2>&1; then
    echo "Found offline runs. Syncing..."

    # Sync each offline run with explicit entity and project
    for run_dir in wandb_logs/wandb/offline-run-*; do
        echo "Syncing $run_dir ..."
        wandb sync --entity ankurgarg-university-of-calgary-in-alberta --project nct-waterbirds "$run_dir"
    done

    echo ""
    echo "=========================================="
    echo "✓ Sync Complete!"
    echo "View your runs at: https://wandb.ai/ankurgarg-university-of-calgary-in-alberta/nct-waterbirds"
    echo "=========================================="
else
    echo "No offline runs found in wandb_logs/wandb/"
    echo "Make sure your training job completed successfully."
fi
