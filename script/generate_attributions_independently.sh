#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --output=logs/attribution_image/attributions_%j.log
#SBATCH --error=logs/attribution_image/attributions_error_%j.log

export PYTHONUNBUFFERED=1

# Define paths (adjust if needed)
PROJECT_DIR="/home/ankur.garg1/NCT"
mkdir -p "$PROJECT_DIR/script/logs"

module load conda
conda activate nct_new

cd "$PROJECT_DIR/script"

# Run ALL methods (Automatic Run ID generation)
python ../src/generate_attributions_independently.py \
    --checkpoint "$PROJECT_DIR/script/checkpoints/final_model.pt" \
    --data_dir "$PROJECT_DIR/data" \
    --attribution_dir "$PROJECT_DIR/script/attributions" \
    --num_per_node 5 \
    --num_transitions 5 \
    --methods all_layer

echo "All Attribution Methods Complete"
