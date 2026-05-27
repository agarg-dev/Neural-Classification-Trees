#!/bin/bash
# Train NCT on a dataset using its tuned preset.
# Usage: bash script/train.sh <dataset> [seed] [data_dir]
set -e

dataset="${1:?Usage: bash script/train.sh <dataset> [seed] [data_dir]}"
seed="${2:-42}"
data_dir="${3:-./data}"

python src/train.py \
    --dataset "${dataset}" \
    --preset "script/configs/${dataset}/${dataset}.yaml" \
    --data_dir "${data_dir}" \
    --checkpoint_dir outputs/checkpoints \
    --attribution_dir outputs/attributions \
    --generate_attributions \
    --seed "${seed}"
