#!/bin/bash

# Fast linear probing: Extract features once, then train
# Usage: ./run_linear_probe_fast.sh <checkpoint_path>

CHECKPOINT=${1:-"/scratch/sp7007/bigrun/checkpoint0130.pth"}
OUTPUT_DIR="/scratch/sp7007/bigrun/linear_probe_results"

mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Fast Linear Probing Evaluation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Architecture: vit_small (patch_size=8)"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

python linear_probe.py \
    --arch vit_small \
    --patch_size 8 \
    --checkpoint $CHECKPOINT \
    --train_csv /scratch/sp7007/data/train_labels.csv \
    --val_csv /scratch/sp7007/data/val_labels.csv \
    --train_dir /scratch/sp7007/data/train \
    --val_dir /scratch/sp7007/data/val \
    --num_labels 100 \
    --epochs 100 \
    --batch_size 1024 \
    --extract_batch_size 256 \
    --lr 0.01 \
    --num_workers 4 \
    --output_dir $OUTPUT_DIR \
    --save_checkpoint \
    --save_features

echo ""
echo "=============================================="
echo "Linear probing completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="