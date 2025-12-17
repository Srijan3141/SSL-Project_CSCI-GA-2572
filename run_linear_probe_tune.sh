CUB

#!/bin/bash
# Linear probing with hyperparameter tuning
# Usage: ./run_linear_probe_tune.sh <checkpoint_path> [--grid_search]

CHECKPOINT=${1:-"/scratch/sp7007/bigrun_resumed_final100/checkpoint0570.pth"}
MODE=${2:-"--grid_search"}  # Use --grid_search or leave empty for single run
OUTPUT_DIR="/scratch/sp7007/cub/570_final"

mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Linear Probing with Hyperparameter Tuning"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Architecture: vit_small (patch_size=8)"
echo "Mode: $MODE"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

if [[ "$MODE" == "--grid_search" ]]; then
    echo "Running GRID SEARCH mode..."
    echo "This will try multiple hyperparameter combinations"
    echo ""
    
    python linear_probe_tune.py \
        --arch vit_small \
        --patch_size 8 \
        --checkpoint $CHECKPOINT \
        --train_csv /scratch/sp7007/cub/kaggle_data/train_labels.csv \
        --val_csv /scratch/sp7007/cub/kaggle_data/val_labels.csv \
        --train_dir /scratch/sp7007/cub/kaggle_data/train \
        --val_dir /scratch/sp7007/cub/kaggle_data/val \
        --num_labels 200 \
        --epochs 100 \
        --extract_batch_size 1024 \
        --num_workers 8 \
        --output_dir $OUTPUT_DIR \
        --grid_search \
        --search_lr 0.005 0.001 0.002 0.0035 \
        --search_wd 1e-6 1e-5 \
        --search_optimizer adamw \
        --search_batch_size 32 \
        --search_normalization l2
else
    echo "Running SINGLE CONFIGURATION mode..."
    echo ""
    
    python linear_probe_tune.py \
        --arch vit_small \
        --patch_size 8 \
        --checkpoint $CHECKPOINT \
        --train_csv /scratch/sp7007/imagenet/kaggle_data_miniimagenet/train_labels.csv \
        --val_csv /scratch/sp7007/imagenet/kaggle_data_miniimagenet/val_labels.csv \
        --train_dir /scratch/sp7007/imagenet/kaggle_data_miniimagenet/train \
        --val_dir /scratch/sp7007/imagenet/kaggle_data_miniimagenet/val \
        --num_labels 100 \
        --epochs 300 \
        --batch_size 512 \
        --extract_batch_size 512 \
        --lr 0.01 \
        --weight_decay 1e-5 \
        --optimizer adamw \
        --normalization l2 \
        --num_workers 8 \
        --output_dir $OUTPUT_DIR \
        --save_checkpoint
fi

echo ""
echo "=============================================="
echo "Linear probing completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="