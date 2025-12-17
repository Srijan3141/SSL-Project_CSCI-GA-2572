#!/bin/bash
# Linear probing with hyperparameter tuning

CHECKPOINT=${1:-"./ibot_results/checkpoint0570.pth"}
MODE=${2:-"--grid_search"}  
OUTPUT_DIR="grid_search_results"

mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "Linear Probing with Hyperparameter Tuning"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
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
        --train_csv /scratch/ss17894/imagenet/train_labels.csv \
        --val_csv /scratch/ss17894/imagenet/val_labels.csv \
        --train_dir /scratch/ss17894/imagenet/train \
        --val_dir /scratch/ss17894/imagenet/val \
        --num_labels 100 \
        --epochs 100 \
        --extract_batch_size 1024 \
        --num_workers 8 \
        --output_dir $OUTPUT_DIR \
        --grid_search \
        --search_lr 0.1 0.01 0.001  \
        --search_wd 1e-6 1e-5 \
        --search_optimizer adamw \
        --search_batch_size 64 128 256 512 \
        --search_normalization l2
else
    echo "Running SINGLE CONFIGURATION mode..."
    echo ""
    
    python linear_probe_tune.py \
        --arch vit_small \
        --patch_size 8 \
        --checkpoint $CHECKPOINT \
        --train_csv /scratch/ss17894/imagenet/train_labels.csv \
        --val_csv /scratch/ss17894/imagenet/val_labels.csv \
        --train_dir /scratch/ss17894/imagenet/train \
        --val_dir /scratch/ss17894/imagenet/val \
        --num_labels 100 \
        --epochs 200 \
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