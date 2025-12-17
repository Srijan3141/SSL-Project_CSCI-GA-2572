#!/bin/bash

# KNN evaluation script
# Run this after training completes

# Path to your trained checkpoint
CHECKPOINT="./output_ibot_96x96/checkpoint.pth"

# Or use a specific epoch checkpoint
# CHECKPOINT="./output_ibot_96x96/checkpoint0800.pth"

python eval_knn.py \
    --arch vit_small \
    --patch_size 8 \
    --checkpoint $CHECKPOINT \
    --train_csv /scratch/sp7007/data/train_labels.csv \
    --val_csv /scratch/sp7007/data/val_labels.csv \
    --train_dir /scratch/sp7007/data/train \
    --val_dir /scratch/ss17894/DL/data/val \
    --k 20 \
    --batch_size 256 \
    --num_workers 10

echo "KNN evaluation completed!"