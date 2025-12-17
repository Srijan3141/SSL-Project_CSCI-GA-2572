#!/bin/bash
# Training script for iBOT on 2xA100 GPUs with 500k 96x96 images

# Output directory
OUTPUT_DIR="./output_ibot_96x96"
mkdir -p $OUTPUT_DIR

# Launch distributed training on 2 GPUs using torchrun
torchrun \
    --nproc_per_node=2 \
    --master_port=29505 \
    main_ibot_custom.py \
    --arch vit_small \
    --patch_size 8 \
    --data_path /scratch/sp7007/pretrain \
    --output_dir /scratch/sp7007/bigrun_resumed_final100 \
    --batch_size_per_gpu 192 \
    --epochs 600 \
    --warmup_epochs 20 \
    --lr 0.0005 \
    --min_lr 1e-6 \
    --weight_decay 0.04 \
    --weight_decay_end 0.4 \
    --global_crops_number 2 \
    --local_crops_number 6 \
    --global_crops_scale 0.4 1.0 \
    --local_crops_scale 0.05 0.4 \
    --pred_ratio 0.3 \
    --pred_shape block \
    --optimizer adamw \
    --use_fp16 true \
    --clip_grad 3.0 \
    --freeze_last_layer 1 \
    --norm_last_layer true \
    --momentum_teacher 0.996 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp 0.04 \
    --saveckp_freq 10 \
    --num_workers 28 \
    --seed 0 \
    --train_csv /scratch/sp7007/data/train_labels.csv \
    --val_csv /scratch/sp7007/data/val_labels.csv \
    --train_dir /scratch/sp7007/data/train \
    --val_dir /scratch/sp7007/data/val \
    --load_from /scratch/sp7007/bigrun_resumed/checkpoint0500.pth


echo "Training completed! Checkpoint saved to $OUTPUT_DIR"