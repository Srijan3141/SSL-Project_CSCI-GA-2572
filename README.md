# iBOT Implementation

**iBOT (Image BERT Pre-Training with Online Tokenizer)** is a self-supervised learning method that combines masked image modeling with self-distillation. This implementation trains a Vision Transformer (ViT) on unlabeled images using both [CLS] token prediction and masked patch reconstruction.

## Table of Contents
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#grid-search-evaluation-for-linear-probing)
- [Final Submission](#final-submission)
- [Citation](#citation)

## Installation

### Create Environment
```bash
# Create new conda environment with Python 3.12
conda create -n ibot_env python=3.12 -y
conda activate ibot_env
```

### Install Dependencies
```bash
# Install PyTorch 2.4.0 with CUDA 12.1
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install numpy==1.26.4 pandas==2.3.3 pillow==11.3.0 tensorboardX==2.6.4 timm==1.0.21 scipy==1.16.2 einops==0.8.1 urllib3 idna pytz python-dateutil
```

## Training

```bash
chmod +x run_training.sh
./run_training.sh
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--arch` | `vit_small` | Model architecture (vit_tiny, vit_small, vit_base, vit_large) |
| `--patch_size` | `8` | Size of image patches |
| `--batch_size_per_gpu` | `192` | Batch size per GPU |
| `--epochs` | `600` | Total training epochs |
| `--lr` | `0.0005` | Peak learning rate |
| `--global_crops_number` | `2` | Number of large crops |
| `--local_crops_number` | `6` | Number of small crops |
| `--pred_ratio` | `0.3` | Ratio of masked patches |
| `--saveckp_freq` | `10` | Checkpoint save frequency (epochs) |

Edit `run_training.sh` to modify these parameters and data paths.

### Resume Training

To resume from a checkpoint, uncomment the following line in `run_training.sh`:
```bash
--load_from checkpoint0500.pth
```

## Grid Search Evaluation for Linear Probing

Evaluates learned representations by training a linear classifier on frozen features.

### Setup

**Before running, edit `run_linear_probe_tune.sh`:**

1. Update data paths to match your dataset location
2. Set `--num_labels` to match your number of classes
3. Change checkpoint path as needed

### Running Evaluation

```bash
chmod +x run_linear_probe_tune.sh

# Grid search (default)
./run_linear_probe_tune.sh

# Single configuration
./run_linear_probe_tune.sh "" --single
```

### Grid Search Parameters

Grid search tests combinations of:
- Learning rate (`--search_lr`)
- Weight decay (`--search_wd`)
- Batch size (`--search_batch_size`)
- Optimizer (`--search_optimizer`)
- Normalization (`--search_normalization`)

## Final Submission

Linear probing with best hyperparameters found in grid search, combining both train and validation sets.

### Download Pre-trained Checkpoint

**Important:** For best results, download our pre-trained checkpoint and place it at `./ibot_results/checkpoint0570.pth`

📥 [**Download checkpoint0570.pth**](https://drive.google.com/file/d/1sQIt9aQD7hUMHGEM90ZYNJCM0Q86n1ur/view?usp=sharing)

### Running Predictions

Update the data paths in the commands below to match your dataset locations.

#### Dataset 1: CUB-200

```bash
python linear_probe_trainandval_and_predict.py \
    --checkpoint ./ibot_results/checkpoint0570.pth \
    --train_csv /scratch/ss17894/cub/train_labels.csv \
    --train_dir /scratch/ss17894/cub/train \
    --val_csv /scratch/ss17894/cub/val_labels.csv \
    --val_dir /scratch/ss17894/cub/val \
    --test_csv /scratch/ss17894/cub/test_images.csv \
    --test_dir /scratch/ss17894/cub/test \
    --num_labels 200 \
    --epochs 35 \
    --batch_size 16 \
    --lr 0.005 \
    --weight_decay 1e-5 \
    --extract_batch_size 1024 \
    --output_dir ./submissions/cub
```

#### Dataset 2: ImageNet-100

```bash
python linear_probe_trainandval_and_predict.py \
    --checkpoint ./ibot_results/checkpoint0570.pth \
    --train_csv /scratch/ss17894/imagenet/train_labels.csv \
    --train_dir /scratch/ss17894/imagenet/train \
    --val_csv /scratch/ss17894/imagenet/val_labels.csv \
    --val_dir /scratch/ss17894/imagenet/val \
    --test_csv /scratch/ss17894/imagenet/test_images.csv \
    --test_dir /scratch/ss17894/imagenet/test \
    --num_labels 100 \
    --epochs 10 \
    --batch_size 512 \
    --lr 0.02 \
    --weight_decay 1e-5 \
    --extract_batch_size 256 \
    --output_dir ./submissions/imagenet
```

#### Dataset 3: SUN397

```bash
python linear_probe_trainandval_and_predict.py \
    --checkpoint ./ibot_results/checkpoint0570.pth \
    --train_csv /scratch/ss17894/SUN/train_labels.csv \
    --train_dir /scratch/ss17894/SUN/kaggle_data_sun397/train \
    --val_csv /scratch/ss17894/SUN/val_labels.csv \
    --val_dir /scratch/ss17894/SUN/val \
    --test_csv /scratch/ss17894/SUN/test_images.csv \
    --test_dir /scratch/ss17894/SUN/test \
    --num_labels 397 \
    --epochs 17 \
    --batch_size 128 \
    --lr 0.01 \
    --weight_decay 1e-5 \
    --extract_batch_size 128 \
    --output_dir ./submissions/sun397
```

## Citation

```bibtex
@inproceedings{zhou2021ibot,
  title={iBOT: Image BERT Pre-Training with Online Tokenizer},
  author={Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```