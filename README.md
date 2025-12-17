# iBOT Implementation

**iBOT (Image BERT Pre-Training with Online Tokenizer)** is a self-supervised learning method that combines masked image modeling with self-distillation. This implementation trains a Vision Transformer (ViT) on unlabeled images using both [CLS] token prediction and masked patch reconstruction.

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
pip install numpy==1.26.4 pandas==2.3.3 pillow==11.3.0 tensorboardX==2.6.4 timm==1.0.21 scipy==1.16.2 einops==0.8.1
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

**Resume training:** Uncomment `--load_from checkpoint0500.pth` in `run_training.sh`



## Citation
```bibtex
@inproceedings{zhou2021ibot,
  title={iBOT: Image BERT Pre-Training with Online Tokenizer},
  author={Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}
```