"""
Linear Probe TrainAndVal and Predict
======================================================================

This script trains a linear classifier on features extracted from a frozen pretrained Vision Transformer (ViT).

Pipeline:
    1. Load pretrained ViT checkpoint.
    2. Extract features for TRAIN, VAL, and TEST images.
    3. Train linear classifier using TRAIN + VAL labels (combined).
    4. Predict labels for TEST images.
    5. Save predictions in Kaggle submission format.

Expected CSV Format:
--------------------
train_labels.csv:
    filename,class_id,class_name
    00000_class042.jpg,42,n02403003
    00001_class042.jpg,42,n02403003
    ...

val_labels.csv:
    filename,class_id,class_name
    00000_class042.jpg,42,n02403003
    00001_class042.jpg,42,n02403003
    ...

test_images.csv:
    filename
    00000_class042.jpg
    00001_class015.jpg
    ...

Submission Format:
------------------
submission.csv:
    id,class_id
    00000_class042.jpg,42
    00001_class015.jpg,15
    ...

Usage:
python linear_probe_trainandval_and_predict.py \
    --checkpoint ./ibot_results/checkpoint0001.pth \
    --train_csv /scratch/ss17894/DL/data/train_labels.csv \
    --train_dir /scratch/ss17894/DL/data/train \
    --val_csv /scratch/ss17894/DL/data/val_labels.csv \
    --val_dir /scratch/ss17894/DL/data/val \
    --test_csv /scratch/ss17894/DL/data/test_images.csv \
    --test_dir /scratch/ss17894/DL/data/test \
    --num_labels 200 \
    --epochs 39 \
    --batch_size 512 \
    --lr 0.01 \
    --weight_decay 1e-5 \
    --extract_batch_size 512 \
    --output_dir ./submission/data1

"""

import os
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms

import models   # Your ViT model definitions


# =============================================================================
# Dataset
# =============================================================================

class CSVImageDataset(Dataset):
    """
    Kaggle-style dataset loader.
    train_labels.csv contains filename + class_id (+ class_name ignored)
    test_images.csv contains filename only.
    """
    def __init__(self, csv_path, img_dir, transform=None, has_labels=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["filename"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row["class_id"] if self.has_labels else -1
        return image, label, row["filename"]


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_features(model, loader, device, split_name='data'):
    features, labels, filenames = [], [], []

    model.eval()
    with torch.no_grad():
        for imgs, labs, fnames in tqdm(loader, desc=f"Extracting {split_name} features"):
            imgs = imgs.to(device)

            feats = model(imgs)
            if feats.ndim == 3:      # ViT output → CLS token
                feats = feats[:, 0]

            features.append(feats.cpu())
            labels.append(torch.tensor(labs))
            filenames.extend(fnames)

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features, labels, filenames


# =============================================================================
# Linear Probe Training
# =============================================================================

def train_linear_probe(train_features, train_labels,
                       num_labels, lr, weight_decay,
                       batch_size, epochs, device):
    """
    Trains a simple linear classifier on frozen features.
    """

    # L2-normalize features
    train_features = F.normalize(train_features, p=2, dim=1)

    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=batch_size,
        shuffle=True
    )

    input_dim = train_features.shape[1]
    classifier = nn.Linear(input_dim, num_labels).to(device)
    classifier.weight.data.normal_(0, 0.01)
    classifier.bias.data.zero_()

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    print("\n" + "="*60)
    print("Training Linear Probe")
    print("="*60)

    for epoch in range(epochs):
        classifier.train()
        total, correct = 0, 0

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            outputs = classifier(X)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = outputs.max(dim=1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {acc:.2f}%")

    return classifier


# =============================================================================
# Prediction & Submission Creation
# =============================================================================

def create_submission(classifier, test_features, filenames, batch_size, device, output_path):
    """
    Predicts class IDs for test images and creates Kaggle submission CSV.
    
    Args:
        classifier: Trained linear classifier
        test_features: Test image features
        filenames: List of test image filenames
        batch_size: Batch size for prediction
        device: Device to use
        output_path: Path to save submission.csv
    """
    
    # L2-normalize features
    test_features = F.normalize(test_features, p=2, dim=1)
    loader = DataLoader(test_features, batch_size=batch_size, shuffle=False)

    print("\nGenerating predictions on test set...")
    preds = []

    classifier.eval()
    with torch.no_grad():
        for X in tqdm(loader, desc="Predicting"):
            X = X.to(device)
            outputs = classifier(X)
            pred = outputs.argmax(dim=1).cpu().tolist()
            preds.extend(pred)

    # Create submission dataframe with correct format
    submission_df = pd.DataFrame({
        'id': filenames,
        'class_id': preds
    })
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Submission file created: {output_path}")
    print(f"{'='*60}")
    print(f"Total predictions: {len(submission_df)}")
    print(f"\nFirst 10 predictions:")
    print(submission_df.head(10))
    print(f"\nClass distribution in predictions:")
    print(submission_df['class_id'].value_counts().head(10))
    
    # Validate submission format
    print(f"\nValidating submission format...")
    assert list(submission_df.columns) == ['id', 'class_id'], f"Invalid columns! Expected ['id', 'class_id'], got {list(submission_df.columns)}"
    assert submission_df['class_id'].min() >= 0, "Invalid class_id < 0"
    assert submission_df['class_id'].max() < 400, f"Invalid class_id >= 200 (max is {submission_df['class_id'].max()})"
    assert submission_df.isnull().sum().sum() == 0, "Missing values found!"
    print("✓ Submission format is valid!")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Linear Probe TrainAndVal and Predict")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--arch", type=str, default="vit_small")
    parser.add_argument("--patch_size", type=int, default=8)

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--train_dir", type=str, required=True)

    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)

    parser.add_argument("--num_labels", type=int, required=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--extract_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--output_dir", type=str, default="./linear_probe_results")
    parser.add_argument("--output_filename", type=str, default="submission.csv",
                        help="Name of the submission CSV file")

    args = parser.parse_args()

    print("="*60)
    print("Linear Probe TrainAndVal and Predict")
    print("Training on TRAIN + VAL combined")
    print("="*60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Device -------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model ---------------------------------------------------------------
    print("\nLoading pretrained ViT backbone...")
    model = models.__dict__[args.arch](
        patch_size=args.patch_size,
        return_all_tokens=False
    )

    state = torch.load(args.checkpoint, map_location="cpu")
    if "teacher" in state: state = state["teacher"]
    if "student" in state: state = state["student"]

    # Clean keys
    state = {k.replace("module.", "").replace("backbone.", ""): v
             for k, v in state.items() if not k.startswith("head")}

    msg = model.load_state_dict(state, strict=False)
    print("Loaded:", msg)

    model = model.to(device).eval()

    # Transforms ---------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    # Data loaders -------------------------------------------------------------
    train_ds = CSVImageDataset(args.train_csv, args.train_dir,
                               transform, has_labels=True)
    
    val_ds = CSVImageDataset(args.val_csv, args.val_dir,
                             transform, has_labels=True)
    
    test_ds = CSVImageDataset(args.test_csv, args.test_dir,
                              transform, has_labels=False)

    train_loader = DataLoader(train_ds,
                              batch_size=args.extract_batch_size,
                              shuffle=False,
                              num_workers=4)

    val_loader = DataLoader(val_ds,
                            batch_size=args.extract_batch_size,
                            shuffle=False,
                            num_workers=4)

    test_loader = DataLoader(test_ds,
                             batch_size=args.extract_batch_size,
                             shuffle=False,
                             num_workers=4)

    # Extract features ---------------------------------------------------------
    print("\nExtracting TRAIN features...")
    train_features, train_labels, _ = extract_features(model, train_loader, device, 'TRAIN')

    print("\nExtracting VAL features...")
    val_features, val_labels, _ = extract_features(model, val_loader, device, 'VAL')

    print("\nExtracting TEST features...")
    test_features, _, test_filenames = extract_features(model, test_loader, device, 'TEST')

    # Combine train and val for training --------------------------------------
    print("\nCombining TRAIN + VAL datasets...")
    combined_features = torch.cat([train_features, val_features], dim=0)
    combined_labels = torch.cat([train_labels, val_labels], dim=0)
    
    print(f"  Train: {len(train_features)} samples")
    print(f"  Val: {len(val_features)} samples")
    print(f"  Combined: {len(combined_features)} samples")

    # Train linear probe on combined data -------------------------------------
    classifier = train_linear_probe(
        combined_features, combined_labels,
        args.num_labels,
        args.lr, args.weight_decay,
        args.batch_size, args.epochs,
        device
    )

    # Create submission --------------------------------------------------------
    output_csv = Path(args.output_dir) / args.output_filename
    create_submission(classifier, test_features, test_filenames,
                     args.batch_size, device, output_csv)


if __name__ == "__main__":
    main()