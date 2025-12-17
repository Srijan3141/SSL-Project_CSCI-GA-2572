"""
Fast linear probing: Extract features once, then train linear classifier
This is much faster than computing features every epoch
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import models


class CSVImageDataset(Dataset):
    """Dataset that loads images from CSV with labels"""
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['class_id']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def extract_features(model, data_loader, device):
    """Extract features once and cache them"""
    print("Extracting features (this only happens once)...")
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(data_loader):
            if i % 10 == 0:
                print(f"  Processing batch {i}/{len(data_loader)}")
            
            imgs = imgs.to(device)
            
            # Extract features
            features = model(imgs)
            
            # Handle different output formats
            if len(features.shape) == 3:
                features = features[:, 0]  # Take [CLS] token
            
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted features shape: {features.shape}")
    return features, labels


def train_linear_classifier(train_features, train_labels, val_features, val_labels, args):
    """Train linear classifier on cached features"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Normalize features
    print("Normalizing features...")
    train_mean = train_features.mean(dim=0, keepdim=True)
    train_std = train_features.std(dim=0, keepdim=True) + 1e-6
    train_features = (train_features - train_mean) / train_std
    val_features = (val_features - train_mean) / train_std
    
    # Create dataloaders from cached features
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build linear classifier
    input_dim = train_features.shape[1]
    linear_classifier = nn.Linear(input_dim, args.num_labels).to(device)
    
    # Initialize weights
    linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
    linear_classifier.bias.data.zero_()
    
    # Optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0,
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_epoch = 0
    
    print(f"\nTraining linear classifier for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}")
    
    for epoch in range(args.epochs):
        # Training
        linear_classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = linear_classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # Validation
        linear_classifier.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = linear_classifier(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            if args.save_checkpoint:
                torch.save({
                    'epoch': epoch,
                    'linear_classifier': linear_classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'train_mean': train_mean,
                    'train_std': train_std,
                }, os.path.join(args.output_dir, 'best_linear_classifier.pth'))
    
    print(f"\nBest validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    return best_acc


def get_args_parser():
    parser = argparse.ArgumentParser('Fast linear probing', add_help=False)
    
    # Model params
    parser.add_argument('--arch', default='vit_small', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'])
    parser.add_argument('--patch_size', default=8, type=int)
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to pretrained checkpoint')
    
    # Data params
    parser.add_argument('--train_csv', default='/scratch/ss17894/DL/data/train_labels.csv', type=str)
    parser.add_argument('--val_csv', default='/scratch/ss17894/DL/data/val_labels.csv', type=str)
    parser.add_argument('--train_dir', default='/scratch/ss17894/DL/data/train', type=str)
    parser.add_argument('--val_dir', default='/scratch/ss17894/DL/data/val', type=str)
    parser.add_argument('--num_labels', default=100, type=int, help='Number of classes')
    
    # Training params
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for training (can be large since features are cached)')
    parser.add_argument('--extract_batch_size', default=256, type=int, help='Batch size for feature extraction')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # Output
    parser.add_argument('--output_dir', default='./linear_probe_results', type=str)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_features', action='store_true', help='Save extracted features to disk')
    
    return parser


def main():
    parser = argparse.ArgumentParser('Fast linear probing', parents=[get_args_parser()])
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if features are already cached
    train_features_path = os.path.join(args.output_dir, 'train_features.pt')
    val_features_path = os.path.join(args.output_dir, 'val_features.pt')
    
    if os.path.exists(train_features_path) and os.path.exists(val_features_path):
        print("\nFound cached features! Loading...")
        cache = torch.load(train_features_path)
        train_features = cache['features']
        train_labels = cache['labels']
        
        cache = torch.load(val_features_path)
        val_features = cache['features']
        val_labels = cache['labels']
        
        print(f"Loaded train features: {train_features.shape}")
        print(f"Loaded val features: {val_features.shape}")
    else:
        print("\nNo cached features found. Extracting features...")
        
        # Build model
        print(f"Building model: {args.arch}")
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=False,
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        if os.path.isfile(args.checkpoint):
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            
            if 'teacher' in state_dict:
                state_dict = state_dict['teacher']
            elif 'student' in state_dict:
                state_dict = state_dict['student']
            
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() 
                          if not k.startswith("head")}
            
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {msg}")
        else:
            raise ValueError(f"No checkpoint found at {args.checkpoint}")
        
        model = model.to(device)
        model.eval()
        
        # Data transforms (no augmentation for feature extraction)
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Load datasets
        print("\nLoading datasets...")
        train_dataset = CSVImageDataset(args.train_csv, args.train_dir, transform)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        val_dataset = CSVImageDataset(args.val_csv, args.val_dir, transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.extract_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        
        # Extract features
        print("\n" + "="*60)
        print("STEP 1: Extracting features from pretrained model")
        print("="*60)
        train_features, train_labels = extract_features(model, train_loader, device)
        val_features, val_labels = extract_features(model, val_loader, device)
        
        # Save features if requested
        if args.save_features:
            print("\nSaving features to disk...")
            torch.save({'features': train_features, 'labels': train_labels}, train_features_path)
            torch.save({'features': val_features, 'labels': val_labels}, val_features_path)
            print(f"Saved to {args.output_dir}")
        
        # Free up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Train linear classifier
    print("\n" + "="*60)
    print("STEP 2: Training linear classifier on cached features")
    print("="*60)
    best_acc = train_linear_classifier(train_features, train_labels, 
                                       val_features, val_labels, args)
    
    print(f"\n{'='*60}")
    print(f"Linear Probing Results")
    print(f"{'='*60}")
    print(f"Architecture: {args.arch}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()