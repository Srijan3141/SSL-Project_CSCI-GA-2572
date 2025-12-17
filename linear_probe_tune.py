"""
Hyperparameter tuning for linear probing
Runs grid search over key hyperparameters and saves results
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
import models
import itertools
import json
import sys
from datetime import datetime


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


def train_linear_classifier(train_features, train_labels, val_features, val_labels, 
                           config, device, verbose=True):
    """Train linear classifier with specific hyperparameters"""
    
    # Normalize features based on config
    if config['normalization'] == 'standardize':
        train_mean = train_features.mean(dim=0, keepdim=True)
        train_std = train_features.std(dim=0, keepdim=True) + 1e-6
        train_features_norm = (train_features - train_mean) / train_std
        val_features_norm = (val_features - train_mean) / train_std
    elif config['normalization'] == 'l2':
        train_features_norm = F.normalize(train_features, p=2, dim=1)
        val_features_norm = F.normalize(val_features, p=2, dim=1)
    else:  # none
        train_features_norm = train_features
        val_features_norm = val_features
    
    # Create dataloaders
    train_dataset = TensorDataset(train_features_norm, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, drop_last=True)
    
    val_dataset = TensorDataset(val_features_norm, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Build linear classifier
    input_dim = train_features_norm.shape[1]
    linear_classifier = nn.Linear(input_dim, config['num_labels']).to(device)
    
    # Initialize weights
    linear_classifier.weight.data.normal_(mean=0.0, std=0.01)
    linear_classifier.bias.data.zero_()
    
    # Optimizer
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            lr=config['lr'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
        )
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            linear_classifier.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            linear_classifier.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
    
    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config['epochs'], eta_min=0
        )
    elif config['scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config['epochs']//3, gamma=0.1
        )
    else:  # none
        scheduler = None
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_epoch = 0
    train_accs = []
    val_accs = []
    
    for epoch in range(config['epochs']):
        # Training
        linear_classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Optional: add noise augmentation
            if config['feature_noise'] > 0:
                features = features + torch.randn_like(features) * config['feature_noise']
            
            outputs = linear_classifier(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        if scheduler is not None:
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
        val_acc = 100. * val_correct / val_total
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        if verbose and (epoch % 10 == 0 or epoch == config['epochs'] - 1):
            print(f"  Epoch [{epoch+1:3d}/{config['epochs']}] "
                  f"Train Acc: {train_acc:5.2f}% | Val Acc: {val_acc:5.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
    
    return {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'final_train_acc': train_accs[-1],
        'final_val_acc': val_accs[-1],
    }


def run_grid_search(train_features, train_labels, val_features, val_labels, 
                   search_space, base_config, output_dir):
    """Run grid search over hyperparameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate all combinations
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    print(f"\nRunning grid search with {len(combinations)} configurations...")
    print(f"Search space: {search_space}\n")
    
    results = []
    
    # Save timestamp for consistent naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, combo in enumerate(combinations):
        config = base_config.copy()
        config.update(dict(zip(keys, combo)))
        
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}/{len(combinations)}")
        print(f"{'='*60}")
        for k, v in config.items():
            if k in search_space:
                print(f"  {k}: {v}")
        
        try:
            result = train_linear_classifier(
                train_features, train_labels, 
                val_features, val_labels,
                config, device, verbose=True
            )
            
            result['config'] = config
            results.append(result)
            
            print(f"\n  Best Val Acc: {result['best_acc']:.2f}% (epoch {result['best_epoch']})")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Sort by best accuracy
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    # Save results
    results_file = os.path.join(output_dir, f'grid_search_results_{timestamp}.json')
    
    # Convert to serializable format
    serializable_results = []
    for r in results:
        serializable_results.append({
            'best_acc': r['best_acc'],
            'best_epoch': r['best_epoch'],
            'final_train_acc': r['final_train_acc'],
            'final_val_acc': r['final_val_acc'],
            'config': r['config']
        })
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save to CSV as well
    csv_file = os.path.join(output_dir, f'grid_search_results_{timestamp}.csv')
    csv_data = []
    for r in serializable_results:
        row = {
            'best_val_acc': r['best_acc'],
            'best_epoch': r['best_epoch'],
            'final_train_acc': r['final_train_acc'],
            'final_val_acc': r['final_val_acc'],
        }
        # Add config params
        for k, v in r['config'].items():
            if k in search_space.keys():
                row[k] = v
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    # Sort by best validation accuracy
    df = df.sort_values('best_val_acc', ascending=False)
    df.to_csv(csv_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"Grid Search Complete")
    print(f"Results saved to:")
    print(f"  JSON: {results_file}")
    print(f"  CSV:  {csv_file}")
    print(f"{'='*60}")
    
    # Print top 5 configurations
    print("\nTop 5 Configurations:")
    print(f"{'='*60}")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Val Acc: {result['best_acc']:.2f}%")
        print(f"   Config:")
        for k, v in result['config'].items():
            if k in search_space:
                print(f"     {k}: {v}")
    
    return results


def get_args_parser():
    parser = argparse.ArgumentParser('Linear probing with hyperparameter tuning', add_help=False)
    
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
    
    # Feature extraction
    parser.add_argument('--extract_batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    
    # Grid search mode
    parser.add_argument('--grid_search', action='store_true', 
                        help='Run grid search over hyperparameters')
    
    # Hyperparameters for grid search
    parser.add_argument('--search_lr', nargs='+', type=float, 
                        default=[0.001, 0.01, 0.1],
                        help='Learning rates to search')
    parser.add_argument('--search_wd', nargs='+', type=float,
                        default=[0, 1e-6, 1e-5, 1e-4],
                        help='Weight decay values to search')
    parser.add_argument('--search_optimizer', nargs='+', type=str,
                        default=['sgd', 'adamw'],
                        help='Optimizers to search')
    parser.add_argument('--search_batch_size', nargs='+', type=int,
                        default=[512, 1024],
                        help='Batch sizes to search')
    parser.add_argument('--search_normalization', nargs='+', type=str,
                        default=['standardize', 'l2'],
                        help='Normalization methods to search')
    
    # Single run hyperparameters
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine', 'step', 'none'])
    parser.add_argument('--normalization', default='standardize', type=str, 
                        choices=['standardize', 'l2', 'none'])
    parser.add_argument('--feature_noise', default=0.0, type=float,
                        help='Add Gaussian noise to features during training')
    
    # Output
    parser.add_argument('--output_dir', default='./linear_probe_results', type=str)
    parser.add_argument('--save_checkpoint', action='store_true')
    
    return parser


def main():
    parser = argparse.ArgumentParser('Linear probing with tuning', parents=[get_args_parser()])
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f'training_log_{timestamp}.txt')
    
    # Redirect stdout to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_file)
    print(f"Logging to: {log_file}")
    
    # Use DataParallel for multi-GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Using {n_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")
    else:
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
        
        # Use DataParallel for multi-GPU feature extraction
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs for feature extraction")
            model = nn.DataParallel(model)
        
        model.eval()
        
        # Data transforms
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
        print("Extracting features from pretrained model")
        print("="*60)
        train_features, train_labels = extract_features(model, train_loader, device)
        val_features, val_labels = extract_features(model, val_loader, device)
        
        # Save features
        print("\nSaving features to disk...")
        torch.save({'features': train_features, 'labels': train_labels}, train_features_path)
        torch.save({'features': val_features, 'labels': val_labels}, val_features_path)
        print(f"Saved to {args.output_dir}")
        
        # Free up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Run grid search or single training
    if args.grid_search:
        print("\n" + "="*60)
        print("Running Grid Search")
        print("="*60)
        
        search_space = {
            'lr': args.search_lr,
            'weight_decay': args.search_wd,
            'optimizer': args.search_optimizer,
            'batch_size': args.search_batch_size,
            'normalization': args.search_normalization,
        }
        
        base_config = {
            'epochs': args.epochs,
            'num_labels': args.num_labels,
            'momentum': args.momentum,
            'scheduler': args.scheduler,
            'feature_noise': args.feature_noise,
        }
        
        results = run_grid_search(
            train_features, train_labels,
            val_features, val_labels,
            search_space, base_config,
            args.output_dir
        )
        
    else:
        print("\n" + "="*60)
        print("Training with single configuration")
        print("="*60)
        
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'momentum': args.momentum,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'normalization': args.normalization,
            'feature_noise': args.feature_noise,
            'num_labels': args.num_labels,
        }
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = train_linear_classifier(
            train_features, train_labels,
            val_features, val_labels,
            config, device, verbose=True
        )
        
        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"{'='*60}")
        print(f"Best Validation Accuracy: {result['best_acc']:.2f}%")
        print(f"Best Epoch: {result['best_epoch']}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()