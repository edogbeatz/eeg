"""
Training Pipeline for Synthetic EEG Data

This script demonstrates how to:
1. Generate synthetic "relaxed vs anxious" EEG data
2. Train a LaBraM model or just the classification head
3. Save the trained model for deployment
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

try:
    from braindecode.models import Labram
    LABRAM_AVAILABLE = True
except ImportError:
    print("⚠️  LaBraM model not available. Install braindecode>=1.1.0")
    LABRAM_AVAILABLE = False

from synthetic_data_generator import SyntheticEEGGenerator

# Training constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CYTON_SAMPLING_RATE = 250
CYTON_N_CHANNELS = 8
DEFAULT_WINDOW_SECONDS = 4
DEFAULT_N_TIMES = CYTON_SAMPLING_RATE * DEFAULT_WINDOW_SECONDS  # 1000 samples


class SyntheticEEGDataset(Dataset):
    """PyTorch Dataset for synthetic EEG data"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        """
        Args:
            data: (n_samples, n_channels, n_times) array
            labels: (n_samples,) array of class labels
        """
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleClassificationHead(nn.Module):
    """Simple classification head for LaBraM features"""
    
    def __init__(self, feature_dim: int = 512, n_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, features):
        return self.classifier(features)


class SyntheticEEGTrainer:
    """Trainer for synthetic EEG classification"""
    
    def __init__(self, 
                 n_channels: int = CYTON_N_CHANNELS,
                 n_times: int = DEFAULT_N_TIMES,
                 n_classes: int = 2,
                 device: torch.device = DEVICE):
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.device = device
        
        # Model components
        self.backbone = None
        self.classifier = None
        self.model = None
        
        # Training state
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def setup_model(self, 
                   freeze_backbone: bool = True,
                   pretrained_path: Optional[Path] = None):
        """Setup the model architecture"""
        if not LABRAM_AVAILABLE:
            raise ValueError("LaBraM not available. Install braindecode>=1.1.0")
        
        # Create LaBraM backbone
        self.backbone = Labram(
            n_chans=self.n_channels,
            n_times=self.n_times,
            n_outputs=self.n_classes,
            neural_tokenizer=True
        )
        
        # Load pretrained weights if available
        if pretrained_path and pretrained_path.exists():
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=self.device)
            state_dict = state_dict.get("state_dict", state_dict)
            self.backbone.load_state_dict(state_dict, strict=False)
        
        # Freeze backbone if requested (feature extractor mode)
        if freeze_backbone:
            print("Freezing LaBraM backbone - training only classification head")
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Only train the final classification layer and layer norm
            for param in self.backbone.final_layer.parameters():
                param.requires_grad = True
            for param in self.backbone.fc_norm.parameters():
                param.requires_grad = True
        else:
            print("Training full LaBraM model")
        
        self.model = self.backbone.to(self.device)
        
        # Setup optimizer (only trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)
        
        print(f"Model setup complete. Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def generate_data(self, 
                     n_samples_per_class: int = 1000,
                     val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """Generate synthetic dataset and create data loaders"""
        print(f"Generating {n_samples_per_class * 2} synthetic samples...")
        
        generator = SyntheticEEGGenerator(
            n_channels=self.n_channels,
            sampling_rate=CYTON_SAMPLING_RATE,
            window_seconds=DEFAULT_WINDOW_SECONDS
        )
        
        dataset = generator.generate_dataset(n_samples_per_class)
        
        # Split into train/val
        n_total = len(dataset['data'])
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_dataset = SyntheticEEGDataset(
            dataset['data'][train_indices], 
            dataset['labels'][train_indices]
        )
        val_dataset = SyntheticEEGDataset(
            dataset['data'][val_indices], 
            dataset['labels'][val_indices]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=32, 
            shuffle=True, 
            num_workers=0  # Avoid multiprocessing issues
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=0
        )
        
        print(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, 
             n_epochs: int = 10,
             n_samples_per_class: int = 1000,
             val_split: float = 0.2,
             save_dir: Path = Path("./trained_models")) -> Dict:
        """Complete training pipeline"""
        print("=== Starting Synthetic EEG Training ===\n")
        
        # Generate data
        train_loader, val_loader = self.generate_data(n_samples_per_class, val_split)
        
        # Training loop
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'n_channels': self.n_channels,
                    'n_times': self.n_times,
                    'n_classes': self.n_classes
                }, save_dir / "best_synthetic_model.pth")
                
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }
        
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.val_losses, label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(self.train_accs, label='Train Acc', marker='o')
        ax2.plot(self.val_accs, label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


def main():
    """Main training script"""
    print("🧠 Synthetic EEG Training Pipeline")
    print("=" * 40)
    
    # Setup trainer
    trainer = SyntheticEEGTrainer()
    
    # Check for existing LaBraM weights
    weights_dir = Path("./weights")
    pretrained_path = weights_dir / "labram_checkpoint.pth"
    
    # Setup model (freeze backbone for faster training on synthetic data)
    trainer.setup_model(
        freeze_backbone=True,  # Only train classification head
        pretrained_path=pretrained_path if pretrained_path.exists() else None
    )
    
    # Train the model
    history = trainer.train(
        n_epochs=15,
        n_samples_per_class=1500,  # 3000 total samples
        val_split=0.2,
        save_dir=Path("./trained_models")
    )
    
    # Plot results
    try:
        trainer.plot_training_history(Path("./trained_models/training_curves.png"))
    except Exception as e:
        print(f"Could not plot training curves: {e}")
    
    print("\n✅ Training complete!")
    print("📁 Check ./trained_models/ for saved model and history")
    print("\n💡 Next steps:")
    print("1. Test the model with: python test_synthetic_model.py")
    print("2. Deploy to FastAPI by updating CHECKPOINT_PATH in main.py")
    print("3. Collect real EEG data for fine-tuning")


if __name__ == "__main__":
    main()
