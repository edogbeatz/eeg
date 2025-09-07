#!/usr/bin/env python3
"""
Real EEG Data Training Pipeline for OpenNeuro Datasets

Downloads and processes real EEG data from OpenNeuro for meditation detection training.
Supports BIDS format and integrates with the existing LaBraM training system.
"""

import os
import shutil
import requests
import zipfile
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from urllib.parse import urlparse

# Import existing training components
from train_synthetic import SyntheticEEGTrainer
from synthetic_data_generator import SyntheticEEGGenerator

# Constants
CYTON_SAMPLING_RATE = 250
CYTON_N_CHANNELS = 8
DEFAULT_WINDOW_SECONDS = 4
DEFAULT_N_TIMES = CYTON_SAMPLING_RATE * DEFAULT_WINDOW_SECONDS


class OpenNeuroDatasetLoader:
    """Loader for OpenNeuro EEG datasets in BIDS format"""
    
    def __init__(self, dataset_id: str = "ds003969", data_dir: Path = Path("./real_data")):
        self.dataset_id = dataset_id
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # OpenNeuro download URL
        self.download_url = f"https://openneuro.org/datasets/{dataset_id}/versions/1.0.0/download"
        self.dataset_path = self.data_dir / f"{dataset_id}.zip"
        self.extracted_path = self.data_dir / dataset_id
        
    def download_dataset(self, force_redownload: bool = False) -> bool:
        """Download the OpenNeuro dataset"""
        if self.dataset_path.exists() and not force_redownload:
            print(f"✅ Dataset {self.dataset_id} already downloaded")
            return True
            
        print(f"📥 Downloading OpenNeuro dataset {self.dataset_id}...")
        print(f"   URL: {self.download_url}")
        print("   This may take several minutes...")
        
        try:
            response = requests.get(self.download_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.dataset_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
            
            print(f"\n✅ Downloaded {self.dataset_id} ({downloaded / (1024*1024):.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            if self.dataset_path.exists():
                self.dataset_path.unlink()
            return False
    
    def extract_dataset(self) -> bool:
        """Extract the downloaded dataset"""
        if not self.dataset_path.exists():
            print(f"❌ Dataset {self.dataset_id} not found. Download first.")
            return False
            
        if self.extracted_path.exists():
            print(f"✅ Dataset {self.dataset_id} already extracted")
            return True
            
        print(f"📂 Extracting {self.dataset_id}...")
        
        try:
            with zipfile.ZipFile(self.dataset_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            print(f"✅ Extracted to {self.extracted_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error extracting dataset: {e}")
            return False
    
    def find_eeg_files(self) -> List[Path]:
        """Find all EEG files in the dataset"""
        if not self.extracted_path.exists():
            print(f"❌ Dataset not extracted. Run extract_dataset() first.")
            return []
        
        # Look for .edf, .bdf, .fif files (common EEG formats)
        eeg_files = []
        for ext in ['*.edf', '*.bdf', '*.fif', '*.set', '*.vhdr']:
            eeg_files.extend(self.extracted_path.rglob(ext))
        
        print(f"📁 Found {len(eeg_files)} EEG files")
        return eeg_files
    
    def load_participants_info(self) -> Optional[pd.DataFrame]:
        """Load participants.tsv file with metadata"""
        participants_file = self.extracted_path / "participants.tsv"
        
        if not participants_file.exists():
            print("⚠️  No participants.tsv found")
            return None
            
        try:
            df = pd.read_csv(participants_file, sep='\t')
            print(f"📊 Loaded {len(df)} participant records")
            return df
        except Exception as e:
            print(f"❌ Error loading participants info: {e}")
            return None


class RealEEGDataProcessor:
    """Process real EEG data for meditation detection training"""
    
    def __init__(self, target_sr: int = CYTON_SAMPLING_RATE, target_channels: int = CYTON_N_CHANNELS):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.window_seconds = DEFAULT_WINDOW_SECONDS
        self.window_samples = int(target_sr * self.window_seconds)
        
    def load_eeg_file(self, file_path: Path) -> Optional[mne.io.Raw]:
        """Load an EEG file using MNE"""
        try:
            print(f"📖 Loading {file_path.name}...")
            
            # Try different file formats
            if file_path.suffix.lower() == '.edf':
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            elif file_path.suffix.lower() == '.bdf':
                raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
            elif file_path.suffix.lower() == '.fif':
                raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
            elif file_path.suffix.lower() == '.set':
                raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
            elif file_path.suffix.lower() == '.vhdr':
                raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            else:
                print(f"⚠️  Unsupported file format: {file_path.suffix}")
                return None
                
            print(f"   Channels: {len(raw.ch_names)}, Duration: {raw.times[-1]:.1f}s, SR: {raw.info['sfreq']:.0f}Hz")
            return raw
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None
    
    def preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply standard EEG preprocessing"""
        try:
            # Make a copy to avoid modifying original
            raw_processed = raw.copy()
            
            # Resample to target sampling rate
            if raw_processed.info['sfreq'] != self.target_sr:
                print(f"   Resampling from {raw_processed.info['sfreq']:.0f}Hz to {self.target_sr}Hz")
                raw_processed.resample(self.target_sr)
            
            # Apply bandpass filter (0.5-40 Hz for meditation detection)
            print("   Applying bandpass filter (0.5-40 Hz)")
            raw_processed.filter(l_freq=0.5, h_freq=40.0, verbose=False)
            
            # Remove bad channels if any
            if hasattr(raw_processed, 'info') and 'bads' in raw_processed.info:
                if raw_processed.info['bads']:
                    print(f"   Removing bad channels: {raw_processed.info['bads']}")
                    raw_processed.drop_channels(raw_processed.info['bads'])
            
            # Select EEG channels only
            eeg_channels = [ch for ch in raw_processed.ch_names if 'EEG' in ch.upper() or 
                           ch.upper() in ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']]
            
            if eeg_channels:
                raw_processed.pick_channels(eeg_channels[:self.target_channels])
                print(f"   Selected {len(raw_processed.ch_names)} EEG channels")
            
            return raw_processed
            
        except Exception as e:
            print(f"❌ Error preprocessing: {e}")
            return raw
    
    def extract_windows(self, raw: mne.io.Raw, label: int) -> List[Tuple[np.ndarray, int]]:
        """Extract overlapping windows from continuous EEG data"""
        data = raw.get_data()  # (n_channels, n_times)
        n_channels, n_times = data.shape
        
        # Ensure we have the right number of channels
        if n_channels > self.target_channels:
            data = data[:self.target_channels, :]
        elif n_channels < self.target_channels:
            # Pad with zeros if we have fewer channels
            padded_data = np.zeros((self.target_channels, n_times))
            padded_data[:n_channels, :] = data
            data = padded_data
        
        # Extract overlapping windows (50% overlap)
        windows = []
        step_size = self.window_samples // 2  # 50% overlap
        
        for start in range(0, n_times - self.window_samples + 1, step_size):
            end = start + self.window_samples
            window = data[:, start:end]
            
            # Apply additional preprocessing
            window = self.preprocess_window(window)
            windows.append((window, label))
        
        print(f"   Extracted {len(windows)} windows")
        return windows
    
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Apply window-level preprocessing"""
        # DC removal
        window = window - window.mean(axis=1, keepdims=True)
        
        # Z-score normalization
        std = window.std(axis=1, keepdims=True)
        window = window / (std + 1e-8)
        
        return window.astype(np.float32)


class RealDataTrainer:
    """Trainer for real EEG meditation detection"""
    
    def __init__(self, dataset_id: str = "ds003969"):
        self.dataset_id = dataset_id
        self.loader = OpenNeuroDatasetLoader(dataset_id)
        self.processor = RealEEGDataProcessor()
        self.trainer = SyntheticEEGTrainer()  # Reuse existing trainer
        
    def download_and_prepare(self) -> bool:
        """Download and prepare the dataset"""
        print(f"🧠 Preparing OpenNeuro dataset {self.dataset_id}")
        print("=" * 50)
        
        # Download dataset
        if not self.loader.download_dataset():
            return False
        
        # Extract dataset
        if not self.loader.extract_dataset():
            return False
        
        return True
    
    def load_and_process_data(self) -> Optional[Dict]:
        """Load and process all EEG data"""
        eeg_files = self.loader.find_eeg_files()
        if not eeg_files:
            print("❌ No EEG files found in dataset")
            return None
        
        all_windows = []
        all_labels = []
        
        # Load participants info to determine meditation sessions
        participants_df = self.loader.load_participants_info()
        
        print(f"\n📊 Processing {len(eeg_files)} EEG files...")
        
        for i, eeg_file in enumerate(eeg_files[:10]):  # Limit to first 10 files for testing
            print(f"\nFile {i+1}/{min(10, len(eeg_files))}: {eeg_file.name}")
            
            # Load EEG file
            raw = self.processor.load_eeg_file(eeg_file)
            if raw is None:
                continue
            
            # Preprocess
            raw_processed = self.processor.preprocess_raw(raw)
            
            # Determine label based on filename or metadata
            # This is dataset-specific - you'll need to adjust based on the actual dataset structure
            label = self.determine_label(eeg_file, participants_df)
            
            # Extract windows
            windows = self.processor.extract_windows(raw_processed, label)
            
            for window, window_label in windows:
                all_windows.append(window)
                all_labels.append(window_label)
        
        if not all_windows:
            print("❌ No windows extracted from dataset")
            return None
        
        # Convert to arrays
        data_array = np.stack(all_windows, axis=0)  # (n_samples, n_channels, n_times)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        print(f"\n✅ Dataset processed successfully!")
        print(f"   Total windows: {len(data_array)}")
        print(f"   Shape: {data_array.shape}")
        print(f"   Label distribution: {np.bincount(labels_array)}")
        
        return {
            'data': data_array,
            'labels': labels_array,
            'metadata': {
                'dataset_id': self.dataset_id,
                'n_samples': len(data_array),
                'n_channels': data_array.shape[1],
                'n_times': data_array.shape[2],
                'sampling_rate': self.processor.target_sr,
                'window_seconds': self.processor.window_seconds,
                'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
    
    def determine_label(self, eeg_file: Path, participants_df: Optional[pd.DataFrame]) -> int:
        """Determine meditation label based on file path and metadata"""
        # This is dataset-specific logic that needs to be customized
        # based on the actual structure of ds003969
        
        file_name = eeg_file.name.lower()
        file_path = str(eeg_file).lower()
        
        # Common meditation/mindfulness keywords
        meditation_keywords = ['meditation', 'meditat', 'mindful', 'rest', 'relax', 'calm']
        active_keywords = ['task', 'active', 'cognitive', 'attention', 'focus', 'working']
        
        # Check filename for meditation indicators
        if any(keyword in file_name or keyword in file_path for keyword in meditation_keywords):
            return 1  # Meditation
        elif any(keyword in file_name or keyword in file_path for keyword in active_keywords):
            return 0  # Non-meditation
        else:
            # Default: assume alternating pattern or random assignment
            # This should be replaced with proper metadata parsing
            return hash(file_name) % 2  # Pseudo-random but consistent
    
    def train_on_real_data(self, 
                          n_epochs: int = 20,
                          val_split: float = 0.2,
                          save_dir: Path = Path("./real_trained_models")) -> Optional[Dict]:
        """Complete training pipeline on real data"""
        print("🧠 Real EEG Meditation Detection Training")
        print("=" * 50)
        
        # Prepare dataset
        if not self.download_and_prepare():
            return None
        
        # Load and process data
        dataset = self.load_and_process_data()
        if dataset is None:
            return None
        
        # Setup trainer
        print("\n🔧 Setting up trainer...")
        weights_dir = Path("./weights")
        pretrained_path = weights_dir / "labram_checkpoint.pth"
        
        self.trainer.setup_model(
            freeze_backbone=True,  # Fine-tune on real data
            pretrained_path=pretrained_path if pretrained_path.exists() else None
        )
        
        # Create data loaders from real data
        train_loader, val_loader = self.create_data_loaders(dataset, val_split)
        
        # Train the model
        print(f"\n🎓 Training on real EEG data...")
        history = self.train_model(train_loader, val_loader, n_epochs, save_dir)
        
        return history
    
    def create_data_loaders(self, dataset: Dict, val_split: float = 0.2):
        """Create PyTorch data loaders from real data"""
        from torch.utils.data import Dataset, DataLoader
        import torch
        
        class RealEEGDataset(Dataset):
            def __init__(self, data: np.ndarray, labels: np.ndarray):
                self.data = torch.from_numpy(data).float()
                self.labels = torch.from_numpy(labels).long()
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        # Split data
        data = dataset['data']
        labels = dataset['labels']
        
        n_total = len(data)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create datasets
        train_dataset = RealEEGDataset(data[train_indices], labels[train_indices])
        val_dataset = RealEEGDataset(data[val_indices], labels[val_indices])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        print(f"📊 Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val")
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, n_epochs: int, save_dir: Path) -> Dict:
        """Train the model on real data"""
        import torch
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0
        history = {
            'train_losses': [],
            'val_losses': [],
            'train_accs': [],
            'val_accs': []
        }
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}/{n_epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.trainer.validate(val_loader)
            
            # Record metrics
            history['train_losses'].append(train_loss)
            history['val_losses'].append(val_loss)
            history['train_accs'].append(train_acc)
            history['val_accs'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.trainer.model.state_dict(),
                    'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'dataset_id': self.dataset_id,
                    'n_channels': CYTON_N_CHANNELS,
                    'n_times': DEFAULT_N_TIMES,
                    'n_classes': 2
                }, save_dir / "best_real_model.pth")
                
                print(f"💾 New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save training history
        history['best_val_acc'] = best_val_acc
        history['dataset_id'] = self.dataset_id
        
        with open(save_dir / "real_training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return history


def main():
    """Main function to train on OpenNeuro dataset"""
    print("🌐 OpenNeuro Real EEG Training Pipeline")
    print("=" * 50)
    
    # Create trainer
    trainer = RealDataTrainer("ds003969")
    
    # Train on real data
    history = trainer.train_on_real_data(
        n_epochs=15,
        val_split=0.2,
        save_dir=Path("./real_trained_models")
    )
    
    if history:
        print("\n✅ Real data training completed!")
        print("📁 Check ./real_trained_models/ for the trained model")
        print("\n💡 Next steps:")
        print("1. Copy best_real_model.pth to weights/labram_checkpoint.pth")
        print("2. Restart the API server to use the real data model")
        print("3. Test meditation detection with improved accuracy!")
    else:
        print("❌ Training failed. Check dataset and try again.")


if __name__ == "__main__":
    main()
