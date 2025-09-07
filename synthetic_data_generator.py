"""
Synthetic EEG Data Generator for LaBraM Training

Generates realistic synthetic EEG data for "relaxed vs anxious" classification
using spectral characteristics and artifacts. Compatible with BrainFlow pipeline.
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import json
from data_augmentation import EEGAugmentation

# Cyton board constants
CYTON_SAMPLING_RATE = 250  # Hz
CYTON_N_CHANNELS = 8
DEFAULT_WINDOW_SECONDS = 4
DEFAULT_N_TIMES = CYTON_SAMPLING_RATE * DEFAULT_WINDOW_SECONDS  # 1000 samples

# EEG frequency bands (Hz)
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 12)
BETA_BAND = (12, 30)
GAMMA_BAND = (30, 40)

# Channel names for 8-channel Cyton setup
CHANNEL_NAMES = ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']


class SyntheticEEGGenerator:
    """Generate realistic synthetic EEG data with different mental states"""
    
    def __init__(self, 
                 n_channels: int = CYTON_N_CHANNELS,
                 sampling_rate: int = CYTON_SAMPLING_RATE,
                 window_seconds: float = DEFAULT_WINDOW_SECONDS):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.window_seconds = window_seconds
        self.n_times = int(sampling_rate * window_seconds)
        
        # Time vector
        self.t = np.arange(self.n_times) / sampling_rate
        
        # Initialize augmentation
        self.augmenter = EEGAugmentation(sampling_rate)
        
        # Random seed for reproducibility
        np.random.seed(42)
    
    def generate_base_noise(self, amplitude: float = 1.0) -> np.ndarray:
        """Generate 1/f pink noise as base EEG signal"""
        # Generate white noise
        white = np.random.randn(self.n_channels, self.n_times * 2)  # Extra length for filtering
        
        # Apply 1/f filter in frequency domain
        fft_white = np.fft.fft(white, axis=1)
        freqs = np.fft.fftfreq(white.shape[1], 1/self.sampling_rate)
        
        # 1/f filter (avoid division by zero)
        filter_1f = 1 / np.sqrt(np.maximum(np.abs(freqs), 0.1))
        filter_1f = filter_1f[np.newaxis, :]  # Broadcast across channels
        
        # Apply filter and convert back
        pink_fft = fft_white * filter_1f
        pink_noise = np.real(np.fft.ifft(pink_fft, axis=1))
        
        # Take middle portion and scale
        start_idx = self.n_times // 2
        pink_signal = pink_noise[:, start_idx:start_idx + self.n_times]
        
        return pink_signal * amplitude
    
    def generate_band_oscillations(self, 
                                   freq_range: Tuple[float, float], 
                                   amplitude: float = 1.0,
                                   phase_coupling: bool = False) -> np.ndarray:
        """Generate oscillations in specific frequency band"""
        low_freq, high_freq = freq_range
        signal = np.zeros((self.n_channels, self.n_times))
        
        for ch in range(self.n_channels):
            # Random frequency within band
            freq = np.random.uniform(low_freq, high_freq)
            
            # Random phase
            phase = np.random.uniform(0, 2*np.pi) if not phase_coupling else 0
            
            # Generate oscillation with some frequency modulation
            freq_mod = 1 + 0.1 * np.sin(2 * np.pi * 0.5 * self.t)  # 0.5 Hz modulation
            instantaneous_phase = 2 * np.pi * freq * self.t * freq_mod + phase
            
            # Add some amplitude modulation
            amp_mod = 1 + 0.2 * np.sin(2 * np.pi * 0.3 * self.t)  # 0.3 Hz modulation
            
            signal[ch] = amplitude * amp_mod * np.sin(instantaneous_phase)
        
        return signal
    
    def add_artifacts(self, signal: np.ndarray, artifact_prob: float = 0.15) -> np.ndarray:
        """Add realistic EEG artifacts (blinks, muscle, etc.)"""
        signal_with_artifacts = signal.copy()
        
        for ch in range(self.n_channels):
            # Eye blink artifacts (mainly frontal channels)
            if ch < 2 and np.random.rand() < artifact_prob:  # Fp1, Fp2
                blink_start = np.random.randint(50, self.n_times - 100)
                blink_duration = np.random.randint(25, 75)
                blink_end = min(blink_start + blink_duration, self.n_times)
                
                # Blink shape (negative deflection)
                blink_window = np.hanning(blink_end - blink_start)
                blink_amplitude = np.random.uniform(50, 150)  # microvolts
                signal_with_artifacts[ch, blink_start:blink_end] -= blink_amplitude * blink_window
            
            # Muscle artifacts (higher frequency, random channels)
            if np.random.rand() < artifact_prob * 0.5:
                muscle_start = np.random.randint(0, self.n_times - 50)
                muscle_duration = np.random.randint(20, 50)
                muscle_end = min(muscle_start + muscle_duration, self.n_times)
                
                # High-frequency muscle activity
                muscle_freq = np.random.uniform(50, 100)
                muscle_t = np.arange(muscle_end - muscle_start) / self.sampling_rate
                muscle_signal = np.random.uniform(10, 30) * np.sin(2 * np.pi * muscle_freq * muscle_t)
                muscle_signal *= np.hanning(len(muscle_signal))  # Smooth onset/offset
                
                signal_with_artifacts[ch, muscle_start:muscle_end] += muscle_signal
            
            # Power line noise (50/60 Hz)
            if np.random.rand() < 0.3:  # 30% chance
                line_freq = 60.0  # Hz (use 50 for Europe)
                line_amplitude = np.random.uniform(2, 8)
                line_noise = line_amplitude * np.sin(2 * np.pi * line_freq * self.t)
                signal_with_artifacts[ch] += line_noise
        
        return signal_with_artifacts
    
    def generate_relaxed_state(self, amplitude_scale: float = 1.0) -> np.ndarray:
        """Generate EEG characteristic of relaxed state"""
        # Base pink noise
        base_signal = self.generate_base_noise(amplitude=10 * amplitude_scale)
        
        # Strong alpha waves (relaxed, eyes closed)
        alpha_signal = self.generate_band_oscillations(
            ALPHA_BAND, 
            amplitude=25 * amplitude_scale,
            phase_coupling=True  # Synchronized alpha
        )
        
        # Moderate theta (drowsy relaxation)
        theta_signal = self.generate_band_oscillations(
            THETA_BAND, 
            amplitude=15 * amplitude_scale
        )
        
        # Low beta (reduced mental activity)
        beta_signal = self.generate_band_oscillations(
            BETA_BAND, 
            amplitude=8 * amplitude_scale
        )
        
        # Combine signals
        eeg_signal = base_signal + alpha_signal + theta_signal + beta_signal
        
        # Add minimal artifacts (relaxed state)
        eeg_signal = self.add_artifacts(eeg_signal, artifact_prob=0.1)
        
        return eeg_signal
    
    def generate_anxious_state(self, amplitude_scale: float = 1.0) -> np.ndarray:
        """Generate EEG characteristic of anxious state"""
        # Base pink noise (higher amplitude due to tension)
        base_signal = self.generate_base_noise(amplitude=15 * amplitude_scale)
        
        # Reduced alpha waves (active, alert state)
        alpha_signal = self.generate_band_oscillations(
            ALPHA_BAND, 
            amplitude=8 * amplitude_scale
        )
        
        # Increased beta waves (anxiety, worry)
        beta_signal = self.generate_band_oscillations(
            BETA_BAND, 
            amplitude=30 * amplitude_scale
        )
        
        # Some gamma activity (high arousal)
        gamma_signal = self.generate_band_oscillations(
            GAMMA_BAND, 
            amplitude=12 * amplitude_scale
        )
        
        # Elevated theta (emotional processing)
        theta_signal = self.generate_band_oscillations(
            THETA_BAND, 
            amplitude=20 * amplitude_scale
        )
        
        # Combine signals
        eeg_signal = base_signal + alpha_signal + beta_signal + gamma_signal + theta_signal
        
        # Add more artifacts (tension, movement)
        eeg_signal = self.add_artifacts(eeg_signal, artifact_prob=0.25)
        
        return eeg_signal
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply standard EEG preprocessing"""
        # DC removal (per channel)
        signal_dc_removed = signal - signal.mean(axis=1, keepdims=True)
        
        # Z-score normalization (per channel)
        signal_std = signal_dc_removed.std(axis=1, keepdims=True)
        signal_normalized = signal_dc_removed / (signal_std + 1e-8)
        
        return signal_normalized.astype(np.float32)
    
    def generate_window(self, 
                       state: str = "relaxed", 
                       preprocess: bool = True,
                       amplitude_variation: float = 0.3,
                       apply_augmentation: bool = False) -> Tuple[np.ndarray, int]:
        """
        Generate a single EEG window
        
        Args:
            state: "relaxed" or "anxious"
            preprocess: Apply DC removal and z-scoring
            amplitude_variation: Random amplitude scaling factor
            apply_augmentation: Apply realistic augmentations to reduce domain gap
            
        Returns:
            Tuple of (eeg_data, label) where:
            - eeg_data: (n_channels, n_times) array
            - label: 0 for relaxed, 1 for anxious
        """
        # Random amplitude variation
        amp_scale = 1.0 + np.random.uniform(-amplitude_variation, amplitude_variation)
        
        if state.lower() == "relaxed":
            signal = self.generate_relaxed_state(amp_scale)
            label = 0
        elif state.lower() == "anxious":
            signal = self.generate_anxious_state(amp_scale)
            label = 1
        else:
            raise ValueError(f"Unknown state: {state}. Use 'relaxed' or 'anxious'")
        
        # Apply augmentation before preprocessing to simulate real-world conditions
        if apply_augmentation:
            signal = self.augmenter.apply_random_augmentations(signal)
        
        if preprocess:
            signal = self.preprocess_signal(signal)
        
        return signal, label
    
    def generate_dataset(self, 
                        n_samples_per_class: int = 500,
                        output_dir: Optional[Path] = None,
                        save_format: str = "npy",
                        augmentation_ratio: float = 0.5) -> Dict:
        """
        Generate a complete dataset
        
        Args:
            n_samples_per_class: Number of samples per class
            output_dir: Directory to save data (optional)
            save_format: "npy" or "pt" (PyTorch)
            augmentation_ratio: Fraction of samples to apply augmentation (0.0 to 1.0)
            
        Returns:
            Dictionary with data, labels, and metadata
        """
        print(f"Generating {n_samples_per_class * 2} synthetic EEG samples...")
        
        all_data = []
        all_labels = []
        
        # Generate relaxed samples
        print("Generating relaxed samples...")
        for i in range(n_samples_per_class):
            if i % 100 == 0:
                print(f"  Progress: {i}/{n_samples_per_class}")
            
            # Apply augmentation to a fraction of samples
            apply_aug = np.random.random() < augmentation_ratio
            data, label = self.generate_window("relaxed", apply_augmentation=apply_aug)
            all_data.append(data)
            all_labels.append(label)
        
        # Generate anxious samples
        print("Generating anxious samples...")
        for i in range(n_samples_per_class):
            if i % 100 == 0:
                print(f"  Progress: {i}/{n_samples_per_class}")
            
            # Apply augmentation to a fraction of samples
            apply_aug = np.random.random() < augmentation_ratio
            data, label = self.generate_window("anxious", apply_augmentation=apply_aug)
            all_data.append(data)
            all_labels.append(label)
        
        # Convert to arrays
        data_array = np.stack(all_data, axis=0)  # (n_samples, n_channels, n_times)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        # Shuffle
        indices = np.random.permutation(len(data_array))
        data_array = data_array[indices]
        labels_array = labels_array[indices]
        
        dataset = {
            'data': data_array,
            'labels': labels_array,
            'metadata': {
                'n_samples': len(data_array),
                'n_channels': self.n_channels,
                'n_times': self.n_times,
                'sampling_rate': self.sampling_rate,
                'window_seconds': self.window_seconds,
                'channel_names': CHANNEL_NAMES[:self.n_channels],
                'class_names': ['relaxed', 'anxious'],
                'augmentation_ratio': augmentation_ratio,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Save if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if save_format == "npy":
                np.save(output_dir / "synthetic_eeg_data.npy", data_array)
                np.save(output_dir / "synthetic_eeg_labels.npy", labels_array)
                
                # Save metadata as JSON
                with open(output_dir / "metadata.json", "w") as f:
                    json.dump(dataset['metadata'], f, indent=2)
                    
                print(f"Saved dataset to {output_dir} (NumPy format)")
                
            elif save_format == "pt":
                import torch
                torch.save({
                    'data': torch.from_numpy(data_array),
                    'labels': torch.from_numpy(labels_array),
                    'metadata': dataset['metadata']
                }, output_dir / "synthetic_eeg_dataset.pt")
                
                print(f"Saved dataset to {output_dir} (PyTorch format)")
        
        print(f"Dataset generation complete!")
        print(f"Shape: {data_array.shape}")
        print(f"Labels: {np.bincount(labels_array)} samples per class")
        
        return dataset


def main():
    """Generate synthetic dataset for training"""
    generator = SyntheticEEGGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset(
        n_samples_per_class=1000,
        output_dir=Path("./synthetic_data"),
        save_format="both"  # Save both formats
    )
    
    print("\nDataset statistics:")
    print(f"Data shape: {dataset['data'].shape}")
    print(f"Data range: [{dataset['data'].min():.3f}, {dataset['data'].max():.3f}]")
    print(f"Data mean: {dataset['data'].mean():.3f}")
    print(f"Data std: {dataset['data'].std():.3f}")
    
    # Generate a few examples for testing
    print("\nGenerating test examples...")
    for state in ["relaxed", "anxious"]:
        data, label = generator.generate_window(state)
        print(f"{state.capitalize()} example:")
        print(f"  Shape: {data.shape}")
        print(f"  Label: {label}")
        print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")


if __name__ == "__main__":
    main()
