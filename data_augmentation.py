"""
Data Augmentation for EEG Signals

Implements various augmentation techniques to reduce domain gap between
synthetic and real EEG data, and improve model generalization.
"""

import numpy as np
import scipy.signal as signal
from typing import Tuple, Optional, List, Dict
import random


class EEGAugmentation:
    """Collection of EEG-specific data augmentation techniques"""
    
    def __init__(self, sampling_rate: int = 250):
        self.sampling_rate = sampling_rate
    
    def add_gaussian_noise(self, 
                          eeg_data: np.ndarray, 
                          noise_level: float = 0.1) -> np.ndarray:
        """
        Add Gaussian noise to EEG signals
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            noise_level: Standard deviation of noise relative to signal std
            
        Returns:
            Augmented EEG data
        """
        signal_std = np.std(eeg_data, axis=1, keepdims=True)
        noise = np.random.normal(0, noise_level * signal_std, eeg_data.shape)
        return eeg_data + noise.astype(eeg_data.dtype)
    
    def amplitude_scaling(self, 
                         eeg_data: np.ndarray, 
                         scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Randomly scale amplitude of each channel
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            scale_range: (min_scale, max_scale) for random scaling
            
        Returns:
            Amplitude-scaled EEG data
        """
        n_channels = eeg_data.shape[0]
        scales = np.random.uniform(scale_range[0], scale_range[1], (n_channels, 1))
        return eeg_data * scales.astype(eeg_data.dtype)
    
    def time_shift(self, 
                   eeg_data: np.ndarray, 
                   max_shift_samples: int = 25) -> np.ndarray:
        """
        Apply random time shift to EEG data
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            max_shift_samples: Maximum shift in samples
            
        Returns:
            Time-shifted EEG data
        """
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        
        if shift == 0:
            return eeg_data
        elif shift > 0:
            # Shift right (delay)
            shifted = np.zeros_like(eeg_data)
            shifted[:, shift:] = eeg_data[:, :-shift]
            # Fill beginning with edge values
            shifted[:, :shift] = eeg_data[:, [0]]
        else:
            # Shift left (advance)
            shift = abs(shift)
            shifted = np.zeros_like(eeg_data)
            shifted[:, :-shift] = eeg_data[:, shift:]
            # Fill end with edge values
            shifted[:, -shift:] = eeg_data[:, [-1]]
        
        return shifted
    
    def frequency_shift(self, 
                       eeg_data: np.ndarray, 
                       max_shift_hz: float = 2.0) -> np.ndarray:
        """
        Apply random frequency shift using Hilbert transform
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            max_shift_hz: Maximum frequency shift in Hz
            
        Returns:
            Frequency-shifted EEG data
        """
        n_channels, n_times = eeg_data.shape
        shift_hz = np.random.uniform(-max_shift_hz, max_shift_hz)
        
        # Time vector
        t = np.arange(n_times) / self.sampling_rate
        
        # Apply frequency shift using complex modulation
        shifted_data = np.zeros_like(eeg_data)
        
        for ch in range(n_channels):
            # Get analytic signal
            analytic = signal.hilbert(eeg_data[ch])
            
            # Apply frequency shift
            shifted_analytic = analytic * np.exp(2j * np.pi * shift_hz * t)
            
            # Take real part
            shifted_data[ch] = np.real(shifted_analytic)
        
        return shifted_data.astype(eeg_data.dtype)
    
    def add_powerline_noise(self, 
                           eeg_data: np.ndarray, 
                           frequency: float = 60.0,
                           amplitude_range: Tuple[float, float] = (0.5, 3.0)) -> np.ndarray:
        """
        Add realistic powerline noise (50/60 Hz)
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            frequency: Powerline frequency (50 or 60 Hz)
            amplitude_range: Range of noise amplitude
            
        Returns:
            EEG data with powerline noise
        """
        n_channels, n_times = eeg_data.shape
        t = np.arange(n_times) / self.sampling_rate
        
        # Random amplitude per channel
        amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], n_channels)
        
        # Add powerline noise with slight frequency variation
        noise = np.zeros_like(eeg_data)
        for ch in range(n_channels):
            freq_variation = frequency + np.random.uniform(-0.5, 0.5)  # Slight freq variation
            phase = np.random.uniform(0, 2*np.pi)  # Random phase
            noise[ch] = amplitudes[ch] * np.sin(2 * np.pi * freq_variation * t + phase)
        
        return eeg_data + noise.astype(eeg_data.dtype)
    
    def channel_dropout(self, 
                       eeg_data: np.ndarray, 
                       dropout_prob: float = 0.1) -> np.ndarray:
        """
        Randomly set channels to zero (simulate disconnected electrodes)
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            dropout_prob: Probability of dropping each channel
            
        Returns:
            EEG data with some channels dropped
        """
        n_channels = eeg_data.shape[0]
        dropout_mask = np.random.random(n_channels) < dropout_prob
        
        augmented = eeg_data.copy()
        augmented[dropout_mask] = 0
        
        return augmented
    
    def add_muscle_artifacts(self, 
                           eeg_data: np.ndarray, 
                           artifact_prob: float = 0.2,
                           duration_range: Tuple[int, int] = (10, 50),
                           amplitude_range: Tuple[float, float] = (10, 30)) -> np.ndarray:
        """
        Add realistic muscle artifacts
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            artifact_prob: Probability of artifact per channel
            duration_range: (min, max) duration in samples
            amplitude_range: (min, max) amplitude of artifacts
            
        Returns:
            EEG data with muscle artifacts
        """
        n_channels, n_times = eeg_data.shape
        augmented = eeg_data.copy()
        
        for ch in range(n_channels):
            if np.random.random() < artifact_prob:
                # Random artifact location and duration
                duration = np.random.randint(duration_range[0], duration_range[1])
                start = np.random.randint(0, max(1, n_times - duration))
                end = min(start + duration, n_times)
                
                # Generate high-frequency muscle activity
                artifact_t = np.arange(end - start) / self.sampling_rate
                freq = np.random.uniform(50, 100)  # High frequency
                amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
                
                # Create artifact with envelope
                envelope = signal.windows.hann(len(artifact_t))
                artifact = amplitude * envelope * np.sin(2 * np.pi * freq * artifact_t)
                
                # Add to signal
                augmented[ch, start:end] += artifact.astype(eeg_data.dtype)
        
        return augmented
    
    def add_eye_blinks(self, 
                      eeg_data: np.ndarray, 
                      blink_prob: float = 0.3,
                      amplitude_range: Tuple[float, float] = (50, 150)) -> np.ndarray:
        """
        Add eye blink artifacts (mainly to frontal channels)
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            blink_prob: Probability of blink artifact
            amplitude_range: (min, max) amplitude of blinks
            
        Returns:
            EEG data with eye blink artifacts
        """
        n_channels, n_times = eeg_data.shape
        augmented = eeg_data.copy()
        
        # Eye blinks mainly affect frontal channels (first 2 channels: Fp1, Fp2)
        frontal_channels = min(2, n_channels)
        
        if np.random.random() < blink_prob:
            # Random blink timing and duration
            duration = np.random.randint(20, 60)  # Blink duration
            start = np.random.randint(50, max(51, n_times - duration - 50))
            end = min(start + duration, n_times)
            
            # Blink amplitude (negative deflection)
            amplitude = np.random.uniform(amplitude_range[0], amplitude_range[1])
            
            # Create blink waveform
            blink_samples = end - start
            blink_window = signal.windows.hann(blink_samples)
            
            # Apply to frontal channels
            for ch in range(frontal_channels):
                # Slightly different amplitude per channel
                ch_amplitude = amplitude * np.random.uniform(0.8, 1.2)
                blink_artifact = -ch_amplitude * blink_window  # Negative deflection
                
                augmented[ch, start:end] += blink_artifact.astype(eeg_data.dtype)
        
        return augmented
    
    def bandpass_filter_variation(self, 
                                 eeg_data: np.ndarray,
                                 base_low: float = 0.5,
                                 base_high: float = 40.0,
                                 variation_hz: float = 2.0) -> np.ndarray:
        """
        Apply slight variations in bandpass filtering
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            base_low: Base low cutoff frequency
            base_high: Base high cutoff frequency
            variation_hz: Maximum variation in cutoff frequencies
            
        Returns:
            Filtered EEG data with slight frequency variations
        """
        # Random variations in cutoff frequencies
        low_cutoff = base_low + np.random.uniform(-variation_hz/4, variation_hz/4)
        high_cutoff = base_high + np.random.uniform(-variation_hz, variation_hz)
        
        # Ensure valid frequency range
        low_cutoff = max(0.1, low_cutoff)
        high_cutoff = min(self.sampling_rate/2 - 1, high_cutoff)
        
        if low_cutoff >= high_cutoff:
            return eeg_data  # Skip if invalid range
        
        # Apply bandpass filter
        sos = signal.butter(4, [low_cutoff, high_cutoff], 
                           btype='band', fs=self.sampling_rate, output='sos')
        
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            filtered_data[ch] = signal.sosfiltfilt(sos, eeg_data[ch])
        
        return filtered_data.astype(eeg_data.dtype)
    
    def apply_random_augmentations(self, 
                                  eeg_data: np.ndarray,
                                  augmentation_config: Optional[Dict] = None) -> np.ndarray:
        """
        Apply a random combination of augmentations
        
        Args:
            eeg_data: (n_channels, n_times) EEG data
            augmentation_config: Configuration for augmentation probabilities
            
        Returns:
            Augmented EEG data
        """
        if augmentation_config is None:
            augmentation_config = {
                'gaussian_noise': 0.7,
                'amplitude_scaling': 0.8,
                'time_shift': 0.5,
                'powerline_noise': 0.3,
                'muscle_artifacts': 0.2,
                'eye_blinks': 0.3,
                'channel_dropout': 0.1,
                'bandpass_variation': 0.4
            }
        
        augmented = eeg_data.copy()
        
        # Apply augmentations based on probabilities
        if np.random.random() < augmentation_config.get('gaussian_noise', 0):
            augmented = self.add_gaussian_noise(augmented, np.random.uniform(0.05, 0.15))
        
        if np.random.random() < augmentation_config.get('amplitude_scaling', 0):
            augmented = self.amplitude_scaling(augmented, (0.85, 1.15))
        
        if np.random.random() < augmentation_config.get('time_shift', 0):
            augmented = self.time_shift(augmented, max_shift_samples=20)
        
        if np.random.random() < augmentation_config.get('powerline_noise', 0):
            frequency = np.random.choice([50.0, 60.0])  # EU vs US
            augmented = self.add_powerline_noise(augmented, frequency, (0.5, 2.0))
        
        if np.random.random() < augmentation_config.get('muscle_artifacts', 0):
            augmented = self.add_muscle_artifacts(augmented, 0.15, (15, 40), (8, 25))
        
        if np.random.random() < augmentation_config.get('eye_blinks', 0):
            augmented = self.add_eye_blinks(augmented, 0.25, (40, 120))
        
        if np.random.random() < augmentation_config.get('channel_dropout', 0):
            augmented = self.channel_dropout(augmented, 0.05)
        
        if np.random.random() < augmentation_config.get('bandpass_variation', 0):
            augmented = self.bandpass_filter_variation(augmented, 0.5, 40.0, 1.5)
        
        return augmented


def demo_augmentations():
    """Demonstrate various augmentation techniques"""
    from synthetic_data_generator import SyntheticEEGGenerator
    
    print("🔧 EEG Data Augmentation Demo")
    print("=" * 40)
    
    # Generate sample data
    generator = SyntheticEEGGenerator()
    original_data, label = generator.generate_window("relaxed", preprocess=True)
    
    print(f"Original data: {original_data.shape}")
    print(f"Data range: [{original_data.min():.3f}, {original_data.max():.3f}]")
    print(f"Data std: {original_data.std():.3f}")
    
    # Initialize augmentation
    augmenter = EEGAugmentation(sampling_rate=250)
    
    # Test individual augmentations
    augmentations = [
        ("Gaussian Noise", lambda x: augmenter.add_gaussian_noise(x, 0.1)),
        ("Amplitude Scaling", lambda x: augmenter.amplitude_scaling(x, (0.8, 1.2))),
        ("Time Shift", lambda x: augmenter.time_shift(x, 20)),
        ("Powerline Noise", lambda x: augmenter.add_powerline_noise(x, 60.0, (1.0, 3.0))),
        ("Muscle Artifacts", lambda x: augmenter.add_muscle_artifacts(x, 0.3, (20, 40), (15, 25))),
        ("Eye Blinks", lambda x: augmenter.add_eye_blinks(x, 0.5, (60, 100))),
        ("Channel Dropout", lambda x: augmenter.channel_dropout(x, 0.2)),
        ("Bandpass Variation", lambda x: augmenter.bandpass_filter_variation(x, 0.5, 40.0, 2.0))
    ]
    
    print(f"\nTesting individual augmentations:")
    print("-" * 40)
    
    for name, aug_func in augmentations:
        try:
            augmented = aug_func(original_data)
            print(f"✅ {name:20s}: range=[{augmented.min():.3f}, {augmented.max():.3f}], std={augmented.std():.3f}")
        except Exception as e:
            print(f"❌ {name:20s}: Error - {e}")
    
    # Test random augmentation combination
    print(f"\nTesting random augmentation combinations:")
    print("-" * 40)
    
    for i in range(5):
        try:
            augmented = augmenter.apply_random_augmentations(original_data)
            print(f"Combination {i+1}: range=[{augmented.min():.3f}, {augmented.max():.3f}], std={augmented.std():.3f}")
        except Exception as e:
            print(f"Combination {i+1}: Error - {e}")
    
    print(f"\n✅ Augmentation demo complete!")


if __name__ == "__main__":
    demo_augmentations()
