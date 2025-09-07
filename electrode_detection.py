"""
Electrode Detection Module for OpenBCI Cyton Board

This module implements electrode detection using impedance testing and live signal monitoring
as described in the OpenBCI documentation.
"""

import math
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  BrainFlow not installed. Install with: pip install brainflow")
    BRAINFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenBCI Cyton constants
CYTON_DRIVE_CURRENT = 6e-9  # 6 nA
CYTON_SERIES_RESISTOR = 2200  # 2.2 kΩ
CYTON_SCALE_FACTOR = 0.02235e-6  # µV/count at gain x24
CYTON_SAMPLING_RATE = 250  # Hz
CYTON_TEST_FREQUENCY = 31.5  # Hz

# Impedance thresholds (kΩ)
IMPEDANCE_THRESHOLDS = {
    "good": 750,      # < 750 kΩ = good contact
    "moderate": 1500,  # 750-1500 kΩ = moderate/OK
    "poor": 5000      # > 1500 kΩ or ~5000 kΩ = disconnected
}

# Signal amplitude thresholds for railing detection
RAILING_THRESHOLD = 0.8  # 80% of full scale
MIN_SIGNAL_STD = 1.0     # Minimum std for connected electrode
MAX_SIGNAL_STD = 100.0   # Maximum std for good contact


class ElectrodeDetector:
    """
    Electrode detection class for OpenBCI Cyton board.
    
    Implements both impedance-based detection and live signal monitoring.
    """
    
    def __init__(self, board: BoardShim):
        """
        Initialize electrode detector.
        
        Args:
            board: Initialized and prepared BoardShim instance
        """
        self.board = board
        self.board_id = board.get_board_id()
        self.eeg_channels = BoardShim.get_eeg_channels(self.board_id)
        self.scale_factor = self._get_scale_factor()
        
    def _get_scale_factor(self) -> float:
        """Get the scale factor for converting ADC counts to volts."""
        # For Cyton at gain x24, scale factor is approximately 0.02235 µV/count
        return CYTON_SCALE_FACTOR
    
    def measure_impedance(self, channel: int, samples: int = 250) -> Dict:
        """
        Measure electrode impedance using OpenBCI's lead-off test.
        
        Args:
            channel: Channel number (1-8)
            samples: Number of samples to collect (default: 1 second at 250Hz)
            
        Returns:
            Dictionary with impedance measurement results
        """
        if channel < 1 or channel > 8:
            raise ValueError("Channel must be between 1 and 8")
        wait = samples / CYTON_SAMPLING_RATE

        try:
            # Stop any existing stream
            try:
                self.board.stop_stream()
            except:
                pass
            
            # Enable lead-off test: negative pin test on channel
            cmd = f"z {channel} 1 0 Z"
            logger.info(f"Sending impedance test command: {cmd}")
            self.board.config_board(cmd)
            
            # Wait for command to take effect
            time.sleep(0.1)

            # Start streaming and collect data
            self.board.start_stream()
            time.sleep(wait)  # Allow enough samples to be collected

            # Collect samples
            data = self.board.get_board_data(samples)
            self.board.stop_stream()
            
            # Get the tested channel data
            if channel - 1 < len(self.eeg_channels):
                row = self.eeg_channels[channel - 1]
                raw_data = data[row]
                
                # Convert ADC counts to volts
                volts = raw_data * self.scale_factor
                
                # Calculate standard deviation
                std_v = np.std(volts)
                
                # Calculate impedance using OpenBCI formula
                # impedance = (sqrt(2) * std_volts) / I_drive - R_series
                impedance_ohms = (math.sqrt(2) * std_v) / CYTON_DRIVE_CURRENT - CYTON_SERIES_RESISTOR
                impedance_kohm = impedance_ohms / 1000
                
                # Classify impedance
                quality = self._classify_impedance(impedance_kohm)
                
                result = {
                    "channel": channel,
                    "impedance_kohm": round(impedance_kohm, 2),
                    "impedance_ohms": round(impedance_ohms, 0),
                    "std_volts": round(std_v, 6),
                    "quality": quality,
                    "status": self._get_status_from_quality(quality),
                    "raw_data_samples": len(raw_data),
                    "test_frequency_hz": CYTON_TEST_FREQUENCY
                }
                
            else:
                raise ValueError(f"Invalid channel {channel}")
                
        except Exception as e:
            logger.error(f"Error measuring impedance for channel {channel}: {e}")
            result = {
                "channel": channel,
                "error": str(e),
                "status": "error"
            }
        
        finally:
            # Always disable lead-off test
            try:
                disable_cmd = f"z {channel} 0 0 Z"
                self.board.config_board(disable_cmd)
                logger.info(f"Disabled impedance test: {disable_cmd}")
            except Exception as e:
                logger.warning(f"Failed to disable impedance test: {e}")
        
        return result
    
    def _classify_impedance(self, impedance_kohm: float) -> str:
        """Classify impedance value into quality categories."""
        if impedance_kohm < IMPEDANCE_THRESHOLDS["good"]:
            return "good"
        elif impedance_kohm < IMPEDANCE_THRESHOLDS["moderate"]:
            return "moderate"
        elif impedance_kohm < IMPEDANCE_THRESHOLDS["poor"]:
            return "poor"
        else:
            return "disconnected"
    
    def _get_status_from_quality(self, quality: str) -> str:
        """Convert quality to status string."""
        status_map = {
            "good": "connected",
            "moderate": "poor_contact", 
            "poor": "poor_contact",
            "disconnected": "disconnected"
        }
        return status_map.get(quality, "unknown")
    
    def measure_all_channels(self, samples: int = 250) -> List[Dict]:
        """
        Measure impedance for all channels sequentially.
        
        Args:
            samples: Number of samples per channel
            
        Returns:
            List of impedance measurement results for each channel
        """
        results = []
        
        for channel in range(1, 9):  # Channels 1-8
            logger.info(f"Measuring impedance for channel {channel}")
            result = self.measure_impedance(channel, samples)
            results.append(result)
            
            # Small delay between measurements
            time.sleep(0.2)
        
        return results
    
    def detect_live_quality(self, data: np.ndarray) -> Dict:
        """
        Detect electrode connection quality from live signal data.
        
        Args:
            data: EEG data array (n_channels, n_samples)
            
        Returns:
            Dictionary with quality assessment for each channel
        """
        n_channels, n_samples = data.shape
        results = {}
        
        for ch in range(n_channels):
            signal = data[ch, :]
            
            # Calculate signal statistics
            signal_std = np.std(signal)
            signal_range = np.max(signal) - np.min(signal)
            signal_energy = np.sum(signal ** 2) / n_samples
            
            # Check for railing (signal saturation)
            max_amplitude = np.max(np.abs(signal))
            is_railed = max_amplitude > (RAILING_THRESHOLD * np.max(np.abs(data)))
            
            # Check for flat line (disconnected electrode)
            is_flat = signal_std < MIN_SIGNAL_STD
            
            # Check for excessive noise
            is_noisy = signal_std > MAX_SIGNAL_STD
            
            # Determine quality
            if is_flat:
                quality = "disconnected"
                status = "disconnected"
            elif is_railed:
                quality = "railed"
                status = "disconnected"
            elif is_noisy:
                quality = "noisy"
                status = "poor_contact"
            else:
                quality = "good"
                status = "connected"
            
            results[f"ch{ch+1}"] = {
                "status": status,
                "quality": quality,
                "std": round(float(signal_std), 3),
                "range": round(float(signal_range), 3),
                "energy": round(float(signal_energy), 6),
                "max_amplitude": round(float(max_amplitude), 3),
                "is_railed": is_railed,
                "is_flat": is_flat,
                "is_noisy": is_noisy
            }
        
        return results
    
    def get_board_status(self) -> Dict:
        """
        Get current board status information.
        
        Returns:
            Dictionary with board status details
        """
        try:
            # Check if board is prepared
            is_prepared = True
            try:
                self.board.get_board_data(1)
            except:
                is_prepared = False
            
            # Get board information
            board_id = self.board.get_board_id()
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            
            # Try to get battery level
            battery_level = None
            try:
                battery_channel = BoardShim.get_battery_channel(board_id)
                if battery_channel is not None:
                    data = self.board.get_board_data(1)
                    if data.shape[0] > battery_channel:
                        battery_level = float(data[battery_channel, -1])
            except:
                pass
            
            return {
                "board_id": board_id,
                "board_name": "OpenBCI Cyton",
                "sampling_rate_hz": sampling_rate,
                "eeg_channels": self.eeg_channels.tolist(),
                "is_prepared": is_prepared,
                "is_streaming": is_prepared,
                "battery_level": battery_level,
                "scale_factor_uv_per_count": self.scale_factor * 1e6,
                "impedance_thresholds_kohm": IMPEDANCE_THRESHOLDS
            }
            
        except Exception as e:
            logger.error(f"Error getting board status: {e}")
            return {
                "error": str(e),
                "is_prepared": False,
                "is_streaming": False
            }


def create_board_connection(serial_port: str) -> Optional[BoardShim]:
    """
    Create and prepare a board connection.
    
    Args:
        serial_port: Serial port for the Cyton board
        
    Returns:
        Prepared BoardShim instance or None if failed
    """
    try:
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        
        board = BoardShim(BoardIds.CYTON_BOARD.value, params)
        board.prepare_session()
        
        logger.info(f"Successfully connected to Cyton board on {serial_port}")
        return board
        
    except Exception as e:
        logger.error(f"Failed to connect to board on {serial_port}: {e}")
        return None


# Example usage and testing functions
def test_impedance_measurement(board: BoardShim, channel: int = 1):
    """Test impedance measurement on a single channel."""
    detector = ElectrodeDetector(board)
    result = detector.measure_impedance(channel)
    print(f"Channel {channel} impedance test:")
    print(f"  Impedance: {result.get('impedance_kohm', 'N/A')} kΩ")
    print(f"  Quality: {result.get('quality', 'N/A')}")
    print(f"  Status: {result.get('status', 'N/A')}")
    return result


def test_all_channels(board: BoardShim):
    """Test impedance measurement on all channels."""
    detector = ElectrodeDetector(board)
    results = detector.measure_all_channels()
    
    print("\nImpedance Test Results:")
    print("Channel | Impedance (kΩ) | Quality    | Status")
    print("-" * 50)
    
    for result in results:
        ch = result.get('channel', 'N/A')
        imp = result.get('impedance_kohm', 'N/A')
        quality = result.get('quality', 'N/A')
        status = result.get('status', 'N/A')
        print(f"   {ch:2d}   |      {imp:6.1f}     | {quality:8s} | {status}")


if __name__ == "__main__":
    # Example usage
    serial_port = "/dev/cu.usbserial-DM01N8KH"  # Update with your port
    
    board = create_board_connection(serial_port)
    if board:
        try:
            detector = ElectrodeDetector(board)
            
            # Test board status
            status = detector.get_board_status()
            print("Board Status:", status)
            
            # Test single channel
            test_impedance_measurement(board, 1)
            
            # Test all channels
            test_all_channels(board)
            
        finally:
            board.release_session()
