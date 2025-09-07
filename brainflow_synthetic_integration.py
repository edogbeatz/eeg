"""
BrainFlow Synthetic Board Integration

Extends the existing BrainFlow pipeline to support:
1. Synthetic Board for testing without hardware
2. Playback File Board for replaying recorded data
3. Integration with synthetic data generator
"""

import time
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import logging

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  BrainFlow not installed. Install with: pip install brainflow")
    BRAINFLOW_AVAILABLE = False

from synthetic_data_generator import SyntheticEEGGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Board constants
CYTON_SAMPLING_RATE = 250
CYTON_N_CHANNELS = 8
DEFAULT_WINDOW_SECONDS = 4


class SyntheticBoardManager:
    """Manages BrainFlow Synthetic Board for testing and development"""
    
    def __init__(self, 
                 sampling_rate: int = CYTON_SAMPLING_RATE,
                 n_channels: int = CYTON_N_CHANNELS):
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.board: Optional[BoardShim] = None
        self.is_streaming = False
        
        # Synthetic data generator for custom patterns
        self.generator = SyntheticEEGGenerator(
            n_channels=n_channels,
            sampling_rate=sampling_rate
        )
    
    def connect(self) -> bool:
        """Connect to BrainFlow Synthetic Board"""
        if not BRAINFLOW_AVAILABLE:
            logger.error("BrainFlow not available")
            return False
        
        try:
            params = BrainFlowInputParams()
            self.board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
            self.board.prepare_session()
            logger.info("Connected to BrainFlow Synthetic Board")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Synthetic Board: {e}")
            return False
    
    def start_stream(self) -> bool:
        """Start data streaming"""
        if not self.board:
            logger.error("Board not connected")
            return False
        
        try:
            self.board.start_stream()
            self.is_streaming = True
            logger.info("Started synthetic data streaming")
            return True
        except Exception as e:
            logger.error(f"Failed to start stream: {e}")
            return False
    
    def get_window(self, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> Optional[np.ndarray]:
        """Get a window of synthetic EEG data"""
        if not self.board or not self.is_streaming:
            logger.error("Board not streaming")
            return None
        
        try:
            # Calculate required samples
            n_samples = int(window_seconds * self.sampling_rate)
            
            # Wait to accumulate enough data
            time.sleep(window_seconds + 0.1)  # Small buffer
            
            # Get data from board
            data = self.board.get_board_data()
            
            if data.shape[1] < n_samples:
                logger.warning(f"Insufficient data: got {data.shape[1]}, need {n_samples}")
                return None
            
            # Extract EEG channels
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
            eeg_data = data[eeg_channels, -n_samples:]
            
            # Take only the channels we need
            eeg_window = eeg_data[:self.n_channels].astype(np.float32)
            
            logger.info(f"Retrieved synthetic window: {eeg_window.shape}")
            return eeg_window
            
        except Exception as e:
            logger.error(f"Failed to get window: {e}")
            return None
    
    def stop_stream(self):
        """Stop data streaming"""
        if self.board and self.is_streaming:
            try:
                self.board.stop_stream()
                self.is_streaming = False
                logger.info("Stopped synthetic data streaming")
            except Exception as e:
                logger.error(f"Failed to stop stream: {e}")
    
    def disconnect(self):
        """Disconnect from board"""
        if self.board:
            try:
                if self.is_streaming:
                    self.stop_stream()
                self.board.release_session()
                self.board = None
                logger.info("Disconnected from Synthetic Board")
            except Exception as e:
                logger.error(f"Failed to disconnect: {e}")
    
    def generate_custom_window(self, 
                             state: str = "relaxed",
                             preprocess: bool = True) -> Tuple[np.ndarray, int]:
        """Generate custom synthetic window using our generator"""
        return self.generator.generate_window(state, preprocess)


class PlaybackBoardManager:
    """Manages BrainFlow Playback File Board for recorded data"""
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 sampling_rate: int = CYTON_SAMPLING_RATE,
                 n_channels: int = CYTON_N_CHANNELS):
        self.file_path = Path(file_path)
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.board: Optional[BoardShim] = None
        self.is_streaming = False
    
    def connect(self) -> bool:
        """Connect to BrainFlow Playback File Board"""
        if not BRAINFLOW_AVAILABLE:
            logger.error("BrainFlow not available")
            return False
        
        if not self.file_path.exists():
            logger.error(f"Playback file not found: {self.file_path}")
            return False
        
        try:
            params = BrainFlowInputParams()
            params.file = str(self.file_path)
            params.master_board = BoardIds.CYTON_BOARD.value  # Simulate Cyton board
            
            self.board = BoardShim(BoardIds.PLAYBACK_FILE_BOARD.value, params)
            self.board.prepare_session()
            logger.info(f"Connected to Playback Board with file: {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Playback Board: {e}")
            return False
    
    def start_stream(self) -> bool:
        """Start playback streaming"""
        if not self.board:
            logger.error("Board not connected")
            return False
        
        try:
            self.board.start_stream()
            self.is_streaming = True
            logger.info("Started playback streaming")
            return True
        except Exception as e:
            logger.error(f"Failed to start playback: {e}")
            return False
    
    def get_window(self, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> Optional[np.ndarray]:
        """Get a window of playback EEG data"""
        if not self.board or not self.is_streaming:
            logger.error("Board not streaming")
            return None
        
        try:
            # Calculate required samples
            n_samples = int(window_seconds * self.sampling_rate)
            
            # Wait for data
            time.sleep(0.1)
            
            # Get available data
            data = self.board.get_current_board_data(n_samples)
            
            if data.shape[1] < n_samples:
                logger.warning(f"Insufficient playback data: got {data.shape[1]}, need {n_samples}")
                # Try to get whatever is available
                data = self.board.get_board_data()
                if data.shape[1] == 0:
                    logger.error("No playback data available")
                    return None
            
            # Extract EEG channels
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)  # Use Cyton channels
            eeg_data = data[eeg_channels, -min(n_samples, data.shape[1]):]
            
            # Take only the channels we need
            eeg_window = eeg_data[:self.n_channels].astype(np.float32)
            
            logger.info(f"Retrieved playback window: {eeg_window.shape}")
            return eeg_window
            
        except Exception as e:
            logger.error(f"Failed to get playback window: {e}")
            return None
    
    def stop_stream(self):
        """Stop playback streaming"""
        if self.board and self.is_streaming:
            try:
                self.board.stop_stream()
                self.is_streaming = False
                logger.info("Stopped playback streaming")
            except Exception as e:
                logger.error(f"Failed to stop playback: {e}")
    
    def disconnect(self):
        """Disconnect from playback board"""
        if self.board:
            try:
                if self.is_streaming:
                    self.stop_stream()
                self.board.release_session()
                self.board = None
                logger.info("Disconnected from Playback Board")
            except Exception as e:
                logger.error(f"Failed to disconnect from playback: {e}")


class UnifiedBoardManager:
    """Unified interface for all board types (Real, Synthetic, Playback)"""
    
    def __init__(self):
        self.current_board = None
        self.board_type = None
    
    def connect_synthetic(self, 
                         sampling_rate: int = CYTON_SAMPLING_RATE,
                         n_channels: int = CYTON_N_CHANNELS) -> bool:
        """Connect to synthetic board"""
        self.disconnect()  # Disconnect any existing board
        
        self.current_board = SyntheticBoardManager(sampling_rate, n_channels)
        if self.current_board.connect():
            self.board_type = "synthetic"
            logger.info("Connected to Synthetic Board")
            return True
        else:
            self.current_board = None
            return False
    
    def connect_playback(self, 
                        file_path: Union[str, Path],
                        sampling_rate: int = CYTON_SAMPLING_RATE,
                        n_channels: int = CYTON_N_CHANNELS) -> bool:
        """Connect to playback board"""
        self.disconnect()  # Disconnect any existing board
        
        self.current_board = PlaybackBoardManager(file_path, sampling_rate, n_channels)
        if self.current_board.connect():
            self.board_type = "playback"
            logger.info(f"Connected to Playback Board: {file_path}")
            return True
        else:
            self.current_board = None
            return False
    
    def connect_real(self, 
                    serial_port: str,
                    sampling_rate: int = CYTON_SAMPLING_RATE,
                    n_channels: int = CYTON_N_CHANNELS) -> bool:
        """Connect to real Cyton board (placeholder - use existing implementation)"""
        # This would use the existing BrainFlowLaBraMPipeline
        logger.info("Real board connection should use BrainFlowLaBraMPipeline")
        return False
    
    def start_stream(self) -> bool:
        """Start streaming from current board"""
        if not self.current_board:
            logger.error("No board connected")
            return False
        
        return self.current_board.start_stream()
    
    def get_window(self, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> Optional[np.ndarray]:
        """Get window from current board"""
        if not self.current_board:
            logger.error("No board connected")
            return None
        
        return self.current_board.get_window(window_seconds)
    
    def get_board_info(self) -> Dict:
        """Get information about current board"""
        if not self.current_board:
            return {"connected": False, "type": None}
        
        return {
            "connected": True,
            "type": self.board_type,
            "streaming": getattr(self.current_board, 'is_streaming', False)
        }
    
    def stop_stream(self):
        """Stop streaming from current board"""
        if self.current_board:
            self.current_board.stop_stream()
    
    def disconnect(self):
        """Disconnect from current board"""
        if self.current_board:
            self.current_board.disconnect()
            self.current_board = None
            self.board_type = None
    
    def generate_custom_synthetic(self, state: str = "relaxed") -> Optional[Tuple[np.ndarray, int]]:
        """Generate custom synthetic data (only for synthetic board)"""
        if self.board_type == "synthetic" and hasattr(self.current_board, 'generate_custom_window'):
            return self.current_board.generate_custom_window(state)
        else:
            logger.error("Custom generation only available for synthetic board")
            return None


def demo_synthetic_integration():
    """Demonstrate synthetic board integration"""
    print("=== BrainFlow Synthetic Integration Demo ===\n")
    
    # Test synthetic board
    print("1. Testing Synthetic Board...")
    manager = UnifiedBoardManager()
    
    if manager.connect_synthetic():
        print("✅ Connected to synthetic board")
        
        if manager.start_stream():
            print("✅ Started streaming")
            
            # Get a few windows
            for i in range(3):
                print(f"\nGetting window {i+1}...")
                window = manager.get_window(2.0)  # 2-second window
                if window is not None:
                    print(f"✅ Window shape: {window.shape}")
                    print(f"   Data range: [{window.min():.3f}, {window.max():.3f}]")
                else:
                    print("❌ Failed to get window")
            
            # Test custom generation
            print("\nTesting custom synthetic generation...")
            for state in ["relaxed", "anxious"]:
                data, label = manager.generate_custom_synthetic(state)
                if data is not None:
                    print(f"✅ {state.capitalize()} window: {data.shape}, label: {label}")
                else:
                    print(f"❌ Failed to generate {state} window")
        
        manager.disconnect()
        print("✅ Disconnected from synthetic board")
    else:
        print("❌ Failed to connect to synthetic board")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    demo_synthetic_integration()
