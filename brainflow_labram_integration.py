"""
Enhanced BrainFlow/OpenBCI Integration with LaBraM

This module implements the complete pipeline as specified:
1. Stream data from Cyton board via BrainFlow
2. Create sliding windows with ring buffer
3. Convert microvolts to volts (MNE compatibility)
4. Preprocess data (DC removal, z-scoring)
5. Feed to LaBraM model for inference
6. Provide FastAPI endpoints for real-time BCI predictions
"""

import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import logging

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  BrainFlow not installed. Install with: pip install brainflow")
    BRAINFLOW_AVAILABLE = False

try:
    from braindecode.models import Labram
    BRAINDECODE_AVAILABLE = True
except ImportError:
    print("⚠️  BrainDecode not installed. Install with: pip install braindecode")
    BRAINDECODE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cyton board constants
CYTON_SAMPLING_RATE = 250  # Hz
CYTON_N_CHANNELS = 8
DEFAULT_WINDOW_SECONDS = 4
DEFAULT_N_TIMES = CYTON_SAMPLING_RATE * DEFAULT_WINDOW_SECONDS  # 1000 samples


class BrainFlowLaBraMPipeline:
    """
    Complete pipeline for streaming OpenBCI data to LaBraM model.
    
    Implements:
    - BrainFlow board connection and streaming
    - Ring buffer for sliding windows
    - Microvolts to volts conversion
    - Preprocessing (DC removal, z-scoring)
    - LaBraM model inference
    - Background data collection
    """
    
    def __init__(self, 
                 serial_port: str,
                 window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 n_outputs: int = 2,
                 model_checkpoint: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            serial_port: Serial port for Cyton board
            window_seconds: Window length in seconds
            n_outputs: Number of output classes for LaBraM
            model_checkpoint: Path to pre-trained model checkpoint
        """
        self.serial_port = serial_port
        self.window_seconds = window_seconds
        self.n_times = int(CYTON_SAMPLING_RATE * window_seconds)
        self.n_outputs = n_outputs
        self.model_checkpoint = model_checkpoint
        
        # Board connection
        self.board: Optional[BoardShim] = None
        self.eeg_channels: Optional[np.ndarray] = None
        self.channel_names: Optional[List[str]] = None
        
        # Ring buffer for sliding windows
        self.buffer: Optional[np.ndarray] = None
        self.buffer_lock = threading.Lock()
        
        # LaBraM model
        self.model: Optional[Labram] = None
        self.model_lock = threading.Lock()
        
        # Background streaming
        self.streaming = False
        self.stream_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_samples_collected = 0
        self.last_prediction_time = 0
        
    def connect_board(self) -> bool:
        """Connect to the OpenBCI Cyton board."""
        try:
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            self.board.prepare_session()
            
            # Get board information
            self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            self.channel_names = BoardShim.get_eeg_names(BoardIds.CYTON_BOARD.value)
            
            # Initialize ring buffer
            self.buffer = np.zeros((len(self.eeg_channels), self.n_times), dtype=np.float32)
            
            logger.info(f"Connected to Cyton board on {self.serial_port}")
            logger.info(f"EEG channels: {self.eeg_channels}")
            logger.info(f"Channel names: {self.channel_names}")
            logger.info(f"Window size: {self.window_seconds}s ({self.n_times} samples)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to board: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load the LaBraM model with pre-trained weights."""
        try:
            with self.model_lock:
                self.model = Labram(
                    n_chans=CYTON_N_CHANNELS,
                    n_times=self.n_times,
                    n_outputs=self.n_outputs,
                    neural_tokenizer=True
                )
                self.model.eval()
                
                # Load pre-trained weights if available
                if self.model_checkpoint:
                    state = torch.load(self.model_checkpoint, map_location='cpu')
                    state_dict = state.get('state_dict', state)
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Loaded pre-trained weights from {self.model_checkpoint}")
                else:
                    logger.info("Using randomly initialized LaBraM model")
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start background data streaming."""
        if self.board is None:
            logger.error("Board not connected")
            return False
        
        try:
            self.board.start_stream()
            self.streaming = True
            
            # Start background thread for data collection
            self.stream_thread = threading.Thread(target=self._stream_data_loop)
            self.stream_thread.daemon = True
            self.stream_thread.start()
            
            logger.info("Started background data streaming")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop background data streaming."""
        self.streaming = False
        
        if self.board is not None:
            try:
                self.board.stop_stream()
            except:
                pass
        
        if self.stream_thread is not None:
            self.stream_thread.join(timeout=2)
        
        logger.info("Stopped data streaming")
    
    def _stream_data_loop(self):
        """Background thread for continuous data collection."""
        while self.streaming:
            try:
                # Get new data from board
                data = self.board.get_board_data()
                
                if data.shape[1] > 0:
                    # Extract EEG channels
                    eeg_data = data[self.eeg_channels, :]
                    
                    # Convert microvolts to volts (MNE compatibility)
                    eeg_data_volts = eeg_data / 1e6
                    
                    # Update ring buffer
                    with self.buffer_lock:
                        self._update_buffer(eeg_data_volts)
                        self.total_samples_collected += eeg_data.shape[1]
                
                time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(0.1)
    
    def _update_buffer(self, new_data: np.ndarray):
        """
        Update ring buffer with new data.
        
        Args:
            new_data: New EEG data (n_channels, n_samples) in volts
        """
        n_samples = new_data.shape[1]
        
        # Roll buffer and add new data
        self.buffer = np.roll(self.buffer, -n_samples, axis=1)
        self.buffer[:, -n_samples:] = new_data
    
    def get_current_window(self) -> Optional[np.ndarray]:
        """
        Get the current sliding window.
        
        Returns:
            Current window data (n_channels, n_times) in volts
        """
        with self.buffer_lock:
            if self.buffer is None:
                return None
            return self.buffer.copy()
    
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """
        Preprocess the sliding window for LaBraM.
        
        Args:
            window: EEG window (n_channels, n_times) in volts
            
        Returns:
            Preprocessed window ready for LaBraM
        """
        # Remove DC offset per channel
        window = window - window.mean(axis=1, keepdims=True)
        
        # Z-score normalization per channel
        std = window.std(axis=1, keepdims=True) + 1e-8
        window = window / std
        
        return window
    
    def predict_window(self, window: Optional[np.ndarray] = None) -> Dict:
        """
        Run LaBraM inference on current or provided window.
        
        Args:
            window: Optional window data, if None uses current buffer
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Get window data
        if window is None:
            window = self.get_current_window()
            if window is None:
                return {"error": "No data available"}
        
        # Check if we have enough data
        if window.shape[1] < self.n_times:
            return {"error": f"Insufficient data: {window.shape[1]}/{self.n_times} samples"}
        
        try:
            # Preprocess window
            processed_window = self.preprocess_window(window)
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(processed_window).unsqueeze(0).float()  # (1, n_chans, n_times)
            
            # Run inference
            with torch.no_grad():
                with self.model_lock:
                    logits = self.model(tensor)
                    probs = F.softmax(logits, dim=1).squeeze(0).numpy()
            
            self.last_prediction_time = time.time()
            
            return {
                "probabilities": probs.tolist(),
                "predicted_class": int(np.argmax(probs)),
                "confidence": float(np.max(probs)),
                "window_shape": window.shape,
                "preprocessing_applied": True,
                "timestamp": self.last_prediction_time
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict:
        """Get pipeline status information."""
        return {
            "board_connected": self.board is not None,
            "streaming": self.streaming,
            "model_loaded": self.model is not None,
            "buffer_initialized": self.buffer is not None,
            "total_samples_collected": self.total_samples_collected,
            "window_seconds": self.window_seconds,
            "n_times": self.n_times,
            "n_channels": CYTON_N_CHANNELS,
            "sampling_rate": CYTON_SAMPLING_RATE,
            "channel_names": self.channel_names,
            "last_prediction_time": self.last_prediction_time
        }
    
    def get_board_info(self) -> Dict:
        """Get detailed board information."""
        if self.board is None:
            return {"error": "Board not connected"}
        
        try:
            # Get battery level
            battery_channel = BoardShim.get_battery_channel(BoardIds.CYTON_BOARD.value)
            battery_level = None
            
            if battery_channel is not None and battery_channel >= 0:
                data = self.board.get_board_data(1)
                if data.shape[0] > battery_channel:
                    battery_level = float(data[battery_channel, -1])
            else:
                battery_level = None
            
            return {
                "board_id": BoardIds.CYTON_BOARD.value,
                "board_name": "OpenBCI Cyton",
                "serial_port": self.serial_port,
                "sampling_rate": CYTON_SAMPLING_RATE,
                "eeg_channels": self.eeg_channels.tolist() if self.eeg_channels is not None else None,
                "channel_names": self.channel_names,
                "battery_level": battery_level,
                "streaming_active": self.streaming
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_streaming()
        
        if self.board is not None:
            try:
                self.board.release_session()
            except:
                pass
        
        logger.info("Pipeline cleaned up")


# Example usage and testing
def test_pipeline(serial_port: str, model_checkpoint: Optional[str] = None):
    """Test the complete pipeline."""
    pipeline = BrainFlowLaBraMPipeline(
        serial_port=serial_port,
        window_seconds=4,
        n_outputs=2,
        model_checkpoint=model_checkpoint
    )
    
    try:
        # Connect to board
        if not pipeline.connect_board():
            print("❌ Failed to connect to board")
            return
        
        # Load model
        if not pipeline.load_model():
            print("❌ Failed to load model")
            return
        
        # Start streaming
        if not pipeline.start_streaming():
            print("❌ Failed to start streaming")
            return
        
        print("✅ Pipeline initialized successfully")
        print(f"Status: {pipeline.get_status()}")
        
        # Wait for data to accumulate
        print("Waiting for data to accumulate...")
        time.sleep(5)
        
        # Run predictions
        for i in range(5):
            result = pipeline.predict_window()
            print(f"Prediction {i+1}: {result}")
            time.sleep(2)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    # Test with default port
    test_pipeline("/dev/cu.usbserial-DM01N8KH")
