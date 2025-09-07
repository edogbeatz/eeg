#!/usr/bin/env python3
"""
Enhanced Collector Script for BrainFlow/LaBraM Integration

This script demonstrates the complete pipeline:
1. Connect to OpenBCI Cyton board via BrainFlow
2. Stream data continuously with ring buffer
3. Convert microvolts to volts (MNE compatibility)
4. Preprocess data (DC removal, z-scoring)
5. Run LaBraM inference on sliding windows
6. Display real-time predictions and electrode status

Usage:
    python enhanced_collector.py [serial_port] [api_url]
    
Example:
    python enhanced_collector.py /dev/cu.usbserial-DM01N8KH http://localhost:8000
"""

import sys
import time
import requests
import numpy as np
import json

try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter
    BRAINFLOW_AVAILABLE = True
except ImportError:
    print("⚠️  BrainFlow not installed. Install with: pip install brainflow")
    BRAINFLOW_AVAILABLE = False


class EnhancedCollector:
    """
    Enhanced collector that demonstrates the complete BrainFlow/LaBraM pipeline.
    """
    
    def __init__(self, serial_port: str, api_url: str = "http://localhost:8000"):
        self.serial_port = serial_port
        self.api_url = api_url.rstrip('/')
        self.board = None
        self.sampling_rate = None
        self.eeg_channels = None
        self.channel_names = None
        
        # Pipeline parameters
        self.window_seconds = 4
        self.n_times = None
        self.buffer = None
        
        # Statistics
        self.total_samples = 0
        self.predictions_made = 0
        self.start_time = None
        
    def connect_board(self) -> bool:
        """Connect to the OpenBCI Cyton board."""
        if not BRAINFLOW_AVAILABLE:
            print("❌ BrainFlow not available. Cannot connect to board.")
            return False
        
        try:
            BoardShim.enable_dev_board_logger()
            params = BrainFlowInputParams()
            params.serial_port = self.serial_port
            
            self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)
            self.board.prepare_session()
            
            # Get board information
            self.sampling_rate = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
            self.eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            self.channel_names = BoardShim.get_eeg_names(BoardIds.CYTON_BOARD.value)
            self.n_times = int(self.sampling_rate * self.window_seconds)
            
            # Initialize ring buffer
            self.buffer = np.zeros((len(self.eeg_channels), self.n_times), dtype=np.float32)
            
            print(f"✅ Connected to Cyton board on {self.serial_port}")
            print(f"   Sampling rate: {self.sampling_rate} Hz")
            print(f"   EEG channels: {self.eeg_channels}")
            print(f"   Channel names: {self.channel_names}")
            print(f"   Window size: {self.window_seconds}s ({self.n_times} samples)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to board: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start data streaming."""
        try:
            self.board.start_stream()
            print(f"✅ Started streaming at {self.sampling_rate} Hz")
            return True
        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            return False
    
    def update_buffer(self, new_data: np.ndarray):
        """Update ring buffer with new data."""
        n_samples = new_data.shape[1]
        self.buffer = np.roll(self.buffer, -n_samples, axis=1)
        self.buffer[:, -n_samples:] = new_data
        self.total_samples += n_samples
    
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """Preprocess window for LaBraM (DC removal + z-scoring)."""
        # Remove DC offset per channel
        window = window - window.mean(axis=1, keepdims=True)
        
        # Z-score normalization per channel
        std = window.std(axis=1, keepdims=True) + 1e-8
        window = window / std
        
        return window
    
    def send_to_api(self, window: np.ndarray) -> dict:
        """Send window data to API for prediction."""
        try:
            # Convert microvolts to volts for MNE compatibility
            window_volts = window / 1e6
            
            payload = {
                "x": window_volts.tolist(),
                "n_outputs": 2
            }
            
            response = requests.post(f"{self.api_url}/predict", json=payload, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def start_pipeline_via_api(self) -> bool:
        """Start the complete pipeline via API."""
        try:
            payload = {"serial_port": self.serial_port}
            response = requests.post(f"{self.api_url}/start-pipeline", json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            print(f"✅ Pipeline started via API: {result['message']}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start pipeline via API: {e}")
            return False
    
    def get_realtime_prediction(self) -> dict:
        """Get real-time prediction from API."""
        try:
            response = requests.post(f"{self.api_url}/predict-realtime", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_pipeline_status(self) -> dict:
        """Get pipeline status from API."""
        try:
            response = requests.get(f"{self.api_url}/pipeline-status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def print_prediction_result(self, result: dict, prediction_num: int):
        """Print formatted prediction result."""
        if "error" in result:
            print(f"❌ Prediction {prediction_num}: {result['error']}")
            return
        
        probs = result.get("probs", [])
        electrode_status = result.get("electrode_status", {})
        
        print(f"\n🧠 Prediction {prediction_num}:")
        print(f"   Probabilities: {[f'{p:.3f}' for p in probs]}")
        print(f"   Predicted class: {probs.index(max(probs)) if probs else 'N/A'}")
        print(f"   Confidence: {max(probs):.3f}" if probs else "N/A")
        
        # Show electrode status
        print("   Electrode Status:")
        for ch_name, status in electrode_status.items():
            quality = status.get("quality", 0)
            status_text = status.get("status", "unknown")
            
            if quality >= 0.8:
                indicator = "✅"
            elif quality >= 0.5:
                indicator = "⚠️"
            else:
                indicator = "❌"
            
            print(f"     {ch_name}: {indicator} {status_text} (quality: {quality:.2f})")
    
    def run_local_collection(self):
        """Run local data collection and processing."""
        print("\n🔄 Starting local data collection...")
        
        try:
            while True:
                time.sleep(1)  # Check every second
                
                # Get data from board
                data = self.board.get_board_data()
                current_samples = data.shape[1]
                
                if current_samples < self.n_times:
                    print(f"📊 Collecting: {current_samples}/{self.n_times} samples ({current_samples/self.sampling_rate:.1f}s)")
                    continue
                
                # Extract EEG data and update buffer
                eeg_data = data[self.eeg_channels, -self.n_times:]
                self.update_buffer(eeg_data)
                
                print(f"✅ Processing window: {eeg_data.shape}")
                print(f"   Data range: {eeg_data.min():.3f} to {eeg_data.max():.3f} µV")
                
                # Send to API for prediction
                result = self.send_to_api(eeg_data)
                self.predictions_made += 1
                
                self.print_prediction_result(result, self.predictions_made)
                
                # Clear buffer for next window
                self.board.get_board_data()
                print("🔄 Buffer cleared, collecting next window...")
                
        except KeyboardInterrupt:
            print("\n⏹️  Collection stopped by user")
    
    def run_api_pipeline(self):
        """Run using the API pipeline."""
        print("\n🔄 Starting API-based pipeline...")
        
        # Start pipeline
        if not self.start_pipeline_via_api():
            return
        
        try:
            # Wait for pipeline to initialize
            time.sleep(3)
            
            while True:
                # Get pipeline status
                status = self.get_pipeline_status()
                if "error" in status:
                    print(f"❌ Pipeline error: {status['error']}")
                    break
                
                # Get real-time prediction
                result = self.get_realtime_prediction()
                self.predictions_made += 1
                
                if "error" in result:
                    print(f"❌ Prediction error: {result['error']}")
                    time.sleep(2)
                    continue
                
                prediction_data = result.get("prediction", {})
                self.print_prediction_result(prediction_data, self.predictions_made)
                
                time.sleep(2)  # Wait between predictions
                
        except KeyboardInterrupt:
            print("\n⏹️  Pipeline stopped by user")
    
    def cleanup(self):
        """Clean up resources."""
        if self.board is not None:
            try:
                self.board.stop_stream()
                self.board.release_session()
                print("✅ Board connection closed")
            except:
                pass
    
    def print_statistics(self):
        """Print collection statistics."""
        if self.start_time:
            duration = time.time() - self.start_time
            print(f"\n📊 Collection Statistics:")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Total samples: {self.total_samples}")
            print(f"   Predictions made: {self.predictions_made}")
            print(f"   Average samples/sec: {self.total_samples/duration:.1f}")
            print(f"   Average predictions/sec: {self.predictions_made/duration:.2f}")


def main():
    """Main function."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        serial_port = "/dev/cu.usbserial-DM01N8KH"  # Default
    else:
        serial_port = sys.argv[1]
    
    if len(sys.argv) < 3:
        api_url = "http://localhost:8000"  # Default
    else:
        api_url = sys.argv[2]
    
    print("🚀 Enhanced BrainFlow/LaBraM Collector")
    print(f"   Serial port: {serial_port}")
    print(f"   API URL: {api_url}")
    print("   Make sure your Cyton board is connected and the API server is running")
    
    collector = EnhancedCollector(serial_port, api_url)
    collector.start_time = time.time()
    
    try:
        # Connect to board
        if not collector.connect_board():
            return
        
        # Start streaming
        if not collector.start_streaming():
            return
        
        # Choose collection mode
        print("\nChoose collection mode:")
        print("1. Local collection (collector processes data locally)")
        print("2. API pipeline (use API's background pipeline)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            collector.run_local_collection()
        elif choice == "2":
            collector.run_api_pipeline()
        else:
            print("Invalid choice, using local collection")
            collector.run_local_collection()
        
    except KeyboardInterrupt:
        print("\n⏹️  Collection interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
    finally:
        collector.cleanup()
        collector.print_statistics()


if __name__ == "__main__":
    main()
