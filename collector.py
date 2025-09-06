# collector.py (run locally)
import time
import requests
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Your deployed API URL
URL = "https://eeg-3j9h.onrender.com/predict"
WINDOW_SECONDS = 4

def main():
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    
    # Set your Cyton port (detected automatically)
    params.serial_port = "/dev/cu.usbserial-DM01N8KH"  # Your Cyton board
    
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    sr = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
    eeg_ids = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)

    board.prepare_session()
    board.start_stream()
    print(f"Streaming at {sr} Hz...")
    print(f"EEG channels: {eeg_ids}")
    print(f"Window size: {WINDOW_SECONDS} seconds ({int(sr * WINDOW_SECONDS)} samples)")

    try:
        n_times = int(sr * WINDOW_SECONDS)
        print(f"Waiting for {n_times} samples ({WINDOW_SECONDS} seconds)...")
        
        while True:
            time.sleep(1)  # Check every second
            data = board.get_board_data()   # (n_rows, n_samples)
            current_samples = data.shape[1]
            
            if current_samples < n_times: 
                print(f"Collecting: {current_samples}/{n_times} samples ({current_samples/sr:.1f}s)")
                continue
            
            # We have enough data!
            eeg = data[eeg_ids, -n_times:]  # pick last window
            print(f"✅ Processing window: {eeg.shape}")
            print(f"   Data range: {eeg.min():.3f} to {eeg.max():.3f}")
            
            payload = {"x": eeg.tolist(), "n_outputs": 2}
            try:
                print("🧠 Sending to API...")
                r = requests.post(URL, json=payload, timeout=15)
                print(f"✅ API Response ({r.status_code}): {r.text[:200]}")
                
                # Reset buffer for next window
                board.get_board_data()  # Clear buffer
                print("🔄 Buffer cleared, collecting next window...")
                
            except Exception as e:
                print(f"❌ API Error: {e}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()
