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
    
    # Set your port; examples:
    # params.serial_port = "/dev/tty.usbserial-DM03..."  # macOS
    # params.serial_port = "COM3"                        # Windows
    # params.serial_port = "/dev/ttyUSB0"                # Linux
    
    # Uncomment and set your actual port:
    # params.serial_port = "YOUR_CYTON_PORT_HERE"
    
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
        while True:
            time.sleep(WINDOW_SECONDS / 2)  # slide by 2 s
            data = board.get_board_data()   # (n_rows, n_samples)
            if data.shape[1] < n_times: 
                print(f"Not enough data yet: {data.shape[1]}/{n_times}")
                continue
            
            eeg = data[eeg_ids, -n_times:]  # pick last window
            print(f"Processing window: {eeg.shape}")
            
            payload = {"x": eeg.tolist(), "n_outputs": 2}
            try:
                r = requests.post(URL, json=payload, timeout=10)
                print(f"API Response ({r.status_code}): {r.text[:120]}")
            except Exception as e:
                print(f"API Error: {e}")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        board.stop_stream()
        board.release_session()

if __name__ == "__main__":
    main()
