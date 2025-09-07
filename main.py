import os
from pathlib import Path
from typing import Optional, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
try:
    from braindecode.models import Labram
    LABRAM_AVAILABLE = True
except ImportError:
    print("⚠️  LaBraM model not available in this version of braindecode")
    LABRAM_AVAILABLE = False
    Labram = None
from electrode_detection import ElectrodeDetector, create_board_connection
from brainflow_labram_integration import BrainFlowLaBraMPipeline

# ---- Cyton-friendly defaults ----
CYTON_SR = 250                    # Hz
WINDOW_SECONDS = 4
DEFAULT_NCH = 8
DEFAULT_NTIMES = CYTON_SR * WINDOW_SECONDS
DEFAULT_NOUT = 2

# ---- Optional weights via ENV ----
WEIGHTS_URL = os.getenv("WEIGHTS_URL")      # public URL to .pth
HF_REPO_ID = os.getenv("HF_REPO_ID")        # huggingface repo id
HF_FILENAME = os.getenv("HF_FILENAME")      # file name in repo
HF_TOKEN = os.getenv("HF_TOKEN")            # token if private/gated
WEIGHTS_DIR = Path(os.getenv("WEIGHTS_DIR", "./weights"))
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = WEIGHTS_DIR / "labram_checkpoint.pth"

app = FastAPI(title="EEG API (Cyton x LaBraM)")

class InferenceRequest(BaseModel):
    # 2D list: [n_chans][n_times]
    x: List[List[float]]
    n_outputs: Optional[int] = None   # falls back to DEFAULT_NOUT

class ElectrodeImpedanceRequest(BaseModel):
    channels: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]  # Default to all channels
    samples: int = 250  # Default to 1 second of data

class BoardConnectionRequest(BaseModel):
    serial_port: str

_model = None
_shape = None
_loaded_ckpt: Optional[Path] = None
_board = None
_electrode_detector = None
_pipeline = None

def maybe_download_checkpoint() -> Optional[Path]:
    global _loaded_ckpt
    if _loaded_ckpt is not None:
        return _loaded_ckpt

    if WEIGHTS_URL:
        import requests
        r = requests.get(WEIGHTS_URL, timeout=60)
        r.raise_for_status()
        CHECKPOINT_PATH.write_bytes(r.content)
        _loaded_ckpt = CHECKPOINT_PATH
        return _loaded_ckpt

    if HF_REPO_ID and HF_FILENAME:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN
        )
        _loaded_ckpt = Path(local_path)
        return _loaded_ckpt

    return None

def get_model(n_chans: int, n_times: int, n_outputs: int):
    global _model, _shape
    key = (n_chans, n_times, n_outputs)
    if _model is None or _shape != key:
        if not LABRAM_AVAILABLE:
            raise HTTPException(500, "LaBraM model not available. Please install braindecode>=1.1.0")
        
        m = Labram(
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            neural_tokenizer=True
        )
        m.eval()
        ckpt = maybe_download_checkpoint()
        if ckpt and ckpt.exists():
            state = torch.load(ckpt, map_location="cpu")
            state = state.get("state_dict", state)
            m.load_state_dict(state, strict=False)
        _model, _shape = m, key
    return _model

def detect_electrode_connections(arr: np.ndarray) -> dict:
    """
    Detect which electrodes are connected to scalp vs disconnected.
    Returns connection quality for each channel.
    """
    n_chans, n_times = arr.shape
    connections = {}
    
    for ch in range(n_chans):
        signal = arr[ch, :]
        
        # Calculate signal statistics
        signal_std = np.std(signal)
        signal_range = np.max(signal) - np.min(signal)
        signal_energy = np.sum(signal ** 2) / n_times
        
        # Check for flat line (disconnected electrode)
        is_flat = signal_std < 1.0  # Very low variation
        
        # Check for excessive noise (poor contact)
        is_noisy = signal_std > 100  # Very high variation (microvolts)
        
        # Check for reasonable signal range (good contact)
        is_good_range = 2 < signal_std < 80  # Typical EEG range
        
        # Determine connection status
        if is_flat:
            status = "disconnected"
            quality = 0.0
        elif is_noisy:
            status = "noisy"
            quality = 0.3
        elif is_good_range:
            status = "connected"
            quality = 1.0
        else:
            status = "poor_contact"
            quality = 0.6
            
        connections[f"ch{ch+1}"] = {
            "status": status,
            "quality": quality,
            "std": float(signal_std),
            "range": float(signal_range),
            "energy": float(signal_energy)
        }
    
    return connections

def preprocess(arr: np.ndarray) -> np.ndarray:
    """
    arr: (n_chans, n_times) in microvolts or volts.
    Minimal, fast, hardware-agnostic cleaning:
    - remove DC per channel
    - z-score per channel (robust to scales)
    """
    arr = arr - arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True) + 1e-8
    arr = arr / std
    return arr

@app.get("/health")
def health():
    return {
        "ok": True,
        "weights_loaded": bool(_loaded_ckpt),
        "defaults": {
            "n_chans": DEFAULT_NCH,
            "n_times": DEFAULT_NTIMES,
            "sample_rate": CYTON_SR
        }
    }

@app.post("/predict")
def predict(req: InferenceRequest):
    x = np.asarray(req.x, dtype=np.float32)        # (ch, time)
    if x.ndim != 2:
        raise HTTPException(400, "x must be 2D: [n_chans][n_times]")
    n_chans, n_times = x.shape
    if n_chans != DEFAULT_NCH:
        raise HTTPException(400, f"expected {DEFAULT_NCH} channels for Cyton")
    
    # Detect electrode connections BEFORE preprocessing
    electrode_status = detect_electrode_connections(x)
    
    x = preprocess(x)
    n_outputs = req.n_outputs or DEFAULT_NOUT
    model = get_model(n_chans, n_times, n_outputs)
    with torch.no_grad():
        t = torch.from_numpy(x).unsqueeze(0)       # (1, ch, time)
        logits = model(t)                          # (1, n_outputs)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()
    return {
        "probs": probs,
        "n_chans": n_chans,
        "n_times": n_times,
        "window_seconds": n_times / CYTON_SR,
        "electrode_status": electrode_status
    }

# Electrode Detection Endpoints

@app.post("/connect-board")
def connect_board(req: BoardConnectionRequest):
    """Connect to OpenBCI Cyton board."""
    global _board, _electrode_detector
    
    try:
        # Release existing connection if any
        if _board is not None:
            try:
                _board.release_session()
            except:
                pass
        
        # Create new connection
        _board = create_board_connection(req.serial_port)
        if _board is None:
            raise HTTPException(400, f"Failed to connect to board on {req.serial_port}")
        
        # Create electrode detector
        _electrode_detector = ElectrodeDetector(_board)
        
        return {
            "success": True,
            "serial_port": req.serial_port,
            "message": "Board connected successfully"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error connecting to board: {str(e)}")

@app.get("/board-status")
def get_board_status():
    """Get current board status and information."""
    global _board, _electrode_detector
    
    if _board is None or _electrode_detector is None:
        return {
            "connected": False,
            "message": "No board connection. Use /connect-board endpoint first."
        }
    
    try:
        status = _electrode_detector.get_board_status()
        status["connected"] = True
        return status
        
    except Exception as e:
        raise HTTPException(500, f"Error getting board status: {str(e)}")

@app.get("/battery")
def get_battery():
    """Get battery level from the board."""
    global _board, _electrode_detector
    
    if _board is None or _electrode_detector is None:
        raise HTTPException(400, "No board connection. Use /connect-board endpoint first.")
    
    try:
        status = _electrode_detector.get_board_status()
        battery_level = status.get("battery_level")
        
        if battery_level is None:
            return {
                "battery_level": None,
                "message": "Battery level not available"
            }
        
        return {
            "battery_level": battery_level,
            "unit": "volts"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error reading battery: {str(e)}")

@app.post("/electrode-impedance")
def measure_electrode_impedance(req: ElectrodeImpedanceRequest):
    """Measure electrode impedance for specified channels."""
    global _board, _electrode_detector
    
    if _board is None or _electrode_detector is None:
        raise HTTPException(400, "No board connection. Use /connect-board endpoint first.")
    
    try:
        results = []
        
        # Validate channels
        for channel in req.channels:
            if channel < 1 or channel > 8:
                raise HTTPException(400, f"Channel {channel} must be between 1 and 8")
        
        # Measure impedance for each channel
        for channel in req.channels:
            result = _electrode_detector.measure_impedance(channel, req.samples)
            results.append(result)
        
        return {
            "results": results,
            "channels_tested": req.channels,
            "samples_per_channel": req.samples,
            "test_duration_seconds": len(req.channels) * (req.samples / CYTON_SR + 0.2)  # Include overhead
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error measuring impedance: {str(e)}")

@app.post("/live-quality")
def get_live_quality(req: InferenceRequest):
    """Get real-time electrode connection quality from live signal data."""
    global _electrode_detector
    
    if _electrode_detector is None:
        raise HTTPException(400, "No board connection. Use /connect-board endpoint first.")
    
    try:
        x = np.asarray(req.x, dtype=np.float32)
        if x.ndim != 2:
            raise HTTPException(400, "x must be 2D: [n_chans][n_times]")
        
        # Use the electrode detector's live quality detection
        quality_results = _electrode_detector.detect_live_quality(x)
        
        return {
            "live_quality": quality_results,
            "n_chans": x.shape[0],
            "n_times": x.shape[1],
            "window_seconds": x.shape[1] / CYTON_SR
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error analyzing live quality: {str(e)}")

@app.post("/disconnect-board")
def disconnect_board():
    """Disconnect from the OpenBCI board."""
    global _board, _electrode_detector, _pipeline
    
    try:
        # Clean up pipeline if running
        if _pipeline is not None:
            _pipeline.cleanup()
            _pipeline = None
        
        # Clean up board connection
        if _board is not None:
            _board.release_session()
            _board = None
            _electrode_detector = None
        
        return {
            "success": True,
            "message": "Board disconnected successfully"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error disconnecting board: {str(e)}")

# Enhanced BrainFlow/LaBraM Integration Endpoints

@app.post("/start-pipeline")
def start_pipeline(req: BoardConnectionRequest):
    """Start the complete BrainFlow/LaBraM pipeline."""
    global _pipeline
    
    try:
        # Clean up existing pipeline
        if _pipeline is not None:
            _pipeline.cleanup()
        
        # Create new pipeline
        _pipeline = BrainFlowLaBraMPipeline(
            serial_port=req.serial_port,
            window_seconds=WINDOW_SECONDS,
            n_outputs=DEFAULT_NOUT,
            model_checkpoint=str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None
        )
        
        # Connect to board
        if not _pipeline.connect_board():
            raise HTTPException(400, f"Failed to connect to board on {req.serial_port}")
        
        # Load model
        if not _pipeline.load_model():
            raise HTTPException(500, "Failed to load LaBraM model")
        
        # Start streaming
        if not _pipeline.start_streaming():
            raise HTTPException(500, "Failed to start data streaming")
        
        return {
            "success": True,
            "message": "BrainFlow/LaBraM pipeline started successfully",
            "status": _pipeline.get_status()
        }
        
    except Exception as e:
        if _pipeline is not None:
            _pipeline.cleanup()
            _pipeline = None
        raise HTTPException(500, f"Error starting pipeline: {str(e)}")

@app.get("/pipeline-status")
def get_pipeline_status():
    """Get current pipeline status."""
    global _pipeline
    
    if _pipeline is None:
        return {
            "running": False,
            "message": "Pipeline not started. Use /start-pipeline endpoint first."
        }
    
    try:
        status = _pipeline.get_status()
        status["running"] = True
        return status
        
    except Exception as e:
        raise HTTPException(500, f"Error getting pipeline status: {str(e)}")

@app.get("/current-window")
def get_current_window():
    """Get the current sliding window data."""
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(400, "Pipeline not started. Use /start-pipeline endpoint first.")
    
    try:
        window = _pipeline.get_current_window()
        if window is None:
            raise HTTPException(400, "No data available")
        
        return {
            "window_data": window.tolist(),
            "shape": window.shape,
            "window_seconds": WINDOW_SECONDS,
            "n_channels": window.shape[0],
            "n_times": window.shape[1],
            "data_range": {
                "min": float(window.min()),
                "max": float(window.max()),
                "mean": float(window.mean()),
                "std": float(window.std())
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error getting current window: {str(e)}")

@app.post("/predict-realtime")
def predict_realtime():
    """Run LaBraM prediction on current window."""
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(400, "Pipeline not started. Use /start-pipeline endpoint first.")
    
    try:
        result = _pipeline.predict_window()
        
        if "error" in result:
            raise HTTPException(400, result["error"])
        
        return {
            "prediction": result,
            "timestamp": result.get("timestamp", 0),
            "pipeline_status": _pipeline.get_status()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error running prediction: {str(e)}")

@app.post("/predict-window")
def predict_window(req: InferenceRequest):
    """Run LaBraM prediction on provided window data."""
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(400, "Pipeline not started. Use /start-pipeline endpoint first.")
    
    try:
        x = np.asarray(req.x, dtype=np.float32)
        if x.ndim != 2:
            raise HTTPException(400, "x must be 2D: [n_chans][n_times]")
        
        n_chans, n_times = x.shape
        if n_chans != DEFAULT_NCH:
            raise HTTPException(400, f"expected {DEFAULT_NCH} channels for Cyton")
        
        # Convert to volts if needed (assuming input is in microvolts)
        if np.max(np.abs(x)) > 1.0:  # Likely microvolts
            x = x / 1e6  # Convert to volts
        
        result = _pipeline.predict_window(x)
        
        if "error" in result:
            raise HTTPException(400, result["error"])
        
        return {
            "prediction": result,
            "input_shape": x.shape,
            "window_seconds": n_times / CYTON_SR,
            "data_converted_to_volts": True
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error running prediction: {str(e)}")

@app.get("/board-info")
def get_board_info():
    """Get detailed board information."""
    global _pipeline
    
    if _pipeline is None:
        raise HTTPException(400, "Pipeline not started. Use /start-pipeline endpoint first.")
    
    try:
        board_info = _pipeline.get_board_info()
        return board_info
        
    except Exception as e:
        raise HTTPException(500, f"Error getting board info: {str(e)}")

@app.post("/stop-pipeline")
def stop_pipeline():
    """Stop the BrainFlow/LaBraM pipeline."""
    global _pipeline
    
    try:
        if _pipeline is not None:
            _pipeline.cleanup()
            _pipeline = None
        
        return {
            "success": True,
            "message": "Pipeline stopped successfully"
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error stopping pipeline: {str(e)}")

