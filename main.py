import os
import time
import threading
import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
from brainflow_synthetic_integration import UnifiedBoardManager, SyntheticBoardManager
from synthetic_data_generator import SyntheticEEGGenerator
from train_synthetic import SyntheticEEGTrainer
from real_data_trainer import RealDataTrainer

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

app = FastAPI(title="EEG Meditation Detection API (Cyton x LaBraM)")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    # 2D list: [n_chans][n_times]
    x: List[List[float]]
    n_outputs: Optional[int] = None   # falls back to DEFAULT_NOUT

class ElectrodeImpedanceRequest(BaseModel):
    channels: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]  # Default to all channels
    samples: int = 250  # Default to 1 second of data

class BoardConnectionRequest(BaseModel):
    serial_port: str

class TrainingConfig(BaseModel):
    n_epochs: int = 15
    n_samples_per_class: int = 1500
    val_split: float = 0.2
    freeze_backbone: bool = True
    learning_rate: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-4
    save_dir: str = "./trained_models"

class TrainingControlRequest(BaseModel):
    action: str  # "start" or "stop"
    config: Optional[TrainingConfig] = None

class RealDataTrainingConfig(BaseModel):
    dataset_id: str = "ds003969"
    n_epochs: int = 20
    val_split: float = 0.2
    save_dir: str = "./real_trained_models"
    force_redownload: bool = False

_model = None
_shape = None
_loaded_ckpt: Optional[Path] = None
_board = None
_electrode_detector = None
_pipeline = None
synthetic_manager = None
synthetic_generator = None

# Training state management
_training_status = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "train_acc": 0.0,
    "val_acc": 0.0,
    "best_val_acc": 0.0,
    "start_time": None,
    "elapsed_time": 0.0,
    "eta": None,
    "config": None,
    "history": {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": []
    }
}
_training_thread = None
_stop_training_flag = False
_trainer = None

def maybe_download_checkpoint() -> Optional[Path]:
    global _loaded_ckpt
    if _loaded_ckpt is not None:
        return _loaded_ckpt

    # Check for local checkpoint first
    if CHECKPOINT_PATH.exists():
        _loaded_ckpt = CHECKPOINT_PATH
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
            # Handle our specific training format
            if "model_state_dict" in state:
                model_state = state["model_state_dict"]
            else:
                model_state = state.get("state_dict", state)
            m.load_state_dict(model_state, strict=False)
            print(f"✅ Loaded trained meditation model from {ckpt}")
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

# Synthetic Board Endpoints for Frontend Testing
_synthetic_board = None
_synthetic_board_lock = False

@app.get("/synthetic-data")
def get_synthetic_data():
    """Get synthetic EEG data for frontend testing (8 channels × 1000 samples)."""
    global _synthetic_board, _synthetic_board_lock
    
    try:
        from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
        import time
        
        # Avoid concurrent board access
        if _synthetic_board_lock:
            # Generate simple synthetic data without BrainFlow
            import numpy as np
            np.random.seed(int(time.time()) % 1000)
            eeg8 = np.random.randn(8, 1000) * 50 + np.sin(np.arange(1000) * 0.02)[:, np.newaxis].T * 20
            electrode_status = detect_electrode_connections(eeg8)
            
            return {
                "data": eeg8.tolist(),
                "shape": eeg8.shape,
                "channels": 8,
                "samples": 1000,
                "sampling_rate": 250,
                "window_seconds": 4.0,
                "electrode_status": electrode_status,
                "data_range": {
                    "min": float(eeg8.min()),
                    "max": float(eeg8.max()),
                    "mean": float(eeg8.mean()),
                    "std": float(eeg8.std())
                }
            }
        
        _synthetic_board_lock = True
        
        try:
            # Create synthetic board
            params = BrainFlowInputParams()
            board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
            board.prepare_session()
            board.start_stream()
            
            # Collect data for ~4 seconds
            time.sleep(4.2)
            data = board.get_board_data()
            
            # Clean up
            board.stop_stream()
            board.release_session()
            
            # Get EEG channels
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
            eeg_data = data[eeg_channels, :]
            
            # Take first 8 channels and last 1000 samples
            eeg8 = eeg_data[:8, -1000:]
            
            # Detect electrode connections
            electrode_status = detect_electrode_connections(eeg8)
            
            return {
                "data": eeg8.tolist(),
                "shape": eeg8.shape,
                "channels": 8,
                "samples": 1000,
                "sampling_rate": 250,
                "window_seconds": 4.0,
                "electrode_status": electrode_status,
                "data_range": {
                    "min": float(eeg8.min()),
                    "max": float(eeg8.max()),
                    "mean": float(eeg8.mean()),
                    "std": float(eeg8.std())
                }
            }
        finally:
            _synthetic_board_lock = False
        
    except Exception as e:
        _synthetic_board_lock = False
        raise HTTPException(500, f"Error generating synthetic data: {str(e)}")

@app.get("/synthetic-stream")
def get_synthetic_stream():
    """Get a continuous stream of synthetic EEG data."""
    global _synthetic_board_lock
    
    try:
        import time
        import numpy as np
        
        # Use simple numpy generation for streaming to avoid BrainFlow conflicts
        np.random.seed(int(time.time() * 1000) % 10000)
        
        # Generate realistic EEG-like data
        t = np.arange(250) / 250.0  # 1 second at 250 Hz
        eeg8 = np.zeros((8, 250))
        
        for ch in range(8):
            # Mix of frequencies typical in EEG
            alpha = 10 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
            beta = 5 * np.sin(2 * np.pi * 20 * t)    # 20 Hz beta
            noise = np.random.randn(250) * 15         # Background noise
            eeg8[ch, :] = alpha + beta + noise + np.random.randn() * 10
        
        return {
            "data": eeg8.tolist(),
            "shape": eeg8.shape,
            "channels": 8,
            "samples": 250,
            "sampling_rate": 250,
            "timestamp": time.time(),
            "data_range": {
                "min": float(eeg8.min()),
                "max": float(eeg8.max()),
                "mean": float(eeg8.mean()),
                "std": float(eeg8.std())
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error generating synthetic stream: {str(e)}")

@app.post("/predict-synthetic")
def predict_synthetic():
    """Run prediction on synthetic EEG data."""
    try:
        # Get synthetic data
        synthetic_response = get_synthetic_data()
        eeg_data = synthetic_response["data"]
        
        # Create inference request
        req = InferenceRequest(x=eeg_data, n_outputs=2)
        
        # Run prediction
        result = predict(req)
        
        # Add synthetic data info
        result["synthetic_data"] = True
        result["data_source"] = "brainflow_synthetic_board"
        
        return result
        
    except Exception as e:
        raise HTTPException(500, f"Error running synthetic prediction: {str(e)}")

# ===== NEW ENHANCED SYNTHETIC DATA ENDPOINTS =====

@app.post("/synthetic/generate-custom")
def generate_custom_synthetic(state: str = "meditation", preprocess: bool = True):
    """Generate custom synthetic EEG data for meditation detection"""
    try:
        generator = SyntheticEEGGenerator()
        data, label = generator.generate_window(state, preprocess)
        
        return {
            "data": data.tolist(),
            "label": label,
            "state": state,
            "shape": data.shape,
            "channels": data.shape[0],
            "samples": data.shape[1],
            "sampling_rate": 250,
            "window_seconds": 4.0,
            "preprocessed": preprocess,
            "data_range": {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std())
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating custom synthetic data: {str(e)}")

@app.post("/synthetic/generate-dataset")
def generate_synthetic_dataset(n_samples_per_class: int = 100):
    """Generate a dataset of synthetic EEG samples"""
    if n_samples_per_class > 1000:
        raise HTTPException(400, "Maximum 1000 samples per class to prevent timeout")
    
    try:
        generator = SyntheticEEGGenerator()
        dataset = generator.generate_dataset(n_samples_per_class)
        
        return {
            "message": f"Generated {dataset['data'].shape[0]} samples",
            "shape": dataset['data'].shape,
            "metadata": dataset['metadata'],
            "class_distribution": {
                "non_meditation": int(np.sum(dataset['labels'] == 0)),
                "meditation": int(np.sum(dataset['labels'] == 1))
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Error generating dataset: {str(e)}")

@app.post("/synthetic/connect-board")
def connect_synthetic_board():
    """Connect to BrainFlow Synthetic Board for streaming"""
    global synthetic_manager
    try:
        if synthetic_manager is None:
            synthetic_manager = UnifiedBoardManager()
        
        success = synthetic_manager.connect_synthetic()
        if success:
            return {
                "status": "connected",
                "board_type": "synthetic",
                "message": "Successfully connected to BrainFlow Synthetic Board"
            }
        else:
            raise HTTPException(500, "Failed to connect to synthetic board")
    except Exception as e:
        raise HTTPException(500, f"Error connecting to synthetic board: {str(e)}")

@app.post("/synthetic/start-stream")
def start_synthetic_stream():
    """Start streaming from synthetic board"""
    global synthetic_manager
    try:
        if synthetic_manager is None:
            raise HTTPException(400, "No synthetic board connected. Call /synthetic/connect-board first")
        
        success = synthetic_manager.start_stream()
        if success:
            return {
                "status": "streaming",
                "message": "Synthetic data streaming started"
            }
        else:
            raise HTTPException(500, "Failed to start synthetic stream")
    except Exception as e:
        raise HTTPException(500, f"Error starting synthetic stream: {str(e)}")

@app.get("/synthetic/get-window")
def get_synthetic_window(window_seconds: float = 4.0):
    """Get a window of data from synthetic board stream"""
    global synthetic_manager
    try:
        if synthetic_manager is None:
            raise HTTPException(400, "No synthetic board connected")
        
        window = synthetic_manager.get_window(window_seconds)
        if window is not None:
            return {
                "data": window.tolist(),
                "shape": window.shape,
                "channels": window.shape[0],
                "samples": window.shape[1],
                "sampling_rate": 250,
                "window_seconds": window_seconds,
                "data_range": {
                    "min": float(window.min()),
                    "max": float(window.max()),
                    "mean": float(window.mean()),
                    "std": float(window.std())
                }
            }
        else:
            raise HTTPException(500, "Failed to get window from synthetic board")
    except Exception as e:
        raise HTTPException(500, f"Error getting synthetic window: {str(e)}")

@app.post("/synthetic/predict-custom")
def predict_custom_synthetic(state: str = "meditation"):
    """Generate custom synthetic data and run prediction"""
    try:
        # Generate custom data
        generator = SyntheticEEGGenerator()
        data, true_label = generator.generate_window(state, preprocess=True)
        
        # Run prediction
        model = get_model(DEFAULT_NCH, DEFAULT_NTIMES, DEFAULT_NOUT)
        x_tensor = torch.from_numpy(data).unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            logits = model(x_tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
        
        predicted_label = int(np.argmax(probs))
        
        return {
            "probs": probs.tolist(),
            "predicted_label": predicted_label,
            "true_label": true_label,
            "state": state,
            "correct": predicted_label == true_label,
            "confidence": float(np.max(probs)),
            "n_chans": data.shape[0],
            "n_times": data.shape[1],
            "window_seconds": 4.0,
            "synthetic_data": True
        }
    except Exception as e:
        raise HTTPException(500, f"Error in synthetic prediction: {str(e)}")

@app.post("/synthetic/disconnect")
def disconnect_synthetic():
    """Disconnect from synthetic board"""
    global synthetic_manager
    try:
        if synthetic_manager:
            synthetic_manager.disconnect()
            synthetic_manager = None
        
        return {
            "status": "disconnected",
            "message": "Synthetic board disconnected"
        }
    except Exception as e:
        raise HTTPException(500, f"Error disconnecting synthetic board: {str(e)}")

@app.get("/synthetic/board-info")
def get_synthetic_board_info():
    """Get information about synthetic board connection"""
    global synthetic_manager
    try:
        if synthetic_manager:
            info = synthetic_manager.get_board_info()
            return info
        else:
            return {"connected": False, "type": None}
    except Exception as e:
        raise HTTPException(500, f"Error getting board info: {str(e)}")

# ===== TRAINING INTERFACE ENDPOINTS =====

def run_training_background(config: TrainingConfig):
    """Background training function that runs in a separate thread"""
    global _training_status, _stop_training_flag, _trainer
    
    try:
        _training_status["is_training"] = True
        _training_status["start_time"] = time.time()
        _training_status["config"] = config.dict()
        
        # Create trainer
        _trainer = SyntheticEEGTrainer(
            n_channels=DEFAULT_NCH,
            n_times=DEFAULT_NTIMES,
            n_classes=DEFAULT_NOUT
        )
        
        # Setup model
        weights_dir = Path("./weights")
        pretrained_path = weights_dir / "labram_checkpoint.pth"
        
        _trainer.setup_model(
            freeze_backbone=config.freeze_backbone,
            pretrained_path=pretrained_path if pretrained_path.exists() else None
        )
        
        # Generate data
        train_loader, val_loader = _trainer.generate_data(
            n_samples_per_class=config.n_samples_per_class,
            val_split=config.val_split
        )
        
        _training_status["total_epochs"] = config.n_epochs
        best_val_acc = 0
        
        # Training loop
        for epoch in range(config.n_epochs):
            if _stop_training_flag:
                print("Training stopped by user")
                break
                
            _training_status["current_epoch"] = epoch + 1
            
            # Train epoch
            train_loss, train_acc = _trainer.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = _trainer.validate(val_loader)
            
            # Update status
            _training_status["train_loss"] = train_loss
            _training_status["val_loss"] = val_loss
            _training_status["train_acc"] = train_acc
            _training_status["val_acc"] = val_acc
            _training_status["elapsed_time"] = time.time() - _training_status["start_time"]
            
            # Calculate ETA
            if epoch > 0:
                avg_epoch_time = _training_status["elapsed_time"] / (epoch + 1)
                remaining_epochs = config.n_epochs - (epoch + 1)
                _training_status["eta"] = avg_epoch_time * remaining_epochs
            
            # Record history
            _training_status["history"]["train_losses"].append(train_loss)
            _training_status["history"]["val_losses"].append(val_loss)
            _training_status["history"]["train_accs"].append(train_acc)
            _training_status["history"]["val_accs"].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _training_status["best_val_acc"] = best_val_acc
                
                save_dir = Path(config.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': _trainer.model.state_dict(),
                    'optimizer_state_dict': _trainer.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'n_channels': DEFAULT_NCH,
                    'n_times': DEFAULT_NTIMES,
                    'n_classes': DEFAULT_NOUT
                }, save_dir / "best_synthetic_model.pth")
        
        # Save final training history
        save_dir = Path(config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_losses': _training_status["history"]["train_losses"],
            'val_losses': _training_status["history"]["val_losses"],
            'train_accs': _training_status["history"]["train_accs"],
            'val_accs': _training_status["history"]["val_accs"],
            'best_val_acc': best_val_acc
        }
        
        with open(save_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        print(f"Training error: {e}")
        _training_status["error"] = str(e)
    finally:
        _training_status["is_training"] = False
        _stop_training_flag = False

@app.post("/training/start")
def start_training(config: TrainingConfig = None):
    """Start model training with specified configuration"""
    global _training_thread, _training_status, _stop_training_flag
    
    if _training_status["is_training"]:
        raise HTTPException(400, "Training is already in progress. Stop current training first.")
    
    if not LABRAM_AVAILABLE:
        raise HTTPException(500, "LaBraM model not available. Please install braindecode>=1.1.0")
    
    try:
        # Reset status
        _training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "train_acc": 0.0,
            "val_acc": 0.0,
            "best_val_acc": 0.0,
            "start_time": None,
            "elapsed_time": 0.0,
            "eta": None,
            "config": None,
            "history": {
                "train_losses": [],
                "val_losses": [],
                "train_accs": [],
                "val_accs": []
            }
        }
        _stop_training_flag = False
        
        # Use default config if none provided
        if config is None:
            config = TrainingConfig()
        
        # Start training in background thread
        _training_thread = threading.Thread(
            target=run_training_background,
            args=(config,),
            daemon=True
        )
        _training_thread.start()
        
        return {
            "success": True,
            "message": "Training started successfully",
            "config": config.dict()
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error starting training: {str(e)}")

@app.post("/training/stop")
def stop_training():
    """Stop current training"""
    global _stop_training_flag, _training_status
    
    if not _training_status["is_training"]:
        raise HTTPException(400, "No training is currently in progress")
    
    try:
        _stop_training_flag = True
        
        return {
            "success": True,
            "message": "Training stop requested. Will stop after current epoch."
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error stopping training: {str(e)}")

@app.get("/training/status")
def get_training_status():
    """Get current training status and progress"""
    global _training_status
    
    try:
        status = dict(_training_status)  # Create a copy using dict()
        
        # Add formatted time information
        if status.get("start_time") is not None:
            status["elapsed_time_formatted"] = format_duration(status.get("elapsed_time", 0))
            if status.get("eta") is not None:
                status["eta_formatted"] = format_duration(status["eta"])
        
        return status
    except Exception as e:
        # Return basic status if there's an error
        return {
            "is_training": False,
            "error": f"Status error: {str(e)}",
            "current_epoch": 0,
            "total_epochs": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "train_acc": 0.0,
            "val_acc": 0.0,
            "best_val_acc": 0.0,
            "elapsed_time": 0.0,
            "history": {
                "train_losses": [],
                "val_losses": [],
                "train_accs": [],
                "val_accs": []
            }
        }

@app.get("/training/history")
def get_training_history():
    """Get training history and metrics"""
    global _training_status
    
    return {
        "history": _training_status["history"],
        "current_epoch": _training_status["current_epoch"],
        "total_epochs": _training_status["total_epochs"],
        "best_val_acc": _training_status["best_val_acc"]
    }

@app.get("/training/config")
def get_default_training_config():
    """Get default training configuration"""
    return TrainingConfig().dict()

@app.post("/training/config/validate")
def validate_training_config(config: TrainingConfig):
    """Validate training configuration parameters"""
    errors = []
    
    if config.n_epochs <= 0:
        errors.append("n_epochs must be positive")
    if config.n_samples_per_class <= 0:
        errors.append("n_samples_per_class must be positive")
    if not 0 < config.val_split < 1:
        errors.append("val_split must be between 0 and 1")
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    if config.weight_decay < 0:
        errors.append("weight_decay must be non-negative")
    
    if errors:
        return {"valid": False, "errors": errors}
    else:
        return {"valid": True, "message": "Configuration is valid"}

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# ===== REAL DATA TRAINING ENDPOINTS =====

_real_trainer = None
_real_training_status = {
    "is_training": False,
    "current_step": "",
    "progress": 0.0,
    "error": None
}

@app.post("/training/real-data/start")
def start_real_data_training(config: RealDataTrainingConfig):
    """Start training on real OpenNeuro EEG data"""
    global _real_trainer, _real_training_status
    
    if _real_training_status["is_training"]:
        raise HTTPException(400, "Real data training already in progress")
    
    try:
        _real_training_status = {
            "is_training": True,
            "current_step": "Initializing",
            "progress": 0.0,
            "error": None,
            "config": config.dict()
        }
        
        # Create trainer (this will be run in background in production)
        _real_trainer = RealDataTrainer(config.dataset_id)
        
        # For now, return success - in production this would be async
        return {
            "success": True,
            "message": f"Real data training started for dataset {config.dataset_id}",
            "config": config.dict(),
            "note": "This is a demo endpoint. Full implementation would run in background."
        }
        
    except Exception as e:
        _real_training_status["is_training"] = False
        _real_training_status["error"] = str(e)
        raise HTTPException(500, f"Error starting real data training: {str(e)}")

@app.get("/training/real-data/status")
def get_real_training_status():
    """Get real data training status"""
    return _real_training_status

@app.post("/training/real-data/download")
def download_openneuro_dataset(dataset_id: str = "ds003969", force_redownload: bool = False):
    """Download OpenNeuro dataset for training"""
    try:
        from real_data_trainer import OpenNeuroDatasetLoader
        
        loader = OpenNeuroDatasetLoader(dataset_id)
        
        # Download
        if loader.download_dataset(force_redownload):
            # Extract
            if loader.extract_dataset():
                # Find files
                eeg_files = loader.find_eeg_files()
                participants_df = loader.load_participants_info()
                
                return {
                    "success": True,
                    "dataset_id": dataset_id,
                    "message": f"Dataset {dataset_id} downloaded and extracted successfully",
                    "eeg_files_found": len(eeg_files),
                    "participants_info": participants_df is not None,
                    "dataset_path": str(loader.extracted_path)
                }
            else:
                raise HTTPException(500, "Failed to extract dataset")
        else:
            raise HTTPException(500, "Failed to download dataset")
            
    except Exception as e:
        raise HTTPException(500, f"Error downloading dataset: {str(e)}")

@app.get("/training/real-data/datasets")
def list_available_datasets():
    """List available OpenNeuro datasets for meditation/EEG"""
    return {
        "available_datasets": [
            {
                "id": "ds003969",
                "name": "Meditation/Mindfulness EEG Dataset",
                "description": "Real EEG data for meditation state detection",
                "url": "https://openneuro.org/datasets/ds003969"
            }
        ],
        "note": "Add more datasets as needed for different meditation studies"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

