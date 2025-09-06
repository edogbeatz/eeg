import os
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from braindecode.models import Labram

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
    x: list[list[float]]
    n_outputs: int | None = None   # falls back to DEFAULT_NOUT

_model = None
_shape = None
_loaded_ckpt: Optional[Path] = None

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
        "window_seconds": n_times / CYTON_SR
    }

