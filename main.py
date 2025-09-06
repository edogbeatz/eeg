from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F

app = FastAPI(title="EEG API (starter)")

@app.get("/health")
def health():
    return {"ok": True}

class InferenceRequest(BaseModel):
    x: list[list[float]]  # [n_chans][n_times]
    n_outputs: int = 2

@app.post("/predict")
def predict(req: InferenceRequest):
    try:
        # Import Labram here to handle any import issues gracefully
        from braindecode.models import Labram
        
        x = torch.tensor(req.x, dtype=torch.float32)   # (ch, time)
        n_chans, n_times = x.shape
        
        # Add small epsilon to avoid zero values that might cause numerical issues
        x = x + 1e-8
        
        # Create a new model instance for each request (for simplicity)
        model = Labram(n_chans=n_chans, n_times=n_times, n_outputs=req.n_outputs, neural_tokenizer=False)
        model.eval()
        
        with torch.no_grad():
            logits = model(x.unsqueeze(0))             # (1, n_outputs)
            probs = F.softmax(logits, dim=1).squeeze(0).tolist()
        
        return {"probs": probs, "n_chans": n_chans, "n_times": n_times, "status": "success"}
    
    except Exception as e:
        # Fallback: return random probabilities for demonstration
        import random
        n_chans = len(req.x) if req.x else 0
        n_times = len(req.x[0]) if req.x and len(req.x) > 0 else 0
        probs = [random.random() for _ in range(req.n_outputs)]
        # Normalize to sum to 1
        total = sum(probs)
        probs = [p/total for p in probs] if total > 0 else [1/req.n_outputs] * req.n_outputs
        
        return {
            "probs": probs, 
            "n_chans": n_chans, 
            "n_times": n_times,
            "status": "fallback",
            "error": str(e)
        }

