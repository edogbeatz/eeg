# EEG API (Cyton x LaBraM)

A production-ready FastAPI server optimized for OpenBCI Cyton 8-channel EEG data with LaBraM brain signal processing.

## 🧠 Features

- **8-channel Cyton optimized** (250 Hz, 4-second windows)
- **LaBraM integration** for brain signal classification
- **Real-time preprocessing** (DC removal, z-score normalization)
- **Pre-trained model loading** (HuggingFace/URL support)
- **Production deployment** on Render

## 🚀 Live API

**Health Check**: https://eeg-3j9h.onrender.com/health
**API Docs**: https://eeg-3j9h.onrender.com/docs
**Prediction**: https://eeg-3j9h.onrender.com/predict

## 📊 API Usage

### Health Check
```bash
curl https://eeg-3j9h.onrender.com/health
```

### EEG Prediction (8×1000 samples)
```bash
curl -X POST "https://eeg-3j9h.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "x": [
      [0.01,0.02,0.01,0.00, ... 1000 samples ...],
      [0.00,0.01,-0.01,0.00, ...],
      [ ... 6 more channels ... ]
    ],
    "n_outputs": 2
  }'
```

## 🔧 Local Setup (for Cyton data collection)

### 1. Install BrainFlow
```bash
pip install -r collector_requirements.txt
```

### 2. Find your Cyton port
- **macOS**: `/dev/tty.usbserial-DM03...` or similar
- **Windows**: `COM3`, `COM4`, etc.
- **Linux**: `/dev/ttyUSB0` or similar

### 3. Update collector.py
Edit `collector.py` and set your port:
```python
params.serial_port = "YOUR_CYTON_PORT_HERE"
```

### 4. Run collector
```bash
python collector.py
```

## 🎯 Cyton Specifications

- **Channels**: 8 (DEFAULT_NCH = 8)
- **Sample Rate**: 250 Hz (CYTON_SR = 250)
- **Window Size**: 4 seconds (1000 samples)
- **Preprocessing**: DC removal + z-score per channel

## 🔄 Real-time Flow

1. **Cyton** streams 8-channel EEG at 250 Hz
2. **Collector** captures 4-second windows (1000 samples)
3. **API** preprocesses and runs LaBraM inference
4. **Response** contains classification probabilities

## 🛠️ Environment Variables (Optional)

Set these in Render dashboard for model loading:

- `WEIGHTS_URL`: Direct URL to .pth checkpoint
- `HF_REPO_ID`: HuggingFace repository ID
- `HF_FILENAME`: Model filename in repo
- `HF_TOKEN`: HuggingFace token (if private)

## 📈 Performance

- **Latency**: ~100-200ms per prediction
- **Throughput**: Handles real-time streaming
- **Memory**: Efficient model caching
- **Robustness**: Hardware-agnostic preprocessing

## 🚀 Deployment

The API is already deployed on Render with:
- **Auto-deploy** from GitHub
- **HTTPS** enabled
- **Scaling** ready
- **Monitoring** available

## 📝 Example Response

```json
{
  "probs": [0.65, 0.35],
  "n_chans": 8,
  "n_times": 1000,
  "window_seconds": 4.0
}
```

## 🔍 Troubleshooting

1. **Port not found**: Check device manager/terminal for Cyton port
2. **API timeout**: Check internet connection to Render
3. **Model errors**: Verify input shape is exactly 8×1000
4. **Preprocessing**: Data is automatically z-scored per channel

## 🧪 Testing

Test with synthetic data:
```python
import numpy as np
import requests

# Generate 8×1000 test data
data = np.random.randn(8, 1000).tolist()

response = requests.post(
    "https://eeg-3j9h.onrender.com/predict",
    json={"x": data, "n_outputs": 2}
)
print(response.json())
```