# EEG Meditation Detection API (OpenBCI Cyton x LaBraM)

A comprehensive FastAPI server for real-time EEG meditation detection with OpenBCI Cyton boards, featuring electrode detection, impedance testing, and LaBraM-based meditation state classification.

## 🧠 Features

### **Core Functionality**
- **Real-time EEG streaming** from OpenBCI Cyton (8 channels, 250 Hz)
- **LaBraM integration** for meditation state detection and classification  
- **Electrode detection** with impedance testing (OpenBCI z-command protocol)
- **Live signal quality monitoring** with railing detection
- **Ring buffer implementation** for sliding window processing
- **MNE compatibility** with microvolts to volts conversion

### **Production Ready**
- **Background data streaming** with thread-safe operations
- **Comprehensive API endpoints** for real-time BCI applications
- **Robust error handling** and status monitoring
- **Pre-trained model loading** (HuggingFace/URL support)
- **Auto-deployment** on Render

## 🚀 Live API

**Health Check**: https://eeg-3j9h.onrender.com/health  
**API Documentation**: https://eeg-3j9h.onrender.com/docs  
**Interactive Testing**: https://eeg-3j9h.onrender.com/redoc

## 📊 API Endpoints

### **Core Prediction**
- `POST /predict` - Run LaBraM meditation detection on EEG data
- `GET /health` - API health and model status

### **BrainFlow/LaBraM Pipeline**
- `POST /start-pipeline` - Initialize complete streaming pipeline
- `GET /pipeline-status` - Get pipeline status and statistics
- `POST /predict-realtime` - Real-time predictions on live stream
- `GET /current-window` - Get current sliding window data
- `POST /stop-pipeline` - Clean pipeline shutdown

### **Electrode Detection**
- `POST /connect-board` - Connect to OpenBCI Cyton board
- `GET /board-status` - Board information and connection status
- `POST /electrode-impedance` - Measure electrode impedance
- `POST /live-quality` - Live electrode quality monitoring
- `GET /battery` - Battery level monitoring
- `POST /disconnect-board` - Disconnect from board

### **Board Information**
- `GET /board-info` - Detailed board specifications
- `GET /board-status` - Connection and streaming status

## 🔧 Local Setup

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Find Your Cyton Port**
- **macOS**: `/dev/cu.usbserial-DM01N8KH` or similar
- **Windows**: `COM3`, `COM4`, etc.
- **Linux**: `/dev/ttyUSB0` or similar

```bash
# List available ports (macOS/Linux)
ls /dev/cu.usbserial-*
```

### 3. **Start API Server**
```bash
uvicorn main:app --reload
```

### 4. **Connect and Stream**
```bash
# Connect to board
curl -X POST "http://localhost:8000/start-pipeline" \
     -H "Content-Type: application/json" \
     -d '{"serial_port": "/dev/cu.usbserial-DM01N8KH"}'

# Get real-time predictions
curl -X POST "http://localhost:8000/predict-realtime"
```

## 🧪 Usage Examples

### **1. Real-time EEG Collection**
```python
# Run the enhanced collector
python enhanced_collector.py /dev/cu.usbserial-DM01N8KH http://localhost:8000
```

### **2. Electrode Impedance Testing**
```python
# Run electrode detection demo
python electrode_detection_example.py /dev/cu.usbserial-DM01N8KH
```

### **3. API Integration**
```python
import requests

# Start pipeline
response = requests.post("http://localhost:8000/start-pipeline", 
                        json={"serial_port": "/dev/cu.usbserial-DM01N8KH"})

# Check impedance for all channels
response = requests.post("http://localhost:8000/electrode-impedance",
                        json={"channels": [1,2,3,4,5,6,7,8], "samples": 250})

# Get real-time prediction
response = requests.post("http://localhost:8000/predict-realtime")
print(response.json())
```

### **4. Manual Data Submission**
```python
import numpy as np
import requests

# Generate test data (8 channels × 1000 samples)
data = np.random.randn(8, 1000) * 50  # Simulate EEG in microvolts
payload = {"x": data.tolist(), "n_outputs": 2}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
```

## 🎯 Technical Specifications

### **OpenBCI Cyton Configuration**
- **Channels**: 8 EEG channels
- **Sampling Rate**: 250 Hz
- **Window Size**: 4 seconds (1000 samples)
- **Resolution**: 24-bit ADC
- **Scale Factor**: 0.02235 µV/count (gain x24)

### **Electrode Detection**
- **Impedance Testing**: 31.5 Hz test signal injection
- **Drive Current**: 6 nA
- **Series Resistor**: 2.2 kΩ
- **Thresholds**: 
  - Good: < 750 kΩ
  - Moderate: 750-1500 kΩ
  - Poor/Disconnected: > 1500 kΩ

### **Data Processing**
- **Preprocessing**: DC removal + z-score normalization per channel
- **Buffer Management**: Ring buffer for sliding windows
- **Unit Conversion**: Microvolts to volts (MNE compatibility)
- **Threading**: Thread-safe background data collection

### **LaBraM Model**
- **Architecture**: Transformer-based foundation model
- **Parameters**: ~5.9M parameters
- **Input Format**: (batch, n_chans, n_times) with neural_tokenizer=True
- **Training Data**: ~2,500 hours from ~20 EEG datasets

## 🔄 Data Flow Pipeline

```
OpenBCI Cyton → BrainFlow → Ring Buffer → Preprocessing → LaBraM → Predictions
     ↓              ↓           ↓            ↓           ↓
   Serial USB   Background   Sliding      DC + Z-score  Classification
   250 Hz       Threading    Windows      Normalization  Probabilities
```

## 📁 Project Structure

```
eeg/
├── main.py                          # FastAPI server with all endpoints
├── electrode_detection.py           # Electrode detection and impedance testing
├── brainflow_labram_integration.py  # Complete BrainFlow/LaBraM pipeline
├── enhanced_collector.py            # Advanced data collection demo
├── electrode_detection_example.py   # Electrode detection demo
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── ELECTRODE_DETECTION_README.md    # Electrode detection documentation
├── BRAINFLOW_LABRAM_ANALYSIS.md     # Implementation analysis
├── Procfile                         # Render deployment config
└── runtime.txt                      # Python version for deployment
```

## 🛠️ Environment Variables (Optional)

Set these for pre-trained model loading:

- `WEIGHTS_URL`: Direct URL to .pth checkpoint
- `HF_REPO_ID`: HuggingFace repository ID
- `HF_FILENAME`: Model filename in repo
- `HF_TOKEN`: HuggingFace token (if private)
- `WEIGHTS_DIR`: Local weights directory (default: ./weights)

## 📈 Performance & Monitoring

### **Latency**
- **Prediction**: ~100-200ms per inference
- **Impedance Test**: ~1-2 seconds per channel
- **Pipeline Startup**: ~3-5 seconds

### **Throughput**
- **Real-time**: 250 samples/second sustained
- **Concurrent**: Multiple API requests supported
- **Memory**: ~32KB ring buffer + model weights

### **Monitoring**
- **Pipeline Status**: Real-time statistics and health checks
- **Electrode Quality**: Continuous monitoring with alerts
- **Error Handling**: Comprehensive logging and recovery

## 🚀 Deployment

### **Render (Production)**
The API is deployed on Render with:
- ✅ Auto-deploy from GitHub
- ✅ HTTPS enabled  
- ✅ Environment variable management
- ✅ Monitoring and logging
- ✅ Automatic scaling

### **Local Development**
```bash
# Clone repository
git clone <repository-url>
cd eeg

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### **Docker (Optional)**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "$PORT"]
```

## 🔍 Troubleshooting

### **Connection Issues**
1. **Board not found**: Check USB connection and port permissions
2. **Port busy**: Close other applications using the Cyton board
3. **Permission denied**: Add user to dialout group (Linux) or check admin rights

### **Impedance Testing**
1. **High impedance values**: Clean electrodes and improve skin contact
2. **Inconsistent readings**: Ensure subject is still during testing
3. **Test failures**: Check electrode connections and board status

### **API Issues**
1. **Timeout errors**: Check board connection and streaming status
2. **Model errors**: Verify input data shape (8 × 1000)
3. **Memory issues**: Restart pipeline to clear buffers

### **Performance Issues**
1. **High latency**: Check CPU usage and optimize model settings
2. **Dropped samples**: Reduce background processes and check USB connection
3. **Memory leaks**: Monitor pipeline status and restart if needed

## 📚 Documentation

- **[Electrode Detection Guide](ELECTRODE_DETECTION_README.md)** - Comprehensive electrode detection documentation
- **[Implementation Analysis](BRAINFLOW_LABRAM_ANALYSIS.md)** - Technical implementation details
- **[API Documentation](https://eeg-3j9h.onrender.com/docs)** - Interactive API explorer
- **[OpenBCI Documentation](https://docs.openbci.com/)** - Official OpenBCI guides
- **[BrainFlow Documentation](https://brainflow.readthedocs.io/)** - BrainFlow SDK reference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **OpenBCI** for the Cyton board and documentation
- **BrainFlow** for the excellent SDK
- **BrainDecode** for LaBraM model implementation
- **FastAPI** for the robust web framework