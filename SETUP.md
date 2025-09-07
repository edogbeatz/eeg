# EEG System Setup Guide

## Quick Setup (New Environment)

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd eeg
```

### 2. Set Up Python Environment

#### Option A: Use the provided activation script (Recommended)
```bash
# Create virtual environment with Python 3.12
python3.12 -m venv eeg-env

# Activate using provided script
source activate_env.sh

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Manual setup
```bash
# Create virtual environment
python3.12 -m venv eeg-env

# Activate manually
source eeg-env/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python3 -c "
from braindecode.models import Labram
print('✅ LaBraM model available')
print('✅ Setup complete!')
"
```

## Requirements

- **Python 3.10+** (3.12 recommended for latest braindecode)
- **macOS/Linux** (Windows may work but not tested)
- **OpenBCI Cyton board** (for real EEG data)

## Key Dependencies

- `braindecode>=1.1.0` - For LaBraM neural network model
- `brainflow` - OpenBCI board communication
- `fastapi` - API server
- `torch` - Deep learning framework
- `numpy`, `scipy` - Scientific computing

## Usage

### Start API Server
```bash
# Make sure virtual environment is activated
source activate_env.sh

# Start server
uvicorn main:app --reload
```

### Test with Mock Data
```bash
# Test realistic EEG simulation
python3 test_cyton_mock.py

# Test continuous processing
python3 test_enhanced_collector_mock.py

# Full API demo
python3 demo_api_server.py
```

### Connect Real OpenBCI Board
```bash
# Find your serial port
ls /dev/cu.usbserial*

# Run enhanced collector
python3 enhanced_collector.py /dev/cu.usbserial-YOUR_PORT http://localhost:8000
```

## Important Notes

⚠️ **Virtual Environment**: The `eeg-env/` directory is excluded from git (1GB+ size).
Each user must create their own virtual environment.

⚠️ **Python Version**: LaBraM model requires Python 3.10+. Use Python 3.12 for best compatibility.

⚠️ **Dependencies**: Some packages require compilation. Allow extra time for first install.

## Troubleshooting

### Python Version Issues
```bash
# Check Python version
python3 --version

# If < 3.10, install newer Python:
# macOS: brew install python@3.12
# Ubuntu: sudo apt install python3.12
```

### Package Installation Errors
```bash
# If you get externally-managed-environment error:
python3 -m venv eeg-env
source eeg-env/bin/activate
pip install -r requirements.txt
```

### BrainFlow Connection Issues
```bash
# Check USB permissions (Linux)
sudo usermod -a -G dialout $USER

# Check serial port
ls /dev/cu.usbserial* # macOS
ls /dev/ttyUSB* # Linux
```

## Development

### Adding New Dependencies
```bash
# Install new package
pip install new-package

# Update requirements
pip freeze > requirements.txt
```

### Running Tests
```bash
# Quick system test
python3 -c "from main import health; print(health())"

# Full test suite
python3 test_cyton_mock.py
```
