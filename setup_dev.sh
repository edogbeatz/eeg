#!/bin/bash
# Development Environment Setup Script
# This script sets up the complete EEG development environment

set -e  # Exit on any error

echo "🚀 EEG System Development Setup"
echo "================================"
echo

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "❌ Python ${PYTHON_VERSION} found, but Python 3.10+ required."
    echo "   Install with: brew install python@3.12  # macOS"
    echo "   Or: sudo apt install python3.12  # Ubuntu"
    exit 1
fi

echo "✅ Python ${PYTHON_VERSION} found"

# Create virtual environment
echo
echo "📦 Creating virtual environment..."
if [ -d "eeg-env" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf eeg-env
fi

python3 -m venv eeg-env
echo "✅ Virtual environment created"

# Activate virtual environment
echo
echo "🔌 Activating virtual environment..."
source eeg-env/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo
echo "⬆️  Upgrading pip..."
pip install --upgrade pip
echo "✅ Pip upgraded"

# Install dependencies
echo
echo "📚 Installing dependencies..."
echo "   This may take several minutes..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Verify installation
echo
echo "🧪 Verifying installation..."

# Test imports
python3 -c "
import sys
print(f'✅ Python {sys.version.split()[0]}')

try:
    import torch
    print('✅ PyTorch imported')
except ImportError as e:
    print(f'❌ PyTorch error: {e}')
    sys.exit(1)

try:
    import fastapi
    print('✅ FastAPI imported')
except ImportError as e:
    print(f'❌ FastAPI error: {e}')
    sys.exit(1)

try:
    import brainflow
    print('✅ BrainFlow imported')
except ImportError as e:
    print(f'❌ BrainFlow error: {e}')
    sys.exit(1)

try:
    from braindecode.models import Labram
    print('✅ LaBraM model available')
except ImportError as e:
    print(f'❌ LaBraM error: {e}')
    print('   This might be due to braindecode version compatibility')

print('✅ Core imports successful')
"

# Test API
echo
echo "🌐 Testing API..."
python3 -c "
from main import health
result = health()
print(f'✅ API health check: {result}')
"

echo
echo "🎉 Setup completed successfully!"
echo
echo "📋 Next steps:"
echo "   1. Activate environment: source activate_env.sh"
echo "   2. Start API server: uvicorn main:app --reload"
echo "   3. Test with mock data: python3 test_cyton_mock.py"
echo "   4. Connect real board: python3 enhanced_collector.py [serial_port] [api_url]"
echo
echo "📚 Documentation:"
echo "   • Setup guide: SETUP.md"
echo "   • API docs: http://localhost:8000/docs (after starting server)"
echo "   • Analysis: BRAINFLOW_LABRAM_ANALYSIS.md"
echo
echo "✅ Development environment ready!"
