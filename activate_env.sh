#!/bin/bash
# Activate the EEG virtual environment
# Usage: source activate_env.sh

echo "🚀 Activating EEG virtual environment with Python 3.12..."
source eeg-env/bin/activate
echo "✅ Virtual environment activated!"
echo "   Python: $(which python3) ($(python3 --version))"
echo "   To deactivate: run 'deactivate'"
