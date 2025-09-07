# EEG Synthetic Data Pipeline Makefile
# Provides convenient commands for the complete synthetic data workflow

.PHONY: help install synth train test serve clean demo frontend

# Default target
help:
	@echo "🧠 EEG Synthetic Data Pipeline"
	@echo "=============================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install Python dependencies"
	@echo "  make synth       - Generate synthetic EEG dataset"
	@echo "  make train       - Train model on synthetic data"
	@echo "  make test        - Test trained model"
	@echo "  make serve       - Start FastAPI server"
	@echo "  make frontend    - Start frontend development server"
	@echo "  make demo        - Run complete demo pipeline"
	@echo "  make clean       - Clean generated files"
	@echo ""
	@echo "Quick start:"
	@echo "  make install && make demo"

# Install dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Generate synthetic dataset
synth:
	@echo "🔬 Generating synthetic EEG dataset..."
	python -c "from synthetic_data_generator import main; main()"
	@echo "✅ Synthetic data generated"

# Train model
train:
	@echo "🎓 Training model on synthetic data..."
	python train_synthetic.py
	@echo "✅ Training complete"

# Test model
test:
	@echo "🧪 Testing trained model..."
	python test_synthetic_model.py
	@echo "✅ Testing complete"

# Start API server
serve:
	@echo "🚀 Starting FastAPI server..."
	@echo "   API will be available at: http://localhost:8000"
	@echo "   Docs available at: http://localhost:8000/docs"
	@echo ""
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Start frontend
frontend:
	@echo "🎨 Starting frontend development server..."
	@echo "   Frontend will be available at: http://localhost:3000"
	@echo ""
	cd frontend && npm run dev

# Run complete demo
demo:
	@echo "🎪 Running complete synthetic EEG demo..."
	@echo ""
	@echo "Step 1: Generating synthetic data..."
	@make synth
	@echo ""
	@echo "Step 2: Training model..."
	@make train
	@echo ""
	@echo "Step 3: Testing model..."
	@make test
	@echo ""
	@echo "🎉 Demo complete! Check ./trained_models/ for results"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Start API server: make serve"
	@echo "  2. Start frontend: make frontend"
	@echo "  3. Test synthetic endpoints at http://localhost:8000/docs"

# Test synthetic board integration
test-synthetic-board:
	@echo "🔌 Testing BrainFlow Synthetic Board integration..."
	python -c "from brainflow_synthetic_integration import demo_synthetic_integration; demo_synthetic_integration()"

# Test API endpoints
test-api:
	@echo "🌐 Testing API endpoints..."
	@echo "Make sure API server is running (make serve)"
	@echo ""
	@echo "Testing health endpoint..."
	curl -s http://localhost:8000/health | python -m json.tool || echo "❌ API not responding"
	@echo ""
	@echo "Testing synthetic data generation..."
	curl -s -X POST http://localhost:8000/synthetic/generate-custom?state=relaxed | python -m json.tool || echo "❌ Synthetic endpoint not working"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf synthetic_data/
	rm -rf trained_models/
	rm -rf __pycache__/
	rm -rf *.pyc
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Development setup
dev-setup:
	@echo "🛠️  Setting up development environment..."
	@make install
	@echo ""
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo ""
	@echo "✅ Development setup complete"
	@echo ""
	@echo "Quick start:"
	@echo "  Terminal 1: make serve"
	@echo "  Terminal 2: make frontend"

# Production build
build:
	@echo "🏗️  Building for production..."
	cd frontend && npm run build
	@echo "✅ Production build complete"

# Run all tests
test-all:
	@echo "🔍 Running all tests..."
	@make test-synthetic-board
	@make test
	@echo "✅ All tests complete"

# Quick synthetic prediction test
quick-test:
	@echo "⚡ Quick synthetic prediction test..."
	python -c "
from synthetic_data_generator import SyntheticEEGGenerator
import numpy as np
print('Generating relaxed and anxious samples...')
gen = SyntheticEEGGenerator()
for state in ['relaxed', 'anxious']:
    data, label = gen.generate_window(state)
    print(f'{state}: shape={data.shape}, label={label}, range=[{data.min():.2f}, {data.max():.2f}]')
print('✅ Synthetic generation working!')
"

# Show project status
status:
	@echo "📋 Project Status"
	@echo "================="
	@echo ""
	@echo "Files:"
	@ls -la *.py | grep -E "(synthetic|train|test)" || echo "  No Python files found"
	@echo ""
	@echo "Synthetic data:"
	@if [ -d "synthetic_data" ]; then \
		echo "  ✅ synthetic_data/ exists"; \
		ls -la synthetic_data/ 2>/dev/null || true; \
	else \
		echo "  ❌ synthetic_data/ not found (run: make synth)"; \
	fi
	@echo ""
	@echo "Trained models:"
	@if [ -d "trained_models" ]; then \
		echo "  ✅ trained_models/ exists"; \
		ls -la trained_models/ 2>/dev/null || true; \
	else \
		echo "  ❌ trained_models/ not found (run: make train)"; \
	fi
	@echo ""
	@echo "Frontend:"
	@if [ -d "frontend/node_modules" ]; then \
		echo "  ✅ Frontend dependencies installed"; \
	else \
		echo "  ❌ Frontend dependencies not installed (run: cd frontend && npm install)"; \
	fi
