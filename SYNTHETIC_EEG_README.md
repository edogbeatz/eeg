# Synthetic EEG Data Pipeline

A comprehensive implementation for bootstrapping EEG-based BCI applications using synthetic data. This system allows you to develop, train, and test your EEG pipeline without requiring physical hardware.

## 🎯 Overview

This synthetic data pipeline addresses the key challenge of getting started with EEG/BCI development: **you need data to train models, but you need hardware to get data**. Our solution:

1. **Generate realistic synthetic EEG data** with different mental states
2. **Train LaBraM models** on synthetic data for initial validation
3. **Test complete pipeline** including frontend visualization
4. **Prepare for real data** with minimal domain gap

## 🚀 Quick Start

```bash
# 1. Install dependencies
make install

# 2. Run complete demo pipeline
make demo

# 3. Start API server (in one terminal)
make serve

# 4. Start frontend (in another terminal)
make frontend
```

Visit http://localhost:3000 to see the EEG dashboard with synthetic data!

## 📁 New Files Added

### Core Implementation
- `synthetic_data_generator.py` - Generate realistic "relaxed vs anxious" EEG data
- `brainflow_synthetic_integration.py` - BrainFlow Synthetic Board integration
- `data_augmentation.py` - Domain gap reduction techniques
- `train_synthetic.py` - Training pipeline for synthetic data
- `test_synthetic_model.py` - Model testing and evaluation
- `Makefile` - Convenient command shortcuts

### Enhanced API Endpoints
New endpoints in `main.py`:
- `/synthetic/generate-custom` - Generate custom mental state data
- `/synthetic/generate-dataset` - Generate training datasets
- `/synthetic/connect-board` - Connect to BrainFlow Synthetic Board
- `/synthetic/start-stream` - Start synthetic data streaming
- `/synthetic/get-window` - Get streaming windows
- `/synthetic/predict-custom` - Generate and predict in one call

## 🧠 Synthetic Data Features

### Mental State Modeling
- **Relaxed State**: High alpha (8-12 Hz), moderate theta, low beta
- **Anxious State**: Low alpha, high beta (12-30 Hz), elevated gamma

### Realistic Artifacts
- **Eye blinks** (frontal channels)
- **Muscle artifacts** (high-frequency bursts)
- **Powerline noise** (50/60 Hz)
- **1/f pink noise** baseline

### Data Augmentation (Domain Gap Reduction)
- Gaussian noise injection
- Amplitude scaling variations
- Time shifts and frequency shifts
- Channel dropout simulation
- Bandpass filter variations
- Realistic artifact injection

## 🎓 Training Pipeline

### 1. Generate Data
```python
from synthetic_data_generator import SyntheticEEGGenerator

generator = SyntheticEEGGenerator()
dataset = generator.generate_dataset(
    n_samples_per_class=1000,
    augmentation_ratio=0.5  # 50% of samples get augmentation
)
```

### 2. Train Model
```python
# Option 1: Command line
python train_synthetic.py

# Option 2: Makefile
make train
```

Features:
- **Freeze backbone**: Only train classification head for speed
- **Load pretrained**: Use existing LaBraM weights if available
- **Validation split**: Automatic train/validation splitting
- **Best model saving**: Saves best validation accuracy model
- **Training curves**: Automatic plotting of loss/accuracy

### 3. Test Model
```python
# Test on fresh synthetic data
python test_synthetic_model.py

# Or use Makefile
make test
```

## 🌐 API Integration

### BrainFlow Synthetic Board
```python
from brainflow_synthetic_integration import UnifiedBoardManager

# Connect to synthetic board
manager = UnifiedBoardManager()
manager.connect_synthetic()
manager.start_stream()

# Get EEG windows
window = manager.get_window(window_seconds=4.0)
```

### Custom Data Generation
```bash
# Generate relaxed state data
curl -X POST "http://localhost:8000/synthetic/generate-custom?state=relaxed"

# Generate and predict
curl -X POST "http://localhost:8000/synthetic/predict-custom?state=anxious"
```

## 🎨 Frontend Integration

The frontend now displays:
- **Real-time EEG waveforms** (enabled from synthetic data)
- **Channel grid visualization** 
- **Prediction results** with confidence scores
- **Data statistics** and quality metrics

Key update in `eeg-dashboard.tsx`:
```typescript
// Waveform visualization now enabled
<EEGWaveform 
  data={eegData.data} 
  samplingRate={eegData.sampling_rate}
  channels={eegData.channels}
/>
```

## 📊 Expected Performance

### Synthetic Data Training
- **Training time**: ~5-10 minutes (1500 samples/class, 15 epochs)
- **Validation accuracy**: 85-95% on synthetic test data
- **Model size**: ~5.9M parameters (LaBraM)
- **Inference speed**: ~100ms per window

### Domain Gap Considerations
- **Synthetic → Synthetic**: 90%+ accuracy expected
- **Synthetic → Real**: 60-80% accuracy expected (before fine-tuning)
- **After fine-tuning**: 80-90% accuracy expected (with minimal real data)

## 🔄 Migration to Real Data

### 1. Collect Real Data
```python
# Use existing real board integration
from brainflow_labram_integration import BrainFlowLaBraMPipeline

pipeline = BrainFlowLaBraMPipeline("/dev/cu.usbserial-DM01N8KH")
# ... collect real EEG windows
```

### 2. Fine-tune Model
```python
# Load synthetic-trained model
model = load_trained_model("./trained_models/best_synthetic_model.pth")

# Fine-tune on real data (even 50-100 samples per class helps significantly)
# Use lower learning rate: lr=1e-4
```

### 3. Playback File Board
```python
# Test with recorded OpenBCI data
manager = UnifiedBoardManager()
manager.connect_playback("path/to/recorded_data.csv")
```

## 🛠️ Development Commands

```bash
# Generate synthetic data only
make synth

# Train model only
make train

# Test model only  
make test

# Test synthetic board integration
make test-synthetic-board

# Quick synthetic generation test
make quick-test

# Clean all generated files
make clean

# Check project status
make status

# Test API endpoints
make test-api
```

## 🔬 Advanced Usage

### Custom Augmentation
```python
from data_augmentation import EEGAugmentation

augmenter = EEGAugmentation(sampling_rate=250)

# Custom augmentation config
config = {
    'gaussian_noise': 0.8,      # 80% chance
    'muscle_artifacts': 0.3,    # 30% chance
    'powerline_noise': 0.4,     # 40% chance
    # ... other augmentations
}

augmented_data = augmenter.apply_random_augmentations(eeg_data, config)
```

### Custom Mental States
```python
# Extend SyntheticEEGGenerator for new states
class ExtendedGenerator(SyntheticEEGGenerator):
    def generate_focused_state(self):
        # High beta, moderate gamma, low alpha
        # ... implementation
        pass
```

### Batch Processing
```python
# Generate large datasets
dataset = generator.generate_dataset(
    n_samples_per_class=5000,   # 10k total samples
    augmentation_ratio=0.7,     # Heavy augmentation
    output_dir="./large_dataset"
)
```

## 📈 Performance Optimization

### Training Speed
- **Freeze backbone**: 10x faster training
- **Smaller batches**: Better for limited GPU memory
- **Mixed precision**: Use `torch.cuda.amp` for speed

### Memory Usage
- **Ring buffer**: Only ~32KB for 4-second windows
- **Streaming**: Process data in chunks
- **Model quantization**: Reduce model size for deployment

## 🔍 Troubleshooting

### Common Issues

**1. BrainFlow Import Error**
```bash
pip install brainflow
```

**2. LaBraM Not Available**
```bash
pip install braindecode>=1.1.0
```

**3. Frontend Not Loading Waveforms**
- Check API is running on port 8000
- Verify CORS settings in `main.py`
- Check browser console for errors

**4. Low Synthetic Performance**
- Increase training epochs (15-30)
- Add more training data (2000+ per class)
- Adjust augmentation ratio (0.3-0.7)

### Debug Commands
```bash
# Test data generation
python -c "from synthetic_data_generator import main; main()"

# Test augmentation
python data_augmentation.py

# Test BrainFlow integration
python brainflow_synthetic_integration.py
```

## 🎯 Next Steps

1. **Generate synthetic data**: `make synth`
2. **Train initial model**: `make train` 
3. **Test pipeline**: `make serve` + `make frontend`
4. **Collect real data**: Use existing Cyton integration
5. **Fine-tune model**: Retrain with real data
6. **Deploy**: Update model weights in production

## 📚 References

- [BrainFlow Synthetic Board Documentation](https://brainflow.readthedocs.io/en/stable/Examples.html)
- [LaBraM Paper](https://arxiv.org/abs/2306.15062)
- [OpenBCI Cyton Board](https://docs.openbci.com/Ganglion%20Data%20Format/)
- [EEG Preprocessing Best Practices](https://mne.tools/stable/auto_tutorials/preprocessing/index.html)

## 🤝 Contributing

The synthetic data pipeline is designed to be extensible:

1. **Add new mental states** in `SyntheticEEGGenerator`
2. **Implement new augmentations** in `EEGAugmentation`  
3. **Extend API endpoints** for custom functionality
4. **Add new board types** in `UnifiedBoardManager`

---

**🎉 You now have a complete synthetic EEG pipeline!** 

This implementation provides everything you need to bootstrap your EEG/BCI development without requiring physical hardware. The synthetic data closely mimics real EEG characteristics, and the augmentation techniques help bridge the domain gap when you transition to real data.

Start with `make demo` and explore the full pipeline!
