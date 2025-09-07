# EEG Training Interface

A comprehensive interface for starting, stopping, and monitoring EEG model training through both API endpoints and the web frontend.

## Features

### Backend API Endpoints

#### Training Control
- `POST /training/start` - Start model training
- `POST /training/stop` - Stop current training
- `GET /training/status` - Get real-time training status
- `GET /training/history` - Get training metrics history
- `GET /training/config` - Get default training configuration
- `POST /training/config/validate` - Validate training parameters

#### Training Configuration
```json
{
  "n_epochs": 15,
  "n_samples_per_class": 1500,
  "val_split": 0.2,
  "freeze_backbone": true,
  "learning_rate": 0.001,
  "batch_size": 32,
  "weight_decay": 0.0001,
  "save_dir": "./trained_models"
}
```

### Frontend Interface

The training interface is integrated into the main EEG dashboard and provides:

#### Real-time Monitoring
- **Training Progress**: Visual progress bar with epoch count
- **Live Metrics**: Train/validation loss and accuracy
- **Time Tracking**: Elapsed time and estimated completion time
- **Status Indicators**: Training state badges and error reporting

#### Configuration Management
- **Interactive Config Dialog**: Easy-to-use form for training parameters
- **Parameter Validation**: Client-side and server-side validation
- **Presets**: Quick access to common training configurations

#### Training Control
- **One-click Start/Stop**: Simple buttons to control training
- **Background Training**: Non-blocking training execution
- **Progress Persistence**: Training continues even if frontend is closed

## Usage

### Starting Training via API

```bash
# Start training with default configuration
curl -X POST http://localhost:8000/training/start

# Start training with custom configuration
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "n_epochs": 20,
    "n_samples_per_class": 2000,
    "freeze_backbone": false,
    "learning_rate": 0.0005
  }'
```

### Monitoring Training Progress

```bash
# Get current training status
curl http://localhost:8000/training/status

# Get training history
curl http://localhost:8000/training/history
```

### Stopping Training

```bash
# Stop current training
curl -X POST http://localhost:8000/training/stop
```

### Using the Web Interface

1. **Open the EEG Dashboard** at `http://localhost:3000`
2. **Locate the Training Interface** section
3. **Configure Training**: Click the "Config" button to adjust parameters
4. **Start Training**: Click "Start Training" to begin
5. **Monitor Progress**: Watch real-time metrics and progress bar
6. **Stop if Needed**: Click "Stop" to halt training early

## Training Process

### Data Generation
- Synthetic EEG data is generated for "relaxed" vs "anxious" states
- Configurable number of samples per class
- Automatic train/validation split

### Model Architecture
- Uses LaBraM (Large Brain Model) as backbone
- Option to freeze backbone for faster training
- Custom classification head for binary classification

### Training Features
- **Automatic Model Saving**: Best model saved based on validation accuracy
- **Training History**: Complete metrics history saved to JSON
- **Early Stopping**: Manual stop capability between epochs
- **Progress Tracking**: Real-time monitoring of all metrics

### Output Files
- `best_synthetic_model.pth` - Best model checkpoint
- `training_history.json` - Complete training metrics
- `training_curves.png` - Training visualization (if matplotlib available)

## Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `n_epochs` | Number of training epochs | 15 | 1-100 |
| `n_samples_per_class` | Samples per class to generate | 1500 | 100-5000 |
| `val_split` | Validation data fraction | 0.2 | 0.1-0.5 |
| `freeze_backbone` | Freeze LaBraM backbone | true | true/false |
| `learning_rate` | Optimizer learning rate | 0.001 | 0.0001-0.01 |
| `batch_size` | Training batch size | 32 | 16-128 |
| `weight_decay` | L2 regularization | 0.0001 | 0-0.001 |
| `save_dir` | Model save directory | "./trained_models" | Any valid path |

## Status Response Format

```json
{
  "is_training": true,
  "current_epoch": 5,
  "total_epochs": 15,
  "train_loss": 0.4521,
  "val_loss": 0.4832,
  "train_acc": 78.5,
  "val_acc": 76.2,
  "best_val_acc": 77.1,
  "elapsed_time": 125.3,
  "elapsed_time_formatted": "2.1m",
  "eta": 250.6,
  "eta_formatted": "4.2m",
  "config": { ... },
  "history": {
    "train_losses": [0.6, 0.5, 0.45, ...],
    "val_losses": [0.65, 0.52, 0.48, ...],
    "train_accs": [65, 72, 78, ...],
    "val_accs": [62, 70, 76, ...]
  }
}
```

## Error Handling

- **Validation Errors**: Invalid configuration parameters are caught and reported
- **Training Errors**: Runtime errors during training are captured and displayed
- **Connection Errors**: Frontend gracefully handles API connectivity issues
- **Resource Errors**: Memory and GPU availability are checked before training

## Requirements

### Backend
- Python 3.8+
- PyTorch
- braindecode>=1.1.0 (for LaBraM model)
- FastAPI
- All dependencies from `requirements.txt`

### Frontend
- Next.js 14+
- React 18+
- Tailwind CSS
- shadcn/ui components

## Troubleshooting

### Common Issues

1. **"LaBraM model not available"**
   - Install braindecode>=1.1.0: `pip install braindecode>=1.1.0`

2. **"Training already in progress"**
   - Stop current training first: `curl -X POST http://localhost:8000/training/stop`

3. **Frontend not updating**
   - Check browser console for errors
   - Verify API server is running on port 8000

4. **Training fails immediately**
   - Check available memory and GPU resources
   - Verify configuration parameters are valid

### Performance Tips

- **Faster Training**: Enable `freeze_backbone=true` to train only the classification head
- **Better Accuracy**: Disable backbone freezing but expect longer training times
- **Memory Management**: Reduce `batch_size` if running out of memory
- **Quick Testing**: Use smaller `n_samples_per_class` for faster iterations

## Integration

The training interface is designed to integrate seamlessly with the existing EEG pipeline:

1. **Data Flow**: Uses the same synthetic data generation as other endpoints
2. **Model Compatibility**: Trained models work with existing prediction endpoints
3. **State Management**: Training state is independent of other system components
4. **Resource Sharing**: Safely coexists with other API operations

## Future Enhancements

- **Real Data Training**: Support for training on real EEG data
- **Distributed Training**: Multi-GPU and multi-node training support
- **Hyperparameter Tuning**: Automated parameter optimization
- **Model Comparison**: Side-by-side comparison of different models
- **Export Options**: Model export to different formats (ONNX, TensorRT)
