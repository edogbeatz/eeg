# OpenBCI Cyton Electrode Detection Implementation

This implementation provides comprehensive electrode detection functionality for the OpenBCI Cyton board using impedance testing and live signal monitoring as specified in the OpenBCI documentation.

## Overview

The electrode detection system implements two main approaches:

1. **Impedance-Based Detection** (Recommended): Uses OpenBCI's lead-off test to inject a 31.5 Hz current and measure electrode impedance
2. **Live Signal Monitoring**: Monitors signal amplitude and detects "railing" to identify disconnected electrodes

## Features

- ✅ Impedance measurement using OpenBCI's `z` command protocol
- ✅ Live signal quality monitoring with railing detection
- ✅ FastAPI endpoints for web integration
- ✅ Comprehensive error handling and logging
- ✅ Proper impedance classification based on OpenBCI thresholds
- ✅ Support for all 8 Cyton channels
- ✅ Battery level monitoring
- ✅ Board status and connection management

## Files Added

- `electrode_detection.py` - Core electrode detection module
- `electrode_detection_example.py` - Demonstration script
- `ELECTRODE_DETECTION_README.md` - This documentation
- Updated `main.py` - Added FastAPI endpoints
- Updated `requirements.txt` - Added brainflow dependency

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your OpenBCI Cyton board is connected and powered on.

## Usage

### 1. Basic Impedance Testing

```python
from electrode_detection import ElectrodeDetector, create_board_connection

# Connect to board
board = create_board_connection("/dev/cu.usbserial-DM01N8KH")
detector = ElectrodeDetector(board)

# Test single channel
result = detector.measure_impedance(channel=1, samples=250)
print(f"Channel 1 impedance: {result['impedance_kohm']} kΩ")
print(f"Quality: {result['quality']}")

# Test all channels
results = detector.measure_all_channels()
```

### 2. Live Signal Monitoring

```python
# Start streaming
board.start_stream()

# Get data and analyze quality
data = board.get_board_data(250)  # 1 second of data
eeg_channels = BoardShim.get_eeg_channels(board.get_board_id())
eeg_data = data[eeg_channels]

# Analyze live quality
quality_results = detector.detect_live_quality(eeg_data)
print(quality_results)
```

### 3. FastAPI Endpoints

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

Available endpoints:

- `POST /connect-board` - Connect to Cyton board
- `GET /board-status` - Get board information
- `GET /battery` - Get battery level
- `POST /electrode-impedance` - Measure electrode impedance
- `POST /live-quality` - Analyze live signal quality
- `POST /disconnect-board` - Disconnect from board

### 4. Example API Usage

```python
import requests

# Connect to board
response = requests.post("http://localhost:8000/connect-board", 
                        json={"serial_port": "/dev/cu.usbserial-DM01N8KH"})

# Measure impedance for all channels
response = requests.post("http://localhost:8000/electrode-impedance",
                        json={"channels": [1, 2, 3, 4, 5, 6, 7, 8], "samples": 250})
results = response.json()

# Get live quality
response = requests.post("http://localhost:8000/live-quality",
                        json={"x": eeg_data.tolist()})
quality = response.json()
```

## Impedance Thresholds

Based on OpenBCI documentation:

| Impedance Range | Quality | Status | Interpretation |
|----------------|---------|--------|----------------|
| < 750 kΩ | Good | Connected | Excellent electrode contact |
| 750-1500 kΩ | Moderate | Poor Contact | Acceptable, may need adjustment |
| 1500-5000 kΩ | Poor | Poor Contact | Poor contact, needs attention |
| > 5000 kΩ | Disconnected | Disconnected | No electrode contact |

## Technical Details

### Impedance Measurement Formula

The implementation uses OpenBCI's impedance calculation formula:

```
impedance = (√2 × std_volts) / I_drive - R_series
```

Where:
- `std_volts`: Standard deviation of measured voltage
- `I_drive`: 6 nA (Cyton drive current)
- `R_series`: 2.2 kΩ (built-in series resistor)

### Signal Processing

- **Sampling Rate**: 250 Hz (Cyton default)
- **Test Frequency**: 31.5 Hz (OpenBCI lead-off test)
- **Scale Factor**: 0.02235 µV/count (gain x24)
- **Measurement Duration**: 1 second per channel (250 samples)

### Railing Detection

Live signal monitoring detects:
- **Flat signals**: Standard deviation < 1.0 µV (disconnected)
- **Railed signals**: Amplitude > 80% of full scale (saturated)
- **Noisy signals**: Standard deviation > 100 µV (poor contact)

## Best Practices

1. **Sequential Testing**: Only test one channel at a time to avoid interference
2. **Pre-Recording**: Run impedance tests before starting EEG recording sessions
3. **Skin Preparation**: Ensure clean, abraded skin for optimal contact
4. **Electrode Placement**: Follow 10-20 system guidelines
5. **Motion Avoidance**: Avoid impedance testing during subject movement

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check serial port: `ls /dev/cu.usbserial-*`
   - Ensure board is powered on
   - Close other applications using the board

2. **High Impedance Values**
   - Check electrode placement
   - Clean and abrade skin
   - Apply conductive gel
   - Check electrode connections

3. **Inconsistent Results**
   - Ensure subject is still during testing
   - Wait for signal to stabilize
   - Check for loose connections

### Error Codes

- `400`: Bad request (invalid channel, no connection)
- `500`: Server error (board communication issues)

## Example Output

```
IMPEDANCE MEASUREMENT RESULTS
Channel   Impedance (kΩ) Quality      Status         
------------------------------------------------------------
1         245.3          ✅ good      connected      
2         890.2          ⚠️  moderate  poor_contact   
3         1250.5         ⚠️  moderate  poor_contact   
4         156.7          ✅ good      connected      
5         5200.1         🔴 disconnected disconnected
6         445.8          ✅ good      connected      
7         2100.3         ❌ poor      poor_contact   
8         678.9          ✅ good      connected      
```

## Integration with Existing Code

The electrode detection functionality integrates seamlessly with the existing EEG processing pipeline:

1. **Pre-Processing**: Run impedance tests before starting data collection
2. **Live Monitoring**: Use live quality detection during streaming
3. **Quality Control**: Filter out poor-quality channels from analysis
4. **User Feedback**: Display electrode status in real-time UI

## References

- [OpenBCI Documentation](https://docs.openbci.com/)
- [BrainFlow Documentation](https://brainflow.readthedocs.io/)
- [ADS1299 Datasheet](https://www.ti.com/lit/ds/symlink/ads1299.pdf)

## License

This implementation follows the same license as the parent project.
