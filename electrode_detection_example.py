#!/usr/bin/env python3
"""
Example script demonstrating electrode detection functionality for OpenBCI Cyton board.

This script shows how to use the electrode detection module to:
1. Connect to the Cyton board
2. Measure impedance for all channels
3. Monitor live signal quality
4. Display results in a user-friendly format

Usage:
    python electrode_detection_example.py [serial_port]
    
Example:
    python electrode_detection_example.py /dev/cu.usbserial-DM01N8KH
"""

import sys
import time
import numpy as np
from electrode_detection import ElectrodeDetector, create_board_connection
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_impedance_results(results):
    """Print impedance measurement results in a formatted table."""
    print_header("IMPEDANCE MEASUREMENT RESULTS")
    
    print(f"{'Channel':<8} {'Impedance (kΩ)':<15} {'Quality':<12} {'Status':<15}")
    print("-" * 60)
    
    for result in results:
        channel = result.get('channel', 'N/A')
        impedance = result.get('impedance_kohm', 'N/A')
        quality = result.get('quality', 'N/A')
        status = result.get('status', 'N/A')
        
        # Color coding for quality
        if quality == 'good':
            quality_display = f"✅ {quality}"
        elif quality == 'moderate':
            quality_display = f"⚠️  {quality}"
        elif quality == 'poor':
            quality_display = f"❌ {quality}"
        elif quality == 'disconnected':
            quality_display = f"🔴 {quality}"
        else:
            quality_display = f"❓ {quality}"
        
        print(f"{channel:<8} {impedance:<15} {quality_display:<12} {status:<15}")
    
    print("\nLegend:")
    print("✅ Good: < 750 kΩ - Excellent electrode contact")
    print("⚠️  Moderate: 750-1500 kΩ - Acceptable contact")
    print("❌ Poor: 1500-5000 kΩ - Poor contact, may need adjustment")
    print("🔴 Disconnected: > 5000 kΩ - No electrode contact")


def print_live_quality_results(quality_results):
    """Print live signal quality results."""
    print_header("LIVE SIGNAL QUALITY")
    
    print(f"{'Channel':<8} {'Status':<15} {'Quality':<12} {'Std (µV)':<12} {'Railed':<8}")
    print("-" * 60)
    
    for ch_name, result in quality_results.items():
        channel = ch_name.replace('ch', '')
        status = result.get('status', 'N/A')
        quality = result.get('quality', 'N/A')
        std = result.get('std', 0) * 1000  # Convert to µV
        is_railed = result.get('is_railed', False)
        
        # Status indicators
        if status == 'connected':
            status_display = f"✅ {status}"
        elif status == 'poor_contact':
            status_display = f"⚠️  {status}"
        elif status == 'disconnected':
            status_display = f"🔴 {status}"
        else:
            status_display = f"❓ {status}"
        
        railed_display = "🔴 YES" if is_railed else "✅ NO"
        
        print(f"{channel:<8} {status_display:<15} {quality:<12} {std:<12.1f} {railed_display:<8}")


def demonstrate_impedance_testing(board):
    """Demonstrate impedance testing functionality."""
    print_header("IMPEDANCE TESTING DEMONSTRATION")
    
    detector = ElectrodeDetector(board)
    
    print("This will test electrode impedance on all channels...")
    print("Note: Impedance testing injects a 31.5 Hz signal and may add noise.")
    print("Only one channel is tested at a time to minimize interference.\n")
    
    input("Press Enter to start impedance testing...")
    
    # Test all channels
    results = detector.measure_all_channels(samples=250)  # 1 second per channel
    
    print_impedance_results(results)
    
    return results


def demonstrate_live_monitoring(board):
    """Demonstrate live signal quality monitoring."""
    print_header("LIVE SIGNAL MONITORING DEMONSTRATION")
    
    detector = ElectrodeDetector(board)
    
    print("This will monitor live signal quality for 10 seconds...")
    print("Watch for changes in electrode connection status.\n")
    
    input("Press Enter to start live monitoring...")
    
    # Start streaming
    board.start_stream()
    
    try:
        for i in range(10):  # Monitor for 10 seconds
            time.sleep(1)
            
            # Get recent data (1 second window)
            data = board.get_board_data(250)  # 1 second at 250Hz
            eeg_channels = BoardShim.get_eeg_channels(board.get_board_id())
            eeg_data = data[eeg_channels]
            
            # Analyze live quality
            quality_results = detector.detect_live_quality(eeg_data)
            
            print(f"\n--- Second {i+1}/10 ---")
            print_live_quality_results(quality_results)
            
    finally:
        board.stop_stream()


def get_board_info(board):
    """Get and display board information."""
    print_header("BOARD INFORMATION")
    
    detector = ElectrodeDetector(board)
    status = detector.get_board_status()
    
    print(f"Board ID: {status.get('board_id', 'N/A')}")
    print(f"Board Name: {status.get('board_name', 'N/A')}")
    print(f"Sampling Rate: {status.get('sampling_rate_hz', 'N/A')} Hz")
    print(f"EEG Channels: {status.get('eeg_channels', 'N/A')}")
    print(f"Scale Factor: {status.get('scale_factor_uv_per_count', 'N/A')} µV/count")
    print(f"Battery Level: {status.get('battery_level', 'N/A')} V")
    print(f"Connected: {status.get('connected', False)}")
    
    print("\nImpedance Thresholds:")
    thresholds = status.get('impedance_thresholds_kohm', {})
    for quality, threshold in thresholds.items():
        print(f"  {quality}: {threshold} kΩ")


def main():
    """Main demonstration function."""
    # Get serial port from command line or use default
    if len(sys.argv) > 1:
        serial_port = sys.argv[1]
    else:
        serial_port = "/dev/cu.usbserial-DM01N8KH"  # Default Cyton port
    
    print_header("OPENBCI CYTON ELECTRODE DETECTION DEMO")
    print(f"Attempting to connect to Cyton board on: {serial_port}")
    print("Make sure your Cyton board is connected and powered on.")
    
    # Connect to board
    board = create_board_connection(serial_port)
    if board is None:
        print(f"❌ Failed to connect to board on {serial_port}")
        print("Please check:")
        print("1. Board is connected and powered on")
        print("2. Correct serial port (try: ls /dev/cu.usbserial-*)")
        print("3. No other applications are using the board")
        return
    
    print(f"✅ Successfully connected to Cyton board!")
    
    try:
        # Get board information
        get_board_info(board)
        
        # Demonstrate impedance testing
        impedance_results = demonstrate_impedance_testing(board)
        
        # Demonstrate live monitoring
        demonstrate_live_monitoring(board)
        
        # Summary
        print_header("SUMMARY")
        print("Electrode detection demonstration completed!")
        print("\nKey findings:")
        
        good_channels = [r for r in impedance_results if r.get('quality') == 'good']
        moderate_channels = [r for r in impedance_results if r.get('quality') == 'moderate']
        poor_channels = [r for r in impedance_results if r.get('quality') in ['poor', 'disconnected']]
        
        print(f"✅ Good connections: {len(good_channels)} channels")
        print(f"⚠️  Moderate connections: {len(moderate_channels)} channels")
        print(f"❌ Poor/disconnected: {len(poor_channels)} channels")
        
        if poor_channels:
            print("\nRecommendations:")
            for channel in poor_channels:
                ch_num = channel.get('channel', 'N/A')
                print(f"  - Channel {ch_num}: Check electrode placement and skin preparation")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    finally:
        # Clean up
        try:
            board.release_session()
            print("\n✅ Board connection closed.")
        except:
            pass


if __name__ == "__main__":
    main()
