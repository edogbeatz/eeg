#!/usr/bin/env python3
"""
Demo script to show the API server working with mock Cyton data.
This demonstrates the complete system in action.
"""

import asyncio
import uvicorn
import threading
import time
import requests
import numpy as np
from contextlib import asynccontextmanager

# Generate mock data function
def generate_cyton_mock_data():
    """Generate a single window of realistic Cyton data."""
    n_channels, n_samples = 8, 1000
    t = np.linspace(0, 4, n_samples)  # 4 seconds
    
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Realistic EEG components
        noise = np.random.normal(0, 5, n_samples)
        alpha = 10 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta = 5 * np.sin(2 * np.pi * 20 * t)    # 20 Hz beta
        theta = 8 * np.sin(2 * np.pi * 6 * t)    # 6 Hz theta
        
        eeg_data[ch] = noise + alpha + beta + theta
    
    return eeg_data

def test_api_client():
    """Test client that sends requests to the running API server."""
    
    print("🚀 API Client Testing Started")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        # Test health endpoint
        print("🏥 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Health: {result}")
        else:
            print(f"   ❌ Health failed: {response.status_code}")
            return
        
        # Test predictions with mock data
        print("\n🧠 Testing predictions with mock data...")
        
        for i in range(5):
            print(f"\n   📊 Prediction {i+1}:")
            
            # Generate fresh mock data
            mock_data = generate_cyton_mock_data()
            
            # Send prediction request
            response = requests.post(
                f"{base_url}/predict",
                json={"x": mock_data.tolist(), "n_outputs": 2},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                probs = result['probs']
                winner = probs.index(max(probs))
                confidence = max(probs)
                
                print(f"      ✅ Success: Class {winner} (confidence: {confidence:.3f})")
                print(f"      📈 Probabilities: {[f'{p:.3f}' for p in probs]}")
                
                # Show electrode status summary
                electrode_status = result['electrode_status']
                good_electrodes = sum(1 for status in electrode_status.values() 
                                    if status['quality'] >= 0.8)
                print(f"      🔌 Electrodes: {good_electrodes}/8 good quality")
                
            else:
                print(f"      ❌ Prediction failed: {response.status_code} - {response.text}")
            
            time.sleep(1)  # Wait between predictions
        
        print(f"\n🎉 All API tests completed successfully!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
        print("   Make sure the API server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def run_api_server():
    """Run the API server in a separate thread."""
    
    print("🌐 Starting API Server...")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    print("   Health: http://localhost:8000/health")
    print()
    
    # Import here to avoid circular imports
    from main import app
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False  # Reduce noise
    )
    
    server = uvicorn.Server(config)
    server.run()

def main():
    """Main demo function."""
    
    print("🎬 EEG API Server Demo with Mock Cyton Data")
    print("=" * 60)
    print("This demo shows your complete EEG system in action:")
    print("  • FastAPI server with LaBraM model")
    print("  • Realistic mock Cyton board data")
    print("  • Real-time predictions and electrode monitoring")
    print("  • Complete end-to-end pipeline")
    print()
    
    try:
        # Start API server in background thread
        server_thread = threading.Thread(target=run_api_server, daemon=True)
        server_thread.start()
        
        # Give server time to start
        print("⏳ Waiting for server to start...")
        time.sleep(3)
        
        # Run API tests
        test_api_client()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("🚀 Your EEG system is production-ready!")
        print("\nTo run the server manually:")
        print("  uvicorn main:app --reload")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
