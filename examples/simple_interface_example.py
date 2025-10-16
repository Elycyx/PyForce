"""
Simple Interface Example
Demonstrates how to use the basic interface: connect(), disconnect(), read(), get(), zero()
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path to import pyforce
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyforce import ForceSensor


def main():
    """Main function - demonstrates basic interface usage"""
    # Create sensor instance
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    # 1. Connect to sensor
    print("Connecting to sensor...")
    if not sensor.connect():
        print("Connection failed!")
        return
    
    print("Connection successful!")
    
    try:
        # 2. Check connection status
        if sensor.is_connected():
            print("Sensor is connected")
        
        # 3. Start data stream (required before reading)
        print("\nStarting data stream with background thread...")
        sensor.start_stream()
        time.sleep(0.5)  # Wait for stream to stabilize
        
        # 4. Zero calibration
        print("\nPerforming zero calibration...")
        if sensor.zero(num_samples=50):
            print(f"Zero calibration successful! Bias: {sensor.bias}")
        else:
            print("Zero calibration failed!")
        
        # 5. Read data (two methods available)
        print("\n" + "="*60)
        print("Method 1: Using read() - returns numpy array")
        print("="*60)
        count = 0
        while count < 10:
            force_data = sensor.read()
            
            if force_data is not None:
                print(f"\nReading #{count+1}: {np.around(force_data, 4)}")
                print(f"  Force: Fx={force_data[0]:.3f}, Fy={force_data[1]:.3f}, Fz={force_data[2]:.3f}")
                print(f"  Torque: Mx={force_data[3]:.3f}, My={force_data[4]:.3f}, Mz={force_data[5]:.3f}")
            else:
                print(f"Reading #{count+1}: Failed")
            
            count += 1
            time.sleep(0.1)  # Control reading frequency
        
        print("\n" + "="*60)
        print("Method 2: Using get() - returns dict with timestamp (ATI-compatible)")
        print("="*60)
        count = 0
        while count < 10:
            data = sensor.get()
            
            if data is not None:
                print(f"\nReading #{count+1}:")
                print(f"  Data: {np.around(data['ft'], 4)}")
                print(f"  Timestamp: {data['timestamp']:.6f}")
            else:
                print(f"Reading #{count+1}: Failed")
            
            count += 1
            time.sleep(0.1)
        
        # Stop data stream
        print("\nStopping data stream...")
        sensor.stop_stream()
        
    except KeyboardInterrupt:
        print("\nUser interrupted")
        sensor.stop_stream()
    
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 6. Disconnect
        print("\nDisconnecting...")
        if sensor.disconnect():
            print("Disconnection successful!")
        else:
            print("Disconnection failed!")


if __name__ == "__main__":
    main()

