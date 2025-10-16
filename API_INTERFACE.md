# Force Sensor API Interface

## Overview

This document describes the basic API interface for the force sensor, suitable for integrating the force sensor into other projects.

## Basic Interface

### 1. Initialization

```python
from pyforce import ForceSensor

sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
```

**Parameters:**
- `ip_addr`: IP address of the sensor (default: '192.168.0.108')
- `port`: Port number of the sensor (default: 4008)

---

### 2. connect() - Connect to Sensor

```python
def connect(self) -> bool:
    """
    Connect to force sensor
    
    Returns:
        bool: True if connection successful, False otherwise
    """
```

**Example:**
```python
if sensor.connect():
    print("Connection successful")
else:
    print("Connection failed")
```

---

### 3. disconnect() - Disconnect from Sensor

```python
def disconnect(self) -> bool:
    """
    Disconnect from force sensor
    
    Returns:
        bool: True if disconnection successful, False otherwise
    """
```

**Example:**
```python
if sensor.disconnect():
    print("Disconnection successful")
```

---

### 4. is_connected() - Check Connection Status

```python
def is_connected(self) -> bool:
    """
    Check if sensor is connected
    
    Returns:
        bool: True if connected, False otherwise
    """
```

**Example:**
```python
if sensor.is_connected():
    print("Sensor is connected")
```

---

### 5. read() - Read Force/Torque Data

```python
def read(self) -> Optional[np.ndarray]:
    """
    Read the most recent force/torque data from sensor
    
    Note: This method returns cached data from the streaming thread,
          ensuring real-time time-aligned measurements without blocking.
          You must call start_stream() before using this method.
    
    Returns:
        np.ndarray: 6-axis force/torque [fx, fy, fz, mx, my, mz]
                   or None if read failed
    """
```

**Return Value:**
- Returns a numpy array containing 6 float values
- `[fx, fy, fz, mx, my, mz]`
  - fx, fy, fz: Forces in X, Y, Z axes (N)
  - mx, my, mz: Torques around X, Y, Z axes (Nm)
- Data is bias-corrected (bias is subtracted)

**Real-Time Performance:**
- Uses a **background thread** that continuously receives data from the sensor
- `read()` returns the **most recent cached data** without blocking
- This ensures **true real-time performance** with guaranteed time alignment
- No buffer flushing needed - you always get the freshest data

**Example:**
```python
# Read most recent data (always time-aligned, non-blocking)
force_data = sensor.read()
if force_data is not None:
    print(f"Force: Fx={force_data[0]:.3f}, Fy={force_data[1]:.3f}, Fz={force_data[2]:.3f}")
    print(f"Torque: Mx={force_data[3]:.3f}, My={force_data[4]:.3f}, Mz={force_data[5]:.3f}")
else:
    print("Read failed")
```

**Note:** You must call `start_stream()` before using `read()`.

---

### 5b. get() - Get Data with Timestamp

```python
def get(self) -> Optional[Dict[str, any]]:
    """
    Get the most recent force/torque data with timestamp
    
    Similar to read() but returns a dictionary with both force data and timestamp.
    This is compatible with the ATI sensor interface.
    
    Returns:
        Dict with 'ft' (np.ndarray) and 'timestamp' (float), or None if read failed
    """
```

**Example:**
```python
data = sensor.get()
if data is not None:
    print(f"Force/Torque: {data['ft']}")
    print(f"Timestamp: {data['timestamp']}")
```

---

### 6. zero() - Zero Calibration

```python
def zero(self, num_samples: int = 100) -> bool:
    """
    Zero/bias the force sensor by averaging multiple samples
    
    Args:
        num_samples: Number of samples to average for bias calculation (default: 100)
        
    Returns:
        bool: True if zeroing successful, False otherwise
    """
```

**Description:**
- Collects the specified number of data samples
- Calculates the average as the zero offset (bias)
- Subsequent `read()` calls will automatically subtract this offset

**Example:**
```python
# Perform zero calibration with 50 samples
if sensor.zero(num_samples=50):
    print(f"Zero calibration successful, bias: {sensor.bias}")
else:
    print("Zero calibration failed")
```

---

## Complete Usage Example

```python
from pyforce import ForceSensor
import time

# 1. Create sensor instance
sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)

# 2. Connect to sensor
if not sensor.connect():
    print("Connection failed")
    exit(1)

try:
    # 3. Start data stream (important!)
    sensor.start_stream()
    time.sleep(0.5)  # Wait for stream to stabilize
    
    # 4. Zero calibration
    if sensor.zero(num_samples=50):
        print("Zero calibration successful")
    
    # 5. Read data in loop
    for i in range(100):
        force_data = sensor.read()
        if force_data is not None:
            print(f"Reading #{i}: {force_data}")
        time.sleep(0.01)  # Control reading frequency
    
    # 6. Stop data stream
    sensor.stop_stream()
    
finally:
    # 7. Disconnect
    sensor.disconnect()
```

---

## ATI-Compatible Interface Example

For compatibility with ATI sensor code:

```python
from pyforce import ForceSensor

# Create and connect
sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
sensor.connect()
sensor.start_stream()

# ATI-style usage
while True:
    data = sensor.get()
    if data is not None:
        print(f"Force/Torque: {data['ft']}")
        print(f"Timestamp: {data['timestamp']}")
    time.sleep(0.1)
```

---

## Key Features

### Background Streaming Thread

The sensor automatically manages a background thread when you call `start_stream()`:

1. **Continuous Data Reception**: The thread continuously receives data from the sensor
2. **Thread-Safe Caching**: Latest data is stored in a thread-safe manner
3. **Non-Blocking Reads**: `read()` and `get()` return immediately with the most recent data
4. **Automatic Cleanup**: Thread stops automatically when you call `stop_stream()` or `disconnect()`

### Time Alignment Guarantee

- The background thread ensures you always get the **most recent** data
- No stale data from socket buffers
- True real-time performance without manual buffer flushing
- Perfect for control loops and real-time applications

---

## Additional Useful Attributes

### bias - Zero Offset

```python
sensor.bias  # numpy array containing 6 floats
```

Get or set the current zero offset.

### logger - Logger

```python
sensor.logger.setLevel(logging.DEBUG)  # Set log level
```

Control log output level.

---

## Important Notes

1. **Must start data stream first**: You must call `start_stream()` before calling `read()` or `get()`
2. **Zero calibration timing**: It's recommended to perform zero calibration after connecting and starting the stream
3. **Exception handling**: Use try-finally to ensure proper sensor disconnection
4. **Thread management**: The background thread is automatically managed - don't worry about it
5. **Time alignment**: Always guaranteed - the background thread ensures you get the freshest data
6. **Non-blocking**: `read()` and `get()` never block, making them perfect for real-time control loops

---

## Advanced Features

In addition to the basic interface above, the sensor also provides these advanced features:

- `collect_data()`: Batch data collection
- `save_data()`: Save data to file
- `plot_data()`: Plot data charts
- `query_info()`: Query sensor information
- `set_sample_rate()`: Set sampling rate
- `set_decouple_matrix()`: Set decouple matrix

For detailed information, please refer to README.md and README_CN.md.

