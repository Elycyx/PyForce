# PyForce
![Windows](https://img.shields.io/badge/OS-Windows-0078D6?logo=windows&logoColor=white)
![Ubuntu](https://img.shields.io/badge/OS-Ubuntu-E95420?logo=ubuntu&logoColor=white)

A modular Python package for collecting, visualizing, and saving force sensor data. This package is designed to be used with the Sunrise (宇立) force sensors.

[中文文档](README_CN.md)

## Features

- **Easy Connection**: Simple API to connect to force sensors via TCP/IP
- **Data Collection**: Real-time data collection with configurable duration
- **Configuration**: Set sensor parameters (sample rate, decouple matrix, compute unit)
- **Data Storage**: Save collected data to text files
- **Visualization**: Generate individual and combined charts
- **Modular Design**: Easy integration into other projects

## Installation


```bash
git clone https://github.com/Elycyx/PyForce.git
cd PyForce
pip install -e .
```


## Quick Import

```python
from pyforce import ForceSensor
```

## Quick Start

### Basic Usage

```python
from pyforce import ForceSensor

# Create sensor instance
sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)

# Connect to sensor
if sensor.connect():
    # Query sensor info
    sensor.query_info()
    
    # Collect data (press Ctrl+C to stop)
    sensor.collect_data()
    
    # Save data
    sensor.save_data()
    
    # Generate charts
    sensor.plot_data()
    
    # Disconnect
    sensor.disconnect()
```

### Timed Data Collection

```python
# Collect data for a specific duration (e.g., 10 seconds)
sensor.collect_data(duration=10, print_data=True)
```

### Configure Sensor Settings

```python
# Set sample rate
sensor.set_sample_rate(100)  # 100 Hz

# Set compute unit
sensor.set_compute_unit("MVPV")

# Set decouple matrix (use your calibrated values)
decouple_matrix = "(0.272516,-62.753809,...)...\r\n"
sensor.set_decouple_matrix(decouple_matrix)
```

### Access Data Programmatically

```python
# Get latest data point
latest = sensor.get_latest_data()
print(f"Fx: {latest['fx']} N, Fz: {latest['fz']} N")

# Get all collected data
all_data = sensor.get_all_data()
print(f"Total points: {len(all_data['timestamps'])}")
```

### Custom Data Processing

```python
import numpy as np

# Collect data
sensor.collect_data(duration=5, print_data=False)

# Get all data
data = sensor.get_all_data()

# Calculate statistics
print(f"Mean Fz: {np.mean(data['fz']):.3f} N")
print(f"Max Fz: {np.max(data['fz']):.3f} N")
```

## API Reference

### ForceSensor Class

#### Constructor

```python
ForceSensor(ip_addr='192.168.0.108', port=4008)
```

- `ip_addr`: Sensor IP address
- `port`: Sensor port

#### Methods

**Connection Management**
- `connect()` → bool: Connect to sensor
- `disconnect()`: Disconnect from sensor

**Sensor Configuration**
- `query_info()` → Dict: Query sensor information
- `set_sample_rate(rate)` → bool: Set sampling frequency (Hz)
- `set_compute_unit(unit)` → bool: Set computation unit
- `set_decouple_matrix(matrix)` → bool: Set decouple matrix
- `set_data_format(format_str)` → bool: Set data upload format

**Data Collection**
- `collect_data(duration=None, print_data=True)` → bool: Collect sensor data
  - `duration`: Collection duration in seconds (None for manual stop)
  - `print_data`: Whether to print real-time data
- `start_stream()` → bool: Start data stream
- `stop_stream()` → bool: Stop data stream
- `clear_data()`: Clear data buffer

**Data Access**
- `get_latest_data()` → Dict: Get latest data point
- `get_all_data()` → Dict: Get all collected data
- `parse_data(data)` → Tuple: Parse raw sensor data packet

**Data Storage & Visualization**
- `save_data(filename=None)` → str: Save data to file
- `plot_data(save_charts=True, show_charts=True, chart_prefix=None)` → List: Generate charts

## Data Format

### Collected Data Structure

Each data point contains:
- `time`: Timestamp (seconds)
- `fx`: Force in X direction (N)
- `fy`: Force in Y direction (N)
- `fz`: Force in Z direction (N)
- `mx`: Torque around X axis (Nm)
- `my`: Torque around Y axis (Nm)
- `mz`: Torque around Z axis (Nm)

### File Output

Data is saved in tab-separated format:

```
Time(s)    Fx(N)    Fy(N)    Fz(N)    Mx(Nm)    My(Nm)    Mz(Nm)
0.0000     1.234    2.345    3.456    0.123     0.234     0.345
...
```

## Examples

See `examples/example_usage.py` for detailed examples:

1. **Basic Usage**: Simple data collection and visualization
2. **Timed Collection**: Collect data for specific duration
3. **Custom Settings**: Configure sensor parameters
4. **Real-time Processing**: Process data without storing all
5. **Multiple Sessions**: Collect and compare multiple datasets
6. **Data Access**: Access and analyze collected data
7. **Callback Integration**: Use callbacks for data processing

Run examples:

```bash
python examples/example_usage.py
```

## Troubleshooting

### Connection Issues

- Check IP address and port are correct
- Ensure sensor is powered on and connected to network
- Verify network connectivity with `ping`

### Data Collection Problems

- If no data received, check sensor configuration
- Verify data format settings match sensor output
- Check firewall settings

### Visualization Issues

- If charts don't display, you may not have a GUI (use `show_charts=False`)
- Charts are saved to `charts/` directory even if display fails
- Ensure matplotlib backend is compatible with your system

## File Structure

```
PyForce/
├── pyforce/              # Main package
│   ├── __init__.py      # Package initialization
│   └── sensor.py        # ForceSensor class
├── examples/            # Usage examples
│   └── example_usage.py
├── setup.py            # Setup configuration
├── pyproject.toml      # Project metadata
├── requirements.txt    # Python dependencies
├── README.md          # This file
├── .gitignore         # Git ignore rules
├── data/              # Data files (auto-created, gitignored)
└── charts/            # Chart images (auto-created, gitignored)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


