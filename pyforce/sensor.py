import numpy as np
import struct
import socket
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import os
from typing import Optional, Tuple, List, Dict
from collections import deque
import threading
import logging
import time


class ForceSensor:
    """
    Force Sensor Data Collection and Visualization Class
    
    Features:
    - Connect to force sensor
    - Query and configure sensor parameters
    - Collect force and torque data
    - Save data to file
    - Generate visualization charts
    """
    
    def __init__(self, ip_addr: str = '192.168.0.108', port: int = 4008):
        """
        Initialize Force Sensor
        
        Args:
            ip_addr: Sensor IP address
            port: Sensor port
        """
        self.ip_addr = ip_addr
        self.port = port
        self.socket = None
        self.connected = False
        
        # Logger setup - use unique logger name to avoid conflicts
        logger_name = f"{self.__class__.__name__}_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # Prevent propagation to root logger
        
        # Bias for zero calibration
        self.bias = np.zeros(6, dtype=np.float32)
        
        # Streaming state
        self._streaming_active = threading.Event()
        self._streaming_thread = None
        self._streaming_lock = threading.Lock()
        self._latest_data = None
        self._latest_timestamp = None
        
        # Streaming statistics
        self._packets_received = 0
        self._packets_parsed = 0
        self._parse_errors = 0
        
        # Data storage
        self.data_fx = []
        self.data_fy = []
        self.data_fz = []
        self.data_mx = []
        self.data_my = []
        self.data_mz = []
        self.timestamps = []
        self.start_time = None
        
    def connect(self) -> bool:
        """
        Connect to sensor
        
        Returns:
            bool: Connection success status
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Disable Nagle's algorithm for lower latency
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Increase receive buffer size to avoid data accumulation
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
            
            self.socket.connect((self.ip_addr, self.port))
            self.connected = True
            self.logger.info(f"Successfully connected to sensor: {self.ip_addr}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from sensor
        
        Returns:
            bool: True if disconnection successful
        """
        try:
            # Stop streaming if active
            if self._streaming_active.is_set():
                self.stop_stream()
            
            if self.socket:
                self.socket.close()
                self.socket = None
            
            self.connected = False
            self.logger.info("Disconnected from force sensor")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
            return False
    
    def send_command(self, command: str, recv_size: int = 1000) -> Optional[bytearray]:
        """
        Send AT command to sensor
        
        Args:
            command: AT command string
            recv_size: Receive buffer size
            
        Returns:
            Received data, or None if failed
        """
        if not self.connected:
            print("Not connected to sensor")
            return None
        
        try:
            self.socket.send(command.encode())
            recv_data = bytearray(self.socket.recv(recv_size))
            return recv_data
        except Exception as e:
            print(f"Command send failed: {e}")
            return None
    
    def query_info(self) -> Dict[str, bytearray]:
        """
        Query sensor information
        
        Returns:
            Dictionary containing sensor information
        """
        info = {}
        
        # Query connection address
        result = self.send_command("AT+EIP=?\r\n")
        if result:
            print('Connection address:', result)
            info['address'] = result
        
        # Query decouple matrix
        result = self.send_command("AT+DCPM=?\r\n")
        if result:
            print('Decouple matrix:', result)
            info['decouple_matrix'] = result
        
        # Query compute unit
        result = self.send_command("AT+DCPCU=?\r\n")
        if result:
            print('Compute unit:', result)
            info['compute_unit'] = result
        
        return info
    
    def set_decouple_matrix(self, matrix: str) -> bool:
        """
        Set decouple matrix
        
        Args:
            matrix: Decouple matrix string
            
        Returns:
            bool: Success status
        """
        command = f"AT+DCPM={matrix}\r\n"
        result = self.send_command(command)
        if result:
            print('New decouple matrix:', result)
            return True
        return False
    
    def set_sample_rate(self, rate: int) -> bool:
        """
        Set sampling frequency
        
        Args:
            rate: Sampling frequency (Hz)
            
        Returns:
            bool: Success status
        """
        command = f"AT+SMPR={rate}\r\n"
        result = self.send_command(command)
        if result:
            print(f'Sample rate set to: {rate} Hz')
            return True
        return False
    
    def set_compute_unit(self, unit: str = "MVPV") -> bool:
        """
        Set matrix computation unit
        
        Args:
            unit: Compute unit
            
        Returns:
            bool: Success status
        """
        command = f"AT+DCPCU={unit}\r\n"
        result = self.send_command(command)
        if result:
            print(f'Compute unit set to: {unit}')
            return True
        return False
    
    def set_data_format(self, format_str: str = "(A01,A02,A03,A04,A05,A06);E;1;(WMA:1)") -> bool:
        """
        Set data upload format
        
        Args:
            format_str: Data format string
            
        Returns:
            bool: Success status
        """
        command = f"AT+SGDM={format_str}\r\n"
        result = self.send_command(command)
        if result:
            print('Data format set')
            return True
        return False
    
    def parse_data(self, data: bytes) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Parse sensor data packet
        
        Args:
            data: Raw data packet
            
        Returns:
            (fx, fy, fz, mx, my, mz) or None
        """
        try:
            # Check if data is long enough
            if len(data) < 30:
                self.logger.debug(f"Data packet too short: {len(data)} bytes, expected at least 30 bytes")
                return None
            
            fx = struct.unpack("f", data[6:10])[0]
            fy = struct.unpack('f', data[10:14])[0]
            fz = struct.unpack('f', data[14:18])[0]
            mx = struct.unpack('f', data[18:22])[0]
            my = struct.unpack('f', data[22:26])[0]
            mz = struct.unpack('f', data[26:30])[0]
            return (fx, fy, fz, mx, my, mz)
        except Exception as e:
            self.logger.debug(f"Data parsing failed: {e}, data length: {len(data)}")
            return None
    
    def is_connected(self) -> bool:
        """
        Check if sensor is connected
        
        Returns:
            bool: True if connected
        """
        return self.connected
    
    def read(self) -> Optional[np.ndarray]:
        """
        Read the most recent force/torque data from sensor
        
        Note: This method returns cached data from the streaming thread,
              ensuring real-time time-aligned measurements without blocking.
              You must call start_stream() before using this method.
        
        Returns:
            np.ndarray: 6-axis force/torque [fx, fy, fz, mx, my, mz] or None if read failed
        """
        if not self.is_connected():
            self.logger.warning("Force sensor not connected")
            return None
        
        if not self._streaming_active.is_set():
            self.logger.warning("Data stream not started. Call start_stream() first.")
            return None
        
        try:
            # Get the latest data from streaming cache (thread-safe)
            with self._streaming_lock:
                if self._latest_data is None:
                    return None
                force_data = self._latest_data.copy()
            
            return force_data
            
        except Exception as e:
            self.logger.error(f"Error reading from force sensor: {e}")
            return None
    
    def get(self) -> Optional[Dict[str, any]]:
        """
        Get the most recent force/torque data with timestamp
        
        Similar to read() but returns a dictionary with both force data and timestamp.
        This is compatible with the ATI sensor interface.
        
        Returns:
            Dict with 'ft' (np.ndarray) and 'timestamp' (float), or None if read failed
        """
        if not self.is_connected():
            self.logger.warning("Force sensor not connected")
            return None
        
        if not self._streaming_active.is_set():
            self.logger.warning("Data stream not started. Call start_stream() first.")
            return None
        
        try:
            # Get the latest data and timestamp from streaming cache (thread-safe)
            with self._streaming_lock:
                if self._latest_data is None:
                    return None
                force_data = self._latest_data.copy()
                timestamp = self._latest_timestamp
            
            return {
                'ft': force_data,
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Error reading from force sensor: {e}")
            return None
    
    def get_stream_stats(self) -> Dict[str, any]:
        """
        Get streaming statistics for debugging
        
        Returns:
            Dict with streaming statistics
        """
        return {
            'streaming_active': self._streaming_active.is_set(),
            'thread_alive': self._streaming_thread.is_alive() if self._streaming_thread else False,
            'packets_received': self._packets_received,
            'packets_parsed': self._packets_parsed,
            'parse_errors': self._parse_errors,
            'success_rate': self._packets_parsed / max(1, self._packets_received),
            'has_data': self._latest_data is not None,
            'last_timestamp': self._latest_timestamp
        }
    
    def zero(self, num_samples: int = 100) -> bool:
        """
        Zero/bias the force sensor by averaging multiple samples
        
        Args:
            num_samples: Number of samples to average for bias calculation
            
        Returns:
            bool: True if zeroing successful
        """
        if not self.is_connected():
            self.logger.warning("Force sensor not connected")
            return False
        
        try:
            # Temporarily save current bias
            old_bias = self.bias.copy()
            # Reset bias to zero for raw readings
            self.bias = np.zeros(6, dtype=np.float32)
            
            # Collect samples
            samples = []
            for i in range(num_samples):
                force = self.read()
                if force is not None:
                    samples.append(force)
                else:
                    self.logger.warning(f"Failed to read sample {i+1}/{num_samples}")
                time.sleep(0.01)  # Small delay between samples
            
            if len(samples) > 0:
                self.bias = np.mean(samples, axis=0)
                self.logger.info(f"Force sensor zeroed with {len(samples)} samples, bias: {self.bias}")
                return True
            else:
                self.logger.error("Failed to collect samples for zeroing")
                # Restore old bias
                self.bias = old_bias
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to zero force sensor: {e}")
            # Restore old bias
            self.bias = old_bias
            return False
    
    def clear_data(self):
        """Clear data buffer"""
        self.data_fx = []
        self.data_fy = []
        self.data_fz = []
        self.data_mx = []
        self.data_my = []
        self.data_mz = []
        self.timestamps = []
        self.start_time = None
    
    def start_stream(self) -> bool:
        """
        Start data stream with background thread for continuous data reception
        
        This starts a background thread that continuously receives data from the sensor,
        ensuring that read() always returns the most recent data.
        
        Returns:
            bool: Success status
        """
        if self._streaming_active.is_set():
            self.logger.warning("Data stream already started")
            return True
        
        # Clear socket receive buffer to avoid reading stale data
        if self.socket:
            old_timeout = self.socket.gettimeout()
            self.socket.settimeout(0.001)  # Very short timeout
            try:
                # Drain any pending data in the socket buffer
                while True:
                    data = self.socket.recv(4096)
                    if not data:
                        break
            except socket.timeout:
                pass  # Expected when buffer is empty
            except Exception as e:
                self.logger.debug(f"Error clearing buffer: {e}")
            finally:
                self.socket.settimeout(old_timeout)
        
        # Send command to start streaming
        result = self.send_command("AT+GSD\r\n")
        if result is None:
            self.logger.error("Failed to start data stream")
            return False
        
        self.start_time = datetime.now()
        self._streaming_active.set()
        
        # Start background thread for continuous data reception
        self._streaming_thread = threading.Thread(
            target=self._streaming_worker,
            daemon=True,
            name="ForceSensorStreaming"
        )
        self._streaming_thread.start()
        
        self.logger.info("Data stream started with background thread")
        return True
    
    def _streaming_worker(self):
        """
        Background worker thread that continuously receives data from sensor
        
        This ensures that read() always gets the most recent data without blocking.
        """
        self.logger.info("Streaming worker thread started")
        
        # Set socket timeout to a very low value for minimal latency
        if self.socket:
            self.socket.settimeout(0.01)  # 10ms timeout for fast response
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self._streaming_active.is_set():
            try:
                # Receive data packet with larger buffer to handle multiple packets
                data = self.socket.recv(4096)
                
                # Update receive counter
                self._packets_received += 1
                
                # Check if we received any data
                if not data:
                    self.logger.warning("Received empty data packet")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors, stopping stream")
                        break
                    continue
                
                # Try to parse the most recent data packet
                # If buffer contains multiple packets, we want the newest one
                result = self.parse_data(data)
                
                if result is not None:
                    consecutive_errors = 0  # Reset error counter on success
                    self._packets_parsed += 1
                    
                    fx, fy, fz, mx, my, mz = result
                    # Create numpy array and apply bias correction
                    force = np.array([fx, fy, fz, mx, my, mz], dtype=np.float32) - self.bias
                    timestamp = time.time()
                    
                    # Update latest data (thread-safe)
                    with self._streaming_lock:
                        self._latest_data = force
                        self._latest_timestamp = timestamp
                    
                    # Log every 100 packets for debugging
                    if self._packets_parsed % 100 == 0:
                        self.logger.debug(f"Received {self._packets_parsed} valid packets, {self._parse_errors} parse errors")
                else:
                    # Parse failed, but don't count as critical error
                    self._parse_errors += 1
                    if self._parse_errors % 50 == 0:
                        self.logger.warning(f"Parse errors: {self._parse_errors}/{self._packets_received} packets")
                
            except socket.timeout:
                # Timeout is normal, just continue
                continue
            except Exception as e:
                if self._streaming_active.is_set():
                    self.logger.error(f"Error in streaming worker: {e}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error("Too many consecutive errors, stopping stream")
                        break
        
        self.logger.info(f"Streaming worker thread stopped. Stats: {self._packets_parsed} parsed, {self._parse_errors} errors")
    
    def stop_stream(self) -> bool:
        """
        Stop data stream and background thread
        
        Returns:
            bool: Success status
        """
        if not self._streaming_active.is_set():
            self.logger.warning("Data stream not started")
            return True
        
        # Signal thread to stop
        self._streaming_active.clear()
        
        # Wait for thread to finish
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)
        
        # Send stop command to sensor
        try:
            self.socket.settimeout(2.0)
            result = self.send_command("AT+GSD=STOP\r\n")
            if result:
                self.logger.info("Data stream stopped")
                return True
        except socket.timeout:
            self.logger.warning("Stop command timeout")
        except Exception as e:
            self.logger.error(f"Error stopping data stream: {e}")
        finally:
            if self.socket:
                self.socket.settimeout(None)
        return False
    
    def collect_data(self, duration: Optional[float] = None, print_data: bool = True) -> bool:
        """
        Collect data
        
        Args:
            duration: Collection duration (seconds), None for manual stop (Ctrl+C)
            print_data: Whether to print real-time data
            
        Returns:
            bool: Collection success status
        """
        if not self.connected:
            print("Not connected to sensor")
            return False
        
        # Start data stream
        if not self.start_stream():
            return False
        
        print("Collecting data... Press Ctrl+C to stop")
        
        try:
            while True:
                # Check if duration reached
                if duration and self.start_time:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    if elapsed >= duration:
                        print(f"\nReached specified duration of {duration} seconds")
                        break
                
                # Receive data
                data = self.socket.recv(1000)
                result = self.parse_data(data)
                
                if result:
                    fx, fy, fz, mx, my, mz = result
                    
                    # Record timestamp
                    current_time = (datetime.now() - self.start_time).total_seconds()
                    self.timestamps.append(current_time)
                    
                    # Save data
                    self.data_fx.append(fx)
                    self.data_fy.append(fy)
                    self.data_fz.append(fz)
                    self.data_mx.append(mx)
                    self.data_my.append(my)
                    self.data_mz.append(mz)
                    
                    # Print data
                    if print_data:
                        F = np.array([fx, fy, fz, mx, my, mz])
                        print(np.around(F, 4))
        
        except KeyboardInterrupt:
            print("\nData collection stopped")
        
        # Stop data stream
        self.stop_stream()
        
        return len(self.timestamps) > 0
    
    def collect_data_with_realtime_plot(self, duration: Optional[float] = None, 
                                        window_size: int = 100, 
                                        update_interval: int = 50) -> bool:
        """
        Collect data with real-time visualization
        
        Args:
            duration: Collection duration (seconds), None for manual stop (Ctrl+C)
            window_size: Number of recent data points to display in the plot
            update_interval: Plot update interval in milliseconds
            
        Returns:
            bool: Collection success status
        """
        if not self.connected:
            print("Not connected to sensor")
            return False
        
        # Start data stream
        if not self.start_stream():
            return False
        
        print("Collecting data with real-time visualization...")
        print("Press Ctrl+C to stop")
        
        # Enable interactive mode
        plt.ion()
        
        # Create figure and subplots
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle('Real-time Force Sensor Data', fontsize=14, fontweight='bold')
        
        # Initialize line objects
        lines = []
        titles = ['Fx - Force X', 'Fy - Force Y', 'Fz - Force Z', 
                  'Mx - Torque X', 'My - Torque Y', 'Mz - Torque Z']
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        
        for i, (ax, title, color) in enumerate(zip(axes.flat, titles, colors)):
            line, = ax.plot([], [], color + '-', linewidth=1.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Force (N)' if i < 3 else 'Torque (Nm)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            lines.append(line)
        
        plt.tight_layout()
        
        # Data deques for efficient rolling window
        time_deque = deque(maxlen=window_size)
        data_deques = [deque(maxlen=window_size) for _ in range(6)]
        
        try:
            while True:
                # Check if duration reached
                if duration and self.start_time:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    if elapsed >= duration:
                        print(f"\nReached specified duration of {duration} seconds")
                        break
                
                # Receive data
                data = self.socket.recv(1000)
                result = self.parse_data(data)
                
                if result:
                    fx, fy, fz, mx, my, mz = result
                    
                    # Record timestamp
                    current_time = (datetime.now() - self.start_time).total_seconds()
                    self.timestamps.append(current_time)
                    
                    # Save data
                    self.data_fx.append(fx)
                    self.data_fy.append(fy)
                    self.data_fz.append(fz)
                    self.data_mx.append(mx)
                    self.data_my.append(my)
                    self.data_mz.append(mz)
                    
                    # Update deques for plotting
                    time_deque.append(current_time)
                    data_deques[0].append(fx)
                    data_deques[1].append(fy)
                    data_deques[2].append(fz)
                    data_deques[3].append(mx)
                    data_deques[4].append(my)
                    data_deques[5].append(mz)
                    
                    # Update plots
                    if len(time_deque) > 1:
                        times = list(time_deque)
                        for i, (line, ax, data_deque) in enumerate(zip(lines, axes.flat, data_deques)):
                            values = list(data_deque)
                            line.set_data(times, values)
                            
                            # Auto-scale axes
                            ax.relim()
                            ax.autoscale_view()
                        
                        # Redraw
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plt.pause(0.001)
        
        except KeyboardInterrupt:
            print("\nData collection stopped")
        
        finally:
            plt.ioff()
        
        # Stop data stream
        self.stop_stream()
        
        return len(self.timestamps) > 0
    
    def save_data(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save data to file
        
        Args:
            filename: File name, None for auto-generation
            
        Returns:
            Saved file path, or None if failed
        """
        if len(self.timestamps) == 0:
            print("No data to save")
            return None
        
        # Create data directory
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Generate filename
        if filename is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'data/force_sensor_data_{timestamp_str}.txt'
        
        try:
            with open(filename, 'w') as f:
                # Write header
                f.write("Time(s)\tFx(N)\tFy(N)\tFz(N)\tMx(Nm)\tMy(Nm)\tMz(Nm)\n")
                # Write data
                for i in range(len(self.timestamps)):
                    f.write(f"{self.timestamps[i]:.4f}\t{self.data_fx[i]:.4f}\t{self.data_fy[i]:.4f}\t{self.data_fz[i]:.4f}\t")
                    f.write(f"{self.data_mx[i]:.4f}\t{self.data_my[i]:.4f}\t{self.data_mz[i]:.4f}\n")
            
            print(f"\nData saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Failed to save data: {e}")
            return None
    
    def plot_data(self, save_charts: bool = True, show_charts: bool = True, 
                  chart_prefix: Optional[str] = None) -> List[str]:
        """
        Generate data visualization charts
        
        Args:
            save_charts: Whether to save charts
            show_charts: Whether to display charts
            chart_prefix: Chart filename prefix, None for auto-generation
            
        Returns:
            List of saved chart file paths
        """
        if len(self.timestamps) == 0:
            print("No data to plot")
            return []
        
        saved_files = []
        
        # Create charts directory
        if save_charts and not os.path.exists('charts'):
            os.makedirs('charts')
        
        # Generate filename prefix
        if chart_prefix is None:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_prefix = f'charts/force_sensor_{timestamp_str}'
        
        # Chart 1: 6 subplots
        fig1, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig1.suptitle('Force Sensor Data - Individual Channels', fontsize=16, fontweight='bold')
        
        # Plot force data
        axes[0, 0].plot(self.timestamps, self.data_fx, 'r-', linewidth=1.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Force (N)')
        axes[0, 0].set_title('Fx - Force in X direction')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.timestamps, self.data_fy, 'g-', linewidth=1.5)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Force (N)')
        axes[0, 1].set_title('Fy - Force in Y direction')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.timestamps, self.data_fz, 'b-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Force (N)')
        axes[1, 0].set_title('Fz - Force in Z direction')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot torque data
        axes[1, 1].plot(self.timestamps, self.data_mx, 'c-', linewidth=1.5)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Torque (Nm)')
        axes[1, 1].set_title('Mx - Torque around X axis')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[2, 0].plot(self.timestamps, self.data_my, 'm-', linewidth=1.5)
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Torque (Nm)')
        axes[2, 0].set_title('My - Torque around Y axis')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(self.timestamps, self.data_mz, 'y-', linewidth=1.5)
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Torque (Nm)')
        axes[2, 1].set_title('Mz - Torque around Z axis')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_charts:
            filename1 = f'{chart_prefix}_individual.png'
            plt.savefig(filename1, dpi=150, bbox_inches='tight')
            saved_files.append(filename1)
            print(f"Individual chart saved to: {filename1}")
        
        # Chart 2: Combined plot
        fig2, ax = plt.subplots(figsize=(16, 8))
        fig2.suptitle('Force Sensor Data - All Channels Combined', fontsize=16, fontweight='bold')
        
        ax.plot(self.timestamps, self.data_fx, 'r-', linewidth=1.5, label='Fx (N)')
        ax.plot(self.timestamps, self.data_fy, 'g-', linewidth=1.5, label='Fy (N)')
        ax.plot(self.timestamps, self.data_fz, 'b-', linewidth=1.5, label='Fz (N)')
        ax.plot(self.timestamps, self.data_mx, 'c-', linewidth=1.5, label='Mx (Nm)')
        ax.plot(self.timestamps, self.data_my, 'm-', linewidth=1.5, label='My (Nm)')
        ax.plot(self.timestamps, self.data_mz, 'y-', linewidth=1.5, label='Mz (Nm)')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Force (N) / Torque (Nm)', fontsize=12)
        ax.set_title('Forces and Torques in All Directions', fontsize=14)
        ax.legend(loc='upper right', fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_charts:
            filename2 = f'{chart_prefix}_combined.png'
            plt.savefig(filename2, dpi=150, bbox_inches='tight')
            saved_files.append(filename2)
            print(f"Combined chart saved to: {filename2}")
        
        print(f"Total {len(self.timestamps)} data points collected")
        
        # Display charts
        if show_charts:
            try:
                plt.show(block=False)
                plt.pause(0.1)
                print("\nChart window opened, program will continue after closing the window")
                plt.show(block=True)
            except Exception as e:
                print(f"\nCannot display chart window (may not have GUI): {e}")
                if save_charts:
                    print("Chart has been saved to file successfully")
        
        return saved_files
    
    def get_latest_data(self) -> Optional[Dict[str, float]]:
        """
        Get the latest data point
        
        Returns:
            Dictionary containing latest data, or None if no data
        """
        if len(self.timestamps) == 0:
            return None
        
        return {
            'time': self.timestamps[-1],
            'fx': self.data_fx[-1],
            'fy': self.data_fy[-1],
            'fz': self.data_fz[-1],
            'mx': self.data_mx[-1],
            'my': self.data_my[-1],
            'mz': self.data_mz[-1]
        }
    
    def get_all_data(self) -> Dict[str, List[float]]:
        """
        Get all collected data
        
        Returns:
            Dictionary containing all data
        """
        return {
            'timestamps': self.timestamps.copy(),
            'fx': self.data_fx.copy(),
            'fy': self.data_fy.copy(),
            'fz': self.data_fz.copy(),
            'mx': self.data_mx.copy(),
            'my': self.data_my.copy(),
            'mz': self.data_mz.copy()
        }


# Usage example (for backward compatibility)
def main():
    """Main function - Demonstrates how to use ForceSensor class"""
    # Create sensor instance
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    # Connect to sensor
    if not sensor.connect():
        return
    
    # Query sensor information
    sensor.query_info()
    
    # Optional: Set sensor parameters (only need to set once)
    # decouple_matrix = "(0.272516,-62.753809,0.493088,63.105530,-0.077489,-0.048031);" \
    #                   "(-0.175049,-36.146383,-0.203728,-36.289948,0.161349,72.595527);" \
    #                   "(-109.853665,0.273471,-107.370475,-0.409348,-108.782132,-0.637754);" \
    #                   "(-1.720345,0.002968,1.668830,0.010925,0.011441,-0.010379);" \
    #                   "(-1.013312,0.002875,-0.977509,-0.006355,1.941401,0.008722);" \
    #                   "(-0.001640,1.228712,0.012064,1.184833,-0.000614,1.271352)\r\n"
    # sensor.set_decouple_matrix(decouple_matrix)
    # sensor.set_sample_rate(100)
    # sensor.set_compute_unit("MVPV")
    
    try:
        # Collect data (press Ctrl+C to stop)
        sensor.collect_data()
        
        # Save data
        sensor.save_data()
        
        # Generate charts
        sensor.plot_data()
        
    finally:
        # Disconnect
        sensor.disconnect()


if __name__ == "__main__":
    main()
