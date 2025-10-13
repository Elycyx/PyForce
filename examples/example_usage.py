"""
Example Usage of ForceSensor Class

This file demonstrates different ways to use the ForceSensor class
for data collection, visualization, and integration into other projects.
"""

from pyforce import ForceSensor
import time


def example_basic_usage():
    """Basic usage example - collect data and save"""
    print("=== Example 1: Basic Usage ===")
    
    # Create sensor instance
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    # Connect to sensor
    if not sensor.connect():
        print("Failed to connect to sensor")
        return
    
    # Query sensor information
    sensor.query_info()
    
    try:
        # Collect data (press Ctrl+C to stop)
        sensor.collect_data()
        
        # Save data to file
        sensor.save_data()
        
        # Generate and save charts
        sensor.plot_data()
        
    finally:
        # Disconnect
        sensor.disconnect()


def example_timed_collection():
    """Collect data for a specific duration"""
    print("\n=== Example 2: Timed Collection ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    try:
        # Collect data for 10 seconds
        print("Collecting data for 10 seconds...")
        sensor.collect_data(duration=10, print_data=True)
        
        # Save data with custom filename
        sensor.save_data(filename='data/test_10sec.txt')
        
        # Plot without showing window (useful for headless systems)
        sensor.plot_data(save_charts=True, show_charts=False, 
                        chart_prefix='charts/test_10sec')
        
    finally:
        sensor.disconnect()


def example_custom_settings():
    """Configure sensor settings before data collection"""
    print("\n=== Example 3: Custom Settings ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    # Set custom sample rate
    sensor.set_sample_rate(100)  # 100 Hz
    
    # Set compute unit
    sensor.set_compute_unit("MVPV")
    
    # Set decouple matrix (example - use your calibrated values)
    # decouple_matrix = "(0.272516,-62.753809,0.493088,63.105530,-0.077489,-0.048031);" \
    #                   "(-0.175049,-36.146383,-0.203728,-36.289948,0.161349,72.595527);" \
    #                   "(-109.853665,0.273471,-107.370475,-0.409348,-108.782132,-0.637754);" \
    #                   "(-1.720345,0.002968,1.668830,0.010925,0.011441,-0.010379);" \
    #                   "(-1.013312,0.002875,-0.977509,-0.006355,1.941401,0.008722);" \
    #                   "(-0.001640,1.228712,0.012064,1.184833,-0.000614,1.271352)\r\n"
    # sensor.set_decouple_matrix(decouple_matrix)
    
    try:
        sensor.collect_data(duration=5)
        sensor.save_data()
        sensor.plot_data()
    finally:
        sensor.disconnect()


def example_realtime_processing():
    """Process data in real-time without storing all"""
    print("\n=== Example 4: Real-time Processing ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    # Start data stream
    sensor.start_stream()
    
    try:
        print("Processing real-time data for 5 seconds...")
        start = time.time()
        
        while time.time() - start < 5:
            # Receive and parse data
            data = sensor.socket.recv(1000)
            result = sensor.parse_data(data)
            
            if result:
                fx, fy, fz, mx, my, mz = result
                
                # Custom processing - e.g., check if force exceeds threshold
                if abs(fz) > 10:  # 10N threshold on Z-axis
                    print(f"Warning: High force detected! Fz = {fz:.2f} N")
                
                # Calculate total force magnitude
                import numpy as np
                total_force = np.sqrt(fx**2 + fy**2 + fz**2)
                print(f"Total force: {total_force:.2f} N")
                
    except KeyboardInterrupt:
        print("\nReal-time processing stopped")
    finally:
        sensor.stop_stream()
        sensor.disconnect()


def example_multiple_sessions():
    """Collect multiple sessions and compare"""
    print("\n=== Example 5: Multiple Sessions ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    try:
        # Session 1
        print("\nSession 1: Collecting data for 3 seconds...")
        sensor.collect_data(duration=3, print_data=False)
        sensor.save_data(filename='data/session1.txt')
        
        # Clear data for next session
        sensor.clear_data()
        
        # Session 2
        print("\nSession 2: Collecting data for 3 seconds...")
        sensor.collect_data(duration=3, print_data=False)
        sensor.save_data(filename='data/session2.txt')
        
        # Plot both sessions (would need to load and compare separately)
        sensor.plot_data(chart_prefix='charts/session2')
        
    finally:
        sensor.disconnect()


def example_data_access():
    """Access and use collected data programmatically"""
    print("\n=== Example 6: Data Access ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    try:
        # Collect some data
        sensor.collect_data(duration=3, print_data=False)
        
        # Get latest data point
        latest = sensor.get_latest_data()
        if latest:
            print(f"\nLatest data point:")
            print(f"  Time: {latest['time']:.3f} s")
            print(f"  Fx: {latest['fx']:.3f} N")
            print(f"  Fy: {latest['fy']:.3f} N")
            print(f"  Fz: {latest['fz']:.3f} N")
        
        # Get all data
        all_data = sensor.get_all_data()
        print(f"\nTotal data points collected: {len(all_data['timestamps'])}")
        
        # Calculate statistics
        import numpy as np
        print(f"\nForce statistics (Fz):")
        print(f"  Mean: {np.mean(all_data['fz']):.3f} N")
        print(f"  Std: {np.std(all_data['fz']):.3f} N")
        print(f"  Max: {np.max(all_data['fz']):.3f} N")
        print(f"  Min: {np.min(all_data['fz']):.3f} N")
        
    finally:
        sensor.disconnect()


def example_integration_callback():
    """Simulate integration with callback function"""
    print("\n=== Example 7: Callback Integration ===")
    
    def data_callback(fx, fy, fz, mx, my, mz, timestamp):
        """Custom callback to process each data point"""
        # Example: Send data to another system, database, etc.
        print(f"[{timestamp:.2f}s] Fz={fz:.2f}N", end='\r')
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.start_stream()
    start_time = time.time()
    
    try:
        print("Running with callback for 5 seconds...")
        while time.time() - start_time < 5:
            data = sensor.socket.recv(1000)
            result = sensor.parse_data(data)
            
            if result:
                fx, fy, fz, mx, my, mz = result
                timestamp = time.time() - start_time
                data_callback(fx, fy, fz, mx, my, mz, timestamp)
                
    except KeyboardInterrupt:
        print("\nCallback processing stopped")
    finally:
        sensor.stop_stream()
        sensor.disconnect()


if __name__ == "__main__":
    # Run different examples
    # Uncomment the example you want to run
    
    # example_basic_usage()
    # example_timed_collection()
    # example_custom_settings()
    # example_realtime_processing()
    # example_multiple_sessions()
    # example_data_access()
    # example_integration_callback()
    
    # Or run basic usage as default
    example_basic_usage()

