"""
Real-time Visualization Example for PyForce

This example demonstrates how to use the real-time visualization feature
to display force sensor data as it's being collected.
"""

from pyforce import ForceSensor


def example_realtime_basic():
    """Basic real-time visualization example"""
    print("=== Example 1: Basic Real-time Visualization ===")
    
    # Create sensor instance
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    # Connect to sensor
    if not sensor.connect():
        print("Failed to connect to sensor")
        return
    
    # Query sensor information
    sensor.query_info()
    
    try:
        # Collect data with real-time visualization
        # The plot will show the last 100 data points
        sensor.collect_data_with_realtime_plot(window_size=100)
        
        # After collection, save the data
        sensor.save_data()
        
        # Optionally, generate static charts as well
        sensor.plot_data()
        
    finally:
        # Disconnect
        sensor.disconnect()


def example_realtime_timed():
    """Real-time visualization with specified duration"""
    print("\n=== Example 2: Timed Real-time Collection ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    try:
        # Collect data for 30 seconds with real-time visualization
        # Show the last 150 data points in the plot
        sensor.collect_data_with_realtime_plot(duration=30, window_size=150)
        
        # Save data
        sensor.save_data()
        
        # Generate final static charts
        sensor.plot_data(show_charts=False)
        
    finally:
        sensor.disconnect()


def example_realtime_large_window():
    """Real-time visualization with large window"""
    print("\n=== Example 3: Large Window Real-time Visualization ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    try:
        # Collect data with a larger window to see more history
        # This will show the last 500 data points
        sensor.collect_data_with_realtime_plot(duration=60, window_size=500)
        
        # Save data with custom filename
        sensor.save_data(filename='data/realtime_test.txt')
        
        # Generate charts
        sensor.plot_data(chart_prefix='charts/realtime_test')
        
    finally:
        sensor.disconnect()


def example_multiple_sessions_with_realtime():
    """Collect multiple sessions with real-time visualization"""
    print("\n=== Example 4: Multiple Sessions with Real-time ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    try:
        # Session 1
        print("\nSession 1: Collecting for 10 seconds...")
        sensor.collect_data_with_realtime_plot(duration=10, window_size=100)
        sensor.save_data(filename='data/session1_realtime.txt')
        
        # Clear data for next session
        sensor.clear_data()
        
        print("\nWaiting 2 seconds before next session...")
        import time
        time.sleep(2)
        
        # Session 2
        print("\nSession 2: Collecting for 10 seconds...")
        sensor.collect_data_with_realtime_plot(duration=10, window_size=100)
        sensor.save_data(filename='data/session2_realtime.txt')
        
    finally:
        sensor.disconnect()


def example_comparison_normal_vs_realtime():
    """Compare normal collection vs real-time visualization"""
    print("\n=== Example 5: Comparison - Normal vs Real-time ===")
    
    sensor = ForceSensor(ip_addr='192.168.0.108', port=4008)
    
    if not sensor.connect():
        return
    
    sensor.query_info()
    
    try:
        # First, collect without visualization
        print("\nCollecting 5 seconds without visualization...")
        sensor.collect_data(duration=5, print_data=False)
        sensor.save_data(filename='data/without_viz.txt')
        
        # Clear data
        sensor.clear_data()
        
        import time
        time.sleep(2)
        
        # Then, collect with real-time visualization
        print("\nCollecting 5 seconds WITH real-time visualization...")
        sensor.collect_data_with_realtime_plot(duration=5, window_size=100)
        sensor.save_data(filename='data/with_realtime_viz.txt')
        
        print("\nBoth datasets saved. You can compare the files.")
        
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    # Run different examples
    # Uncomment the example you want to run
    
    example_realtime_basic()
    # example_realtime_timed()
    # example_realtime_large_window()
    # example_multiple_sessions_with_realtime()
    # example_comparison_normal_vs_realtime()

