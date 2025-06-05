"""
Analytics Panel for Smart Traffic Management System
Provides real-time analytics, graphs, heatmaps and reports
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
import seaborn as sns
from matplotlib.animation import FuncAnimation
import logging
from typing import List, Dict, Any

# UI Configuration
UI_CONFIG = {
    'window_title': 'Smart Traffic Management System',
    'window_size': (1400, 900),
    'min_window_size': (1200, 700),
    'theme': 'modern',
    'update_interval': 500,  # Increased from 100ms to 500ms
    'graph_update_interval': 2000,  # Update graphs every 2 seconds
    'stats_update_interval': 1000,  # Update stats every second
    'video_size': (320, 240),
    'colors': {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'success': '#F18F01',
        'danger': '#C73E1D',
        'background': '#F5F5F5',
        'card_bg': '#FFFFFF',
        'text_primary': '#333333',
        'text_secondary': '#666666'
    }
}

# Signal configuration
SIGNAL_CONFIG = {
    'names': ['Signal A', 'Signal B', 'Signal C', 'Signal D'],
    'colors': {
        'red': '#FF4444',
        'yellow': '#FFAA00', 
        'green': '#44FF44'
    },
    'default_timing': {
        'red': 30,
        'yellow': 3,
        'green': 27
    }
}

class LiveStatsPanel(ttk.Frame):
    """Panel for displaying live statistics"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.stats_labels = {}
        self.running = False
        self.last_update = 0
        self.update_interval = UI_CONFIG['stats_update_interval'] / 1000.0  # Convert to seconds
        self.cached_data = {}
        self.cache_timeout = 5.0  # Cache timeout in seconds
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup statistics UI"""
        # Title
        title_label = ttk.Label(self, text="Live Traffic Statistics", 
                              font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Stats container
        stats_frame = ttk.Frame(self)
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create stats labels
        self.create_stat_label(stats_frame, 'total_vehicles', "Total Vehicles: 0")
        self.create_stat_label(stats_frame, 'avg_efficiency', "Average Efficiency: 0%")
        self.create_stat_label(stats_frame, 'peak_time', "Peak Traffic Time: N/A")
        self.create_stat_label(stats_frame, 'peak_count', "Peak Vehicle Count: 0")
        
        # Signal-specific stats
        for i in range(4):
            signal_frame = ttk.LabelFrame(stats_frame, text=f"Signal {chr(65+i)}")
            signal_frame.pack(fill=tk.X, pady=5)
            
            self.create_stat_label(signal_frame, f'signal_{i}_vehicles', "Vehicles: 0")
            self.create_stat_label(signal_frame, f'signal_{i}_weight', "Traffic Weight: 0.0")
            self.create_stat_label(signal_frame, f'signal_{i}_state', "State: RED")
            self.create_stat_label(signal_frame, f'signal_{i}_efficiency', "Efficiency: 0%")
    
    def create_stat_label(self, parent, key, initial_text):
        """Create a statistics label"""
        label = ttk.Label(parent, text=initial_text)
        label.pack(anchor=tk.W, padx=5, pady=2)
        self.stats_labels[key] = label
    
    def start_updates(self):
        """Start updating statistics"""
        self.running = True
        self.update_stats()
    
    def stop_updates(self):
        """Stop updating statistics"""
        self.running = False
    
    def update_stats(self):
        """Update statistics display"""
        if not self.running:
            return
            
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
            
        try:
            # Check cache first
            if (current_time - self.cached_data.get('timestamp', 0)) < self.cache_timeout:
                analytics = self.cached_data.get('data')
            else:
                # Get fresh analytics data
                analytics = self.traffic_manager.get_analytics_data()
                self.cached_data = {
                    'timestamp': current_time,
                    'data': analytics
                }
            
            if not analytics:
                return
                
            # Update global stats
            self.stats_labels['total_vehicles'].config(
                text=f"Total Vehicles: {analytics.get('total_vehicles', 0)}")
            
            # Update efficiency
            efficiency = analytics.get('avg_efficiency', 0)
            self.stats_labels['avg_efficiency'].config(
                text=f"Average Efficiency: {efficiency:.1f}%")
            
            # Update peak time if available
            peak_time = analytics.get('peak_time')
            if peak_time:
                peak_time_str = peak_time.strftime("%H:%M:%S")
            else:
                peak_time_str = "N/A"
            self.stats_labels['peak_time'].config(
                text=f"Peak Traffic Time: {peak_time_str}")
            
            # Update signal-specific stats
            for signal_data in analytics.get('signals', []):
                signal_id = signal_data.get('signal_id')
                if signal_id is not None:
                    self._update_signal_stats(signal_id, signal_data)
            
            self.last_update = current_time
            
            # Schedule next update
            if self.running:
                self.after(int(self.update_interval * 1000), self.update_stats)
                
        except Exception as e:
            logging.error(f"Error updating statistics: {str(e)}")
            if self.running:
                self.after(5000, self.update_stats)
    
    def _update_signal_stats(self, signal_id, signal_data):
        """Update statistics for a specific signal"""
        try:
            # Update vehicle count
            self.stats_labels[f'signal_{signal_id}_vehicles'].config(
                text=f"Vehicles: {signal_data.get('total_vehicles', 0)}")
            
            # Update traffic weight
            self.stats_labels[f'signal_{signal_id}_weight'].config(
                text=f"Traffic Weight: {signal_data.get('avg_weight', 0.0):.1f}")
            
            # Update efficiency
            self.stats_labels[f'signal_{signal_id}_efficiency'].config(
                text=f"Efficiency: {signal_data.get('efficiency', 0):.1f}%")
            
            # Get current signal state
            signal_state = self.traffic_manager.get_signal(signal_id)
            if signal_state:
                state = signal_state.get('state', 'RED')
                remaining = signal_state.get('remaining_time', 0)
                self.stats_labels[f'signal_{signal_id}_state'].config(
                    text=f"State: {state} ({remaining}s)")
        except Exception as e:
            logging.error(f"Error updating signal {signal_id} stats: {str(e)}")
    
    def __del__(self):
        """Cleanup resources"""
        self.running = False
        self.cached_data = None


class GraphsPanel(ttk.Frame):
    """Panel for displaying real-time graphs"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.figures = {}
        self.canvases = {}
        self.data_history = {'timestamps': [], 'vehicle_counts': [[] for _ in range(4)], 
                           'efficiencies': [[] for _ in range(4)]}
        self.max_history = 30  # Reduced from 50 to 30 data points
        self.running = False
        self.last_update = 0
        self.update_interval = UI_CONFIG['graph_update_interval'] / 1000.0  # Convert to seconds
        self.cached_data = {}
        self.cache_timeout = 5.0
        
        self.setup_ui()
        self.start_graph_updates()
    
    def setup_ui(self):
        """Setup graphs panel UI"""
        # Create graphs
        self.create_vehicle_count_graph()
        self.create_efficiency_graph()
        self.create_traffic_weight_graph()
        self.create_timing_graph()
    
    def create_vehicle_count_graph(self):
        """Create vehicle count graph"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Vehicle Count Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vehicles')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.figures['vehicle_count'] = fig
        self.canvases['vehicle_count'] = canvas
    
    def create_efficiency_graph(self):
        """Create efficiency graph"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Signal Efficiency Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Efficiency (%)')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.figures['efficiency'] = fig
        self.canvases['efficiency'] = canvas
    
    def create_traffic_weight_graph(self):
        """Create traffic weight graph"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Traffic Weight Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Traffic Weight')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.figures['traffic_weight'] = fig
        self.canvases['traffic_weight'] = canvas
    
    def create_timing_graph(self):
        """Create signal timing graph"""
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Signal Timing Distribution')
        ax.set_xlabel('Signal')
        ax.set_ylabel('Time (seconds)')
        ax.grid(True)
        
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.figures['timing'] = fig
        self.canvases['timing'] = canvas
    
    def start_graph_updates(self):
        """Start updating graphs"""
        self.running = True
        self.update_graphs()
    
    def stop_graph_updates(self):
        """Stop updating graphs"""
        self.running = False
    
    def update_graphs(self):
        """Update all graphs with current data"""
        if not self.running:
            return
            
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            if self.running:
                self.after(int((self.update_interval - (current_time - self.last_update)) * 1000), self.update_graphs)
            return
            
        try:
            # Check cache first
            if (current_time - self.cached_data.get('timestamp', 0)) < self.cache_timeout:
                analytics = self.cached_data.get('data')
            else:
                # Get fresh analytics data
                analytics = self.traffic_manager.get_analytics_data()
                self.cached_data = {
                    'timestamp': current_time,
                    'data': analytics
                }
            
            if not analytics:
                if self.running:
                    self.after(int(self.update_interval * 1000), self.update_graphs)
                return
            
            # Update data history
            self.data_history['timestamps'].append(datetime.now())
            if len(self.data_history['timestamps']) > self.max_history:
                self.data_history['timestamps'].pop(0)
            
            # Get data for all signals
            for i in range(4):
                # Get signal data from cache if available
                signal_data = analytics.get('signals', [])[i] if i < len(analytics.get('signals', [])) else None
                if not signal_data:
                    continue
                
                # Update vehicle counts
                vehicle_count = signal_data.get('total_vehicles', 0)
                self.data_history['vehicle_counts'][i].append(vehicle_count)
                if len(self.data_history['vehicle_counts'][i]) > self.max_history:
                    self.data_history['vehicle_counts'][i].pop(0)
                
                # Update efficiencies
                efficiency = signal_data.get('efficiency', 0)
                self.data_history['efficiencies'][i].append(efficiency)
                if len(self.data_history['efficiencies'][i]) > self.max_history:
                    self.data_history['efficiencies'][i].pop(0)
            
            # Only update visible graphs based on current tab
            current_tab = self.winfo_viewable()
            if current_tab:
                self.update_visible_graphs()
            
            self.last_update = current_time
            
            # Schedule next update
            if self.running:
                self.after(int(self.update_interval * 1000), self.update_graphs)
                
        except Exception as e:
            logging.error(f"Error updating graphs: {str(e)}")
            if self.running:
                self.after(5000, self.update_graphs)
    
    def update_visible_graphs(self):
        """Update only the graphs that are currently visible"""
        try:
            # Update vehicle count graph if visible
            if self.canvases['vehicle_count'].winfo_viewable():
                self.update_vehicle_count_graph()
            
            # Update efficiency graph if visible
            if self.canvases['efficiency'].winfo_viewable():
                self.update_efficiency_graph()
            
            # Update traffic weight graph if visible
            if self.canvases['traffic_weight'].winfo_viewable():
                self.update_traffic_weight_graph()
            
            # Update timing graph if visible
            if self.canvases['timing'].winfo_viewable():
                self.update_timing_graph()
                
        except Exception as e:
            logging.error(f"Error updating visible graphs: {str(e)}")
    
    def update_vehicle_count_graph(self):
        """Update vehicle count graph"""
        try:
            fig = self.figures['vehicle_count']
            ax = fig.axes[0]
            ax.clear()
            
            # Plot only visible data points to reduce rendering load
            visible_points = min(self.max_history, len(self.data_history['timestamps']))
            if visible_points == 0:
                return
                
            timestamps = self.data_history['timestamps'][-visible_points:]
            
            # Plot data for each signal with simplified line style
            for i in range(4):
                counts = self.data_history['vehicle_counts'][i][-visible_points:]
                ax.plot(timestamps, counts, label=f'Signal {chr(65+i)}', 
                       linewidth=1, marker=None)  # Removed markers for better performance
            
            ax.set_title('Vehicle Count Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Vehicles')
            ax.grid(False)  # Disabled grid for better performance
            ax.legend(loc='upper right', frameon=False)  # Simplified legend
            
            # Format x-axis with fewer ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))  # Limit to 5 ticks
            plt.setp(ax.get_xticklabels(), rotation=45)
            fig.tight_layout()
            
            # Use faster canvas draw
            self.canvases['vehicle_count'].draw_idle()
            
        except Exception as e:
            logging.error(f"Error updating vehicle count graph: {str(e)}")
    
    def update_efficiency_graph(self):
        """Update efficiency graph"""
        try:
            fig = self.figures['efficiency']
            ax = fig.axes[0]
            ax.clear()
            
            # Plot only visible data points
            visible_points = min(self.max_history, len(self.data_history['timestamps']))
            if visible_points == 0:
                return
                
            timestamps = self.data_history['timestamps'][-visible_points:]
            
            # Plot data for each signal with simplified style
            for i in range(4):
                efficiencies = self.data_history['efficiencies'][i][-visible_points:]
                ax.plot(timestamps, efficiencies, label=f'Signal {chr(65+i)}',
                       linewidth=1, marker=None)
            
            ax.set_title('Signal Efficiency Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Efficiency (%)')
            ax.grid(False)
            ax.legend(loc='upper right', frameon=False)
            
            # Format x-axis with fewer ticks
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            plt.setp(ax.get_xticklabels(), rotation=45)
            fig.tight_layout()
            
            # Use faster canvas draw
            self.canvases['efficiency'].draw_idle()
            
        except Exception as e:
            logging.error(f"Error updating efficiency graph: {str(e)}")
    
    def update_traffic_weight_graph(self):
        """Update traffic weight graph"""
        try:
            if not hasattr(self, 'cached_weights') or time.time() - self.last_weight_update > 2.0:
                # Get current traffic weights
                weights = []
                for i in range(4):
                    traffic_data = self.traffic_manager.get_traffic_data(i)
                    weights.append(traffic_data.get('weight', 0))
                self.cached_weights = weights
                self.last_weight_update = time.time()
            
            fig = self.figures['traffic_weight']
            ax = fig.axes[0]
            ax.clear()
            
            # Create simplified bar chart
            signals = [f'Signal {chr(65+i)}' for i in range(4)]
            ax.bar(signals, self.cached_weights, width=0.6)
            
            ax.set_title('Current Traffic Weight by Signal')
            ax.set_xlabel('Signal')
            ax.set_ylabel('Traffic Weight')
            ax.grid(False)
            
            # Format with minimal decorations
            ax.tick_params(axis='both', which='both', length=0)
            fig.tight_layout()
            
            # Use faster canvas draw
            self.canvases['traffic_weight'].draw_idle()
            
        except Exception as e:
            logging.error(f"Error updating traffic weight graph: {str(e)}")
    
    def update_timing_graph(self):
        """Update timing graph"""
        try:
            if not hasattr(self, 'cached_timing') or time.time() - self.last_timing_update > 2.0:
                # Get timing data for each signal
                green_times = []
                yellow_times = []
                red_times = []
                
                for i in range(4):
                    signal_data = self.traffic_manager.get_signal(i)
                    state = signal_data.get('state', 'RED')
                    remaining = signal_data.get('remaining_time', 0)
                    
                    if state == 'GREEN':
                        green_times.append(remaining)
                        yellow_times.append(0)
                        red_times.append(0)
                    elif state == 'YELLOW':
                        green_times.append(0)
                        yellow_times.append(remaining)
                        red_times.append(0)
                    else:  # RED
                        green_times.append(0)
                        yellow_times.append(0)
                        red_times.append(remaining)
                
                self.cached_timing = (green_times, yellow_times, red_times)
                self.last_timing_update = time.time()
            
            fig = self.figures['timing']
            ax = fig.axes[0]
            ax.clear()
            
            # Create simplified stacked bar chart
            signals = [f'Signal {chr(65+i)}' for i in range(4)]
            width = 0.6
            
            green_times, yellow_times, red_times = self.cached_timing
            
            ax.bar(signals, green_times, width, label='Green', color='green')
            ax.bar(signals, yellow_times, width, bottom=green_times, 
                  label='Yellow', color='yellow')
            ax.bar(signals, red_times, width, 
                  bottom=np.array(green_times) + np.array(yellow_times),
                  label='Red', color='red')
            
            ax.set_title('Current Signal Timing')
            ax.set_xlabel('Signal')
            ax.set_ylabel('Time (seconds)')
            ax.grid(False)
            ax.legend(loc='upper right', frameon=False)
            
            # Format with minimal decorations
            ax.tick_params(axis='both', which='both', length=0)
            fig.tight_layout()
            
            # Use faster canvas draw
            self.canvases['timing'].draw_idle()
            
        except Exception as e:
            logging.error(f"Error updating timing graph: {str(e)}")


class ReportsPanel(ttk.Frame):
    """Panel for generating and exporting reports"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup reports UI"""
        # Title
        title_label = ttk.Label(self, text="Reports & Export", 
                              font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Report options frame
        options_frame = ttk.LabelFrame(self, text="Report Options", padding=10)
        options_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Time range selection
        time_frame = ttk.Frame(options_frame)
        time_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(time_frame, text="Time Range:").pack(side=tk.LEFT)
        self.time_range_var = tk.StringVar(value="Last Hour")
        time_combo = ttk.Combobox(time_frame, textvariable=self.time_range_var,
                                 values=["Last Hour", "Last 6 Hours", "Last 24 Hours", 
                                        "Last Week", "Custom Range"], state="readonly")
        time_combo.pack(side=tk.LEFT, padx=5)
        
        # Export buttons
        export_frame = ttk.Frame(options_frame)
        export_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(export_frame, text="Export CSV", 
                  command=self.export_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Generate Summary Report", 
                  command=self.generate_summary_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(export_frame, text="Export Analytics", 
                  command=self.export_analytics).pack(side=tk.LEFT, padx=5)
        
        # Report preview
        preview_frame = ttk.LabelFrame(self, text="Report Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(preview_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.report_text = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=scrollbar.set)
        
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Generate initial preview
        self.generate_preview()
    
    def get_time_range(self):
        """Get datetime range based on selection"""
        now = datetime.now()
        range_map = {
            "Last Hour": timedelta(hours=1),
            "Last 6 Hours": timedelta(hours=6),
            "Last 24 Hours": timedelta(days=1),
            "Last Week": timedelta(weeks=1)
        }
        
        if self.time_range_var.get() in range_map:
            start_time = now - range_map[self.time_range_var.get()]
            return start_time, now
        else:
            # Custom range - for now, default to last hour
            return now - timedelta(hours=1), now
    
    def export_csv(self):
        """Export traffic data to CSV"""
        try:
            start_time, end_time = self.get_time_range()
            df = self.traffic_manager.get_analytics_data_range(start_time, end_time)
            
            if df.empty:
                messagebox.showwarning("No Data", "No data available for the selected time range.")
                return
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Traffic Data"
            )
            
            if filename:
                df.to_csv(filename, index=False)
                messagebox.showinfo("Export Successful", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        try:
            start_time, end_time = self.get_time_range()
            df = self.traffic_manager.get_analytics_data_range(start_time, end_time)
            
            if df.empty:
                messagebox.showwarning("No Data", "No data available for the selected time range.")
                return
            
            # Generate summary statistics
            summary = self.create_summary_report(df, start_time, end_time)
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Summary Report"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(summary)
                messagebox.showinfo("Report Generated", f"Summary report saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report: {str(e)}")
    
    def export_analytics(self):
        """Export detailed analytics data"""
        try:
            start_time, end_time = self.get_time_range()
            df = self.traffic_manager.get_analytics_data_range(start_time, end_time)
            
            if df.empty:
                messagebox.showwarning("No Data", "No data available for the selected time range.")
                return
            
            # Create analytics summary
            analytics = self.create_analytics_export(df, start_time, end_time)
            
            # Ask user for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Analytics Data"
            )
            
            if filename:
                import json
                with open(filename, 'w') as f:
                    json.dump(analytics, f, indent=2, default=str)
                messagebox.showinfo("Analytics Exported", f"Analytics data saved to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export analytics: {str(e)}")
    
    def create_summary_report(self, df, start_time, end_time):
        """Create a formatted summary report"""
        report = f"""
SMART TRAFFIC MANAGEMENT SYSTEM
SUMMARY REPORT
{'='*50}

Report Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL STATISTICS
{'='*50}
Total Records: {len(df)}
Total Vehicles Detected: {df['vehicle_count'].sum()}
Average Traffic Weight: {df['traffic_weight'].mean():.2f}
Average Efficiency Score: {df['efficiency_score'].mean():.2f}%

SIGNAL PERFORMANCE
{'='*50}
"""
        
        for signal_id in range(4):
            signal_data = df[df['signal_id'] == signal_id]
            if not signal_data.empty:
                report += f"""
Signal {chr(65 + signal_id)}:
  - Total Vehicles: {signal_data['vehicle_count'].sum()}
  - Average Vehicles/Hour: {signal_data['vehicle_count'].mean():.1f}
  - Average Traffic Weight: {signal_data['traffic_weight'].mean():.2f}
  - Average Green Time: {signal_data['green_time'].mean():.1f}s
  - Average Efficiency: {signal_data['efficiency_score'].mean():.1f}%
  - Peak Traffic: {signal_data['vehicle_count'].max()} vehicles
"""
        
        # Traffic patterns
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_traffic = df.groupby('hour')['vehicle_count'].sum()
            peak_hour = hourly_traffic.idxmax()
            
            report += f"""
TRAFFIC PATTERNS
{'='*50}
Peak Traffic Hour: {peak_hour}:00 ({hourly_traffic[peak_hour]} vehicles)
Lowest Traffic Hour: {hourly_traffic.idxmin()}:00 ({hourly_traffic.min()} vehicles)

RECOMMENDATIONS
{'='*50}
"""
            
            # Add recommendations based on data analysis
            avg_efficiency = df['efficiency_score'].mean()
            if avg_efficiency < 60:
                report += "- Consider optimizing signal timing algorithms\n"
            if df['vehicle_count'].std() > df['vehicle_count'].mean() * 0.5:
                report += "- High traffic variability detected - consider adaptive timing\n"
            
            report += f"- System efficiency is {'good' if avg_efficiency > 70 else 'acceptable' if avg_efficiency > 50 else 'needs improvement'}\n"
        
        return report
    
    def create_analytics_export(self, df, start_time, end_time):
        """Create detailed analytics for export"""
        analytics = {
            'report_info': {
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'generated_at': datetime.now().isoformat(),
                'total_records': len(df)
            },
            'overall_stats': {
                'total_vehicles': int(df['vehicle_count'].sum()),
                'avg_traffic_weight': float(df['traffic_weight'].mean()),
                'avg_efficiency': float(df['efficiency_score'].mean()),
                'avg_green_time': float(df['green_time'].mean())
            },
            'signal_stats': {},
            'hourly_patterns': {},
            'performance_metrics': {}
        }
        
        # Signal-specific statistics
        for signal_id in range(4):
            signal_data = df[df['signal_id'] == signal_id]
            if not signal_data.empty:
                analytics['signal_stats'][f'signal_{signal_id}'] = {
                    'total_vehicles': int(signal_data['vehicle_count'].sum()),
                    'avg_vehicles': float(signal_data['vehicle_count'].mean()),
                    'max_vehicles': int(signal_data['vehicle_count'].max()),
                    'avg_weight': float(signal_data['traffic_weight'].mean()),
                    'avg_efficiency': float(signal_data['efficiency_score'].mean()),
                    'avg_green_time': float(signal_data['green_time'].mean())
                }
        
        # Hourly patterns
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            hourly_data = df.groupby('hour').agg({
                'vehicle_count': ['sum', 'mean'],
                'efficiency_score': 'mean'
            }).round(2)
            
            analytics['hourly_patterns'] = hourly_data.to_dict()
        
        return analytics
    
    def generate_preview(self):
        """Generate a preview of the current report"""
        try:
            start_time, end_time = self.get_time_range()
            df = self.traffic_manager.get_analytics_data_range(start_time, end_time)
            
            self.report_text.delete(1.0, tk.END)
            
            if df.empty:
                self.report_text.insert(tk.END, "No data available for the selected time range.")
                return
            
            preview = f"""TRAFFIC MANAGEMENT SYSTEM - PREVIEW
{'='*50}
Time Range: {self.time_range_var.get()}
Records: {len(df)}
Total Vehicles: {df['vehicle_count'].sum()}
Average Efficiency: {df['efficiency_score'].mean():.1f}%

SIGNAL SUMMARY:
{'='*20}
"""
            
            for signal_id in range(4):
                signal_data = df[df['signal_id'] == signal_id]
                if not signal_data.empty:
                    preview += f"Signal {chr(65 + signal_id)}: {signal_data['vehicle_count'].sum()} vehicles, "
                    preview += f"{signal_data['efficiency_score'].mean():.1f}% efficiency\n"
            
            self.report_text.insert(tk.END, preview)
            
        except Exception as e:
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, f"Error generating preview: {str(e)}")


class AnalyticsPanel(ttk.Frame):
    """Main analytics panel combining all analytics components"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        
        self.setup_ui()
        self.pack(fill=tk.BOTH, expand=True)
    
    def setup_ui(self):
        """Setup main analytics UI"""
        # Create notebook for different analytics sections
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Live Statistics Tab
        self.live_stats_panel = LiveStatsPanel(self.notebook, self.traffic_manager)
        self.notebook.add(self.live_stats_panel, text="Live Stats")
        self.live_stats_panel.start_updates()
        
        # Graphs Tab
        self.graphs_panel = GraphsPanel(self.notebook, self.traffic_manager)
        self.notebook.add(self.graphs_panel, text="Graphs")
        
        # Reports Tab
        self.reports_panel = ReportsPanel(self.notebook, self.traffic_manager)
        self.notebook.add(self.reports_panel, text="Reports & Export")
    
    def update_analytics(self):
        """Update all analytics components"""
        if hasattr(self.live_stats_panel, 'update_stats'):
            self.live_stats_panel.update_stats()
        if hasattr(self.graphs_panel, 'update_graphs'):
            self.graphs_panel.update_graphs()
    
    def __del__(self):
        """Cleanup when panel is destroyed"""
        if hasattr(self, 'live_stats_panel'):
            self.live_stats_panel.stop_updates()
        if hasattr(self, 'graphs_panel'):
            self.graphs_panel.stop_graph_updates()


# Additional utility functions for analytics
def calculate_traffic_efficiency(vehicle_count, green_time, max_capacity=50):
    """Calculate traffic efficiency score"""
    if green_time == 0:
        return 0.0
    
    # Simple efficiency calculation based on vehicles processed vs green time
    vehicles_per_second = vehicle_count / max(green_time, 1)
    max_vehicles_per_second = max_capacity / 30  # Assuming 30 seconds optimal green time
    
    efficiency = min((vehicles_per_second / max_vehicles_per_second) * 100, 100)
    return round(efficiency, 2)


def calculate_traffic_weight(vehicle_count, vehicle_types=None):
    """Calculate weighted traffic score based on vehicle types"""
    if vehicle_types is None:
        # Default weight calculation based on count
        return vehicle_count * 1.0
    
    # Weight factors for different vehicle types
    weights = {
        'car': 1.0,
        'bus': 2.0,
        'truck': 2.5,
        'motorcycle': 0.5,
        'bicycle': 0.3
    }
    
    total_weight = 0.0
    for vehicle_type, count in vehicle_types.items():
        total_weight += count * weights.get(vehicle_type, 1.0)
    
    return round(total_weight, 2)


def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def generate_traffic_insights(df):
    """Generate insights from traffic data"""
    insights = []
    
    if df.empty:
        return ["No data available for analysis"]
    
    # Traffic volume insights
    total_vehicles = df['vehicle_count'].sum()
    avg_vehicles = df['vehicle_count'].mean()
    
    if total_vehicles > 1000:
        insights.append("High traffic volume detected - consider optimization")
    elif total_vehicles < 100:
        insights.append("Low traffic period - signals could be optimized for efficiency")
    
    # Efficiency insights
    avg_efficiency = df['efficiency_score'].mean()
    if avg_efficiency < 50:
        insights.append("Low system efficiency - review signal timing algorithms")
    elif avg_efficiency > 80:
        insights.append("Excellent system efficiency - current settings are optimal")
    
    # Signal balance insights
    signal_variance = df.groupby('signal_id')['vehicle_count'].sum().std()
    if signal_variance > avg_vehicles:
        insights.append("Unbalanced traffic distribution across signals")
    
    # Time-based insights
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        peak_hour = df.groupby('hour')['vehicle_count'].sum().idxmax()
        insights.append(f"Peak traffic hour: {peak_hour}:00")
    
    return insights if insights else ["Traffic patterns are within normal ranges"]