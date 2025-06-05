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

# UI Configuration
UI_CONFIG = {
    'window_title': 'Smart Traffic Management System',
    'window_size': (1400, 900),
    'min_window_size': (1200, 700),
    'theme': 'modern',
    'update_interval': 100,  # milliseconds
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
        
        self.setup_ui()
        self.start_updates()
    
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
        self.create_stat_label(stats_frame, 'avg_efficiency', "Avg Efficiency: 0%")
        self.create_stat_label(stats_frame, 'peak_signal', "Peak Signal: None")
    
    def create_stat_label(self, parent, key, initial_text):
        """Create a statistics label"""
        label = ttk.Label(parent, text=initial_text, font=('Arial', 10))
        label.pack(pady=2)
        self.stats_labels[key] = label
    
    def start_updates(self):
        """Start the statistics update"""
        self.running = True
        self.update_stats()
    
    def stop_updates(self):
        """Stop the statistics update"""
        self.running = False
    
    def update_stats(self):
        """Update all statistics displays"""
        if not self.running:
            return
            
        try:
            # Get current traffic data
            current_data = self.traffic_manager.get_live_statistics()
            
            if 'signals' in current_data:
                # Calculate overall statistics
                vehicle_counts = [signal['metrics'].get('vehicle_count', 0) 
                                for signal in current_data['signals']]
                efficiencies = [signal['metrics'].get('efficiency_score', 0.0) 
                              for signal in current_data['signals']]
                
                total_vehicles = sum(vehicle_counts)
                avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0
                peak_signal_idx = vehicle_counts.index(max(vehicle_counts)) if vehicle_counts else 0
                
                # Update labels using after() for thread safety
                self.after_idle(self.update_label, 'total_vehicles', f"Total Vehicles: {total_vehicles}")
                self.after_idle(self.update_label, 'avg_efficiency', f"Avg Efficiency: {avg_efficiency:.1f}%")
                self.after_idle(self.update_label, 'peak_signal', f"Peak Signal: Signal {chr(65 + peak_signal_idx)}")
                
        except Exception as e:
            print(f"Error updating live stats: {e}")
        
        # Schedule next update
        self.after(1000, self.update_stats)
    
    def update_label(self, key, text):
        """Thread-safe label update"""
        if key in self.stats_labels:
            self.stats_labels[key].config(text=text)


class GraphsPanel(ttk.Frame):
    """Panel for displaying real-time graphs"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.figures = {}
        self.canvases = {}
        self.data_history = {'timestamps': [], 'vehicle_counts': [[] for _ in range(4)], 
                           'efficiencies': [[] for _ in range(4)]}
        self.max_history = 50  # Keep last 50 data points
        
        self.setup_ui()
        self.start_graph_updates()
    
    def setup_ui(self):
        """Setup graphs UI"""
        # Title
        title_label = ttk.Label(self, text="Real-Time Analytics", 
                              font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Notebook for different graph types
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Vehicle Count Graph
        self.setup_vehicle_count_graph()
        
        # Efficiency Graph
        self.setup_efficiency_graph()
        
        # Traffic Flow Heatmap
        self.setup_heatmap()
        
        # Comparative Analysis
        self.setup_comparative_analysis()
    
    def setup_vehicle_count_graph(self):
        """Setup vehicle count over time graph"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Vehicle Count")
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Real-Time Vehicle Count')
        ax.set_xlabel('Time')
        ax.set_ylabel('Vehicle Count')
        ax.grid(True, alpha=0.3)
        
        self.figures['vehicle_count'] = fig
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['vehicle_count'] = canvas
    
    def setup_efficiency_graph(self):
        """Setup efficiency over time graph"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Efficiency")
        
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title('Signal Efficiency Over Time')
        ax.set_xlabel('Time')
        ax.set_ylabel('Efficiency (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        self.figures['efficiency'] = fig
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['efficiency'] = canvas
    
    def setup_heatmap(self):
        """Setup traffic flow heatmap"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Traffic Heatmap")
        
        fig = Figure(figsize=(10, 6), dpi=100)
        self.figures['heatmap'] = fig
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['heatmap'] = canvas
    
    def setup_comparative_analysis(self):
        """Setup comparative analysis charts"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Comparative Analysis")
        
        fig = Figure(figsize=(12, 8), dpi=100)
        self.figures['comparative'] = fig
        
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvases['comparative'] = canvas
    
    def start_graph_updates(self):
        """Start updating graphs"""
        self.running = True
        self.update_graphs()
    
    def stop_updates(self):
        """Stop graph updates"""
        self.running = False
    
    def update_graphs(self):
        """Update all graphs with latest data"""
        if not self.running:
            return
            
        try:
            # Get current data
            current_data = self.traffic_manager.get_live_statistics()
            current_time = datetime.now()
            
            if 'signals' in current_data:
                # Update data history
                self.data_history['timestamps'].append(current_time)
                
                for i in range(4):
                    signal = current_data['signals'][i]
                    count = signal['metrics'].get('vehicle_count', 0)
                    efficiency = signal['metrics'].get('efficiency_score', 0.0)
                    
                    self.data_history['vehicle_counts'][i].append(count)
                    self.data_history['efficiencies'][i].append(efficiency)
                
                # Maintain max history
                if len(self.data_history['timestamps']) > self.max_history:
                    self.data_history['timestamps'] = self.data_history['timestamps'][-self.max_history:]
                    for i in range(4):
                        self.data_history['vehicle_counts'][i] = self.data_history['vehicle_counts'][i][-self.max_history:]
                        self.data_history['efficiencies'][i] = self.data_history['efficiencies'][i][-self.max_history:]
                
                # Schedule graph updates using after_idle
                self.after_idle(self.update_vehicle_count_graph)
                self.after_idle(self.update_efficiency_graph)
                self.after_idle(self.update_heatmap)
                self.after_idle(self.update_comparative_analysis)
            
        except Exception as e:
            print(f"Error updating graphs: {e}")
        
        # Schedule next update
        self.after(2000, self.update_graphs)  # Update every 2 seconds
    
    def update_vehicle_count_graph(self):
        """Update vehicle count graph"""
        try:
            fig = self.figures['vehicle_count']
            fig.clear()
            ax = fig.add_subplot(111)
            
            times = self.data_history['timestamps']
            for i in range(4):
                counts = self.data_history['vehicle_counts'][i]
                ax.plot(times, counts, label=f'Signal {chr(65 + i)}')
            
            ax.set_title('Real-Time Vehicle Count')
            ax.set_xlabel('Time')
            ax.set_ylabel('Vehicle Count')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Format x-axis to show only time
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            fig.tight_layout()
            self.canvases['vehicle_count'].draw()
            
        except Exception as e:
            print(f"Error updating vehicle count graph: {e}")
    
    def update_efficiency_graph(self):
        """Update efficiency graph"""
        try:
            fig = self.figures['efficiency']
            fig.clear()
            ax = fig.add_subplot(111)
            
            times = self.data_history['timestamps']
            for i in range(4):
                efficiencies = self.data_history['efficiencies'][i]
                ax.plot(times, efficiencies, label=f'Signal {chr(65 + i)}')
            
            ax.set_title('Signal Efficiency Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Efficiency (%)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            ax.legend()
            
            # Format x-axis to show only time
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            fig.tight_layout()
            self.canvases['efficiency'].draw()
            
        except Exception as e:
            print(f"Error updating efficiency graph: {e}")
    
    def update_heatmap(self):
        """Update traffic flow heatmap"""
        try:
            fig = self.figures['heatmap']
            fig.clear()
            
            # Get historical data for heatmap (last hour = 3600 seconds)
            df = self.traffic_manager.get_analytics_data(3600)
            
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Create hourly heatmap data
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                heatmap_data = df.groupby(['signal_id', 'hour'])['vehicle_count'].mean().unstack(fill_value=0)
                
                ax = fig.add_subplot(111)
                sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                           ax=ax, cbar_kws={'label': 'Avg Vehicle Count'})
                ax.set_title('Traffic Flow Heatmap (Last Hour)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Signal ID')
                
                # Set y-axis labels to signal names
                ax.set_yticklabels([f'Signal {chr(65 + int(i))}' for i in heatmap_data.index])
            else:
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'No data available for heatmap', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Traffic Flow Heatmap')
            
            fig.tight_layout()
            self.canvases['heatmap'].draw()
            
        except Exception as e:
            print(f"Error updating heatmap: {e}")
    
    def update_comparative_analysis(self):
        """Update comparative analysis charts"""
        try:
            fig = self.figures['comparative']
            fig.clear()
            
            current_data = self.traffic_manager.get_live_statistics()
            
            if 'signals' in current_data:
                # Create 2x2 subplot grid
                ax1 = fig.add_subplot(221)  # Current vehicle distribution
                ax2 = fig.add_subplot(222)  # Efficiency comparison
                ax3 = fig.add_subplot(223)  # Traffic weight distribution
                ax4 = fig.add_subplot(224)  # Green time utilization
                
                # Get data from signals
                vehicle_counts = []
                efficiencies = []
                weights = []
                green_times = []
                
                for signal in current_data['signals']:
                    # Get metrics
                    metrics = signal['metrics']
                    vehicle_counts.append(metrics.get('vehicle_count', 0))
                    efficiencies.append(metrics.get('efficiency_score', 0.0))
                    weights.append(metrics.get('traffic_weight', 0.0))
                    
                    # Get state info
                    state_info = signal['state']
                    if isinstance(state_info, dict):
                        green_time = state_info.get('remaining_time', 0) if state_info.get('state') == 'GREEN' else 0
                    else:
                        green_time = 0
                    green_times.append(green_time)
                
                signal_names = [f'Signal {chr(65 + i)}' for i in range(4)]
                
                # Vehicle distribution pie chart
                if sum(vehicle_counts) > 0:
                    ax1.pie(vehicle_counts, labels=signal_names, autopct='%1.1f%%')
                    ax1.set_title('Current Vehicle Distribution')
                else:
                    ax1.text(0.5, 0.5, 'No vehicles detected', ha='center', va='center')
                    ax1.set_title('Current Vehicle Distribution')
                
                # Efficiency bar chart
                bars = ax2.bar(signal_names, efficiencies)
                ax2.set_title('Signal Efficiency')
                ax2.set_ylabel('Efficiency (%)')
                ax2.set_ylim(0, 100)
                
                # Add value labels on bars
                for bar, eff in zip(bars, efficiencies):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{eff:.1f}%', ha='center', va='bottom')
                
                # Traffic weight comparison
                ax3.bar(signal_names, weights)
                ax3.set_title('Traffic Weight')
                ax3.set_ylabel('Weight')
                
                # Green time utilization
                ax4.bar(signal_names, green_times)
                ax4.set_title('Green Time')
                ax4.set_ylabel('Seconds')
                
                fig.tight_layout()
                self.canvases['comparative'].draw()
            
        except Exception as e:
            print(f"Error updating comparative analysis: {e}")


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
    
    def setup_ui(self):
        """Setup main analytics UI"""
        # Create notebook for different analytics sections
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Live Statistics Tab
        self.live_stats_panel = LiveStatsPanel(self.notebook, self.traffic_manager)
        self.notebook.add(self.live_stats_panel, text="Live Stats")
        
        # Graphs Tab
        self.graphs_panel = GraphsPanel(self.notebook, self.traffic_manager)
        self.notebook.add(self.graphs_panel, text="Real-Time Graphs")
        
        # Reports Tab
        self.reports_panel = ReportsPanel(self.notebook, self.traffic_manager)
        self.notebook.add(self.reports_panel, text="Reports & Export")
    
    def cleanup(self):
        """Cleanup method to stop all update threads"""
        if hasattr(self.live_stats_panel, 'stop_updates'):
            self.live_stats_panel.stop_updates()


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