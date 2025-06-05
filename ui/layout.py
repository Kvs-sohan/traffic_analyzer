"""
Main UI Layout for Smart Traffic Management System
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime
import numpy as np
import logging
import json
import os

from .analytics import AnalyticsPanel
from .settings import SimplifiedTrafficSettings

# Define styles and colors
BUTTON_STYLES = {
    'Success': {'background': '#28a745', 'foreground': 'white'},
    'Danger': {'background': '#dc3545', 'foreground': 'white'},
    'Primary': {'background': '#007bff', 'foreground': 'white'}
}

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

class VideoPanel(ttk.Frame):
    """Panel for displaying video feed with signal status"""
    
    def __init__(self, parent, signal_id, signal_name):
        super().__init__(parent)
        self.signal_id = signal_id
        self.signal_name = signal_name
        self.current_frame = None
        self.video_label = None
        self.photo = None  # Keep a reference to avoid garbage collection
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup video panel UI"""
        # Main frame with border
        main_frame = ttk.LabelFrame(self, text=self.signal_name, padding=5)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Video display area
        self.video_label = ttk.Label(main_frame)
        self.video_label.pack(pady=5)
        
        # Create a black frame as placeholder
        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.update_video(black_frame)
        
        # Status frame
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=5)
        
        # Signal status indicator
        self.status_indicator = ttk.Label(status_frame, text="● RED", 
                                        foreground=SIGNAL_CONFIG['colors']['red'],
                                        font=('Arial', 12, 'bold'))
        self.status_indicator.pack(side=tk.LEFT)
        
        # Timer
        self.timer_label = ttk.Label(status_frame, text="Time: 0s", 
                                   font=('Arial', 10))
        self.timer_label.pack(side=tk.RIGHT)
        
        # Metrics frame
        metrics_frame = ttk.Frame(main_frame)
        metrics_frame.pack(fill=tk.X, pady=2)
        
        # Vehicle count
        self.vehicle_label = ttk.Label(metrics_frame, text="Vehicles: 0 | Weight: 0.0",
                                     font=('Arial', 9))
        self.vehicle_label.pack()
        
        # Efficiency score
        self.efficiency_label = ttk.Label(metrics_frame, text="Efficiency: 0.0%",
                                        font=('Arial', 9))
        self.efficiency_label.pack()
    
    def update_video(self, frame):
        """Update video display with new frame"""
        try:
            if frame is None:
                return
                
            # Resize frame to fit display area
            frame = cv2.resize(frame, (320, 240))
            
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_rgb)
            self.photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.video_label.configure(image=self.photo)
            
        except Exception as e:
            logging.error(f"Error updating video for signal {self.signal_id}: {str(e)}")
    
    def update_status(self, signal_state, metrics):
        """Update signal status and metrics"""
        # Update signal indicator
        state = signal_state.get('current_state', 'red').upper()
        color = SIGNAL_CONFIG['colors'].get(state.lower(), '#666666')
        self.status_indicator.configure(text=f"● {state}", foreground=color)
        
        # Update timer
        remaining_time = signal_state.get('remaining_time', 0)
        self.timer_label.configure(text=f"Time: {remaining_time}s")
        
        # Update metrics
        vehicle_count = metrics.get('vehicle_count', 0)
        traffic_weight = metrics.get('traffic_weight', 0.0)
        efficiency = metrics.get('efficiency_score', 0.0)
        
        self.vehicle_label.configure(
            text=f"Vehicles: {vehicle_count} | Weight: {traffic_weight:.1f}"
        )
        self.efficiency_label.configure(text=f"Efficiency: {efficiency:.1f}%")

class ControlPanel(ttk.Frame):
    """Control panel for system operations"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.is_running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup control panel UI"""
        # Title
        title_label = ttk.Label(self, text="System Control", 
                              font=('Arial', 12, 'bold'))
        title_label.pack(pady=5)
        
        # Control buttons frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=10)
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(buttons_frame, text="Start System",
                                       command=self.toggle_system,
                                       style='Success.TButton')
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Load cameras button
        load_cameras_btn = ttk.Button(buttons_frame, text="Load Cameras",
                                    command=self.load_cameras)
        load_cameras_btn.pack(side=tk.LEFT, padx=5)
        
        # New Areas button
        new_areas_btn = ttk.Button(buttons_frame, text="New Areas",
                                 command=self.configure_new_areas)
        new_areas_btn.pack(side=tk.LEFT, padx=5)
        
        # Load areas button
        load_areas_btn = ttk.Button(buttons_frame, text="Load Saved Areas",
                                  command=self.load_saved_areas)
        load_areas_btn.pack(side=tk.LEFT, padx=5)
        
        # Export data button
        export_btn = ttk.Button(buttons_frame, text="Export Data",
                              command=self.export_data)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(self, text="System Status", padding=10)
        status_frame.pack(fill=tk.X, pady=10)
        
        # System status labels
        self.status_label = ttk.Label(status_frame, text="Status: Stopped",
                                    font=('Arial', 10))
        self.status_label.pack()
        
        self.cameras_label = ttk.Label(status_frame, text="Cameras: 0/4 Active")
        self.cameras_label.pack()
        
        self.uptime_label = ttk.Label(status_frame, text="Uptime: 0:00:00")
        self.uptime_label.pack()
        
        # Performance frame
        perf_frame = ttk.LabelFrame(self, text="Performance", padding=10)
        perf_frame.pack(fill=tk.X, pady=5)
        
        self.fps_labels = []
        for i in range(4):
            fps_label = ttk.Label(perf_frame, text=f"Signal {chr(65+i)}: 0 FPS",
                                font=('Arial', 9))
            fps_label.pack()
            self.fps_labels.append(fps_label)
    
    def toggle_system(self):
        """Toggle system start/stop"""
        if not self.is_running:
            self.start_system()
        else:
            self.stop_system()
    
    def start_system(self):
        """Start the traffic management system"""
        try:
            self.traffic_manager.start_system()
            self.is_running = True
            self.start_stop_btn.configure(text="Stop System", style='Danger.TButton')
            self.status_label.configure(text="Status: Running")
            messagebox.showinfo("Success", "System started successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {e}")
    
    def stop_system(self):
        """Stop the traffic management system"""
        try:
            self.traffic_manager.stop_system()
            self.is_running = False
            self.start_stop_btn.configure(text="Start System", style='Success.TButton')
            self.status_label.configure(text="Status: Stopped")
            messagebox.showinfo("Success", "System stopped successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop system: {e}")
    
    def load_cameras(self):
        """Load camera sources dialog"""
        dialog = CameraLoadDialog(self, self.traffic_manager)
        self.wait_window(dialog)
    
    def configure_new_areas(self):
        """Open dialog to configure new detection areas"""
        dialog = AreaLoadDialog(self, self.traffic_manager)
        self.wait_window(dialog)
    
    def load_saved_areas(self):
        """Load previously saved detection areas"""
        try:
            # Try to load areas from the saved configuration
            config_file = os.path.join('config', 'areas.json')
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    saved_areas = json.load(f)
                
                # Update the areas in traffic manager
                success = self.traffic_manager.area_manager.load_areas(config_file)
                if success:
                    messagebox.showinfo("Success", "Previously saved areas loaded successfully!")
                else:
                    messagebox.showerror("Error", "Failed to load saved areas")
            else:
                messagebox.showwarning("No Saved Areas", 
                                     "No previously saved areas found. Use 'New Areas' to configure areas.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load saved areas: {str(e)}")
    
    def export_data(self):
        """Export traffic data"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if filename:
                result = self.traffic_manager.export_data(format='csv')
                if result:
                    messagebox.showinfo("Success", f"Data exported to {filename}")
                else:
                    messagebox.showerror("Error", "Failed to export data")
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def update_status(self, system_status):
        """Update system status display"""
        # Update camera status
        active_cameras = system_status.get('active_cameras', 0)
        total_cameras = system_status.get('total_cameras', 4)
        self.cameras_label.configure(text=f"Cameras: {active_cameras}/{total_cameras} Active")
        
        # Update uptime
        uptime = system_status.get('uptime', 0)
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        self.uptime_label.configure(text=f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        # Update FPS
        fps_data = system_status.get('fps', [0, 0, 0, 0])
        for i, fps in enumerate(fps_data):
            if i < len(self.fps_labels):
                self.fps_labels[i].configure(text=f"Signal {chr(65+i)}: {fps} FPS")

class CameraLoadDialog(tk.Toplevel):
    """Dialog for loading camera sources"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.camera_entries = []
        
        self.title("Load Camera Sources")
        self.geometry("500x300")
        self.transient(parent)
        self.grab_set()
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup camera load dialog UI"""
        # Title
        title_label = ttk.Label(self, text="Configure Camera Sources", 
                              font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)
        
        # Info label
        info_label = ttk.Label(self, text="Enter RTSP URLs or video file paths:")
        info_label.pack(pady=5)
        
        # Camera entries frame
        entries_frame = ttk.Frame(self)
        entries_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create entries for 4 cameras
        for i in range(4):
            # Label
            label = ttk.Label(entries_frame, text=f"Signal {chr(65+i)}:")
            label.grid(row=i, column=0, sticky=tk.W, pady=5)
            
            # Entry
            entry = ttk.Entry(entries_frame, width=50)
            entry.grid(row=i, column=1, sticky=tk.EW, padx=10, pady=5)
            self.camera_entries.append(entry)
            
            # Browse button for file sources
            browse_btn = ttk.Button(entries_frame, text="Browse",
                                  command=lambda idx=i: self.browse_file(idx))
            browse_btn.grid(row=i, column=2, padx=5, pady=5)
        
        entries_frame.columnconfigure(1, weight=1)
        
        # Buttons frame
        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=10)
        
        # Load button
        load_btn = ttk.Button(buttons_frame, text="Load Cameras",
                            command=self.load_cameras)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Cancel button
        cancel_btn = ttk.Button(buttons_frame, text="Cancel",
                              command=self.destroy)
        cancel_btn.pack(side=tk.LEFT, padx=5)
        
        # Load current sources
        self.load_current_sources()
    
    def browse_file(self, index):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title=f"Select video file for Signal {chr(65+index)}",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.camera_entries[index].delete(0, tk.END)
            self.camera_entries[index].insert(0, filename)
    
    def load_current_sources(self):
        """Load current camera sources"""
        try:
            sources = self.traffic_manager.camera_manager.sources
            for i, source in enumerate(sources[:4]):
                if i < len(self.camera_entries):
                    self.camera_entries[i].delete(0, tk.END)
                    self.camera_entries[i].insert(0, source)
        except Exception as e:
            print(f"Error loading current sources: {e}")
    
    def load_cameras(self):
        """Load the specified camera sources"""
        try:
            sources = []
            for entry in self.camera_entries:
                source = entry.get().strip()
                if source:
                    sources.append(source)
                else:
                    sources.append("")  # Empty source for demo frame
            
            success = self.traffic_manager.load_camera_sources(sources)
            if success:
                messagebox.showinfo("Success", "Camera sources loaded successfully!")
                self.destroy()
            else:
                messagebox.showerror("Error", "Failed to load camera sources")
        except Exception as e:
            messagebox.showerror("Error", f"Loading failed: {e}")

class AreaLoadDialog(tk.Toplevel):
    """Dialog for configuring detection areas"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.points = []
        self.drawing = False
        
        self.title("Configure Detection Areas")
        self.geometry("800x600")
        self.transient(parent)
        self.grab_set()
        
        # Main frame
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        top_panel = ttk.Frame(main_frame)
        top_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # Title and signal selection on left
        controls_left = ttk.Frame(top_panel)
        controls_left.pack(side=tk.LEFT)
        
        title = ttk.Label(controls_left, text="Detection Area Configuration", 
                         font=('Arial', 12, 'bold'))
        title.pack(side=tk.LEFT, padx=10)
        
        ttk.Label(controls_left, text="| Select Signal:").pack(side=tk.LEFT, padx=5)
        
        self.signal_var = tk.StringVar(value="0")
        for i in range(4):
            ttk.Radiobutton(controls_left, text=f"Signal {chr(65+i)}",
                          variable=self.signal_var, value=str(i),
                          command=self.on_signal_change).pack(side=tk.LEFT, padx=5)
        
        # Buttons on right
        buttons_frame = ttk.Frame(top_panel)
        buttons_frame.pack(side=tk.RIGHT, padx=10)
        
        # Save button
        self.save_btn = tk.Button(buttons_frame, text="Save Area", 
                                command=self.save_area,
                                bg='green', fg='white',
                                width=10,
                                state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        tk.Button(buttons_frame, text="Reset", 
                 command=self.reset_area,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # Close button
        tk.Button(buttons_frame, text="Close", 
                 command=self.destroy,
                 width=10).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Click to add points")
        self.status_label.pack(pady=5)
        
        # Canvas
        self.canvas = tk.Canvas(main_frame, bg='black', width=640, height=480)
        self.canvas.pack(pady=10, padx=10)
        
        # Bind events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_move)
        self.canvas.bind("<Button-3>", self.on_right_click)
        
        # Load initial frame
        self.load_current_frame()
    
    def on_click(self, event):
        """Handle left click"""
        x, y = event.x, event.y
        self.points.append((x, y))
        self.draw_area()
        
        if len(self.points) >= 3:
            self.save_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Area complete - click Save Area")
        else:
            self.status_label.config(text=f"Add {3 - len(self.points)} more points")
    
    def on_right_click(self, event):
        """Handle right click"""
        if self.points:
            self.points.pop()
            self.draw_area()
            
            if len(self.points) < 3:
                self.save_btn.config(state=tk.DISABLED)
                self.status_label.config(text=f"Add {3 - len(self.points)} more points")
    
    def on_move(self, event):
        """Handle mouse movement"""
        if self.points:
            self.draw_area(temp_point=(event.x, event.y))
    
    def draw_area(self, temp_point=None):
        """Draw the area"""
        self.canvas.delete("area")
        
        if self.points:
            # Draw lines between points
            for i in range(len(self.points) - 1):
                self.canvas.create_line(self.points[i], self.points[i + 1],
                                     fill="green", width=2, tags="area")
            
            # Draw temporary line to cursor
            if temp_point:
                self.canvas.create_line(self.points[-1], temp_point,
                                     fill="green", width=2, tags="area")
            
            # Draw points
            for point in self.points:
                self.canvas.create_oval(point[0]-3, point[1]-3,
                                     point[0]+3, point[1]+3,
                                     fill="red", tags="area")
            
            # Close the polygon if we have enough points
            if len(self.points) >= 3:
                self.canvas.create_line(self.points[-1], self.points[0],
                                     fill="green", width=2, tags="area")
    
    def save_area(self):
        """Save the area"""
        if len(self.points) >= 3:
            signal_id = int(self.signal_var.get())
            if self.traffic_manager.update_detection_area(signal_id, self.points):
                messagebox.showinfo("Success", 
                                  f"Area saved for Signal {chr(65+signal_id)}")
                self.reset_area()
                
                # Move to next signal
                if signal_id < 3:
                    self.signal_var.set(str(signal_id + 1))
                    self.on_signal_change()
                else:
                    self.destroy()
            else:
                messagebox.showerror("Error", "Failed to save area")
    
    def reset_area(self):
        """Reset the current area"""
        self.points = []
        self.save_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Click to add points")
        self.draw_area()
    
    def on_signal_change(self):
        """Handle signal selection change"""
        self.reset_area()
        self.load_current_frame()
    
    def load_current_frame(self):
        """Load the current frame from camera"""
        try:
            signal_id = int(self.signal_var.get())
            frame = self.traffic_manager.camera_manager.get_frame(signal_id)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 480))
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
                # Load existing area
                area = self.traffic_manager.area_manager.get_area(signal_id)
                if area:
                    self.points = area
                    self.draw_area()
            else:
                self.canvas.create_text(320, 240, text="No camera feed available",
                                     fill="white", font=('Arial', 14))
        except Exception as e:
            print(f"Error loading frame: {e}")
            self.canvas.create_text(320, 240, text="Error loading camera feed",
                                 fill="white", font=('Arial', 14))

class TrafficUI:
    """Main UI class for Smart Traffic Management System"""
    
    def __init__(self, root, traffic_manager):
        self.traffic_manager = traffic_manager
        self.root = root
        self.video_panels = []
        self.control_panel = None
        self.analytics_panel = None
        self.settings_panel = None
        self.update_thread = None
        self.running = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup main UI"""
        # Configure styles
        self.setup_styles()
        
        # Create main notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main monitoring tab
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="Traffic Monitoring")
        self.setup_main_tab(main_frame)
        
        # Analytics tab
        analytics_frame = ttk.Frame(notebook)
        notebook.add(analytics_frame, text="Analytics")
        self.analytics_panel = AnalyticsPanel(analytics_frame, self.traffic_manager)
        
        # Settings tab
        settings_frame = ttk.Frame(notebook)
        notebook.add(settings_frame, text="Settings")
        self.settings_panel = SimplifiedTrafficSettings(settings_frame, self.traffic_manager)
        self.settings_panel.pack(fill=tk.BOTH, expand=True)
        
        # Setup update thread
        self.start_update_thread()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        
        # Configure button styles
        style.configure('Success.TButton', background='#28a745')
        style.configure('Danger.TButton', background='#dc3545')
        style.configure('Warning.TButton', background='#ffc107')
    
    def setup_main_tab(self, parent):
        """Setup main monitoring tab"""
        # Main container
        main_container = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for video feeds
        video_container = ttk.Frame(main_container)
        main_container.add(video_container, weight=3)
        
        # Video grid
        video_grid = ttk.Frame(video_container)
        video_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create 2x2 grid for video panels
        for i in range(4):
            row = i // 2
            col = i % 2
            
            signal_name = SIGNAL_CONFIG['names'][i]
            panel = VideoPanel(video_grid, i, signal_name)
            panel.grid(row=row, column=col, sticky=tk.NSEW, padx=2, pady=2)
            self.video_panels.append(panel)
        
        # Configure grid weights
        video_grid.grid_rowconfigure(0, weight=1)
        video_grid.grid_rowconfigure(1, weight=1)
        video_grid.grid_columnconfigure(0, weight=1)
        video_grid.grid_columnconfigure(1, weight=1)
        
        # Right panel for controls
        control_container = ttk.Frame(main_container)
        main_container.add(control_container, weight=1)
        
        self.control_panel = ControlPanel(control_container, self.traffic_manager)
        self.control_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def start_update_thread(self):
        """Start UI update thread"""
        self.running = True
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
    
    def update_loop(self):
        """Main update loop for UI"""
        while self.running:
            try:
                self.update_ui()
                time.sleep(UI_CONFIG['update_interval'] / 1000.0)
            except Exception as e:
                print(f"UI update error: {e}")
                time.sleep(1.0)
    
    def update_ui(self):
        """Update UI elements"""
        if not self.root or not self.root.winfo_exists():
            return
        
        try:
            # Update video panels
            for i, panel in enumerate(self.video_panels):
                # Get current frame
                frame = self.traffic_manager.get_current_frame(i)
                if frame is not None:
                    panel.update_video(frame)
                
                # Get signal state and metrics
                signal_state = self.traffic_manager.get_signal(i)
                metrics = self.traffic_manager.get_current_metrics(i)
                panel.update_status(signal_state, metrics)
            
            # Update control panel
            if self.control_panel:
                system_status = self.traffic_manager.get_system_status()
                self.control_panel.update_status(system_status)
            
            # Update analytics panel
            if self.analytics_panel and hasattr(self.analytics_panel, 'update_analytics'):
                self.analytics_panel.update_analytics()
                
        except Exception as e:
            print(f"Error updating UI: {e}")
    
    def run(self):
        """Run the UI main loop"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.on_closing()
    
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        
        # Stop traffic manager if running
        if self.traffic_manager and hasattr(self.traffic_manager, 'running'):
            if self.traffic_manager.running:
                self.traffic_manager.stop_system()
        
        # Wait for update thread to finish
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        if self.root:
            self.root.quit()
            self.root.destroy()