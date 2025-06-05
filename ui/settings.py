"""
Simplified Settings Panel for Smart Traffic Management System
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser
import json
import os
import cv2
import threading
from pathlib import Path

class SimplifiedTrafficSettings(ttk.Frame):
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        
        # Configuration storage
        self.camera_configs = {}
        self.detection_areas = {i: [] for i in range(4)}
        self.camera_entries = {}  # Initialize camera_entries dict
        
        self.setup_ui()
        self.load_config()
    
    def setup_ui(self):
        """Create the main interface"""
        # Create notebook for tabs
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera Settings Tab
        self.camera_frame = ttk.Frame(notebook)
        notebook.add(self.camera_frame, text="Camera Settings")
        self.setup_camera_tab()
        
        # Detection Areas Tab
        self.areas_frame = ttk.Frame(notebook)
        notebook.add(self.areas_frame, text="Detection Areas")
        self.setup_areas_tab()
        
        # System Settings Tab
        self.system_frame = ttk.Frame(notebook)
        notebook.add(self.system_frame, text="System Settings")
        self.setup_system_tab()
    
    def setup_camera_tab(self):
        """Setup camera configuration tab"""
        ttk.Label(self.camera_frame, text="Camera Settings", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Scrollable frame for camera configs
        canvas = tk.Canvas(self.camera_frame)
        scrollbar = ttk.Scrollbar(self.camera_frame, orient="vertical", command=canvas.yview)
        cameras_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        cameras_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=cameras_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Create camera configurations
        for i in range(4):
            self.create_camera_config(cameras_frame, i)
    
    def create_camera_config(self, parent, signal_idx):
        """Create configuration for one camera"""
        signal_name = f"Signal {chr(65 + signal_idx)}"
        
        # Signal frame
        frame = ttk.LabelFrame(parent, text=signal_name, padding=5)
        frame.pack(fill=tk.X, pady=2)
        
        # Source input
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="Source:", width=8).pack(side=tk.LEFT)
        
        # Entry for source (RTSP URL, file path, or camera ID)
        source_entry = ttk.Entry(input_frame, width=40)
        source_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Browse button for files
        browse_btn = ttk.Button(input_frame, text="Browse", width=8,
                               command=lambda: self.browse_file(signal_idx))
        browse_btn.pack(side=tk.RIGHT, padx=2)
        
        # Test button
        test_btn = ttk.Button(input_frame, text="Test", width=6,
                             command=lambda: self.test_camera(signal_idx))
        test_btn.pack(side=tk.RIGHT, padx=2)
        
        # Status indicator
        status_frame = ttk.Frame(frame)
        status_frame.pack(fill=tk.X, pady=2)
        
        status_canvas = tk.Canvas(status_frame, width=15, height=15)
        status_canvas.pack(side=tk.LEFT, padx=5)
        status_canvas.create_oval(2, 2, 13, 13, fill='gray', outline='black')
        
        status_label = ttk.Label(status_frame, text="Not tested", font=('Arial', 8))
        status_label.pack(side=tk.LEFT)
        
        # Store references
        self.camera_entries[signal_idx] = {
            'source_entry': source_entry,
            'browse_btn': browse_btn,
            'status_canvas': status_canvas,
            'status_label': status_label
        }
        
        # Set default source if available
        if signal_idx < len(self.traffic_manager.camera_manager.sources):
            source = self.traffic_manager.camera_manager.sources[signal_idx]
            source_entry.insert(0, source)
    
    def update_source_type(self):
        """Update UI based on source type"""
        source_type = self.source_type_var.get()
        
        for i in range(4):
            entries = self.camera_entries[i]
            
            if source_type == "FILE":
                entries['browse_btn'].pack(side=tk.RIGHT, padx=2)
                entries['source_entry'].delete(0, tk.END)
                entries['source_entry'].insert(0, "Select video file...")
            else:
                entries['browse_btn'].pack_forget()
                if source_type == "RTSP":
                    entries['source_entry'].delete(0, tk.END)
                    entries['source_entry'].insert(0, "rtsp://")
                elif source_type == "USB":
                    entries['source_entry'].delete(0, tk.END)
                    entries['source_entry'].insert(0, str(i))
    
    def browse_file(self, signal_idx):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title=f"Select video for Signal {chr(65 + signal_idx)}",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if filename:
            self.camera_entries[signal_idx]['source_entry'].delete(0, tk.END)
            self.camera_entries[signal_idx]['source_entry'].insert(0, filename)
    
    def test_camera(self, signal_idx):
        """Test individual camera connection"""
        entries = self.camera_entries[signal_idx]
        source = entries['source_entry'].get()
        
        if not source or source.startswith("Select video") or source == "rtsp://":
            self.update_camera_status(signal_idx, "error", "No source specified")
            return
        
        self.update_camera_status(signal_idx, "testing", "Testing...")
        
        def test_thread():
            try:
                # Convert USB camera ID to integer
                if self.source_type_var.get() == "USB":
                    source_val = int(source)
                else:
                    source_val = source
                
                cap = cv2.VideoCapture(source_val)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        self.update_camera_status(signal_idx, "success", "Connected")
                    else:
                        self.update_camera_status(signal_idx, "error", "No video stream")
                else:
                    self.update_camera_status(signal_idx, "error", "Connection failed")
                cap.release()
            except Exception as e:
                self.update_camera_status(signal_idx, "error", f"Error: {str(e)[:20]}")
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def test_all_cameras(self):
        """Test all camera connections"""
        for i in range(4):
            self.test_camera(i)
    
    def update_camera_status(self, signal_idx, status, message):
        """Update camera connection status"""
        entries = self.camera_entries[signal_idx]
        
        colors = {"success": "green", "testing": "orange", "error": "red"}
        color = colors.get(status, "gray")
        
        canvas = entries['status_canvas']
        canvas.delete("all")
        canvas.create_oval(2, 2, 13, 13, fill=color, outline='black')
        
        entries['status_label'].config(text=message)
    
    def setup_areas_tab(self):
        """Setup detection areas configuration tab"""
        ttk.Label(self.areas_frame, text="Detection Areas", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Signal selection
        selection_frame = ttk.Frame(self.areas_frame)
        selection_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(selection_frame, text="Configure for:").pack(side=tk.LEFT)
        self.current_signal_var = tk.StringVar(value="Signal A")
        signal_combo = ttk.Combobox(selection_frame, textvariable=self.current_signal_var,
                                   values=[f"Signal {chr(65 + i)}" for i in range(4)],
                                   state="readonly", width=15)
        signal_combo.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(selection_frame, text="Capture Frame", 
                  command=self.capture_frame).pack(side=tk.LEFT, padx=10)
        
        # Preview and controls
        content_frame = ttk.Frame(self.areas_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Canvas for drawing areas
        canvas_frame = ttk.LabelFrame(content_frame, text="Video Preview", padding=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.areas_canvas = tk.Canvas(canvas_frame, bg='black', width=480, height=360)
        self.areas_canvas.pack()
        
        # Controls
        controls_frame = ttk.Frame(canvas_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls_frame, text="Add Area", 
                  command=self.add_detection_area).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Clear Areas", 
                  command=self.clear_areas).pack(side=tk.LEFT, padx=5)
        
        # Areas list
        list_frame = ttk.LabelFrame(content_frame, text="Configured Areas", padding=10)
        list_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        self.areas_listbox = tk.Listbox(list_frame, width=25, height=15)
        self.areas_listbox.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(list_frame, text="Remove Selected", 
                  command=self.remove_area).pack(fill=tk.X, pady=5)
    
    def setup_system_tab(self):
        """Setup system settings tab"""
        ttk.Label(self.system_frame, text="System Settings", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        settings_frame = ttk.LabelFrame(self.system_frame, text="General Settings", padding=20)
        settings_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Detection sensitivity
        ttk.Label(settings_frame, text="Detection Sensitivity:").pack(anchor=tk.W)
        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, 
                                    variable=self.sensitivity_var, orient=tk.HORIZONTAL)
        sensitivity_scale.pack(fill=tk.X, pady=5)
        
        # Signal timing
        timing_frame = ttk.Frame(settings_frame)
        timing_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(timing_frame, text="Min Green Time (seconds):").pack(side=tk.LEFT)
        self.min_green_var = tk.IntVar(value=10)
        ttk.Spinbox(timing_frame, from_=5, to=60, textvariable=self.min_green_var, 
                   width=10).pack(side=tk.RIGHT)
        
        # Save/Load buttons
        button_frame = ttk.Frame(self.system_frame)
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        ttk.Button(button_frame, text="Save All Settings", 
                  command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Settings", 
                  command=self.load_config_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", 
                  command=self.reset_settings).pack(side=tk.LEFT, padx=5)
    
    def capture_frame(self):
        """Capture frame from current signal camera"""
        messagebox.showinfo("Info", "Frame capture functionality would connect to camera here")
    
    def add_detection_area(self):
        """Add a detection area"""
        signal_idx = ord(self.current_signal_var.get()[-1]) - ord('A')
        area_name = f"Area {len(self.detection_areas[signal_idx]) + 1}"
        
        # Simulate adding an area
        self.detection_areas[signal_idx].append({
            'name': area_name,
            'points': [(100, 100), (200, 100), (200, 200), (100, 200)]
        })
        
        self.update_areas_list()
    
    def clear_areas(self):
        """Clear all areas for current signal"""
        signal_idx = ord(self.current_signal_var.get()[-1]) - ord('A')
        self.detection_areas[signal_idx] = []
        self.update_areas_list()
    
    def remove_area(self):
        """Remove selected area"""
        selection = self.areas_listbox.curselection()
        if selection:
            signal_idx = ord(self.current_signal_var.get()[-1]) - ord('A')
            del self.detection_areas[signal_idx][selection[0]]
            self.update_areas_list()
    
    def update_areas_list(self):
        """Update the areas listbox"""
        self.areas_listbox.delete(0, tk.END)
        signal_idx = ord(self.current_signal_var.get()[-1]) - ord('A')
        
        for i, area in enumerate(self.detection_areas[signal_idx]):
            self.areas_listbox.insert(tk.END, area['name'])
    
    def save_config(self):
        """Save all configuration"""
        try:
            config = {
                'source_type': self.source_type_var.get(),
                'cameras': {},
                'detection_areas': self.detection_areas,
                'system_settings': {
                    'sensitivity': self.sensitivity_var.get(),
                    'min_green_time': self.min_green_var.get()
                }
            }
            
            # Save camera configurations
            for i in range(4):
                source = self.camera_entries[i]['source_entry'].get()
                config['cameras'][i] = {'source': source}
            
            # Create config directory if it doesn't exist  
            config_dir = Path(__file__).parent.parent / 'config'
            config_dir.mkdir(exist_ok=True)

            # Save to file
            config_file = config_dir / 'traffic_config.json'
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            messagebox.showinfo("Success", "Configuration saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_config(self):
        """Load configuration from default file"""
        try:
            config_file = Path(__file__).parent.parent / 'config' / 'traffic_config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.apply_config(config)
        except Exception as e:
            print(f"Error loading configuration: {e}")
    
    def load_config_dialog(self):
        """Load configuration from file dialog"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                
                self.apply_config(config)
                messagebox.showinfo("Success", "Configuration loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def apply_config(self, config):
        """Apply configuration to UI"""
        # Source type
        if 'source_type' in config:
            self.source_type_var.set(config['source_type'])
            self.update_source_type()
        
        # Camera sources
        if 'cameras' in config:
            for i, camera_config in config['cameras'].items():
                idx = int(i)
                if idx in self.camera_entries:
                    source = camera_config.get('source', '')
                    self.camera_entries[idx]['source_entry'].delete(0, tk.END)
                    self.camera_entries[idx]['source_entry'].insert(0, source)
        
        # Detection areas
        if 'detection_areas' in config:
            self.detection_areas = config['detection_areas']
            self.update_areas_list()
        
        # System settings
        if 'system_settings' in config:
            settings = config['system_settings']
            self.sensitivity_var.set(settings.get('sensitivity', 0.5))
            self.min_green_var.set(settings.get('min_green_time', 10))
    
    def reset_settings(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Confirm", "Reset all settings to defaults?"):
            # Reset to defaults
            self.source_type_var.set("RTSP")
            self.sensitivity_var.set(0.5)
            self.min_green_var.set(10)
            self.detection_areas = {i: [] for i in range(4)}
            
            # Clear camera sources
            for i in range(4):
                self.camera_entries[i]['source_entry'].delete(0, tk.END)
                self.update_camera_status(i, "reset", "Not tested")
            
            self.update_source_type()
            self.update_areas_list()
            
            messagebox.showinfo("Success", "Settings reset to defaults!")


def main():
    root = tk.Tk()
    app = SimplifiedTrafficSettings(root)
    root.mainloop()

if __name__ == "__main__":
    main()