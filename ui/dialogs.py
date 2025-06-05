class AreaLoadDialog(tk.Toplevel):
    """Dialog for configuring detection areas"""
    
    def __init__(self, parent, traffic_manager):
        super().__init__(parent)
        self.traffic_manager = traffic_manager
        self.current_signal = 0  # Start with Signal A
        self.points = []
        self.drawing = False
        
        # Configure dialog
        self.title("Configure Detection Areas")
        self.geometry("800x600")
        
        # Create top control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Signal selection on left
        signal_frame = ttk.Frame(control_frame)
        signal_frame.pack(side=tk.LEFT)
        
        ttk.Label(signal_frame, text="Configuring:").pack(side=tk.LEFT)
        self.signal_label = ttk.Label(signal_frame, text="Signal A", font=("Arial", 12, "bold"))
        self.signal_label.pack(side=tk.LEFT, padx=5)
        
        # Buttons on right
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)
        
        save_style = ttk.Style()
        save_style.configure("Green.TButton", foreground="green")
        
        self.save_btn = ttk.Button(button_frame, text="Save Area", style="Green.TButton",
                                 command=self.save_current_area)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Reset", command=self.reset_points).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.close_dialog).pack(side=tk.LEFT, padx=5)
        
        # Status message
        self.status_var = tk.StringVar()
        self.status_var.set("Left click to add points, right click to remove last point")
        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self, width=640, height=480, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        # Load existing area if available
        self.load_current_area()
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", self.remove_last_point)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
    
    def load_current_area(self):
        """Load existing area for current signal"""
        self.points = []
        area = self.traffic_manager.area_manager.get_area(self.current_signal)
        if area:
            self.points = area.copy()
            self.draw_area()
            self.status_var.set(f"Loaded existing area for Signal {chr(65 + self.current_signal)}")
        else:
            self.status_var.set(f"Configure new area for Signal {chr(65 + self.current_signal)}")
    
    def save_current_area(self):
        """Save current area and move to next signal"""
        if len(self.points) < 3:
            self.status_var.set("⚠️ Need at least 3 points to define an area!")
            return
        
        # Save current area
        success = self.traffic_manager.area_manager.set_area(
            self.current_signal, 
            self.points,
            f"Signal {chr(65 + self.current_signal)}"
        )
        
        if success:
            # Move to next signal or close if done
            self.current_signal += 1
            if self.current_signal < 4:
                self.signal_label.config(text=f"Signal {chr(65 + self.current_signal)}")
                self.points = []
                self.draw_area()
                self.status_var.set(f"Configure area for Signal {chr(65 + self.current_signal)}")
            else:
                messagebox.showinfo("Success", "All areas configured successfully!")
                self.close_dialog()
        else:
            self.status_var.set("❌ Failed to save area configuration")
    
    def add_point(self, event):
        """Add point to current area"""
        self.points.append([event.x, event.y])
        self.draw_area()
    
    def remove_last_point(self, event):
        """Remove last point from current area"""
        if self.points:
            self.points.pop()
            self.draw_area()
    
    def reset_points(self):
        """Reset current area points"""
        self.points = []
        self.draw_area()
        self.status_var.set("Area reset - start adding new points")
    
    def draw_area(self):
        """Draw current area on canvas"""
        self.canvas.delete("all")
        
        # Draw points
        for x, y in self.points:
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red")
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points) - 1):
                x1, y1 = self.points[i]
                x2, y2 = self.points[i + 1]
                self.canvas.create_line(x1, y1, x2, y2, fill="yellow")
            
            # Close the polygon if we have at least 3 points
            if len(self.points) >= 3:
                x1, y1 = self.points[-1]
                x2, y2 = self.points[0]
                self.canvas.create_line(x1, y1, x2, y2, fill="yellow")
    
    def close_dialog(self):
        """Close the dialog"""
        self.grab_release()
        self.destroy() 