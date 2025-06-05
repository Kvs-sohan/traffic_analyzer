#!/usr/bin/env python3
"""
Smart Traffic Management System
Main application entry point
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.layout import VideoPanel
from core.traffic_manager import SmartTrafficManager
from core.database import TrafficDatabase
from utils.helpers import setup_logging

def main():
    """Main application entry point"""
    try:
        # Setup logging
        setup_logging()
        
        # Initialize database
        db_manager = TrafficDatabase()
        
        # Initialize traffic manager
        traffic_manager = SmartTrafficManager(db_manager)
        
        # Create main window
        root = tk.Tk()
        app = VideoPanel(root, traffic_manager)
        
        # Set window properties
        root.title("Smart Traffic Management System")
        root.geometry("1400x900")
        root.minsize(1200, 800)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f'+{x}+{y}')
        
        # Start the application
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()