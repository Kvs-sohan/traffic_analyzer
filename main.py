"""
Smart Traffic Management System
Main application entry point
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import traceback
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.layout import TrafficUI
from core.traffic_manager import SmartTrafficManager
from core.database import TrafficDatabase
from utils.helpers import setup_logging

def main():
    """Main application entry point"""
    try:
        # Setup logging
        logger = setup_logging()
        logger.info("Starting Traffic Management System...")
        
        # Initialize database
        logger.info("Initializing database...")
        db_manager = TrafficDatabase()
        
        # Initialize traffic manager
        logger.info("Initializing traffic manager...")
        traffic_manager = SmartTrafficManager()
        
        # Create main window and initialize UI
        logger.info("Creating main window...")
        root = tk.Tk()
        
        # Set window properties
        root.title("Smart Traffic Management System")
        root.geometry("1400x900")
        root.minsize(1200, 800)
        
        # Center window on screen
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f'+{x}+{y}')
        
        # Initialize UI with the root window
        logger.info("Initializing UI components...")
        app = TrafficUI(root, traffic_manager)
        
        # Start the application
        logger.info("Starting main application loop...")
        app.run()
        
    except Exception as e:
        error_msg = f"Failed to start application:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logging.error(error_msg)
        messagebox.showerror("Error", error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()