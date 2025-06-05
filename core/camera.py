"""
Camera Manager Module
"""

import os
import json
import cv2
import numpy as np
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

class CameraManager:
    """Manages camera sources and video capture"""
    
    def __init__(self):
        self.sources = []
        self.captures = []
        self.frame_sizes = [(640, 480)] * 4  # Default frame sizes
        self.current_frames = [None] * 4
        self.running = False
        self.update_thread = None
        self.frame_locks = [threading.Lock() for _ in range(4)]
        self.load_sources()
    
    def load_sources(self, sources_file="config/sources.json"):
        """Load camera sources from config file"""
        try:
            config_path = Path(sources_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.sources = json.load(f)
                logging.info(f"✅ Loaded {len(self.sources)} camera sources")
            else:
                # Default demo sources
                demo_dir = Path("data/videos")
                demo_dir.mkdir(parents=True, exist_ok=True)
                
                self.sources = [
                    str(demo_dir / "signal_0.mp4"),
                    str(demo_dir / "signal_1.mp4"),
                    str(demo_dir / "signal_2.mp4"),
                    str(demo_dir / "signal_3.mp4")
                ]
                self.save_sources(sources_file)
                logging.info("Created default camera sources configuration")
        except Exception as e:
            logging.error(f"❌ Error loading sources: {str(e)}")
            self.sources = []
    
    def save_sources(self, sources_file="config/sources.json"):
        """Save camera sources to config file"""
        try:
            config_path = Path(sources_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.sources, f, indent=2)
            logging.info("✅ Saved camera sources configuration")
        except Exception as e:
            logging.error(f"❌ Error saving sources: {str(e)}")
    
    def initialize_captures(self):
        """Initialize video captures for all sources"""
        self.close_captures()  # Close any existing captures
        
        for i, source in enumerate(self.sources):
            if source:
                try:
                    # Try to convert to integer for USB cameras
                    try:
                        source_val = int(source)
                    except ValueError:
                        source_val = source
                    
                    cap = cv2.VideoCapture(source_val)
                    if not cap.isOpened():
                        logging.error(f"❌ Failed to open source: {source}")
                        cap = None
                    else:
                        # Set frame size
                        width, height = self.frame_sizes[i]
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        
                        # Verify we can read a frame
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            logging.error(f"❌ Could not read frame from source: {source}")
                            cap.release()
                            cap = None
                        else:
                            logging.info(f"✅ Opened source: {source} ({frame.shape[1]}x{frame.shape[0]})")
                            # Reset to start
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception as e:
                    logging.error(f"❌ Error initializing capture {i}: {str(e)}")
                    cap = None
            else:
                cap = None
            
            self.captures.append(cap)
    
    def close_captures(self):
        """Close all video captures"""
        for cap in self.captures:
            if cap is not None:
                cap.release()
        self.captures = []
    
    def start_capture(self):
        """Start capturing frames from all sources"""
        if self.running:
            return
        
        try:
            # Initialize captures
            self.initialize_captures()
            
            # Start update thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            
            logging.info("✅ Started camera capture")
            
        except Exception as e:
            logging.error(f"❌ Error starting capture: {str(e)}")
            self.stop_capture()
    
    def stop_capture(self):
        """Stop capturing frames"""
        self.running = False
        
        # Wait for update thread
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        # Close captures
        self.close_captures()
        
        # Clear current frames
        for i in range(len(self.current_frames)):
            with self.frame_locks[i]:
                self.current_frames[i] = None
        
        logging.info("✅ Stopped camera capture")
    
    def _update_loop(self):
        """Main update loop for capturing frames"""
        while self.running:
            try:
                for i, cap in enumerate(self.captures):
                    if cap is not None:
                        ret, frame = cap.read()
                        
                        if ret and frame is not None:
                            # Resize frame if needed
                            if frame.shape[:2] != self.frame_sizes[i]:
                                frame = cv2.resize(frame, self.frame_sizes[i])
                            
                            # Update current frame
                            with self.frame_locks[i]:
                                self.current_frames[i] = frame
                            
                            # Loop video if at end
                            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            # Try to reopen capture
                            cap.release()
                            self.captures[i] = None
                            logging.warning(f"Lost connection to source {i}, attempting to reconnect...")
                            self._reconnect_source(i)
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error in update loop: {str(e)}")
                time.sleep(1.0)  # Sleep on error to prevent spam
    
    def _reconnect_source(self, index):
        """Attempt to reconnect to a source"""
        try:
            if index < len(self.sources) and self.sources[index]:
                source = self.sources[index]
                try:
                    source_val = int(source)
                except ValueError:
                    source_val = source
                
                cap = cv2.VideoCapture(source_val)
                if cap.isOpened():
                    self.captures[index] = cap
                    logging.info(f"✅ Reconnected to source {index}")
                else:
                    logging.error(f"❌ Failed to reconnect to source {index}")
        except Exception as e:
            logging.error(f"❌ Error reconnecting to source {index}: {str(e)}")
    
    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """
        Get the current frame from a camera
        
        Args:
            index: Camera index (0-3)
            
        Returns:
            Current frame or None if not available
        """
        if 0 <= index < len(self.current_frames):
            with self.frame_locks[index]:
                frame = self.current_frames[index]
                if frame is not None:
                    return frame.copy()
        return None
    
    def set_frame_size(self, index: int, width: int, height: int):
        """Set frame size for a camera"""
        if 0 <= index < len(self.frame_sizes):
            self.frame_sizes[index] = (width, height)
            if index < len(self.captures) and self.captures[index] is not None:
                cap = self.captures[index]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def get_active_sources(self) -> List[int]:
        """Get list of active camera indices"""
        return [i for i, cap in enumerate(self.captures) if cap is not None]
    
    def get_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return {
            'sources': self.sources,
            'frame_sizes': self.frame_sizes
        }
    
    def load_config(self, config: Dict[str, Any]):
        """Load camera configuration"""
        if 'sources' in config:
            self.sources = config['sources']
        if 'frame_sizes' in config:
            self.frame_sizes = config['frame_sizes']
    
    def __del__(self):
        """Clean up resources"""
        self.close_captures() 