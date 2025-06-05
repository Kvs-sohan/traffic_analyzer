"""
Main Traffic Management System
"""

import cv2
import numpy as np
import threading
import time
import json
import os
import logging
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from .detector import VehicleDetector
from .signal import SignalController
from .database import TrafficDatabase
from .camera import CameraManager
from .area_manager import AreaManager

class CameraManager:
    """Handles camera/video source management"""
    
    def __init__(self):
        self.sources = []
        self.captures = []
        self.rtsp_urls = []
        
    def load_sources(self, sources_file="config/sources.json"):
        """Load camera sources from config file"""
        try:
            if os.path.exists(sources_file):
                with open(sources_file, 'r') as f:
                    config = json.load(f)
                    self.sources = config.get('sources', [])
                    self.rtsp_urls = config.get('rtsp_urls', [])
                logging.info(f"Loaded {len(self.sources)} camera sources")
            else:
                # Default sources
                self.sources = [
                    "data/videos/signal_0.mp4",
                    "data/videos/signal_1.mp4", 
                    "data/videos/signal_2.mp4",
                    "data/videos/signal_3.mp4"
                ]
                # Create default video files if they don't exist
                self._create_default_videos()
                self.save_sources(sources_file)
                logging.info("Created default camera sources")
        except Exception as e:
            logging.error("Error loading camera sources: %s", str(e))
            raise RuntimeError(f"Failed to load camera sources: {str(e)}")
    
    def initialize_captures(self):
        """Initialize video captures for all sources"""
        try:
            # Release any existing captures
            self.release_all()
            self.captures = []
            
            # Initialize new captures
            for source in self.sources:
                try:
                    if isinstance(source, str) and source.isdigit():
                        cap = cv2.VideoCapture(int(source))
                    else:
                        cap = cv2.VideoCapture(source)
                    
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video source: {source}")
                    
                    # Set capture properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    self.captures.append(cap)
                    logging.info(f"Initialized capture for source: {source}")
                except Exception as e:
                    logging.error(f"Error initializing capture for source {source}: {str(e)}")
                    self.captures.append(None)
            
            if not any(self.captures):
                raise RuntimeError("No video sources could be initialized")
                
            logging.info(f"Initialized {len([c for c in self.captures if c is not None])} captures")
            
        except Exception as e:
            logging.error("Error initializing video captures: %s", str(e))
            raise
    
    def release_all(self):
        """Release all video captures"""
        try:
            for cap in self.captures:
                if cap is not None:
                    cap.release()
            self.captures = []
            logging.info("Released all video captures")
        except Exception as e:
            logging.error("Error releasing video captures: %s", str(e))
    
    def _create_default_videos(self):
        """Create default video files if they don't exist"""
        try:
            # Ensure data/videos directory exists
            os.makedirs("data/videos", exist_ok=True)
            
            # Create a blank video file for each signal if it doesn't exist
            for i in range(4):
                video_path = f"data/videos/signal_{i}.mp4"
                if not os.path.exists(video_path):
                    # Create a blank video file with a black frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Add some text to the frame
                    cv2.putText(frame, f"Signal {chr(65+i)}", (220, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
                    for _ in range(30):  # 1 second of video
                        out.write(frame)
                    out.release()
            logging.info("Created default video files")
        except Exception as e:
            logging.error("Error creating default videos: %s", str(e))
            raise
    
    def save_sources(self, sources_file="config/sources.json"):
        """Save camera sources to config file"""
        try:
            os.makedirs(os.path.dirname(sources_file), exist_ok=True)
            config = {
                'sources': self.sources,
                'rtsp_urls': self.rtsp_urls
            }
            with open(sources_file, 'w') as f:
                json.dump(config, f, indent=2)
            logging.info("Saved camera sources configuration")
        except Exception as e:
            logging.error("Error saving camera sources: %s", str(e))
            raise
    
    def get_frame(self, camera_id):
        """Get frame from specific camera"""
        if 0 <= camera_id < len(self.captures) and self.captures[camera_id]:
            ret, frame = self.captures[camera_id].read()
            if ret:
                return frame
            else:
                # Try to restart capture for video files
                source = self.sources[camera_id]
                if not source.startswith('rtsp://') and not source.startswith('http://'):
                    self.captures[camera_id].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.captures[camera_id].read()
                    if ret:
                        return frame
        
        # Return demo frame if no camera available
        return self._generate_demo_frame(camera_id)
    
    def _generate_demo_frame(self, camera_id):
        """Generate demo frame for testing"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Background
        cv2.rectangle(frame, (0, 0), (640, 480), (50, 50, 50), -1)
        
        # Road
        cv2.rectangle(frame, (0, 200), (640, 280), (70, 70, 70), -1)
        cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 2)
        
        # Simulate vehicles
        num_vehicles = np.random.randint(0, 6)
        for i in range(num_vehicles):
            x = np.random.randint(50, 590)
            y = np.random.randint(210, 270)
            cv2.rectangle(frame, (x, y), (x+40, y+20), (0, 100, 255), -1)
        
        # Signal info
        cv2.putText(frame, f"Signal {chr(65 + camera_id)} - Demo", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (10, 460), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

class AreaManager:
    """Manages detection areas configuration"""
    
    def __init__(self):
        self.areas = {}
        self.load_areas()
    
    def load_areas(self, areas_file="config/areas.json"):
        """Load detection areas from config file"""
        try:
            if os.path.exists(areas_file):
                with open(areas_file, 'r') as f:
                    self.areas = json.load(f)
                print(f"✅ Loaded areas for {len(self.areas)} signals")
                return True
            else:
                # Default areas for each signal
                self.areas = {
                    "0": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal A"},
                    "1": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal B"},
                    "2": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal C"},
                    "3": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal D"}
                }
                self.save_areas(areas_file)
                return True
        except Exception as e:
            print(f"❌ Error loading areas: {e}")
            # Set default areas if loading fails
            self.areas = {
                "0": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal A"},
                "1": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal B"},
                "2": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal C"},
                "3": {"points": [[50, 200], [590, 200], [590, 280], [50, 280]], "name": "Signal D"}
            }
            return False
    
    def save_areas(self, areas_file="config/areas.json"):
        """Save detection areas to config file"""
        try:
            os.makedirs(os.path.dirname(areas_file), exist_ok=True)
            with open(areas_file, 'w') as f:
                json.dump(self.areas, f, indent=2)
            print("✅ Saved detection areas configuration")
            return True
        except Exception as e:
            print(f"❌ Error saving areas: {e}")
            return False
    
    def get_area(self, signal_id):
        """Get detection area for specific signal"""
        area_info = self.areas.get(str(signal_id))
        if area_info:
            return area_info["points"]
        return None
    
    def set_area(self, signal_id, points, name=None):
        """Set detection area for specific signal"""
        self.areas[str(signal_id)] = {
            "points": points,
            "name": name or f"Signal {chr(65 + signal_id)}"
        }
        return self.save_areas()  # Save immediately when area is updated

class TrafficAnalyzer:
    """Handles traffic analysis and metrics calculation"""
    
    def __init__(self):
        self.vehicle_history = {i: deque(maxlen=60) for i in range(4)}  # Last 60 seconds
        self.traffic_weights = {
            'car': 1.0,
            'truck': 2.0,
            'bus': 2.5,
            'motorcycle': 0.5,
            'bicycle': 0.3
        }
    
    def calculate_traffic_weight(self, detections):
        """Calculate weighted traffic metric based on vehicle types"""
        total_weight = 0.0
        for detection in detections:
            vehicle_type = detection.get('class', 'car')
            weight = self.traffic_weights.get(vehicle_type, 1.0)
            confidence = detection.get('confidence', 1.0)
            total_weight += weight * confidence
        return total_weight
    
    def update_history(self, signal_id, vehicle_count, traffic_weight):
        """Update traffic history for analysis"""
        timestamp = time.time()
        self.vehicle_history[signal_id].append({
            'timestamp': timestamp,
            'count': vehicle_count,
            'weight': traffic_weight
        })
    
    def get_traffic_trend(self, signal_id, duration=30):
        """Get traffic trend for last N seconds"""
        current_time = time.time()
        history = self.vehicle_history[signal_id]
        
        recent_data = [
            entry for entry in history 
            if current_time - entry['timestamp'] <= duration
        ]
        
        if not recent_data:
            return 0.0
        
        avg_weight = sum(entry['weight'] for entry in recent_data) / len(recent_data)
        return avg_weight
    
    def calculate_efficiency_score(self, green_time, traffic_weight, vehicle_count):
        """Calculate efficiency score for green phase"""
        if green_time <= 0:
            return 0.0
        
        # Base efficiency on vehicles processed per second
        vehicles_per_second = vehicle_count / green_time
        weight_per_second = traffic_weight / green_time
        
        # Normalize to 0-100 scale
        efficiency = min(100, (vehicles_per_second * 20) + (weight_per_second * 10))
        return round(efficiency, 2)

class SmartTrafficManager:
    """Main traffic management system coordinating all components"""
    
    def __init__(self):
        try:
            logging.info("Initializing Smart Traffic Manager...")
            
            # Initialize core components
            self.camera_manager = CameraManager()
            logging.info("Camera Manager initialized")
            
            self.area_manager = AreaManager()
            logging.info("Area Manager initialized")
            
            self.detector = VehicleDetector()
            logging.info("Vehicle Detector initialized")
            
            self.signal_controller = SignalController()
            logging.info("Signal Controller initialized")
            
            self.database = TrafficDatabase()
            logging.info("Traffic Database initialized")
            
            self.analyzer = TrafficAnalyzer()
            logging.info("Traffic Analyzer initialized")
            
            # System state
            self.running = False
            self.processing_threads = []
            self.current_frames = [None] * 4
            self.current_detections = [[] for _ in range(4)]
            self.current_metrics = [{
                'vehicle_count': 0,
                'traffic_weight': 0.0,
                'efficiency_score': 0.0
            } for _ in range(4)]
            
            # Initialize signals list
            self.signals = self.signal_controller.signals
            if not self.signals:
                raise RuntimeError("Failed to initialize signals")
            
            # Signal sequence control
            self.current_signal_index = 0  # Start with Signal A
            self.yellow_start_time = None
            self.yellow_duration = 3  # 3 seconds yellow time
            self.next_signal_green_time = None  # Store calculated green time
            self.active_detection_signal = 0  # Track which signal is being analyzed by YOLO
            self.cycle_start_time = None  # Track when current signal phase started
            
            # Performance tracking
            self.fps_counters = [0] * 4
            
            # Initialize components
            self._initialize_system()
            
        except Exception as e:
            logging.error(f"Failed to initialize Smart Traffic Manager: {str(e)}")
            raise

    def _initialize_system(self):
        """Initialize all system components"""
        try:
            self.camera_manager.load_sources()
            self.camera_manager.initialize_captures()
            logging.info("Traffic Management System components initialized successfully")
        except Exception as e:
            logging.error("System initialization error: %s", str(e))
            raise

    def get_historical_data(self, hours=24, signal_id=None):
        """Get historical traffic data for analysis and visualization"""
        try:
            # Use existing analytics data method
            duration = hours * 3600  # Convert hours to seconds
            analytics = self.get_analytics_data(duration)
        
            # Format for UI compatibility
            historical_data = {
                'status': 'success',
                'time_range': f'Last {hours} hours',
                'traffic_trends': analytics.get('traffic_trends', []),
                'signal_efficiency': analytics.get('signal_efficiency', []),
                'total_vehicles': analytics.get('total_vehicles', 0),
                'peak_hours': analytics.get('peak_hours', [])
            }
        
            # Filter by signal if specified
            if signal_id is not None:
                historical_data['traffic_trends'] = [
                    trend for trend in historical_data['traffic_trends'] 
                    if trend.get('signal_id') == signal_id
                ]
                historical_data['signal_efficiency'] = [
                    eff for eff in historical_data['signal_efficiency'] 
                    if eff.get('signal_id') == signal_id
                ]
        
            return historical_data
        
        except Exception as e:
            print(f"❌ Error getting historical data: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'traffic_trends': [],
                'signal_efficiency': [],
                'total_vehicles': 0,
                'peak_hours': []
            }
    
    def start_system(self):
        """Start the traffic management system"""
        try:
            if self.running:
                return
            
            self.running = True
            logging.info("Starting traffic management system...")
            
            # Initialize first signal (Signal A) as GREEN
            self.signals[0].set_state('GREEN', self.signals[0].default_green_time)
            self.current_signal_index = 0
            self.active_detection_signal = 1  # Start analyzing Signal B
            self.cycle_start_time = time.time()
            
            # Start processing threads for each signal
            for i in range(4):
                thread = threading.Thread(
                    target=self._process_signal,
                    args=(i,),
                    daemon=True
                )
                thread.start()
                self.processing_threads.append(thread)
            
            # Start signal control thread
            control_thread = threading.Thread(
                target=self._control_signals,
                daemon=True
            )
            control_thread.start()
            self.processing_threads.append(control_thread)
            
            logging.info("Traffic management system started successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start system: {str(e)}")
            self.running = False
            return False
    
    def stop_system(self):
        """Stop the traffic management system"""
        try:
            if not self.running:
                return
            
            self.running = False
            logging.info("Stopping traffic management system...")
            
            # Wait for threads to finish
            for thread in self.processing_threads:
                thread.join(timeout=1.0)
            
            self.processing_threads.clear()
            logging.info("Traffic management system stopped successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping system: {str(e)}")
            return False
    
    def _process_signal(self, signal_id):
        """Process signal video feed and detect vehicles"""
        while self.running:
            try:
                # Only process YOLO detection for the active detection signal
                if signal_id != self.active_detection_signal:
                    time.sleep(0.1)
                    continue
                
                # Get frame from camera
                frame = self.camera_manager.get_frame(signal_id)
                if frame is None:
                    continue
                
                # Store current frame
                self.current_frames[signal_id] = frame.copy()
                
                # Get detection area
                area = self.area_manager.get_area(signal_id)
                if not area:
                    # Use default area if none defined
                    height, width = frame.shape[:2]
                    area = [(0, 0), (width, 0), (width, height), (0, height)]
                
                # Run vehicle detection
                vehicle_count, traffic_weight, processed_frame = self.detector.detect_vehicles_in_area(frame, area)
                
                # Store detections for analytics
                self.current_detections[signal_id] = [{'class': 'car', 'confidence': 1.0}] * vehicle_count
                
                # Store current frame with detections
                if processed_frame is not None:
                    self.current_frames[signal_id] = processed_frame
                
                # Update metrics
                self.current_metrics[signal_id].update({
                    'vehicle_count': vehicle_count,
                    'traffic_weight': traffic_weight
                })
                
                # If current signal is yellow, calculate next green time
                current_signal = self.signals[self.current_signal_index]
                if (current_signal.current_state == 'YELLOW' and 
                    signal_id == (self.current_signal_index + 1) % 4):
                    # Calculate green time based on detected vehicles
                    self.next_signal_green_time = self._calculate_green_time(vehicle_count, traffic_weight)
                    logging.info(f"Calculated green time for next signal: {self.next_signal_green_time}s")
                
                # Update FPS counter
                self.fps_counters[signal_id] += 1
                
                # Small sleep to prevent CPU overload
                time.sleep(0.01)
                
            except Exception as e:
                logging.error(f"Error processing signal {signal_id}: {str(e)}")
                time.sleep(1.0)
    
    def _control_signals(self):
        """Control traffic signals based on traffic conditions"""
        while self.running:
            try:
                current_time = time.time()
                current_signal = self.signals[self.current_signal_index]
                
                # Handle yellow signal transition
                if current_signal.current_state == 'YELLOW':
                    if current_time - self.yellow_start_time >= self.yellow_duration:
                        # Yellow phase complete, switch to next signal
                        current_signal.set_state('RED')
                        
                        # Move to next signal
                        self.current_signal_index = (self.current_signal_index + 1) % 4
                        next_signal = self.signals[self.current_signal_index]
                        
                        # Use pre-calculated green time from YOLO detection
                        green_time = self.next_signal_green_time or next_signal.default_green_time
                        
                        # Set next signal to green
                        next_signal.set_state('GREEN', green_time)
                        logging.info(f"Signal {self.current_signal_index} turned GREEN for {green_time} seconds")
                        
                        # Reset next signal green time
                        self.next_signal_green_time = None
                        
                        # Update active detection signal to next in sequence
                        self.active_detection_signal = (self.current_signal_index + 1) % 4
                
                # Handle green signal completion
                elif current_signal.current_state == 'GREEN' and current_signal.remaining_time <= 0:
                    # Switch to yellow
                    current_signal.set_state('YELLOW', self.yellow_duration)
                    self.yellow_start_time = current_time
                    logging.info(f"Signal {self.current_signal_index} turned YELLOW")
                
                # Update remaining times
                if current_signal.current_state == 'GREEN':
                    elapsed = current_time - self.cycle_start_time if hasattr(self, 'cycle_start_time') else 0
                    current_signal.remaining_time = max(0, current_signal.remaining_time - elapsed)
                    self.cycle_start_time = current_time
                
                time.sleep(0.1)  # Update every 100ms
                
            except Exception as e:
                logging.error(f"Error in signal control: {str(e)}")
                time.sleep(1.0)
    
    def get_current_frame(self, signal_id):
        """Get the current frame for a signal"""
        try:
            if 0 <= signal_id < len(self.current_frames):
                return self.current_frames[signal_id]
        except Exception as e:
            logging.error(f"Error getting frame for signal {signal_id}: {str(e)}")
        return None
    
    def get_current_detections(self, signal_id):
        """Get current detections for specific signal"""
        if 0 <= signal_id < 4:
            return self.current_detections[signal_id]
        return []
    
    def get_current_metrics(self, signal_id):
        """Get current metrics for specific signal"""
        if 0 <= signal_id < 4:
            return self.current_metrics[signal_id]
        return {}
    
    def get_signal(self, signal_id):
        """Get current signal state"""
        return self.signal_controller.get_signal(signal_id)
    
    def get_system_status(self):
        """Get overall system status"""
        return {
            'running': self.running,
            'active_cameras': sum(1 for cap in self.camera_manager.captures if cap),
            'total_cameras': len(self.camera_manager.sources),
            'fps': self.fps_counters,
            'uptime': time.time() - getattr(self, 'start_time', time.time())
        }
    
    def get_analytics_data(self, duration=3600):
        """Get analytics data for dashboard"""
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=duration)
        
        # Get data from database
        df = self.database.get_analytics_data(start_time, end_time)
        
        if df.empty:
            return pd.DataFrame()
        
        # Process data for visualization
        analytics = {
            'traffic_trends': [],
            'signal_efficiency': [],
            'peak_hours': [],
            'total_vehicles': 0
        }
        
        # Group by signal and time
        for signal_id in range(4):
            signal_data = df[df['signal_id'] == signal_id]
            
            if not signal_data.empty:
                # Traffic trend
                trend = {
                    'signal_id': signal_id,
                    'timestamps': [row['timestamp'] for _, row in signal_data.iterrows()],
                    'vehicle_counts': [row['vehicle_count'] for _, row in signal_data.iterrows()],
                    'traffic_weights': [row['traffic_weight'] for _, row in signal_data.iterrows()]
                }
                analytics['traffic_trends'].append(trend)
                
                # Efficiency
                avg_efficiency = signal_data['efficiency_score'].mean()
                analytics['signal_efficiency'].append({
                    'signal_id': signal_id,
                    'efficiency': avg_efficiency
                })
                
                # Total vehicles
                analytics['total_vehicles'] += signal_data['vehicle_count'].sum()
        
        return df
    
    def export_data(self, start_date=None, end_date=None, format='csv'):
        """Export traffic data to file"""
        try:
            return self.database.export_data(start_date, end_date, format)
        except Exception as e:
            print(f"❌ Export error: {e}")
            return None
    
    def load_camera_sources(self, sources):
        """Load new camera sources"""
        try:
            # Stop current processing
            was_running = self.running
            if was_running:
                self.stop_system()
            
            # Update sources
            self.camera_manager.sources = sources
            self.camera_manager.save_sources()
            self.camera_manager.initialize_captures()
            
            # Restart if was running
            if was_running:
                self.start_system()
            
            return True
        except Exception as e:
            print(f"❌ Error loading camera sources: {e}")
            return False
    
    def update_detection_area(self, signal_id, points):
        """Update detection area for specific signal"""
        try:
            self.area_manager.set_area(signal_id, points)
            return True
        except Exception as e:
            print(f"❌ Error updating detection area: {e}")
            return False
    
    def get_detection_areas(self):
        """Get all detection areas"""
        return self.area_manager.areas
    
    def calibrate_detector(self, confidence_threshold=0.5, nms_threshold=0.4):
        """Calibrate vehicle detector parameters"""
        try:
            self.detector.set_confidence_threshold(confidence_threshold)
            self.detector.set_nms_threshold(nms_threshold)
            return True
        except Exception as e:
            print(f"❌ Error calibrating detector: {e}")
            return False
    
    def get_live_statistics(self):
        """Get live statistics for all signals"""
        try:
            # Get current data from database (last 5 minutes)
            df = self.get_analytics_data(duration=300)  # 5 minutes
            
            # Initialize response structure
            response = {
                'signals': []
            }
            
            # Process each signal
            for signal_id in range(4):
                signal_data = df[df['signal_id'] == signal_id]
                
                if not signal_data.empty:
                    latest_data = signal_data.iloc[-1]
                    metrics = {
                        'vehicle_count': int(latest_data['vehicle_count']),
                        'traffic_weight': float(latest_data['traffic_weight']),
                        'efficiency_score': float(latest_data['efficiency_score'])
                    }
                    
                    state = {
                        'state': 'GREEN' if latest_data['green_time'] > 0 else 'RED',
                        'remaining_time': int(latest_data['green_time'])
                    }
                else:
                    # Default values if no data available
                    metrics = {
                        'vehicle_count': 0,
                        'traffic_weight': 0.0,
                        'efficiency_score': 0.0
                    }
                    state = {
                        'state': 'RED',
                        'remaining_time': 0
                    }
                
                response['signals'].append({
                    'metrics': metrics,
                    'state': state
                })
            
            return response
            
        except Exception as e:
            print(f"Error getting live statistics: {str(e)}")
            return {'signals': []}
    
    def optimize_signal_timing(self, enable_optimization=True):
        """Enable/disable adaptive signal timing optimization"""
        try:
            self.signal_controller.set_adaptive_mode(enable_optimization)
            return True
        except Exception as e:
            print(f"❌ Error setting optimization mode: {e}")
            return False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'running') and self.running:
            self.stop_system()

    def get_analytics_data(self):
        """Get analytics data for UI display"""
        try:
            analytics = {
                'total_vehicles': 0,
                'peak_time': None,
                'peak_count': 0,
                'signals': []
            }
            
            # Get data for each signal
            for signal_id in range(4):
                signal_data = self.get_signal(signal_id)
                traffic_data = self.get_traffic_data(signal_id)
                
                # Get signal statistics
                stats = self.database.get_signal_statistics(signal_id)
                
                signal_analytics = {
                    'signal_id': signal_id,
                    'total_vehicles': stats.get('total_cycles', 0),
                    'avg_weight': stats.get('avg_traffic_weight', 0.0),
                    'efficiency': stats.get('avg_efficiency', 0.0),
                    'current_state': signal_data.get('state', 'RED'),
                    'remaining_time': signal_data.get('remaining_time', 0)
                }
                
                analytics['signals'].append(signal_analytics)
                analytics['total_vehicles'] += signal_analytics['total_vehicles']
                
                # Update peak count
                if stats.get('max_vehicles', 0) > analytics['peak_count']:
                    analytics['peak_count'] = stats['max_vehicles']
            
            # Get peak time from database
            df = self.database.get_analytics_data()
            if not df.empty:
                peak_row = df.loc[df['vehicle_count'].idxmax()]
                analytics['peak_time'] = pd.to_datetime(peak_row['timestamp'])
            
            return analytics
            
        except Exception as e:
            logging.error(f"Error getting analytics data: {str(e)}")
            return {}
    
    def get_traffic_data(self, signal_id):
        """Get current traffic data for a signal"""
        try:
            if not 0 <= signal_id < len(self.signals):
                raise ValueError(f"Invalid signal ID: {signal_id}")
            
            # Get current frame
            frame = self.current_frames[signal_id]
            if frame is None:
                return {'count': 0, 'weight': 0.0}
            
            # Get detection area
            area = self.area_manager.get_area(signal_id)
            if not area:
                return {'count': 0, 'weight': 0.0}
            
            # Detect vehicles
            count, weight, _ = self.detector.detect_vehicles_in_area(frame, area)
            
            return {
                'count': count,
                'weight': weight
            }
            
        except Exception as e:
            logging.error(f"Error getting traffic data: {str(e)}")
            return {'count': 0, 'weight': 0.0}

    def _calculate_green_time(self, vehicle_count, traffic_weight):
        """Calculate green signal time based on traffic conditions"""
        try:
            # Base time calculation
            base_time = 15  # Minimum green time
            
            # Add time based on vehicle count
            if vehicle_count > 0:
                vehicle_factor = min(vehicle_count * 2, 30)  # Up to 30 seconds for vehicles
                base_time += vehicle_factor
            
            # Add time based on traffic weight
            weight_factor = min(traffic_weight * 3, 15)  # Up to 15 seconds for weight
            base_time += weight_factor
            
            # Ensure time is within bounds
            min_green = 15
            max_green = 60
            green_time = max(min_green, min(base_time, max_green))
            
            return int(green_time)
            
        except Exception as e:
            logging.error(f"Error calculating green time: {str(e)}")
            return 30  # Default 30 seconds if calculation fails