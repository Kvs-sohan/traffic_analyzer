"""
Vehicle Detection Module using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import traceback
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

class VehicleDetector:
    """Vehicle detection using YOLO model"""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.vehicle_weights = {
            'car': 1.0,
            'truck': 2.5,
            'bus': 2.0,
            'motorcycle': 0.5,
            'bicycle': 0.3
        }
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            import torch
            
            logging.info("Loading YOLO model...")
            # Use specific version to avoid compatibility issues
            self.model = torch.hub.load('ultralytics/yolov5:v7.0', 'yolov5s', pretrained=True, trust_repo=True)
            
            # Set model parameters
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.classes = [2, 3, 5, 7]  # Filter for vehicle classes
            self.model.max_det = 50  # Maximum detections per image
            
            # Verify model loaded successfully
            if self.model is not None:
                self.model_loaded = True
                logging.info("YOLO model loaded successfully")
            else:
                raise RuntimeError("Failed to load YOLO model")
                
        except Exception as e:
            logging.error(f"❌ Error loading YOLO model: {str(e)}")
            logging.error(f"Detailed error: {traceback.format_exc()}")
            logging.warning("Using simulation mode for detection")
            self.model = None
            self.model_loaded = False
    
    def detect_vehicles_in_area(self, frame: np.ndarray, area_points: List[Tuple[int, int]]) -> Tuple[int, float, Optional[np.ndarray]]:
        """
        Detect vehicles in the specified area
        
        Args:
            frame: Input frame
            area_points: List of points defining detection area
            
        Returns:
            Tuple of (vehicle_count, traffic_weight, processed_frame)
        """
        if frame is None:
            logging.warning("Received None frame in detect_vehicles_in_area")
            return 0, 0.0, None
        
        try:
            # Verify frame is valid
            if frame.size == 0 or len(frame.shape) != 3:
                logging.warning(f"Invalid frame shape: {frame.shape}")
                return 0, 0.0, frame
            
            # Create mask for the defined area
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            area_points_np = np.array(area_points, dtype=np.int32)
            cv2.fillPoly(mask, [area_points_np], 255)
            
            # Process frame
            processed_frame = frame.copy()
            
            # Draw detection area with semi-transparency
            overlay = processed_frame.copy()
            cv2.fillPoly(overlay, [area_points_np], (0, 255, 0, 128))
            cv2.addWeighted(overlay, 0.3, processed_frame, 0.7, 0, processed_frame)
            cv2.polylines(processed_frame, [area_points_np], True, (0, 255, 0), 2)
            
            if self.model_loaded and self.model:
                # Real YOLO detection
                vehicle_count, traffic_weight = self._yolo_detection(processed_frame, area_points, mask)
            else:
                # Simulation detection
                vehicle_count, traffic_weight = self._simulate_detection(processed_frame, area_points, mask)
            
            # Add text overlay with detection info
            cv2.putText(processed_frame, f"Vehicles: {vehicle_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Traffic Weight: {traffic_weight:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return vehicle_count, traffic_weight, processed_frame
            
        except Exception as e:
            logging.error(f"Error in vehicle detection: {str(e)}")
            logging.error(f"Detailed error: {traceback.format_exc()}")
            return 0, 0.0, frame
    
    def _yolo_detection(self, frame: np.ndarray, area_points: List[Tuple[int, int]], mask: np.ndarray) -> Tuple[int, float]:
        """
        Real YOLO-based vehicle detection
        
        Args:
            frame: Input frame
            area_points: List of points defining detection area
            mask: Binary mask for detection area
            
        Returns:
            Tuple of (vehicle_count, traffic_weight)
        """
        try:
            results = self.model(frame, verbose=False)
            
            vehicle_count = 0
            traffic_weight = 0.0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.model.names[cls].lower()
                        
                        # Check if it's a vehicle and confidence is high enough
                        if class_name in self.vehicle_classes and conf > 0.5:
                            # Check if center point is in detection area
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            if self._point_in_polygon((center_x, center_y), area_points):
                                vehicle_count += 1
                                traffic_weight += self.vehicle_weights.get(class_name, 1.0)
                                
                                # Draw bounding box with class name and confidence
                                color = (0, 255, 0)  # Green for detected vehicles
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                label = f"{class_name}: {conf:.2f}"
                                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Draw center point
                                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
            
            return vehicle_count, traffic_weight
            
        except Exception as e:
            logging.error(f"Error in YOLO detection: {str(e)}")
            logging.error(f"Detailed error: {traceback.format_exc()}")
            return self._simulate_detection(frame, area_points, mask)
    
    def _simulate_detection(self, frame: np.ndarray, area_points: List[Tuple[int, int]], mask: np.ndarray) -> Tuple[int, float]:
        """
        Simulate vehicle detection when YOLO model is not available
        
        Args:
            frame: Input frame
            area_points: List of points defining detection area
            mask: Binary mask for detection area
            
        Returns:
            Tuple of (vehicle_count, traffic_weight)
        """
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply mask
            masked = cv2.bitwise_and(gray, gray, mask=mask)
            
            # Simple motion detection
            blurred = cv2.GaussianBlur(masked, (21, 21), 0)
            thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = 500
            vehicle_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # Draw contours
            cv2.drawContours(frame, vehicle_contours, -1, (0, 255, 0), 2)
            
            # Simulate traffic weight based on contour areas
            total_area = sum(cv2.contourArea(c) for c in vehicle_contours)
            traffic_weight = total_area / 10000  # Normalize
            
            return len(vehicle_contours), min(traffic_weight, 10.0)
            
        except Exception as e:
            logging.error(f"Error in simulation detection: {str(e)}")
            return 0, 0.0
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def detect_vehicles(self, frame):
        """
        Backward compatibility method - redirects to detect_vehicles_in_area
        """
        # Use full frame area as detection area
        height, width = frame.shape[:2]
        area_points = [(0, 0), (width, 0), (width, height), (0, height)]
        
        count, weight, processed_frame = self.detect_vehicles_in_area(frame, area_points)
        
        return [{'class': 'car', 'confidence': 1.0}] * count  # Return dummy detections for compatibility