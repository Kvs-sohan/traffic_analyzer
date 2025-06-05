"""
Vehicle Detection Module using YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

class VehicleDetector:
    def __init__(self):
        self.model = None
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
        """Load YOLO model for vehicle detection"""
        try:
            # Try to load YOLOv8 model
            self.model = YOLO('yolov8n.pt')  # Will download if not present
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            print("Using simulation mode for detection")
            self.model = None
    
    def detect_vehicles_in_area(self, frame, area_points):
        """
        Detect vehicles in the specified area
        Returns: vehicle_count, traffic_weight, processed_frame
        """
        if frame is None:
            return 0, 0.0, None
        
        try:
            # Create mask for the defined area
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            area_points_np = np.array(area_points, dtype=np.int32)
            cv2.fillPoly(mask, [area_points_np], 255)
            
            # Process frame
            processed_frame = frame.copy()
            
            # Draw detection area
            cv2.polylines(processed_frame, [area_points_np], True, (0, 255, 0), 2)
            
            if self.model:
                # Real YOLO detection
                vehicle_count, traffic_weight = self._yolo_detection(processed_frame, area_points, mask)
            else:
                # Simulation detection
                vehicle_count, traffic_weight = self._simulate_detection(processed_frame, area_points, mask)
            
            return vehicle_count, traffic_weight, processed_frame
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            return 0, 0.0, frame
    
    def _yolo_detection(self, frame, area_points, mask):
        """Real YOLO-based vehicle detection"""
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
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                                cv2.putText(frame, f"{class_name}: {conf:.2f}", 
                                          (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            return vehicle_count, traffic_weight
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return self._simulate_detection(frame, area_points, mask)
    
    def _simulate_detection(self, frame, area_points, mask):
        """Simulate vehicle detection for demo purposes"""
        try:
            # Generate random detections for demo
            vehicle_count = np.random.randint(0, 8)
            
            # Calculate traffic weight based on simulated vehicle types
            traffic_weight = 0.0
            for _ in range(vehicle_count):
                vehicle_type = np.random.choice(list(self.vehicle_weights.keys()))
                traffic_weight += self.vehicle_weights[vehicle_type]
                
                # Draw random bounding boxes for demo
                height, width = frame.shape[:2]
                x = np.random.randint(50, width - 100)
                y = np.random.randint(50, height - 80)
                w = np.random.randint(60, 100)
                h = np.random.randint(30, 60)
                
                # Check if the box center is roughly in the detection area
                center_x, center_y = x + w//2, y + h//2
                if self._point_in_polygon((center_x, center_y), area_points):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(frame, f"{vehicle_type}", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return vehicle_count, traffic_weight
            
        except Exception as e:
            print(f"Error in simulation: {e}")
            return 0, 0.0
    
    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
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