"""
Utility helper functions for Smart Traffic Management System
Contains common functions used across the application
"""
import os
import json
import logging
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional


def setup_logging(log_level: str = "INFO", log_file: str = "traffic_system.log") -> logging.Logger:
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Log file path
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")


def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=4)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
    
    Returns:
        True if directory exists or was created
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def validate_video_file(video_path: str) -> bool:
    """
    Validate if video file exists and is readable
    
    Args:
        video_path: Path to video file
    
    Returns:
        True if valid video file
    """
    if not os.path.exists(video_path):
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception:
        return False


def resize_frame(frame: np.ndarray, target_width: int = 640, target_height: int = 480) -> np.ndarray:
    """
    Resize frame to target dimensions
    
    Args:
        frame: Input frame
        target_width: Target width
        target_height: Target height
    
    Returns:
        Resized frame
    """
    return cv2.resize(frame, (target_width, target_height))


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate frames per second
    
    Args:
        start_time: Start time timestamp
        frame_count: Number of frames processed
    
    Returns:
        FPS value
    """
    elapsed_time = datetime.now().timestamp() - start_time
    if elapsed_time > 0:
        return frame_count / elapsed_time
    return 0.0


def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Format timestamp to standard string format
    
    Args:
        timestamp: Datetime object (uses current time if None)
    
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    Parse timestamp string to datetime object
    
    Args:
        timestamp_str: Timestamp string in format "YYYY-MM-DD HH:MM:SS"
    
    Returns:
        Datetime object
    """
    return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


def calculate_time_difference(start_time: datetime, end_time: datetime) -> float:
    """
    Calculate time difference in seconds
    
    Args:
        start_time: Start datetime
        end_time: End datetime
    
    Returns:
        Time difference in seconds
    """
    return (end_time - start_time).total_seconds()


def get_area_center(area_coords: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Calculate center point of an area defined by coordinates
    
    Args:
        area_coords: List of (x, y) coordinate tuples
    
    Returns:
        Center point (x, y)
    """
    if not area_coords:
        return (0, 0)
    
    x_coords = [coord[0] for coord in area_coords]
    y_coords = [coord[1] for coord in area_coords]
    
    center_x = sum(x_coords) // len(x_coords)
    center_y = sum(y_coords) // len(y_coords)
    
    return (center_x, center_y)


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm
    
    Args:
        point: Point coordinates (x, y)
        polygon: List of polygon vertices
    
    Returns:
        True if point is inside polygon
    """
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


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
    
    Returns:
        Distance between points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def normalize_coordinates(coords: List[Tuple[int, int]], frame_width: int, frame_height: int) -> List[Tuple[float, float]]:
    """
    Normalize coordinates to 0-1 range
    
    Args:
        coords: List of coordinate tuples
        frame_width: Frame width
        frame_height: Frame height
    
    Returns:
        Normalized coordinates
    """
    normalized = []
    for x, y in coords:
        norm_x = x / frame_width
        norm_y = y / frame_height
        normalized.append((norm_x, norm_y))
    return normalized


def denormalize_coordinates(normalized_coords: List[Tuple[float, float]], frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
    """
    Convert normalized coordinates back to pixel coordinates
    
    Args:
        normalized_coords: List of normalized coordinate tuples
        frame_width: Frame width
        frame_height: Frame height
    
    Returns:
        Pixel coordinates
    """
    denormalized = []
    for norm_x, norm_y in normalized_coords:
        x = int(norm_x * frame_width)
        y = int(norm_y * frame_height)
        denormalized.append((x, y))
    return denormalized


def calculate_signal_efficiency(vehicles_processed: int, green_time: int, max_capacity: int = 50) -> float:
    """
    Calculate signal efficiency percentage
    
    Args:
        vehicles_processed: Number of vehicles processed
        green_time: Green light duration in seconds
        max_capacity: Maximum vehicles that can be processed
    
    Returns:
        Efficiency percentage (0-100)
    """
    if green_time == 0:
        return 0.0
    
    theoretical_max = min(max_capacity, green_time * 2)  # Assume 2 vehicles per second max
    if theoretical_max == 0:
        return 0.0
    
    efficiency = (vehicles_processed / theoretical_max) * 100
    return min(efficiency, 100.0)


def validate_signal_timing(green_time: int, yellow_time: int, red_time: int) -> bool:
    """
    Validate signal timing parameters
    
    Args:
        green_time: Green light duration
        yellow_time: Yellow light duration
        red_time: Red light duration
    
    Returns:
        True if timing is valid
    """
    # Basic validation rules
    if green_time < 5 or green_time > 120:  # 5-120 seconds
        return False
    if yellow_time < 2 or yellow_time > 10:  # 2-10 seconds
        return False
    if red_time < 5 or red_time > 180:  # 5-180 seconds
        return False
    
    return True


def get_optimal_signal_timing(vehicle_count: int, base_green_time: int = 30) -> Dict[str, int]:
    """
    Calculate optimal signal timing based on vehicle count
    
    Args:
        vehicle_count: Number of vehicles detected
        base_green_time: Base green light duration
    
    Returns:
        Dictionary with timing values
    """
    # Adjust green time based on vehicle count
    if vehicle_count == 0:
        green_time = 10  # Minimum green time
    elif vehicle_count <= 5:
        green_time = base_green_time
    elif vehicle_count <= 15:
        green_time = base_green_time + 10
    else:
        green_time = base_green_time + 20
    
    # Ensure timing is within valid ranges
    green_time = max(10, min(green_time, 90))
    
    return {
        'green_time': green_time,
        'yellow_time': 4,
        'red_time': 30
    }


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to readable string
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    Clamp value between min and max
    
    Args:
        value: Input value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
    
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
    
    Returns:
        Division result or default value
    """
    return numerator / denominator if denominator != 0 else default


def get_file_size(file_path: str) -> int:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
    
    Returns:
        File size in bytes, -1 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return -1


def create_detection_area_from_config(area_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create detection area configuration from config dictionary
    
    Args:
        area_config: Area configuration dictionary
    
    Returns:
        Processed area configuration
    """
    return {
        'id': area_config.get('id', 0),
        'name': area_config.get('name', 'Unknown Area'),
        'coordinates': area_config.get('coordinates', []),
        'signal_id': area_config.get('signal_id', 0),
        'enabled': area_config.get('enabled', True)
    }


def generate_report_filename(prefix: str = "traffic_report", extension: str = "csv") -> str:
    """
    Generate timestamped filename for reports
    
    Args:
        prefix: Filename prefix
        extension: File extension
    
    Returns:
        Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def is_valid_coordinate(coord: Tuple[int, int], frame_width: int, frame_height: int) -> bool:
    """
    Check if coordinate is within frame boundaries
    
    Args:
        coord: Coordinate tuple (x, y)
        frame_width: Frame width
        frame_height: Frame height
    
    Returns:
        True if coordinate is valid
    """
    x, y = coord
    return 0 <= x < frame_width and 0 <= y < frame_height


def convert_seconds_to_time_string(seconds: int) -> str:
    """
    Convert seconds to HH:MM:SS format
    
    Args:
        seconds: Total seconds
    
    Returns:
        Time string in HH:MM:SS format
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"