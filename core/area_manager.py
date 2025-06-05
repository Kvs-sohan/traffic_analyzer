"""
Area Manager Module
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

class AreaManager:
    """Manages detection areas for traffic signals"""
    
    def __init__(self):
        self.areas = {}  # Dictionary to store areas for each signal
        self.load_areas()
    
    def load_areas(self, config_file: str = "config/areas.json") -> bool:
        """
        Load detection areas from configuration file
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            True if areas were loaded successfully
        """
        try:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                if isinstance(config, dict) and 'areas' in config:
                    # Load areas from config
                    for area_config in config['areas']:
                        signal_id = area_config.get('signal_id')
                        if signal_id is not None:
                            self.areas[signal_id] = area_config.get('points', [])
                    
                    logging.info(f"✅ Loaded {len(config['areas'])} detection areas")
                    return True
            else:
                # Create default areas
                self._create_default_areas()
                logging.info("Created default detection areas")
                return True
                
        except Exception as e:
            logging.error(f"❌ Error loading detection areas: {str(e)}")
            self._create_default_areas()
            return False
    
    def save_areas(self, config_file: str = "config/areas.json") -> bool:
        """
        Save detection areas to configuration file
        
        Args:
            config_file: Path to save configuration
            
        Returns:
            True if areas were saved successfully
        """
        try:
            # Create config directory if it doesn't exist
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare areas configuration
            config = {
                'areas': []
            }
            
            for signal_id, points in self.areas.items():
                area_config = {
                    'signal_id': signal_id,
                    'name': f"Signal {chr(65 + signal_id)} Area",
                    'points': points,
                    'enabled': True
                }
                config['areas'].append(area_config)
            
            # Save to file
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"✅ Saved {len(config['areas'])} detection areas")
            return True
            
        except Exception as e:
            logging.error(f"❌ Error saving detection areas: {str(e)}")
            return False
    
    def get_area(self, signal_id: int) -> Optional[List[Tuple[int, int]]]:
        """
        Get detection area for a signal
        
        Args:
            signal_id: Signal ID
            
        Returns:
            List of area points or None if not found
        """
        return self.areas.get(signal_id)
    
    def set_area(self, signal_id: int, points: List[Tuple[int, int]], name: str = None) -> bool:
        """
        Set detection area for a signal
        
        Args:
            signal_id: Signal ID
            points: List of area points
            name: Optional area name
            
        Returns:
            True if area was set successfully
        """
        try:
            if not isinstance(points, list) or len(points) < 3:
                raise ValueError("Area must have at least 3 points")
            
            self.areas[signal_id] = points
            return self.save_areas()
            
        except Exception as e:
            logging.error(f"❌ Error setting detection area: {str(e)}")
            return False
    
    def clear_area(self, signal_id: int) -> bool:
        """
        Clear detection area for a signal
        
        Args:
            signal_id: Signal ID
            
        Returns:
            True if area was cleared successfully
        """
        try:
            if signal_id in self.areas:
                del self.areas[signal_id]
                return self.save_areas()
            return True
            
        except Exception as e:
            logging.error(f"❌ Error clearing detection area: {str(e)}")
            return False
    
    def clear_all_areas(self) -> bool:
        """
        Clear all detection areas
        
        Returns:
            True if areas were cleared successfully
        """
        try:
            self.areas.clear()
            return self.save_areas()
            
        except Exception as e:
            logging.error(f"❌ Error clearing all areas: {str(e)}")
            return False
    
    def _create_default_areas(self):
        """Create default detection areas"""
        # Default areas for each signal (simple rectangles)
        default_areas = {
            0: [(100, 100), (300, 100), (300, 300), (100, 300)],  # Signal A
            1: [(340, 100), (540, 100), (540, 300), (340, 300)],  # Signal B
            2: [(100, 340), (300, 340), (300, 540), (100, 540)],  # Signal C
            3: [(340, 340), (540, 340), (540, 540), (340, 540)]   # Signal D
        }
        
        self.areas = default_areas.copy()
        self.save_areas()
    
    def normalize_coordinates(self, points: List[Tuple[int, int]], frame_width: int, frame_height: int) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates to normalized coordinates (0-1 range)
        
        Args:
            points: List of pixel coordinates
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            List of normalized coordinates
        """
        return [
            (x / frame_width, y / frame_height)
            for x, y in points
        ]
    
    def denormalize_coordinates(self, normalized_points: List[Tuple[float, float]], frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """
        Convert normalized coordinates back to pixel coordinates
        
        Args:
            normalized_points: List of normalized coordinates
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            List of pixel coordinates
        """
        return [
            (int(x * frame_width), int(y * frame_height))
            for x, y in normalized_points
        ] 