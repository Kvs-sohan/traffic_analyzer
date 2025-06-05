"""
UI Module for Smart Traffic Management System
"""

from .layout import TrafficUI
from .analytics import AnalyticsPanel
from .settings import SimplifiedTrafficSettings

__all__ = ['TrafficUI', 'AnalyticsPanel', 'SimplifiedTrafficSettings']

# Version info
__version__ = "1.0.0"
__author__ = "Smart Traffic Team"

# UI Configuration
UI_CONFIG = {
    'window_title': 'Smart Traffic Management System',
    'window_size': (1400, 900),
    'min_window_size': (1200, 700),
    'theme': 'modern',
    'update_interval': 100,  # milliseconds
    'video_size': (320, 240),
    'colors': {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'success': '#F18F01',
        'danger': '#C73E1D',
        'background': '#F5F5F5',
        'card_bg': '#FFFFFF',
        'text_primary': '#333333',
        'text_secondary': '#666666'
    }
}

# Signal configuration
SIGNAL_CONFIG = {
    'names': ['Signal A', 'Signal B', 'Signal C', 'Signal D'],
    'colors': {
        'red': '#FF4444',
        'yellow': '#FFAA00', 
        'green': '#44FF44'
    },
    'default_timing': {
        'red': 30,
        'yellow': 3,
        'green': 27
    }
}

def get_ui_config():
    """Get UI configuration dictionary"""
    return UI_CONFIG

def get_signal_config():
    """Get signal configuration dictionary"""
    return SIGNAL_CONFIG