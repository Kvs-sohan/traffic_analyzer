import os
import json
import cv2

class CameraManager:
    """Manages camera sources and video capture"""
    
    def __init__(self):
        self.sources = []
        self.captures = []
        self.load_sources()
        
    def load_sources(self, sources_file="config/sources.json"):
        """Load camera sources from config file"""
        try:
            if os.path.exists(sources_file):
                with open(sources_file, 'r') as f:
                    self.sources = json.load(f)
                print(f"✅ Loaded {len(self.sources)} camera sources")
            else:
                # Default demo sources
                self.sources = [
                    "data/videos/signal_0.mp4",
                    "data/videos/signal_1.mp4",
                    "data/videos/signal_2.mp4",
                    "data/videos/signal_3.mp4"
                ]
                self.save_sources(sources_file)
        except Exception as e:
            print(f"❌ Error loading sources: {e}")
            self.sources = []
    
    def save_sources(self, sources_file="config/sources.json"):
        """Save camera sources to config file"""
        try:
            os.makedirs(os.path.dirname(sources_file), exist_ok=True)
            with open(sources_file, 'w') as f:
                json.dump(self.sources, f, indent=2)
        except Exception as e:
            print(f"❌ Error saving sources: {e}")
    
    def initialize_captures(self):
        """Initialize video captures for all sources"""
        self.close_captures()  # Close any existing captures
        
        for source in self.sources:
            if source:
                try:
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        print(f"❌ Failed to open source: {source}")
                        cap = None
                    else:
                        print(f"✅ Opened source: {source}")
                except Exception as e:
                    print(f"❌ Error initializing capture for {source}: {e}")
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
    
    def get_frame(self, signal_id):
        """Get current frame from specified signal camera"""
        try:
            if 0 <= signal_id < len(self.captures) and self.captures[signal_id] is not None:
                ret, frame = self.captures[signal_id].read()
                if ret:
                    return frame
                else:
                    # If we reached the end of the video, reset to beginning
                    self.captures[signal_id].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.captures[signal_id].read()
                    return frame if ret else None
        except Exception as e:
            print(f"❌ Error getting frame from signal {signal_id}: {e}")
        return None
    
    def __del__(self):
        """Clean up resources"""
        self.close_captures() 