"""
Traffic Signal Management Module
"""

from collections import deque
from datetime import datetime
import time
import logging
from typing import List, Dict, Any

class TrafficSignal:
    def __init__(self, signal_id):
        self.signal_id = signal_id
        self.min_green_time = 7
        self.max_green_time = 60
        self.default_green_time = 15
        self.yellow_time = 3
        self.all_red_time = 2
        
        # Current state
        self.current_state = 'RED'
        self.remaining_time = 0
        
        # Traffic history for adaptive timing
        self.vehicle_history = deque(maxlen=10)
        self.priority_mode = False
        
    def calculate_adaptive_green_time(self, vehicle_count, traffic_weight, time_of_day=None):
        """
        Calculate optimal green time based on traffic conditions
        """
        base_time = self.min_green_time
        
        # Traffic density factor
        density_time = min(traffic_weight * 2, self.max_green_time - base_time)
        
        # Time of day factor (rush hours get slight boost)
        time_factor = 1.0
        if time_of_day:
            hour = time_of_day.hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                time_factor = 1.2
            elif 22 <= hour or hour <= 6:  # Night time
                time_factor = 0.8
        
        # Historical traffic pattern factor
        if len(self.vehicle_history) > 3:
            avg_traffic = sum(self.vehicle_history) / len(self.vehicle_history)
            if traffic_weight > avg_traffic * 1.5:  # High traffic spike
                time_factor *= 1.3
        
        # Calculate final time
        calculated_time = int(base_time + (density_time * time_factor))
        
        # Store in history
        self.vehicle_history.append(traffic_weight)
        
        return max(self.min_green_time, min(calculated_time, self.max_green_time))
    
    def set_state(self, state, remaining_time=0):
        """Set signal state and remaining time"""
        self.current_state = state
        self.remaining_time = remaining_time
    
    def update_timing_config(self, min_green=None, max_green=None, yellow=None):
        """Update timing configuration"""
        if min_green is not None:
            self.min_green_time = max(5, min_green)
        if max_green is not None:
            self.max_green_time = min(120, max_green)
        if yellow is not None:
            self.yellow_time = max(2, yellow)
    
    def get_state_info(self):
        """Get current state information"""
        return {
            'signal_id': self.signal_id,
            'state': self.current_state,
            'remaining_time': self.remaining_time,
            'vehicle_history': list(self.vehicle_history)
        }
    
    def calculate_efficiency(self, vehicles_processed, green_time_used):
        """Calculate signal efficiency based on throughput"""
        if green_time_used == 0:
            return 0.0
        
        # Theoretical maximum vehicles per second
        max_throughput = 0.5  # vehicles per second
        theoretical_max = max_throughput * green_time_used
        
        if theoretical_max == 0:
            return 0.0
        
        # Actual efficiency percentage
        efficiency = min((vehicles_processed / theoretical_max) * 100, 100)
        return round(efficiency, 2)

class SignalController:
    """Controls traffic signal states and timing"""
    
    def __init__(self):
        try:
            logging.info("Initializing Signal Controller...")
            
            # Initialize signal states
            self.signals = []
            self.current_sequence = []
            self.adaptive_mode = True
            self.emergency_mode = False
            
            # Create signals
            for i in range(4):
                self.signals.append(TrafficSignal(i))
            
            # Set initial states - Signal A starts as GREEN, others as RED
            self.signals[0].set_state('GREEN', self.signals[0].default_green_time)
            for i in range(1, 4):
                self.signals[i].set_state('RED')
            
            # Initialize timing
            self.cycle_start_time = time.time()
            self.yellow_duration = 3  # Fixed 3 seconds yellow time
            self.current_signal_index = 0
            
            logging.info("Signal Controller initialized successfully")
            
        except Exception as e:
            logging.error("Failed to initialize Signal Controller: %s", str(e))
            raise RuntimeError(f"Signal Controller initialization failed: {str(e)}")
    
    def get_signal(self, signal_id: int) -> Dict[str, Any]:
        """Get current state of a signal"""
        try:
            if 0 <= signal_id < len(self.signals):
                return self.signals[signal_id].get_state_info()
            raise ValueError(f"Invalid signal ID: {signal_id}")
        except Exception as e:
            logging.error("Error getting signal state: %s", str(e))
            raise
    
    def set_adaptive_mode(self, enabled):
        """Enable/disable adaptive timing mode"""
        self.adaptive_mode = enabled
        logging.info(f"Adaptive mode {'enabled' if enabled else 'disabled'}")
    
    def get_system_status(self):
        """Get overall system status"""
        return {
            'emergency_mode': self.emergency_mode,
            'signals_status': [signal.get_state_info() for signal in self.signals]
        }
    
    def update_timing(self, traffic_weights: List[float]):
        """Update signal timing based on traffic weights"""
        if not self.adaptive_mode:
            return
            
        try:
            # Prepare traffic data for optimal sequence calculation
            traffic_data = []
            for i, weight in enumerate(traffic_weights):
                traffic_data.append({
                    'signal_id': i,
                    'vehicle_count': int(weight * 2),  # Estimate vehicle count from weight
                    'traffic_weight': weight
                })
            
            # Get optimal sequence
            sequence = self.calculate_optimal_sequence(traffic_data)
            
            # Update signal timings
            for data in sequence:
                signal_id = data['signal_id']
                if 0 <= signal_id < len(self.signals):
                    signal = self.signals[signal_id]
                    # Update green time based on traffic weight
                    green_time = self._calculate_green_time(data['traffic_weight'])
                    signal.update_timing_config(
                        min_green=max(7, int(green_time * 0.5)),
                        max_green=min(120, int(green_time * 1.5))
                    )
            
            logging.info("Signal timing updated successfully")
            
        except Exception as e:
            logging.error("Error updating signal timing: %s", str(e))
            raise
    
    def _calculate_green_time(self, traffic_weight: float) -> int:
        """Calculate green time based on traffic weight"""
        try:
            base_time = 30  # Base green time in seconds
            min_time = 10   # Minimum green time
            max_time = 90   # Maximum green time
            
            # Calculate green time based on traffic weight
            green_time = int(base_time * (1 + traffic_weight))
            
            # Ensure time is within bounds
            green_time = max(min_time, min(green_time, max_time))
            
            return green_time
            
        except Exception as e:
            logging.error("Error calculating green time: %s", str(e))
            raise
    
    def calculate_optimal_sequence(self, traffic_data):
        """
        Calculate optimal signal sequence based on traffic conditions
        traffic_data: list of dicts with signal_id, vehicle_count, traffic_weight
        """
        prioritized_signals = []
        
        for data in traffic_data:
            signal_id = data['signal_id']
            if not 0 <= signal_id < len(self.signals):
                continue
                
            signal = self.signals[signal_id]
            vehicle_count = data['vehicle_count']
            traffic_weight = data['traffic_weight']
            
            # Calculate priority score
            base_priority = traffic_weight
            
            # Add urgency factor based on waiting time
            if len(signal.vehicle_history) > 0:
                avg_historical = sum(signal.vehicle_history) / len(signal.vehicle_history)
                if traffic_weight > avg_historical * 1.5:
                    base_priority *= 1.3
            
            # Emergency mode override
            if self.emergency_mode and vehicle_count > 0:
                base_priority *= 2.0
            
            # Calculate green time
            green_time = signal.calculate_adaptive_green_time(
                vehicle_count, traffic_weight, datetime.now()
            )
            
            prioritized_signals.append({
                'signal_id': signal_id,
                'priority': base_priority,
                'green_time': green_time,
                'vehicle_count': vehicle_count,
                'traffic_weight': traffic_weight
            })
        
        # Sort by priority (descending)
        prioritized_signals.sort(key=lambda x: x['priority'], reverse=True)
        
        return prioritized_signals
    
    def set_emergency_mode(self, enabled):
        """Enable/disable emergency mode"""
        self.emergency_mode = enabled
        for signal in self.signals:
            signal.priority_mode = enabled
    
    def update_signals(self):
        """Update signal states based on timing"""
        try:
            current_time = time.time()
            current_signal = self.signals[self.current_signal_index]
            
            # Update remaining time
            if current_signal.current_state != 'RED':
                elapsed = current_time - self.cycle_start_time
                current_signal.remaining_time = max(0, current_signal.remaining_time - elapsed)
                self.cycle_start_time = current_time
            
            # Handle state transitions
            if current_signal.remaining_time <= 0:
                if current_signal.current_state == 'GREEN':
                    # Change to yellow
                    current_signal.set_state('YELLOW', self.yellow_duration)
                    logging.info(f"Signal {self.current_signal_index} turned YELLOW")
                    
                elif current_signal.current_state == 'YELLOW':
                    # Change to red and activate next signal
                    current_signal.set_state('RED')
                    self.current_signal_index = (self.current_signal_index + 1) % len(self.signals)
                    next_signal = self.signals[self.current_signal_index]
                    
                    # Calculate green time for next signal
                    if self.adaptive_mode:
                        green_time = next_signal.calculate_adaptive_green_time(
                            vehicle_count=len(next_signal.vehicle_history),
                            traffic_weight=sum(next_signal.vehicle_history) / len(next_signal.vehicle_history) if next_signal.vehicle_history else 1.0,
                            time_of_day=datetime.now()
                        )
                    else:
                        green_time = next_signal.default_green_time
                    
                    next_signal.set_state('GREEN', green_time)
                    logging.info(f"Signal {self.current_signal_index} turned GREEN for {green_time} seconds")
            
            return True
            
        except Exception as e:
            logging.error(f"Error updating signals: {str(e)}")
            return False