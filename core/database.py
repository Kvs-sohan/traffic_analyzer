"""
Database Management Module for Traffic Data
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os
import logging

class TrafficDatabase:
    def __init__(self, db_path="data/traffic_data.db"):
        self.db_path = db_path
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.init_database()
            logging.info("Database initialized successfully at %s", db_path)
        except Exception as e:
            logging.error("Failed to initialize database: %s", str(e))
            raise RuntimeError(f"Database initialization failed: {str(e)}")
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create traffic_logs table with exact required fields
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_id INTEGER NOT NULL,
                    vehicle_count INTEGER NOT NULL,
                    traffic_weight REAL NOT NULL,
                    green_time INTEGER NOT NULL,
                    efficiency_score REAL NOT NULL
                )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON traffic_logs(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_signal_id 
                ON traffic_logs(signal_id)
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Database tables and indexes created successfully")
            
        except Exception as e:
            logging.error("Error initializing database tables: %s", str(e))
            raise
    
    def log_traffic_data(self, signal_id, vehicle_count, traffic_weight, green_time, efficiency_score):
        """
        Log traffic data to database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO traffic_logs 
                (timestamp, signal_id, vehicle_count, traffic_weight, green_time, efficiency_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, signal_id, vehicle_count, traffic_weight, green_time, efficiency_score))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"❌ Error logging traffic data: {e}")
    
    def get_recent_data(self, hours=24, signal_id=None):
        """
        Get recent traffic data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if signal_id is not None:
                query = '''
                    SELECT * FROM traffic_logs 
                    WHERE signal_id = ? 
                    AND datetime(timestamp) >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(hours)
                df = pd.read_sql_query(query, conn, params=(signal_id,))
            else:
                query = '''
                    SELECT * FROM traffic_logs 
                    WHERE datetime(timestamp) >= datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(hours)
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving recent data: {e}")
            return pd.DataFrame()
    
    def get_analytics_data(self, start_date=None, end_date=None):
        """
        Get analytics data for specified date range
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            if start_date and end_date:
                query = '''
                    SELECT * FROM traffic_logs 
                    WHERE date(timestamp) BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            else:
                query = '''
                    SELECT * FROM traffic_logs 
                    ORDER BY timestamp
                '''
                df = pd.read_sql_query(query, conn)
            
            conn.close()
            return df
            
        except Exception as e:
            print(f"❌ Error retrieving analytics data: {e}")
            return pd.DataFrame()
    
    def get_signal_statistics(self, signal_id, days=7):
        """
        Get statistics for a specific signal
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    COUNT(*) as total_cycles,
                    AVG(vehicle_count) as avg_vehicles,
                    AVG(traffic_weight) as avg_traffic_weight,
                    AVG(green_time) as avg_green_time,
                    AVG(efficiency_score) as avg_efficiency,
                    MAX(vehicle_count) as max_vehicles,
                    MIN(vehicle_count) as min_vehicles
                FROM traffic_logs 
                WHERE signal_id = ? 
                AND date(timestamp) >= date('now', '-{} days')
            '''.format(days)
            
            cursor = conn.cursor()
            cursor.execute(query, (signal_id,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'total_cycles': result[0] or 0,
                    'avg_vehicles': round(result[1] or 0, 2),
                    'avg_traffic_weight': round(result[2] or 0, 2),
                    'avg_green_time': round(result[3] or 0, 2),
                    'avg_efficiency': round(result[4] or 0, 2),
                    'max_vehicles': result[5] or 0,
                    'min_vehicles': result[6] or 0
                }
            else:
                return {}
                
        except Exception as e:
            print(f"❌ Error retrieving signal statistics: {e}")
            return {}
    
    def export_to_csv(self, filepath=None, start_date=None, end_date=None):
        """
        Export traffic data to CSV file
        """
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"data/traffic_export_{timestamp}.csv"
            
            # Get data
            df = self.get_analytics_data(start_date, end_date)
            
            if df.empty:
                print("⚠️ No data to export")
                return False
            
            # Ensure export directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Export to CSV
            df.to_csv(filepath, index=False)
            print(f"✅ Data exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error exporting data: {e}")
            return False
    
    def clear_old_data(self, days_to_keep=30):
        """
        Clear old data beyond specified days
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM traffic_logs 
                WHERE date(timestamp) < date('now', '-{} days')
            '''.format(days_to_keep))
            
            deleted_rows = cursor.rowcount
            conn.commit()
            conn.close()
            
            print(f"✅ Cleared {deleted_rows} old records")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing old data: {e}")
            return False
    
    def get_database_info(self):
        """
        Get database information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM traffic_logs")
            total_records = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM traffic_logs")
            date_range = cursor.fetchone()
            
            # Get records per signal
            cursor.execute('''
                SELECT signal_id, COUNT(*) 
                FROM traffic_logs 
                GROUP BY signal_id 
                ORDER BY signal_id
            ''')
            signal_counts = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_records': total_records,
                'date_range': date_range,
                'signal_counts': dict(signal_counts) if signal_counts else {},
                'database_size': os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            }
            
        except Exception as e:
            print(f"❌ Error getting database info: {e}")
            return {}