import sqlite3
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path

class DatabaseManager:
    """Manages SQLite database connections and queries."""
    
    def __init__(self, db_path: str):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test database connection and verify file exists."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT 1")
        except sqlite3.Error as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    def get_tables(self) -> List[str]:
        """Get list of all user tables in database."""
        with sqlite3.connect(self.db_path) as conn:
            tables = pd.read_sql("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table' 
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """, conn)
        return tables['name'].tolist()
    
    def get_table_info(self, table_name: str) -> Dict:
        """Get detailed information about a specific table."""
        with sqlite3.connect(self.db_path) as conn:
            # Get column information
            columns = pd.read_sql(f"PRAGMA table_info({table_name})", conn)
            
            # Get row count
            row_count = pd.read_sql(
                f"SELECT COUNT(*) as count FROM {table_name}", 
                conn
            ).iloc[0]['count']
            
            # Get foreign key information
            foreign_keys = pd.read_sql(
                f"PRAGMA foreign_key_list({table_name})", 
                conn
            )
            
        return {
            'columns': columns.to_dict('records'),
            'row_count': row_count,
            'foreign_keys': foreign_keys.to_dict('records')
        }
    
    def get_database_summary(self) -> Dict:
        """Get summary of entire database structure."""
        tables = self.get_tables()
        return {
            table: self.get_table_info(table)
            for table in tables
        }
    
    def execute_query(self, query: str):
        """Execute a SQL query and return results.
        
        Args:
            query: SQL query string
            
        Returns:
            sqlite3.Cursor: Query results
        """
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(query) 