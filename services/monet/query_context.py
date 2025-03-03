"""
Query context management for MONet service.

This module handles query context and history tracking for the MONet service.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .data_manager import MONetDataManager

class QueryContext:
    """Manages query context and history for MONet service."""
    
    def __init__(self, data_manager: MONetDataManager):
        """Initialize query context.
        
        Args:
            data_manager: MONet data manager instance
        """
        self.data_manager = data_manager
        self.history: List[Dict[str, Any]] = []
        self.last_query_time: Optional[datetime] = None
        
    def add_query(self, query: Dict[str, Any], result: Any = None) -> None:
        """Add a query to the history.
        
        Args:
            query: Query dictionary
            result: Optional query result
        """
        self.history.append({
            'query': query,
            'result': result,
            'timestamp': datetime.now()
        })
        self.last_query_time = datetime.now()
        
    def get_recent_queries(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most recent queries.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of recent queries
        """
        return self.history[-limit:]
        
    def clear_history(self) -> None:
        """Clear query history."""
        self.history = []
        self.last_query_time = None 