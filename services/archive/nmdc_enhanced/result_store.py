"""
Result store for NMDC Enhanced Service.

This module provides a simple key-value store for maintaining results
across different commands and service calls.
"""

from typing import Dict, Any, List, Optional

class ResultStore:
    """Store for sharing results between service calls.
    
    This class provides a central storage mechanism for storing and retrieving
    results, allowing for data persistence across different commands.
    """
    
    def __init__(self):
        """Initialize an empty result store."""
        self._results: Dict[str, Any] = {}
    
    def store(self, key: str, value: Any) -> None:
        """Store a result value with the given key.
        
        Args:
            key: Unique identifier for the result
            value: Result value to store
        """
        self._results[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a result value by key.
        
        Args:
            key: Key of the result to retrieve
            default: Default value to return if key not found
            
        Returns:
            The stored result value or the default value if key not found
        """
        return self._results.get(key, default)
    
    def remove(self, key: str) -> None:
        """Remove a result from the store.
        
        Args:
            key: Key of the result to remove
        """
        if key in self._results:
            del self._results[key]
    
    def has_key(self, key: str) -> bool:
        """Check if a key exists in the store.
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in self._results
    
    def keys(self) -> List[str]:
        """Get all keys in the store.
        
        Returns:
            List of all keys
        """
        return list(self._results.keys())
    
    def clear(self) -> None:
        """Clear all results from the store."""
        self._results.clear() 