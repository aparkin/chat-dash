"""
Exceptions for query operations.

This module defines custom exceptions used in query operations
to handle various error conditions.
"""

class DataIntegrityError(Exception):
    """Raised when data integrity issues are detected.
    
    This is a serious error that indicates a mismatch between
    expected data (based on deterministic UUIDs) and actual
    database state.
    """
    pass 