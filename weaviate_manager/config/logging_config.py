"""
Logging configuration for the Weaviate Database Manager.

This module provides a centralized logging configuration that:
- Sets up both file and console logging handlers
- Configures appropriate log levels and formatters
- Manages noisy third-party library logging
- Ensures proper cleanup of logging resources

Features:
- Console output for user-facing messages (INFO and above)
- Detailed file logging for debugging (DEBUG and above)
- Timestamp-based log formatting
- Suppression of verbose library logs
- Resource cleanup utilities

The logging system is designed to provide both user-friendly
feedback and comprehensive debugging information when needed.
"""

import logging
import sys
from pathlib import Path

def setup_logging(log_file: str = 'weaviate_import.log') -> None:
    """
    Configure comprehensive logging with both file and console handlers.
    
    Sets up a dual-output logging system:
    1. Console Handler:
       - Outputs to stderr
       - Shows INFO level and above
       - Uses simplified formatting
       - Provides immediate user feedback
    
    2. File Handler:
       - Writes to specified log file
       - Records DEBUG level and above
       - Uses detailed formatting with timestamps
       - Maintains complete history for debugging
    
    Also configures third-party library logging to minimize noise
    while still capturing errors.
    
    Args:
        log_file: Path to the log file (default: 'weaviate_import.log')
    
    Side Effects:
        - Creates/overwrites the specified log file
        - Configures global logging settings
        - Modifies third-party library log levels
    """
    # Create handlers
    console_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(log_file)
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    console_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # Set log levels
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Suppress noisy libraries
    for logger_name in [
        "weaviate",
        "weaviate.batch",
        "weaviate.auth",
        "weaviate.client",
        "weaviate.collections",
        "weaviate.connect",
        "urllib3.connectionpool"
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

def cleanup_logging() -> None:
    """
    Clean up logging handlers and resources.
    
    This function:
    - Closes all logging handlers properly
    - Removes handlers from the root logger
    - Ensures log files are properly flushed
    
    Should be called:
    - When the application exits
    - Before reconfiguring logging
    - As part of cleanup in error handling
    
    Side Effects:
        - Closes all logging handlers
        - Removes handlers from root logger
        - Flushes any buffered log messages
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler) 