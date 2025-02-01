"""Utility functions for the weaviate_manager package."""

import logging

def log_progress(message: str) -> None:
    """Log a progress message at INFO level."""
    logging.info(f"\n{message}") 