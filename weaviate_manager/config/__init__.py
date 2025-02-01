"""
Configuration package for Weaviate Manager.

This package contains all configuration-related modules including:
- Settings management
- Logging configuration
- Environment handling
"""

from .logging_config import setup_logging

__all__ = ['setup_logging'] 