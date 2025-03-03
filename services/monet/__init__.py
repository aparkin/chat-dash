"""
MONet service for soil data analysis.

This module provides access to EMSL's Molecular Observation Network (MONet) soil data.
"""

from .service import MONetService

# Re-export for backward compatibility
__all__ = ['MONetService'] 