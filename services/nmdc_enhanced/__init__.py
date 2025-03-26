"""
NMDC Enhanced Service package.

This package provides enhanced data discovery and integration capabilities
for interacting with the NMDC API.
"""

from .service import NMDCEnhancedService
from .data_manager import NMDCEnhancedDataManager
from .models import NMDCEnhancedConfig, QueryResult

__all__ = [
    'NMDCEnhancedService',
    'NMDCEnhancedDataManager',
    'NMDCEnhancedConfig',
    'QueryResult'
]

__version__ = "0.1.0" 