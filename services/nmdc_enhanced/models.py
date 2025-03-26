"""
Data models for NMDC Enhanced service.

This module defines the data structures and types used by the NMDC Enhanced service.
It provides:
1. Configuration model for service settings
2. Query result container with metadata
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd
from pydantic import BaseModel

class NMDCEnhancedConfig(BaseModel):
    """Configuration for the NMDC Enhanced service."""

    # Service configuration
    name: str = "nmdc_enhanced"

    # API configuration
    api_base_url: str = "https://data.microbiomedata.org/api"
    api_fallback_url: str = "https://api.microbiomedata.org"
    api_summary_url: str = "https://data.microbiomedata.org/api/summary"
    
    # LLM configuration
    model_name: str = "anthropic/claude-sonnet"
    temperature: float = 0.4
    
    # Cache and defaults
    cache_expiry_hours: int = 24
    default_preview_rows: int = 5
    
    # Request configuration
    request_timeout: int = 60  # Seconds
    max_retries: int = 3
    retry_delay: float = 0.5
    batch_size: int = 500  # Number of items to request at once
    
    # Cache configuration
    use_cache: bool = True
    cache_max_age_hours: int = 24  # Maximum age of cache in hours
    enable_background_refresh: bool = True  # Whether to refresh data in background
    
    # Visualization defaults
    map_center_lat: float = 39.8283
    map_center_lon: float = -98.5795
    map_zoom: int = 3

@dataclass
class QueryResult:
    """Container for query execution results.
    
    Stores both the result DataFrame and associated metadata
    including query details, description, and timing information.
    
    Attributes:
        dataframe: Pandas DataFrame containing query results
        metadata: Dictionary of query metadata (query, params, etc.)
        description: Human-readable description of the query
        creation_time: When the query was executed
    """
    dataframe: pd.DataFrame
    metadata: Dict[str, Any]
    description: str
    creation_time: datetime = field(default_factory=datetime.now)
    
    def to_preview(self, max_rows: int = 5) -> Dict[str, Any]:
        """Convert to preview format for storage.
        
        Creates a compact preview representation suitable for
        storing in the successful_queries_store.
        
        Args:
            max_rows: Maximum number of rows to include in preview
            
        Returns:
            Dictionary containing:
            - query: Original query definition
            - description: Query description
            - result_count: Total number of results
            - execution_time: When query was run
            - dataframe: Preview rows of results AND full dataframe
        """
        return {
            'query': self.metadata.get('query'),
            'description': self.description,
            'result_count': len(self.dataframe),
            'execution_time': self.creation_time.isoformat(),
            'dataframe': self.dataframe.to_dict('records'),  # Store full dataframe for later use
            'preview': self.dataframe.head(max_rows).to_dict('records')  # Store preview for display
        } 