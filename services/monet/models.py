"""
Data models for MONet service.

This module defines the data structures and types used by the MONet service.
It provides:
1. Configuration model for service settings
2. Geographic models for spatial queries
3. Query result container with metadata
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd

@dataclass
class MONetConfig:
    """Configuration for MONet service.
    
    This class maintains service configuration including:
    - Service name and endpoints
    - Model settings for LLM interactions
    - Cache and preview settings
    
    Attributes:
        name: Service identifier (default: "monet")
        base_url: Base URL for MONet API
        model_name: LLM model to use for interpretations
        temperature: Temperature setting for LLM responses
        cache_expiry_hours: How long to cache data (0 disables background refresh)
        default_preview_rows: Number of rows to show in previews
    """
    name: str = "monet"
    base_url: str = "https://sc-data.emsl.pnnl.gov"
    model_name: str = "anthropic/claude-sonnet"
    temperature: float = 0.4
    cache_expiry_hours: int = 24  # Changed from 1 to 24 hours to match NMDC pattern
    default_preview_rows: int = 5

@dataclass
class GeoPoint:
    """Geographic point with optional search radius.
    
    Used for point-radius search queries to find samples
    within a specified distance of a location.
    
    Attributes:
        latitude: Decimal degrees (-90 to 90)
        longitude: Decimal degrees (-180 to 180)
        radius_km: Optional search radius in kilometers
    """
    latitude: float
    longitude: float
    radius_km: Optional[float] = None

    def __hash__(self):
        return hash((self.latitude, self.longitude, self.radius_km))
    
    def __eq__(self, other):
        if not isinstance(other, GeoPoint):
            return False
        return (self.latitude == other.latitude and
                self.longitude == other.longitude and
                self.radius_km == other.radius_km)

@dataclass
class GeoBBox:
    """Geographic bounding box.
    
    Used for spatial queries to find samples within
    a rectangular geographic region.
    
    Attributes:
        min_lat: Minimum latitude (-90 to 90)
        max_lat: Maximum latitude (-90 to 90)
        min_lon: Minimum longitude (-180 to 180)
        max_lon: Maximum longitude (-180 to 180)
    """
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    
    def __hash__(self):
        return hash((self.min_lat, self.max_lat, self.min_lon, self.max_lon))
    
    def __eq__(self, other):
        if not isinstance(other, GeoBBox):
            return False
        return (self.min_lat == other.min_lat and
                self.max_lat == other.max_lat and
                self.min_lon == other.min_lon and
                self.max_lon == other.max_lon)

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
            - dataframe: Preview rows of results
        """
        return {
            'query': self.metadata.get('query'),
            'description': self.description,
            'result_count': len(self.dataframe),
            'execution_time': self.creation_time.isoformat(),
            'dataframe': self.dataframe.head(max_rows).to_dict('records')
        } 