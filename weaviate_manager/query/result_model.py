"""
Core data structures and models for search results.

This module defines the standard format for search results and provides
utilities for working with result data structures.
"""

from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json

@dataclass
class QueryInfo:
    """Information about the executed query."""
    text: str
    type: str  # 'semantic', 'keyword', or 'hybrid'
    parameters: Dict[str, Any]
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

@dataclass
class ResultSummary:
    """Summary statistics for search results."""
    total_matches: int
    collection_counts: Dict[str, int]
    unified_articles: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

@dataclass
class ItemProperties:
    """Container for item properties."""
    values: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return self.values  # Return values directly without nesting

@dataclass
class CollectionItem:
    """A single item from any collection."""
    id: str
    score: float
    properties: ItemProperties = field(default_factory=ItemProperties)
    cross_references: Dict[str, List[str]] = field(default_factory=dict)
    certainty: Optional[float] = None  # Optional, only included if meaningful
    score_explanation: Optional[str] = None  # Optional, only for hybrid search
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            'id': self.id,
            'score': self.score,
            'properties': self.properties.to_dict(),
        }
        
        # Only include optional fields if they have meaningful values
        if self.cross_references:
            d['cross_references'] = self.cross_references
        if self.certainty is not None:
            d['certainty'] = self.certainty
        if self.score_explanation:
            d['score_explanation'] = self.score_explanation
            
        return d

@dataclass
class StandardResults:
    """Standard format for search results."""
    query_info: QueryInfo
    summary: ResultSummary
    collection_results: Dict[str, List[CollectionItem]]
    unified_results: Optional[List[CollectionItem]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        # Start with base dictionary
        result = {
            'query_info': self.query_info.to_dict(),
            'summary': self.summary.to_dict(),
            'collection_results': {
                name: [item.to_dict() for item in items]
                for name, items in self.collection_results.items()
            }
        }
        
        # Only include unified results if present, ensuring no duplicates
        if self.unified_results is not None:
            seen_ids = set()
            filtered_items = []
            for item in self.unified_results:
                if item.id not in seen_ids:
                    filtered_items.append(item.to_dict())
                    seen_ids.add(item.id)
            result['unified_results'] = filtered_items
            
        return result
    
    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, cls=ResultJSONEncoder)

class ResultJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for result classes."""
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Weaviate UUID objects
        if hasattr(obj, '__class__') and obj.__class__.__name__ == '_WeaviateUUIDInt':
            return str(obj)
        return super().default(obj) 