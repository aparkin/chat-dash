"""
Data models for query results and related structures.

This module provides dataclasses for representing search results,
query parameters, and related data structures used in the query system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

@dataclass
class SearchResult:
    """
    Container for search results with metadata.
    
    Attributes:
        query: Original search parameters
        total_results: Total number of matching items
        items: List of result items
        aggregations: Optional aggregation results
        collection: Optional collection name for cross-collection search
        
    The items list contains dictionaries with at least:
    - id: UUID of the item
    - certainty: Similarity score (for vector/hybrid search)
    Additional fields depend on the collection and query parameters.
    """
    query: Dict[str, Any]
    total_results: int
    items: List[Dict[str, Any]]
    aggregations: Optional[Dict[str, Any]] = None
    collection: Optional[str] = None

@dataclass
class QueryParameters:
    """
    Parameters for configuring a search query.
    
    Attributes:
        collections: Collections to search (None for all)
        search_type: Type of search (semantic, keyword, hybrid)
        limit: Maximum results per collection
        offset: Number of results to skip
        min_certainty: Minimum similarity score
        include_references: Include reference information
        include_authors: Include author information
        include_entities: Include named entity information
        filters: Optional query filters
    """
    collections: Optional[List[str]] = None
    search_type: str = "hybrid"
    limit: int = 10
    offset: int = 0
    min_certainty: float = 0.7
    include_references: bool = False
    include_authors: bool = False
    include_entities: bool = False
    filters: Optional[Dict[str, Any]] = None

@dataclass
class EntityNetwork:
    """
    Network of relationships around an entity.
    
    Attributes:
        entity: The central entity
        articles: Articles mentioning the entity
        authors: Authors of related articles
        related_entities: Co-occurring entities
        depth: Depth of relationship traversal
        metadata: Additional network statistics
    """
    entity: Dict[str, Any]
    articles: List[Dict[str, Any]] = field(default_factory=list)
    authors: List[Dict[str, Any]] = field(default_factory=list)
    related_entities: List[Dict[str, Any]] = field(default_factory=list)
    depth: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict) 