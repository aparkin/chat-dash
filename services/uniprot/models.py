"""
Data models for UniProt service.

This module defines the data structures and types used by the UniProt service.
It provides:
1. Configuration model for service settings
2. Protein data models
3. Query result container with metadata
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd

@dataclass
class UniProtConfig:
    """Configuration for UniProt service.
    
    This class maintains service configuration including:
    - Service name and API endpoints
    - Model settings for LLM interactions
    - Cache and preview settings
    
    Attributes:
        name: Service identifier (default: "uniprot")
        base_url: Base URL for UniProt API
        search_endpoint: Endpoint for search operations
        stream_endpoint: Endpoint for streaming results
        id_mapping_endpoint: Endpoint for ID mapping operations
        model_name: LLM model to use for interpretations
        temperature: Temperature setting for LLM responses
        cache_expiry_hours: How long to cache data
        default_preview_rows: Number of rows to show in previews
        rate_limit_per_second: API rate limit (requests per second)
    """
    name: str = "uniprot"
    base_url: str = "https://rest.uniprot.org"
    search_endpoint: str = "/uniprotkb/search"
    stream_endpoint: str = "/uniprotkb/stream"
    id_mapping_endpoint: str = "/idmapping/run"
    model_name: str = "anthropic/claude-sonnet"
    temperature: float = 0.4
    cache_expiry_hours: int = 24
    default_preview_rows: int = 5
    rate_limit_per_second: int = 3

@dataclass
class ProteinEntry:
    """Model for a UniProt protein entry.
    
    Contains core protein information and optional fields for
    detailed protein data.
    
    Attributes:
        accession: UniProt accession number (primary identifier)
        entry_name: UniProt entry name
        protein_name: Descriptive protein name
        gene_names: List of associated gene names
        organism: Source organism name
        length: Protein sequence length
        sequence: Amino acid sequence
        reviewed: Whether entry is reviewed (SwissProt) or unreviewed (TrEMBL)
        annotations: Dictionary of additional annotations
    """
    accession: str
    entry_name: str
    protein_name: str
    gene_names: List[str] = field(default_factory=list)
    organism: str = None
    length: int = None
    sequence: str = None
    reviewed: bool = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.accession)
    
    def __eq__(self, other):
        if not isinstance(other, ProteinEntry):
            return False
        return self.accession == other.accession

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