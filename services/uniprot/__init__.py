"""
UniProt service for protein data retrieval and analysis.

This service provides access to the UniProtKB (UniProt Knowledge Base) 
protein information through a consistent interface. It allows for protein
data retrieval, filtering, and analysis through both direct queries
and natural language processing.

Key Features:
- Protein search by various criteria
- Single protein lookup by accession
- Natural language query interpretation
- Result dataset conversion
- Protein data analysis
"""

from .service import UniProtService

__all__ = ["UniProtService"] 