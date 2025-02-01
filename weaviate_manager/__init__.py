"""
Weaviate Database Manager

A comprehensive system for managing scientific literature in a Weaviate vector database.
This module provides functionality for:

- Data Import: Batch processing of scientific articles and related entities
- Schema Management: Creation and validation of database schema
- Text Processing: Intelligent compression and token management
- Cross-References: Management of relationships between entities
- Database Inspection: Tools for analyzing database state and content
- Query Management: Semantic and hybrid search with cross-references

The system is designed to handle:
- Scientific articles with full text and metadata
- Author information and name variants
- Citations and bibliographic references
- Named entities and their relationships
- Citation contexts and scoring

For usage examples and configuration options, see the README.
"""

__version__ = "0.1.0"

from .database.client import get_client, check_connection
from .query.manager import QueryManager

__all__ = [
    "get_client",
    "check_connection",
    "QueryManager",
    "__version__"
] 