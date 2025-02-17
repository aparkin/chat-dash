"""
Services package for ChatDash application.

This package provides modular services for handling different types of chat interactions.
"""

from typing import Dict, Any

from .base import (
    PreviewIdentifier,
    ServiceMessage,
    ServiceResponse,
    ChatService,
    ServiceRegistry
)
from .test_service import StoreReportService
from .literature_service import LiteratureService
from .index_search_service import IndexSearchService
from .visualization_service import VisualizationService
from .database_service import DatabaseService
from .dataset_service import DatasetService
from .chat_llm_service import ChatLLMService

# Create global service registry
registry = ServiceRegistry()

# Initialize basic services
store_report_service = StoreReportService()
visualization_service = VisualizationService()
database_service = DatabaseService()
dataset_service = DatasetService()
literature_service = LiteratureService()

# Register basic services
registry.register(store_report_service)
registry.register(visualization_service)
registry.register(database_service)
registry.register(dataset_service)
registry.register(literature_service)
registry.register(ChatLLMService())

def initialize_index_search(text_searcher: Any, text_searcher_db: Any) -> None:
    """Initialize the index search service with available searchers.
    
    This function should be called from ChatDash.py after searchers are initialized.
    
    Args:
        text_searcher: Dataset text searcher instance
        text_searcher_db: Database text searcher instance
    """
    index_sources = {
        'datasets': text_searcher,
        'database': text_searcher_db,
        # Future sources can be added here:
        # 'documents': document_searcher,
        # 'code': code_searcher,
    }
    index_search_service = IndexSearchService(index_sources=index_sources)
    registry.register(index_search_service)

# Export for convenience
service_registry = registry

__all__ = [
    'PreviewIdentifier',
    'ServiceMessage',
    'ServiceResponse',
    'ChatService',
    'ServiceRegistry',
    'registry',
    'initialize_index_search'
] 