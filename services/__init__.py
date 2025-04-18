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
from .nmdc_service import NMDCService
from .monet.service import MONetService
from .uniprot.service import UniProtService
from .usgs import get_service as get_usgs_service
from .nmdc_enhanced import NMDCEnhancedService

# Create global service registry
registry = ServiceRegistry()

# Initialize basic services
store_report_service = StoreReportService()
visualization_service = VisualizationService()
database_service = DatabaseService()
dataset_service = DatasetService()
literature_service = LiteratureService()
nmdc_service = NMDCService()
monet_service = MONetService()
uniprot_service = UniProtService()
usgs_water_service = get_usgs_service()
nmdc_enhanced_service = NMDCEnhancedService(name="nmdc_enhanced")

# Register basic services
registry.register(store_report_service)
registry.register(visualization_service)
registry.register(database_service)
registry.register(dataset_service)
registry.register(literature_service)
registry.register(ChatLLMService())
registry.register(nmdc_service)
registry.register(monet_service)
registry.register(uniprot_service)
registry.register(usgs_water_service)
registry.register(nmdc_enhanced_service)

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
    'initialize_index_search',
    'get_usgs_service'
]

__version__ = '0.1.0' 