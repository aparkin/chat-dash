"""
Services package for ChatDash application.

This package provides modular services for handling different types of chat interactions.
"""

from .base import (
    PreviewIdentifier,
    ServiceContext,
    ServiceMessage,
    ServiceResponse,
    ChatService,
    ServiceRegistry
)
from .test_service import StoreReportService
from .literature_service import LiteratureService

# Create global service registry
registry = ServiceRegistry()

# Register services
registry.register(StoreReportService())
registry.register(LiteratureService())

__all__ = [
    'PreviewIdentifier',
    'ServiceContext',
    'ServiceMessage',
    'ServiceResponse',
    'ChatService',
    'ServiceRegistry',
    'registry'
] 