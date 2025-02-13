"""
Base classes and infrastructure for chat services.

This module provides the foundational classes and interfaces for implementing
modular chat services in the ChatDash application.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

class PreviewIdentifier:
    """Utility class for managing preview identifiers and their prefixes.
    
    This class provides class-level functionality for registering and managing
    prefixes used in preview IDs across different services.
    """
    
    # Class-level registry
    _registered_prefixes: Set[str] = set()
    
    def __init__(self):
        """This class should not be instantiated."""
        raise NotImplementedError("PreviewIdentifier is a utility class and should not be instantiated")
    
    @classmethod
    def register_prefix(cls, prefix: str) -> None:
        """Register a new prefix. Raises ValueError if already registered."""
        if prefix in cls._registered_prefixes:
            raise ValueError(f"Prefix '{prefix}' is already registered")
        cls._registered_prefixes.add(prefix)
    
    @classmethod
    def create_id(cls, prefix: str = None, previous_id: str = None) -> str:
        """Create a new ID using either a prefix or based on a previous ID.
        
        Args:
            prefix: Service-specific prefix (must be registered first)
            previous_id: Previous ID to create alternative version from
            
        Returns:
            str: New ID in format {prefix}_{timestamp}_{suffix}
            
        Raises:
            ValueError: If prefix not registered or invalid parameters
        """
        if prefix is not None and previous_id is not None:
            raise ValueError("Cannot specify both prefix and previous_id")
        
        if prefix is None and previous_id is None:
            raise ValueError("Must specify either prefix or previous_id")
            
        if prefix is not None:
            # Creating new original ID
            if prefix not in cls._registered_prefixes:
                raise ValueError(f"Prefix '{prefix}' is not registered")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return f"{prefix}_{timestamp}_orig"
            
        # Handle alternative versions
        parts = previous_id.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid previous ID format: {previous_id}")
            
        prefix = parts[0]
        if prefix not in cls._registered_prefixes:
            raise ValueError(f"Invalid prefix in previous ID: {prefix}")
            
        # Preserve timestamp from previous ID
        timestamp = parts[1]
        if not timestamp.isdigit() or len(timestamp) != 8:
            timestamp = datetime.now().strftime('%Y%m%d')  # Fallback if timestamp invalid
            
        time = parts[2]
        if not time.isdigit() or len(time) != 6:
            time = datetime.now().strftime('%H%M%S')  # Fallback if time invalid
            
        suffix = parts[-1]
        if suffix == 'orig':
            return f"{prefix}_{timestamp}_{time}_alt1"
        elif suffix.startswith('alt'):
            try:
                n = int(suffix[3:])  # Extract number after 'alt'
                return f"{prefix}_{timestamp}_{time}_alt{n+1}"
            except ValueError:
                raise ValueError(f"Invalid alternative suffix in ID: {suffix}")
        else:
            raise ValueError(f"Invalid suffix in previous ID: {suffix}")
    
    @classmethod
    def get_prefix(cls, id_str: str) -> Optional[str]:
        """Extract the prefix from an ID string."""
        parts = id_str.split('_')
        if len(parts) >= 3 and parts[0] in cls._registered_prefixes:
            return parts[0]
        return None

@dataclass
class ServiceContext:
    """Standard format for context added to LLM prompt.
    
    This class defines the contract for service context that will be added to the LLM's
    system message. Each service should provide its context in a consistent format:
    
    Required Data Fields:
        action: str - The action that was performed (e.g., 'query_execution', 'info')
        status: str - Status of the action ('completed', 'error', etc.)
        content: Dict - Action-specific content
        
    Required Metadata Fields:
        timestamp: str - When the action was performed
        prompts: List[str] - Guidance for the LLM about this context
    """
    source: str
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize with default empty dicts and validate structure."""
        self.data = self.data or {}
        self.metadata = self.metadata or {}
        
        # Ensure required fields
        if 'action' not in self.data:
            self.data['action'] = 'unknown'
        if 'status' not in self.data:
            self.data['status'] = 'unknown'
        if 'content' not in self.data:
            self.data['content'] = {}
            
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()
        if 'prompts' not in self.metadata:
            self.metadata['prompts'] = []
    
    def format_context(self) -> str:
        """Format context data for system message.
        
        This base implementation provides specialized formatting based on the action type.
        Each action type gets its own custom formatting to provide the most relevant context
        to the LLM.
        
        Returns:
            str: Formatted context string to append to system message
        """
        # Start with source identification
        formatted = f"\n\nContext from {self.source}:"
        
        # Get the action type
        action = self.data.get('action', 'unknown')
        status = self.data.get('status', 'unknown')
        
        # Format based on action type
        if action == 'database_info':
            formatted += self._format_database_info()
        elif action == 'query_execution':
            formatted += self._format_query_execution()
        elif action == 'dataset_conversion':
            formatted += self._format_dataset_conversion()
        else:
            # Default formatting for unknown actions
            formatted += f"\nAction: {action}"
            formatted += f"\nStatus: {status}"
            
            # Add any content
            content = self.data.get('content', {})
            if content:
                formatted += "\n\nContent:"
                for key, value in content.items():
                    formatted += f"\n- {key}: {value}"
        
        # Add analysis prompts if present
        prompts = self.metadata.get('analysis_prompts', [])
        if prompts:
            formatted += "\n\nAnalysis Instructions:"
            for prompt in prompts:
                formatted += f"\n- {prompt}"
                
        return formatted
    
    def _format_database_info(self) -> str:
        """Format context for database info action."""
        formatted = "\n\nDatabase Structure Overview:"
        structure = self.data.get('structure', {})
        
        if structure:
            formatted += f"\nNumber of tables: {len(structure)}"
            formatted += "\n\nTable Summary:"
            for table, info in structure.items():
                formatted += f"\n- {table}: {info['row_count']} rows, {len(info['columns'])} columns"
                if info.get('foreign_keys'):
                    formatted += f", {len(info['foreign_keys'])} foreign key relationships"
        
        return formatted
    
    def _format_query_execution(self) -> str:
        """Format context for query execution action."""
        cmd_state = self.data.get('command_state', {})
        results = self.data.get('results', {})
        
        formatted = "\n\nQuery Execution Results:"
        formatted += f"\nQuery ID: {cmd_state.get('query_id', 'unknown')}"
        formatted += f"\nExecution Time: {cmd_state.get('execution_time', 'unknown')}"
        
        if results:
            formatted += f"\nRows Retrieved: {results.get('total_rows', 0)}"
            formatted += f"\nColumns: {', '.join(results.get('columns', []))}"
            formatted += f"\nReferenced Tables: {', '.join(results.get('referenced_tables', []))}"
            
            # Add preview if available
            preview = results.get('preview', [])
            if preview:
                formatted += "\n\nPreview of Results:"
                # Format preview as a table
                if isinstance(preview, list) and preview:
                    # Get headers
                    headers = list(preview[0].keys())
                    # Add header row
                    formatted += f"\n| {' | '.join(headers)} |"
                    # Add separator
                    formatted += f"\n| {' | '.join(['---' for _ in headers])} |"
                    # Add data rows
                    for row in preview:
                        formatted += f"\n| {' | '.join(str(row[h]) for h in headers)} |"
        
        return formatted
    
    def _format_dataset_conversion(self) -> str:
        """Format context for dataset conversion action."""
        cmd_state = self.data.get('command_state', {})
        result = self.data.get('conversion_result', {})
        
        formatted = "\n\nDataset Conversion Results:"
        formatted += f"\nSource Query ID: {cmd_state.get('query_id', 'unknown')}"
        formatted += f"\nNew Dataset Name: {result.get('dataset_name', 'unknown')}"
        formatted += f"\nRows: {result.get('rows', 0)}"
        formatted += f"\nColumns: {', '.join(result.get('columns', []))}"
        
        # Add source information
        source = result.get('source', {})
        if source:
            formatted += "\n\nSource Information:"
            formatted += f"\n- Type: {source.get('type', 'unknown')}"
            formatted += f"\n- Query ID: {source.get('query_id', 'unknown')}"
            if source.get('query_text'):
                formatted += f"\n- Query Text: {source['query_text']}"
        
        # Add preview if available
        preview = result.get('preview', [])
        if preview:
            formatted += "\n\nPreview of Dataset:"
            # Format preview as a table
            if isinstance(preview, list) and preview:
                # Get headers
                headers = list(preview[0].keys())
                # Add header row
                formatted += f"\n| {' | '.join(headers)} |"
                # Add separator
                formatted += f"\n| {' | '.join(['---' for _ in headers])} |"
                # Add data rows
                for row in preview:
                    formatted += f"\n| {' | '.join(str(row[h]) for h in headers)} |"
        
        return formatted
    
    def to_dict(self) -> dict:
        """Convert context to dictionary format.
        
        Returns the raw context data and metadata for storage or transmission.
        Services should not need to override this method.
        
        Returns:
            dict: Raw context data and metadata
        """
        return {
            'source': self.source,
            'data': self.data,
            'metadata': self.metadata
        }

@dataclass
class ServiceMessage:
    """Standard format for service-generated chat messages."""
    service: str
    content: str
    message_type: str = 'info'
    role: str = 'system'
    
    def to_chat_message(self) -> dict:
        """Convert to chat message format."""
        # Format content to explicitly mark it as a service response
        formatted_content = f"""
Type: {self.message_type}

{self.content}

"""
        
        return {
            'role': self.role,
            'content': formatted_content,
            'service': self.service,
            'type': self.message_type,
            'timestamp': datetime.now().isoformat()
        }

@dataclass
class ServiceResponse:
    """Container for service execution results."""
    messages: List[ServiceMessage]
    context: Optional[ServiceContext] = None
    store_updates: Dict[str, Any] = None
    state_updates: Dict[str, Any] = None
    
    def __post_init__(self):
        self.store_updates = self.store_updates or {}
        self.state_updates = self.state_updates or {}

class ChatService(ABC):
    """Base class for chat services."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def can_handle(self, message: str) -> bool:
        """Determine if this service can handle the message."""
        pass
    
    @abstractmethod
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Extract parameters and requirements from message."""
        pass
    
    @abstractmethod
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the service request."""
        pass
        
    def detect_content_blocks(self, text: str) -> List[Tuple[str, int, int]]:
        """Detect service-specific content blocks in text.
        
        Returns:
            List of tuples (block_content, start_pos, end_pos)
            Empty list if no blocks found
        """
        return []
        
    def add_ids_to_blocks(self, text: str) -> str:
        """Add service-specific IDs to content blocks in text.
        
        This is used to process LLM outputs that contain content
        relevant to this service.
        
        Returns:
            Text with IDs added to relevant blocks
        """
        return text

class ServiceRegistry:
    """Registry and dispatcher for chat services."""
    
    def __init__(self):
        self._services: Dict[str, ChatService] = {}
    
    def register(self, service: ChatService):
        """Register a new service."""
        self._services[service.name] = service
    
    def detect_handlers(self, message: str) -> List[Tuple[str, ChatService]]:
        """Find services that can handle the message."""
        handlers = []
        for name, service in self._services.items():
            if service.can_handle(message):
                handlers.append((name, service))
        return handlers
    
    def get_service(self, name: str) -> Optional[ChatService]:
        """Get a service by name."""
        return self._services.get(name) 