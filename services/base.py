"""
Base classes and infrastructure for chat services.

This module provides the foundational classes and interfaces for implementing
modular chat services in the ChatDash application.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum, auto

class MessageType(Enum):
    """Enum defining standard message types for service responses.
    
    Types:
    - RESULT: Primary output from a service (search results, analysis output, etc.)
    - SUMMARY: Secondary analysis or interpretation (LLM summaries, insights, etc.)
    - ERROR: Error messages and warnings
    - INFO: General informational messages
    - DEBUG: Debug information (only shown in development)
    - WARNING: Warning messages
    - PREVIEW: Preview of query results or data with ID for reference
    - SUGGESTION: Suggested queries or actions
    """
    RESULT = auto()
    SUMMARY = auto()
    ERROR = auto()
    INFO = auto()
    DEBUG = auto()
    WARNING = auto()
    PREVIEW = auto()
    SUGGESTION = auto()
    
    def __str__(self) -> str:
        """Return lowercase string representation for compatibility."""
        return self.name.lower()

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
        # Find the registered prefix in the previous_id
        found_prefix = None
        for registered_prefix in cls._registered_prefixes:
            if previous_id.startswith(registered_prefix + '_'):
                found_prefix = registered_prefix
                break
                
        if not found_prefix:
            raise ValueError(f"No registered prefix found in previous ID: {previous_id}")
            
        # Remove prefix and split remaining parts
        remaining = previous_id[len(found_prefix) + 1:]  # +1 for the underscore
        parts = remaining.split('_')
        
        if len(parts) < 3:  # We need at least timestamp, time, and suffix
            raise ValueError(f"Invalid previous ID format: {previous_id}")
            
        # Extract timestamp and time
        timestamp = parts[0]
        time = parts[1]
        suffix = parts[-1]
        
        # Validate timestamp and time
        if not timestamp.isdigit() or len(timestamp) != 8:
            timestamp = datetime.now().strftime('%Y%m%d')  # Fallback if timestamp invalid
        if not time.isdigit() or len(time) != 6:
            time = datetime.now().strftime('%H%M%S')  # Fallback if time invalid
            
        # Generate new suffix
        if suffix == 'orig':
            new_suffix = 'alt1'
        elif suffix.startswith('alt'):
            try:
                n = int(suffix[3:])  # Extract number after 'alt'
                new_suffix = f'alt{n+1}'
            except ValueError:
                raise ValueError(f"Invalid alternative suffix in ID: {suffix}")
        else:
            raise ValueError(f"Invalid suffix in previous ID: {suffix}")
            
        return f"{found_prefix}_{timestamp}_{time}_{new_suffix}"
    
    @classmethod
    def get_prefix(cls, id_str: str) -> Optional[str]:
        """Extract the prefix from an ID string."""
        parts = id_str.split('_')
        if len(parts) >= 3 and parts[0] in cls._registered_prefixes:
            return parts[0]
        return None

@dataclass
class ServiceMessage:
    """Standard format for service-generated chat messages.
    
    Attributes:
        service: Name of the service generating the message
        content: Message content (markdown formatted text)
        message_type: Type of message (see MessageType enum)
        role: Message role in chat (system, assistant, user)
        
    Message Types:
    - RESULT: Primary output (search results, analysis, etc.)
    - SUMMARY: Secondary analysis (LLM summaries, insights)
    - ERROR: Error messages and warnings
    - INFO: General information
    - DEBUG: Debug information (development only)
    
    Example:
        ```python
        # Result message with primary output
        ServiceMessage(
            service="search",
            content="Found 5 matches...",
            message_type=MessageType.RESULT
        )
        
        # Summary message with LLM analysis
        ServiceMessage(
            service="search",
            content="Analysis shows...",
            message_type=MessageType.SUMMARY
        )
        ```
    """
    service: str
    content: str
    message_type: MessageType = MessageType.INFO
    role: str = 'assistant'
    
    def __post_init__(self):
        """Validate message content after initialization."""
        if not self.content or not self.content.strip():
            raise ValueError("Service message content cannot be empty")
        self.content = self.content.strip()
        
        # Convert string message types to enum
        if isinstance(self.message_type, str):
            try:
                self.message_type = MessageType[self.message_type.upper()]
            except KeyError:
                self.message_type = MessageType.INFO
    
    def to_chat_message(self) -> dict:
        """Convert to chat message format."""
        # Format content to explicitly mark it as a service response
        formatted_content = f"Type: {str(self.message_type)}\n\n{self.content}"
        
        return {
            'role': self.role,
            'content': formatted_content.strip(),
            'service': self.service,
            'type': str(self.message_type),
            'timestamp': datetime.now().isoformat()
        }

@dataclass
class ServiceResponse:
    """Container for service execution results."""
    messages: List[ServiceMessage]
    store_updates: Dict[str, Any] = None
    state_updates: Dict[str, Any] = None
    
    def __post_init__(self):
        self.store_updates = self.store_updates or {}
        self.state_updates = self.state_updates or {}

class ChatService(ABC):
    """Base class for chat services."""
    
    def __init__(self, name: str):
        self.name = name
    
    @staticmethod
    def _get_creation_timestamp() -> str:
        """Get standardized creation timestamp for dataset metadata.
        
        Returns:
            str: Timestamp in format YYYY-MM-DD HH:MM:SS
        """
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def _validate_dataset_metadata(metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate required fields in dataset metadata.
        
        Required fields:
        - source: str
        - creation_time: str (YYYY-MM-DD HH:MM:SS)
        - rows: int
        - columns: List[str]
        
        Args:
            metadata: Dataset metadata dictionary
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        required_fields = {
            'source': str,
            'creation_time': str,
            'rows': int,
            'columns': list
        }
        
        for field, expected_type in required_fields.items():
            if field not in metadata:
                return False, f"Missing required metadata field: {field}"
            if not isinstance(metadata[field], expected_type):
                return False, f"Invalid type for metadata field {field}: expected {expected_type.__name__}"
        
        return True, None
    
    @abstractmethod
    def can_handle(self, message: str) -> bool:
        """Determine if this service can handle the message."""
        pass
    
    @abstractmethod
    async def parse_request(self, message: str) -> Dict[str, Any]:
        """Extract parameters and requirements from message."""
        pass
    
    @abstractmethod
    async def execute(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the parsed request."""
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

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get service status information.
        
        This is an optional method that services can implement to report their status.
        The default implementation returns None, indicating the service is ready.
        
        Services that need time to initialize (like MONet) can override this.
        
        Returns:
            Optional[Dict[str, Any]]: Status information or None if default status is acceptable
        """
        return None

    @abstractmethod
    def get_help_text(self) -> str:
        """Get a compact help text for this service suitable for the help string in ChatDash.
        
        This method should return a concise, markdown-formatted string that:
        1. Shows all available commands with exact syntax
        2. Groups related commands together
        3. Includes required and optional parameters
        4. Uses consistent formatting with other services
        
        Returns:
            str: Markdown formatted help text for the service
        """
        pass
    
    @abstractmethod
    def get_llm_prompt_addition(self) -> str:
        """Get text to add to the LLM prompt to explain this service's capabilities.
        
        This method should return a compact, focused string that:
        1. Lists exact command patterns the LLM should suggest
        2. Provides critical details about command usage
        3. Notes important limitations or requirements
        4. Avoids redundant or obvious information
        
        Returns:
            str: Detailed service documentation for the LLM prompt
        """
        pass

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

    def get_help_text(self) -> str:
        """Get combined help text from all registered services.
        
        Orders services in a logical sequence:
        1. Dataset operations (core data handling)
        2. Database operations (data querying)
        3. Literature operations (search and analysis)
        4. Visualization (data display)
        5. Index search (unified search)
        6. Store report (system status)
        
        Returns:
            str: Combined markdown-formatted help text from all services
        """
        # Define service order (core services first)
        service_order = [
            'dataset',      # Core data handling
            'database',     # Data querying
            'literature',   # Literature search
            'visualization',# Data visualization
            'index_search', # Unified search
            'store_report' # System status
        ]
        
        help_texts = []
        # First add services in specified order
        for service_name in service_order:
            service = self._services.get(service_name)
            if service:
                service_help = service.get_help_text()
                if service_help and service_help.strip():
                    help_texts.append(service_help)
        
        # Then add any remaining services alphabetically
        remaining_services = sorted(
            (name, service) for name, service in self._services.items()
            if name not in service_order
        )
        for name, service in remaining_services:
            service_help = service.get_help_text()
            if service_help and service_help.strip():
                help_texts.append(service_help)
        
        return "\n\n".join(help_texts) 