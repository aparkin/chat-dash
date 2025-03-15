"""Template Service for ChatDash.

This module provides a template for creating new ChatDash services.
It demonstrates best practices and required implementations for:

1. Message Handling:
   - Command detection
   - Request parsing
   - Execution flow
   - Error handling

2. LLM Integration:
   - Context management
   - Query processing
   - Response generation
   - Result summarization

3. Service Framework:
   - Required abstract methods
   - Common utilities
   - State management
   - Response formatting

Usage:
    ```python
    class MyNewService(ChatService, LLMServiceMixin):
        def __init__(self):
            ChatService.__init__(self, "my_service")
            LLMServiceMixin.__init__(self, "my_service")
    ```

Configuration:
    The following environment variables can be set:
    - {SERVICE_NAME}_MODEL: Model to use (default: "anthropic/claude-sonnet")
    - {SERVICE_NAME}_TEMPERATURE: Temperature setting (default: 0.4)
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import json
from enum import Enum, auto

from .base import ChatService, ServiceResponse, ServiceMessage, MessageType
from .llm_service import LLMServiceMixin

class RequestType(Enum):
    """Types of requests this service can handle."""
    INFO = auto()           # Service information request
    DIRECT_QUERY = auto()   # Direct query execution
    NATURAL_QUERY = auto()  # Natural language query
    QUERY_SEARCH = auto()   # Search for previous queries
    QUERY_EXPLAIN = auto()  # Explain query results
    DATASET_CONVERT = auto() # Convert results to dataset

class TemplateService(ChatService, LLMServiceMixin):
    """Template implementation of a ChatDash service.
    
    This class demonstrates the required implementations and best practices
    for creating new ChatDash services.
    """
    
    def __init__(self):
        """Initialize service with name and configuration."""
        ChatService.__init__(self, "template")
        LLMServiceMixin.__init__(self, "template")
        
        # Command detection patterns
        self.query_block_re = re.compile(r'```template\s*(.*?)\s*```', re.DOTALL)
        self.execution_res = [
            re.compile(pattern) for pattern in [
                r'^template\.(?:search|query|execute)\s*$',
                r'^template\.(?:search|query|execute)\s+template_query_\d{8}_\d{6}(?:_orig|_alt\d+)\b'
            ]
        ]
        
        # Service state
        self._api_schema = None  # Cache for API/data schema
        self._capabilities = None  # Cache for service capabilities
    
    def can_handle(self, message: str) -> bool:
        """Check if message can be handled by this service.
        
        Detects:
        1. Service code blocks (```template {...}```)
        2. Execution commands (template.search, template.query)
        3. Natural language queries (template: ...)
        4. Service info requests (tell me about template)
        
        Args:
            message: The message to check
            
        Returns:
            bool: True if service can handle this message
        """
        message = message.strip()
        message_lower = message.lower()
        
        # Check for code blocks
        if self.query_block_re.search(message):
            return True
            
        # Check for execution commands
        for pattern in self.execution_res:
            if pattern.search(message):
                return True
                
        # Check for natural language query
        if message_lower.startswith('template:'):
            return True
            
        # Check for service info
        if message_lower == 'tell me about template':
            return True
            
        return False
    
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Parse message into request parameters.
        
        Extracts:
        1. Request type (info/query/search)
        2. Query content or ID
        3. Additional parameters
        
        Args:
            message: The message to parse
            
        Returns:
            Dict containing request parameters
            
        Raises:
            ValueError: If message cannot be parsed
        """
        message = message.strip()
        message_lower = message.lower()
        
        # Check for code blocks
        if match := self.query_block_re.search(message):
            try:
                query = json.loads(match.group(1).strip())
                return {
                    'type': RequestType.DIRECT_QUERY,
                    'query': query,
                    'raw_text': match.group(1).strip()
                }
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid query format: {str(e)}")
        
        # Check for execution commands
        if message_lower in ['template.search', 'template.query', 'template.execute']:
            return {
                'type': RequestType.QUERY_SEARCH,
                'query_id': None  # Use most recent
            }
        
        # Check for specific query execution
        if match := re.match(
            r'^template\.(?:search|query|execute)\s+(template_query_\d{8}_\d{6}(?:_orig|_alt\d+))\b',
            message_lower
        ):
            return {
                'type': RequestType.QUERY_SEARCH,
                'query_id': match.group(1)
            }
        
        # Check for service info
        if message_lower == 'tell me about template':
            return {
                'type': RequestType.INFO
            }
        
        # Check for natural language query
        if message_lower.startswith('template:'):
            return {
                'type': RequestType.NATURAL_QUERY,
                'query': message[9:].strip()
            }
        
        # Check for dataset conversion
        if match := re.match(
            r'^convert\s+(template_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b',
            message_lower
        ):
            return {
                'type': RequestType.DATASET_CONVERT,
                'query_id': match.group(1)
            }
        
        raise ValueError(f"Unable to parse request: {message}")
    
    def execute(self, request: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the parsed request.
        
        Args:
            request: Request parameters from parse_request
            context: Execution context with:
                - chat_history: List of chat messages
                - query_store: Previously executed queries
                - other service-specific context
                
        Returns:
            ServiceResponse with:
            - messages: List of service messages
            - context_updates: Updates to context
            - store_updates: Updates to query store
        """
        try:
            # Store context for use in LLM calls
            self.context = context
            request_type = request['type']
            
            # Handle different request types
            if request_type == RequestType.INFO:
                return self._handle_info_request(context)
            elif request_type == RequestType.DIRECT_QUERY:
                return self._handle_direct_query(request['query'], context)
            elif request_type == RequestType.NATURAL_QUERY:
                return self._handle_natural_query(request['query'], context)
            elif request_type == RequestType.QUERY_SEARCH:
                return self._handle_query_execution(request['query_id'], context)
            elif request_type == RequestType.DATASET_CONVERT:
                return self._handle_dataset_conversion(request['query_id'], context)
            else:
                raise ValueError(f"Unknown request type: {request_type}")
                
        except Exception as e:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error executing request: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process a message using the service's LLM.
        
        Required by LLMServiceMixin. Used for:
        - Natural language query interpretation
        - Query explanations
        - Result analysis
        
        Args:
            message: The message to process
            chat_history: List of previous chat messages
            
        Returns:
            str: The LLM's response
        """
        try:
            # Get data context
            data_context = self._get_data_context()
            
            # Create focused system prompt
            system_prompt = f"""You are helping with a template service query.

Data Context:
{json.dumps(data_context, indent=2)}

Please:
1. Interpret the user's request
2. Suggest appropriate queries or actions
3. Explain the expected results

Keep responses clear and focused on the specific service capabilities."""

            # Call LLM with context
            response = self._call_llm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ]
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
    
    def summarize(self, content: str, chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of query results or other content.
        
        Required by LLMServiceMixin. Used to summarize:
        - Query results
        - Data analysis
        - Error explanations
        
        Args:
            content: The content to summarize
            chat_history: List of previous chat messages
            
        Returns:
            str: Generated summary
        """
        try:
            # Create focused summary prompt
            prompt = f"""Please provide a concise summary of the following content:

{content}

Focus on:
1. Key findings and patterns
2. Important relationships
3. Potential next steps

Keep the summary clear and actionable."""

            # Get summary from LLM
            summary = self._call_llm([{"role": "user", "content": prompt}])
            return summary.strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def get_help_text(self) -> str:
        """Get help text for service capabilities.
        
        Returns markdown-formatted help text showing:
        1. Available commands
        2. Query formats
        3. Example usage
        """
        return """
🔧 **Template Service**
- `template: {natural language query}` - Ask questions in plain English
- ```template {...}``` - Execute direct query
- `template.search [query_id]` - Execute saved query
- `convert {query_id} to dataset` - Convert results to dataset
"""

    def get_llm_prompt_addition(self) -> str:
        """Get service documentation for LLM prompts.
        
        Returns focused documentation on:
        1. Command patterns
        2. Query formats
        3. Critical details
        4. Limitations
        """
        return """
Template Service Capabilities:
- Natural language query interpretation
- Direct query execution with JSON format
- Result analysis and summarization
- Dataset conversion and integration

Command Patterns:
- template: {question} - Natural language queries
- ```template {...}``` - Direct query blocks
- template.search [id] - Execute saved queries
- convert {id} to dataset - Create datasets

Important Notes:
- JSON queries must follow service schema
- Results are automatically summarized
- Queries are saved for future reference
"""

    def _get_data_context(self) -> Dict[str, Any]:
        """Get current data context for LLM prompts.
        
        Returns dict containing:
        1. Available data types
        2. Query patterns
        3. Current state
        """
        return {
            "data_types": [
                "example_type_1",
                "example_type_2"
            ],
            "query_patterns": {
                "pattern1": "Description of pattern 1",
                "pattern2": "Description of pattern 2"
            },
            "limitations": [
                "Limitation 1",
                "Limitation 2"
            ]
        }
    
    def _handle_info_request(self, context: Dict[str, Any]) -> ServiceResponse:
        """Handle request for service information."""
        info_text = """
# Template Service

This service demonstrates the standard patterns for:
1. Query handling
2. LLM integration
3. Result processing

## Available Commands
- Natural language: template: {query}
- Direct queries: ```template {...}```
- Query execution: template.search [id]
- Dataset conversion: convert {id} to dataset

## Data Types
- Type 1: Description
- Type 2: Description

## Query Patterns
- Pattern 1: Usage
- Pattern 2: Usage
"""
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service=self.name,
                    content=info_text,
                    message_type=MessageType.INFO
                )
            ]
        )
    
    def _handle_direct_query(self, query: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Handle direct query execution."""
        # Implementation specific to service
        pass
    
    def _handle_natural_query(self, query: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle natural language query."""
        # Implementation specific to service
        pass
    
    def _handle_query_execution(self, query_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle saved query execution."""
        # Implementation specific to service
        pass
    
    def _handle_dataset_conversion(self, query_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle conversion of results to dataset."""
        # Implementation specific to service
        pass 