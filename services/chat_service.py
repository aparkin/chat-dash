"""Chat service implementation for ChatDash."""

from typing import Dict, Any, Optional, List
from .base import ChatService, ServiceMessage, ServiceResponse
from .llm_service import LLMServiceMixin

class ChatDashService(ChatService, LLMServiceMixin):
    """Chat service implementation for ChatDash."""
    
    def __init__(self):
        """Initialize the chat service."""
        ChatService.__init__(self, "ChatDash")
        LLMServiceMixin.__init__(self, "ChatDash")
        
    def can_handle(self, message: ServiceMessage) -> bool:
        """Check if this service can handle the message.
        
        Args:
            message: The service message to check
            
        Returns:
            True if this service can handle the message
        """
        return True  # For testing, handle all messages
        
    async def process_message(self, message: ServiceMessage) -> ServiceResponse:
        """Process a message.
        
        Args:
            message: The service message to process
            
        Returns:
            ServiceResponse with results
        """
        params = await self.parse_request(message)
        return await self.execute(params)
        
    async def parse_request(self, message: ServiceMessage) -> Dict[str, Any]:
        """Parse the user's request into parameters.
        
        Args:
            message: The service message to parse
            
        Returns:
            Dict containing parsed parameters
        """
        # For testing, just return the message content
        return {"content": message.content}
        
    async def execute(self, params: Dict[str, Any]) -> ServiceResponse:
        """Execute the parsed request.
        
        Args:
            params: Dictionary of parameters from parse_request
            
        Returns:
            ServiceResponse with results
        """
        # For testing, call LLM with simple message
        messages = [{"role": "user", "content": params["content"]}]
        response = await self._call_llm(messages, {})
        return ServiceResponse(messages=[ServiceMessage(
            service=self.service_name,
            content=response
        )])
        
    def get_help_text(self) -> str:
        """Get help text for this service."""
        return "ChatDash service for testing. Responds with mock LLM responses."
        
    def get_llm_prompt_addition(self) -> str:
        """Get additional text to add to LLM prompts."""
        return "You are a test chat service."
        
    async def summarize(self, content: str) -> str:
        """Summarize content.
        
        Args:
            content: The content to summarize
            
        Returns:
            Summarized content
        """
        messages = [{"role": "user", "content": f"Please summarize: {content}"}]
        return await self._call_llm(messages, {}) 