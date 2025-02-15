"""
LLM Service Mixin for ChatDash services.

This module provides a mixin class that services can use to add LLM capabilities
for processing messages and generating responses.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
import tiktoken
import openai
import traceback

# Load environment variables
load_dotenv()

@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    model_name: str
    temperature: float

class LLMServiceMixin(ABC):
    """Mixin class providing LLM capabilities to services."""
    
    def __init__(self, service_name: str):
        """Initialize LLM service configuration.
        
        Args:
            service_name: Name of the service using this mixin
        """
        self.service_name = service_name
        self.llm_config = self._load_llm_config()
        # Use cl100k_base tokenizer as it's suitable for newer models
        self._encoding = tiktoken.get_encoding("cl100k_base")
        
        # Use the same OpenAI client configuration as ChatDash
        openai_config = {
            'api_key': os.getenv('CBORG_API_KEY', ''),
            'base_url': os.getenv('CBORG_BASE_URL', "https://api.cborg.lbl.gov")
        }
        self._client = openai.OpenAI(**openai_config)
        
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from environment variables."""
        # Get service-specific model name or fall back to default
        model_env_var = f"{self.service_name.upper()}_MODEL"
        model_name = os.getenv(model_env_var, "anthropic/claude-sonnet")  # Use Claude as default
        
        # Load temperature with service-specific override
        temperature = float(os.getenv(f"{self.service_name.upper()}_TEMPERATURE", "0.4"))
        
        return LLMConfig(
            model_name=model_name,
            temperature=temperature
        )
    
    def _call_llm(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
        """Make an API call to the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt to prepend
            
        Returns:
            str: The LLM's response
            
        Raises:
            Exception: If the API call fails
        """
        try:
            print("\n=== Debug: LLM Call ===")
            # Prepare messages
            formatted_messages = []
            
            # Add system prompt first if provided
            if system_prompt and system_prompt.strip():
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt.strip()
                })
            
            # Process and validate each message
            for msg in messages:
                # Skip system message if we already added system prompt
                if msg.get("role") == "system" and system_prompt:
                    continue
                    
                content = msg.get('content', '').strip()
                if not content:  # Skip empty messages
                    print(f"Warning: Skipping empty message with role {msg.get('role')}")
                    continue
                
                # Ensure role is valid
                role = msg.get("role", "user")
                if role not in ["system", "user", "assistant"]:
                    role = "user"  # Default to user for unknown roles
                
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
            
            # Ensure we have at least one message
            if not formatted_messages:
                raise ValueError("No valid messages to send to LLM")
            
            print(f"Prepared {len(formatted_messages)} messages")
            print("Message roles:", [m["role"] for m in formatted_messages])
            print("Message lengths:", [len(m["content"]) for m in formatted_messages])
            
            # Call the API using the same pattern as ChatDash
            response = self._client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=formatted_messages,
                temperature=self.llm_config.temperature,
                max_tokens=8192  # Same as ChatDash
            )
            
            result = response.choices[0].message.content
            if not result or not result.strip():
                raise ValueError("LLM returned empty response")
                
            print(f"Got response, length: {len(result)}")
            return result.strip()
            
        except Exception as e:
            print(f"LLM API call failed: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            raise Exception(f"LLM API call failed: {str(e)}")
    
    def count_tokens(self, text: str) -> int:
        """Estimate the number of tokens in the text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        return len(self._encoding.encode(text))
    
    def _filter_relevant_history(self, chat_history: List[Dict[str, Any]], 
                               max_tokens: Optional[int] = None) -> List[Dict[str, Any]]:
        """Filter chat history to include relevant messages for this service.
        
        Args:
            chat_history: Complete chat history
            max_tokens: Maximum number of tokens to include in history
            
        Returns:
            List[Dict[str, Any]]: Filtered chat history
        """
        # Filter messages related to this service
        relevant_messages = [
            msg for msg in chat_history 
            if msg.get('service') == self.service_name
        ]
        
        if max_tokens:
            # Count tokens from most recent messages until we hit the limit
            filtered_messages = []
            token_count = 0
            
            for msg in reversed(relevant_messages):
                msg_tokens = self.count_tokens(msg.get('content', ''))
                if token_count + msg_tokens > max_tokens:
                    break
                    
                filtered_messages.insert(0, msg)
                token_count += msg_tokens
                
            return filtered_messages
            
        return relevant_messages
    
    @abstractmethod
    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process a message using the service's LLM.
        
        Args:
            message: The message to process
            chat_history: List of previous chat messages
            
        Returns:
            str: The LLM's response
        """
        pass
    
    @abstractmethod
    def summarize(self, content: str, chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of the given content.
        
        Args:
            content: The content to summarize
            chat_history: List of previous chat messages
            
        Returns:
            str: The generated summary
        """
        pass 