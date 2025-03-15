"""
LLM Service Mixin for ChatDash services.

This module provides a mixin class that services can use to add LLM capabilities
for processing messages and generating responses. It implements a standardized approach
to LLM integration with the following key features:

Architecture:
1. Context Management:
   - Smart filtering of chat history
   - Token budget management
   - Domain-specific context building

2. Configuration:
   - Service-specific model selection
   - Environment-based configuration
   - Flexible temperature settings

3. Error Handling:
   - Structured retry logic
   - Validation history tracking
   - Clear error hierarchies

4. State Management:
   - Immutable state updates
   - Clear state ownership
   - Centralized state tracking

Usage:
    ```python
    class MyService(ChatService, LLMServiceMixin):
        def __init__(self):
            ChatService.__init__(self, "my_service")
            LLMServiceMixin.__init__(self, "my_service")
            
        def process_message(self, message: str, chat_history: List[Dict]) -> str:
            # Build context
            system_prompt, context_messages, limits = self._prepare_llm_context(
                message, chat_history, domain_context
            )
            
            # Get LLM response with retry logic
            response = self._call_llm(context_messages, system_prompt)
            
            # Process and validate response
            return self._process_response(response)
    ```

Configuration:
    The following environment variables can be set per service:
    - {SERVICE_NAME}_MODEL: Model to use (default: "anthropic/claude-sonnet")
    - {SERVICE_NAME}_TEMPERATURE: Temperature setting (default: 0.4)
    Example:
    ```bash
    DATASET_MODEL="anthropic/claude-opus"
    DATASET_TEMPERATURE="0.7"
    ```

Dependencies:
    - openai: OpenAI API client
    - tiktoken: Token counting
    - python-dotenv: Environment variable management
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
from pathlib import Path

# Load environment variables
# Find the project root directory (where .env is located)
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'

# Try to load from .env file
load_dotenv(dotenv_path=dotenv_path)

# OpenAI Settings
if True:  # Toggle for development environment
    CBORG=True
    OPENAI_BASE_URL = os.getenv('CBORG_BASE_URL', "https://api.cborg.lbl.gov")
    OPENAI_API_KEY = os.getenv('CBORG_API_KEY', '')  # Must be set in environment
else:  # Production environment
    CBORG=False
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Must be set in environment

print(f"OPENAI_BASE_URL: {OPENAI_BASE_URL}")
# TODO: why isn't the above getting the right BASE_URL?
#OPENAI_BASE_URL='https://api.openai.com/v1'

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

@dataclass
class LLMConfig:
    """Configuration for LLM service.
    
    Attributes:
        model_name: Name of the LLM model to use
        temperature: Temperature setting for response generation
    """
    model_name: str
    temperature: float

class LLMServiceMixin(ABC):
    """Mixin class providing LLM capabilities to services.
    
    This mixin provides a standardized way to integrate LLM capabilities into
    ChatDash services. It handles:
    1. LLM configuration and initialization
    2. Context management and token budgeting
    3. Message processing and response generation
    4. Error handling and retry logic
    
    Implementation Requirements:
    1. Services must initialize both ChatService and LLMServiceMixin
    2. Services must implement process_message() for custom message handling
    3. Services should use _prepare_llm_context() for context management
    4. Services should use _call_llm() for all LLM interactions
    
    Token Management:
    The mixin implements smart token management to:
    - Stay within model context limits
    - Reserve space for system prompts
    - Allow for validation iterations
    - Maintain relevant chat history
    
    Error Handling:
    Provides structured error handling with:
    - Retry logic for validation failures
    - History tracking for validation errors
    - Clear error messages and tracebacks
    """
    
    def __init__(self, service_name: str):
        """Initialize LLM service configuration.
        
        Args:
            service_name: Name of the service using this mixin.
                Used for service-specific configuration.
        """
        self.service_name = service_name
        self.llm_config = self._load_llm_config()
        # Use cl100k_base tokenizer as it's suitable for newer models
        self._encoding = tiktoken.get_encoding("cl100k_base")
        
        # Use the same OpenAI client configuration as ChatDash
        openai_config = {
            'api_key': OPENAI_API_KEY,
            'base_url': OPENAI_BASE_URL
        }
        self._client = openai.OpenAI(**openai_config)
        
        # Initialize context
        self.context = {}
        
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
        """Make an API call to the LLM."""
        try:
            print("\n=== Debug: LLM Call ===", flush=True)
            print(f"Service: {self.service_name}", flush=True)
            
            # Prepare messages
            formatted_messages = []
            
            # Add system prompt first if provided
            if system_prompt and system_prompt.strip():
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt.strip()
                })
                print("\nSystem prompt:", flush=True)
                print("---", flush=True)
                print(system_prompt.strip()[0:200], flush=True)
                print("---", flush=True)
            
            # Process and validate each message
            for msg in messages:
                # Skip system message if we already added system prompt
                if msg.get("role") == "system" and system_prompt:
                    print(f"Skipping system message due to system prompt", flush=True)
                    continue
                    
                content = msg.get('content', '').strip()
                if not content:  # Skip empty messages
                    print(f"Warning: Skipping empty message with role {msg.get('role')}", flush=True)
                    continue
                
                # Ensure role is valid
                role = msg.get("role", "user")
                if role not in ["system", "user", "assistant"]:
                    print(f"Warning: Invalid role '{role}', defaulting to 'user'", flush=True)
                    role = "user"  # Default to user for unknown roles
                
                formatted_messages.append({
                    "role": role,
                    "content": content
                })
                print(f"\nAdded message with role '{role}':", flush=True)
                print("---", flush=True)
                print(content, flush=True)
                print("---", flush=True)
            
            # Ensure we have at least one message
            if not formatted_messages:
                raise ValueError("No valid messages to send to LLM")
            
            print(f"\nPrepared {len(formatted_messages)} messages", flush=True)
            print("Message roles:", [m["role"] for m in formatted_messages], flush=True)
            print("Message lengths:", [len(m["content"]) for m in formatted_messages], flush=True)
            
            # Get model from context or config
            model = self.context.get('model', self.llm_config.model_name)
            print(f"\nUsing model: {model}", flush=True)
            
            if not CBORG:
                # Map model names to OpenAI API model IDs
                model_mapping = {   
                    'anthropic/claude-sonnet': 'o3-mini',  # Map to GPT-4 as closest equivalent
                    'anthropic/claude-opus': 'o3-mini',
                    'anthropic/claude-haiku': 'gpt-3.5-turbo',
                    'openai/gpt-4o': 'gpt-4o',
                    'openai/gpt-4o-mini': 'gpt-4o-mini',
                    'openai/o1': 'o1',
                    'openai/o1-mini': 'o1-mini',
                    'lbl/cborg-chat:latest': 'gpt-4o',  # Keep as is for CBORG
                    'lbl/cborg-coder:latest': 'gpt-4o',
                    'lbl/cborg-deepthought:latest': 'gpt-4o'
                }
                model = model_mapping.get(model, 'gpt-4')
                print(f"Mapped to OpenAI model: {model}", flush=True)
            
            print("\nMaking API call...", flush=True)
            try:
                print("API parameters:", flush=True)
                print(f"- Model: {model}", flush=True)
                print(f"- Temperature: {self.llm_config.temperature}", flush=True)
                print(f"- Messages: {len(formatted_messages)}", flush=True)
                
                # Models that don't support temperature parameter
                no_temperature_models = ['o1', 'o1-mini']
                
                # Prepare API call parameters
                api_params = {
                    'model': model,
                    'messages': formatted_messages
                }
                
                # Only include temperature for models that support it
                if model not in no_temperature_models:
                    api_params['temperature'] = self.llm_config.temperature
                
                try:
                    response = self._client.chat.completions.create(**api_params)
                except Exception as api_error:
                    error_msg = str(api_error)
                    if "temperature" in error_msg.lower():
                        print(f"Temperature parameter error detected. Retrying without temperature.", flush=True)
                        # Remove temperature and retry
                        if 'temperature' in api_params:
                            del api_params['temperature']
                        response = self._client.chat.completions.create(**api_params)
                    else:
                        # Re-raise if it's not a temperature-related error
                        raise
                
                # Extract response content
                response_content = response.choices[0].message.content.strip()
                print("\nExtracted content:", flush=True)
                print("---", flush=True)
                print(repr(response_content), flush=True)
                print("---", flush=True)
                
                if not response_content:
                    raise ValueError("LLM returned empty response")
                
                return response_content
                
            except Exception as e:
                print(f"\nAPI call error: {str(e)}", flush=True)
                print(f"Error type: {type(e)}", flush=True)
                print(f"Traceback:\n{traceback.format_exc()}", flush=True)
                raise
                
        except Exception as e:
            print(f"\nTop-level error in _call_llm: {str(e)}", flush=True)
            print(f"Error type: {type(e)}", flush=True)
            print(f"Traceback:\n{traceback.format_exc()}", flush=True)
            raise
    
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