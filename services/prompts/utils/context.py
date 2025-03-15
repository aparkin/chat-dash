"""Utilities for managing prompt context and token budgets."""

from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TokenBudget:
    """Token budget allocation for different prompt components."""
    system: int = 1000
    context: int = 800
    history: int = 1000
    query: int = 500
    response: int = 1000
    
    @property
    def total(self) -> int:
        """Get total token budget."""
        return self.system + self.context + self.history + self.query + self.response
    
    def scale(self, factor: float) -> 'TokenBudget':
        """Scale all budgets by a factor."""
        return TokenBudget(
            system=int(self.system * factor),
            context=int(self.context * factor),
            history=int(self.history * factor),
            query=int(self.query * factor),
            response=int(self.response * factor)
        )

class ContextManager:
    """Manages context for prompts with token budget awareness."""
    
    def __init__(
        self,
        service_name: str,
        token_budget: Optional[TokenBudget] = None
    ):
        self.service_name = service_name
        self.token_budget = token_budget or TokenBudget()
        self.context_cache: Dict[str, Tuple[str, datetime]] = {}
    
    def build_context(
        self,
        message: str,
        chat_history: List[Dict],
        data_context: Dict[str, any]
    ) -> Dict[str, str]:
        """Build context for a prompt within token budget.
        
        Args:
            message: Current user message
            chat_history: Chat history
            data_context: Current data context
            
        Returns:
            Dict containing formatted context sections
        """
        return {
            'message': self._truncate_text(message, self.token_budget.query),
            'history': self._format_history(chat_history),
            'context': self._format_data_context(data_context),
            'timestamp': datetime.now().isoformat()
        }
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format chat history within token budget."""
        formatted = []
        budget = self.token_budget.history
        
        for msg in reversed(history[-5:]):  # Keep last 5 messages
            formatted_msg = f"{msg['role']}: {msg['content']}"
            if len(formatted_msg) > budget:
                break
            formatted.insert(0, formatted_msg)
            budget -= len(formatted_msg)
        
        return "\n".join(formatted)
    
    def _format_data_context(self, context: Dict[str, any]) -> str:
        """Format data context within token budget."""
        formatted = []
        budget = self.token_budget.context
        
        for key, value in context.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            formatted_item = f"{key}:\n{value}"
            if len(formatted_item) > budget:
                continue
            formatted.append(formatted_item)
            budget -= len(formatted_item)
        
        return "\n\n".join(formatted)
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token budget."""
        if len(text) <= max_tokens:
            return text
        return text[:max_tokens - 3] + "..."
    
    def cache_context(self, key: str, context: str) -> None:
        """Cache context for reuse."""
        self.context_cache[key] = (context, datetime.now())
    
    def get_cached_context(self, key: str) -> Optional[str]:
        """Get cached context if available and recent."""
        if key in self.context_cache:
            context, timestamp = self.context_cache[key]
            if (datetime.now() - timestamp).seconds < 300:  # 5 minute cache
                return context
            del self.context_cache[key]
        return None

class ResponseFormatter:
    """Formats LLM responses for consistency."""
    
    @staticmethod
    def format_command_response(
        command: str,
        params: Dict[str, any],
        result: str
    ) -> str:
        """Format command execution response."""
        return f"""### Command Execution
```
Command: {command}
Parameters: {json.dumps(params, indent=2)}
```

### Result
{result}
"""
    
    @staticmethod
    def format_error_response(
        error: str,
        context: str,
        solution: str
    ) -> str:
        """Format error response."""
        return f"""### Error
```
{error}
```

### Context
{context}

### Solution
{solution}
"""
    
    @staticmethod
    def format_summary(
        overview: str,
        key_points: List[str],
        recommendations: List[str]
    ) -> str:
        """Format summary response."""
        points = "\n".join(f"- {point}" for point in key_points)
        recs = "\n".join(f"- {rec}" for rec in recommendations)
        
        return f"""### Overview
{overview}

### Key Points
{points}

### Recommendations
{recs}
"""

def estimate_tokens(text: str) -> int:
    """Estimate token count for text.
    
    This is a rough estimation using character count.
    For more accurate counts, use a proper tokenizer.
    """
    return len(text) // 4  # Rough estimate of 4 chars per token 