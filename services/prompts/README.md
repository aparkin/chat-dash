# ChatDash Service Prompts

This directory contains the prompt templates and management system for ChatDash services. Each service should organize its prompts following these guidelines.

## Prompt Organization

```
prompts/
├── README.md
├── base/
│   ├── system.py       # Base system prompts
│   ├── query.py        # Query processing prompts
│   └── summary.py      # Result summarization prompts
├── specialized/
│   ├── database/       # Database-specific prompts
│   ├── api/            # API service prompts
│   └── analysis/       # Analysis service prompts
└── utils/
    ├── context.py      # Context building utilities
    ├── formatting.py   # Response formatting
    └── validation.py   # Prompt validation
```

## Prompt Types

1. **System Prompts**
   - Define service behavior and constraints
   - Set response format requirements
   - Establish domain focus
   - Configure error handling

2. **Query Processing**
   - Natural language interpretation
   - Query validation
   - Parameter extraction
   - Command generation

3. **Summarization**
   - Result analysis
   - Pattern detection
   - Next step suggestions
   - Context integration

## Best Practices

1. **Context Management**
   ```python
   def build_context(service_name: str, data_context: Dict, chat_history: List[Dict]) -> Dict:
       """Build prompt context with proper token management."""
       return {
           'service': service_name,
           'data': format_data_context(data_context),
           'history': filter_relevant_history(chat_history),
           'capabilities': get_service_capabilities(service_name)
       }
   ```

2. **Prompt Templates**
   ```python
   def load_prompt(prompt_type: str, context: Dict) -> str:
       """Load and format a prompt template."""
       template = get_template(prompt_type)
       return template.format(**context)
   ```

3. **Response Formatting**
   ```python
   def format_response(response: str, format_type: str) -> str:
       """Apply consistent formatting to responses."""
       formatter = get_formatter(format_type)
       return formatter(response)
   ```

## Example Prompts

1. **System Prompt**
   ```python
   SYSTEM_PROMPT = """You are a {service_type} expert assistant.
   
   Available Data:
   {data_context}
   
   Service Capabilities:
   {capabilities}
   
   Response Requirements:
   1. Format all responses in markdown
   2. Use code blocks for commands
   3. Include rationale for suggestions
   4. Reference specific data points
   """
   ```

2. **Query Processing**
   ```python
   QUERY_PROMPT = """Analyze this user request:
   {user_query}
   
   Available Commands:
   {available_commands}
   
   Recent Context:
   {chat_history}
   
   Generate:
   1. Command interpretation
   2. Parameter suggestions
   3. Expected results
   4. Alternative approaches
   """
   ```

3. **Summarization**
   ```python
   SUMMARY_PROMPT = """Summarize these results:
   {content}
   
   Focus on:
   1. Key findings and patterns
   2. Important relationships
   3. Potential next steps
   
   Domain Context:
   {domain_context}
   
   Format as:
   1. Overview (2-3 sentences)
   2. Key Points (bullet points)
   3. Suggestions (code blocks)
   """
   ```

## Token Management

1. **Budget Allocation**
   ```python
   def allocate_tokens(total_budget: int) -> Dict[str, int]:
       """Allocate token budget across prompt components."""
       return {
           'system': int(total_budget * 0.3),
           'context': int(total_budget * 0.2),
           'history': int(total_budget * 0.3),
           'query': int(total_budget * 0.2)
       }
   ```

2. **Context Filtering**
   ```python
   def filter_context(context: Dict, max_tokens: int) -> Dict:
       """Filter context to fit within token budget."""
       return {
           k: truncate_to_tokens(v, max_tokens)
           for k, v in context.items()
       }
   ```

## Implementation Guide

1. **Service Integration**
   ```python
   class MyService(ChatService, LLMServiceMixin):
       def __init__(self):
           super().__init__("my_service")
           self.prompt_manager = PromptManager("my_service")
   
       def process_message(self, message: str, chat_history: List[Dict]) -> str:
           context = self.prompt_manager.build_context(
               message=message,
               chat_history=chat_history,
               data_context=self._get_data_context()
           )
           return self._call_llm(
               messages=context['messages'],
               system_prompt=context['system_prompt']
           )
   ```

2. **Custom Prompts**
   ```python
   class MyServicePrompts:
       """Custom prompts for specific service needs."""
       
       @staticmethod
       def create_analysis_prompt(data: Dict) -> str:
           """Create specialized analysis prompt."""
           return f"""Analyze this {data['type']} data:
           {data['content']}
           
           Focus on:
           1. {data['focus_points']}
           2. {data['relationships']}
           """
   ```

## Testing

1. **Prompt Validation**
   ```python
   def test_prompt(prompt: str, test_inputs: List[Dict]) -> bool:
       """Validate prompt with test inputs."""
       for test in test_inputs:
           result = format_prompt(prompt, test)
           if not validate_prompt(result):
               return False
       return True
   ```

2. **Response Validation**
   ```python
   def validate_response(response: str, requirements: List[str]) -> bool:
       """Validate LLM response against requirements."""
       return all(
           requirement in response
           for requirement in requirements
       )
   ```

## Common Issues

1. **Token Management**
   - Monitor token usage
   - Implement smart truncation
   - Cache common contexts
   - Use efficient formatting

2. **Context Relevance**
   - Filter irrelevant history
   - Prioritize recent interactions
   - Maintain domain focus
   - Update context dynamically

3. **Response Quality**
   - Validate command syntax
   - Check result formatting
   - Ensure complete responses
   - Handle edge cases

## Future Improvements

1. **Dynamic Prompts**
   - Context-aware templates
   - Adaptive formatting
   - Learning from feedback
   - Performance optimization

2. **Enhanced Context**
   - Better history filtering
   - Improved relevance scoring
   - Cross-service context
   - User preference integration

3. **Quality Assurance**
   - Automated testing
   - Response validation
   - Performance monitoring
   - User feedback integration 