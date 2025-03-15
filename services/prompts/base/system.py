"""Base system prompts for ChatDash services."""

from typing import Dict, List

BASE_SYSTEM_PROMPT = """You are a specialized {service_type} assistant for the ChatDash platform.

Service Capabilities:
{capabilities}

Available Commands:
{commands}

Response Requirements:
1. Format all responses in markdown
2. Use code blocks for commands and data
3. Include rationale for suggestions
4. Reference specific data points
5. Maintain consistent formatting

Current Context:
{context}
"""

COMMAND_PROCESSING_PROMPT = """Process this command request:
{command}

Available Parameters:
{parameters}

Validation Rules:
{validation_rules}

Expected Output:
1. Command interpretation
2. Parameter validation
3. Execution plan
4. Expected results
"""

NATURAL_QUERY_PROMPT = """Interpret this natural language query:
{query}

Domain Knowledge:
{domain_context}

Available Operations:
{operations}

Generate:
1. Query interpretation
2. Suggested commands
3. Expected insights
4. Alternative approaches
"""

SUMMARY_BASE_PROMPT = """Summarize these results:
{content}

Focus Areas:
1. Key findings
2. Important patterns
3. Actionable insights
4. Next steps

Domain Context:
{domain_context}

Format as:
1. Brief overview
2. Key points (bulleted)
3. Recommendations
4. Related queries
"""

ERROR_HANDLING_PROMPT = """Handle this error condition:
{error}

Context:
{error_context}

Generate:
1. Error analysis
2. Recovery steps
3. User guidance
4. Prevention suggestions
"""

def build_system_prompt(
    service_type: str,
    capabilities: Dict[str, str],
    commands: List[str],
    context: Dict[str, any]
) -> str:
    """Build a system prompt for a service.
    
    Args:
        service_type: Type of service (e.g., 'database', 'api')
        capabilities: Dict of service capabilities
        commands: List of available commands
        context: Current service context
        
    Returns:
        Formatted system prompt
    """
    return BASE_SYSTEM_PROMPT.format(
        service_type=service_type,
        capabilities=format_capabilities(capabilities),
        commands=format_commands(commands),
        context=format_context(context)
    )

def format_capabilities(capabilities: Dict[str, str]) -> str:
    """Format service capabilities for prompt inclusion."""
    return "\n".join(f"- {name}: {desc}" for name, desc in capabilities.items())

def format_commands(commands: List[str]) -> str:
    """Format available commands for prompt inclusion."""
    return "\n".join(f"- {cmd}" for cmd in commands)

def format_context(context: Dict[str, any]) -> str:
    """Format current context for prompt inclusion."""
    return "\n".join(f"{k}: {v}" for k, v in context.items())

def get_error_prompt(error: str, context: Dict[str, any]) -> str:
    """Get error handling prompt for a specific error."""
    return ERROR_HANDLING_PROMPT.format(
        error=error,
        error_context=format_context(context)
    )

def get_summary_prompt(content: str, domain_context: Dict[str, any]) -> str:
    """Get summary prompt for content."""
    return SUMMARY_BASE_PROMPT.format(
        content=content,
        domain_context=format_context(domain_context)
    )

def get_command_prompt(
    command: str,
    parameters: Dict[str, str],
    validation_rules: Dict[str, str]
) -> str:
    """Get command processing prompt."""
    return COMMAND_PROCESSING_PROMPT.format(
        command=command,
        parameters=format_parameters(parameters),
        validation_rules=format_validation_rules(validation_rules)
    )

def format_parameters(parameters: Dict[str, str]) -> str:
    """Format command parameters for prompt inclusion."""
    return "\n".join(f"- {name}: {desc}" for name, desc in parameters.items())

def format_validation_rules(rules: Dict[str, str]) -> str:
    """Format validation rules for prompt inclusion."""
    return "\n".join(f"- {name}: {rule}" for name, rule in rules.items()) 