"""Specialized prompts for database services."""

from typing import Dict, List, Optional
from ...base.system import BASE_SYSTEM_PROMPT

DATABASE_SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + """
Database Specific Guidelines:
1. Always validate SQL queries for safety
2. Use parameterized queries when possible
3. Consider query performance implications
4. Respect database access permissions
5. Handle large result sets appropriately

Database Schema:
{schema}

Access Permissions:
{permissions}
"""

SQL_QUERY_PROMPT = """Analyze this SQL query request:
{query}

Database Context:
{schema}

Validation Requirements:
1. SQL injection prevention
2. Performance optimization
3. Permission validation
4. Result set management

Generate:
1. Validated SQL query
2. Expected result format
3. Performance analysis
4. Error handling approach
"""

SCHEMA_ANALYSIS_PROMPT = """Analyze this database schema:
{schema}

Focus Areas:
1. Table relationships
2. Index usage
3. Constraint validation
4. Query optimization

Generate:
1. Schema overview
2. Key relationships
3. Performance considerations
4. Usage recommendations
"""

QUERY_OPTIMIZATION_PROMPT = """Optimize this query:
{query}

Current Performance:
{performance_metrics}

Optimization Goals:
1. Reduce execution time
2. Minimize resource usage
3. Improve result set handling
4. Maintain data accuracy

Generate:
1. Optimized query
2. Performance comparison
3. Trade-off analysis
4. Implementation risks
"""

DATA_SUMMARY_PROMPT = """Summarize this query result:
{result_data}

Context:
{query_context}

Focus Areas:
1. Key patterns
2. Anomalies
3. Business insights
4. Follow-up queries

Format as:
1. Overview
2. Key findings
3. Recommendations
4. Related analyses
"""

def build_database_prompt(
    schema: Dict[str, any],
    permissions: Dict[str, List[str]],
    context: Optional[Dict[str, any]] = None
) -> str:
    """Build database-specific system prompt.
    
    Args:
        schema: Database schema information
        permissions: Access permissions
        context: Additional context
        
    Returns:
        Formatted database prompt
    """
    return DATABASE_SYSTEM_PROMPT.format(
        schema=format_schema(schema),
        permissions=format_permissions(permissions),
        **(context or {})
    )

def format_schema(schema: Dict[str, any]) -> str:
    """Format database schema for prompt inclusion."""
    formatted = []
    
    for table, details in schema.items():
        columns = details.get('columns', {})
        indices = details.get('indices', [])
        constraints = details.get('constraints', [])
        
        formatted.append(f"Table: {table}")
        formatted.append("Columns:")
        for col, col_type in columns.items():
            formatted.append(f"  - {col}: {col_type}")
        
        if indices:
            formatted.append("Indices:")
            for idx in indices:
                formatted.append(f"  - {idx}")
        
        if constraints:
            formatted.append("Constraints:")
            for const in constraints:
                formatted.append(f"  - {const}")
        
        formatted.append("")  # Empty line between tables
    
    return "\n".join(formatted)

def format_permissions(permissions: Dict[str, List[str]]) -> str:
    """Format access permissions for prompt inclusion."""
    formatted = []
    
    for role, access in permissions.items():
        formatted.append(f"Role: {role}")
        for perm in access:
            formatted.append(f"  - {perm}")
        formatted.append("")  # Empty line between roles
    
    return "\n".join(formatted)

def get_query_prompt(
    query: str,
    schema: Dict[str, any],
    context: Optional[Dict[str, any]] = None
) -> str:
    """Get SQL query analysis prompt."""
    return SQL_QUERY_PROMPT.format(
        query=query,
        schema=format_schema(schema),
        **(context or {})
    )

def get_optimization_prompt(
    query: str,
    performance_metrics: Dict[str, any]
) -> str:
    """Get query optimization prompt."""
    return QUERY_OPTIMIZATION_PROMPT.format(
        query=query,
        performance_metrics=format_metrics(performance_metrics)
    )

def format_metrics(metrics: Dict[str, any]) -> str:
    """Format performance metrics for prompt inclusion."""
    formatted = []
    
    for metric, value in metrics.items():
        formatted.append(f"{metric}: {value}")
    
    return "\n".join(formatted)

def get_summary_prompt(
    result_data: str,
    query_context: Dict[str, any]
) -> str:
    """Get data summary prompt."""
    return DATA_SUMMARY_PROMPT.format(
        result_data=result_data,
        query_context=format_query_context(query_context)
    )

def format_query_context(context: Dict[str, any]) -> str:
    """Format query context for prompt inclusion."""
    formatted = []
    
    for key, value in context.items():
        if isinstance(value, (dict, list)):
            value = str(value)  # Simple conversion for complex types
        formatted.append(f"{key}: {value}")
    
    return "\n".join(formatted) 