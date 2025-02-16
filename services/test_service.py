"""
Test service implementation for validating service infrastructure.

This service provides a simple store reporting functionality to test the service pattern.
"""

from typing import Dict, Any
from .base import ChatService, ServiceResponse, ServiceMessage, PreviewIdentifier

class StoreReportService(ChatService):
    """Service that reports on the contents of data stores."""
    
    def __init__(self):
        super().__init__("store_report")
    
    def can_handle(self, message: str) -> bool:
        """Check if message is exactly 'report on stores'."""
        return message.strip().lower() == 'report on stores'
    
    def parse_request(self, message: str) -> Dict[str, Any]:
        """No parameters needed for this service."""
        return {}
    
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Generate reports on datasets and successful queries stores."""
        messages = []
        
        # Get store data from context
        datasets = context.get('datasets_store', {})
        queries = context.get('successful_queries_store', {})
        
        # Track summary information for context
        summary_data = {
            'dataset_count': len(datasets),
            'query_count': len(queries),
            'dataset_details': [],
            'query_details': []
        }
        
        # Generate dataset report
        if datasets:
            # Create table rows with proper metadata access
            table_rows = []
            for name, data in datasets.items():
                metadata = data.get('metadata', {})
                rows = len(data['df'])
                cols = len(metadata.get('columns', []))
                source = metadata.get('source', 'Unknown')
                
                # Add to summary data
                summary_data['dataset_details'].append({
                    'name': name,
                    'rows': rows,
                    'columns': cols,
                    'source': source
                })
                
                table_rows.append(
                    f"| {name} | {rows} | {cols} | {source} |"
                )
            
            messages.append(ServiceMessage(
                service=self.name,
                content="### Dataset Store Contents\n\n" +
                       "| Dataset Name | Rows | Columns | Source |\n" +
                       "|--------------|------|----------|--------|\n" +
                       "\n".join(table_rows),
                message_type="info",
                role="assistant"
            ))
        else:
            messages.append(ServiceMessage(
                service=self.name,
                content="No datasets currently loaded.",
                message_type="info",
                role="assistant"
            ))
        
        # Generate queries report
        if queries:
            query_rows = []
            for query_id, data in queries.items():
                # Get prefix using PreviewIdentifier
                prefix = PreviewIdentifier.get_prefix(query_id)
                # Map prefix to type
                query_type = {
                    'literature': 'Literature',
                    'query': 'SQL'
                }.get(prefix, 'Unknown')
                
                exec_time = data.get('metadata', {}).get('execution_time', 'Unknown')
                details = self._get_query_details(data)
                
                # Add to summary data
                summary_data['query_details'].append({
                    'id': query_id,
                    'type': query_type,
                    'execution_time': exec_time,
                    'details': details
                })
                
                query_rows.append(
                    f"| {query_id} | {query_type} | {exec_time} | {details} |"
                )
            
            messages.append(ServiceMessage(
                service=self.name,
                content="### Successful Queries Store Contents\n\n" +
                       "| Query ID | Type | Execution Time | Details |\n" +
                       "|----------|------|----------------|----------|\n" +
                       "\n".join(query_rows),
                message_type="info",
                role="assistant"
            ))
        else:
            messages.append(ServiceMessage(
                service=self.name,
                content="No successful queries in store.",
                message_type="info",
                role="assistant"
            ))
        
        return ServiceResponse(
            messages=messages,
        )
    
    def _get_query_details(self, query_data: Dict[str, Any]) -> str:
        """Extract relevant details from query data."""
        try:
            if 'threshold' in query_data:  # Literature query
                return f"Threshold: {query_data['threshold']}"
            elif 'sql' in query_data:  # SQL query
                sql = query_data['sql']
                # Truncate SQL if too long
                return f"SQL: {sql[:50]}..." if len(sql) > 50 else f"SQL: {sql}"
            return "Details not available"
        except Exception as e:
            return f"Error extracting details: {str(e)}"

    def get_help_text(self) -> str:
        """Get help text for store report service commands."""
        return """
📊 **Store Reports**
- View summary of all data stores: `report on stores`
  Shows:
  - Dataset store contents and statistics
  - Query store contents and execution history
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for store reporting capabilities."""
        return """
Store Report Service Capabilities:
1. Dataset Store Reporting
   - Lists all loaded datasets
   - Shows dataset metadata (rows, columns, source)
   - Provides dataset statistics

2. Query Store Reporting
   - Lists successful queries
   - Shows query execution times
   - Provides query metadata
   - Tracks query results

3. Report Format
   - Markdown formatted tables
   - Organized by store type
   - Includes summary statistics
   - Shows detailed metadata
""" 