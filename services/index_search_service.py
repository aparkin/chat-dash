"""
Index Search Service implementation.

This service provides unified text search capabilities across all indexed data sources
in the ChatDash application. It leverages existing text search infrastructure while
providing a consistent interface for searching and result presentation.

Future Enhancements:
1. Source Registration System
   - Dynamic registration of new index sources at runtime
   - Source-specific configuration management
   - Source health monitoring and status reporting
   - Source capability discovery

2. Advanced Query Features
   - Regular expression support
   - Exact match options
   - Field-specific search constraints
   - Boolean operators (AND, OR, NOT)
   - Source-specific query parameters

3. Result Management
   - Search history tracking
   - Result caching
   - Incremental search updates
   - Export capabilities
   - Result persistence

4. Performance Optimizations
   - Parallel search execution
   - Index preloading
   - Result pagination
   - Memory-efficient processing
   - Query optimization per source
"""

from typing import Dict, Any, List, Optional, Set
import re
from datetime import datetime
import pandas as pd

from .base import (
    ChatService,
    ServiceResponse,
    ServiceMessage,
    PreviewIdentifier
)

class SearchParameters:
    """Container for parsed search parameters.
    
    Future:
    - Add support for source-specific parameters
    - Add query type specifications (exact, fuzzy, regex)
    - Add field constraints and filters
    - Add source-specific thresholds
    """
    def __init__(
        self,
        query: str,
        threshold: float = 0.6,
        sources: Optional[Set[str]] = None,
        coverage: float = 0.0
    ):
        self.query = query
        self.threshold = threshold
        self.sources = sources  # Now determined by available index sources
        self.coverage = coverage

class SearchResult:
    """Standardized container for search results.
    
    Future:
    - Add result scoring normalization
    - Add result ranking algorithms
    - Add result clustering/grouping
    - Add source-specific metadata
    """
    def __init__(
        self,
        source_name: str,
        source_type: str,
        similarity: float,
        matched_text: str,
        details: Dict[str, Any]
    ):
        self.source_name = source_name
        self.source_type = source_type
        self.similarity = similarity
        self.matched_text = matched_text
        self.details = details
        self.timestamp = datetime.now()

class IndexSearchService(ChatService):
    """Service for unified text search across all indexed sources.
    
    This service provides:
    1. Unified search interface across multiple data sources
    2. Consistent result formatting and presentation
    3. Configurable search parameters and thresholds
    4. Source-specific optimizations
    
    Future Enhancements:
    1. Source Management
       - Dynamic source registration
       - Source health monitoring
       - Source-specific configuration
       - Source capability discovery
    
    2. Query Processing
       - Query preprocessing/normalization
       - Query expansion/suggestion
       - Context-aware search
       - Source-specific query optimization
    
    3. Result Processing
       - Result deduplication
       - Result ranking/scoring
       - Result caching
       - Source-specific result formatting
    """
    
    def __init__(self, index_sources: Dict[str, Any]):
        """Initialize with dictionary of index sources.
        
        Args:
            index_sources: Dictionary mapping source names to their searchers.
                         Each searcher must implement search_text(query, threshold, coverage)
        
        Example:
            index_sources = {
                'datasets': dataset_searcher,
                'database': database_searcher,
                # Future sources:
                # 'documents': document_searcher,
                # 'code': code_searcher,
            }
        """
        super().__init__("index_search")
        # Register our prefix for search result IDs
        PreviewIdentifier.register_prefix("search")
        
        self.index_sources = index_sources
        
        # Single pattern for command-style syntax
        self.command_pattern = (
            r'^search\s+'  # Command start
            r'(?:(\w+)\s+)?'  # Optional source name
            r'indices?\s+'  # Required 'index' or 'indices'
            r'(?:with\s+threshold\s+(\d*\.?\d+)\s+)?'  # Optional threshold
            r'for\s+'  # Required 'for' separator
            r'(.+?)'  # Search text (non-greedy match)
            r'\s*$'  # End of string (no trailing content)
        )
    
    def can_handle(self, message: str) -> bool:
        """Detect if message matches our command syntax.
        
        Valid formats:
            search indices for <text>
            search <source> indices for <text>
            search indices with threshold <0.X> for <text>
            search <source> indices with threshold <0.X> for <text>
            convert search_<ID> to dataset
        """
        message = message.lower().strip()
        
        # Check for dataset conversion command
        if message.startswith('convert search_'):
            convert_match = re.match(r'^convert\s+(search_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message)
            return bool(convert_match)
            
        # Check for search command
        match = re.match(self.command_pattern, message)
        if match:
            _, _, query = match.groups()
            return bool(query and query.strip())
            
        return False
    
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Extract search parameters from command-style message.
        
        Examples:
            "search indices for example"
            "search database indices for test"
            "search dataset indices with threshold 0.8 for sample"
            "convert search_20250212_123456_orig to dataset"
        """
        message = message.lower().strip()
        
        # Check for dataset conversion
        convert_match = re.match(r'^convert\s+(search_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message)
        if convert_match:
            return {
                'command': 'convert',
                'search_id': convert_match.group(1)
            }
        
        # Must be a search command
        match = re.match(self.command_pattern, message)
        if not match:
            raise ValueError("Message does not match expected command format")
            
        source_name, threshold_str, query = match.groups()
        
        # Get available sources
        available_sources = set(self.index_sources.keys())
        
        # Handle source specification
        if source_name:
            if source_name not in available_sources:
                raise ValueError(
                    f"Unknown source '{source_name}'. Available sources: {', '.join(available_sources)}"
                )
            sources = {source_name}
        else:
            # Default to all available sources
            sources = available_sources
        
        # Handle threshold
        threshold = None
        if threshold_str:
            try:
                threshold = float(threshold_str)
                if not 0 <= threshold <= 1:
                    raise ValueError("Threshold must be between 0 and 1")
            except ValueError as e:
                raise ValueError(f"Invalid threshold value: {str(e)}")
        
        return {
            'command': 'search',
            'query': query.strip(),
            'threshold': threshold if threshold is not None else 0.6,  # Default threshold
            'sources': sources
        }
    
    def _create_results_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert search results to a structured DataFrame.
        
        This creates a normalized view of results across different sources,
        making it suitable for dataset conversion and analysis.
        """
        rows = []
        for result in results:
            for col_name, details in result['details'].items():
                for matched_value in details['matches']:
                    rows.append({
                        'source_type': result['source_type'],
                        'source_name': result['source_name'],
                        'column_name': col_name,
                        'matched_value': matched_value,
                        'similarity': details['similarities'][matched_value],
                        'occurrences': details['counts'][matched_value]
                    })
        
        df = pd.DataFrame(rows)
        
        # Format columns
        if not df.empty:
            # Format similarity as percentage
            df['similarity'] = df['similarity'].apply(lambda x: f"{x*100:.1f}%")
            
            # Sort by similarity (after converting to string)
            df = df.sort_values(by=['source_type', 'source_name', 'similarity'], 
                              ascending=[True, True, False])
            
            # Rename columns for better readability
            df = df.rename(columns={
                'source_type': 'Source Type',
                'source_name': 'Source Name',
                'column_name': 'Column',
                'matched_value': 'Matched Value',
                'similarity': 'Similarity',
                'occurrences': 'Count'
            })
        
        return df
    
    def _handle_dataset_conversion(self, params: dict, context: dict) -> ServiceResponse:
        """Handle conversion of search results to dataset.
        
        Args:
            params: Parameters including search_id
            context: Execution context with search store
            
        Returns:
            ServiceResponse with conversion result
        """
        search_id = params.get('search_id')
        if not search_id:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No search ID provided for conversion.",
                    message_type="error"
                )],
                state_updates={'chat_input': ''}
            )
        
        # Get stored execution
        executions = context.get('successful_queries_store', {})
        stored = executions.get(search_id)
        
        if not stored:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"No execution found for search ID: {search_id}",
                    message_type="error"
                )],
                state_updates={'chat_input': ''}
            )
        
        try:
            # Create DataFrame from stored results
            df = pd.DataFrame(stored['dataframe'])
            
            # Get datasets store
            datasets = context.get('datasets_store', {})
            
            # Create dataset with special metadata
            datasets[search_id] = {
                'df': df.to_dict('records'),
                'metadata': {
                    'source': f"Search Query: {search_id}",
                    'query': stored['query'],
                    'threshold': stored['threshold'],
                    'execution_time': stored['metadata']['execution_time'],
                    'rows': len(df),
                    'columns': list(df.columns),
                    'sources': stored['metadata']['sources']
                }
            }
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Search results converted to dataset '{search_id}'

- Rows: {len(df)}
- Columns: {', '.join(df.columns)}
- Source: Search Query {search_id}
- Original query: {stored['query']}
- Threshold: {stored['threshold']}
- Sources: {', '.join(stored['metadata']['sources'])}""",
                    message_type="info"
                )],
                store_updates={
                    'datasets_store': datasets,  # Update datasets store
                    'successful_queries_store': {  # Update queries store, removing the converted query
                        k: v for k, v in executions.items() if k != search_id
                    }
                },
                state_updates={'chat_input': ''}
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Index search service error: ❌ Search result conversion failed: {str(e)}",
                    message_type="error"
                )],
                state_updates={'chat_input': ''}
            )
    
    def execute(self, params: dict, context: dict) -> ServiceResponse:
        """Execute the search request."""
        command = params.get('command')
        
        if command == 'convert':
            return self._handle_dataset_conversion(params, context)
        
        # Must be a search command
        query = params.get('query', '')
        threshold = params.get('threshold', 0.3)
        sources = params.get('sources', {'database', 'datasets'})
        
        print(f"\n=== Index Search Execution Debug ===")
        print(f"Query: '{query}'")
        print(f"Threshold: {threshold}")
        print(f"Requested sources: {sources}")
        
        # Track warnings
        warnings = []
        results = []
        
        # Check available searchers
        available_searchers = list(self.index_sources.keys())
        print(f"\nAvailable searchers: {available_searchers}")
        
        # Check database state
        if 'database' in sources:
            db_state = context.get('database_state', {})
            searcher = self.index_sources.get('database')
            print(f"\nDatabase State Debug:")
            print(f"- Connected: {db_state.get('connected', False)}")
            print(f"- Has searcher: {searcher is not None}")
            print(f"- Searcher fitted: {searcher.fitted if searcher and hasattr(searcher, 'fitted') else False}")
            
            if not db_state.get('connected'):
                warnings.append("Database search requested but no database is connected. Use the database dropdown in Data Management to connect to a database.")
        
        # Check dataset state
        if 'datasets' in sources:
            datasets = context.get('datasets_store', {})
            searcher = self.index_sources.get('datasets')
            print(f"\nDataset State Debug:")
            print(f"- Has datasets: {bool(datasets)}")
            print(f"- Has searcher: {searcher is not None}")
            print(f"- Searcher fitted: {searcher.fitted if searcher and hasattr(searcher, 'fitted') else False}")
            
            if not datasets:
                warnings.append("Dataset search requested but no datasets are loaded. Upload datasets using the file upload area in Data Management.")
        
        # Execute search on available sources
        for source in sources:
            searcher = self.index_sources.get(source)
            if searcher is None:
                continue
                
            if source == 'datasets':
                print(f"\nChecking source: datasets")
                print(f"Searcher fitted: {searcher.fitted if hasattr(searcher, 'fitted') else False}")
                if hasattr(searcher, 'fitted') and searcher.fitted:
                    print("Executing search on datasets...")
                    dataset_results = searcher.search_text(query, threshold)
                    print(f"Got {len(dataset_results)} results")
                    results.extend(dataset_results)
                    print("Added results from datasets")
            
            elif source == 'database':
                print(f"\nChecking source: database")
                print(f"Searcher fitted: {searcher.fitted if hasattr(searcher, 'fitted') else False}")
                if hasattr(searcher, 'fitted') and searcher.fitted:
                    print("Executing search on database...")
                    db_results = searcher.search_text(query, threshold)
                    print(f"Got {len(db_results)} results")
                    results.extend(db_results)
                    print("Added results from database")
        
        print(f"\nTotal results: {len(results)}")
        
        # Generate unique ID for this search
        search_id = PreviewIdentifier.create_id(prefix="search")
        
        # Create DataFrame from results for storage
        results_df = self._create_results_dataframe(results)
        
        # Store in successful_queries_store
        store_updates = {
            'successful_queries_store': {
                search_id: {
                    'query': query,
                    'threshold': threshold,
                    'dataframe': results_df.to_dict('records'),
                    'metadata': {
                        'execution_time': datetime.now().isoformat(),
                        'total_results': len(results),
                        'sources': list(sources)
                    }
                }
            }
        }
        
        # Create concise summary for chat
        summary = []
        if results:
            summary.append(f"Found matches for '{query}' (threshold={threshold}):")
            summary.append(f"\nSearch ID: {search_id}")
            
            # Add DataFrame preview
            preview = f"\nResults preview:\n```\n{results_df.head().to_string()}\n```"
            summary.append(preview)
            
            # Group by source type
            by_source = {'database': [], 'dataset': []}
            for result in results:
                by_source[result['source_type']].append(result)
            
            # Summarize database results
            if by_source['database']:
                summary.append("\nDatabase matches:")
                for result in by_source['database']:
                    table_summary = []
                    for col, details in result['details'].items():
                        match_count = len(details['matches'])
                        total_rows = sum(details['counts'].values())
                        table_summary.append(f"{col}: {match_count} unique matches ({total_rows} total rows)")
                    summary.append(f"- {result['source_name']}: {', '.join(table_summary)}")
            
            # Summarize dataset results
            if by_source['dataset']:
                summary.append("\nDataset matches:")
                for result in by_source['dataset']:
                    dataset_summary = []
                    for col, details in result['details'].items():
                        match_count = len(details['matches'])
                        total_rows = sum(details['counts'].values())
                        dataset_summary.append(f"{col}: {match_count} unique matches ({total_rows} total rows)")
                    summary.append(f"- {result['source_name']}: {', '.join(dataset_summary)}")
                    
            # Add conversion hint
            summary.append(f"\nTo convert results to a dataset, use: convert {search_id} to dataset")
        else:
            summary.append(f"No matches found for '{query}' with threshold {threshold}")
        
        if warnings:
            summary.append("\nWarnings:")
            for warning in warnings:
                summary.append(f"- {warning}")
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service="index_search",
                    content="\n".join(summary)
                )
            ],
            store_updates=store_updates
        ) 