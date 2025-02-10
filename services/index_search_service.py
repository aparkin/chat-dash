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

from .base import (
    ChatService,
    ServiceResponse,
    ServiceMessage,
    ServiceContext
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
        """
        message = message.lower().strip()
        match = re.match(self.command_pattern, message)
        
        # Additional validation to ensure 'for' clause exists
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
        """
        message = message.lower().strip()
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
            'query': query.strip(),
            'threshold': threshold if threshold is not None else 0.6,  # Default threshold
            'sources': sources
        }
    
    def execute(self, params: dict, context: dict) -> ServiceResponse:
        """Execute the search request."""
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
        
        # Create concise summary for LLM
        summary = []
        if results:
            summary.append(f"Found matches for '{query}' (threshold={threshold}):")
            
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
        else:
            summary.append(f"No matches found for '{query}' with threshold {threshold}")
        
        if warnings:
            summary.append("\nWarnings:")
            for warning in warnings:
                summary.append(f"- {warning}")
        
        # Create service context with proper structure
        service_context = ServiceContext(
            source="index_search",
            data={
                'query': query,
                'threshold': threshold,
                'results': results,
                'warnings': warnings
            },
            metadata={
                'total_matches': len(results),
                'source_counts': {
                    source_type: len(source_results)
                    for source_type, source_results in by_source.items()
                }
            }
        )
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service="index_search",
                    content="\n".join(summary)
                )
            ],
            context=service_context
        )
    
    def _format_search_results(self, results: List[Dict], errors: List[str]) -> str:
        """Format search results into a readable preview.
        
        Future:
        - Add customizable formatting templates
        - Add result grouping/clustering
        - Add interactive result navigation
        - Add source-specific formatting
        """
        if not results and not errors:
            return "No matches found in any data source."
        
        sections = []
        
        # Add header with search coverage
        sections.append("### Search Results")
        
        # Group results by source type
        by_source = {}
        for result in results:
            source_type = result['source_type']
            if source_type not in by_source:
                by_source[source_type] = []
            by_source[source_type].append(result)
        
        # Format results for each source type
        for source_type, source_results in by_source.items():
            sections.append(f"\n#### {source_type.title()} Matches")
            
            for result in source_results:
                sections.append(f"\n{result['source_type']}: **{result['source_name']}** (Score: {result['similarity']:.2f})")
                
                for col_name, details in result['details'].items():
                    matches = len(details['matches'])
                    sections.append(f"\nColumn `{col_name}`: {matches} matching values")
                    
                    # Show matches sorted by similarity
                    all_matches = sorted(
                        details['matches'],
                        key=lambda x: details['similarities'][x],
                        reverse=True
                    )
                    for value in all_matches:
                        count = details['counts'][value]
                        similarity = details['similarities'][value]
                        sections.append(f"- '{value}' ({count} occurrences, {similarity:.0f}% match)")
        
        # Add errors if any
        if errors:
            sections.append("\n### Search Errors")
            for error in errors:
                sections.append(f"- {error}")
        
        return "\n".join(sections) 