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
import json
import io

from .base import (
    ChatService,
    ServiceResponse,
    ServiceMessage,
    PreviewIdentifier,
    MessageType
)
from .llm_service import LLMServiceMixin

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

class IndexSearchService(ChatService, LLMServiceMixin):
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
        ChatService.__init__(self, "index_search")
        LLMServiceMixin.__init__(self, "index_search")
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
            # Format similarity as percentage - fuzz.ratio returns 0-100
            df['similarity'] = df['similarity'].apply(lambda x: f"{x:.1f}%")
            
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
                    message_type=MessageType.ERROR
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
                    message_type=MessageType.ERROR
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
                    message_type=MessageType.RESULT
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
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )
    
    def execute(self, params: dict, context: dict) -> ServiceResponse:
        """Execute the search request."""
        # Store context for use in _call_llm
        self.context = context

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
        
        # Create initial response
        response_content = "\n".join(summary)
        messages = [
            ServiceMessage(
                service="index_search",
                content=response_content,
                message_type=MessageType.RESULT
            )
        ]
        
        # Generate LLM summary if we have results
        if results:
            try:
                # Pass the actual DataFrame to summarize instead of the preview text
                llm_summary = self.summarize(results_df, context.get('chat_history', []))
                if llm_summary:
                    messages.append(
                        ServiceMessage(
                            service="index_search",
                            content=f"\n### Analysis Summary\n\n{llm_summary}",
                            message_type=MessageType.SUMMARY
                        )
                    )
            except Exception as e:
                print(f"Error generating LLM summary: {str(e)}")
                # Continue without summary
        
        return ServiceResponse(
            messages=messages,
            store_updates=store_updates
        )

    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """
        Required by LLMServiceMixin but not used by IndexSearchService.
        We don't use LLM for processing messages, only for summarization.
        """
        raise NotImplementedError("IndexSearchService does not use LLM for message processing")

    def _compress_search_results(self, df: pd.DataFrame, max_tokens: int) -> Dict[str, Any]:
        """
        Compress search results into hierarchical structure within token budget.
        
        Args:
            df: DataFrame with search results
            max_tokens: Maximum tokens allowed for the compressed structure
            
        Returns:
            Dict with compressed structure:
            {
                'source_type': {
                    'source_name': {
                        'column': {
                            'matches': [{'term': str, 'similarity': float, 'count': int}, ...],
                            'total_matches': int,
                            'total_rows': int
                        }
                    }
                },
                'threshold_used': float  # If compression was needed
            }
        """
        # Create initial hierarchical structure
        compressed = {}
        for _, row in df.iterrows():
            source_type = row['Source Type']
            source_name = row['Source Name']
            column = row['Column']
            
            # Convert similarity back to float from percentage string
            similarity = float(row['Similarity'].rstrip('%'))
            
            # Initialize nested dictionaries
            if source_type not in compressed:
                compressed[source_type] = {}
            if source_name not in compressed[source_type]:
                compressed[source_type][source_name] = {}
            if column not in compressed[source_type][source_name]:
                compressed[source_type][source_name][column] = {
                    'matches': [],
                    'total_matches': 0,
                    'total_rows': 0
                }
            
            # Add match
            compressed[source_type][source_name][column]['matches'].append({
                'term': row['Matched Value'],
                'similarity': similarity,
                'count': row['Count']
            })
            compressed[source_type][source_name][column]['total_matches'] += 1
            compressed[source_type][source_name][column]['total_rows'] += row['Count']
        
        # Convert to string and check token count
        compressed_str = json.dumps(compressed, indent=2)
        token_count = self.count_tokens(compressed_str)
        
        # If within budget, return as is
        if token_count <= max_tokens:
            return {'data': compressed}
            
        # If over budget, find minimum similarity threshold that fits
        similarities = df['Similarity'].apply(lambda x: float(x.rstrip('%'))).unique()
        similarities.sort()
        
        for threshold in similarities:
            # Filter df by threshold
            filtered_df = df[df['Similarity'].apply(lambda x: float(x.rstrip('%'))) >= threshold]
            
            # Create compressed structure with filtered data
            filtered_compressed = {}
            for _, row in filtered_df.iterrows():
                source_type = row['Source Type']
                source_name = row['Source Name']
                column = row['Column']
                similarity = float(row['Similarity'].rstrip('%'))
                
                if source_type not in filtered_compressed:
                    filtered_compressed[source_type] = {}
                if source_name not in filtered_compressed[source_type]:
                    filtered_compressed[source_type][source_name] = {}
                if column not in filtered_compressed[source_type][source_name]:
                    filtered_compressed[source_type][source_name][column] = {
                        'matches': [],
                        'total_matches': 0,
                        'total_rows': 0
                    }
                
                filtered_compressed[source_type][source_name][column]['matches'].append({
                    'term': row['Matched Value'],
                    'similarity': similarity,
                    'count': row['Count']
                })
                filtered_compressed[source_type][source_name][column]['total_matches'] += 1
                filtered_compressed[source_type][source_name][column]['total_rows'] += row['Count']
            
            # Check if this fits within token budget
            filtered_str = json.dumps(filtered_compressed, indent=2)
            if self.count_tokens(filtered_str) <= max_tokens:
                return {
                    'data': filtered_compressed,
                    'threshold_used': threshold
                }
        
        # If we get here, even highest threshold doesn't fit
        # Take top N matches by similarity
        top_df = df.nlargest(10, 'Similarity')
        final_compressed = {}
        for _, row in top_df.iterrows():
            source_type = row['Source Type']
            source_name = row['Source Name']
            column = row['Column']
            similarity = float(row['Similarity'].rstrip('%'))
            
            if source_type not in final_compressed:
                final_compressed[source_type] = {}
            if source_name not in final_compressed[source_type]:
                final_compressed[source_type][source_name] = {}
            if column not in final_compressed[source_type][source_name]:
                final_compressed[source_type][source_name][column] = {
                    'matches': [],
                    'total_matches': 0,
                    'total_rows': 0
                }
            
            final_compressed[source_type][source_name][column]['matches'].append({
                'term': row['Matched Value'],
                'similarity': similarity,
                'count': row['Count']
            })
            final_compressed[source_type][source_name][column]['total_matches'] += 1
            final_compressed[source_type][source_name][column]['total_rows'] += row['Count']
        
        return {
            'data': final_compressed,
            'truncated': True,
            'shown_matches': len(top_df)
        }

    def _calculate_context_limits(self, system_prompt: str) -> Dict[str, int]:
        """Calculate token limits for search result summarization.
        
        Balances token budget between search results and chat history,
        prioritizing search results to maintain as much detail as possible.
        
        Args:
            system_prompt: The system prompt to be used
            
        Returns:
            Dict with token limits:
            {
                'system_prompt': int,    # Fixed overhead for system prompt
                'search_results': int,   # Primary allocation for search results
                'chat_history': int,     # Smaller allocation for context
                'total_available': int   # Total context window size
            }
        """
        # Model context window
        MAX_CONTEXT_TOKENS = 8192
        
        # Calculate system prompt overhead
        system_tokens = self.count_tokens(system_prompt)
        
        # Reserve space for system prompt and safety margin
        available_tokens = MAX_CONTEXT_TOKENS - system_tokens - 500  # 500 token safety margin
        
        # Prioritize search results (70%) over chat history (30%)
        allocations = {
            'system_prompt': system_tokens,
            'search_results': int(available_tokens * 0.7),  # Majority for search results
            'chat_history': int(available_tokens * 0.3),    # Smaller portion for context
            'total_available': MAX_CONTEXT_TOKENS
        }
        
        return allocations

    def summarize(self, df: pd.DataFrame, chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of search results with biological/environmental context.
        
        Args:
            df: DataFrame containing the search results
            chat_history: List of previous chat messages
            
        Returns:
            str: Generated summary focusing on:
                - Broad topics and subclasses found
                - Biological/environmental significance
                - Organization across data sources
                - Relationships and patterns
        """
        # Create system prompt for summarization
        system_prompt = """You are a scientific data analyst specializing in biological and environmental data analysis.
        Your task is to analyze search results from various data sources and provide a comprehensive summary that:

        1. Term Origins and Distribution:
           - For each source (database or dataset), explicitly list:
             * The source name (e.g., table name or dataset name)
             * The specific columns where terms were found
             * The matched terms with their similarity scores
           - Highlight which terms appear in multiple sources by:
             * Identifying overlapping terms
             * Listing the exact source names and columns where they appear
             * Noting any patterns in how these shared terms are used

        2. Topic Analysis:
           - Identify broad topics and their subclasses from the matched values
           - Group conceptually related terms
           - Highlight any scientific nomenclature or technical terms
           - Note which sources contribute to each topic group

        3. Scientific Significance:
           - Biological significance of matched terms
           - Environmental implications
           - Physical/chemical properties and their importance
           - Known relationships to species, ecosystems, or processes

        4. Data Organization:
           - How different sources organize these concepts
           - Patterns in data distribution across sources
           - Relationships between different columns/sources
           - Quality assessment based on match scores

        5. Domain-Specific Context:
           For matches related to:
           a) Geographic data (lat/long):
              - Link to specific geographic or environmental conditions
              - Relevant species distributions
              - Regional environmental characteristics
           
           b) Temporal data (dates):
              - Seasonal patterns
              - Climatic conditions
              - Historical context
           
           c) Depth/elevation data:
              - Hydrographic conditions
              - Oceanographic implications
              - Altitude-related patterns
           
           d) Chemical concentrations:
              - Environmental distributions
              - Toxicological effects
              - Chemical processes
           
           e) Species data:
              - Distribution patterns
              - Ecological interactions
              - Functional roles
           
           f) Temperature data:
              - Effects on species/ecosystems
              - Climate relationships
              - Physiological implications

        Format your response as:
        1. Term Distribution (list each source and its terms)
        2. Cross-Source Relationships (identify overlapping terms)
        3. Key Findings (2-3 bullet points)
        4. Detailed Analysis
        5. Scientific Context
        6. Suggested Further Analysis

        IMPORTANT:
        - Always specify exact source names and column names when discussing terms
        - Explicitly identify when the same or similar terms appear across different sources
        - For overlapping terms, list ALL sources and columns where they appear
        - Maintain clarity about which source each term comes from

        Focus on the most significant findings based on match scores and occurrence counts.
        """
        
        # Calculate token limits
        limits = self._calculate_context_limits(system_prompt)
        
        try:
            # Compress results to fit token budget
            compressed = self._compress_search_results(df, limits['search_results'])
            
            # Create context for LLM
            context = {
                'results': compressed['data'],
                'total_matches': len(df),
                'sources': df['Source Type'].nunique(),
                'threshold_used': compressed.get('threshold_used'),
                'truncated': compressed.get('truncated', False),
                'shown_matches': compressed.get('shown_matches')
            }
            
            # Convert to string for LLM
            content = json.dumps(context, indent=2)
            
            # Get relevant chat history within token limit
            messages = self._filter_history_by_tokens(chat_history, limits['chat_history'])
            
            # Create message list
            llm_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyze these search results:\n\n{content}"}
            ]
            
            # Add filtered history
            for msg in messages:
                llm_messages.append({
                    "role": msg.get('role', 'user'),
                    "content": msg.get('content', '')
                })
            
            # Get LLM response
            response = self._call_llm(llm_messages)
            return response
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def _filter_history_by_tokens(self, chat_history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Filter chat history to fit within token budget.
        
        Prioritizes:
        1. Most recent search-related messages
        2. Messages with search results or analysis
        3. Recent user queries
        
        Args:
            chat_history: Complete chat history
            max_tokens: Maximum tokens allowed for history
            
        Returns:
            List[Dict[str, Any]]: Filtered chat history within token budget
        """
        if not chat_history:
            return []
            
        filtered_messages = []
        token_count = 0
        
        # First pass: get most recent search-related messages
        for msg in reversed(chat_history):
            msg_tokens = self.count_tokens(msg.get('content', ''))
            
            # Skip if would exceed budget
            if token_count + msg_tokens > max_tokens:
                break
                
            # Prioritize messages from this service
            if msg.get('service') == self.name:
                filtered_messages.insert(0, msg)
                token_count += msg_tokens
                continue
                
            # Include search-related user messages
            if msg['role'] == 'user' and any(term in msg.get('content', '').lower() 
                                           for term in ['search', 'find', 'look for', 'query']):
                filtered_messages.insert(0, msg)
                token_count += msg_tokens
                continue
        
        # Second pass: if space remains, add other recent context
        remaining_tokens = max_tokens - token_count
        if remaining_tokens > 500:  # Only add more if significant space remains
            for msg in reversed(chat_history):
                # Skip messages we already included
                if msg in filtered_messages:
                    continue
                    
                msg_tokens = self.count_tokens(msg.get('content', ''))
                if token_count + msg_tokens > max_tokens:
                    break
                    
                filtered_messages.insert(0, msg)
                token_count += msg_tokens
        
        return filtered_messages 

    def get_help_text(self) -> str:
        """Get help text for index search service commands."""
        return """
🔎 **Unified Index Search**
- Search all sources: `search indices for [text]`
- Search specific source: `search [source] indices for [text]`
- With threshold: `search indices with threshold [0.0-1.0] for [text]`
- Convert to dataset: `convert search_[ID] to dataset`
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for index search capabilities."""
        return """
Index Search Commands:
1. Search Commands:
   "search indices for [text]" - search all sources
   "search [source] indices for [text]" - search specific source
   "search indices with threshold [0.0-1.0] for [text]"
   - Available sources: datasets, database
   - Default threshold: 0.6

2. Result Conversion:
   "convert search_[ID] to dataset"
   - Saves search results as dataset
   - Preserves source and match info""" 