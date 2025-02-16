"""
Literature service implementation.

This service handles literature search queries and refinements through Weaviate.
Key features:
1. Literature Search:
   - Natural language queries for scientific literature
   - Configurable relevance thresholds
   - Cross-reference tracking between articles, references, and entities

2. Result Management:
   - Structured DataFrame representation of results
   - Rich previews with metadata
   - Dataset conversion capabilities

3. Analysis:
   - LLM-powered result summarization
   - Scientific relationship analysis
   - Token-aware batch processing

4. Command Types:
   - Initial search: "find papers about X"
   - Refinement: "refine lit_query_ID with threshold 0.X"
   - Conversion: "convert lit_query_ID to dataset"
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
from datetime import datetime
import re
import traceback
import json
from ydata_profiling import ProfileReport

from .base import (
    ChatService, 
    ServiceResponse, 
    ServiceMessage, 
    PreviewIdentifier
)

from weaviate_integration import WeaviateConnection
from weaviate_manager.query.manager import QueryManager
from .llm_service import LLMServiceMixin

class LiteratureService(ChatService, LLMServiceMixin):
    """Service for literature search and refinement.
    
    This service provides:
    1. Natural language literature search using Weaviate
    2. Result refinement with adjustable thresholds
    3. Cross-reference tracking between articles and entities
    4. LLM-powered result summarization
    5. Dataset conversion capabilities
    
    Command Patterns:
    - Search: "find papers about X"
    - Refinement: "refine lit_query_ID with threshold 0.X"
    - Conversion: "convert lit_query_ID to dataset"
    
    Implementation Notes:
    - Uses on-demand Weaviate connections for resilience
    - Implements token-aware batch processing for large results
    - Provides rich previews with metadata and available actions
    """
    
    # Command patterns for parsing and validation
    CONVERT_PATTERN = r'^convert\s+(lit_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b'
    
    # Literature search patterns with examples
    LITERATURE_PATTERNS = [
        # Pattern: [find] [me] <papers|literature|references> about <X>
        # Examples: 
        # - "find papers about gene regulation"
        # - "find me literature about CRISPR"
        r'(?:find\s+)?(?:me\s+)?(?:papers|literature|references)\s+about\s+(.+?)(?:\?|$)',
        
        # Pattern: [what] [do|does] [the] <papers|literature|references> say about <X>
        # Examples:
        # - "what do the papers say about metabolic pathways"
        # - "what does the literature say about protein folding"
        r'(?:what\s+)?(?:do|does\s+)?(?:the\s+)?(?:papers|literature|references)\s+say\s+about\s+(.+?)(?:\?|$)'
    ]
    
    # Threshold adjustment patterns with examples
    THRESHOLD_PATTERNS = [
        # Pattern: use/set/apply [a] threshold [of] X
        # Example: "use threshold of 0.7"
        r"(?:use|set|apply|with)\s+(?:a\s+)?(?:threshold|cutoff|score|limit)\s+(?:of\s+)?(\d+\.?\d*)",
        
        # Pattern: threshold [of] X
        # Example: "threshold 0.8"
        r"threshold\s+(?:of\s+)?(\d+\.?\d*)",
        
        # Pattern: cutoff [of] X
        # Example: "cutoff 0.5"
        r"cutoff\s+(?:of\s+)?(\d+\.?\d*)"
    ]
    
    # Refinement patterns with examples
    REFINE_PATTERNS = [
        # Pattern: refine/update/modify query_ID
        # Example: "refine lit_query_20250215_123456_orig"
        r"(?:refine|update|modify)\s+(lit_query_\d{8}_\d{6}(?:_orig|_alt\d+))",
        
        # Pattern: query_ID with/using/at threshold
        # Example: "lit_query_20250215_123456_orig with threshold"
        r"(lit_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+(?:with|using|at)\s+threshold"
    ]
    
    # Configuration constants
    DEFAULT_THRESHOLD = 0.3  # Default similarity threshold for initial searches
    
    def __init__(self):
        """Initialize the literature service.
        
        This initializes:
        1. Base service functionality (ChatService)
        2. LLM capabilities (LLMServiceMixin)
        3. ID prefix registration for result tracking
        4. Pre-compiled regex patterns for command parsing
        """
        ChatService.__init__(self, "literature")
        LLMServiceMixin.__init__(self, "literature")
        # Register our prefix for storing query results
        PreviewIdentifier.register_prefix("lit_query")
        
        # Pre-compile patterns for efficiency
        self._convert_re = re.compile(self.CONVERT_PATTERN)
        self._literature_res = [re.compile(p) for p in self.LITERATURE_PATTERNS]
        self._threshold_res = [re.compile(p) for p in self.THRESHOLD_PATTERNS]
        self._refine_res = [re.compile(p) for p in self.REFINE_PATTERNS]
    
    def _transform_weaviate_results(self, json_results: dict) -> pd.DataFrame:
        """Transform Weaviate JSON results into a unified DataFrame.
        
        This method:
        1. Processes raw results from each collection
        2. Handles unified Article records with cross-references
        3. Creates a consistent DataFrame structure
        4. Preserves metadata and summary information
        
        Args:
            json_results: Dict containing:
                - query_info: Query parameters and metadata
                - raw_results: Direct hits from collections
                - unified_results: Article records with cross-references
                
        Returns:
            pd.DataFrame with:
            Columns:
                - score: Search relevance score
                - id: Record identifier
                - collection: Source collection name
                - source: Original collection or source field
                - {collection}_{property}: Dynamic property columns
                - cross_references: JSON string of cross-references
                
            Attributes:
                - query_info: Original query parameters
                - summary: Collection counts and statistics
        
        Note:
            Returns empty DataFrame with error info if processing fails
        """
        try:
            print("\n=== Processing Weaviate Results ===")
            
            # Initialize empty records list and property tracking
            records = []
            all_properties = {}  # {collection: {property_name: data_type}}
            
            # Process raw_results first to discover all possible properties
            raw_results = json_results.get('raw_results', {})
            print(f"\nProcessing raw results from {len(raw_results)} collections")
            
            for collection_name, collection_results in raw_results.items():
                if collection_name == 'Article':
                    continue  # Skip Articles in raw_results as they'll be handled in unified
                    
                print(f"\nProcessing collection: {collection_name}")
                print(f"Found {len(collection_results)} records")
                
                # Track properties for this collection
                all_properties[collection_name] = set()
                
                for record in collection_results:
                    # Create base record
                    transformed = {
                        'score': record.get('score', 0.0),
                        'id': str(record.get('uuid', '')),
                        'collection': collection_name,
                        'source': collection_name,
                        'cross_references': None  # Raw results don't have cross-references
                    }
                    
                    # Process properties
                    properties = record.get('properties', {})
                    for prop_name, value in properties.items():
                        column_name = f"{collection_name}_{prop_name}"
                        transformed[column_name] = value
                        all_properties[collection_name].add(prop_name)
                    
                    records.append(transformed)
            
            # Process unified results (Articles)
            unified_results = json_results.get('unified_results', [])
            print(f"\nProcessing {len(unified_results)} unified Article results")
            
            if unified_results:
                all_properties['Article'] = set()
                
                for record in unified_results:
                    # Create base record
                    transformed = {
                        'score': record.get('score', 0.0),
                        'id': str(record.get('id', '')),
                        'collection': 'Article',
                        'source': record.get('source', 'Unknown')
                    }
                    
                    # Process Article properties
                    properties = record.get('properties', {})
                    for prop_name, value in properties.items():
                        column_name = f"Article_{prop_name}"
                        transformed[column_name] = value
                        all_properties['Article'].add(prop_name)
                    
                    # Handle cross-references
                    traced = record.get('traced_elements', {})
                    if traced:
                        # Convert to simplified format for storage
                        refs = {}
                        for ref_collection, elements in traced.items():
                            if elements:  # Only store non-empty references
                                refs[ref_collection] = [
                                    {
                                        'id': str(elem.get('id', '')),
                                        'score': elem.get('score', 0.0)
                                    } for elem in elements
                                ]
                        transformed['cross_references'] = (
                            json.dumps(refs) if refs else None
                        )
                    else:
                        transformed['cross_references'] = None
                    
                    records.append(transformed)
            
            # Create DataFrame
            if not records:
                print("No results found")
                empty_df = pd.DataFrame(columns=[
                    'score', 'id', 'collection', 'source', 'cross_references'
                ])
                empty_df.attrs['query_info'] = json_results.get('query_info', {})
                empty_df.attrs['summary'] = {'total_results': 0}
                return empty_df
            
            # Create DataFrame and ensure all columns exist
            df = pd.DataFrame(records)
            
            # Create summary information
            summary = {
                'total_results': len(df),
                'collection_counts': df['collection'].value_counts().to_dict(),
                'score_range': {
                    'min': df['score'].min(),
                    'max': df['score'].max(),
                    'mean': df['score'].mean()
                },
                'property_coverage': {
                    collection: list(props) 
                    for collection, props in all_properties.items()
                }
            }
            
            # Store metadata
            df.attrs['query_info'] = json_results.get('query_info', {})
            df.attrs['summary'] = summary
            
            # Sort by score descending
            df = df.sort_values('score', ascending=False)
            
            print("\n=== Results Summary ===")
            print(f"Total records: {len(df)}")
            print("\nBy collection:")
            for collection, count in summary['collection_counts'].items():
                print(f"- {collection}: {count} records")
            print(f"\nScore range: {summary['score_range']['min']:.3f} - {summary['score_range']['max']:.3f}")
            
            return df
            
        except Exception as e:
            print(f"Error transforming Weaviate results: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            # Return empty DataFrame with error information
            empty_df = pd.DataFrame(columns=[
                'score', 'id', 'collection', 'source', 'cross_references'
            ])
            empty_df.attrs['query_info'] = json_results.get('query_info', {})
            empty_df.attrs['summary'] = {'error': str(e)}
            return empty_df
    
    def _execute_weaviate_query(self, query: str, min_score: float = 0.3) -> dict:
        """Execute a query through weaviate_manager.
        
        This method:
        1. Creates an on-demand Weaviate connection
        2. Initializes a query manager for this operation
        3. Executes a comprehensive search across collections
        4. Returns unified results with cross-references
        
        Args:
            query: Search query text
            min_score: Minimum similarity score (0-1)
            
        Returns:
            Dict containing:
            - query_info: Query parameters and metadata
            - raw_results: Direct hits from collections
            - unified_results: Article records with cross-references
            
        Note:
            Uses context managers to ensure proper cleanup of connections
        """
        print("\n=== Weaviate Query Debug ===")
        print(f"Query: '{query}'")
        print(f"Min score: {min_score}")
        
        try:
            # Create connection when needed
            connection = WeaviateConnection()
            with connection.get_client() as client:
                if not client:
                    print("Error: No Weaviate client available")
                    return {}
                
                print("Client connection successful")
                
                # Create query manager for this operation
                query_manager = QueryManager(client)
                print("QueryManager initialized")
                
                print("Executing comprehensive search...")
                results = query_manager.comprehensive_search(
                    query_text=query,
                    search_type="hybrid",
                    min_score=min_score,
                    unify_results=True,  # Get unified article view
                    verbose=True  # For debugging
                )
                
                print("\nSearch Results:")
                print(f"- Raw results: {bool(results)}")
                print(f"- Has unified_results: {bool(results and 'unified_results' in results)}")
                if results and 'unified_results' in results:
                    print(f"- Number of unified results: {len(results['unified_results'])}")
                    if results['unified_results']:
                        first_result = results['unified_results'][0]
                        print(f"- First result score: {first_result.get('score', 'N/A')}")
                        print(f"- First result collection: {first_result.get('collection', 'N/A')}")
                
                if not results or not results.get('unified_results'):
                    print("No results found")
                    return {}
                    
                return results
                
        except Exception as e:
            print(f"Error in execute_weaviate_query: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return {}
    
    def _format_weaviate_results_preview(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """Generate a formatted preview of Weaviate search results.
        
        Creates a structured markdown preview with:
        1. Query Information
           - Original query text
           - Search type
           - Minimum score used
        
        2. Results Summary
           - Total result count
           - Results by collection
           - Score distribution
        
        3. Collection-specific Previews
           - Article previews with abstracts
           - Reference previews with titles
           - Named entity previews with types
        
        Args:
            df: DataFrame from transform_weaviate_results
            max_rows: Maximum rows to show per collection
            
        Returns:
            str: Formatted markdown preview
            
        Note:
            Handles different collection types with appropriate formatting
        """
        try:
            # Get metadata from DataFrame attributes
            query_info = df.attrs.get('query_info', {})
            summary = df.attrs.get('summary', {})
            
            # Start building output
            sections = []
            
            # 1. Query Information
            sections.append("### Search Query Information")
            sections.append(f"- Query: {query_info.get('text', 'Not specified')}")
            sections.append(f"- Type: {query_info.get('type', 'Not specified')}")
            sections.append(f"- Minimum Score: {query_info.get('min_score', 'Not specified')}")
            sections.append("")
            
            # 2. Result Summary
            sections.append("### Results Summary")
            sections.append(f"Total Results: {summary.get('total_results', len(df))}")
            if 'collection_counts' in summary:
                sections.append("\nResults by Collection:")
                for collection, count in summary['collection_counts'].items():
                    sections.append(f"- {collection}: {count}")
            sections.append("")
            
            # 3. Collection-specific previews
            sections.append("### Result Previews")
            
            for collection in df['collection'].unique():
                collection_df = df[df['collection'] == collection].head(max_rows)
                if len(collection_df) == 0:
                    continue
                    
                sections.append(f"\n#### {collection} Preview")
                
                # Build preview table based on collection type
                if collection == 'Article':
                    sections.append("\n| Score | ID | Filename | Abstract Preview |")
                    sections.append("|-------|-----|----------|-----------------|")
                    for _, row in collection_df.iterrows():
                        # Get filename and abstract
                        filename = row.get('Article_filename', 'N/A')
                        abstract = row.get('Article_abstract', '')
                        # Create abstract preview
                        abstract_preview = abstract[:50] + "..." if abstract and len(abstract) > 50 else abstract or 'N/A'
                        # Format row with proper spacing
                        sections.append(
                            f"| {row['score']:.3f} | {row['id'][:8]}... | "
                            f"{filename} | {abstract_preview} |"
                        )
                    sections.append("")  # Add spacing after table
                    
                elif collection == 'Reference':
                    sections.append("\n| Score | ID | Title |")
                    sections.append("|-------|-----|-------|")
                    for _, row in collection_df.iterrows():
                        title = row.get('Reference_title', 'N/A')
                        # Truncate long titles
                        title_preview = title[:50] + "..." if len(title) > 50 else title
                        sections.append(
                            f"| {row['score']:.3f} | {row['id'][:8]}... | {title_preview} |"
                        )
                    sections.append("")  # Add spacing after table
                    
                elif collection == 'NamedEntity':
                    sections.append("\n| Score | ID | Name | Type |")
                    sections.append("|-------|-----|------|------|")
                    for _, row in collection_df.iterrows():
                        name = row.get('NamedEntity_name', 'N/A')
                        entity_type = row.get('NamedEntity_type', 'N/A')
                        sections.append(
                            f"| {row['score']:.3f} | {row['id'][:8]}... | {name} | {entity_type} |"
                        )
                    sections.append("")  # Add spacing after table
                
                # Add note if there are more results
                total_count = len(df[df['collection'] == collection])
                if total_count > max_rows:
                    sections.append(f"... and {total_count - max_rows} more {collection} results\n")
            
            # Add score distribution analysis
            sections.append("### Score Distribution")
            thresholds = [0.3, 0.5, 0.7, 0.9]
            sections.append("\n| Minimum Score | Results | By Collection |")
            sections.append("|---------------|----------|---------------|")
            
            for threshold in thresholds:
                filtered_df = df[df['score'] >= threshold]
                if len(filtered_df) > 0:
                    collection_counts = filtered_df['collection'].value_counts()
                    counts_str = ", ".join(
                        f"{col}: {count}" 
                        for col, count in collection_counts.items()
                    )
                    sections.append(
                        f"| {threshold:.1f} | {len(filtered_df)} | {counts_str} |"
                    )
            
            # Join sections and ensure no trailing whitespace
            return "\n".join(sections).rstrip()
            
        except Exception as e:
            return f"Error formatting results preview: {str(e)}"
    
    def _format_literature_preview(self, df: pd.DataFrame, query_id: str, threshold: float) -> str:
        """Format literature results preview with metadata and actions.
        
        Creates a complete preview containing:
        1. Standard Results Preview
           - Query information
           - Results summary
           - Collection-specific previews
        
        2. DataFrame Preview
           - Preview of most relevant columns
           - Limited to first few rows
        
        3. Available Actions
           - Refinement instructions with current query ID
           - Dataset conversion instructions
        
        Args:
            df: DataFrame from transform_weaviate_results
            query_id: Unique identifier for this query
            threshold: Current relevance threshold
            
        Returns:
            str: Formatted preview with instructions
            
        Note:
            Explicitly states available actions to prevent confusion
        """
        # Get the standard preview
        preview = self._format_weaviate_results_preview(df)
        
        # Add DataFrame preview with ID
        preview_rows = min(5, len(df))  # Show up to 5 rows
        df_preview = df.head(preview_rows)
        
        # Select most relevant columns for preview
        preview_columns = ['score', 'collection']
        if 'Article_title' in df.columns:
            preview_columns.append('Article_title')
        elif 'Article_filename' in df.columns:
            preview_columns.append('Article_filename')
        if 'Article_abstract' in df.columns:
            preview_columns.append('Article_abstract')
        
        df_section = f"""
Results preview:

Query ID: {query_id}
```
{df_preview[preview_columns].to_string()}
```

Current threshold: {threshold}

### Available actions
1. Refine results with different threshold:
   refine {query_id} with threshold 0.X

2. Save results as dataset:
   convert {query_id} to dataset"""
        
        # Combine sections and ensure no trailing whitespace
        return (preview + df_section).rstrip()
    
    def _parse_message(self, message: str) -> Dict[str, Any]:
        """Parse and classify user messages.
        
        This is the central parsing logic used by both can_handle and parse_request.
        It identifies and extracts parameters for three types of commands:
        
        1. Literature Search:
           - Matches natural language queries about papers/literature
           - Extracts the search topic
           - Uses default threshold
        
        2. Refinement:
           - Matches threshold adjustment requests
           - Extracts query ID and new threshold
           - Validates threshold range
        
        3. Conversion:
           - Matches dataset conversion commands
           - Extracts query ID to convert
        
        Args:
            message: The user's message (will be normalized)
            
        Returns:
            Dict containing:
            - type: 'conversion', 'refinement', 'initial', or None
            - params: Dict of extracted parameters
            - can_handle: Whether this service can handle the message
            
        Note:
            Message is normalized to lowercase for consistent matching
        """
        message = message.lower().strip()
        
        # Check for conversion command
        if message.startswith('convert lit_query_'):
            convert_match = self._convert_re.match(message)
            if convert_match:
                return {
                    'type': 'conversion',
                    'params': {'query_id': convert_match.group(1)},
                    'can_handle': True
                }
        
        # Check for refinement request
        threshold = None
        for pattern in self._threshold_res:
            match = pattern.search(message)
            if match:
                try:
                    threshold = float(match.group(1))
                    if not 0 <= threshold <= 1:
                        threshold = None
                except ValueError:
                    continue
                break
        
        query_id = None
        for pattern in self._refine_res:
            match = pattern.search(message)
            if match:
                query_id = match.group(1)
                break
        
        if threshold is not None and query_id is not None and query_id.startswith('lit_query_'):
            return {
                'type': 'refinement',
                'params': {
                    'query_id': query_id,
                    'threshold': threshold
                },
                'can_handle': True
            }
        
        # Check for literature query
        for pattern in self._literature_res:
            match = pattern.search(message)
            if match:
                query = match.group(1).strip()
                if query:
                    return {
                        'type': 'initial',
                        'params': {
                            'query': query,
                            'threshold': self.DEFAULT_THRESHOLD
                        },
                        'can_handle': True
                    }
        
        return {
            'type': None,
            'params': {},
            'can_handle': False
        }
    
    def can_handle(self, message: str) -> bool:
        """Detect if this service can handle the message.
        
        Checks for:
        1. Literature search queries ("find papers about X")
        2. Refinement requests ("refine lit_query_ID with threshold X")
        3. Dataset conversion commands ("convert lit_query_ID to dataset")
        
        Args:
            message: The user's message
            
        Returns:
            bool: True if service can handle this message
        """
        return self._parse_message(message)['can_handle']
    
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Parse message into request parameters.
        
        Extracts:
        1. Command type (search/refinement/conversion)
        2. Query text or ID
        3. Threshold values
        
        Args:
            message: The user's message
            
        Returns:
            Dict containing:
            - type: Command type
            - Additional parameters based on type
            
        Raises:
            ValueError: If message cannot be handled
        """
        result = self._parse_message(message)
        if not result['can_handle']:
            raise ValueError("Message cannot be handled by this service")
        
        return {
            'type': result['type'],
            **result['params']
        }
        
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the requested operation.
        
        Operations:
        1. Initial search: Execute new literature query
        2. Refinement: Adjust threshold on existing results
        3. Conversion: Convert results to dataset
        
        Args:
            params: Operation parameters from parse_request
            context: Execution context with:
                - successful_queries_store: Previous query results
                - datasets_store: Available datasets
                - chat_history: Previous messages
                
        Returns:
            ServiceResponse containing:
            - Messages with results/preview
            - Store updates if needed
            - State updates if needed
        """
        if params['type'] == 'conversion':
            return self._handle_conversion(params, context)
        elif params['type'] == 'refinement':
            return self._execute_refinement(params, context)
        else:
            return self._execute_search(params, context)
            
    def _handle_conversion(self, params: dict, context: dict) -> ServiceResponse:
        """Handle conversion of literature results to dataset.
        
        This method:
        1. Retrieves stored query results
        2. Creates a DataFrame from the results
        3. Generates a profile report
        4. Creates a new dataset with metadata
        5. Updates the datasets store
        
        Args:
            params: Parameters including query_id
            context: Execution context with:
                - successful_queries_store: Previous query results
                - datasets_store: Available datasets
            
        Returns:
            ServiceResponse with:
            - Success/error message
            - Updated stores if successful
            
        Note:
            Removes the query from successful_queries_store after conversion
        """
        query_id = params.get('query_id')
        if not query_id:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No query ID provided for conversion.",
                    message_type="error"
                )]
            )
        
        # Get stored query
        queries_store = context.get('successful_queries_store', {})
        stored_query = queries_store.get(query_id)
        
        if not stored_query:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"No results found for query ID: {query_id}",
                    message_type="error"
                )]
            )
        
        try:
            # Create DataFrame from stored results
            df = pd.DataFrame(stored_query['dataframe'])
            
            # Get datasets store
            datasets = context.get('datasets_store', {})
            
            # Create metadata
            metadata = {
                'source': f"Literature Query: {query_id}",
                'query': stored_query['query'],
                'threshold': stored_query['threshold'],
                'execution_time': stored_query['metadata']['execution_time'],
                'rows': len(df),
                'columns': list(df.columns),
                'query_info': stored_query['metadata']['query_info']
            }
            
            try:
                profile = ProfileReport(
                    df,
                    minimal=True,
                    title=f"Profile Report for {query_id}",
                    html={'style': {'full_width': True}},
                    progress_bar=False,
                    correlations={'pearson': {'calculate': True}},
                    missing_diagrams={'matrix': False},
                    samples=None
                )
                profile_html = profile.to_html()
            except Exception as e:
                print(f"Warning: Profile report generation failed: {str(e)}")
                profile_html = None

            # Create dataset with metadata
            datasets[query_id] = {
                'df': df.to_dict('records'),
                'metadata': metadata,
                'profile_report': profile_html
            }
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Literature results converted to dataset '{query_id}'

- Rows: {len(df)}
- Columns: {', '.join(df.columns)}
- Source: Literature Query {query_id}
- Original query: {stored_query['query']}
- Threshold: {stored_query['threshold']}""",
                    message_type="info"
                )],
                store_updates={
                    'datasets_store': datasets,  # Update datasets store
                    'successful_queries_store': {  # Update queries store, removing converted query
                        k: v for k, v in queries_store.items() if k != query_id
                    }
                }
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error converting literature results: {str(e)}",
                    message_type="error"
                )]
            )
    
    def _execute_search(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute initial literature search."""
        # Execute search
        results = self._execute_weaviate_query(params['query'], params['threshold'])
        
        # Transform results
        df = self._transform_weaviate_results(results)
        
        # Create preview ID using our internal method
        preview_id = PreviewIdentifier.create_id(prefix="lit_query")
        
        # Format preview (this will be shown in chat)
        preview = self._format_literature_preview(df, preview_id, params['threshold'])
        
        # Store full results in successful_queries_store
        store_updates = {
            'successful_queries_store': {
                preview_id: {
                    'query': params['query'],
                    'threshold': params['threshold'],
                    'dataframe': df.to_dict('records'),
                    'metadata': {
                        'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_results': len(df),
                        'query_info': df.attrs.get('query_info', {}),
                        'summary': df.attrs.get('summary', {})
                    }
                }
            }
        }
        
        # Create messages list with preview
        messages = [
            ServiceMessage(
                service=self.name,
                content=preview,
                role="assistant"
            )
        ]
        
        # Generate LLM summary if we have results
        if not df.empty:
            try:
                llm_summary = self.summarize(df, context.get('chat_history', []))
                if llm_summary:
                    messages.append(
                        ServiceMessage(
                            service=self.name,
                            content=f"\n### Analysis Summary\n\n{llm_summary}",
                            message_type="info",
                            role="assistant"
                        )
                    )
            except Exception as e:
                print(f"Error generating LLM summary: {str(e)}")
                # Continue without summary
        
        return ServiceResponse(
            messages=messages,
            store_updates=store_updates
        )
            
    def _execute_refinement(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Handle refinement of existing query."""
        # Get stored query
        queries_store = context.get('successful_queries_store', {})
        stored_query = queries_store.get(params['query_id'])
        
        if not stored_query:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"❌ Query {params['query_id']} not found in history.",
                    message_type="error",
                    role="assistant"
                )]
            )
            
        # Execute with new threshold
        results = self._execute_weaviate_query(stored_query['query'], params['threshold'])
        
        # Transform results
        df = self._transform_weaviate_results(results)
        
        # Create new preview ID
        preview_id = PreviewIdentifier.create_id(prefix="lit_query")
        
        # Format comparison preview
        old_df = pd.DataFrame(stored_query['dataframe'])
        preview = (
            f"Refined search results:\n\n"
            f"Previous results (threshold {stored_query['threshold']}): {len(old_df)} matches\n"
            f"New results (threshold {params['threshold']}): {len(df)} matches\n\n"
        ) + self._format_literature_preview(df, preview_id, params['threshold'])
        
        # Store new results in successful_queries_store
        store_updates = {
            'successful_queries_store': {
                preview_id: {
                    'query': stored_query['query'],
                    'threshold': params['threshold'],
                    'dataframe': df.to_dict('records'),
                    'metadata': {
                        'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_results': len(df),
                        'query_info': df.attrs.get('query_info', {}),
                        'summary': df.attrs.get('summary', {}),
                        'refined_from': params['query_id']
                    }
                }
            }
        }
        
        # Create messages list with preview
        messages = [
            ServiceMessage(
                service=self.name,
                content=preview,
                role="assistant"
            )
        ]
        
        # Generate LLM summary if we have results
        if not df.empty:
            try:
                llm_summary = self.summarize(df, context.get('chat_history', []))
                if llm_summary:
                    messages.append(
                        ServiceMessage(
                            service=self.name,
                            content=f"\n### Analysis Summary\n\n{llm_summary}",
                            message_type="info",
                            role="assistant"
                        )
                    )
            except Exception as e:
                print(f"Error generating LLM summary: {str(e)}")
                # Continue without summary
        
        return ServiceResponse(
            messages=messages,
            store_updates=store_updates
        )

    def _calculate_token_estimate(self, text: Optional[str]) -> int:
        """Estimate tokens for a piece of text.
        
        Uses the same token counting method as the LLM service
        for consistency and accuracy with Claude models.
        
        Args:
            text: Text to estimate tokens for, or None
            
        Returns:
            int: Token count (0 if text is None)
        """
        if text is None or not text.strip():
            return 0
        return self.count_tokens(text)
    
    def _organize_article_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Create list of article entries with their related references and entities.
        
        Organizes articles (sorted by score) with their cross-referenced items.
        Only includes high-priority fields to manage token usage.
        
        Args:
            df: DataFrame containing articles, references, and entities
            
        Returns:
            List of dicts, each containing:
            - article: Dict with filename, abstract, score
            - references: List of referenced titles
            - entities: List of named entities (type and name)
        """
        try:
            print("\n=== Organizing Article Relationships ===")
            articles = []
            
            # Get articles sorted by score
            article_rows = df[df['collection'] == 'Article'].sort_values('score', ascending=False)
            print(f"Processing {len(article_rows)} articles")
            
            for _, row in article_rows.iterrows():
                # Initialize article entry with high-priority fields
                article_entry = {
                    'article': {
                        'filename': row.get('Article_filename', 'Unknown'),
                        'abstract': row.get('Article_abstract', ''),
                        'score': float(row.get('score', 0.0))
                    },
                    'references': [],
                    'entities': []
                }
                
                # Process cross-references if present
                cross_refs = row.get('cross_references')
                if cross_refs:
                    try:
                        refs = json.loads(cross_refs)
                        
                        # Add referenced items
                        for ref_type, items in refs.items():
                            if ref_type == 'Reference':
                                for item in items:
                                    ref_id = item['id']
                                    ref_row = df[df['id'] == ref_id]
                                    if not ref_row.empty:
                                        article_entry['references'].append({
                                            'title': ref_row.iloc[0].get('Reference_title', 'Unknown Title')
                                        })
                            elif ref_type == 'NamedEntity':
                                for item in items:
                                    entity_id = item['id']
                                    entity_row = df[df['id'] == entity_id]
                                    if not entity_row.empty:
                                        article_entry['entities'].append({
                                            'type': entity_row.iloc[0].get('NamedEntity_type', 'Unknown Type'),
                                            'name': entity_row.iloc[0].get('NamedEntity_name', 'Unknown Name')
                                        })
                    except json.JSONDecodeError as e:
                        print(f"Error decoding cross-references: {str(e)}")
                        continue
                
                articles.append(article_entry)
            
            print(f"Organized {len(articles)} articles with their relationships")
            return articles
            
        except Exception as e:
            print(f"Error organizing article relationships: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _create_content_batches(
        self,
        articles: List[Dict],
        system_prompt: str,
        max_tokens: int = 8192
    ) -> List[List[Dict]]:
        """Create batches of articles that fit within token limits.
        
        Takes into account:
        - System prompt size
        - Content formatting overhead
        - Response space needed
        
        Args:
            articles: List of article entries from _organize_article_relationships
            system_prompt: The system prompt to be used
            max_tokens: Maximum total tokens allowed
            
        Returns:
            List of article batches that fit within token limits
        """
        try:
            print("\n=== Creating Content Batches ===")
            
            # Calculate token overhead
            system_tokens = self.count_tokens(system_prompt)
            formatting_tokens = 500  # Reserve space for JSON formatting
            response_tokens = 1500   # Reserve space for response
            
            # Calculate available tokens for content
            available_tokens = max_tokens - system_tokens - formatting_tokens - response_tokens
            print(f"Token budget: {available_tokens} for content")
            
            batches = []
            current_batch = []
            current_tokens = 0
            
            for article in articles:
                # Estimate tokens for this article entry
                article_tokens = (
                    self._calculate_token_estimate(article['article']['abstract']) +
                    self._calculate_token_estimate(article['article']['filename']) +
                    sum(self._calculate_token_estimate(ref['title']) 
                        for ref in article['references']) +
                    sum((self._calculate_token_estimate(ent['type']) + 
                         self._calculate_token_estimate(ent['name']))
                        for ent in article['entities'])
                )
                
                # Add formatting overhead for this entry
                article_tokens += 100  # JSON formatting per article
                
                if current_tokens + article_tokens > available_tokens:
                    if current_batch:  # Only add non-empty batches
                        batches.append(current_batch)
                        current_batch = []
                        current_tokens = 0
                
                current_batch.append(article)
                current_tokens += article_tokens
            
            # Add any remaining articles
            if current_batch:
                batches.append(current_batch)
            
            print(f"Created {len(batches)} batches")
            for i, batch in enumerate(batches):
                print(f"Batch {i+1}: {len(batch)} articles")
            
            return batches
            
        except Exception as e:
            print(f"Error creating content batches: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return []

    def _summarize_batch(
        self,
        batch: List[Dict],
        search_query: str,
        batch_index: int = 0,
        total_batches: int = 1
    ) -> str:
        """Summarize a batch of articles with their references and entities.
        
        Args:
            batch: List of article entries with their relationships
            search_query: Original search query
            batch_index: Index of this batch (0-based)
            total_batches: Total number of batches
            
        Returns:
            str: Batch summary focusing on scientific topics and relationships
        """
        try:
            print(f"\n=== Summarizing Batch {batch_index + 1}/{total_batches} ===")
            print(f"Articles in batch: {len(batch)}")
            
            # Create system prompt
            system_prompt = """You are a scientific literature analyst specializing in analyzing relationships between articles, their references, and scientific concepts.

Your task is to analyze a batch of scientific articles and their relationships, focusing on:

1. Scientific Topics and Concepts:
   - Major topics and themes across articles
   - Subtopics and their relationships
   - Research approaches (computational/experimental/theoretical)
   - Methods, discoveries, and hypotheses

2. Cross-Article Patterns:
   - Common themes between articles
   - Complementary or contrasting approaches
   - Evolution of ideas or methods
   - Shared references or entities

3. Named Entities Analysis:
   - Group and categorize entity types
   - Identify patterns in entity usage
   - Connect entities to research themes
   - Note significant entity co-occurrences

4. Reference Analysis:
   - Common reference patterns
   - How references support main themes
   - Evolution of ideas through citations
   - Research lineage suggested by references

Format your response as:
1. Major Topics (2-3 bullet points identifying the main scientific themes)
2. Research Approaches (computational/experimental/theoretical breakdown)
3. Key Findings (significant patterns or insights)
4. Entity Relationships (how named entities relate to topics)
5. Reference Patterns (how citations support the research)

IMPORTANT:
- Focus on scientific meaning and relationships
- Identify clear patterns and themes
- Link entities and references to main topics
- Note any significant gaps or biases
- If this is one of multiple batches, focus on patterns within this batch only

Keep your analysis focused and concise, emphasizing the most significant patterns and relationships."""

            # Format batch content for LLM
            content = {
                'search_query': search_query,
                'batch_info': {
                    'index': batch_index + 1,
                    'total_batches': total_batches
                },
                'articles': [{
                    'filename': article['article']['filename'],
                    'abstract': article['article']['abstract'],
                    'score': article['article']['score'],
                    'references': [ref['title'] for ref in article['references']],
                    'entities': [{
                        'type': ent['type'],
                        'name': ent['name']
                    } for ent in article['entities']]
                } for article in batch]
            }
            
            # Get LLM response
            response = self._call_llm([{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Please analyze these scientific articles and their relationships:\n{json.dumps(content, indent=2)}"
            }])
            
            print(f"Generated summary for batch {batch_index + 1}")
            return response.strip()
            
        except Exception as e:
            print(f"Error summarizing batch: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error generating batch summary: {str(e)}"
            
    def _combine_batch_summaries(self, summaries: List[str], search_query: str) -> str:
        """Combine multiple batch summaries into a cohesive analysis.
        
        Args:
            summaries: List of batch summaries
            search_query: Original search query
            
        Returns:
            str: Combined analysis focusing on cross-batch patterns
        """
        if len(summaries) == 1:
            return summaries[0]
            
        try:
            print("\n=== Combining Batch Summaries ===")
            
            system_prompt = """You are a scientific literature analyst tasked with combining multiple batch summaries into a cohesive analysis.

Your task is to:
1. Identify common themes across all batches
2. Note unique contributions from each batch
3. Synthesize a comprehensive view of the literature

Focus on:
- Major scientific themes and their variations
- Research approach patterns
- Entity and reference patterns
- Evolution of ideas across batches

Format your response as:
1. Overall Themes (3-4 major scientific topics)
2. Research Approaches (patterns across all batches)
3. Key Findings (significant insights from all batches)
4. Entity Relationships (cross-batch patterns)
5. Reference Patterns (citation patterns across batches)
6. Gaps and Future Directions

IMPORTANT:
- Emphasize cross-batch patterns
- Note any contradictions or variations
- Maintain focus on scientific meaning
- Consider how findings relate to the search query"""

            # Format content for LLM
            content = {
                'search_query': search_query,
                'batch_count': len(summaries),
                'batch_summaries': [
                    {'batch_index': i + 1, 'summary': summary}
                    for i, summary in enumerate(summaries)
                ]
            }
            
            # Get LLM response
            response = self._call_llm([{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": f"Please combine these batch summaries into a comprehensive analysis:\n{json.dumps(content, indent=2)}"
            }])
            
            print("Generated combined summary")
            return response.strip()
            
        except Exception as e:
            print(f"Error combining summaries: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error generating combined summary: {str(e)}"

    def summarize(self, df: pd.DataFrame, chat_history: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive summary of literature search results.
        
        This method:
        1. Organizes articles with their references and entities
        2. Creates token-aware batches
        3. Summarizes each batch
        4. Combines summaries if multiple batches
        
        Args:
            df: DataFrame containing search results
            chat_history: List of previous chat messages (for context)
            
        Returns:
            str: Comprehensive analysis of the literature results
        """
        try:
            start_time = datetime.now()
            print("\n=== Literature Search Summarization ===")
            print(f"Starting summarization at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get search query from DataFrame metadata
            query_info = df.attrs.get('query_info', {})
            search_query = query_info.get('text', 'literature search')
            print(f"Search query: {search_query}")
            
            # Stage 1: Organize articles
            stage_start = datetime.now()
            print("\nStage 1: Organizing Articles")
            articles = self._organize_article_relationships(df)
            if not articles:
                return "No articles found to summarize."
            stage_duration = datetime.now() - stage_start
            print(f"Stage 1 completed in: {stage_duration.total_seconds():.2f} seconds")
            print(f"Organized {len(articles)} articles")
            
            # Stage 2: Create batches
            stage_start = datetime.now()
            print("\nStage 2: Creating Batches")
            system_prompt = self._summarize_batch.__doc__  # Use method's docstring as base
            batches = self._create_content_batches(articles, system_prompt)
            if not batches:
                return "Error creating content batches for summarization."
            stage_duration = datetime.now() - stage_start
            print(f"Stage 2 completed in: {stage_duration.total_seconds():.2f} seconds")
            print(f"Created {len(batches)} batches")
            
            # Stage 3: Process batches
            stage_start = datetime.now()
            print("\nStage 3: Processing Batches")
            batch_summaries = []
            batch_times = []
            
            for i, batch in enumerate(batches):
                batch_start = datetime.now()
                print(f"\nProcessing batch {i+1}/{len(batches)}")
                print(f"Batch size: {len(batch)} articles")
                
                summary = self._summarize_batch(
                    batch=batch,
                    search_query=search_query,
                    batch_index=i,
                    total_batches=len(batches)
                )
                
                if summary and not summary.startswith("Error"):
                    batch_summaries.append(summary)
                    batch_duration = datetime.now() - batch_start
                    batch_times.append(batch_duration.total_seconds())
                    print(f"Batch {i+1} completed in: {batch_duration.total_seconds():.2f} seconds")
            
            if not batch_summaries:
                return "Error generating batch summaries."
                
            stage_duration = datetime.now() - stage_start
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            print(f"\nStage 3 completed in: {stage_duration.total_seconds():.2f} seconds")
            print(f"Average batch processing time: {avg_batch_time:.2f} seconds")
            
            # Stage 4: Combine summaries
            stage_start = datetime.now()
            print("\nStage 4: Combining Summaries")
            final_summary = self._combine_batch_summaries(batch_summaries, search_query)
            stage_duration = datetime.now() - stage_start
            print(f"Stage 4 completed in: {stage_duration.total_seconds():.2f} seconds")
            
            # Final timing summary
            total_duration = datetime.now() - start_time
            print("\n=== Summarization Complete ===")
            print(f"Total processing time: {total_duration.total_seconds():.2f} seconds")
            print(f"Articles processed: {len(articles)}")
            print(f"Batches processed: {len(batches)}")
            print(f"Average time per article: {total_duration.total_seconds() / len(articles):.2f} seconds")
            
            return final_summary.strip()
            
        except Exception as e:
            print(f"Error in literature summarization: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error generating literature summary: {str(e)}"

    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """
        Required by LLMServiceMixin but not used by LiteratureService.
        We only use LLM capabilities for summarization, not message processing.
        
        Args:
            message: The message to process
            chat_history: List of previous chat messages
            
        Raises:
            NotImplementedError: This method is not used by LiteratureService
        """
        raise NotImplementedError("LiteratureService does not use LLM for message processing")