"""
Literature service implementation.

This service handles literature search queries and refinements through Weaviate.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
from datetime import datetime
import re
import traceback
import json

from .base import (
    ChatService, 
    ServiceResponse, 
    ServiceMessage, 
    PreviewIdentifier
)

from weaviate_integration import WeaviateConnection
from weaviate_manager.query.manager import QueryManager

class LiteratureService(ChatService):
    """Service for literature search and refinement."""
    
    def __init__(self):
        super().__init__("literature")
        # Register our prefix with the ID generator
        PreviewIdentifier.register_prefix("literature")
    
    def _transform_weaviate_results(self, json_results: dict) -> pd.DataFrame:
        """Transform Weaviate JSON results into a unified DataFrame with consistent structure.
        
        Args:
            json_results: Dict containing:
                - query_info: Query parameters and metadata
                - raw_results: Direct hits from collections
                - unified_results: Unified Article records with cross-references
                
        Returns:
            pd.DataFrame with columns:
                - score: Search relevance score
                - id: Record identifier (uuid from raw_results, id from unified_results)
                - collection: Source collection name
                - source: Original collection for raw_results, source field for unified
                - {collection}_{property}: Dynamic property columns
                - cross_references: JSON string of cross-references (or None)
                
            DataFrame.attrs contains:
                - query_info: Original query parameters
                - summary: Collection counts and statistics
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
        
        Args:
            query: Search query text
            min_score: Minimum score threshold
            
        Returns:
            Dict containing query results or empty dict if no results/error
        """
        print("\n=== Weaviate Query Debug ===")
        print(f"Query: '{query}'")
        print(f"Min score: {min_score}")
        
        try:
            connection = WeaviateConnection()
            with connection.get_client() as client:
                if not client:
                    print("Error: No Weaviate client available")
                    return {}
                
                print("Client connection successful")
                
                # Use the QueryManager from weaviate_manager
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
        """Generate a formatted preview of Weaviate search results for chat display.
        
        Args:
            df: DataFrame from transform_weaviate_results
            max_rows: Maximum number of rows to show per collection
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
        """Format literature results preview with query ID and conversion instructions.
        
        Args:
            df: DataFrame from transform_weaviate_results
            query_id: Unique identifier for this literature query
            threshold: Current relevance threshold
            
        Returns:
            str: Formatted preview with ID and instructions
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
    
    def _extract_query_id_from_message(self, message: str) -> Optional[str]:
        """Extract literature query ID from refinement command.
        
        Args:
            message: User's chat message
            
        Returns:
            str: Query ID if found, None otherwise
            
        Examples:
            >>> _extract_query_id_from_message("refine literature_20250207_123456_orig with threshold 0.7")
            'literature_20250207_123456_orig'
        """
        patterns = [
            r"(?:refine|update|modify)\s+(literature_\d{8}_\d{6}(?:_orig|_alt\d+))",
            r"(literature_\d{8}_\d{6}(?:_orig|_alt\d+))\s+(?:with|using|at)\s+threshold"
        ]
        
        message = message.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_threshold_from_message(self, message: str) -> Optional[float]:
        """Extract threshold value from user message.
        
        Args:
            message: User's chat message
            
        Returns:
            Float threshold value or None if not found/invalid
            
        Examples:
            >>> _extract_threshold_from_message("Use threshold 0.3")
            0.3
            >>> _extract_threshold_from_message("Set cutoff to 0.7")
            0.7
        """
        patterns = [
            r"(?:use|set|apply|with)\s+(?:a\s+)?(?:threshold|cutoff|score|limit)\s+(?:of\s+)?(\d+\.?\d*)",
            r"threshold\s+(?:of\s+)?(\d+\.?\d*)",
            r"cutoff\s+(?:of\s+)?(\d+\.?\d*)"
        ]
        
        message = message.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                try:
                    threshold = float(match.group(1))
                    if 0 <= threshold <= 1:
                        return threshold
                except ValueError:
                    continue
        
        return None
    
    def _generate_literature_query_id(self, query: str, threshold: float, previous_id: str = None) -> str:
        """Generate a unique ID for literature queries.
        
        Args:
            query: The literature search query
            threshold: The relevance threshold used
            previous_id: Optional previous ID for refinements
            
        Returns:
            str: Query ID in format literature_YYYYMMDD_HHMMSS_(orig|altN)
        """
        if previous_id is not None:
            return PreviewIdentifier.create_id(previous_id=previous_id)
        return PreviewIdentifier.create_id(prefix="literature")
    
    def _is_literature_query(self, message: str) -> Tuple[bool, Optional[str]]:
        """Detect if a message is requesting literature information.
        
        Returns:
            Tuple[bool, Optional[str]]: (is_literature_query, extracted_query)
        """
        patterns = [
            # [find] [me] <papers|literature|references> about <X>
            r'(?:find\s+)?(?:me\s+)?(?:papers|literature|references)\s+about\s+(.+?)(?:\?|$)',
            
            # [what] [do|does] [the] <papers|literature|references> say about <X>
            r'(?:what\s+)?(?:do|does\s+)?(?:the\s+)?(?:papers|literature|references)\s+say\s+about\s+(.+?)(?:\?|$)'
        ]
        
        print("\n=== Literature Query Detection Debug ===")
        print(f"Input message: '{message}'")
        normalized = message.lower().strip()
        print(f"Normalized message: '{normalized}'")
        print("\nTrying patterns:")
        
        for pattern in patterns:
            print(f"\nTrying pattern: {pattern}")
            match = re.search(pattern, normalized)
            if match:
                query = match.group(1).strip()
                print(f"Match found! Extracted query: '{query}'")
                return True, query
        
        print("No literature query patterns matched")
        return False, None
    
    def can_handle(self, message: str) -> bool:
        """Detect both initial queries and refinement requests."""
        # Check for literature query
        is_lit, _ = self._is_literature_query(message)
        if is_lit:
            return True
            
        # Check for refinement request
        threshold = self._extract_threshold_from_message(message)
        query_id = self._extract_query_id_from_message(message)
        if threshold is not None and query_id is not None:
            # Only handle refinements of our own queries
            return query_id.startswith('literature_')
            
        return False
        
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Parse both initial queries and refinement requests."""
        # Check for refinement first
        threshold = self._extract_threshold_from_message(message)
        query_id = self._extract_query_id_from_message(message)
        
        if threshold is not None and query_id is not None:
            return {
                'type': 'refinement',
                'query_id': query_id,
                'threshold': threshold
            }
            
        # Must be an initial query
        _, query = self._is_literature_query(message)
        return {
            'type': 'initial',
            'query': query,
            'threshold': 0.3  # Default threshold
        }
        
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute either initial search or refinement."""
        if params['type'] == 'refinement':
            return self._execute_refinement(params, context)
        else:
            return self._execute_search(params, context)
            
    def _execute_search(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute initial literature search."""
        # Execute search
        results = self._execute_weaviate_query(params['query'], params['threshold'])
        
        # Transform results
        df = self._transform_weaviate_results(results)
        
        # Create preview ID using our internal method
        preview_id = self._generate_literature_query_id(params['query'], params['threshold'])
        
        # Format preview (this will be shown in chat)
        preview = self._format_literature_preview(df, preview_id, params['threshold'])
        
        # Store full results
        store_updates = {
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
        
        return ServiceResponse(
            messages=[ServiceMessage(
                service=self.name,
                content=preview,
                role="assistant"
            )],
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
        
        # Create new preview ID using same method as initial search
        preview_id = self._generate_literature_query_id(stored_query['query'], params['threshold'], params['query_id'])
        
        # Format comparison preview
        old_df = pd.DataFrame(stored_query['dataframe'])
        preview = (
            f"Refined search results:\n\n"
            f"Previous results (threshold {stored_query['threshold']}): {len(old_df)} matches\n"
            f"New results (threshold {params['threshold']}): {len(df)} matches\n\n"
        ) + self._format_literature_preview(df, preview_id, params['threshold'])
        
        # Store new results
        store_updates = {
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
        
        return ServiceResponse(
            messages=[ServiceMessage(
                service=self.name,
                content=preview,
                role="assistant"
            )],
            store_updates=store_updates
        )