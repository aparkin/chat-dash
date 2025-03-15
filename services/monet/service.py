"""
MONet service implementation.

This service provides access to EMSL's Molecular Observation Network (MONet) soil data.
It implements a modular interface for:
1. Direct query execution with filtering and geographic constraints
2. Natural language query processing with LLM-powered interpretation
3. Result analysis and interpretation with statistical summaries
4. Dataset conversion with automatic profiling

Key Features:
- Flexible query system with support for:
  * Numeric comparisons and ranges
  * Text matching (contains, exact, starts_with)
  * Geographic constraints (point radius, bounding box)
  * Date filtering
- Two-step query process:
  1. Query validation and interpretation
  2. Execution with preview and analysis
- Automatic data profiling using ydata-profiling
- Integration with ChatDash dataset system
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import re
from datetime import datetime
import json
import pandas as pd
import traceback
from ydata_profiling import ProfileReport

from services.base import (
    ChatService,
    ServiceResponse,
    ServiceMessage,
    PreviewIdentifier,
    MessageType
)
from services.llm_service import LLMServiceMixin

from .models import MONetConfig, QueryResult
from .data_manager import MONetDataManager
from .query_builder import MONetQueryBuilder
from .query_context import QueryContext
from .prompts import load_prompt

class MONetService(ChatService, LLMServiceMixin):
    """Service for querying and analyzing MONet soil database.
    
    This service provides a comprehensive interface to the MONet soil database,
    supporting both direct JSON queries and natural language interactions.
    
    Key Features:
    1. Query Types:
       - Direct JSON queries with filters and geographic constraints
       - Natural language queries with LLM interpretation
       - Query execution by ID
       - Dataset conversion with profiling
       
    2. Filter Operations:
       - Numeric: >, <, >=, <=, ==, range
       - Text: contains, exact, starts_with
       - Date: range, >, <, ==
       - Geographic: point radius, bounding box
       
    3. Data Management:
       - Query validation and interpretation
       - Result preview with statistics
       - Automatic data profiling
       - Dataset conversion
       
    4. Commands:
       - Natural language: monet: [question]
       - Execute query: monet.search [query_id]
       - Convert to dataset: convert [query_id] to dataset
       - Service info: tell me about monet
       
    The service follows a two-step query process:
    1. Validation and Interpretation:
       - Query is validated for correctness
       - LLM provides interpretation and suggestions
       - Query ID is generated for future reference
       
    2. Execution and Analysis:
       - Query is executed on the database
       - Results are previewed with statistics
       - LLM provides analysis and insights
       - Results can be converted to datasets
    """
    
    # Regex patterns for extracting monet code blocks
    query_block_re = re.compile(r'```monet\s*(?:\n|)(.*?)(?:\n|)\s*```', re.DOTALL)
    
    def __init__(self):
        """Initialize MONet service."""
        # Create default config
        config = MONetConfig()
        
        ChatService.__init__(self, config.name)
        LLMServiceMixin.__init__(self, config.name)
        
        # Initialize stores
        self.successful_queries_store = {}
        
        try:
            # Register preview ID prefix (ignore if already registered)
            PreviewIdentifier.register_prefix('monet_query')
        except ValueError:
            pass  # Prefix already registered
        
        # Initialize data manager and query builder
        print("\nInitializing MONet service...")
        self.data_manager = MONetDataManager(config)
        # Force data loading now instead of lazy loading
        _ = self.data_manager.unified_df
        print("MONet service initialization complete.")
        
        self.query_builder = MONetQueryBuilder(self.data_manager)
        self.query_context = QueryContext(self.data_manager)
        
        # Command patterns
        self.execution_patterns = [
            r'^monet\.(?:search|query)\s+(?:monet_query_)?\d{8}_\d{6}(?:_orig|_alt\d+)\b',
            r'^monet\.(?:search|query)\.?$',
            r'tell\s+me\s+about\s+monet\b',
            r'^convert\s+monet_query_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b'
        ]
        self.execution_res = [re.compile(p, re.IGNORECASE) for p in self.execution_patterns]
    
    def can_handle(self, message: str) -> bool:
        """Check if message can be handled by this service."""
        message = message.strip()
        
        # Check for monet code blocks
        if match := self.query_block_re.search(message):
            try:
                # Validate it's a proper JSON query
                query_text = match.group(1).strip()
                json.loads(query_text)
                return True
            except json.JSONDecodeError:
                # Still return True so we can provide helpful error messages
                return True
            
        # Check for execution commands
        for pattern in self.execution_res:
            if pattern.search(message):
                return True
                
        # Check for natural language query
        if message.lower().startswith('monet:'):
            return True
            
        return False
    
    def parse_request(self, message: str) -> Dict[str, Any]:
        """Parse message into request parameters."""
        message = message.strip()
        message_lower = message.lower()
        
        # Check for direct query
        if match := self.query_block_re.search(message):
            query_text = match.group(1).strip()
            try:
                query = json.loads(query_text)
                return {
                    'type': 'direct_query',
                    'query': query,
                    'raw_text': query_text  # Keep original text for error context
                }
            except json.JSONDecodeError as e:
                return {
                    'type': 'invalid_query',
                    'error': str(e),
                    'raw_text': query_text
                }
        
        # Check for service info
        if message_lower == 'tell me about monet':
            return {'type': 'info'}
        
        # Check for query execution
        if message_lower in ['monet.search', 'monet.query']:
            return {
                'type': 'execute_query',
                'query_id': None  # Use most recent
            }
            
        if match := re.match(r'^monet\.(?:search|query)\s+(monet_query_\d{8}_\d{6}(?:_orig|_alt\d+))\b', message_lower):
            return {
                'type': 'execute_query',
                'query_id': match.group(1)
            }
        
        # Check for dataset conversion
        if match := re.match(r'^convert\s+(monet_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message_lower):
            return {
                'type': 'convert_dataset',
                'query_id': match.group(1)
            }
        
        # Check for natural language query
        if message_lower.startswith('monet:'):
            return {
                'type': 'natural_query',
                'query': message[6:].strip()
            }
        
        raise ValueError(f"Unable to parse request from message: {message}")
    
    def execute(self, request: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the parsed request."""
        request_type = request['type']
        
        try:
            if request_type == 'info':
                return self._handle_info_request(context)
            elif request_type == 'direct_query':
                return self._handle_direct_query(request['query'], context)
            elif request_type == 'invalid_query':
                return self._handle_invalid_query(request['error'], request['raw_text'], context)
            elif request_type == 'natural_query':
                return self._handle_natural_query(request['query'], context)
            elif request_type == 'execute_query':
                return self._handle_query_execution(request['query_id'], context)
            elif request_type == 'convert_dataset':
                return self._handle_dataset_conversion(request['query_id'], context)
            else:
                raise ValueError(f"Unknown request type: {request_type}")
        except Exception as e:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=str(e),
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_info_request(self, context: Dict[str, Any]) -> ServiceResponse:
        """Handle 'tell me about monet' request."""
        # Get data context
        data_context = self.data_manager.get_dataframe_context()
        
        # Generate structured summary
        structured_summary = []
        
        # Dataset overview
        stats = data_context['statistics']
        structured_summary.extend([
            "### MONet Soil Database Overview",
            f"Total Samples: {stats['total_rows']:,}",
            f"Total Columns: {stats['total_columns']:,}",
            "\n### Measurement Types:"
        ])
        
        # Group measurements by type
        measurements = {
            "Core Measurements": [col for col in stats['numeric_columns'] if any(x in col for x in ['carbon', 'nitrogen', 'sulfur'])],
            "Nutrient Measurements": [col for col in stats['numeric_columns'] if any(x in col for x in ['nh4', 'no3', 'sulfate'])],
            "Element Measurements": [col for col in stats['numeric_columns'] if any(x in col for x in ['calcium', 'magnesium', 'potassium', 'sodium', 'iron', 'zinc', 'copper'])],
            "Soil Properties": [col for col in stats['numeric_columns'] if any(x in col for x in ['bulk_density', 'ph', 'cation'])],
            "Microbial Properties": [col for col in stats['numeric_columns'] if any(x in col for x in ['mbc', 'mbn', 'respiration'])]
        }
        
        for category, cols in measurements.items():
            if cols:
                structured_summary.append(f"\n{category}:")
                for col in cols:
                    desc = data_context['column_descriptions'].get(col, '')
                    col_stats = stats['numeric_columns'][col]
                    structured_summary.append(
                        f"- {desc}\n"
                        f"  Range: {col_stats['min']:.2f} to {col_stats['max']:.2f}, "
                        f"Mean: {col_stats['mean']:.2f}"
                    )
        
        # Geographic coverage
        geo = data_context['geographic_coverage']
        if geo:
            structured_summary.extend([
                "\n### Geographic Coverage:",
                f"Latitude: {geo['latitude_range']['min']:.2f}° to {geo['latitude_range']['max']:.2f}°",
                f"Longitude: {geo['longitude_range']['min']:.2f}° to {geo['longitude_range']['max']:.2f}°",
                f"Total Locations: {geo['total_locations']:,}"
            ])
        
        # Temporal coverage
        if 'collection_date' in stats['temporal_columns']:
            date_range = stats['temporal_columns']['collection_date']
            structured_summary.extend([
                "\n### Temporal Coverage:",
                f"Collection Period: {date_range['min']} to {date_range['max']}"
            ])
        
        # Get LLM interpretation
        interpretation = self._call_llm(
            messages=[{'role': 'user', 'content': 'Interpret this soil database information'}],
            system_prompt=load_prompt('info', {'data_context': data_context})
        )
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service=self.name,
                    content="\n".join(structured_summary),
                    message_type=MessageType.INFO
                ),
                ServiceMessage(
                    service=self.name,
                    content=interpretation,
                    message_type=MessageType.SUMMARY
                )
            ]
        )
    
    def _handle_direct_query(self, query: Dict, context: Dict[str, Any]) -> ServiceResponse:
        """Handle direct query validation and interpretation.
        
        This is the first step of the two-step query process:
        1. Validate query and get LLM interpretation
        2. Generate ID for future execution
        
        The query will be executed when the user runs monet.search
        """
        # Validate query
        valid, message = self.query_builder.validate_query(query)
        
        # Get LLM interpretation
        prompt_context = {
            'query_details': {
                'query': query,
                'validation': (valid, message)
            },
            'data_context': self.data_manager.get_dataframe_context(),
            'chat_history': self._format_chat_history(context.get('chat_history', []))
        }
        
        interpretation = self._call_llm(
            messages=[{'role': 'user', 'content': 'Validate and explain this query'}],
            system_prompt=load_prompt('validation', prompt_context)
        )
        
        if not valid:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=interpretation,
                        message_type=MessageType.ERROR
                    )
                ]
            )
        
        # Generate query ID for future reference
        query_id = PreviewIdentifier.create_id('monet_query')
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service=self.name,
                    content=interpretation,
                    message_type=MessageType.SUGGESTION
                ),
                ServiceMessage(
                    service=self.name,
                    content=f"Query ID: {query_id}\n\n```monet\n{json.dumps(query, indent=2)}\n```\n\nTo execute this query, run: monet.search {query_id}",
                    message_type=MessageType.PREVIEW
                )
            ]
        )
    
    def _handle_invalid_query(self, error: str, raw_text: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle invalid query by providing helpful suggestions."""
        # Get LLM interpretation and suggestions
        prompt_context = {
            'error': error,
            'query_text': raw_text,
            'data_context': self.data_manager.get_dataframe_context(),
            'chat_history': self._format_chat_history(context.get('chat_history', []))
        }
        
        # Get LLM help
        system_prompt = f"""You are helping fix an invalid MONet query.
The user tried to write a query but it had JSON errors.

Original query text:
{raw_text}

Error:
{error}

Please:
1. Explain what's wrong with the query
2. Provide a corrected version in a ```monet``` code block
3. Explain what the corrected query will do

Remember to format the corrected query as valid JSON inside a ```monet``` code block."""

        suggestions = self._call_llm(
            messages=[{'role': 'user', 'content': 'Please help fix this query'}],
            system_prompt=system_prompt
        )
        
        # Add query IDs to any suggestions
        suggestions_with_ids = self._add_query_ids(suggestions)
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service=self.name,
                    content=suggestions_with_ids,
                    message_type=MessageType.SUGGESTION
                )
            ]
        )
    
    def _handle_natural_query(self, query: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle natural language query."""
        try:
            # Prepare context for LLM
            prompt_context = {
                'data_context': self.data_manager.get_dataframe_context(),
                'user_request': query,
                'chat_history': self._format_chat_history(context.get('chat_history', []))
            }
            
            # Load prompt and get LLM suggestions
            system_prompt = load_prompt('natural', prompt_context)
            suggestions = self._call_llm(
                messages=[{'role': 'user', 'content': query}],
                system_prompt=system_prompt
            )
            
            # Add query IDs to suggestions
            try:
                suggestions_with_ids = self._add_query_ids(suggestions)
            except Exception as e:
                print(f"Error adding query IDs: {str(e)}")
                suggestions_with_ids = suggestions
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=suggestions_with_ids,
                        message_type=MessageType.SUGGESTION
                    )
                ]
            )
        except Exception as e:
            print(f"Error in natural language query: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            raise
    
    def _handle_query_execution(self, query_id: Optional[str], context: Dict[str, Any]) -> ServiceResponse:
        """Handle query execution request."""
        chat_history = context.get('chat_history', [])
        stored_queries = context.get('successful_queries_store', {})
        
        if query_id is None:
            # Find most recent query ID from chat history
            query_id = self._find_recent_query_id(chat_history)
            if not query_id:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="No recent queries found to execute",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
        
        # Check if we already have executed results for this query
        stored_query = stored_queries.get(query_id, {})
        if stored_query.get('executed', False):
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content="Query has already been executed. Use 'convert [query_id] to dataset' to access results.",
                        message_type=MessageType.INFO
                    )
                ]
            )
        
        # Get query definition - first check store, then chat history
        query = None
        if 'query' in stored_query:
            query = stored_query['query']
        else:
            query = self._find_query_from_id(query_id, chat_history)
            
        if not query:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Could not find query definition for ID: {query_id}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
        
        # Execute query
        result = self.query_builder.execute_query(query)
        
        # Generate preview with query ID
        preview = self._format_preview(result, query_id)
        
        # Get LLM interpretation
        prompt_context = {
            'query_details': {'query': query},
            'results_stats': self._format_results_stats(result),
            'chat_history': self._format_chat_history(chat_history)
        }
        
        interpretation = self._call_llm(
            messages=[{'role': 'user', 'content': 'Interpret these query results'}],
            system_prompt=load_prompt('results', prompt_context)
        )
        
        # Store results
        store_updates = {
            'successful_queries_store': {
                query_id: {
                    **result.to_preview(),
                    'executed': True  # Mark as executed
                }
            }
        }
        
        return ServiceResponse(
            messages=[
                ServiceMessage(
                    service=self.name,
                    content=preview,
                    message_type=MessageType.PREVIEW
                ),
                ServiceMessage(
                    service=self.name,
                    content=interpretation,
                    message_type=MessageType.SUMMARY
                )
            ],
            store_updates=store_updates
        )
    
    def _handle_dataset_conversion(self, query_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle dataset conversion request."""
        # Get stored query results
        stored_queries = context.get('successful_queries_store', {})
        stored_query = stored_queries.get(query_id)
        
        if not stored_query:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"No stored query found with ID: {query_id}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
        
        try:
            # Execute query again to get fresh DataFrame
            query = stored_query['query']
            result = self.query_builder.execute_query(query)
            df = result.dataframe
            
            # Convert DataFrame to proper format for storage
            df_dict = df.to_dict('records')
            
            # Generate profile report
            try:
                print(f"Generating profile for dataset {query_id} with {len(df)} rows and {len(df.columns)} columns...")
                profile = ProfileReport(df, 
                                    title=f"MONet Query {query_id} Profile",
                                    minimal=True,  # For faster processing
                                    html={'style': {'full_width': True}},
                                    progress_bar=False,
                                    correlations={'pearson': {'calculate': True}},
                                    missing_diagrams={'matrix': False},
                                    samples=None)
                profile_html = profile.to_html()
                print(f"Successfully generated profile for dataset {query_id}")
            except Exception as e:
                print(f"Warning: Profile generation failed: {str(e)}")
                print(f"Traceback:\n{traceback.format_exc()}")
                profile_html = None
            
            # Create dataset entry with proper metadata and full DataFrame
            dataset_entry = {
                'metadata': {
                    'source': f"MONet Query {query_id}",
                    'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'query': stored_query['query'],
                    'rows': len(df),
                    'columns': list(df.columns),
                    'selectable': True,  # Enable dataset selection
                    'transformations': []  # Track any transformations
                },
                'df': df_dict,  # Store full DataFrame
                'profile_report': profile_html  # Store HTML version of profile
            }
            
            # Update stores
            store_updates = {
                'datasets_store': {query_id: dataset_entry},
                'successful_queries_store': {
                    k: v for k, v in stored_queries.items() if k != query_id
                }
            }
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"✓ Query results converted to dataset '{query_id}' with profile",
                        message_type=MessageType.INFO
                    )
                ],
                store_updates=store_updates
            )
            
        except Exception as e:
            print(f"Error in dataset conversion: {str(e)}")
            print(f"Traceback:\n{traceback.format_exc()}")
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error converting query results to dataset: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _format_chat_history(self, history: List[Dict]) -> str:
        """Format chat history for LLM context."""
        formatted = []
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()
            if content:
                formatted.append(f"{role}: {content}")
        return "\n".join(formatted)
    
    def _format_preview(self, result: QueryResult, query_id: Optional[str] = None) -> str:
        """Format query results for preview.
        
        Shows:
        1. Primary columns (ID, name, location)
        2. Query-relevant columns
        3. Truncated text values
        
        Args:
            result: Query result to format
            query_id: Optional query ID to include in preview text
        """
        df = result.dataframe
        
        # Always show these primary columns if they exist
        primary_cols = [
            'id', 'title', 'latitude', 'longitude', 'id_sample','core_section',
            'collection_date'
        ]
        preview_cols = [col for col in primary_cols if col in df.columns]
        
        # Add query-relevant columns
        if hasattr(result, 'metadata') and 'query' in result.metadata:
            query = result.metadata['query']
            if 'filters' in query:
                for filter_group in query['filters']:
                    preview_cols.extend(col for col in filter_group.keys() 
                                      if col in df.columns and col not in preview_cols)
        
        # Add up to 3 measurement columns that have non-null values
        measurement_cols = [col for col in df.columns 
                           if col.endswith('_has_numeric_value') 
                           and df[col].notna().any()
                           and col not in preview_cols]
        preview_cols.extend(measurement_cols[:3])
        
        # Create preview DataFrame
        preview_df = df[preview_cols].head()
        
        # Format values
        formatted_df = preview_df.copy()
        for col in formatted_df.columns:
            if col.endswith('_has_numeric_value'):
                # Format numeric values with 3 decimal places
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")
            elif formatted_df[col].dtype == 'object':
                # Truncate text values to 30 chars
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: f"{str(x)[:30]}..." if pd.notna(x) and len(str(x)) > 30 else x
                )
        
        preview = [
            "### Query Results\n",
            f"Total rows: {len(df)}",
            f"Total columns: {len(df.columns)}",
            f"\nShowing {len(preview_cols)} columns: {', '.join(preview_cols)}",
            "\nPreview:\n",
            formatted_df.to_markdown(index=False, tablefmt="pipe")
        ]
        
        # Add dataset conversion instruction with actual query ID if available
        if query_id:
            preview.append(f"\nTo convert to dataset use: convert {query_id} to dataset")
        else:
            preview.append("\nTo convert to dataset use: convert [query_id] to dataset")
        
        return "\n".join(preview)
    
    def _format_results_stats(self, result: QueryResult) -> str:
        """Format results statistics for LLM."""
        stats = []
        df = result.dataframe
        
        # Basic counts
        stats.append(f"Total Results: {len(df)}")
        
        # If no results, return early with informative message
        if len(df) == 0:
            stats.append("\nNo results found matching the query criteria.")
            return "\n".join(stats)
        
        # Geographic distribution
        if 'latitude' in df.columns and 'longitude' in df.columns:
            lat_stats = df['latitude'].describe()
            lon_stats = df['longitude'].describe()
            stats.append("\nGeographic Coverage:")
            stats.append(f"- Latitude: {lat_stats['min']:.2f} to {lat_stats['max']:.2f}°")
            stats.append(f"- Longitude: {lon_stats['min']:.2f} to {lon_stats['max']:.2f}°")
        
        # Measurement statistics
        measurement_cols = [col for col in df.columns if col.endswith('_has_numeric_value')]
        if measurement_cols:
            stats.append("\nMeasurement Statistics:")
            for col in measurement_cols:
                try:
                    base_name = col.replace('_has_numeric_value', '')
                    col_stats = df[col].describe()
                    stats.append(f"\n{base_name}:")
                    stats.append(f"- Range: {col_stats['min']:.2f} to {col_stats['max']:.2f}")
                    stats.append(f"- Mean: {col_stats['mean']:.2f}")
                    stats.append(f"- Std Dev: {col_stats['std']:.2f}")
                except Exception as e:
                    print(f"Warning: Error calculating statistics for column {col}: {str(e)}")
                    continue
        
        return "\n".join(stats)
    
    def _add_query_ids(self, text: str) -> str:
        """Add query IDs to monet code blocks in text."""
        if '```monet' not in text:
            return text
        
        modified = []
        current_pos = 0
        previous_id = None  # Store the most recently generated ID
        
        matches = list(self.query_block_re.finditer(text))
        
        for i, match in enumerate(matches):
            start, end = match.span()
            content = match.group(1)
            
            # Add text before this block
            modified.append(text[current_pos:start])
            
            try:
                # Clean and parse the content
                clean_content = content.strip()
                query = json.loads(clean_content)
                
                if isinstance(query, dict):
                    # Generate query ID
                    if previous_id is None:
                        # First block - create new ID
                        query_id = PreviewIdentifier.create_id(prefix='monet_query')
                    else:
                        # Subsequent blocks - create alternative version from previous ID
                        query_id = PreviewIdentifier.create_id(previous_id=previous_id)
                    
                    previous_id = query_id  # Update for next iteration
                    
                    # Format and add the block with ID
                    formatted_query = json.dumps(query, indent=2)
                    block_text = f"```monet\n{formatted_query}\n```\nQuery ID: {query_id}\n"
                    modified.append(block_text)
                else:
                    modified.append(text[start:end])
            except json.JSONDecodeError:
                modified.append(text[start:end])
            except Exception:
                modified.append(text[start:end])
            
            current_pos = end
        
        # Add any remaining text
        modified.append(text[current_pos:])
        
        return ''.join(modified)
    
    def _find_query_from_id(self, query_id: str, chat_history: List[Dict]) -> Optional[Dict]:
        """Find query definition from chat history by ID."""
        for msg in reversed(chat_history):
            content = msg.get('content', '')
            
            # First find the monet code block that's associated with this ID
            block_patterns = [
                # ID before block
                f"Query ID: {query_id}[\\s\\n]*```monet[\\s\\n]*(.*?)[\\s\\n]*```",
                # ID after block
                f"```monet[\\s\\n]*(.*?)[\\s\\n]*```[\\s\\n]*Query ID: {query_id}",
                # Just find any monet block
                r"```monet\s*\n(.*?)\n\s*```"
            ]
            
            for pattern in block_patterns:
                if match := re.search(pattern, content, re.DOTALL):
                    try:
                        # Extract and clean the query text
                        query_text = match.group(1).strip()
                        
                        # Try to find a complete JSON object
                        json_start = query_text.find('{')
                        json_end = query_text.rfind('}')
                        
                        if json_start >= 0 and json_end > json_start:
                            json_text = query_text[json_start:json_end + 1]
                            
                            try:
                                query = json.loads(json_text)
                                if isinstance(query, dict):
                                    return query
                            except json.JSONDecodeError:
                                continue
                    except Exception as e:
                        print(f"Error processing query match: {str(e)}")
                        continue
        
        return None
    
    def _find_recent_query_id(self, chat_history: List[Dict]) -> Optional[str]:
        """Find most recent query ID in chat history."""
        for msg in reversed(chat_history):
            content = msg.get('content', '')
            if match := re.search(r'Query ID: (monet_query_\d{8}_\d{6}(?:_orig|_alt\d+))', content):
                return match.group(1)
        return None

    def get_help_text(self) -> str:
        """Get help text for MONet service."""
        return """
🌍 **MONet Soil Database**
- Search MONet database: `monet: [natural language query]`
- Create a direct query: 
  ```monet
  {
    "filters": [{"field": "pH", "op": ">", "value": 6}],
    "geo": {"type": "point_radius", "coordinates": [35.9, -79.05], "radius_km": 100}
  }
  ```
- Execute a query: `monet.search monet_query_id`
- Convert query to dataset: `convert monet_query_id to dataset`
"""

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Get current service status information."""
        # Only report status if data isn't loaded yet
        if not hasattr(self.data_manager, '_unified_df') or self.data_manager._unified_df is None:
            return {
                'status': 'initializing',
                'ready': False
            }
        return None  # Use default status when ready

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for MONet capabilities."""
        return """
MONet Soil Database Capabilities:

1. Query Types:
   - Direct JSON queries with filters and geographic constraints
   - Natural language queries converted to structured format
   - Query execution by ID
   - Dataset conversion

2. Filter Types:
   a) Measurement Filters:
      - Numeric comparisons (>, <, >=, <=, ==)
      - Range queries ([min, max])
      - Units handled automatically
   
   b) Geographic Filters:
      - Point with radius search
      - Bounding box search
      - Coordinate validation
   
   c) Text Filters:
      - Contains (case-insensitive)
      - Exact match
      - Starts with
   
   d) Date Filters:
      - Range queries
      - Before/after comparisons
      - Exact date match

3. Query Structure:
   ```monet
   {
     "filters": [
       {
         "column_name": [
           {"operation": "range", "value": [min, max]},
           {"operation": ">=", "value": number}
         ]
       }
     ],
     "geo_point": {
       "latitude": float,
       "longitude": float,
       "radius_km": float
     }
   }
   ```

4. Response Types:
   - Query validation and explanation
   - Result previews with statistics
   - Geographic distribution analysis
   - Scientific interpretation
"""

    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> ServiceResponse:
        """Process a natural language message to generate query suggestions."""
        try:
            # Get data context
            data_context = self.data_manager.get_dataframe_context()
            
            # Format data context for LLM
            context_summary = []
            
            # Add measurement ranges
            context_summary.append("Available Measurements:")
            for col, stats in data_context['statistics'].get('numeric_columns', {}).items():
                if col.endswith('_has_numeric_value'):
                    desc = data_context['column_descriptions'].get(col, col)
                    context_summary.append(
                        f"- {desc}\n"
                        f"  Range: {stats['min']:.2f} to {stats['max']:.2f}, "
                        f"Mean: {stats['mean']:.2f}"
                    )
            
            # Add geographic coverage
            geo = data_context.get('geographic_coverage', {})
            if geo:
                context_summary.extend([
                    "\nGeographic Coverage:",
                    f"Latitude: {geo.get('latitude_range', {}).get('min', 0):.2f}° to {geo.get('latitude_range', {}).get('max', 0):.2f}°",
                    f"Longitude: {geo.get('longitude_range', {}).get('min', 0):.2f}° to {geo.get('longitude_range', {}).get('max', 0):.2f}°"
                ])
            
            # Prepare context for LLM
            prompt_context = {
                'data_context': "\n".join(context_summary),
                'user_request': message,
                'chat_history': self._format_chat_history(chat_history)
            }
            
            # Get LLM suggestions
            suggestions = self._call_llm(
                messages=[{'role': 'user', 'content': message}],
                system_prompt=load_prompt('natural', prompt_context)
            )
            
            # Add query IDs to suggestions
            suggestions_with_ids = self._add_query_ids(suggestions)
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=suggestions_with_ids,
                        message_type=MessageType.SUGGESTION
                    )
                ]
            )
            
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error processing natural language query: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )

    def summarize(self, content: Union[pd.DataFrame, str], chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of the given content."""
        try:
            if isinstance(content, pd.DataFrame):
                # Get statistics about the DataFrame
                stats = self._format_results_stats(content)
                
                # Get LLM interpretation
                prompt_context = {
                    'query_details': {'type': 'dataframe_summary'},
                    'results_stats': stats,
                    'chat_history': self._format_chat_history(chat_history)
                }
                
                response = self._call_llm(
                    messages=[{'role': 'user', 'content': 'Summarize this data'}],
                    system_prompt=load_prompt('results', prompt_context)
                )
                
                # Add query IDs to any suggestions
                return self._add_query_ids(response)
            else:
                # For text content, use a simpler prompt
                response = self._call_llm(
                    messages=[
                        {
                            'role': 'system',
                            'content': 'Summarize the following soil science content, focusing on key findings and implications. If suggesting queries, format them in ```monet``` code blocks.'
                        },
                        {
                            'role': 'user',
                            'content': str(content)
                        }
                    ]
                )
                
                # Add query IDs to any suggestions
                return self._add_query_ids(response)
                
        except Exception as e:
            return str(e) 