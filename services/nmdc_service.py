"""
NMDC API service implementation.

This service handles NMDC API query execution and data interactions in the ChatDash application.
It provides a modular interface for:
1. API query detection and validation
2. Safe query execution
3. Result formatting and storage
4. API state management
"""

from typing import Dict, Any, Optional, List, Tuple
import re
from datetime import datetime
import pandas as pd
import requests
import json
from enum import Enum, auto
import numpy as np
from pathlib import Path
import traceback
from urllib.parse import urljoin
from ydata_profiling import ProfileReport

from .base import (
    ChatService, 
    ServiceResponse, 
    ServiceMessage, 
    PreviewIdentifier,
    MessageType
)
from .llm_service import LLMServiceMixin

class RequestType(Enum):
    API_QUERY = auto()
    QUERY_SEARCH = auto()
    NMDC_INFO = auto()
    SERVICE_TEST = auto()
    NATURAL_QUERY = auto()
    QUERY_EXPLAIN = auto()
    DATASET_CONVERSION = auto()

class NMDCService(ChatService, LLMServiceMixin):
    """Service for NMDC API query handling and execution."""
    
    BASE_URL = "https://api.microbiomedata.org"
    
    def __init__(self):
        ChatService.__init__(self, "nmdc")
        LLMServiceMixin.__init__(self, "nmdc")
        
        # Register our prefix for query IDs - using unique prefix for NMDC
        try:
            PreviewIdentifier.register_prefix("nmdc_query")
        except ValueError:
            # Prefix already registered, which is fine
            pass
        
        # NMDC query block pattern - match both ```nmdc and ``` blocks
        self.query_block_pattern = r'^```nmdc\s+(.*?)```'
        
        # Query execution patterns
        self.execution_patterns = [
            # Direct execution commands
            r'^nmdc\.search\s+(?:nmdc_query_)?\d{8}_\d{6}(?:_orig|_alt\d+)\b',  # Handle search for query IDs
            r'^nmdc\.query\s+(?:nmdc_query_)?\d{8}_\d{6}(?:_orig|_alt\d+)\b',   # Handle query for query IDs
            r'^nmdc\.(?:search|query)\.?$',  # Simple execution commands with optional period
            
            # NMDC info request
            r'tell\s+me\s+about\s+nmdc\b',
            
            # Dataset conversion
            r'^convert\s+nmdc_query_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b',
            
            # Self-test command
            r'^test\s+nmdc\s+service\b'
        ]
        
        # Compile patterns for efficiency
        self.query_block_re = re.compile(self.query_block_pattern, re.IGNORECASE | re.DOTALL)
        self.execution_res = [re.compile(p, re.IGNORECASE) for p in self.execution_patterns]
        
        # Cache for API schema
        self._api_schema = None
        
        # Common endpoints
        self.common_endpoints = {
            'biosample': '/biosamples',
            'study': '/studies',
            'data_object': '/data_objects'
        }
        
    def can_handle(self, message: str) -> bool:
        """Detect if message contains NMDC API queries or commands.
        
        Handles:
        1. NMDC query blocks (```nmdc {...}```)
        2. Query search commands (search nmdc_query_20240101_123456)
        3. NMDC info requests (tell me about nmdc)
        4. Self-test command (test nmdc service)
        5. Natural language NMDC commands (nmdc: ...)
        6. Query explanation requests (explain nmdc_query_ID)
        """
        # Clean and normalize message
        message = message.strip()
        message_lower = message.lower()
        
        # Check for NMDC query blocks
        if self.query_block_re.search(message):
            return True
            
        # Check for execution commands and test command
        for pattern in self.execution_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return True
                
        # Check for nmdc: command
        if message_lower.startswith("nmdc:"):
            return True
        
        # Check for explain command with valid query ID
        if message_lower.startswith("explain "):
            # Get query IDs (handling potential comma-separated list)
            remainder = message_lower[8:].strip()
            if not remainder:
                return False
            
            query_ids = [id.strip() for id in remainder.split(',')]
            return all(
                re.match(r'^nmdc_query_\d{8}_\d{6}(?:_orig|_alt\d+)$', query_id)
                for query_id in query_ids
            )
        
        return False
    
    def parse_request(self, message: str) -> Tuple[RequestType, Dict[str, Any]]:
        """Parse message to determine request type and extract relevant parameters.
        
        Returns:
            Tuple[RequestType, Dict[str, Any]]: Request type and parameters dict
        """
        message = message.strip()
        message_lower = message.lower()
        
        # Check for NMDC query blocks
        if match := self.query_block_re.search(message):
            try:
                query = json.loads(match.group(1).strip())
                return RequestType.API_QUERY, {"query": query}
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid NMDC query JSON: {str(e)}")
        
        # Check for simple query search
        if message_lower.rstrip('.') in ['nmdc.search', 'nmdc.query']:
            return RequestType.QUERY_SEARCH, {"query_id": None}  # Will use most recent query
        
        # Check for specific query search
        if match := re.match(r'^nmdc\.(?:search|query)\s+(?:nmdc_query_)?(\d{8}_\d{6}(?:_orig|_alt\d+))(?:\s+|$)', message_lower):
            query_id = match.group(1)
            if not query_id.startswith('nmdc_query_'):
                query_id = f"nmdc_query_{query_id}"
            return RequestType.QUERY_SEARCH, {"query_id": query_id}
        
        # Check for NMDC info request
        if message_lower == 'tell me about nmdc':
            return RequestType.NMDC_INFO, {}
        
        # Check for test command
        if message_lower == "test nmdc service":
            return RequestType.SERVICE_TEST, {}
        
        # Check for nmdc: command
        if message_lower.startswith("nmdc:"):
            return RequestType.NATURAL_QUERY, {"query": message[5:].strip()}
        
        # Check for dataset conversion
        if match := re.match(r'^convert\s+(nmdc_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message_lower):
            return RequestType.DATASET_CONVERSION, {"query_id": match.group(1)}
        
        # Check for explain command
        if message_lower.startswith("explain "):
            remainder = message_lower[8:].strip()
            query_ids = [id.strip() for id in remainder.split(',')]
            if all(re.match(r'^nmdc_query_\d{8}_\d{6}(?:_orig|_alt\d+)$', query_id) for query_id in query_ids):
                return RequestType.QUERY_EXPLAIN, {"query_ids": query_ids}
            
        raise ValueError(f"Unable to parse request from message: {message}")
    
    def _fetch_api_schema(self) -> Dict[str, Any]:
        """Fetch and cache the NMDC API schema.
        
        Returns:
            Dict containing the API schema information
        """
        if self._api_schema is None:
            try:
                # Fetch OpenAPI schema
                response = requests.get(urljoin(self.BASE_URL, "/openapi.json"))
                response.raise_for_status()
                schema = response.json()
                
                # Process schema into more usable format
                processed_schema = {
                    'endpoints': {},
                    'models': {},
                    'relationships': {}
                }
                
                # Extract endpoints
                for path, methods in schema['paths'].items():
                    for method, details in methods.items():
                        if method.lower() == 'get':
                            processed_schema['endpoints'][path] = {
                                'method': method,
                                'summary': details.get('summary', ''),
                                'description': details.get('description', ''),
                                'parameters': details.get('parameters', []),
                                'responses': details.get('responses', {})
                            }
                
                # Extract models
                for name, schema in schema.get('components', {}).get('schemas', {}).items():
                    processed_schema['models'][name] = {
                        'type': schema.get('type', ''),
                        'properties': schema.get('properties', {}),
                        'required': schema.get('required', [])
                    }
                
                # TODO: Extract relationships between models
                
                self._api_schema = processed_schema
                
            except Exception as e:
                raise Exception(f"Failed to fetch API schema: {str(e)}")
        
        return self._api_schema
    
    def _validate_query(self, query: Dict) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate NMDC API query for correctness.
        
        Args:
            query: Query dictionary with endpoint and parameters
            
        Returns:
            Tuple[bool, str, dict]: (is_valid, error_message, metadata)
        """
        try:
            schema = self._fetch_api_schema()
            
            # Check if endpoint exists
            endpoint = query.get('endpoint')
            if not endpoint:
                return False, "No endpoint specified in query", {}
            
            if endpoint not in schema['endpoints']:
                return False, f"Invalid endpoint: {endpoint}", {}
            
            endpoint_schema = schema['endpoints'][endpoint]
            
            # Validate parameters
            params = query.get('filters', {})
            required_params = [p['name'] for p in endpoint_schema['parameters'] if p.get('required', False)]
            
            for param in required_params:
                if param not in params:
                    return False, f"Missing required parameter: {param}", {}
            
            # Add metadata about the endpoint
            metadata = {
                'endpoint': endpoint,
                'summary': endpoint_schema['summary'],
                'description': endpoint_schema['description'],
                'parameters': endpoint_schema['parameters']
            }
            
            return True, "", metadata
            
        except Exception as e:
            return False, f"Validation error: {str(e)}", {}
    
    def _execute_query(self, query: Dict) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """Execute NMDC API query and return results.
        
        Args:
            query: Query dictionary with endpoint and parameters
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any], str]: (results, metadata, preview)
        """
        try:
            # First validate the query
            is_valid, error_msg, validation_metadata = self._validate_query(query)
            if not is_valid:
                raise Exception(error_msg)
            
            # Execute query
            endpoint = query['endpoint']
            
            # Handle filter format - convert from dict to string if needed
            filter_param = query.get('filter', {})
            if isinstance(filter_param, dict):
                # Convert dict to filter string format
                filter_parts = []
                for key, value in filter_param.items():
                    if isinstance(value, (int, float)):
                        filter_parts.append(f"{key}:{value}")
                    else:
                        # Quote string values that contain spaces
                        if ' ' in str(value):
                            filter_parts.append(f'{key}:"{value}"')
                        else:
                            filter_parts.append(f"{key}:{value}")
                filter_str = ",".join(filter_parts) if filter_parts else ""
            else:
                # If it's already a string, ensure proper format
                filter_str = str(filter_param)
                # Replace any equals signs with colons
                filter_str = filter_str.replace('=', ':')
                # Add quotes around values with spaces if not already quoted
                parts = filter_str.split(',')
                formatted_parts = []
                for part in parts:
                    if ':' in part:
                        field, value = part.split(':', 1)
                        value = value.strip()
                        if ' ' in value and not (value.startswith('"') and value.endswith('"')):
                            value = f'"{value}"'
                        formatted_parts.append(f"{field.strip()}:{value}")
                    else:
                        formatted_parts.append(part)
                filter_str = ",".join(formatted_parts)
            
            per_page = query.get('per_page', 2000)  # Default to 2000 per page
            all_results = []
            start_time = datetime.now()
            
            # Initial request with cursor
            params = {
                'filter': filter_str if filter_str else None,  # Only include if not empty
                'per_page': per_page,
                'cursor': '*'  # Include cursor in initial request
            }
            
            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}
            
            # Make initial API request
            url = urljoin(self.BASE_URL, endpoint)
            print(f"\nDEBUG: Making initial request to {url}")
            print(f"DEBUG: Parameters: {params}")
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                # If initial request with cursor fails, try without cursor
                if 'cursor' in params:
                    print("\nDEBUG: Initial request with cursor failed, retrying without cursor")
                    del params['cursor']
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                else:
                    raise e
            
            # Parse response
            data = response.json()
            if not isinstance(data, dict) or 'results' not in data:
                raise ValueError("Unexpected response format - missing 'results' key")
            
            results = data['results']
            all_results.extend(results)
            
            # Get next cursor from meta
            cursor = data.get('meta', {}).get('next_cursor')
            
            # Continue with pagination if there are more results
            while cursor:
                # Add cursor parameter for subsequent requests
                params['cursor'] = cursor
                
                print(f"\nDEBUG: Making paginated request to {url}")
                print(f"DEBUG: Parameters: {params}")
                
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                if not isinstance(data, dict) or 'results' not in data:
                    raise ValueError("Unexpected response format - missing 'results' key")
                
                results = data['results']
                all_results.extend(results)
                
                # Get next cursor
                cursor = data.get('meta', {}).get('next_cursor')
            
            # Convert to DataFrame
            df = pd.json_normalize(all_results)

            # Process DataFrame for consistency
            try:
                # Define common missing value indicators
                missing_values = [
                    '-', 'NA', 'na', 'N/A', 'n/a',
                    'NaN', 'nan', 'NAN',
                    'None', 'none', 'NONE',
                    'NULL', 'null', 'Null',
                    'ND', 'nd', 'N/D', 'n/d',
                    '', ' '  # Empty strings and spaces
                ]
                
                # Track transformations for metadata
                transformations = []
                
                # Replace missing values
                df = df.replace(missing_values, np.nan)
                if df.isna().any().any():
                    transformations.append("Standardized missing values")
                
                # Attempt numeric conversion for string columns
                numeric_conversions = []
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        # Check if column contains only numeric values (allowing NaN)
                        non_nan = df[col].dropna()
                        if len(non_nan) > 0 and non_nan.astype(str).str.match(r'^-?\d*\.?\d+$').all():
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            numeric_conversions.append(col)
                    except Exception:
                        continue
                
                if numeric_conversions:
                    transformations.append(f"Converted columns to numeric: {', '.join(numeric_conversions)}")
                
                # Clean column names
                old_columns = list(df.columns)
                df.columns = df.columns.str.replace(r'[.\[\]{}]', '_', regex=True)
                if any(old != new for old, new in zip(old_columns, df.columns)):
                    transformations.append("Cleaned column names")
                
            except Exception as e:
                print(f"Warning: Error during DataFrame processing: {str(e)}")
                transformations.append(f"Note: Some data cleaning failed: {str(e)}")
            
            # Format preview
            preview_df = self._create_preview_df(df, max_rows=5, max_cols=8)
            preview = "\n\n```\n" + preview_df.to_string() + "\n```\n\n"
            if len(df.columns) > 8:
                preview += f"(Showing 8 of {len(df.columns)} columns)\n"
            
            # Add response metadata
            metadata = {
                'rows': len(df),
                'columns': list(df.columns),
                'response_time': (datetime.now() - start_time).total_seconds(),
                'transformations': transformations,
                **validation_metadata
            }
            
            return df, metadata, preview
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    def execute(self, request: Tuple[RequestType, Dict[str, Any]], context: dict) -> ServiceResponse:
        """Execute NMDC service request.
        
        Args:
            request: Tuple of (RequestType, params) from parse_request
            context: Execution context
            
        Returns:
            ServiceResponse with messages and updates
        """
        try:
            # Store context for use in _call_llm
            self.context = context
            request_type, request_params = request
            
            if request_type == RequestType.API_QUERY:
                # Execute API query
                query = request_params.get('query')
                if not query:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No query provided.",
                            message_type=MessageType.ERROR
                        )]
                    )
                
                # Execute query
                try:
                    results, metadata, preview = self._execute_query(query)
                    
                    # Generate query ID
                    query_id = PreviewIdentifier.create_id(prefix="nmdc_query")
                    
                    # Store successful query
                    store_updates = {
                        'successful_queries_store': {
                            query_id: {
                                'query': query,
                                'metadata': metadata,
                                'execution_time': datetime.now().isoformat(),
                                'dataframe': results.to_dict('records')
                            }
                        }
                    }
                    
                    # Format response
                    response = f"""Query executed successfully!

Results preview:

Query ID: {query_id}
{preview}

Total rows: {metadata['rows']}

Endpoint: {metadata['endpoint']}
Response time: {metadata['response_time']:.2f}s

To convert your result to a dataset you can use 'convert {query_id} to dataset'"""
                    
                    # Create messages list with preview
                    messages = [
                        ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )
                    ]
                    
                    # Generate LLM summary if we have results
                    if not results.empty:
                        try:
                            llm_summary = self.summarize(results, context['chat_history'], context)
                            if llm_summary:
                                messages.append(
                                    ServiceMessage(
                                        service=self.name,
                                        content=f"\n### Analysis Summary\n\n{llm_summary}",
                                        message_type=MessageType.SUMMARY
                                    )
                                )
                        except Exception as e:
                            print(f"Error generating LLM summary: {str(e)}")
                            # Continue without summary
                    
                    return ServiceResponse(
                        messages=messages,
                        store_updates=store_updates,
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error executing query: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.QUERY_SEARCH:
                # Find query to execute
                query_text, query_id = self.find_recent_query(
                    context['chat_history'],
                    request_params.get('query_id')
                )
                
                if not query_text:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No NMDC query found in recent chat history. Please make sure a query has been suggested before using 'nmdc.search' or 'nmdc.query'.",
                            message_type=MessageType.ERROR
                        )]
                    )
                
                # Parse and execute found query
                try:
                    query = json.loads(query_text)
                    results, metadata, preview = self._execute_query(query)
                    
                    # Store successful query
                    store_updates = {
                        'successful_queries_store': {
                            query_id: {
                                'query': query,
                                'metadata': metadata,
                                'execution_time': datetime.now().isoformat(),
                                'dataframe': results.to_dict('records')
                            }
                        }
                    }
                    
                    # Format response
                    response = f"""Query executed successfully!

Results preview:

Query ID: {query_id}
{preview}

Total rows: {metadata['rows']}

Endpoint: {metadata['endpoint']}
Response time: {metadata['response_time']:.2f}s

To convert your result to a dataset you can use 'convert {query_id} to dataset'"""
                    
                    messages = [
                        ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )
                    ]
                    
                    # Generate LLM summary if we have results
                    if not results.empty:
                        try:
                            llm_summary = self.summarize(results, context['chat_history'], context)
                            if llm_summary:
                                messages.append(
                                    ServiceMessage(
                                        service=self.name,
                                        content=f"\n### Analysis Summary\n\n{llm_summary}",
                                        message_type=MessageType.SUMMARY
                                    )
                                )
                        except Exception as e:
                            print(f"Error generating LLM summary: {str(e)}")
                            # Continue without summary
                    
                    return ServiceResponse(
                        messages=messages,
                        store_updates=store_updates,
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error executing query: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.NMDC_INFO:
                # Get API schema
                try:
                    schema = self._fetch_api_schema()
                    
                    # Format API overview
                    overview = ["### NMDC API Overview"]
                    
                    # Add endpoint summaries
                    overview.append("\n#### Available Endpoints")
                    for path, info in schema['endpoints'].items():
                        overview.append(f"\n**{path}**")
                        if info['summary']:
                            overview.append(f"- Summary: {info['summary']}")
                        if info['description']:
                            overview.append(f"- Description: {info['description']}")
                        if info['parameters']:
                            overview.append("- Parameters:")
                            for param in info['parameters']:
                                required = " (Required)" if param.get('required') else ""
                                overview.append(f"  - {param['name']}: {param.get('description', '')}{required}")
                    
                    # Add model information
                    overview.append("\n#### Data Models")
                    for name, info in schema['models'].items():
                        overview.append(f"\n**{name}**")
                        if info['properties']:
                            overview.append("Properties:")
                            for prop_name, prop_info in info['properties'].items():
                                required = " (Required)" if prop_name in info.get('required', []) else ""
                                overview.append(f"- {prop_name}: {prop_info.get('type', '')}{required}")
                    
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="\n".join(overview),
                            message_type=MessageType.INFO
                        )],
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error fetching API information: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.SERVICE_TEST:
                return self._run_self_test(context)
            
            elif request_type == RequestType.NATURAL_QUERY:
                # Get query text
                query_text = request_params.get('query')
                if not query_text:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No query text provided.",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Get API schema
                try:
                    schema = self._fetch_api_schema()
                    
                    # Generate prompt
                    prompt = self._create_natural_query_prompt(query_text, schema, context)
                    
                    # Get LLM response
                    response = self._call_llm([{"role": "user", "content": prompt}])
                    
                    # Process response to ensure proper query ID formatting
                    response = self.add_ids_to_blocks(response)
                    
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )],
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error generating API queries: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.QUERY_EXPLAIN:
                # Get query IDs to explain
                query_ids = request_params.get('query_ids', [])
                focus = request_params.get('focus', '')
                
                if not query_ids:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content="No query IDs provided to explain.",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
                
                # Get API schema
                try:
                    schema = self._fetch_api_schema()
                    
                    # Get query details
                    queries = []
                    missing_queries = []
                    for query_id in query_ids:
                        query_details = self._get_query_details(query_id, context)
                        if query_details:
                            queries.append(query_details)
                        else:
                            missing_queries.append(query_id)
                    
                    if missing_queries:
                        return ServiceResponse(
                            messages=[ServiceMessage(
                                service=self.name,
                                content=f"Could not find the following queries: {', '.join(missing_queries)}",
                                message_type=MessageType.ERROR
                            )],
                            state_updates={'chat_input': ''}
                        )
                    
                    # Generate prompt
                    prompt = self._create_explain_prompt(queries, schema, focus)
                    
                    # Get LLM analysis
                    analysis = self._call_llm([{"role": "user", "content": prompt}])
                    
                    # Process response to ensure proper query ID formatting
                    analysis = self.add_ids_to_blocks(analysis)
                    
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=analysis,
                            message_type=MessageType.RESULT
                        )],
                        state_updates={'chat_input': ''}
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[ServiceMessage(
                            service=self.name,
                            content=f"Error generating query explanation: {str(e)}",
                            message_type=MessageType.ERROR
                        )],
                        state_updates={'chat_input': ''}
                    )
            
            elif request_type == RequestType.DATASET_CONVERSION:
                return self._handle_dataset_conversion(request_params, context)
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Invalid request type",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}  # Clear input even on error
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Service error: {str(e)}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}  # Clear input on error
            )
    
    def find_recent_query(self, chat_history: list, query_id: str = None) -> tuple[str, str]:
        """Find NMDC query in chat history.
        
        Looks for NMDC query blocks in both assistant messages and service messages.
        """
        print(f"\nSearching for query in chat history...")
        print(f"Target query ID: {query_id}")
        
        for msg in reversed(chat_history):
            # Check both assistant messages and NMDC service messages
            if ('```' in msg['content'].lower() and 
                (msg['role'] == 'assistant' or 
                 (msg.get('service') == self.name and msg['role'] == 'system'))):
                content = msg['content']
                print(f"\nFound NMDC block in message:")
                print(f"Message role: {msg['role']}")
                print(f"Message service: {msg.get('service')}")
                
                # Extract all NMDC blocks with IDs
                query_blocks = []
                for match in re.finditer(r'```nmdc\s*(.*?)```', content, re.DOTALL):
                    block = match.group(1).strip()
                    
                    # First try to find and extract the query ID
                    id_match = re.search(r'--\s*Query ID:\s*(nmdc_query_\d{8}_\d{6}(?:_orig|_alt\d+))\b', block)
                    if id_match:
                        found_id = id_match.group(1)
                        # Remove ID comment from query by splitting on -- and taking first part
                        query_parts = block.split('--')
                        query = query_parts[0].strip()
                        
                        try:
                            # Validate it's proper JSON
                            query_json = json.loads(query)
                            if isinstance(query_json, dict) and 'endpoint' in query_json:
                                print(f"\nFound query with ID {found_id}:")
                                print(f"Query text to execute:\n{query}")
                                query_blocks.append((query, found_id))
                        except json.JSONDecodeError:
                            print(f"Invalid JSON in block with ID {found_id}")
                            continue
                
                if query_blocks:
                    if query_id:
                        # Find specific query
                        for query, found_id in query_blocks:
                            if found_id == query_id:
                                print(f"\nFound requested query: {query_id}")
                                print(f"Query text to execute:\n{query}")
                                return query, found_id
                    else:
                        # Find most recent original query
                        for query, found_id in query_blocks:
                            if found_id.endswith('_orig'):
                                print(f"\nFound most recent original query: {found_id}")
                                print(f"Query text to execute:\n{query}")
                                return query, found_id
        
        print("No matching query found")
        return None, None
    
    def _run_self_test(self, context: Dict[str, Any]) -> ServiceResponse:
        """Run self-tests for NMDC service functionality."""
        test_results = []
        passed = 0
        total = 0
        
        def run_test(name: str, test_fn) -> bool:
            nonlocal passed, total
            total += 1
            try:
                test_fn()
                test_results.append(f"✓ {name}")
                passed += 1
                return True
            except Exception as e:
                test_results.append(f"✗ {name}: {str(e)}")
                return False
        
        # 1. Test query parsing
        def test_query_parsing():
            message = """```nmdc
{
    "endpoint": "/biosample/search",
    "filters": {"field": "value"}
}
```"""
            request_type, params = self.parse_request(message)
            assert request_type == RequestType.API_QUERY
            assert isinstance(params['query'], dict)
            assert params['query']['endpoint'] == "/biosample/search"
        
        run_test("Query parsing", test_query_parsing)
        
        # 2. Test command parsing
        def test_command_parsing():
            message = "test nmdc service"
            request_type, params = self.parse_request(message)
            assert request_type == RequestType.SERVICE_TEST
            assert params == {}
        
        run_test("Command parsing", test_command_parsing)
        
        # 3. Test API schema fetching
        def test_schema_fetch():
            # Set up mock schema for testing
            self._api_schema = {
                'endpoints': {
                    '/biosample/search': {
                        'method': 'get',
                        'summary': 'Search biosamples',
                        'description': 'Search for biosamples in NMDC',
                        'parameters': []
                    }
                },
                'models': {
                    'Biosample': {
                        'type': 'object',
                        'properties': {
                            'id': {'type': 'string'},
                            'name': {'type': 'string'}
                        }
                    }
                }
            }
            schema = self._fetch_api_schema()
            assert isinstance(schema, dict)
            assert 'endpoints' in schema
            assert 'models' in schema
        
        run_test("API schema fetching", test_schema_fetch)
        
        # 4. Test query validation
        def test_query_validation():
            # Ensure schema is set up
            if not self._api_schema:
                test_schema_fetch()
                
            query = {
                "endpoint": "/biosample/search",
                "filters": {}
            }
            is_valid, error, metadata = self._validate_query(query)
            assert is_valid, f"Query validation failed: {error}"
            assert not error, f"Unexpected error: {error}"
            assert metadata['endpoint'] == "/biosample/search"
        
        run_test("Query validation", test_query_validation)
        
        # Format results
        summary = [
            "### NMDC Service Self-Test Results",
            f"\nPassed: {passed}/{total} tests\n",
            "Detailed Results:"
        ] + test_results
        
        return ServiceResponse(
            messages=[ServiceMessage(
                service=self.name,
                content="\n".join(summary),
                message_type=MessageType.INFO
            )]
        )
    
    def _get_query_details(self, query_id: str, context: Dict[str, Any]) -> Optional[Dict]:
        """Get detailed information about a query."""
        # Check successful queries store first
        queries_store = context.get('successful_queries_store', {})
        stored = queries_store.get(query_id)
        
        if stored:
            return {
                'id': query_id,
                'query': stored['query'],
                'metadata': stored['metadata'],
                'execution_time': stored['execution_time']
            }
        
        # Fall back to chat history
        query_text, found_id = self.find_recent_query(context.get('chat_history', []), query_id)
        if query_text:
            return {
                'id': query_id,
                'query': json.loads(query_text),
                'metadata': {
                    'note': 'This query has not been executed yet.'
                }
            }
        
        return None
    
    def _handle_dataset_conversion(self, params: dict, context: dict) -> ServiceResponse:
        """Handle conversion of query results to dataset."""
        query_id = params.get('query_id')
        if not query_id:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No query ID provided for conversion.",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )
        
        # Get stored execution
        executions = context.get('successful_queries_store', {})
        stored = executions.get(query_id)
        
        if not stored:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"No execution found for query ID: {query_id}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )
        
        try:
            # Re-execute query to get fresh results
            results, metadata, _ = self._execute_query(stored['query'])
            
            # Process DataFrame for consistency
            try:
                # Define common missing value indicators
                missing_values = [
                    '-', 'NA', 'na', 'N/A', 'n/a',
                    'NaN', 'nan', 'NAN',
                    'None', 'none', 'NONE',
                    'NULL', 'null', 'Null',
                    'ND', 'nd', 'N/D', 'n/d',
                    '', ' '
                ]
                
                # Track transformations
                transformations = []
                
                # Format complex columns first
                for col in results.columns:
                    try:
                        # Check if column contains complex data (lists, dicts, etc.)
                        if results[col].apply(lambda x: isinstance(x, (list, dict, tuple, pd.Series, np.ndarray))).any():
                            results[col] = results[col].apply(lambda x: self._format_preview_value(x, max_length=100))
                            transformations.append(f"Formatted complex data in column: {col}")
                    except Exception as e:
                        print(f"Warning: Error formatting column {col}: {str(e)}")
                        continue
                
                # Replace missing values
                results = results.replace(missing_values, np.nan)
                if results.isna().any().any():
                    transformations.append("Standardized missing values")
                
                # Attempt numeric conversion for string columns
                numeric_conversions = []
                for col in results.select_dtypes(include=['object']).columns:
                    try:
                        non_nan = results[col].dropna()
                        if len(non_nan) > 0 and non_nan.astype(str).str.match(r'^-?\d*\.?\d+$').all():
                            results[col] = pd.to_numeric(results[col], errors='coerce')
                            numeric_conversions.append(col)
                    except Exception:
                        continue
                
                if numeric_conversions:
                    transformations.append(f"Converted columns to numeric: {', '.join(numeric_conversions)}")
                
                # Clean column names
                old_columns = list(results.columns)
                results.columns = results.columns.str.replace(r'[.\[\]{}]', '_', regex=True)
                if any(old != new for old, new in zip(old_columns, results.columns)):
                    transformations.append("Cleaned column names")
                
            except Exception as e:
                print(f"Warning: Error during DataFrame processing: {str(e)}")
                transformations.append(f"Note: Some data cleaning failed: {str(e)}")
            
            # Get datasets store
            datasets = context.get('datasets_store', {})
            
            # Generate profile report
            try:
                profile = ProfileReport(
                    results,
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
                'df': results.to_dict('records'),
                'metadata': {
                    'source': f"NMDC API Query: {query_id}",
                    'query': stored['query'],
                    'endpoint': metadata['endpoint'],
                    'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(results),
                    'columns': list(results.columns),
                    'transformations': transformations
                },
                'profile_report': profile_html
            }
            
            # Format transformation message
            transform_msg = "\n- " + "\n- ".join(transformations) if transformations else ""
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"""✓ Query results converted to dataset '{query_id}'

- Rows: {len(results)}
- Columns: {', '.join(results.columns)}
- Source: NMDC API Query {query_id}
- Endpoint: {metadata['endpoint']}
Data Transformations:{transform_msg}""",
                    message_type=MessageType.INFO
                )],
                store_updates={
                    'datasets_store': datasets,
                    'successful_queries_store': {
                        k: v for k, v in executions.items() if k != query_id
                    }
                },
                state_updates={'chat_input': ''}
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"NMDC service error: ❌ Query result conversion failed: {str(e)}",
                    message_type=MessageType.ERROR
                )],
                state_updates={'chat_input': ''}
            )
    
    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process natural language NMDC queries.
        
        Args:
            message: The message to process
            chat_history: List of previous chat messages
            
        Returns:
            str: Processed response
        """
        try:
            # First, look for any referenced query IDs in the message
            query_pattern = r'nmdc_query_\d{8}_\d{6}(?:_orig|_alt\d+)'
            referenced_queries = re.finditer(query_pattern, message)
            
            # Build context about referenced queries
            query_context = []
            for match in referenced_queries:
                query_id = match.group(0)
                # Find the query in chat history
                query_text, _ = self.find_recent_query(chat_history, query_id)
                if query_text:
                    try:
                        # Parse query to ensure it's valid and to format it nicely
                        query_json = json.loads(query_text)
                        query_context.append(f"\nReferenced Query {query_id}:\n{json.dumps(query_json, indent=2)}")
                    except json.JSONDecodeError:
                        continue
            
            # Get API schema
            schema = self._fetch_api_schema()
            
            # Create context for query generation
            context = {'chat_history': chat_history}
            
            # Generate prompt, including referenced query information if any
            prompt = self._create_natural_query_prompt(
                message,
                schema,
                context,
                query_context="\n".join(query_context) if query_context else None
            )
            
            # Get LLM response
            response = self._call_llm([{"role": "user", "content": prompt}])
            
            # Process response to ensure proper query ID formatting
            response = self.add_ids_to_blocks(response)
            
            return response
            
        except Exception as e:
            return f"Error processing message: {str(e)}"

    def _create_natural_query_prompt(self, query_text: str, schema: Dict, context: Dict, query_context: Optional[str] = None) -> str:
        """Create specialized prompt for natural language to NMDC API query translation."""
        # Format API structure
        api_structure = []
        
        # Add common endpoints first
        api_structure.append("\n### Common Search Endpoints")
        for name, endpoint in self.common_endpoints.items():
            if endpoint_info := schema['endpoints'].get(endpoint):
                api_structure.append(f"\n**{name.title()} Search** (`{endpoint}`)")
                if endpoint_info['summary']:
                    api_structure.append(f"Summary: {endpoint_info['summary']}")
                if endpoint_info['parameters']:
                    api_structure.append("Parameters:")
                    for param in endpoint_info['parameters']:
                        required = " (Required)" if param.get('required') else ""
                        api_structure.append(f"- {param['name']}: {param.get('description', '')}{required}")

        prompt_template = """You are an NMDC API query generator. Your task is to translate natural language questions into NMDC API queries.

API Structure:
{structure}

User's Question: {query}
{referenced_queries}

NMDC Query Guidelines:
1. Query Format - ALWAYS use this exact format with ```nmdc code blocks:
   ```nmdc
   {{
       "endpoint": "/biosamples",  # Use exact endpoint paths
       "filter": "env_broad_scale=ocean and latitude>45",  # Use filter string format
       "per_page": 10,            # Use "per_page" not "limit"
       "cursor": "*"              # Use cursor-based pagination
   }}
   ```

2. Required Endpoint Formats - ALWAYS use these exact paths:
   - Biosamples: MUST use "/biosamples" (not "/biosample/search")
   - Studies: MUST use "/studies" (not "/study/search")
   - Data Objects: MUST use "/data_objects" (not "/data_object/search")

3. Common Filter Examples:
   - Environmental: "env_broad_scale=ocean"
   - Geographic: "latitude>45 and longitude<-122"
   - Temporal: "collection_date>2020-01-01"
   - Multiple conditions: "env_broad_scale=ocean and depth>1000"

4. Query Best Practices:
   - Use filter string format with field=value pairs
   - Join multiple conditions with "and"
   - Use comparison operators: =, >, <, >=, <=
   - Use "per_page" for pagination size
   - Use cursor-based pagination with "cursor"
   - Consider relationships between biosamples, studies, and data objects
   - Add comments explaining complex filter logic
   - ALWAYS use ```nmdc code blocks for queries
   - ALWAYS use proper JSON format with double quotes
   - ALWAYS use the exact endpoint paths shown above

Your response should:
1. Analyze the natural language question to understand the user's intent
2. If there are referenced queries, consider how they relate to the current request
3. Choose the most appropriate endpoint from the REQUIRED formats above
4. Generate appropriate query/queries using EXACTLY the format shown above
5. Explain your approach and any assumptions made
6. Suggest alternative queries if there are different ways to answer the question

Remember: 
- ALWAYS wrap queries in ```nmdc code blocks
- ALWAYS use proper JSON format with double quotes
- ALWAYS use the exact endpoint paths listed above
- NEVER use variations like "/biosample/search" or "/study/search"
- Users can execute queries using:
  - 'nmdc.search' or 'nmdc.query' to run the primary query
  - 'nmdc.search ID' to run a specific query by its ID"""

        return prompt_template.format(
            structure="\n".join(api_structure),
            query=query_text,
            referenced_queries=f"\nReferenced Queries:\n{query_context}" if query_context else ""
        )

    def get_help_text(self) -> str:
        """Get help text for NMDC service commands."""
        return """
🔍 **NMDC API Operations**
The National Microbiome Data Collaborative (NMDC) service provides access to standardized microbiome data, setting the standard in biological data integration and accessibility.

Commands:
- View API info: `tell me about nmdc`
- Execute queries:
  - Direct query: ```nmdc {...}```
  - Last query: `nmdc.search` or `nmdc.query`
  - Specific query: `nmdc.search ID` or `nmdc.query ID`
- Natural language: `nmdc: [your question]`
- Analyze queries: `explain nmdc_query_[ID]` or multiple: `explain ID1, ID2`
- Convert to dataset: `convert nmdc_query_[ID] to dataset`
- Test service: `test nmdc service`
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for NMDC capabilities."""
        return """
NMDC Service Overview:
The National Microbiome Data Collaborative (NMDC) service provides standardized access to microbiome data and metadata. This service enables:
- Standardized microbiome data discovery
- Cross-study data integration
- Consistent metadata representation
- Biological data accessibility and reuse

Commands:
1. Query Execution:
   ```nmdc
   {
       "endpoint": "/biosample/search",
       "filters": {"field": "value"}
   }
   ```
   - Returns DataFrame results
   - Common endpoints:
     * /biosamples - Search biological samples and their metadata
     * /studies - Search research studies and protocols
     * /data_objects - Search associated data files and resources

2. Query Management:
   "nmdc.search" or "nmdc.query" - run last query
   "nmdc.search ID" or "nmdc.query ID" - run specific query
   "explain nmdc_query_[ID]" - analyze query structure and results
   "explain ID1, ID2" - compare multiple queries
   "nmdc: [question]" - natural language to query
   "test nmdc service" - run service diagnostics

3. Result Conversion:
   "convert nmdc_query_[ID] to dataset"
   - Saves query results as dataset
   - Preserves query and execution info
   - Includes data cleaning and profiling
   - Maintains standardized metadata"""

    def summarize(self, df: pd.DataFrame, chat_history: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Generate comprehensive summary of NMDC API query results.
        
        Args:
            df: DataFrame containing query results
            chat_history: List of previous chat messages
            context: Execution context with:
                - successful_queries_store: Query history with metadata
                
        Returns:
            str: Summary of the results
        """
        try:
            # 1. Get query information from successful_queries_store
            queries_store = context.get('successful_queries_store', {})
            current_query = None
            query_metadata = None
            
            # Find the most recent query that matches our results
            df_columns = set(df.columns)
            for query_id, stored in queries_store.items():
                stored_df = pd.DataFrame(stored['dataframe'])
                if (len(stored_df) == len(df) and 
                    set(stored_df.columns) == df_columns):  # Compare column sets instead of lists
                    current_query = stored['query']
                    query_metadata = stored['metadata']
                    break
            
            # 2. Create base system prompt
            system_prompt = f"""You are an NMDC data analyst. Analyze these query results in the context of:

1. Query Information:
Endpoint: {query_metadata['endpoint'] if query_metadata else 'Unknown'}
{json.dumps(current_query, indent=2) if current_query else ''}

2. Result Statistics:
- Total rows: {len(df)}
- Columns: {', '.join(df.columns)}

Provide analysis focusing on:
1. Data Content:
   - Key patterns and relationships
   - Statistical insights
   - Data quality observations
   - Notable values or trends

2. Domain-Specific Analysis:
   - Environmental/geological context
   - Taxonomic patterns
   - Geographic distribution
   - Temporal patterns
   - Methodological insights

3. Recommendations:
   - Additional analyses to consider
   - Related data to explore
   - Quality considerations
   - Potential biases or limitations

Focus on providing concrete insights based on the actual data."""

            # 3. Package results for analysis
            result_content = self._package_results_for_analysis(df)
            
            # 4. Get LLM response
            response = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": result_content}
            ])
            
            return response.strip()
            
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error generating summary: {str(e)}"

    def _package_results_for_analysis(self, df: pd.DataFrame) -> str:
        """Package query results for analysis.
        
        Args:
            df: DataFrame containing query results
            
        Returns:
            str: JSON-formatted string containing packaged results
        """
        try:
            # Start with basic result information
            content = {
                'summary': {
                    'total_rows': len(df),
                    'columns': list(df.columns),
                    'data_types': {str(col): str(dtype) for col, dtype in df.dtypes.items()}
                }
            }
            
            # Add numeric column statistics
            content['numeric_analysis'] = {}
            for col in df.select_dtypes(include=['number']).columns:
                stats = df[col].describe()
                content['numeric_analysis'][str(col)] = {
                    'mean': float(stats['mean']) if not pd.isna(stats['mean']) else None,
                    'std': float(stats['std']) if not pd.isna(stats['std']) else None,
                    'min': float(stats['min']) if not pd.isna(stats['min']) else None,
                    'max': float(stats['max']) if not pd.isna(stats['max']) else None,
                    'quartiles': {
                        '25%': float(stats['25%']) if not pd.isna(stats['25%']) else None,
                        '50%': float(stats['50%']) if not pd.isna(stats['50%']) else None,
                        '75%': float(stats['75%']) if not pd.isna(stats['75%']) else None
                    }
                }
            
            # Add categorical column analysis
            content['categorical_analysis'] = {}
            for col in df.select_dtypes(exclude=['number']).columns:
                value_counts = df[col].value_counts()
                # Convert values to strings to ensure they're hashable
                value_counts.index = value_counts.index.map(str)
                # Only include top categories if there are many
                if len(value_counts) <= 10:
                    content['categorical_analysis'][str(col)] = {
                        'unique_values': len(value_counts),
                        'top_values': {str(k): int(v) for k, v in value_counts.items()}
                    }
                else:
                    content['categorical_analysis'][str(col)] = {
                        'unique_values': len(value_counts),
                        'top_values': {str(k): int(v) for k, v in value_counts.head(10).items()},
                        'other_categories': len(value_counts) - 10
                    }
            
            # Add null value analysis
            null_counts = df.isnull().sum()
            if null_counts.any():
                content['null_analysis'] = {str(col): int(count) for col, count in null_counts.items()}
            
            # Add sample rows (up to 5)
            if len(df) > 0:
                # Convert sample rows to serializable format
                samples = []
                for _, row in df.head(min(5, len(df))).iterrows():
                    sample = {}
                    for col in df.columns:
                        val = row[col]
                        try:
                            # Handle different value types
                            if isinstance(val, (pd.Series, np.ndarray)):
                                # For array-like values, check if any element is NaN
                                if pd.isna(val).any():
                                    sample[str(col)] = None
                                else:
                                    # Convert array to list if not NaN
                                    sample[str(col)] = val.tolist() if hasattr(val, 'tolist') else list(val)
                            elif isinstance(val, (list, tuple)):
                                # Handle Python sequences
                                sample[str(col)] = list(val)
                            elif isinstance(val, (int, float)):
                                # Handle numeric values
                                sample[str(col)] = None if pd.isna(val) else val
                            else:
                                # Handle other types (strings, etc)
                                sample[str(col)] = None if pd.isna(val) else str(val)
                        except Exception as e:
                            # If any error occurs during conversion, set to None
                            print(f"Warning: Error converting value for column {col}: {str(e)}")
                            sample[str(col)] = None
                    samples.append(sample)
                content['samples'] = samples
            
            return json.dumps(content, indent=2)
            
        except Exception as e:
            print(f"Error packaging results: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return json.dumps({
                'error': str(e),
                'basic_info': {
                    'total_rows': len(df),
                    'columns': list(df.columns)
                }
            })

    def add_ids_to_blocks(self, text: str) -> str:
        """Add query IDs to NMDC query blocks in text.
        
        This method processes LLM responses to ensure each NMDC query block
        has a proper query ID for tracking and execution.
        
        Args:
            text: Text containing NMDC query blocks
            
        Returns:
            str: Text with query IDs added to blocks
        """
        if not text or '```' not in text:
            return text
            
        # Track if we've marked a block as primary
        has_primary = False
        # Track alternative numbers used
        alt_numbers = set()
        
        def replace_block(match) -> str:
            nonlocal has_primary, alt_numbers
            block = match.group(1).strip()
            
            # Skip if block already has an ID
            if '--' in block and 'Query ID:' in block:
                # Extract existing alt number if present
                if '_alt' in block:
                    try:
                        alt_num = int(re.search(r'_alt(\d+)', block).group(1))
                        alt_numbers.add(alt_num)
                    except (AttributeError, ValueError):
                        pass
                return match.group(0)
            
            # Skip if block doesn't look like a JSON query
            if not (block.startswith('{') and block.endswith('}')):
                return match.group(0)
            
            try:
                # Validate it's a proper JSON query
                query = json.loads(block)
                if not isinstance(query, dict) or 'endpoint' not in query:
                    return match.group(0)
                
                # Generate new query ID
                query_id = PreviewIdentifier.create_id(prefix="nmdc_query")
                
                # Add suffix based on whether this is primary or alternative
                if not has_primary:
                    query_id = query_id.replace('_orig', '_orig')  # Ensure primary query
                    has_primary = True
                else:
                    # Find the next available alternative number
                    alt_num = 1
                    while alt_num in alt_numbers:
                        alt_num += 1
                    alt_numbers.add(alt_num)
                    query_id = query_id.replace('_orig', f'_alt{alt_num}')
                
                # Format block with ID
                return f"```nmdc\n{block}\n\n-- Query ID: {query_id}\n```"
            except json.JSONDecodeError:
                return match.group(0)
        
        # First pass to collect existing alternative numbers
        for match in re.finditer(r'_alt(\d+)', text):
            try:
                alt_numbers.add(int(match.group(1)))
            except ValueError:
                continue
        
        # Replace all query blocks
        processed = re.sub(
            self.query_block_pattern,
            replace_block,
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        
        return processed

    def _format_preview_value(self, value: Any, max_length: int = 50) -> str:
        """Format a value for preview display.
        
        Args:
            value: The value to format
            max_length: Maximum length for string values
            
        Returns:
            str: Formatted value suitable for display
        """
        try:
            print(f"\nDEBUG: Formatting value of type {type(value)}")
            print(f"\nDEBUG: Value content: {value}")
            
            # Handle lists first (before NaN check)
            if isinstance(value, list):
                print("DEBUG: Handling list")
                size = len(value)
                if size == 0:
                    return "[]"
                preview = str(value[:3])  # Only show first 3 elements
                if len(preview) > max_length:
                    preview = preview[:max_length-3] + "..."
                return f"[list({size}): {preview}]"
            
            # Handle NaN/None values
            try:
                print("DEBUG: Checking for NaN")
                if pd.isna(value):
                    print("DEBUG: Value is NaN")
                    return 'NA'
            except Exception as e:
                print(f"DEBUG: Error in NaN check: {str(e)}")
            
            print("DEBUG: Passed NaN check")
            
            # For other collections (tuple, set, dict, array-like)
            if isinstance(value, (tuple, set, dict, pd.Series, np.ndarray)):
                print("DEBUG: Handling other collection")
                try:
                    size = len(value)
                    preview = str(value)[:max_length]
                    if len(preview) < len(str(value)):
                        preview = preview[:max_length-3] + "..."
                    return f"[{type(value).__name__}({size}): {preview}]"
                except Exception as e:
                    print(f"DEBUG: Error in collection handling: {str(e)}")
                    return f"[{type(value).__name__}]"
            
            # Handle numeric values with precision
            if isinstance(value, float):
                print("DEBUG: Handling float")
                return f"{value:.3g}"
            
            # For all other values, convert to string and truncate if needed
            print("DEBUG: Handling as string")
            preview = str(value)
            if len(preview) > max_length:
                return preview[:max_length-3] + "..."
            return preview
            
        except Exception as e:
            print(f"Error in _format_preview_value: {str(e)}")
            print(f"Value type: {type(value)}")
            print(f"Value: {value}")
            return "[Error formatting value]"

    def _create_preview_df(self, df: pd.DataFrame, max_rows: int = 5, max_cols: int = None) -> pd.DataFrame:
        """Create a preview-friendly version of the DataFrame.
        
        Args:
            df: Original DataFrame
            max_rows: Maximum number of rows to show
            max_cols: Maximum number of columns to show (None for all)
            
        Returns:
            pd.DataFrame: Preview-friendly version
        """
        try:
            # Select rows
            preview_df = df.head(max_rows).copy()
            
            # Select columns if needed
            if max_cols and len(preview_df.columns) > max_cols:
                # Prioritize ID and name columns
                priority_cols = [col for col in preview_df.columns if any(key in col.lower() for key in ['id', 'name', 'type'])]
                other_cols = [col for col in preview_df.columns if col not in priority_cols]
                
                # Take all priority columns plus enough other columns to reach max_cols
                remaining_slots = max_cols - len(priority_cols)
                if remaining_slots > 0:
                    selected_cols = priority_cols + other_cols[:remaining_slots]
                else:
                    selected_cols = priority_cols[:max_cols]
                
                preview_df = preview_df[selected_cols]
            
            # Format all values
            formatted_data = {}
            for col in preview_df.columns:
                try:
                    formatted_data[col] = preview_df[col].apply(self._format_preview_value)
                except Exception as e:
                    print(f"Error formatting column {col}: {str(e)}")
                    # Use string representation as fallback
                    formatted_data[col] = preview_df[col].astype(str).apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
            
            return pd.DataFrame(formatted_data, index=preview_df.index)
            
        except Exception as e:
            print(f"Error in _create_preview_df: {str(e)}")
            # Return a simple string representation as fallback
            return df.head(max_rows).astype(str)

    def _create_explain_prompt(self, queries: List[Dict], schema: Dict, focus: str = '') -> str:
        """Create prompt for explaining NMDC queries.
        
        Args:
            queries: List of query details including query, metadata, and execution info
            schema: API schema information
            focus: Optional focus area for the explanation
            
        Returns:
            str: Formatted prompt for LLM
        """
        # Format API structure
        api_structure = []
        
        # Add relevant endpoints
        api_structure.append("\n### Relevant Endpoints")
        for query in queries:
            endpoint = query['query'].get('endpoint')
            if endpoint_info := schema['endpoints'].get(endpoint):
                api_structure.append(f"\n**{endpoint}**")
                if endpoint_info['summary']:
                    api_structure.append(f"Summary: {endpoint_info['summary']}")
                if endpoint_info['parameters']:
                    api_structure.append("Parameters:")
                    for param in endpoint_info['parameters']:
                        required = " (Required)" if param.get('required') else ""
                        api_structure.append(f"- {param['name']}: {param.get('description', '')}{required}")
        
        # Format queries
        query_details = []
        for query in queries:
            details = [
                f"\nQuery ID: {query['id']}",
                "Query:",
                json.dumps(query['query'], indent=2),
                "\nMetadata:",
                json.dumps(query['metadata'], indent=2)
            ]
            query_details.append("\n".join(details))
        
        prompt_template = """You are an NMDC API query analyst. Your task is to explain and analyze NMDC API queries.

API Information:
{structure}

Queries to Analyze:
{queries}

Focus Areas:
1. Query Structure:
   - Endpoint selection and appropriateness
   - Filter conditions and their effects
   - Pagination settings
   
2. Expected Results:
   - Types of data returned
   - Potential result size
   - Key fields in the response
   
3. Query Relationships:
   - How queries relate to each other
   - Progressive refinement patterns
   - Alternative approaches
   
4. Best Practices:
   - Optimization opportunities
   - Potential issues or limitations
   - Suggested improvements

{focus_prompt}

Your response should:
1. Explain each query's purpose and structure
2. Analyze the effectiveness of the approach
3. Suggest potential improvements or alternatives
4. Note any concerns or limitations
5. Explain relationships between queries if multiple are provided"""

        focus_section = ""
        if focus:
            focus_section = f"\nSpecific Focus:\n{focus}"
        
        return prompt_template.format(
            structure="\n".join(api_structure),
            queries="\n\n".join(query_details),
            focus_prompt=focus_section
        ) 