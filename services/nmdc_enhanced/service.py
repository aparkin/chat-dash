"""
NMDC Enhanced Service for ChatDash.

A specialized service that provides advanced data discovery and analysis capabilities
for the National Microbiome Data Collaborative (NMDC) database. This service enables
researchers to explore environmental microbiome data through:

- Natural language querying
- Advanced data integration
- Statistical analysis
- Geographic and environmental context
- Dataset management and conversion

The service is designed to be user-friendly while maintaining data integrity
and providing comprehensive analytical capabilities for microbiome research.
"""

import logging
import json
import asyncio
import re
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import pandas as pd
from datetime import datetime
from copy import deepcopy
import traceback
from enum import Enum
import threading
import numpy as np
from ydata_profiling import ProfileReport
import requests
import tiktoken

from services.base import ChatService, ServiceResponse, ServiceMessage, MessageType, PreviewIdentifier
from services.llm_service import LLMServiceMixin
from .data_manager import NMDCEnhancedDataManager
from .models import NMDCEnhancedConfig, QueryResult

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging for this module

# Define service type locally if needed
class ServiceType(Enum):
    """Types of services available in the system."""
    GENERIC = "generic"
    LLM = "llm"
    NMDC_ENHANCED = "nmdc_enhanced"
    DATASET = "dataset"
    DATABASE = "database"
    WEAVIATE = "weaviate"

class NMDCEnhancedService(ChatService, LLMServiceMixin):
    """Enhanced service for NMDC data discovery and integration.
    
    Provides advanced querying and analysis capabilities for the National Microbiome Data Collaborative (NMDC) database.
    Supports both natural language and direct SQL queries, with rich data integration and analysis features.
    
    Key Features:
    1. Natural Language & SQL Queries
    2. Unified Data Integration
    3. Statistical Analysis
    4. Geographic & Environmental Context
    5. Dataset Management
    
    Commands:
    - nmdc_enhanced.help: Show help and usage information
    - nmdc_enhanced.about: Show detailed statistics and analysis
    - nmdc_enhanced.entities: List available data types and counts
    - nmdc_enhanced: [question]: Process natural language queries
    - nmdc_enhanced.query [id]: Execute saved queries
    - convert [id] to dataset: Create datasets from results
    
    The service integrates with LLM capabilities for query generation and result analysis,
    while maintaining strict data access controls and query safety measures.
    """
    
    def __init__(
        self,
        name: str,
        llm_client: Optional[Any] = None,
        data_manager: Optional["NMDCEnhancedDataManager"] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the NMDC Enhanced service.

        Args:
            name: The name of the service
            llm_client: A client for interacting with LLMs
            data_manager: An NMDC Enhanced Data Manager instance
            config: Configuration for the service
        """
        # Initialize parent classes
        ChatService.__init__(self, name=name)
        LLMServiceMixin.__init__(self, service_name=name)
        
        self.name = name
        self.llm_client = llm_client
        self.config = config or {}
        
        try:
            # Initialize data manager
            if data_manager:
                self.data_manager = data_manager
            else:
                # Create default config for data manager with immediate loading
                data_manager_config = NMDCEnhancedConfig(
                    use_cache=self.config.get("use_cache", True),
                    cache_expiration_hours=self.config.get("cache_expiration_hours", 24),
                    max_retries=self.config.get("max_retries", 3)
                )
                self.data_manager = NMDCEnhancedDataManager(config=data_manager_config, load_data=True)
            
            # Initialize query engine with unified DataFrame
            if hasattr(self.data_manager, '_unified_df') and self.data_manager._unified_df is not None and not self.data_manager._unified_df.empty:
                from .query_engine import NMDCQueryEngine
                self._query_engine = NMDCQueryEngine(memory_limit="4GB")
                self._query_engine.register_dataframe(self.data_manager._unified_df, "unified")
                
                # Ensure statistics and column descriptions are calculated
                if not hasattr(self.data_manager, '_stats') or not self.data_manager._stats:
                    self.data_manager._calculate_unified_statistics()
                if not hasattr(self.data_manager, '_column_descriptions') or not self.data_manager._column_descriptions:
                    self.data_manager._column_descriptions = self.data_manager._get_base_column_descriptions()
            
            # Start background refresh if enabled
            if self.config.get("enable_background_refresh", False) and self.config.get("use_cache", True):
                self.data_manager.start_background_refresh()
                
        except Exception as e:
            logger.error(f"Error initializing NMDCEnhancedService: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # Initialize query and result storage
        self.successful_queries_store = {}
        self.last_result_id = None
        self._query_builder = None
        
        # Register prefixes for PreviewIdentifier
        try:
            PreviewIdentifier.register_prefix("nmdc_enhanced_query")
            PreviewIdentifier.register_prefix("nmdc_enhanced_result")
        except ValueError:
            pass
        
        logger.info(f"Initialized {self.name} service with data manager: {self.data_manager}")
    
    def can_handle(self, message: str) -> bool:
        """Check if this service can handle the message."""
        message = message.strip()
        
        # Handle "tell me about" pattern first
        if "tell me about nmdc_enhanced" in message.lower():
            return True
        
        # Handle direct SQL queries with triple backticks - must start with ```nmdc_enhanced at the start of message
        if re.search(r'^```nmdc_enhanced\s+(.*?)\s*```', message, re.DOTALL):
            return True
        
        # Handle direct service commands
        if message.lower().startswith("nmdc_enhanced."):
            command = message[len("nmdc_enhanced."):].strip()
            
            # Handle query/search commands
            if command.startswith(("query", "search")):
                return True
                
            # Handle other commands
            if command in ["info", "about", "help", "entities"]:
                return True
                
            # Handle convert to dataset
            if command.startswith("convert") and "to dataset" in command:
                match = re.search(r'(nmdc_enhanced_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset', command)
                if match:
                    return True
        
        # Handle natural language queries with colon prefix
        if message.startswith("nmdc_enhanced:"):
            return True
            
        # Handle non-prefixed convert command
        if message.startswith("convert "):
            match = re.search(r'convert\s+(nmdc_enhanced_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset', message)
            if match:
                return True
            
        return False

    def parse_request(self, message: str) -> Dict[str, Any]:
        """Parse service request to extract command and parameters."""
        message = message.strip()
        
        # Handle "tell me about" pattern first
        if "tell me about nmdc_enhanced" in message.lower():
            return {"command_type": "info", "content": {"command": "about"}}
        
        # Handle direct SQL queries with triple backticks
        sql_match = re.search(r'^```nmdc_enhanced\s+(.*?)\s*```', message, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
            if any(keyword in sql_query.lower() for keyword in ["select", "with"]):
                return {
                    "command_type": "structured_query",
                    "content": {"query": sql_query}
                }
        
        # Handle direct service commands
        if message.lower().startswith("nmdc_enhanced."):
            parts = message[len("nmdc_enhanced."):].strip().split(" ", 1)
            command = parts[0].lower()
            args = parts[1].strip() if len(parts) > 1 else ""
            
            # Handle query/search commands
            if command in ["query", "search"]:
                return {
                    "command_type": "execute_query",
                    "content": {"query_id": args if args else None}
                }
            
            # Handle service info commands
            if command in ["info", "about"]:
                return {"command_type": "info", "content": {"command": command}}
            elif command == "help":
                return {"command_type": "info", "content": {"command": "help"}}
            elif command == "entities":
                return {"command_type": "entities", "content": {}}
            
            # Handle convert to dataset
            if command.startswith("convert") and "to dataset" in args.lower():
                match = re.search(r'(nmdc_enhanced_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset', args)
                if match:
                    query_id = match.group(1)
                    return {
                        "command_type": "convert_dataset",
                        "content": {"query_id": query_id}
                    }
        
        # Handle natural language queries with colon prefix
        if message.lower().startswith("nmdc_enhanced:"):
            nl_query = message[len("nmdc_enhanced:"):].strip()
            return {
                "command_type": "natural_language_query",
                "content": {"query": nl_query}
            }
        
        # Handle non-prefixed convert command
        if message.lower().startswith("convert "):
            match = re.search(r'convert\s+(nmdc_enhanced_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset', message)
            if match:
                query_id = match.group(1)
                return {
                    "command_type": "convert_dataset",
                    "content": {"query_id": query_id}
                }
        
        # Default to treating as natural language query
        return {
            "command_type": "natural_language_query",
            "content": {"query": message}
        }

    def execute(self, request: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the service request."""
        command_type = request.get("command_type", "")
        content = request.get("content", {})
        
        try:
            if command_type == "natural_language_query":
                return self._handle_natural_language_query(content)
            elif command_type == "structured_query":
                return self._handle_structured_query(content)
            elif command_type == "execute_query":
                return self._handle_query_execution(content, context)
            elif command_type == "info":
                # Get the command from the parsed content
                command = content.get("command")
                return self._handle_info_request({"command": command})
            elif command_type == "entities":
                return self._handle_entities()
            elif command_type == "convert_dataset":
                return self._handle_dataset_conversion(content, context)
            else:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Unknown command type: {command_type}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
        except Exception as e:
            logger.error(f"Error executing request: {str(e)}")
            logger.error(traceback.format_exc())
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error executing request: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )

    def _handle_info_request(self, context: Dict[str, Any] = None) -> ServiceResponse:
        """Handle info request to provide service information."""
        try:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj

            # Check if this is a help request
            if context and context.get("command") == "help":
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            message_type=MessageType.INFO,
                            content=self.get_help_text()
                        )
                    ]
                )
            
            # Check if we have valid unified data
            has_unified_data = (
                hasattr(self.data_manager, '_unified_df') and 
                self.data_manager._unified_df is not None and 
                not self.data_manager._unified_df.empty
            )
            
            if not has_unified_data:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            message_type=MessageType.ERROR,
                            content="No unified data available. Please try again later."
                        )
                    ]
                )
            
            # Get pre-calculated statistics and convert numpy types
            logger.debug("Calculating statistics from unified DataFrame")
            stats = self._query_engine.calculate_statistics(self.data_manager._unified_df)
            logger.debug(f"Raw stats keys: {list(stats.keys())}")
            
            # Convert all stats to serializable format with detailed logging
            serializable_stats = convert_numpy(stats)
            
            # Verify serialization works
            try:
                logger.debug("Attempting to serialize stats for verification")
                json.dumps(serializable_stats)
                logger.debug("Stats serialization successful")
            except Exception as e:
                logger.error(f"Stats serialization failed: {str(e)}")
                logger.error(f"Problematic stats: {serializable_stats}")
                raise
            
            # Build deterministic response content
            content = []
            
            # === Scope and Purpose ===
            content.append("## Scope and Purpose")
            content.append("The National Microbiome Data Collaborative (NMDC) is building a FAIR microbiome data sharing network through infrastructure, data standards, and community building to address pressing challenges in environmental science.")
            content.append("")
            content.append("This service provides access to NMDC data, enabling exploration of microbiome samples across diverse ecosystems and research studies. You can query the data using natural language or SQL to find patterns, relationships, and insights in environmental microbiome samples.")
            content.append("")
            
            # === Summary Statistics ===
            content.append("## Summary Statistics")
            
            # Dataset sizes
            if hasattr(self.data_manager, '_studies_df') and self.data_manager._studies_df is not None:
                content.append(f"- **Studies**: {len(self.data_manager._studies_df):,}")
            if hasattr(self.data_manager, '_biosamples_df') and self.data_manager._biosamples_df is not None:
                content.append(f"- **Biosamples**: {len(self.data_manager._biosamples_df):,}")
            if has_unified_data:
                content.append(f"- **Integrated Records**: {len(self.data_manager._unified_df):,}")
            content.append("")
            
            # Numerical column statistics
            if "numeric_columns" in serializable_stats:
                logger.debug("Processing numeric columns")
                # Separate study columns from physical variables
                study_cols = {}
                physical_cols = {}
                
                for col, col_stats in serializable_stats["numeric_columns"].items():
                    logger.debug(f"Processing column: {col}")
                    if col.startswith('study_'):
                        study_cols[col] = col_stats
                    else:
                        physical_cols[col] = col_stats
                
                # Study Data Table
                if study_cols:
                    content.append("### Study Data")
                    content.append("```")
                    content.append("┌────────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
                    content.append("│ Variable           │        Min │        Max │       Mean │     Median │    Std Dev │")
                    content.append("├────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
                    
                    for col, col_stats in study_cols.items():
                        # Clean up column name by removing 'study_' prefix
                        clean_col = col.replace('study_', '')
                        # Truncate if too long
                        if len(clean_col) > 18:
                            clean_col = clean_col[:15] + "..."
                        
                        content.append(
                            f"│ {clean_col:<18} │ {col_stats.get('min', 'N/A'):>10.2f} │ "
                            f"{col_stats.get('max', 'N/A'):>10.2f} │ {col_stats.get('mean', 'N/A'):>10.2f} │ "
                            f"{col_stats.get('median', 'N/A'):>10.2f} │ {col_stats.get('std', 'N/A'):>10.2f} │"
                        )
                    
                    content.append("└────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")
                    content.append("```")
                    content.append("")
                
                # Physical Variables Table
                if physical_cols:
                    content.append("### Physical Variables")
                    content.append("```")
                    content.append("┌────────────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
                    content.append("│ Variable           │        Min │        Max │       Mean │     Median │    Std Dev │")
                    content.append("├────────────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")
                    
                    for col, col_stats in physical_cols.items():
                        # Truncate if too long
                        clean_col = col
                        if len(clean_col) > 18:
                            clean_col = clean_col[:15] + "..."
                        
                        content.append(
                            f"│ {clean_col:<18} │ {col_stats.get('min', 'N/A'):>10.2f} │ "
                            f"{col_stats.get('max', 'N/A'):>10.2f} │ {col_stats.get('mean', 'N/A'):>10.2f} │ "
                            f"{col_stats.get('median', 'N/A'):>10.2f} │ {col_stats.get('std', 'N/A'):>10.2f} │"
                        )
                    
                    content.append("└────────────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")
                    content.append("```")
                    content.append("")
            
            # Environment summaries
            env_cols = [col for col in self.data_manager._unified_df.columns 
                       if col.startswith('env_') and col.endswith('_label')]
            if env_cols:
                content.append("### Environmental Contexts")
                for col in env_cols:
                    unique_values = self.data_manager._unified_df[col].dropna().unique()
                    if len(unique_values) > 0:
                        content.append(f"- **{col.replace('_label', '')}**: {', '.join(str(v) for v in unique_values[:5])}")
                        if len(unique_values) > 5:
                            content.append(f"  ... and {len(unique_values) - 5} more")
                content.append("")
            
            # Omics data summary
            omics_cols = [col for col in self.data_manager._unified_df.columns 
                         if col.startswith('study_omics_')]
            if omics_cols:
                content.append("### Available Omics Data")
                for col in omics_cols:
                    # Get the total sum of non-null values
                    total = self.data_manager._unified_df[col].sum()
                    if total > 0:
                        content.append(f"- **{col.replace('study_omics_', '')}**: {total:,.0f} total measurements")
                content.append("")
            
            # Create deterministic response
            deterministic_response = ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        message_type=MessageType.INFO,
                        content="\n".join(content)
                    )
                ]
            )
            
            # For about/info commands, add LLM analysis of the statistics
            if context and context.get("command") in ["about", "info"]:
                logger.debug("Preparing LLM analysis")
                # Prepare data for LLM analysis
                analysis_prompt = f"""You are an expert in environmental microbiology and data analysis. 
Please analyze the following NMDC database statistics and provide insights about:

1. Geographic Distribution:
   - Coverage of sampling locations
   - Distribution across different environments
   - Notable patterns in spatial coverage

2. Environmental Context:
   - Types of environments represented
   - Distribution of environmental conditions
   - Notable patterns in environmental variables

3. Biological Context:
   - Types of samples and measurements
   - Distribution of biological variables
   - Notable patterns in biological data

4. Research Implications:
   - Potential research applications
   - Unique strengths of the dataset
   - Areas for potential expansion

Statistics:
{json.dumps(serializable_stats, indent=2)}

Please provide a concise but comprehensive analysis focusing on these aspects."""

                # Get LLM analysis
                llm_analysis = self._call_llm(
                    messages=[{"role": "user", "content": analysis_prompt}],
                    system_prompt="You are an expert in environmental microbiology and data analysis."
                )
                
                # Add LLM analysis as a separate message
                deterministic_response.messages.append(
                    ServiceMessage(
                        service=self.name,
                        message_type=MessageType.SUMMARY,
                        content=llm_analysis
                    )
                )
            
            return deterministic_response
            
        except Exception as e:
            logger.error(f"Error handling info request: {str(e)}")
            logger.error(traceback.format_exc())
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        message_type=MessageType.ERROR,
                        content=f"Error generating service information: {str(e)}"
                    )
                ]
            )
    
    def _handle_dataset_conversion(self, content: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Convert query results to a dataset."""
        try:
            # Get query ID
            query_id = content.get("query_id")
            if not query_id:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="No query ID provided for dataset conversion",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Validate query ID format
            if not query_id.startswith("nmdc_enhanced_query_"):
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Invalid query ID format. Expected prefix 'nmdc_enhanced_query_', got: {query_id}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Get query result from store
            stored_result = self.successful_queries_store.get(query_id)
            if not stored_result:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"No results found for query ID: {query_id}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Convert stored result to DataFrame
            df = pd.DataFrame.from_records(stored_result['dataframe'])
            
            if df.empty:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Cannot convert empty results to dataset",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Generate dataset ID
            dataset_id = query_id
            
            # Create profile report
            try:
                profile = ProfileReport(df, title=f"NMDC Query Results - {dataset_id}")
                profile_html = profile.to_html()
            except Exception as e:
                logger.error(f"Error generating profile report: {str(e)}")
                profile_html = None
            
            # Create dataset entry
            dataset_entry = {
                'metadata': {
                    'source': f"NMDC Enhanced Query {dataset_id}",
                    'creation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'query': stored_result.get('query'),
                    'query_id': dataset_id,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'selectable': True,
                    'transformations': []
                },
                'df': df.to_dict('records'),
                'profile_report': profile_html
            }
            
            # Update stores
            store_updates = {
                'datasets_store': {dataset_id: dataset_entry}
            }
            
            # Create consolidated message
            message_parts = [f"✓ Query results converted to dataset '{dataset_id}'"]
            if profile_html:
                message_parts.append("Generated profile report for the dataset")
            message_parts.append(f"You can access this dataset using the Datasets tab or by referencing dataset ID '{dataset_id}'")
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content="\n".join(message_parts),
                        message_type=MessageType.INFO
                    )
                ],
                store_updates=store_updates
            )
            
        except Exception as e:
            logger.error(f"Error converting result to dataset: {str(e)}")
            logger.error(traceback.format_exc())
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error converting result to dataset: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_query_execution(self, content: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute a query by ID."""
        try:
            # Get query ID from content
            query_id = content.get("query_id")
            if not query_id:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="No query ID provided. Please provide a query ID or run a new query.",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Find the query in chat history
            sql_query = None
            chat_history = context.get("chat_history", [])
            for message in chat_history:
                if message.get("role") == "assistant":
                    message_content = message.get("content", "")
                    
                    # First find all SQL blocks
                    sql_blocks = list(re.finditer(r'```sql\s*(.*?)\s*```', message_content, re.DOTALL))
                    
                    # For each SQL block, look for its associated query ID
                    for block_match in sql_blocks:
                        block_end = block_match.end()
                        # Look for the query ID after this block
                        id_match = re.search(
                            r'Query ID:\s*(nmdc_enhanced_query_\d{8}_\d{6}(?:_orig|_alt\d+))',
                            message_content[block_end:],
                            re.DOTALL
                        )
                        
                        if id_match:
                            found_id = id_match.group(1)
                            if found_id == query_id:
                                sql_query = block_match.group(1).strip()
                                # Remove any backticks that might be in the SQL
                                sql_query = sql_query.replace('`', '"')
                                break
                    
                    if sql_query:
                        break
            
            if not sql_query:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Query ID '{query_id}' not found in chat history. Please make sure you're using a valid query ID from a previous query suggestion.",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Execute the query
            query_result, error_msg = self._query_engine.execute_query(sql_query)
            
            if error_msg:
                logger.error(f"Query execution error: {error_msg}")
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Error executing query: {error_msg}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            if query_result is None or query_result.dataframe.empty:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Query executed successfully but returned no results",
                            message_type=MessageType.INFO
                        )
                    ]
                )
            
            try:
                # Calculate additional statistics
                stats = self._calculate_statistics(query_result.dataframe)
                query_result.metadata['statistics'] = stats
                
                # Generate preview
                preview = self._generate_preview(query_result.dataframe)
                
                # Get LLM summary
                summary = self._generate_result_summary(query_result.dataframe, sql_query, stats)
                
                # Store in successful_queries_store using QueryResult format
                self.successful_queries_store[query_id] = {
                    'query': sql_query,
                    'dataframe': query_result.dataframe.to_dict('records'),
                    'metadata': {
                        'statistics': stats,
                        'preview': preview,
                        'summary': summary,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'query_result'
                    }
                }
                
                # Format response message
                response = f"""## Query Results (ID: {query_id})

### Preview
{preview}

### Statistics Summary
{self._format_statistics(stats)}

### Analysis
{summary}

To convert these results to a dataset, run: `convert {query_id} to dataset`"""
                
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=response,
                            message_type=MessageType.RESULT
                        )
                    ]
                )
                
            except Exception as e:
                logger.error(f"Error processing query results: {str(e)}")
                logger.error(traceback.format_exc())
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Error processing query results: {str(e)}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(traceback.format_exc())
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error executing query: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for the result DataFrame."""
        stats = {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "numeric_stats": {},
            "categorical_stats": {},
            "geographic_stats": {},
            "omics_stats": {}
        }
        
        # Numeric column statistics using pandas describe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            desc_stats = df[col].describe()
            stats["numeric_stats"][col] = {
                "count": int(desc_stats["count"]),
                "mean": float(desc_stats["mean"]),
                "std": float(desc_stats["std"]),
                "min": float(desc_stats["min"]),
                "25%": float(desc_stats["25%"]),
                "50%": float(desc_stats["50%"]),
                "75%": float(desc_stats["75%"]),
                "max": float(desc_stats["max"])
            }
        
        # Categorical column statistics (term frequencies)
        categorical_cols = [
            "ecosystem", "ecosystem_category", "ecosystem_type", "ecosystem_subtype",
            "specific_ecosystem", "habitat", "location", "community", "geo_loc_name",
            "env_broad_scale.label", "env_local_scale.label", "env_medium.label"
        ]
        
        for col in categorical_cols:
            if col in df.columns:
                value_counts = df[col].value_counts()
                stats["categorical_stats"][col] = {
                    "unique_count": int(len(value_counts)),
                    "term_frequencies": {str(k): int(v) for k, v in value_counts.to_dict().items()}
                }
        
        # Geographic statistics
        if "latitude" in df.columns and "longitude" in df.columns:
            stats["geographic_stats"] = {
                "latitude": {
                    "min": float(df["latitude"].min()),
                    "max": float(df["latitude"].max()),
                    "mean": float(df["latitude"].mean()),
                    "std": float(df["latitude"].std())
                },
                "longitude": {
                    "min": float(df["longitude"].min()),
                    "max": float(df["longitude"].max()),
                    "mean": float(df["longitude"].mean()),
                    "std": float(df["longitude"].std())
                }
            }
        
        # Omics data statistics
        omics_cols = [col for col in df.columns if col.startswith("study_omics_")]
        for col in omics_cols:
            stats["omics_stats"][col] = {
                "samples_with_data": int((df[col] > 0).sum()),
                "total_measurements": int(df[col].sum())
            }
        
        return stats

    def _generate_preview(self, df: pd.DataFrame) -> str:
        """Generate an aesthetic preview of the DataFrame."""
        # Select most relevant columns
        priority_columns = [
            "id", "name", "ecosystem", "ecosystem_category", 
            "latitude", "longitude", "country", "depth",
            "env_broad_scale_label", "env_local_scale_label"
        ]
        
        preview_columns = [col for col in priority_columns if col in df.columns]
        preview_columns.extend([col for col in df.columns if col not in preview_columns][:5])
        
        preview_df = df[preview_columns].head()
        
        # Convert to string with custom formatting
        preview_lines = []
        
        # Add header
        headers = [str(col) for col in preview_df.columns]
        col_widths = [max(len(str(col)), preview_df[col].astype(str).str.len().max()) for col in preview_df.columns]
        
        # Header line
        header = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths)) + " |"
        separator = "|" + "|".join("-" * (w + 2) for w in col_widths) + "|"
        
        preview_lines.append(header)
        preview_lines.append(separator)
        
        # Add data rows
        for _, row in preview_df.iterrows():
            row_values = [str(val) for val in row]
            row_str = "| " + " | ".join(f"{str(val):<{w}}" for val, w in zip(row_values, col_widths)) + " |"
            preview_lines.append(row_str)
            
        return "```\n" + "\n".join(preview_lines) + "\n```"

    def _format_statistics(self, stats: Dict[str, Any]) -> str:
        """Format statistics into a readable string."""
        lines = [
            f"Total Rows: {stats['row_count']}",
            f"Total Columns: {stats['column_count']}\n"
        ]
        
        # Numeric Statistics
        if stats["numeric_stats"]:
            lines.append("**Numeric Statistics:**")
            for col, col_stats in stats["numeric_stats"].items():
                lines.append(f"- {col}:")
                lines.append(f"  * Count: {col_stats['count']}")
                lines.append(f"  * Mean: {col_stats['mean']:.2f}")
                lines.append(f"  * Standard Deviation: {col_stats['std']:.2f}")
                lines.append(f"  * Min: {col_stats['min']:.2f}")
                lines.append(f"  * 25th Percentile: {col_stats['25%']:.2f}")
                lines.append(f"  * Median: {col_stats['50%']:.2f}")
                lines.append(f"  * 75th Percentile: {col_stats['75%']:.2f}")
                lines.append(f"  * Max: {col_stats['max']:.2f}")
            lines.append("")
        
        # Categorical Statistics
        if stats["categorical_stats"]:
            lines.append("**Categorical Distributions:**")
            for col, col_stats in stats["categorical_stats"].items():
                lines.append(f"- {col}:")
                lines.append(f"  * Unique Values: {col_stats['unique_count']}")
                lines.append("  * Term Frequencies:")
                for term, freq in col_stats["term_frequencies"].items():
                    lines.append(f"    - {term}: {freq}")
            lines.append("")
        
        # Geographic Statistics
        if stats["geographic_stats"]:
            lines.append("**Geographic Coverage:**")
            if "latitude" in stats["geographic_stats"]:
                lat_stats = stats["geographic_stats"]["latitude"]
                lon_stats = stats["geographic_stats"]["longitude"]
                lines.append(f"- Latitude:")
                lines.append(f"  * Min: {lat_stats['min']:.2f}")
                lines.append(f"  * Max: {lat_stats['max']:.2f}")
                lines.append(f"  * Mean: {lat_stats['mean']:.2f}")
                lines.append(f"  * Standard Deviation: {lat_stats['std']:.2f}")
            if "longitude" in stats["geographic_stats"]:
                lines.append(f"- Longitude:")
                lines.append(f"  * Min: {lon_stats['min']:.2f}")
                lines.append(f"  * Max: {lon_stats['max']:.2f}")
                lines.append(f"  * Mean: {lon_stats['mean']:.2f}")
                lines.append(f"  * Standard Deviation: {lon_stats['std']:.2f}")
            lines.append("")
        
        # Omics Statistics
        if stats["omics_stats"]:
            lines.append("**Omics Data Summary:**")
            for col, col_stats in stats["omics_stats"].items():
                lines.append(f"- {col}:")
                lines.append(f"  * Samples with Data: {col_stats['samples_with_data']}")
                lines.append(f"  * Total Measurements: {col_stats['total_measurements']}")
            lines.append("")
        
        return "\n".join(lines)

    def _generate_result_summary(self, df: pd.DataFrame, query: str, stats: Dict[str, Any]) -> str:
        """Generate a natural language summary of the results using LLM."""
        system_prompt = """You are an expert data analyst specializing in environmental microbiome data. 
Your task is to provide a concise summary of query results from the NMDC database.
Focus on the geographic, environmental, and physical characteristics of the samples found.
Highlight any patterns or notable distributions in the data.
Keep your summary clear and scientific, but accessible to researchers."""

        # Prepare context for LLM
        context = {
            "query": query,
            "statistics": stats,
            "row_count": len(df),
            "column_count": len(df.columns)
        }
        
        prompt = f"""Please analyze these NMDC query results:

Original Query:
```sql
{query}
```

Result Statistics:
{json.dumps(stats, indent=2)}

Provide a concise summary focusing on:
1. Geographic distribution of samples
2. Environmental contexts represented
3. Physical characteristics
4. Notable patterns or clusters in the data
5. How the results relate to the original query intent

Keep the summary clear and scientific, but accessible."""

        # Get LLM summary
        summary = self._call_llm(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt
        )
        
        return summary

    def get_help_text(self) -> str:
        """Get help text for NMDC Enhanced service."""
        return """
🔬 **NMDC Enhanced Service**

The NMDC Enhanced Service provides advanced data discovery and integration capabilities for the National Microbiome Data Collaborative (NMDC) database.

### Commands:
- `nmdc_enhanced.help` - Show this help message
- `nmdc_enhanced.about` or `tell me about nmdc_enhanced` - Show service information and statistics
- `nmdc_enhanced.entities` - List available entity types and their counts
- `nmdc_enhanced: [your question]` - Ask a natural language question about the data
- `nmdc_enhanced.query [query_id]` - Execute a previously built query
- `convert [result_id] to dataset` - Convert query results to a dataset

### Query Formats:
1. Natural Language:
   ```
   nmdc_enhanced: Find soil samples from Washington state
   ```

2. Direct SQL (must start with ```nmdc_enhanced):
   ```nmdc_enhanced
   SELECT *
   FROM unified
   WHERE ecosystem = 'soil'
   AND geo_loc_name LIKE '%Washington%'
   ```

### Example Natural Language Queries:
- Find soil samples from Washington state
- Show samples with carbon measurements from forest environments
- Get soil samples from the Wrighton lab
- List studies about soil microbiology
- Show biosamples from forest environments
- Find data objects with metagenome sequencing

### Results Include:
- Sample metadata (ID, collection date, location)
- Study information (title, PI, objectives)
- Physical measurements and environmental parameters
- Counts of related data objects by type
- Geographic and environmental context

For detailed statistics and analysis, use `nmdc_enhanced.about` or `tell me about nmdc_enhanced`."""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for NMDC Enhanced capabilities."""
        return """
NMDC Enhanced Service Capabilities:

1. Purpose:
   - Access and analyze microbiome data from the National Microbiome Data Collaborative
   - Query environmental samples across diverse ecosystems and studies
   - Analyze physical measurements and environmental parameters
   - Generate insights about microbial communities and their environments

2. Query Methods:
   a) Natural Language:
      - Format: `nmdc_enhanced: [your question]`
      - Example: `nmdc_enhanced: Find soil samples from Washington state`
      - System converts questions to optimized SQL

   b) Direct SQL:
      - Format: Must start with ```nmdc_enhanced
      - Example:
        ```nmdc_enhanced
        SELECT *
        FROM unified
        WHERE ecosystem = 'soil'
        AND "tot_org_carb.has_numeric_value" > 5.0
        ```
      - Requires proper column quoting for names with dots
      - Uses DuckDB SQL syntax

3. Commands:
   - Execute queries: `nmdc_enhanced.query [query_id]`
   - Get help: `nmdc_enhanced.help`
   - View info: `nmdc_enhanced.about`
   - Convert results: `convert [query_id] to dataset`

4. Data Categories:
   a) Study Information:
      - Title, description, investigators
      - Collection dates and locations
      - Study objectives and design
   
   b) Environmental Context:
      - Ecosystem type (e.g., marine, terrestrial)
      - Location and coordinates
      - Habitat classification
      - Environmental conditions
   
   c) Sample Measurements:
      - Carbon content (total, organic, inorganic)
      - Nitrogen content and ratios
      - pH and salinity
      - Temperature and oxygen
      - Other chemical parameters
"""

    def process_message(self, message: str, chat_history: List[Dict]) -> ServiceResponse:
        """Process a message using LLM capabilities.
        
        Args:
            message: User message to process
            chat_history: List of previous chat messages
            
        Returns:
            ServiceResponse containing messages and updates
        """
        try:
            # Parse the request to determine command type
            parsed_request = self.parse_request(message)
            command_type = parsed_request.get("command_type", "natural_language_query")
            content = parsed_request.get("content", {})
            
            # Build domain context for LLM
            domain_context = {
                "command_type": command_type,
                "data_manager": self.data_manager,
                "query_engine": self._query_engine,
                "query_builder": self._query_builder
            }
            
            # Prepare LLM context
            system_prompt, context_messages, limits = self._prepare_llm_context(
                message, chat_history, domain_context
            )
            
            # Get LLM response with retry logic
            response = self._call_llm(context_messages, system_prompt)
            
            # Process and validate response
            processed_response = self._process_response(response)
            
            # Create service response
            messages = [
                ServiceMessage(
                    service=self.name,
                    content=processed_response,
                    message_type=MessageType.RESULT
                )
            ]
            
            return ServiceResponse(messages=messages)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            logger.error(traceback.format_exc())
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error processing message: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )

    def summarize(self, content, chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of the given content.
        
        Args:
            content: Content to summarize (DataFrame or text)
            chat_history: History of the conversation
            
        Returns:
            Summary text
        """
        try:
            # Get data context from data manager
            data_context = self.data_manager.get_dataframe_context() or {}
            
            if isinstance(content, pd.DataFrame):
                df = content
                row_count = len(df)
                col_count = len(df.columns)
                
                # Get basic statistics
                stats_text = f"Dataset contains {row_count:,} rows and {col_count:,} columns.\n\n"
                
                # Add sample values for key columns
                key_columns = ['title', 'ecosystem_type', 'ecosystem_category', 'depth', 'ph', 'scientific_name']
                for col in key_columns:
                    if col in df.columns:
                        unique_values = df[col].dropna().unique()
                        if len(unique_values) > 0:
                            sample_values = unique_values[:5]
                            stats_text += f"{col}: {', '.join(str(v) for v in sample_values)}"
                            if len(unique_values) > 5:
                                stats_text += f" and {len(unique_values) - 5} more values"
                            stats_text += "\n"
                
                # Create prompt for LLM
                prompt = f"""
You are an AI assistant helping with analyzing microbiome data from the NMDC database.

Here are some statistics about the dataset:
{stats_text}

Please provide a concise summary of what this data represents, focusing on:
1. The general contents and scope of the data
2. Key patterns or distributions (ecosystem types, geographic areas, etc.)
3. Potential scientific relevance or applications
4. Suggestions for further analysis
"""
                
                # Get LLM interpretation
                llm_response = self.llm_client.chat_completion(
                    [{"role": "system", "content": prompt}],
                    model=self.config.get("model_name", ""),
                    temperature=self.config.get("temperature", 0.7)
                )
                
                return llm_response.choices[0].message.content
                
            else:
                # For text content, use a simpler prompt
                prompt = """
You are an AI assistant helping with microbiome data analysis. 
Summarize the following content related to the NMDC (National Microbiome Data Collaborative) database.
Focus on key scientific findings, patterns, and implications. Be concise and informative.
"""
                
                # Get LLM interpretation
                llm_response = self.llm_client.chat_completion(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": str(content)}
                    ],
                    model=self.config.get("model_name", ""),
                    temperature=self.config.get("temperature", 0.7)
                )
                
                return llm_response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error summarizing content: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Error generating summary: {str(e)}"

    def _handle_natural_language_query(self, content: Dict[str, Any]) -> ServiceResponse:
        """Handle a natural language query to generate structured query suggestions."""
        try:
            # Get natural language query - handle both string and dict input
            if isinstance(content, dict):
                nl_query = content.get("query", "")
            else:
                nl_query = str(content)
            
            nl_query = nl_query.strip()
            logger.debug(f"Processing natural language query: {nl_query[:100]}...")
            
            if not nl_query:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Please provide a natural language query after 'nmdc_enhanced:'",
                            message_type=MessageType.ERROR
                        )
                    ]
                )

            # Initialize query builder if needed
            if not hasattr(self, '_query_builder'):
                logger.debug("Initializing query builder")
                from .query_builder import NMDCQueryBuilder
                self._query_builder = NMDCQueryBuilder(self.llm_client, self.config)
            
            # Get data context from data manager
            data_context = self.data_manager.get_dataframe_context() or {}
            logger.debug(f"Retrieved data context with {len(data_context)} keys")
            
            # Initialize table structure and environmental context
            table_structure = [
                "Available Columns and Their Meanings:",
                "",
                "Key Columns with Detailed Descriptions:",
                "- depth: Depth of the sample (e.g., water column, soil layer)",
                "- longitude: Geographical longitude coordinate of the sampling site",
                "- latitude: Geographical latitude coordinate of the sampling site",
                "- ecosystem: Type of ecosystem (e.g., marine, terrestrial)",
                "- ecosystem_category: Broader classification of the ecosystem (e.g., freshwater, desert)",
                "- ecosystem_type: Specific type of ecosystem (e.g., wetland, forest)",
                "- ecosystem_subtype: Subclassification of the ecosystem (e.g., coral reef, grassland)",
                "- specific_ecosystem: Detailed description of the ecosystem (e.g., tropical rainforest)",
                "- habitat: Specific habitat of the sample (e.g., coral reef, soil)",
                "- location: General location of the sample (e.g., 'ocean', 'forest')",
                "- community: Biological community present in the sample (e.g., microbial, plant)",
                "- samp_name: Name or identifier of the sample",
                "- geo_loc_name: Geographical name of the sampling location",
                "- env_broad_scale.label: Broad-scale environmental classification (e.g., biome, climate zone)",
                "- env_local_scale.label: Local-scale environmental classification (e.g., pond, forest stand)",
                "- env_medium.label: Label describing the environmental medium",
                "- elev: Elevation of the sampling site above sea level",
                "- ph: pH level of the sample",
                "- tot_org_carb.has_numeric_value: Numerical value of total organic carbon",
                "- diss_org_carb.has_numeric_value: Numerical value of dissolved organic carbon",
                "- diss_inorg_carb.has_numeric_value: Numerical value of dissolved inorganic carbon",
                "- tot_nitro_content.has_numeric_value: Numerical value of total nitrogen content",
                "- carb_nitro_ratio.has_numeric_value: Numerical value of the carbon-to-nitrogen ratio",
                "- temp.has_numeric_value: Numerical value of temperature",
                "- salinity: Salinity of the sample",
                "- diss_oxygen.has_numeric_value: Numerical value of dissolved oxygen",
                "- ammonium.has_numeric_value: Numerical value of ammonium concentration",
                "- tot_phosp.has_numeric_value: Numerical value of total phosphorus",
                "- chlorophyll.has_numeric_value: Numerical value of chlorophyll concentration",
                "- diss_inorg_nitro.has_numeric_value: Numerical value of dissolved inorganic nitrogen",
                "",
              ]
            
            # Add all other columns from the unified DataFrame
            all_columns = list(self.data_manager._unified_df.columns)
            described__columns={
                "depth",
                "longitude",
                "latitude",
                "ecosystem",
                "ecosystem_category",
                "ecosystem_type",
                "ecosystem_subtype",
                "specific_ecosystem",
                "habitat",
                "location",
                "community",
                "samp_name",
                "geo_loc_name",
                "env_broad_scale.label",
                "env_local_scale.label",
                "env_medium.label",
                "elev",
                "ph",
                "tot_org_carb.has_numeric_value",
                "diss_org_carb.has_numeric_value",
                "diss_inorg_carb.has_numeric_value",
                "tot_nitro_content.has_numeric_value",
                "carb_nitro_ratio.has_numeric_value",
                "temp.has_numeric_value",
                "salinity",
                "diss_oxygen.has_numeric_value",
                "ammonium.has_numeric_value",
                "tot_phosp.has_numeric_value",
                "chlorophyll.has_numeric_value",
                "diss_inorg_nitro.has_numeric_value"
            }
            other_columns = [col for col in all_columns if col not in described__columns]
            # Initialize environmental context
            env_context = []
            
            # Add environmental context if available
            if 'geographic_coverage' in data_context:
                env_context.extend([
                    "Environmental Context:",
                    f"Geographic Coverage: {self._format_geographic_coverage(data_context['geographic_coverage'])}",
                ])

            # Create system prompt with enhanced context
            system_prompt = f"""You are an expert data analyst specializing in converting natural language questions into SQL queries for the National Microbiome Data Collaborative (NMDC) database. Your task is to generate SQL queries that help scientists analyze environmental microbiome data.

The unified data source represents a set of records about physical samples from the environment that are part of different studies. Each row contains information about:
- The study (including principal investigator, dates, descriptions, study_id)
- The sample and its metadata (including id, environmental and ecosystem categories)
- Physical measurements of the sample
- Indications of other omics measurements that may be accessible via other commands

When formulating a search, you must infer which of columns in the unified table are most important to include. 
The full list of available columns in the unified table includes this subset:
{chr(10).join(f"- {col}" for col in other_columns)}

and the subset in the described subset below. To help with your inference, here are key example columns with their formal descriptions so you can understand these data and how they are labeled:

{chr(10).join(table_structure)}

{chr(10).join(env_context)}

QUERY GENERATION PROCESS:
1. Column Selection Analysis:
   - First, analyze which columns are most relevant to the user's question
   - Explain your column selection logic
   - IMPORTANT: The main table is named "unified" (not "unified_table" or any other variation)
   - CRITICAL: Only use columns that actually exist in the unified table as listed above.
   - STRICT: Do not assume or infer the existence of related columns (e.g., if "carb_nitro_ratio.has_numeric_value" exists, do not assume "carb_nitro_ratio.has_unit" exists)
   - STRICT: Do not use column names that are not exactly as shown in the lists above
   - STRICT: Do not modify column names or add suffixes/prefixes
   - STRICT: Do not use column names that contain parts of listed columns (e.g., if "carb_nitro_ratio.has_numeric_value" exists, do not use "carb_nitro_ratio" or "carb_nitro_ratio.has_unit")
   - Consider both direct matches and related columns that might help answer the question

2. Query Strategy:
   - Develop a clear strategy for answering the question
   - Consider multiple approaches (e.g., direct filtering, statistical analysis, aggregation)
   - Explain the rationale for your chosen approach

DuckDB SQL Constraints:
1. Use DuckDB-compatible SQL syntax (similar to PostgreSQL)
2. ONLY SELECT statements are allowed - the DataFrame is read-only
3. Column names with special characters or spaces must use double quotes
4. For column names containing dots (e.g., 'tot_org_carb.has_numeric_value'):
   - Use double quotes around the entire column name to treat it as a single identifier
   - Example: Use `"tot_org_carb.has_numeric_value"` instead of `tot_org_carb.has_numeric_value`
5. Use CAST() for explicit type conversions (e.g., CAST(column AS FLOAT))
6. For string operations, use || for concatenation
7. For aggregations:
   - Use PERCENTILE_CONT(fraction) WITHIN GROUP (ORDER BY column) for percentiles
   - Example: PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY "column_name")
   - GROUP BY must include all non-aggregated columns
   - Window functions must have OVER clause
8. For NULL handling:
   - Use COALESCE() or IFNULL() for NULL replacements
   - When using COALESCE with numeric operations, explicitly cast values to the same type
   - Example: COALESCE(CAST(column AS FLOAT), 0.0) instead of COALESCE(column, 0)
   - IS NULL / IS NOT NULL for NULL checks

IMPORTANT Safety Constraints:
1. The DataFrame is READ-ONLY - only SELECT statements are allowed
2. The following operations are NOT allowed and will be rejected:
   - CREATE, INSERT, UPDATE, DELETE statements
   - DROP, ALTER, TRUNCATE statements
   - Any other data modification operations

RESPONSE FORMAT:
Your response must follow this exact format:
1. Column Selection Analysis:
   - List the columns you've identified as relevant
   - Explain why each column was chosen
   - Note any potential data quality considerations

2. For each query suggest formatted SQL with explanation as follows:
   ```sql
   -- Purpose: Clear description of what this query does
   -- Columns Used: List of columns and their purpose
   -- Strategy: Brief explanation of the query approach
   SELECT ...
   ```

5. Brief conclusion with:
   - Comparison of the different approaches
   - Any caveats or limitations
   - Suggestions for result interpretation"""

            # Calculate token counts for each section
            enc = tiktoken.get_encoding("cl100k_base")
            
            # Split system prompt into sections
            sections = {
                "Role and Task": system_prompt.split("The unified data source")[0],
                "Data Source Description": system_prompt.split("The unified data source")[1].split("The full list of available columns")[0],
                "Column List": system_prompt.split("The full list of available columns in the unified table includes this subset:")[1].split("and the subset in the described subset below")[0],
                "Column Descriptions": system_prompt.split("ns so you can understand these data and how they are labeled:")[1].split("Environmental Context:")[0],
                "Query Generation Process": system_prompt.split("QUERY GENERATION PROCESS")[1].split("DuckDB SQL Constraints")[0],
                "SQL Constraints": system_prompt.split("DuckDB SQL Constraints")[1].split("IMPORTANT Safety Constraints")[0],
                "Safety Constraints": system_prompt.split("IMPORTANT Safety Constraints")[1].split("RESPONSE FORMAT")[0],
                "Response Format": system_prompt.split("RESPONSE FORMAT")[1]
            }
            
            # Log token counts for each section
            logger.debug("=== System Prompt Token Analysis ===")
            total_tokens = len(enc.encode(system_prompt))
            for section_name, section_text in sections.items():
                tokens = len(enc.encode(section_text))
                logger.debug(f"{section_name}: {tokens:,} tokens")
            logger.debug(f"Total system prompt tokens: {total_tokens:,}")
            
            # Prepare messages for LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate SQL queries for this question: {nl_query}"}
            ]
            
            # Get LLM response
            logger.debug("Making LLM call with system prompt length: %d", len(system_prompt))
            llm_response = self._call_llm(messages=messages[1:], system_prompt=system_prompt)
            
            if not llm_response:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Failed to generate SQL query. Please try rephrasing your question.",
                            message_type=MessageType.ERROR
                        )
                    ]
                )

            try:
                logger.debug("LLM response length: %d", len(llm_response))
                
                # Process the response to add query IDs to SQL blocks
                processed_response = llm_response
                previous_id = None  # Track the previous ID for alternatives
                
                # Find all SQL blocks using only triple backtick format
                sql_blocks = list(re.finditer(r'```sql\s*(.*?)\s*```', llm_response, re.DOTALL))
                logger.debug("Found %d SQL blocks", len(sql_blocks))
                
                for i, match in enumerate(sql_blocks):
                    logger.debug("Processing SQL block %d", i)
                    sql_block = match.group(1).strip()  # Get just the SQL content
                    logger.debug("SQL block %d length: %d", i, len(sql_block))
                    
                    # Generate query ID - first one is original, rest are alternatives
                    if i == 0:
                        query_id = PreviewIdentifier.create_id(prefix="nmdc_enhanced_query")
                        previous_id = query_id
                    else:
                        query_id = PreviewIdentifier.create_id(previous_id=previous_id)
                        previous_id = query_id
                    
                    # Reconstruct the SQL block without any stray backticks
                    clean_sql_block = f"```sql\n{sql_block}\n```"
                    
                    # Add the ID and execution instruction after the SQL block
                    replacement = f"{clean_sql_block}\nQuery ID: {query_id}\nTo execute this query, run: `nmdc_enhanced.query {query_id}`\n"
                    processed_response = processed_response.replace(match.group(0), replacement)
                    logger.debug("Processed response length after block %d: %d", i, len(processed_response))
                
                logger.debug("Final processed response length: %d", len(processed_response))
                
                # Return just the processed response with query suggestions
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=processed_response,
                            message_type=MessageType.SUGGESTION
                        )
                    ]
                )
                
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                logger.error(traceback.format_exc())
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Error processing natural language query: {str(e)}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
        except Exception as e:
            logger.error(f"Error handling natural language query: {str(e)}")
            logger.error(traceback.format_exc())
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error processing natural language query: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )

    def _handle_structured_query(self, content: Dict[str, Any]) -> ServiceResponse:
        """Handle a structured SQL query directly provided by the user.
        
        This method validates the SQL query against DuckDB syntax and NMDC-specific rules,
        suggests improvements, and returns query IDs for execution.
        
        Args:
            content: Dictionary containing the SQL query under the 'query' key
            
        Returns:
            ServiceResponse with validation results and suggestions
        """
        try:
            # Get query from content
            query = content.get("query", "").strip()
            if not query:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Please provide a SQL query",
                            message_type=MessageType.ERROR
                        )
                    ]
                )

            # Create specialized prompt for SQL validation
            validation_prompt = f"""You are an expert SQL analyst specializing in validating and improving SQL queries for the National Microbiome Data Collaborative (NMDC) database.

Your task is to:
1. Validate the SQL query against DuckDB syntax rules
2. Check for proper column names and quoting
3. Suggest improvements for clarity and performance
4. Provide alternative approaches if relevant

The unified data source represents a set of records about physical samples from the environment that are part of different studies. Each row contains information about:
- The study (including principal investigator, dates, descriptions, study_id)
- The sample and its metadata (including id, environmental and ecosystem categories)
- Physical measurements of the sample
- Indications of other omics measurements that may be accessible via other commands

Available Columns:
{chr(10).join(f"- {col}" for col in self.data_manager._unified_df.columns)}

DuckDB SQL Constraints:
1. Use DuckDB-compatible SQL syntax (similar to PostgreSQL)
2. ONLY SELECT statements are allowed - the DataFrame is read-only
3. Column names with special characters or spaces must use double quotes
4. For column names containing dots (e.g., 'tot_org_carb.has_numeric_value'):
   - Use double quotes around the entire column name to treat it as a single identifier
   - Example: Use `"tot_org_carb.has_numeric_value"` instead of `tot_org_carb.has_numeric_value`
5. Use CAST() for explicit type conversions (e.g., CAST(column AS FLOAT))
6. For string operations, use || for concatenation
7. For aggregations:
   - Use PERCENTILE_CONT(fraction) WITHIN GROUP (ORDER BY column) for percentiles
   - Example: PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY "column_name")
   - GROUP BY must include all non-aggregated columns
   - Window functions must have OVER clause
8. For NULL handling:
   - Use COALESCE() or IFNULL() for NULL replacements
   - When using COALESCE with numeric operations, explicitly cast values to the same type
   - Example: COALESCE(CAST(column AS FLOAT), 0.0) instead of COALESCE(column, 0)
   - IS NULL / IS NOT NULL for NULL checks
9. ONLY use EXACT column names that are listed above

RESPONSE FORMAT:
Your response must follow this exact format:
1. Query Analysis:
   - Validate the query against DuckDB syntax rules
   - Check column names and quotes
   - Identify potential performance issues
   - Note any safety concerns

2. Validated/Corrected query with explanation:
   ```sql
   -- Purpose: Clear description of what this query does
   -- Changes Made: List any corrections or improvements
   -- Strategy: Explanation of the query approach
   SELECT ...
   ```

3. Any alternative queries with explanation:
   ```sql
   -- Purpose: Clear description of what this query does
   -- Differences: How this differs from the original
   -- Benefits: Why this might be better
   SELECT ...
   ```

4. Brief conclusion with:
   - Summary of key improvements
   - Performance considerations
   - Suggestions for query optimization"""

            # Get LLM response
            llm_response = self._call_llm(
                messages=[{"role": "user", "content": f"Validate and improve this SQL query:\n\n{query}"}],
                system_prompt=validation_prompt
            )
            
            if not llm_response:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Failed to validate SQL query. Please try again.",
                            message_type=MessageType.ERROR
                        )
                    ]
                )

            try:
                # Process the response to add query IDs to SQL blocks
                processed_response = llm_response
                previous_id = None  # Track the previous ID for alternatives
                
                # Find all SQL blocks using only triple backtick format
                sql_blocks = list(re.finditer(r'```sql\s*(.*?)\s*```', llm_response, re.DOTALL))
                
                for i, match in enumerate(sql_blocks):
                    sql_block = match.group(1).strip()  # Get just the SQL content
                    
                    # Generate query ID - first one is original, rest are alternatives
                    if i == 0:
                        query_id = PreviewIdentifier.create_id(prefix="nmdc_enhanced_query")
                        previous_id = query_id
                    else:
                        query_id = PreviewIdentifier.create_id(previous_id=previous_id)
                        previous_id = query_id
                    
                    # Reconstruct the SQL block without any stray backticks
                    clean_sql_block = f"```sql\n{sql_block}\n```"
                    
                    # Add the ID and execution instruction after the SQL block
                    replacement = f"{clean_sql_block}\nQuery ID: {query_id}\nTo execute this query, run: `nmdc_enhanced.query {query_id}`\n"
                    processed_response = processed_response.replace(match.group(0), replacement)
                
                # Return the processed response with query suggestions
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=processed_response,
                            message_type=MessageType.SUGGESTION
                        )
                    ]
                )
                
            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                logger.error(traceback.format_exc())
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Error processing SQL validation: {str(e)}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
        except Exception as e:
            logger.error(f"Error handling structured query: {str(e)}")
            logger.error(traceback.format_exc())
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error validating SQL query: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )

    def _prepare_llm_context(
        self,
        message: str,
        chat_history: List[Dict],
        domain_context: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
        """Prepare context for LLM processing.
        
        Args:
            message: Current user message
            chat_history: List of previous chat messages
            domain_context: Service-specific context
            
        Returns:
            Tuple of (system_prompt, context_messages, token_limits)
        """
        # Get data context from data manager
        data_context = self.data_manager.get_dataframe_context() or {}
        
        # Build system prompt
        system_prompt = f"""You are an AI assistant helping with queries for the National Microbiome Data Collaborative (NMDC) database.

Available Data:
- Studies: {data_context.get('studies_count', 0):,}
- Biosamples: {data_context.get('biosamples_count', 0):,}
- Unified rows: {data_context.get('unified_rows', 0):,}

Key searchable fields:
{chr(10).join(f"- {col}: {desc}" for col, desc in data_context.get('column_descriptions', {}).items())}

Geographic Coverage:
{self._format_geographic_coverage(data_context.get('geographic_coverage', {}))}

Your task is to help users query and analyze this data. You can:
1. Suggest specific queries based on user needs
2. Explain data relationships and patterns
3. Help interpret query results
4. Provide insights about the data

Current command type: {domain_context.get('command_type', 'natural_language_query')}
"""
        
        # Build context messages from chat history
        context_messages = []
        
        # Add relevant chat history (last 5 messages)
        for msg in chat_history[-5:]:
            if msg.get('content'):
                context_messages.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', '')
                })
        
        # Add current message
        context_messages.append({
            'role': 'user',
            'content': message
        })
        
        # Set token limits
        token_limits = {
            'max_tokens': 4000,  # Maximum response length
            'reserved_tokens': 1000,  # Reserved for system prompt and context
            'max_history_tokens': 2000  # Maximum tokens for chat history
        }
        
        return system_prompt, context_messages, token_limits
    
    def _format_geographic_coverage(self, geo_coverage: Dict[str, Any]) -> str:
        """Format geographic coverage information.
        
        Args:
            geo_coverage: Dictionary containing geographic coverage data
            
        Returns:
            Formatted string describing geographic coverage
        """
        if not geo_coverage:
            return "No geographic coverage information available"
            
        lat_range = geo_coverage.get('latitude_range', {})
        lon_range = geo_coverage.get('longitude_range', {})
        
        if not lat_range or not lon_range:
            return "Incomplete geographic coverage information"
            
        return f"""- Latitude: {lat_range.get('min', 'N/A'):.2f}° to {lat_range.get('max', 'N/A'):.2f}°
- Longitude: {lon_range.get('min', 'N/A'):.2f}° to {lon_range.get('max', 'N/A'):.2f}°"""

    def _process_response(self, response: str) -> str:
        """Process and validate the LLM response.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Processed and validated response
        """
        try:
            # Clean up response
            response = response.strip()
            
            # Validate response is not empty
            if not response:
                return "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
            
            # Check for error indicators
            if any(term in response.lower() for term in ['error', 'exception', 'failed', 'unable to']):
                logger.warning(f"LLM response contains error indicators: {response[:200]}...")
            
            # Format response for better readability
            formatted_response = []
            
            # Split into paragraphs
            paragraphs = response.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # Add bullet points if needed
                if para.startswith('- '):
                    formatted_response.append(para)
                else:
                    formatted_response.append(para)
            
            return '\n\n'.join(formatted_response)
            
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}")
            logger.error(traceback.format_exc())
            return "I apologize, but I encountered an error while processing the response. Please try again."

    def _handle_entities(self) -> ServiceResponse:
        """Handle entities command to show NMDC database statistics.
        
        Returns:
            ServiceResponse with formatted statistics
        """
        try:
            # Fetch stats from NMDC API
            response = requests.get("https://data.microbiomedata.org/api/stats")
            response.raise_for_status()
            stats = response.json()
            
            # Format data size for readability
            data_size_gb = stats["data_size"] / (1024**3)  # Convert bytes to GB
            
            # Create markdown content with title
            content = "## NMDC Entity Statistics\n\n```\n"
            
            # Top border
            content += "┌─────────────────────────────────────────────────┬──────────────┐\n"
            # Header - right align Count to match values
            content += "│ Category                                        │       Count  │\n"
            # Header separator
            content += "├─────────────────────────────────────────────────┼──────────────┤\n"
            
            # Format each row with proper spacing
            rows = [
                ("Studies", f"{stats['studies']:,}"),
                ("Locations", f"{stats['locations']:,}"),
                ("Habitats", f"{stats['habitats']:,}"),
                ("Total Data Size", f"{data_size_gb:.2f} GB"),
                ("Metagenomes", f"{stats['metagenomes']:,}"),
                ("Metatranscriptomes", f"{stats['metatranscriptomes']:,}"),
                ("Proteomics", f"{stats['proteomics']:,}"),
                ("Metabolomics", f"{stats['metabolomics']:,}"),
                ("Lipidomics", f"{stats['lipodomics']:,}"),
                ("Organic Matter Characterization", f"{stats['organic_matter_characterization']:,}")
            ]
            
            # Add each row with proper spacing
            for category, count in rows:
                # Left-align category with proper padding
                padded_category = f"│ {category}".ljust(50)
                # Right-align count with proper spacing
                padded_count = f"{count}".rjust(10)
                content += f"{padded_category}│ {padded_count} │\n"
            
            # Bottom border
            content += "└─────────────────────────────────────────────────┴──────────────┘\n"
            content += "```"
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=content,
                        message_type=MessageType.RESULT
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Error handling entities command: {str(e)}")
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error retrieving NMDC statistics: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )