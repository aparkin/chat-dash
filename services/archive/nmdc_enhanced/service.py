"""
NMDC Enhanced Service for ChatDash.

This service provides advanced data discovery and integration capabilities
for the National Microbiome Data Collaborative (NMDC) API.
"""

import logging
import json
import asyncio
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Set
import pandas as pd
import datetime
from copy import deepcopy

from services.base import ChatService, ServiceResponse, ServiceMessage, MessageType, PreviewIdentifier
from .schema_manager import SchemaManager
from .api_client import NMDCApiClient
from .data_integrator import DataIntegrator
from .data_object_processor import DataObjectProcessor
from .result_store import ResultStore
from services.llm_service import LLMServiceMixin  # Import the LLM mixin

# Configure logging
logger = logging.getLogger(__name__)

class NMDCEnhancedService(ChatService, LLMServiceMixin):  # Add LLMServiceMixin
    """Enhanced service for NMDC data discovery and integration.
    
    This service provides:
    1. Schema introspection (entities, fields, etc.)
    2. Natural language query processing
    3. Data integration across multiple entity types
    4. Specialized views (e.g., soil layer taxonomic distribution)
    5. Environmental data visualization
    """
    
    def __init__(self, result_store=None):
        """Initialize the NMDC Enhanced Service.
        
        Args:
            result_store: Optional external result store to use.
                         If None, will create its own internal store.
        """
        super().__init__(name="nmdc_enhanced")
        LLMServiceMixin.__init__(self, service_name="nmdc_enhanced")  # Initialize LLM mixin
        
        # Initialize components
        self.schema_manager = SchemaManager(preload_all=True)
        self.api_client = NMDCApiClient()
        self.data_integrator = DataIntegrator(self.api_client, self.schema_manager)
        self.data_object_processor = DataObjectProcessor(self.api_client)
        
        # Use provided result store or create internal one
        self.result_store = result_store if result_store is not None else ResultStore()
        
        # Query tracking (using standard service pattern)
        self.active_queries = {}
        self.last_query_id = None
        self.last_result_id = None
        
        # Register prefixes for PreviewIdentifier - use service-specific prefixes
        try:
            PreviewIdentifier.register_prefix("nmdc_enhanced_query")
        except ValueError:
            # Prefix already registered, which is fine
            pass
            
        try:
            PreviewIdentifier.register_prefix("nmdc_enhanced_result")
        except ValueError:
            # Prefix already registered, which is fine
            pass
        
        # Register commands
        self.commands = {
            "help": self._handle_help,
            "about": self._handle_about,
            "entities": self._handle_entities,
            "fields": self._handle_fields,
            "query": self._handle_query_execution,
            "search": self._handle_query_execution,
            "soil_taxonomy": self._handle_soil_layer_taxonomic_integration,
            "environment": self._handle_environment_visualization,
            "dataset_conversion": self._handle_dataset_conversion
        }
        
        # Entity type mappings for NL processing
        self.entity_type_mappings = {
            "study": ["study", "studies", "project", "projects"],
            "biosample": ["biosample", "biosamples", "sample", "samples"],
            "data_object": ["data object", "data objects", "file", "files", "data", "dataset", "datasets"]
        }
        
        logger.info("NMDC Enhanced Service initialized")
    
    def can_handle(self, message: str) -> bool:
        """Determine if this service can handle the message.
        
        Args:
            message: User message
            
        Returns:
            True if this service can handle the message, False otherwise
        """
        message = message.lower().strip()
        
        # Check for direct command pattern (both formats for backward compatibility)
        if message.startswith("nmdc_enhanced.") or message.startswith("nmdc_enhanced:"):
            return True
            
        # Check for "tell me about" pattern
        if "tell me about nmdc_enhanced" in message or "about nmdc_enhanced" in message:
            return True
            
        # Check for natural language query pattern with service mention
        if "nmdc_enhanced" in message and any(term in message for term in [
            "show", "find", "get", "query", "search", "sample", "samples", "biosample", "biosamples",
            "locate", "discover", "soil", "water", "sediment", "forest", "carbon", "nitrogen", "ph"
        ]):
            return True
            
        # Check for dataset conversion command
        if re.match(r'^convert\s+(nmdc_enhanced_result_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message):
            return True
            
        return False
    
    def parse_request(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse service request to extract query parameters.
        
        Args:
            message: User message to parse
            
        Returns:
            Dictionary of parsed parameters or None if not applicable
        """
        message = message.lower().strip()
        
        # Check for command pattern with dot (standard pattern)
        if message.startswith("nmdc_enhanced."):
            # Extract command and arguments
            parts = message[len("nmdc_enhanced."):].strip().split(" ", 1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            return {
                "command": command,
                "args": args,
                "original_message": message
            }
            
        # Check for command pattern with colon (for sample queries)
        if message.startswith("nmdc_enhanced:"):
            args = message[len("nmdc_enhanced:"):].strip()
            
            # This is a query building request
            return {
                "command": "_handle_query_building",
                "args": args,
                "original_message": message
            }
        
        # Handle "tell me about" pattern
        if "tell me about nmdc_enhanced" in message or "about nmdc_enhanced" in message:
            return {
                "command": "about",
                "args": "",
                "original_message": message
            }
        
        # Handle natural language sample queries
        if "nmdc_enhanced" in message and any(term in message for term in [
            "show", "find", "get", "query", "search", "sample", "samples", "biosample", "biosamples",
            "locate", "discover", "soil", "water", "sediment", "forest", "carbon", "nitrogen", "ph"
        ]):
            return {
                "command": "_handle_query_building",
                "args": message,
                "original_message": message
            }
            
        # Handle dataset conversion command
        match = re.match(r'^convert\s+(nmdc_enhanced_result_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message)
        if match:
            return {
                "command": "_handle_dataset_conversion",
                "args": match.group(1),
                "original_message": message
            }
        
        # Not a request for this service
        return None
    
    def execute(self, request: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute service request.
        
        Args:
            request: Parsed request dictionary
            context: Context dictionary with additional information
            
        Returns:
            Service response
        """
        command = request.get("command", "")
        args = request.get("args", "")
        
        # Get handler for command
        handler = self.commands.get(command)
        
        # If command is a direct method name (like _handle_query_building), use that
        if not handler and command.startswith("_handle_"):
            handler_name = command
            if hasattr(self, handler_name) and callable(getattr(self, handler_name)):
                handler = getattr(self, handler_name)
        
        # If still no handler, default to help
        if not handler:
            logger.info(f"Command '{command}' not found, defaulting to help")
            return self._handle_help(args, request)
        
        # Execute handler
        try:
            if asyncio.iscoroutinefunction(handler):
                # For async handlers, use asyncio.run
                return asyncio.run(handler(args, request))
            return handler(args, request)
        except Exception as e:
            logger.error(f"Error executing command: {str(e)}")
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Error executing command: {str(e)}",
                    message_type=MessageType.ERROR
                )
            ])
    
    def _handle_help(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle help command.
        
        Args:
            args: Command arguments
            parsed: Parsed request
            
        Returns:
            Service response
        """
        help_text = """
## NMDC Enhanced Service

The NMDC Enhanced Service provides data discovery and integration capabilities for the National Microbiome Data Collaborative (NMDC) database, with a focus on sample data and physical measurements.

### Sample Query Pattern:
1. **Build queries** using `nmdc_enhanced: [your question]` to create a sample query
2. **Execute queries** using `nmdc_enhanced.query [query_id]` with the ID provided from the query building step

### Available Commands:
- `nmdc_enhanced.help` - Show this help message
- `nmdc_enhanced.about` - Show information about the service
- `nmdc_enhanced.entities` - List available entity types in the NMDC database
- `nmdc_enhanced.fields [entity_type]` - List searchable fields for an entity type
- `nmdc_enhanced: [natural_language_query]` - Build a sample query (e.g., `nmdc_enhanced: find soil samples from Washington state`)
- `nmdc_enhanced.query [query_id]` - Execute a previously built query using its ID
- `nmdc_enhanced.environment` - Show environmental hierarchy and distribution in NMDC data

### Examples:
1. **Build a query**: `nmdc_enhanced: samples with carbon measurements from forest environments`
2. **Execute the query**: `nmdc_enhanced.query 1` (use the query ID returned from the build step)
3. **Execute last query**: `nmdc_enhanced.query last` (executes the most recently built query)

The results will include sample metadata, associated study information, physical measurements where available, and counts of related data objects by type.
"""
        return ServiceResponse(messages=[
            ServiceMessage(
                service=self.name,
                content=help_text,
                message_type=MessageType.INFO
            )
        ])
    
    def _handle_about(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle about command.
        
        Args:
            args: Command arguments
            parsed: Parsed request
            
        Returns:
            Service response
        """
        about_text = """
## NMDC Enhanced Service

The NMDC Enhanced Service provides advanced capabilities for data discovery and integration with the NMDC API. It enhances the base NMDC service with:

1. **Improved Schema Introspection:**
   - Automatic loading of entity types and fields
   - Relationship tracking between entities
   - Field statistics and common values

2. **Advanced Query Building:**
   - Support for complex queries with multiple conditions
   - Automatic handling of relationships between entities
   - Query validation against schema

3. **Natural Language Processing:**
   - Convert natural language questions to structured queries
   - Identify entities, fields, and relationships from text
   - Extract query parameters from unstructured text

4. **Data Integration:**
   - Combine data from multiple NMDC entities
   - Create integrated views of related data
   - Generate specialized visualizations and analyses

5. **Specialized Workflows:**
   - Soil layer taxonomic integration
   - Environmental data visualization
   - Study-specific data integration

Use `nmdc_enhanced:help` to see available commands.
"""
        is_initialized = self.schema_manager.is_initialized()
        
        # Add initialization status
        if is_initialized:
            about_text += "\n\n**Service Status:** Fully initialized and ready for queries."
            
            # Add some basic stats
            try:
                study_count = self.schema_manager.get_entity_count("study")
                biosample_count = self.schema_manager.get_entity_count("biosample")
                dataobject_count = self.schema_manager.get_entity_count("data_object")
                
                about_text += f"\n\n**Data Available:**\n"
                about_text += f"- Studies: {study_count:,}\n"
                about_text += f"- Biosamples: {biosample_count:,}\n"
                about_text += f"- Data Objects: {dataobject_count:,}"
            except Exception as e:
                logger.error(f"Error getting entity counts: {str(e)}")
        else:
            about_text += "\n\n**Service Status:** Initialization in progress. Some functionality may be limited."
            
        return ServiceResponse(messages=[
            ServiceMessage(
                service=self.name,
                content=about_text,
                message_type=MessageType.INFO
            )
        ])
    
    def _handle_entities(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle entities command.
        
        Args:
            args: Command arguments
            parsed: Parsed request
            
        Returns:
            Service response with list of available entities
        """
        try:
            # Get entity types with counts
            entity_types = self.schema_manager.get_entity_types()
            
            content = "## Available NMDC Entity Types\n\n"
            content += "| Entity Type | Record Count |\n"
            content += "|-------------|-------------:|\n"
            
            for entity in sorted(entity_types):
                count = self.schema_manager.get_entity_count(entity)
                content += f"| {entity} | {count:,} |\n"
                
            content += "\n\nUse `nmdc_enhanced:fields [entity_type]` to see available fields for an entity type."
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=content,
                    message_type=MessageType.RESULT
                )
            ])
        except Exception as e:
            logger.error(f"Error handling entities command: {str(e)}")
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Error retrieving entity types: {str(e)}",
                    message_type=MessageType.ERROR
                )
            ])
    
    def _handle_fields(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle fields command.
        
        Args:
            args: Command arguments (entity type)
            parsed: Parsed request
            
        Returns:
            Service response with list of fields for an entity type
        """
        try:
            entity_type = args.strip().lower()
            
            if not entity_type:
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content="Please specify an entity type. Use `nmdc_enhanced:entities` to see available entity types.",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Get entity types to validate
            entity_types = self.schema_manager.get_entity_types()
            
            if entity_type not in [et.lower() for et in entity_types]:
                # Find the closest match
                matches = [et for et in entity_types if entity_type in et.lower()]
                suggestion = ""
                if matches:
                    suggestion = f"\n\nDid you mean one of these?\n" + "\n".join([f"- {m}" for m in matches])
                
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Entity type '{entity_type}' not found.{suggestion}\n\nUse `nmdc_enhanced:entities` to see available entity types.",
                        message_type=MessageType.WARNING
                    )
                ])
            
            # Find the correct case for the entity type
            entity_type = next(et for et in entity_types if et.lower() == entity_type)
            
            # Get searchable attributes
            attributes = self.schema_manager.get_searchable_attributes(entity_type)
            
            content = f"## Fields for {entity_type}\n\n"
            content += "| Field Name | Type | Population (% of records) |\n"
            content += "|------------|------|------------------------:|\n"
            
            entity_count = self.schema_manager.get_entity_count(entity_type)
            
            for attr in attributes:
                if entity_count > 0:
                    percent = (attr.count / entity_count) * 100
                    percent_str = f"{percent:.1f}%"
                else:
                    percent_str = "N/A"
                
                content += f"| {attr.name} | {attr.type} | {percent_str} ({attr.count:,}) |\n"
                
            content += f"\n\nUse these field names in queries for {entity_type} entities."
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=content,
                    message_type=MessageType.RESULT
                )
            ])
        except Exception as e:
            logger.error(f"Error handling fields command: {str(e)}")
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Error retrieving fields: {str(e)}",
                    message_type=MessageType.ERROR
                )
            ])
    
    async def _handle_data_integration(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle data integration command.
        
        Args:
            args: Command arguments (integration query)
            parsed: Parsed request
            
        Returns:
            Service response with integration results
        """
        try:
            query = args.strip()
            original_query = parsed.get("original_message", "")
            
            if not query:
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content="Please provide a query for data integration. Examples:\n"
                               "- `nmdc_enhanced.query find studies about soil microbiology`\n"
                               "- `nmdc_enhanced.query show biosamples from forest environments`\n"
                               "- `nmdc_enhanced.query get data objects with metagenome sequencing`",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Process query using simple pattern matching
            logger.info(f"Processing integration query: {query}")
            
            # Extract entity type and conditions
            entity_insights = self._extract_entity_and_conditions(query)
            entity_type = entity_insights["entity_type"]
            conditions = entity_insights["conditions"]
            
            # If no conditions found, provide guidance
            if not conditions:
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"I couldn't extract specific search criteria from your query. Please try being more specific about what you're looking for in {entity_type} records.\n\n"
                               f"For example:\n"
                               f"- For studies: mention topics like 'soil microbiology' or specific research areas\n"
                               f"- For biosamples: mention environments like 'soil', 'marine', or specific locations\n"
                               f"- For data objects: mention types like 'metagenome' or 'amplicon sequencing'",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Construct structured query
            structured_query = {
                "entity_type": entity_type,
                "conditions": []
            }
            
            # Add conditions
            for field, value in conditions.items():
                structured_query["conditions"].append({
                    "field": field,
                    "operator": "contains",
                    "value": value
                })
            
            logger.info(f"Constructed structured query: {structured_query}")
            
            # Check if API client is accessible
            try:
                # Check if entity type is currently supported by the API
                supported_entities = await self.api_client.get_supported_entities()
                if entity_type not in supported_entities:
                    return ServiceResponse(messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"The entity type '{entity_type}' is not currently supported by the NMDC API.\n\n"
                                   f"Currently supported entity types are: {', '.join(supported_entities)}\n\n"
                                   f"Please try your query with one of the supported entity types.",
                            message_type=MessageType.WARNING
                        )
                    ])
                                
                # Execute integration query
                result_df = await self.data_integrator.integrate(
                    entity_type=entity_type,
                    conditions=structured_query["conditions"]
                )
                
                if result_df is None or result_df.empty:
                    # Format condition pairs
                    conditions_text = ", ".join([f"{cond['field']}={cond['value']}" for cond in structured_query['conditions']])
                    
                    return ServiceResponse(messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"No results found for your query searching for {entity_type} with criteria: {conditions_text}.\n\n"
                                   "Try broadening your search or using different terms. You can also try:\n"
                                   f"- `nmdc_enhanced.entities` to see available entity types\n"
                                   f"- `nmdc_enhanced.fields {entity_type}` to see available fields for {entity_type}",
                            message_type=MessageType.INFO
                        )
                    ])
                
                # Generate unique ID for result
                query_id = str(uuid.uuid4())
                
                # Store result in result store
                self.result_store.store(f"query_{query_id}", result_df)
                
                # Generate preview of results
                preview = self._generate_result_preview(result_df, entity_type)
                
                # Prepare response
                response_content = f"## Integration Results\n\n"
                response_content += f"Query: {original_query}\n\n"
                response_content += f"Found {len(result_df)} {entity_type} records.\n\n"
                response_content += preview
                response_content += f"\n\n*Result ID: {query_id}*"
                
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content=response_content,
                        message_type=MessageType.RESULT
                    )
                ])
            except AttributeError:
                # If get_supported_entities is not implemented yet
                logger.warning("get_supported_entities not implemented, proceeding with query")
                # Fallback to standard behavior
            except Exception as api_error:
                logger.error(f"API error during integration: {str(api_error)}")
                # Check if it's a 404 error
                if "404" in str(api_error):
                    return ServiceResponse(messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"The NMDC API endpoint for {entity_type} could not be accessed. This entity type might not be available in the current API version.\n\n"
                                   f"The NMDC Enhanced Service relies on the live NMDC API, which may have limited availability. "
                                   f"Please try using the general `nmdc` service for broader information about microbiome data.",
                            message_type=MessageType.ERROR
                        )
                    ])
                else:
                    # General API error
                    return ServiceResponse(messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Error accessing the NMDC API: {str(api_error)}\n\n"
                                   "The NMDC API might be temporarily unavailable. You can try:\n"
                                   "1. Using a different entity type in your query\n"
                                   "2. Using the standard `nmdc` service for general information\n"
                                   "3. Trying again later when the API may be available",
                            message_type=MessageType.ERROR
                        )
                    ])
            
        except Exception as e:
            logger.error(f"Error handling data integration: {str(e)}")
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Error processing integration query: {str(e)}\n\n"
                           "This could be due to limitations in the current NMDC API or how your query is being processed. "
                           "Please try a different query format or use the standard `nmdc` service for general information.",
                    message_type=MessageType.ERROR
                )
            ])
    
    async def _handle_soil_layer_taxonomic_integration(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle soil layer taxonomic integration command.
        
        Args:
            args: Command arguments
            parsed: Parsed request
            
        Returns:
            Service response with soil layer taxonomic integration results
        """
        try:
            query = args.strip()
            
            # Try to extract study ID from query
            study_id_match = re.search(r"study\s*id[:\s]+(\S+)", query, re.IGNORECASE)
            study_id = study_id_match.group(1) if study_id_match else None
            
            if not study_id:
                # If no study ID was provided, return instructions
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content="Please provide a study ID for soil layer taxonomic integration.\n\n"
                               "Example: `nmdc_enhanced:soil_taxonomy study id: gold:Gs0134277`",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Create dataset for soil layer taxonomic distribution
            result_df = await self.data_integrator.get_soil_layer_taxonomic_distribution(study_id)
            
            if result_df is None or result_df.empty:
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"No soil layer taxonomic data found for study ID: {study_id}",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Generate unique ID for result
            query_id = str(uuid.uuid4())
            
            # Store result in result store
            self.result_store.store(f"soil_taxonomy_{query_id}", result_df)
            
            # Generate preview of results
            preview = self._generate_result_preview(result_df, "soil_layer_taxonomic")
            
            # Prepare response
            response_content = f"## Soil Layer Taxonomic Distribution\n\n"
            response_content += f"Study ID: {study_id}\n\n"
            response_content += f"Found {len(result_df)} taxonomic distribution records.\n\n"
            response_content += preview
            response_content += f"\n\n*Result ID: {query_id}*"
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=response_content,
                    message_type=MessageType.RESULT
                )
            ])
            
        except Exception as e:
            logger.error(f"Error handling soil layer taxonomic integration: {str(e)}")
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Error processing soil layer taxonomic integration: {str(e)}",
                    message_type=MessageType.ERROR
                )
            ])
    
    async def _handle_environment_visualization(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle environment visualization command.
        
        Args:
            args: Command arguments
            parsed: Parsed request
            
        Returns:
            Service response with environment visualization
        """
        try:
            # Get environmental hierarchy
            env_hierarchy = self.schema_manager.get_env_hierarchy()
            
            if not env_hierarchy:
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content="Environmental hierarchy data is not available from the NMDC API.",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Format hierarchy for display
            content = "## NMDC Environmental Hierarchy\n\n"
            content += "| Environment Type | Count |\n"
            content += "|-----------------|------:|\n"
            
            # Extract top-level categories
            top_levels = {}
            
            # Handle different possible formats of env_hierarchy
            nodes = []
            # Case 1: env_hierarchy is a dict with a "nodes" key
            if isinstance(env_hierarchy, dict) and "nodes" in env_hierarchy:
                nodes = env_hierarchy["nodes"]
            # Case 2: env_hierarchy is a list of nodes directly
            elif isinstance(env_hierarchy, list):
                nodes = env_hierarchy
            # Case 3: env_hierarchy has some other format
            else:
                logger.warning(f"Environmental hierarchy has unexpected format: {type(env_hierarchy)}")
                # Try to convert to string and log for debugging
                logger.debug(f"Environment hierarchy content: {str(env_hierarchy)[:500]}...")
                
            # Process nodes if they're a list
            if isinstance(nodes, list):
                for node in nodes:
                    if isinstance(node, dict) and "name" in node and "value" in node:
                        top_levels[node["name"]] = node["value"]
            else:
                logger.warning(f"Environmental hierarchy nodes is not a list: {type(nodes)}")
            
            # If no valid data found, return informative message
            if not top_levels:
                # Try to provide more detailed information about what was received
                env_type = type(env_hierarchy).__name__
                content_preview = str(env_hierarchy)[:100] + "..." if len(str(env_hierarchy)) > 100 else str(env_hierarchy)
                
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"No environmental hierarchy data could be extracted from the NMDC API response.\n\n"
                               f"The API returned data of type: {env_type}\n\n"
                               f"This service may need to be updated to handle the current API format.",
                        message_type=MessageType.INFO
                    )
                ])
            
            # Sort by count (descending)
            for name, count in sorted(top_levels.items(), key=lambda x: x[1], reverse=True):
                content += f"| {name} | {count:,} |\n"
            
            # Add note about geospatial data
            content += "\n\nGeospatial distribution is also available for NMDC samples."
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=content,
                    message_type=MessageType.RESULT
                )
            ])
            
        except Exception as e:
            logger.error(f"Error handling environment visualization: {str(e)}")
            # Include the actual error and stack trace in the response
            import traceback
            error_detail = traceback.format_exc()
            logger.debug(f"Environment visualization error details: {error_detail}")
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Unable to retrieve environmental hierarchy from NMDC API. Error: {str(e)}\n\n"
                           f"The NMDC Enhanced Service relies on the NMDC API structure, which may have changed. "
                           f"Please try using the standard `nmdc` service for general information.",
                    message_type=MessageType.ERROR
                )
            ])
    
    def _extract_entity_and_conditions(self, query: str) -> Dict[str, Any]:
        """Extract entity type and conditions from a natural language query.
        
        This is a simplified replacement for spaCy-based NL processor.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary with entity_type and conditions
        """
        query = query.lower()
        
        # Default entity type
        entity_type = "study"
        conditions = {}
        
        # Try to determine entity type from query
        for et, terms in self.entity_type_mappings.items():
            if any(term in query for term in terms):
                entity_type = et
                break
                
        # Extract conditions based on entity type
        if entity_type == "study":
            # Look for study-specific patterns
            
            # Principal investigator
            pi_match = re.search(r"(?:by|from)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", query, re.IGNORECASE)
            if pi_match:
                conditions["principal_investigator_name"] = pi_match.group(1)
            
            # Topics/keywords
            topic_match = re.search(r"(?:about|on|related to|concerning|involving)\s+([^\.,:;]+)", query)
            if topic_match:
                conditions["description"] = topic_match.group(1).strip()
            
        elif entity_type == "biosample":
            # Look for biosample-specific patterns
            
            # Environment terms
            env_terms = [
                ("soil", "soil"), 
                ("marine", "marine"), 
                ("freshwater", "freshwater"),
                ("forest", "forest"), 
                ("wetland", "wetland"), 
                ("aquatic", "aquatic")
            ]
            
            for term, value in env_terms:
                if term in query:
                    conditions["env_medium"] = value
                    break
            
            # Location
            location_match = re.search(r"(?:from|in|at|located in|collected from)\s+([^\.,:;]+)", query)
            if location_match:
                location = location_match.group(1).strip()
                
                # Check if location contains environment term
                if not conditions.get("env_medium"):
                    for term, value in env_terms:
                        if term in location.lower():
                            conditions["env_medium"] = value
                            break
                    else:
                        # If no environment term, use as general location
                        conditions["description"] = location
                        
        elif entity_type == "data_object":
            # Look for data object-specific patterns
            
            # Data types
            data_types = [
                ("metagenome", "metagenome"),
                ("metaproteome", "metaproteome"),
                ("metatranscriptome", "metatranscriptome"),
                ("amplicon", "amplicon"),
                ("16s", "16S")
            ]
            
            for term, value in data_types:
                if term in query:
                    conditions["type"] = value
                    break
                    
            # File type
            file_type_match = re.search(r"(?:type|format|kind of)\s+([^\.,:;]+)", query)
            if file_type_match:
                conditions["file_type"] = file_type_match.group(1).strip()
        
        # If no specific conditions found, use the query as a general search
        if not conditions:
            # Remove common words
            clean_query = re.sub(r'\b(?:show|find|get|query|search|integrate|me|about|for|in|the|with|and|or)\b', '', query)
            clean_query = re.sub(r'\bnmdc_enhanced\b', '', clean_query)
            clean_query = clean_query.strip()
            
            if clean_query:
                if entity_type == "study":
                    conditions["description"] = clean_query
                elif entity_type == "biosample":
                    conditions["description"] = clean_query
                elif entity_type == "data_object":
                    conditions["description"] = clean_query
        
        return {
            "entity_type": entity_type,
            "conditions": conditions
        }
    
    def _generate_result_preview(self, df, entity_type: str) -> str:
        """Generate preview of result DataFrame.
        
        Args:
            df: Result DataFrame
            entity_type: Type of entity in result
            
        Returns:
            Markdown formatted preview
        """
        if df is None or df.empty:
            return "No results to preview."
            
        # Create a copy to avoid modifying the original
        preview_df = df.head(5).copy()
        
        # For dataframes with many columns, select a subset of important columns
        important_columns = self._get_important_columns(entity_type, list(preview_df.columns))
        
        if important_columns:
            preview_df = preview_df[important_columns]
        
        # Process any complex JSON structures in the dataframe
        preview_df = self._simplify_dataframe_values(preview_df)
        
        # Format preview
        try:
            # Adjust column width by truncating long strings
            for col in preview_df.columns:
                if preview_df[col].dtype == 'object':
                    preview_df[col] = preview_df[col].astype(str).apply(
                        lambda x: x[:40] + '...' if len(x) > 40 else x
                    )
            
            # Convert to markdown table - already inside a code block so no need for header
            md_table = preview_df.to_markdown(index=False)
            return md_table
            
        except Exception as e:
            logger.error(f"Error formatting markdown table: {str(e)}")
            # Fallback to a simpler representation
            preview = "Error creating formatted table. Showing simplified view:\n\n"
            for i, row in preview_df.head(5).iterrows():
                preview += f"Record {i+1}:\n"
                for col in preview_df.columns:
                    preview += f"  {col}: {row[col]}\n"
                preview += "\n"
                
            return preview
    
    def _simplify_dataframe_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simplify complex values in a DataFrame.
        
        This method extracts meaningful information from complex structures
        like nested JSON, particularly focusing on NMDC-specific formats.
        
        Args:
            df: DataFrame to simplify
            
        Returns:
            DataFrame with simplified values
        """
        result_df = df.copy()
        
        # Process each column
        for col in result_df.columns:
            # Skip processing if column has no values
            if result_df[col].isna().all():
                continue
                
            # Check if column contains dictionaries/complex objects
            if result_df[col].notna().any():
                first_value = result_df[col].iloc[0]
                
                # Process NMDC controlled vocabulary terms
                if isinstance(first_value, dict) and 'has_raw_value' in first_value:
                    # Create a new column with just the raw values
                    new_col = f"{col}_value"
                    
                    # Extract the raw values or term names, whichever is more informative
                    result_df[new_col] = result_df[col].apply(
                        lambda x: self._extract_term_value(x) if isinstance(x, dict) else x
                    )
                    
                    # Replace the original column
                    result_df[col] = result_df[new_col]
                    result_df = result_df.drop(columns=[new_col])
                
                # Process lists
                elif isinstance(first_value, list):
                    # Combine list elements into a string
                    result_df[col] = result_df[col].apply(
                        lambda x: ", ".join(str(item) for item in x) if isinstance(x, list) else x
                    )
        
        return result_df
    
    def _extract_term_value(self, term_obj):
        """Extract the most meaningful value from an NMDC term object.
        
        Args:
            term_obj: NMDC term object
            
        Returns:
            Extracted value
        """
        if not isinstance(term_obj, dict):
            return str(term_obj)
            
        # Get the raw value if available
        if 'has_raw_value' in term_obj:
            raw_value = term_obj['has_raw_value']
            # Clean up raw value by removing ID patterns and underscores
            raw_value = re.sub(r'_+', '', raw_value)  # Remove underscore sequences
            raw_value = re.sub(r'\s+ENVO:\d+', '', raw_value)  # Remove ENVO IDs
            return raw_value.strip()
            
        # If there's a term with a name, use that
        if 'term' in term_obj and isinstance(term_obj['term'], dict) and 'name' in term_obj['term']:
            return term_obj['term']['name']
            
        # Fallback to returning the whole object as a string
        return str(term_obj)
    
    def _get_important_columns(self, entity_type: str, available_columns: List[str]) -> List[str]:
        """Get list of important columns for preview based on entity type.
        
        Args:
            entity_type: Type of entity
            available_columns: List of available columns
            
        Returns:
            List of important column names
        """
        # Define important columns for common entity types
        important_columns_map = {
            "study": ["id", "name", "description", "study_category"],
            "biosample": ["id", "name", "env_broad_scale", "env_local_scale", "env_medium"],
            "data_object": ["id", "name", "description", "file_size_bytes"],
            "soil_layer_taxonomic": ["depth", "taxonomy", "abundance", "study_id"]
        }
        
        # Get important columns for entity type
        important_columns = important_columns_map.get(entity_type, [])
        
        # Filter to only include columns that exist in the DataFrame
        important_columns = [col for col in important_columns if col in available_columns]
        
        # If no important columns are available, return all columns up to a limit
        if not important_columns:
            return list(available_columns)[:5]
        
        return important_columns
    
    def get_help_text(self) -> str:
        """Get help text for this service.
        
        Returns:
            Help text
        """
        return """
## NMDC Enhanced Service

The NMDC Enhanced Service provides advanced sample discovery capabilities for NMDC data.

### Query Pattern:
- **Build**: `nmdc_enhanced: [your question]` to formulate a sample query
- **Execute**: `nmdc_enhanced.query [query_id]` to run a previously built query

### Example Queries:
- `nmdc_enhanced: find samples from Washington state`
- `nmdc_enhanced: samples with carbon measurements from forest environments`
- `nmdc_enhanced: soil samples from the Wrighton lab`

### Other Commands:
- `nmdc_enhanced.help` - Show help information
- `nmdc_enhanced.about` - Show service information
- `nmdc_enhanced.entities` - List available entity types
- `nmdc_enhanced.fields [entity_type]` - List fields for an entity type
- `nmdc_enhanced.soil_taxonomy [study_id]` - Get soil layer taxonomic data
- `nmdc_enhanced.environment` - Show environmental hierarchy

### Examples:
- `nmdc_enhanced: I'm looking for studies about soil microbiology`
- `nmdc_enhanced.query abc123de` (using a query ID from a previous build step)
- `nmdc_enhanced.fields biosample`
"""
    
    def get_llm_prompt_addition(self) -> str:
        """Get text to add to the LLM prompt explaining this service.
        
        Returns:
            Text for LLM prompt
        """
        return """
The NMDC Enhanced Service provides sample discovery capabilities for the National Microbiome Data Collaborative (NMDC) database. The service follows a two-step query pattern:

1. Build sample queries using `nmdc_enhanced: [your question]` which will analyze the request and create a structured query with a unique ID. For example:
   - `nmdc_enhanced: find soil samples from Washington state`
   - `nmdc_enhanced: samples with carbon measurements from forest environments`
   - `nmdc_enhanced: samples from the Wrighton lab`

2. Execute queries using `nmdc_enhanced.query [query_id]` with the ID provided from the building step.

The service will return enriched sample data including:
- Sample metadata (ID, collection date, location, etc.)
- Associated study information
- Physical measurements when available
- Counts of related data objects by type

When users ask questions about microbiome samples or data, suggest using the NMDC Enhanced Service to find relevant samples.
"""
    
    def _handle_query_building(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle collaborative query building with the user.
        
        This method builds a structured query focused on samples with physical data
        and displays it in a standardized format for user review.
        
        Args:
            args: User's natural language query
            parsed: Parsed request
            
        Returns:
            Service response with structured query
        """
        query_text = args.strip()
        original_message = parsed.get("original_message", "")
        
        if not query_text:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content="Please specify what kind of samples you're looking for in the NMDC database. For example:\n"
                           "- `nmdc_enhanced: find samples from Washington state`\n"
                           "- `nmdc_enhanced: samples with carbon measurements from the west coast`\n"
                           "- `nmdc_enhanced: samples from Wrighton lab`\n"
                           "- `nmdc_enhanced: soil samples`",
                    message_type=MessageType.INFO
                )
            ])
        
        # Extract search parameters from query
        search_params = self._extract_sample_search_parameters(query_text)
        
        # Create a properly structured query in the format expected by our API client
        structured_query = {
            "entity_type": "biosample",
            "filters": []
        }
        
        # Get the schema manager's field mappings for better field selection
        field_mappings = self.schema_manager.get_field_mappings()
        
        # Add location filter if provided
        if search_params.get("location"):
            location_value = search_params["location"]
            
            # Get field suggestions from schema manager
            location_fields = self.schema_manager.get_field_suggestions_for_value("location", location_value)
            
            if location_fields:
                # Use the first suggested field
                field_info = location_fields[0]
                structured_query["filters"].append({
                    "field": field_info["field"],
                    "operator": field_info["operator"],
                    "value": field_info["value"]
                })
            else:
                # Fallback to default geo_loc_name
                structured_query["filters"].append({
                    "field": "geo_loc_name",
                    "operator": "search",
                    "value": location_value
                })
        
        # Add environment filter if provided
        if search_params.get("environment"):
            env_value = search_params["environment"]
            
            # Get field suggestions from schema manager
            env_fields = self.schema_manager.get_field_suggestions_for_value("environment", env_value)
            
            if env_fields:
                # Use the first suggested field
                field_info = env_fields[0]
                structured_query["filters"].append({
                    "field": field_info["field"],
                    "operator": field_info["operator"],
                    "value": field_info["value"]
                })
            else:
                # Fallback to specific_ecosystem
                structured_query["filters"].append({
                    "field": "specific_ecosystem",
                    "operator": "search", 
                    "value": env_value
                })
        
        # Add researcher filter if provided  
        if search_params.get("researcher"):
            # Best field for researcher is principal_investigator_name
            structured_query["filters"].append({
                "field": "principal_investigator_name",
                "operator": "search",
                "value": search_params["researcher"]
            })
        
        # Add measurement filters if provided
        if search_params.get("measurements"):
            for measurement in search_params["measurements"]:
                # Get field suggestions from schema manager
                measurement_fields = self.schema_manager.get_field_suggestions_for_value("measurement", measurement)
                
                if measurement_fields:
                    # Use the first suggested field
                    field_info = measurement_fields[0]
                    structured_query["filters"].append({
                        "field": field_info["field"],
                        "operator": field_info["operator"],
                        "value": field_info["value"]
                    })
                else:
                    # Fallback to description search
                    structured_query["filters"].append({
                        "field": "description",
                        "operator": "search",
                        "value": measurement
                    })
        
        # Generate a standardized query ID using PreviewIdentifier
        try:
            # Create a new unique ID - always use the simple form without referencing previous IDs
            # This avoids the "Cannot specify both prefix and previous_id" error
            query_id = PreviewIdentifier.create_id('nmdc_enhanced_query')
            
            # Store the query in active_queries dictionary for later reference
            self.active_queries[query_id] = structured_query
            
            # Only update last_query_id after successful ID creation and query storage
            self.last_query_id = query_id
        except Exception as e:
            # Fallback ID generation in case of errors with PreviewIdentifier
            logger.warning(f"Error creating query ID: {str(e)}")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            query_id = f"nmdc_enhanced_query_{timestamp}_orig"
            
            # Store the query even with fallback ID
            self.active_queries[query_id] = structured_query
            self.last_query_id = query_id
        
        # Format the structured query as pretty JSON for display
        import json
        
        # Create an example of the API URL that will be used
        api_url_example = f"{self.api_client.API_BASE_URL}/{self.api_client._get_entity_endpoint(structured_query['entity_type'])}?filter="
        
        # Convert filters to the API filter string format
        filter_str = self.api_client._conditions_to_filter_str(
            [{
                "field": f["field"],
                "op": f["operator"],
                "value": f["value"]
            } for f in structured_query["filters"]]
        )
        
        api_url_example += filter_str
        
        # Format query for display - show both our structured format and the API format
        structured_query_str = json.dumps(structured_query, indent=2)
        
        # Create a response that includes both explanation and structured query
        response_content = f"## NMDC Sample Query\n\n"
        response_content += f"Based on your request: '{query_text}'\n\n"
        response_content += "### Structured Query\n\n"
        response_content += f"```nmdc\n{structured_query_str}\n```\n"
        response_content += f"Query ID: {query_id}\n\n"
        
        response_content += f"### API Request Format\n\n"
        response_content += f"This query will translate to the following NMDC API request:\n\n"
        response_content += f"```\nGET {api_url_example}\n```\n\n"
        
        response_content += f"### Result Preview\n\n"
        response_content += f"This query will search for biosample records matching your criteria.\n"
        response_content += f"Results will include:\n"
        response_content += f"- Sample metadata (ID, collection date, location)\n"
        response_content += f"- Associated study information when available\n"
        response_content += f"- Environmental information\n\n"
        
        response_content += f"### Execute Query\n\n"
        response_content += f"To execute this query, use: `nmdc_enhanced.query {query_id}`"
        
        return ServiceResponse(messages=[
            ServiceMessage(
                service=self.name,
                content=response_content,
                message_type=MessageType.INFO
            )
        ])
    
    def _extract_sample_search_parameters(self, query: str) -> Dict[str, Any]:
        """Extract search parameters for samples from natural language query.
        
        This method uses the schema manager to find the best fields to use based on
        the content of the user's query.
        
        Args:
            query: Natural language query
            
        Returns:
            Dictionary of search parameters
        """
        query = query.lower()
        
        # Initialize parameters
        params = {
            "location": None,
            "environment": None,
            "researcher": None,
            "measurements": []
        }
        
        # Get field mappings from schema manager
        field_mappings = self.schema_manager.get_field_mappings()
        
        # Environment detection
        if any(env in query for env in ["soil", "dirt", "earth", "ground"]):
            params["environment"] = "soil"
        elif any(env in query for env in ["water", "marine", "aquatic", "ocean", "sea", "lake", "river"]):
            params["environment"] = "water"
        elif any(env in query for env in ["forest", "woodland", "woods", "trees"]):
            params["environment"] = "forest"
        elif any(env in query for env in ["wetland", "marsh", "swamp", "bog"]):
            params["environment"] = "wetland"
        
        # Location detection - use geospatial hierarchy if available
        states = [
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado", 
            "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho", 
            "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", 
            "maine", "maryland", "massachusetts", "michigan", "minnesota", 
            "mississippi", "missouri", "montana", "nebraska", "nevada", 
            "new hampshire", "new jersey", "new mexico", "new york", 
            "north carolina", "north dakota", "ohio", "oklahoma", "oregon", 
            "pennsylvania", "rhode island", "south carolina", "south dakota", 
            "tennessee", "texas", "utah", "vermont", "virginia", "washington", 
            "west virginia", "wisconsin", "wyoming"
        ]
        
        # Look for states in the query
        for state in states:
            if state in query:
                params["location"] = state
                break
        
        # Look for regions
        regions = ["west coast", "east coast", "midwest", "south", "northeast", "southwest", "northwest"]
        if not params["location"]:
            for region in regions:
                if region in query:
                    params["location"] = region
                    break
        
        # Try to extract any other location using pattern matching
        if not params["location"]:
            # Look for "from [location]" pattern
            location_match = re.search(r'from\s+([a-z\s]+?)(?:\s+with|\s+in|\s+and|$)', query)
            if location_match:
                location = location_match.group(1).strip()
                # Exclude common non-location terms
                non_location_terms = ["soil", "water", "forest", "samples", "study", "biosample", "data"]
                if location and not any(term == location for term in non_location_terms):
                    params["location"] = location
        
        # Researcher detection
        researcher_match = re.search(r'(?:from|by)\s+(?:the\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:lab|group|team)', query)
        if researcher_match:
            params["researcher"] = researcher_match.group(1)
        
        # Measurement detection using the field mappings
        measurement_fields = field_mappings.get("measurement", {}).get("fields", [])
        if measurement_fields:
            # Extract field names and create a lowercase lookup set
            measurement_terms = set()
            for field in measurement_fields:
                # Split field names like "carbon_content" into ["carbon", "content"]
                parts = field["field"].replace("_", " ").lower().split()
                for part in parts:
                    if len(part) > 3:  # Skip very short words
                        measurement_terms.add(part)
            
            # Look for measurement terms in the query
            found_measurements = []
            for term in measurement_terms:
                if term in query and term not in found_measurements:
                    found_measurements.append(term)
            
            params["measurements"] = found_measurements
        
        # If no measurements found, check for common terms
        if not params["measurements"]:
            measurement_terms = {
                "carbon": ["carbon"],
                "nitrogen": ["nitrogen"],
                "phosphorus": ["phosphorus", "phosphate"],
                "temperature": ["temperature", "temp"],
                "pH": ["ph", "acidity"],
                "moisture": ["moisture", "humidity", "water content"],
                "oxygen": ["oxygen", "o2"]
            }
            
            for measure_type, terms in measurement_terms.items():
                if any(term in query for term in terms):
                    params["measurements"].append(measure_type)
        
        return params
    
    async def _handle_query_execution(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Execute a previously built query against the NMDC API."""
        # Get query ID - handle both input formats
        query_id = parsed.get('query_id', None)
        if not query_id:
            # Try extracting from args string if not in parsed dict
            query_id = args.strip()
            
        if not query_id:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content="No query ID provided.",
                    message_type=MessageType.ERROR
                )
            ])
        
        # Check if query exists in active_queries (original search location)
        if query_id not in self.active_queries:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Query with ID {query_id} not found.",
                    message_type=MessageType.ERROR
                )
            ])
        
        # Get query definition
        query_def = self.active_queries[query_id]
        
        # Extract needed info from query
        entity_type = query_def.get('entity_type', '').lower()
        conditions = query_def.get('filters', [])  # Match field name used in structured_query
        
        # Validate entity type and conditions
        if not entity_type:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content="Query is missing entity type.",
                    message_type=MessageType.ERROR
                )
            ])
        
        if not conditions:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content="Query has no conditions.",
                    message_type=MessageType.ERROR
                )
            ])
        
        # Convert our internal condition format to API condition format
        api_conditions = []
        for condition in conditions:
            field = condition.get('field', '')
            operator = condition.get('operator', 'search')
            value = condition.get('value', '')
            
            # Skip empty conditions
            if not field:
                continue
                
            # Convert our internal filter format to the API format expected by the client
            if operator == "search":
                api_conditions.append({
                    "field": field,
                    "op": "search",
                    "value": value
                })
            elif operator == "exists":
                # For exists operator, we need to use a different approach
                # Just specify the field name to check if it exists
                api_conditions.append({
                    "field": field,
                    "op": "exists",
                    "value": None
                })
            else:
                # For other operators, use as provided but with "op" instead of "operator"
                api_conditions.append({
                    "field": field,
                    "op": operator,
                    "value": value
                })
        
        # Log the query we're about to execute
        logger.info(f"Executing {entity_type} query with conditions: {api_conditions}")
        
        # Check that entity type is valid
        supported_entities = await self.api_client.get_supported_entities()
        if entity_type not in supported_entities:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"The entity type '{entity_type}' is not currently supported by the NMDC API.\n\n"
                           f"Currently supported entities: {', '.join(supported_entities)}",
                    message_type=MessageType.ERROR
                )
            ])
        
        # Execute query against the API
        raw_df, metadata = await self.api_client.search_entities(
            entity_type=entity_type,
            conditions=api_conditions
        )
        
        # No results
        if raw_df is None or raw_df.empty:
            # Format filter string for display
            filter_str = self.api_client._conditions_to_filter_str(api_conditions)
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"No results found for your query.\n\n"
                           f"Query ID: {query_id}\n"
                           f"Entity type: {entity_type}\n"
                           f"Filter: {filter_str}\n\n"
                           f"You may want to try a simpler query or modify your search terms.",
                    message_type=MessageType.INFO
                )
            ])
        
        # Generate a standardized result ID using PreviewIdentifier
        try:
            # Create a new unique result ID - always use the simple form
            # This avoids the "Cannot specify both prefix and previous_id" error
            result_id = PreviewIdentifier.create_id('nmdc_enhanced_result')
            
            # Store the result ID for future use
            self.last_result_id = result_id
        except Exception as e:
            # Fallback ID generation in case of errors with PreviewIdentifier
            logger.warning(f"Error creating result ID: {str(e)}")
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            result_id = f"nmdc_enhanced_result_{timestamp}_orig"
            self.last_result_id = result_id
        
        # Check if we got results from a fallback query
        if metadata.get("fallback_query", False):
            warning_message = "\n\n⚠️ **Note**: The original query failed, so these results are from a generic query and may not match your specific criteria."
        else:
            warning_message = ""
        
        # Transform raw results into analysis-ready format
        logger.info(f"Beginning transformation process for {len(raw_df) if raw_df is not None else 0} {entity_type} records")
        
        # Store the raw DataFrame in result store with a different ID for debugging
        raw_result_id = f"{result_id}_raw"
        
        # Make a deep copy of the raw DataFrame to preserve the original structure
        # This prevents potential side effects from the transformation process
        try:
            # First convert to dict, then back to DataFrame for clean copy
            raw_dict = raw_df.to_dict(orient='records')
            raw_df_copy = pd.DataFrame(raw_dict)
            self.result_store.store(raw_result_id, raw_df_copy)
            logger.info(f"Stored raw DataFrame as {raw_result_id} for debugging")
        except Exception as e:
            logger.error(f"Error storing raw DataFrame: {str(e)}")
            # If copying fails, store the original with a warning
            self.result_store.store(raw_result_id, raw_df)
            logger.warning(f"Stored original raw DataFrame without deepcopy as {raw_result_id}")
        
        # Transform the raw DataFrame into analysis-ready format
        # Use a fresh copy to avoid modifying the stored raw data
        dictionary_columns = set()  # Default empty set
        try:
            raw_df_fresh = pd.DataFrame(raw_dict)
            result_df, dictionary_columns = self._transform_api_results(raw_df_fresh, entity_type)
            logger.info(f"Transformation complete - produced DataFrame with shape {result_df.shape}")
        except Exception as e:
            logger.error(f"Error during transformation: {str(e)}")
            import traceback
            logger.error(f"Transformation error details: {traceback.format_exc()}")
            # Use original raw_df as fallback if transformation fails
            result_df = raw_df
            # Identify dictionary columns in the raw data for fallback
            for col in raw_df.columns:
                sample = raw_df[col].dropna().head(5)
                if not sample.empty and any(isinstance(val, dict) for val in sample):
                    dictionary_columns.add(col)
            logger.warning("Using untransformed raw DataFrame as fallback due to transformation error")
        
        # Store the transformed DataFrame in result store
        try:
            # Again, make a clean copy to prevent reference issues
            result_dict = result_df.to_dict(orient='records')
            result_df_copy = pd.DataFrame(result_dict)
            self.result_store.store(result_id, result_df_copy)
            logger.info(f"Stored transformed DataFrame as {result_id}")
        except Exception as e:
            logger.error(f"Error storing transformed DataFrame: {str(e)}")
            # Direct storage without copy as fallback
            self.result_store.store(result_id, result_df)
            logger.warning(f"Stored transformed DataFrame without deepcopy as {result_id}")
        
        # Enrich results with study information if applicable (using the transformed DataFrame)
        if entity_type == "biosample":
            try:
                # Check for the appropriate study ID field
                study_id_fields = ["associated_studies", "study_id"]
                if any(field in result_df.columns for field in study_id_fields):
                    result_df = await self._enrich_with_study_info(result_df)
            except Exception as e:
                logger.warning(f"Error enriching with study info: {str(e)}")
                warning_message += "\n\n⚠️ Could not retrieve associated study information due to an API error."
        
        # Verify the columns after transformation - check if we've created the term columns correctly
        expected_pattern_columns = []
        for col in dictionary_columns:
            # Look for pattern-based component columns like env_broad_scale_term_name
            component_columns = [c for c in result_df.columns if c.startswith(f"{col}_")]
            if component_columns:
                logger.info(f"Column '{col}' was expanded into components: {component_columns}")
                expected_pattern_columns.extend(component_columns)
            else:
                # Include sample values in the warning for diagnostic purposes
                sample_values = []
                try:
                    # Get up to 3 non-null sample values
                    for val in result_df[col].dropna().head(3):
                        sample_val = str(val)
                        if len(sample_val) > 100:
                            sample_val = sample_val[:100] + "..."
                        sample_values.append(sample_val)
                    
                    logger.warning(f"Column '{col}' was not expanded into components. Sample values: {sample_values}")
                except Exception as e:
                    logger.warning(f"Column '{col}' was not expanded into components. Error getting sample values: {str(e)}")
        
        logger.info(f"After transformation, found {len(expected_pattern_columns)} component columns from dictionaries")
        
        # Generate result preview
        preview_title = f"Result Table (ID: {result_id})"
        preview_content = self._generate_result_preview(result_df, entity_type)
            
        # Generate rich analysis summary
        rich_summary = self._generate_rich_result_summary(result_df, entity_type)
            
        # Build the combined response content
        response_content = f"# NMDC Query Results\n"
        response_content += f"Query ID: `{query_id}`\n"
        response_content += f"Result ID: `{result_id}`\n"
        response_content += f"Entity type: {entity_type}\n"
        response_content += f"Found {len(result_df)} records\n\n"
        
        # Add data preview in a code block for better readability
        response_content += f"## {preview_title}\n```\n{preview_content}\n```\n\n"
        
        # Add rich summary analysis
        response_content += f"## Analysis\n{rich_summary}\n\n"
        
        # Add next steps guidance
        response_content += f"## Next Steps\n"
        response_content += f"- Convert to dataset: `convert {result_id} to dataset`\n"
        response_content += f"- Filter results: Create a more specific query\n"
        response_content += f"- Related data: Explore {('biosamples' if entity_type == 'study' else 'studies' if entity_type == 'biosample' else 'data')} connected to these results\n"
        
        # Return a single combined response
        return ServiceResponse(messages=[
            ServiceMessage(
                service=self.name,
                content=response_content,
                message_type=MessageType.RESULT
            )
        ])
    
    async def _enrich_with_study_info(self, biosample_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich biosample DataFrame with study information.
        
        Args:
            biosample_df: DataFrame with biosample data
            
        Returns:
            Enriched DataFrame
        """
        # Check for associated_studies field which is what the API actually uses
        study_id_field = None
        if "associated_studies" in biosample_df.columns:
            study_id_field = "associated_studies"
        elif "study_id" in biosample_df.columns:
            study_id_field = "study_id"
        
        if not study_id_field:
            logger.warning("No study ID field found in biosample data")
            return biosample_df
        
        # Get all unique study IDs from the biosamples
        all_study_ids = set()
        
        # Handle case where associated_studies is a list
        if study_id_field == "associated_studies" and isinstance(biosample_df[study_id_field].iloc[0], list):
            for studies in biosample_df[study_id_field].dropna():
                if isinstance(studies, list):
                    all_study_ids.update(studies)
                elif isinstance(studies, str):
                    all_study_ids.add(studies)
        else:
            # Standard case with single study ID
            all_study_ids = set(biosample_df[study_id_field].dropna().unique())
        
        study_ids = list(all_study_ids)
        
        if not study_ids:
            logger.warning("No study IDs extracted from biosample data")
            return biosample_df
        
        # Get study data using either POST search or individual GET requests
        try:
            logger.info(f"Retrieving information for {len(study_ids)} studies")
            
            # First try the POST-based search method
            study_df = None
            try:
                study_df, total = await self.api_client.search_studies_post(study_ids)
                if study_df is not None and not study_df.empty:
                    logger.info(f"Successfully retrieved {len(study_df)} studies using POST search")
                else:
                    logger.warning("POST search for studies returned no results, trying individual GET requests")
                    study_df = None
            except Exception as e:
                logger.warning(f"Error with POST search for studies: {str(e)}")
                study_df = None
                
            # If POST search failed, try individual GET requests
            if study_df is None or study_df.empty:
                logger.info("Retrieving studies individually")
                study_df = await self.api_client.get_studies_individually(study_ids)
            
            if study_df.empty:
                logger.warning("No study data found using any method")
                return biosample_df
            
            # Create a mapping from study ID to study information
            study_info = {}
            for _, row in study_df.iterrows():
                study_info[row['id']] = {
                    'study_name': row.get('name', ''),
                    'study_description': row.get('description', ''),
                    'pi_name': row.get('principal_investigator_name', '')
                }
            
            # Add study information to the biosample dataframe
            biosample_df['study_name'] = None
            biosample_df['study_description'] = None
            biosample_df['pi_name'] = None
            
            # Handle both cases (single study_id or list of associated_studies)
            if study_id_field == "associated_studies" and isinstance(biosample_df[study_id_field].iloc[0], list):
                # For each biosample, use the first associated study to get info
                for idx, row in biosample_df.iterrows():
                    studies = row[study_id_field]
                    if isinstance(studies, list) and studies:
                        study_id = studies[0]  # Use the first associated study
                        if study_id in study_info:
                            for key, value in study_info[study_id].items():
                                biosample_df.at[idx, key] = value
            else:
                # Standard case with single study ID
                for idx, row in biosample_df.iterrows():
                    study_id = row[study_id_field]
                    if study_id in study_info:
                        for key, value in study_info[study_id].items():
                            biosample_df.at[idx, key] = value
            
            logger.info(f"Successfully enriched {len(biosample_df)} biosamples with study information")
            return biosample_df
            
        except Exception as e:
            logger.error(f"Error enriching with study information: {str(e)}")
            return biosample_df
    
    def _generate_sample_preview(self, df: pd.DataFrame) -> str:
        """Generate a preview of sample data for display.
        
        Args:
            df: DataFrame with sample data
            
        Returns:
            Markdown-formatted preview
        """
        if df is None or df.empty:
            return "No data to preview."
        
        # Select a subset of important columns for preview
        preview_columns = [
            "id", "env_medium", "location_name", 
            "pi_name", "study_name"
        ]
        
        # Add any physical measurement columns that have data
        physical_columns = [col for col in df.columns if any([
            col.endswith("_mg_per_g"),
            col.endswith("_percent"),
            col.endswith("_celsius"),
            col == "ph_measurement"
        ])]
        
        # Add data object count columns
        count_columns = [col for col in df.columns if col.endswith("_count")]
        
        # Combine all columns, ensuring they exist in the DataFrame
        all_columns = preview_columns + physical_columns + count_columns
        available_columns = [col for col in all_columns if col in df.columns]
        
        # Limit to first 5 rows
        preview_df = df[available_columns].head(5)
        
        # Format as markdown table
        preview = "### Sample Preview (first 5 results)\n\n"
        preview += preview_df.to_markdown(index=False)
        
        return preview
    
    def _transform_api_results(self, raw_df: pd.DataFrame, entity_type: str) -> Tuple[pd.DataFrame, Set[str]]:
        """Transform raw API results into an analysis-ready DataFrame.
        
        This method performs several critical transformations:
        1. Handles simple key-value pairs directly
        2. Converts lists of basic types to comma-separated strings
        3. Extracts values from dictionaries based on pattern recognition
        4. Specially handles the "annotations" field by flattening its structure
        5. Enumerates omics_processing outputs by file type
        
        Args:
            raw_df: The raw DataFrame from the API
            entity_type: The entity type (biosample, study, etc.)
            
        Returns:
            A tuple of (transformed DataFrame, set of dictionary column names)
        """
        try:
            if raw_df is None or raw_df.empty:
                logger.warning("Received empty or None DataFrame in _transform_api_results")
                return pd.DataFrame(), set()
    
            # Log DataFrame shape and columns for debugging
            logger.debug(f"Input DataFrame has shape: {raw_df.shape}")
            logger.debug(f"Input columns: {raw_df.columns.tolist()}")
            
            # ====== STEP 1: Fix any serialized JSON columns ======
            # This handles cases where dictionaries or lists got serialized to strings
            raw_df = self._fix_serialized_json_columns(raw_df)
            
            # ====== STEP 2: Process each record individually ======
            # Create a list to hold transformed records
            transformed_records = []
            
            # Set of columns we've identified as containing dictionaries or lists
            dictionary_columns = set()
            list_columns = set()
            
            # Define standard field patterns
            field_patterns = {
                'value_fields': ['has_raw_value', 'has_value', 'value', 'raw_value', 'string_value', 
                               'numeric_value', 'has_numeric_value', 'text_value'],
                'label_fields': ['label', 'name', 'title', 'description', 'summary', 'comment', 'notes',
                               'display_name', 'alternate_name', 'full_name', 'short_name', 'text'],
                'id_fields': ['id', 'identifier', 'accession', 'external_id', 'internal_id', 
                            'uuid', 'doi', 'url', 'uri', 'permalink'],
                'unit_fields': ['has_unit', 'unit', 'units', 'unit_id', 'measurement_unit', 'has_measurement_unit'],
                'ontology_fields': ['term', 'term_id', 'term_name', 'term_label', 'ontology_term',
                                  'category', 'type', 'class', 'concept', 'pred']
            }
            
            # Combine all field patterns for use in value extraction
            common_fields = []
            for field_list in field_patterns.values():
                common_fields.extend(field_list)
                
            # Identify dictionary and list columns first
            for col in raw_df.columns:
                if raw_df[col].isna().all():
                    continue
                    
                non_null_values = raw_df[col].dropna().head(10)
                if non_null_values.empty:
                    continue
                    
                first_value = non_null_values.iloc[0]
                if isinstance(first_value, dict):
                    dictionary_columns.add(col)
                    logger.debug(f"Identified '{col}' as a dictionary column")
                elif isinstance(first_value, list):
                    list_columns.add(col)
                    logger.debug(f"Identified '{col}' as a list column")
            
            logger.info(f"Found {len(dictionary_columns)} dictionary columns and {len(list_columns)} list columns")
            
            # Process each row to handle various data types appropriately
            for idx, row in enumerate(raw_df.itertuples(index=False)):
                # Convert named tuple to dictionary for easier processing
                record = {col: getattr(row, col) for col in raw_df.columns}
                transformed_record = {}
                
                # Process each field in the record based on its type
                for key, value in record.items():
                    try:
                        # Skip None or NaN values
                        if value is None or (isinstance(value, float) and pd.isna(value)):
                            transformed_record[key] = None
                            continue
                            
                        # Handle simple scalar types directly
                        if isinstance(value, (str, int, float, bool)):
                            transformed_record[key] = value
                            continue
                            
                        # Handle dictionaries - use pattern-based extraction
                        if key in dictionary_columns or isinstance(value, dict):
                            # Apply pattern-based extraction to get main value and components
                            main_value, components = self._extract_by_pattern(value, key, common_fields)
                            
                            # Store the main value
                            transformed_record[key] = main_value
                            
                            # Add component values as new columns
                            for comp_name, comp_value in components.items():
                                column_name = f"{key}_{comp_name}"
                                transformed_record[column_name] = comp_value
                                
                        # Handle lists with appropriate conversion strategy
                        elif key in list_columns or isinstance(value, list):
                            if not value:  # Empty list
                                transformed_record[key] = ""
                                continue
                                
                            # Special case: associated_studies
                            if key == "associated_studies":
                                transformed_record[key] = self._process_associated_studies(value)
                                
                            # Special case: omics_processing for biosamples
                            elif key == "omics_processing" and entity_type == "biosample":
                                file_counts, file_ids = self._extract_file_info_from_omics(value)
                                
                                # Add file counts and IDs as separate columns
                                for file_type, count in file_counts.items():
                                    clean_type = self._clean_column_name(f"{file_type}_count")
                                    transformed_record[clean_type] = count
                                    
                                for file_type, ids in file_ids.items():
                                    clean_ids = self._clean_column_name(f"{file_type}_ids")
                                    transformed_record[clean_ids] = ", ".join(ids)
                                    
                                # Add a flag to indicate omics_processing was processed
                                transformed_record["has_omics_processing"] = True
                                transformed_record["omics_processing_count"] = len(value)
                                
                            # Lists of dictionaries - extract and join values
                            elif value and all(isinstance(item, dict) for item in value if item is not None):
                                extracted_values = []
                                for item in value:
                                    if item is not None:
                                        extracted = self._extract_value_from_dict(item, common_fields)
                                        if extracted is not None and not pd.isna(extracted):
                                            extracted_values.append(str(extracted))
                                
                                if extracted_values:
                                    transformed_record[key] = ", ".join(extracted_values)
                                else:
                                    transformed_record[key] = value
                                    
                            # Lists of simple types - convert to comma-separated string
                            elif value and all(isinstance(item, (str, int, float, bool)) for item in value if item is not None):
                                valid_items = [str(item) for item in value if item is not None and not pd.isna(item)]
                                transformed_record[key] = ", ".join(valid_items) if valid_items else ""
                                
                            # Other list types - store as is
                            else:
                                transformed_record[key] = value
                        
                        # Special case: annotations
                        elif key == "annotations" and isinstance(value, dict):
                            # Skip certain fields that are redundant or represented elsewhere
                            skip_fields = ["lat_lon", "depth", "type"]
                            
                            # Process each annotation as a separate column
                            for annotation_key, annotation_value in value.items():
                                if annotation_key not in skip_fields:
                                    if isinstance(annotation_value, dict):
                                        extracted = self._extract_value_from_dict(annotation_value, common_fields)
                                        transformed_record[annotation_key] = extracted
                                    else:
                                        transformed_record[annotation_key] = annotation_value
                                        
                    except Exception as e:
                        logger.warning(f"Error processing field '{key}' for record {idx}: {str(e)}")
                        transformed_record[key] = value
                
                transformed_records.append(transformed_record)
            
            # Create transformed DataFrame
            if not transformed_records:
                logger.warning("No records were successfully transformed")
                return pd.DataFrame(), dictionary_columns
                
            result_df = pd.DataFrame(transformed_records)
            logger.info(f"Transformation complete: {len(result_df)} records with {len(result_df.columns)} columns")
            
            return result_df, dictionary_columns
            
        except Exception as e:
            logger.error(f"Error transforming API results: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return the original DataFrame to avoid data loss on error
            return raw_df, set()
    
    def _fix_serialized_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix any columns containing serialized JSON strings by converting them back to dictionaries."""
        if df is None or df.empty:
            return df
            
        serialized_columns = []
        
        def parse_if_json_str(val):
            """Parse a value if it's likely a JSON string."""
            if val is None or not isinstance(val, str):
                return val
                
            # Check if this looks like a JSON dictionary
            if (val.startswith('{') and val.endswith('}')) or (val.startswith('"') and val.endswith('"') and '"{' in val):
                try:
                    # For double-escaped JSON
                    if val.startswith('"') and val.endswith('"') and ('"{' in val or '}"' in val or '""' in val):
                        import ast
                        try:
                            # Unescape first level
                            unescaped = ast.literal_eval(val)
                            if isinstance(unescaped, str) and unescaped.startswith('{') and unescaped.endswith('}'):
                                # Parse second level
                                return json.loads(unescaped)
                            return unescaped
                        except (SyntaxError, ValueError) as e:
                            logger.debug(f"Failed to unescape JSON string: {val[:100]} - {str(e)}")
                    
                    # Regular JSON parsing
                    if val.startswith('{') and val.endswith('}'):
                        return json.loads(val)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed for: {val[:100]} - {str(e)}")
                    
            return val
            
        # Check each column for serialized JSON strings
        for col in df.columns:
            # Skip empty columns
            if df[col].isna().all():
                continue
                
            # Sample values to check for JSON strings
            sample = df[col].dropna().head(5)
            if not sample.empty:
                has_json_strings = any(
                    isinstance(val, str) and 
                    ((val.startswith('{') and val.endswith('}')) or
                     (val.startswith('"') and val.endswith('"') and '"{' in val))
                    for val in sample
                )
                
                if has_json_strings:
                    logger.warning(f"Found serialized JSON strings in column '{col}', attempting to parse...")
                    serialized_columns.append(col)
                    
                    # Parse JSON strings back to dictionaries
                    df[col] = df[col].apply(parse_if_json_str)
                    
                    # Verify the fix worked
                    fixed_sample = df[col].dropna().head(5)
                    dict_count = sum(1 for val in fixed_sample if isinstance(val, dict))
                    logger.info(f"After fixing '{col}': found {dict_count} dictionaries out of {len(fixed_sample)} non-null values")
        
        if serialized_columns:
            logger.warning(f"Fixed {len(serialized_columns)} columns with serialized JSON: {serialized_columns}")
            
        return df
        
    def _extract_by_pattern(self, value, field_name=None, common_fields=None):
        """Extract meaningful values from a dictionary using pattern recognition.
        
        Args:
            value: Dictionary to extract values from
            field_name: Name of the field (for context)
            common_fields: List of common field names to check
            
        Returns:
            Tuple of (main_value, components_dict)
        """
        # Default empty components
        components = {}
        
        # Handle non-dictionary or empty values
        if not isinstance(value, dict) or not value:
            return value, components
            
        # Keep the original value as default
        processed_value = value
        
        # Pattern 1: QuantityValue with numeric value and unit
        if 'type' in value and value.get('type') in ['nmdc:QuantityValue', 'QuantityValue']:
            # First try to get the raw value
            if 'has_raw_value' in value:
                processed_value = value['has_raw_value']
                
                # Extract numeric value and unit components if available
                if 'has_numeric_value' in value and value['has_numeric_value'] is not None:
                    components['value'] = value['has_numeric_value']
                if 'has_unit' in value and value['has_unit'] is not None:
                    components['unit'] = value['has_unit']
                    
            # If no raw value but has numeric value, use that
            elif 'has_numeric_value' in value:
                processed_value = value['has_numeric_value']
                if 'has_unit' in value and value['has_unit'] is not None:
                    components['unit'] = value['has_unit']
        
        # Pattern 2: TextValue or TimestampValue
        elif 'type' in value and value.get('type') in ['nmdc:TextValue', 'TextValue', 
                                                    'nmdc:TimestampValue', 'TimestampValue']:
            if 'has_raw_value' in value:
                processed_value = value['has_raw_value']
                
        # Pattern 3: ControlledTermValue or TermLink with term dictionary
        elif 'type' in value and value.get('type') in ['nmdc:TermLink', 'TermLink', 
                                                  'nmdc:ControlledTermValue', 
                                                  'nmdc:ControlledIdentifiedTermValue']:
            # Try to extract from term dictionary
            if 'term' in value and isinstance(value['term'], dict):
                term = value['term']
                
                # Extract name/label and ID
                if 'name' in term:
                    processed_value = term['name']
                    components['term_name'] = term['name']
                elif 'label' in term:
                    processed_value = term['label']
                    components['term_label'] = term['label']
                    
                # Add ID if available
                if 'id' in term:
                    components['term_id'] = term['id']
                
            # If no term extraction but has raw value, use that
            elif 'has_raw_value' in value:
                processed_value = value['has_raw_value']
                
        # Pattern 4: GeolocationValue with lat/lon
        elif 'type' in value and value.get('type') in ['nmdc:GeolocationValue', 'GeolocationValue']:
            if 'has_raw_value' in value:
                processed_value = value['has_raw_value']
                
            # Extract coordinates as components
            if 'latitude' in value:
                components['lat'] = value['latitude']
            if 'longitude' in value:
                components['lon'] = value['longitude']
        
        # General pattern: Try common fields in priority order
        else:
            # First try the raw value fields
            for field in ['has_raw_value', 'value', 'name', 'label', 'id']:
                if field in value and value[field] is not None:
                    processed_value = value[field]
                    break
            
            # If still using the original dict and we have common fields, use extract_value
            if processed_value is value and common_fields:
                extracted = self._extract_value_from_dict(value, common_fields)
                if extracted is not None and not pd.isna(extracted):
                    processed_value = extracted
        
        return processed_value, components
    
    def _process_associated_studies(self, studies_list):
        """Process associated studies list to extract study IDs."""
        if not studies_list:
            return ""
            
        # First, handle the case where we already have strings
        if all(isinstance(item, str) for item in studies_list if item is not None):
            return ", ".join(item for item in studies_list if item is not None)
            
        # Handle more complex items by extracting IDs
        extracted_values = []
        for item in studies_list:
            if item is None:
                continue
            elif isinstance(item, dict) and 'id' in item:
                extracted_values.append(item['id'])
            elif isinstance(item, str):
                extracted_values.append(item)
            else:
                # Convert other non-None types to string
                extracted_values.append(str(item))
                
        return ", ".join(extracted_values) if extracted_values else ""
    
    def _extract_file_info_from_omics(self, omics_list):
        """Extract file type counts and IDs from omics_processing structure."""
        file_type_counts = {}
        file_type_ids = {}
        
        # Skip if not a valid list
        if not omics_list or not isinstance(omics_list, list):
            return {}, {}
            
        # Process each omics_processing entry
        for op in omics_list:
            if not isinstance(op, dict):
                continue
                
            # Look for omics_data which contains the file outputs
            if "omics_data" in op and isinstance(op["omics_data"], list):
                for od in op["omics_data"]:
                    if not isinstance(od, dict):
                        continue
                        
                    # Check for outputs list
                    if "outputs" in od and isinstance(od["outputs"], list):
                        for output in od["outputs"]:
                            if not isinstance(output, dict):
                                continue
                                
                            # Count by file type
                            file_type = output.get("file_type_description", "Unknown")
                            if file_type not in file_type_counts:
                                file_type_counts[file_type] = 0
                                file_type_ids[file_type] = []
                                
                            file_type_counts[file_type] += 1
                            if "id" in output:
                                file_type_ids[file_type].append(output["id"])
                                
        return file_type_counts, file_type_ids
    
    def _extract_value_from_dict(self, value_dict: Dict, common_fields: List[str]) -> Any:
        """Extract the most meaningful value from a dictionary.
        
        Args:
            value_dict: Dictionary to extract value from
            common_fields: List of common field names to check in priority order
            
        Returns:
            The extracted value, or None if no suitable value found
        """
        if value_dict is None or not isinstance(value_dict, dict) or not value_dict:
            return None
            
        # First check for type indicators to handle special patterns
        if 'type' in value_dict:
            type_value = value_dict['type']
            # Special handling for known types
            if type_value in ['nmdc:QuantityValue', 'QuantityValue']:
                if 'has_raw_value' in value_dict:
                    return value_dict['has_raw_value']
                elif 'has_numeric_value' in value_dict:
                    return value_dict['has_numeric_value']
            elif type_value in ['nmdc:TextValue', 'TextValue']:
                if 'has_raw_value' in value_dict:
                    return value_dict['has_raw_value']
            elif type_value in ['nmdc:TermLink', 'TermLink']:
                if 'term' in value_dict and isinstance(value_dict['term'], dict):
                    term = value_dict['term']
                    if 'name' in term:
                        return term['name']
                    elif 'label' in term:
                        return term['label']
                    elif 'id' in term:
                        return term['id']
        
        # Check common fields in priority order
        for field in common_fields:
            if field in value_dict and value_dict[field] is not None:
                return value_dict[field]
                
        # If nothing found, try to get the first non-None value
        for key, val in value_dict.items():
            if val is not None and key != 'type' and not isinstance(val, (dict, list)):
                return val
                
        # If we still don't have a value, convert dict to string as last resort
        return str(value_dict)
        
    def _can_convert_to_float(self, val) -> bool:
        """Check if a value can be converted to float."""
        if val is None or pd.isna(val):
            return False
            
        if isinstance(val, (int, float)):
            return True
            
        if not isinstance(val, str):
            return False
            
        # Try to convert string to float
        try:
            float(val)
            return True
        except (ValueError, TypeError):
            return False
            
        return False
    
    def _extract_clean_term_value(self, term_obj):
        """Extract a clean value from an NMDC term object.
        
        For analysis-ready DataFrames, we extract just the value without
        markups or identifiers.
        
        Args:
            term_obj: NMDC term object
            
        Returns:
            Clean extracted value
        """
        if not isinstance(term_obj, dict):
            return str(term_obj) if term_obj is not None else ''
            
        # Get the raw value if available
        if 'has_raw_value' in term_obj:
            raw_value = term_obj['has_raw_value']
            # Clean up raw value by removing ID patterns and underscores
            raw_value = re.sub(r'_+', '', raw_value)  # Remove underscore sequences
            raw_value = re.sub(r'\s+ENVO:\d+', '', raw_value)  # Remove ENVO IDs
            return raw_value.strip()
            
        # If there's a term with a name, use that
        if 'term' in term_obj and isinstance(term_obj['term'], dict) and 'name' in term_obj['term']:
            return term_obj['term']['name']
            
        # Fallback to empty string if nothing is found
        return ''
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean a column name by removing special characters and normalizing whitespace."""
        # Replace special characters with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', column_name)
        # Normalize underscores (replace multiple with single)
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Convert to lowercase for consistency
        return cleaned.lower()
    
    def _handle_dataset_conversion(self, args: str, parsed: Dict[str, Any]) -> ServiceResponse:
        """Handle conversion of query results to dataset."""
        # Get result ID from command - handle both input formats
        result_id = parsed.get('result_id', None)
        if not result_id:
            # Try extracting from args string if not in parsed dict
            result_id = args.strip()
            
        if not result_id:
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content="No result ID provided.",
                    message_type=MessageType.ERROR
                )
            ])
            
        # Check if result exists
        if not self.result_store.has_key(result_id):
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Result with ID {result_id} not found.",
                    message_type=MessageType.ERROR
                )
            ])
        
        try:
            # Retrieve the DataFrame from our result store
            df = self.result_store.get(result_id)
            
            # DEBUG: Check if dictionaries are preserved when retrieving from store
            dictionary_columns = []
            serialized_columns = []
            for col in df.columns:
                # Sample a few values to check if they're dictionaries
                sample = df[col].dropna().head(5)
                if not sample.empty and any(isinstance(val, dict) for val in sample):
                    dictionary_columns.append(col)
                    logger.info(f"Retrieved dictionary column from store: {col}")
                
                # Check if any values are strings that look like serialized JSON
                if not sample.empty and any(isinstance(val, str) and val.startswith('{') and val.endswith('}') for val in sample):
                    serialized_columns.append(col)
                    logger.warning(f"Found serialized JSON strings in column {col} after retrieval from store")
            
            logger.info(f"After retrieval, found {len(dictionary_columns)} dictionary columns and {len(serialized_columns)} serialized columns")
            
            # If we found serialized columns but no dictionary columns, the store might have serialized our dictionaries
            if serialized_columns and not dictionary_columns:
                logger.warning("Dictionary serialization detected in result store! Attempting repair...")
                # Fix serialized columns
                for col in serialized_columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith('{') and x.endswith('}') else x)
                logger.info("Repaired serialized columns, proceeding with transformation")
            
            if df is None or df.empty:
                return ServiceResponse(messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Result with ID {result_id} exists but contains no data.",
                        message_type=MessageType.ERROR
                    )
                ])
            
            # Track transformations applied
            transformations = []
            
            # Check if the DataFrame was already transformed
            # We can detect this by looking for component columns that come from dictionary extraction
            # For example, if we see env_broad_scale_term_name, the DataFrame was already transformed
            pattern_columns = []
            for col in df.columns:
                if '_term_name' in col or '_term_id' in col or '_value' in col or '_unit' in col:
                    pattern_columns.append(col)
            
            already_transformed = len(pattern_columns) > 0
            if already_transformed:
                logger.info(f"DataFrame appears to be already transformed (found {len(pattern_columns)} component columns)")
                transformations.append(f"Using pre-transformed DataFrame with {len(pattern_columns)} component columns")
            elif dictionary_columns:
                # IMPORTANT: Apply our transformation if needed
                # Detect the entity type from the data
                entity_type = self._detect_entity_type(df)
                logger.info(f"Detected entity type: {entity_type}")
                
                # Apply transformation to extract values from dictionaries
                logger.info(f"Applying dictionary transformation to {len(dictionary_columns)} columns")
                transformed_df = self._transform_api_results(df, entity_type)
                transformations.append(f"Extracted data from {len(dictionary_columns)} dictionary columns")
                # Use the transformed DataFrame for further processing
                df = transformed_df
            
            # NOW proceed with Dash compatibility conversions
            # Check for any remaining dictionaries and convert them to strings
            dict_count = 0
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, dict)).any():
                    dict_count += 1
                    logger.warning(f"Converting remaining dictionary values in column '{col}' to strings")
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
            
            if dict_count > 0:
                transformations.append(f"Converted {dict_count} remaining dictionary columns to strings")
            
            # Next, check for any lists and convert them to strings
            list_count = 0
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, list)).any():
                    list_count += 1
                    logger.warning(f"Converting list values in column '{col}' to comma-separated strings")
                    df[col] = df[col].apply(lambda x: ", ".join(str(i) for i in x) if isinstance(x, list) else x)
            
            if list_count > 0:
                transformations.append(f"Converted {list_count} list columns to comma-separated strings")
            
            # Clean column names
            old_columns = list(df.columns)
            df.columns = df.columns.str.replace(r'[.\[\]{}]', '_', regex=True)
            if any(old != new for old, new in zip(old_columns, df.columns)):
                transformations.append("Cleaned column names")
            
            # Generate a dataset ID using the same ID from the result
            dataset_id = result_id.replace("nmdc_enhanced_result", "dataset")
            
            # Generate profile report
            try:
                from ydata_profiling import ProfileReport
                profile = ProfileReport(
                    df,
                    minimal=True,
                    title=f"Profile Report for {result_id}",
                    html={'style': {'full_width': True}},
                    progress_bar=False,
                    correlations={'pearson': {'calculate': True}},
                    missing_diagrams={'matrix': False},
                    samples=None
                )
                profile_html = profile.to_html()
            except Exception as e:
                logger.warning(f"Profile report generation failed: {str(e)}")
                profile_html = None
                transformations.append("Note: Profile report generation failed")
            
            # Create dataset entry - ensure all values are serializable
            # Convert DataFrame to records but ensure all values are JSON serializable
            records = []
            for _, row in df.iterrows():
                record = {}
                for col, val in row.items():
                    if pd.isna(val):
                        record[col] = None
                    elif isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
                        record[col] = val.isoformat()
                    else:
                        record[col] = val
                records.append(record)
            
            dataset_entry = {
                'df': records,
                'metadata': {
                    'source': f"{result_id}",
                    'creation_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df),
                    'columns': list(df.columns),
                    'transformations': transformations,
                    'entity_type': self._detect_entity_type(df)  # Try to detect the entity type
                },
                'profile_report': profile_html
            }
            
            # Format transformation message
            transform_msg = "\n- " + "\n- ".join(transformations) if transformations else ""
            
            # Generate dataset preview
            try:
                preview_df = df.head(5)
                # For safety, convert any potential problematic values to strings
                for col in preview_df.columns:
                    if preview_df[col].apply(lambda x: not isinstance(x, (str, int, float, bool, type(None)))).any():
                        preview_df[col] = preview_df[col].astype(str)
                
                preview_table = preview_df.to_markdown(index=False)
            except Exception as e:
                logger.warning(f"Error generating preview table: {str(e)}")
                preview_table = "Error generating preview table"
            
            # Create a profile summary
            profile_summary = self._generate_profile_summary(df)
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"""# Converting NMDC Enhanced Result to Dataset
Converting result {result_id} to dataset...

## New Dataset Created
Dataset ID: {dataset_id} Name: {result_id} Source: {result_id} Records: {len(df)}

## Dataset Preview (5 of {len(df)} rows)
{preview_table}

## Dataset Profile
{profile_summary}

## Transformations Applied{transform_msg}

The dataset is now available for analysis with ID: {dataset_id}""",
                        message_type=MessageType.RESULT
                    )
                ],
                store_updates={
                    'datasets_store': {
                        dataset_id: dataset_entry
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting result to dataset: {str(e)}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            return ServiceResponse(messages=[
                ServiceMessage(
                    service=self.name,
                    content=f"Error converting result to dataset: {str(e)}",
                    message_type=MessageType.ERROR
                )
            ])
            
    def _detect_entity_type(self, df: pd.DataFrame) -> str:
        """Attempt to detect the entity type from the DataFrame columns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Detected entity type or 'unknown'
        """
        columns = set(df.columns)
        
        # Biosample indicators
        biosample_columns = {
            'environment_broad', 'environment_local', 'environment_medium', 
            'ecosystem_specific', 'location', 'sample_collection_date', 'depth'
        }
        
        # Study indicators
        study_columns = {
            'principal_investigator_name', 'study_category', 'project_name',
            'gold_study_id', 'ecosystem_category'
        }
        
        # Data object indicators
        data_object_columns = {
            'file_size_bytes', 'compression_type', 'data_format', 
            'md5_checksum', 'url'
        }
        
        # Calculate overlap
        biosample_score = len(columns.intersection(biosample_columns))
        study_score = len(columns.intersection(study_columns))
        data_object_score = len(columns.intersection(data_object_columns))
        
        # Determine type based on highest score
        if biosample_score > study_score and biosample_score > data_object_score:
            return 'biosample'
        elif study_score > biosample_score and study_score > data_object_score:
            return 'study'
        elif data_object_score > biosample_score and data_object_score > study_score:
            return 'data_object'
        
        # Default if we can't determine
        return 'unknown'
        
    def _generate_profile_summary(self, df: pd.DataFrame) -> str:
        """Generate a simple profile summary of the DataFrame.
        
        Args:
            df: DataFrame to profile
            
        Returns:
            Markdown formatted profile summary
        """
        summary = "Column Types:\n\n"
        
        # Categorize columns by type
        text_columns = []
        numeric_columns = []
        date_columns = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
            elif pd.api.types.is_datetime64_dtype(df[col]):
                date_columns.append(col)
            else:
                text_columns.append(col)
        
        # Add column type summaries
        if text_columns:
            summary += f"Text columns: {', '.join(text_columns)}\n"
        if numeric_columns:
            summary += f"Numeric columns: {', '.join(numeric_columns)}\n"
        if date_columns:
            summary += f"Date columns: {', '.join(date_columns)}\n"
        
        # Add data completeness section
        summary += "\nData Completeness:\n\n"
        for col in df.columns:
            non_null_percent = (df[col].count() / len(df)) * 100
            if non_null_percent == 100:
                summary += f"100% complete: {col}\n"
            else:
                summary += f"{non_null_percent:.1f}% complete: {col}\n"
        
        # Add value distributions for key columns
        summary += "\nValue Distributions:\n\n"
        
        # Environment columns
        env_columns = [col for col in df.columns if any(term in col.lower() for term in ['env', 'environment', 'ecosystem'])]
        for col in env_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                summary += f"{col}: {unique_count} unique values\n"
        
        # Location columns
        loc_columns = [col for col in df.columns if any(term in col.lower() for term in ['geo', 'loc', 'location'])]
        for col in loc_columns:
            if col in df.columns:
                unique_count = df[col].nunique()
                summary += f"{col}: {unique_count} unique locations\n"
        
        return summary
    
    def _generate_rich_result_summary(self, df: pd.DataFrame, entity_type: str) -> str:
        """Generate a comprehensive yet token-efficient analysis summary.
        
        Includes statistical descriptions of numeric columns and key categorical distributions,
        followed by an LLM-generated interpretation of the data.
        
        Args:
            df: The result DataFrame
            entity_type: Type of entity (biosample, study, etc.)
            
        Returns:
            Formatted analysis summary
        """
        if df is None or df.empty:
            return "No results available for analysis."
            
        total_records = len(df)
        summary = ""
        
        try:
            # Identify numeric and categorical columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            # 1. STATISTICAL SUMMARY FOR NUMERIC COLUMNS
            if numeric_cols:
                # Get basic statistics for numeric columns, limit to most useful ones
                key_stats = ['count', 'mean', 'min', 'max']
                numeric_stats = df[numeric_cols].describe().loc[key_stats]
                
                # Format only a subset of the most important numeric columns (max 5)
                important_numeric = [col for col in self._get_important_columns(entity_type, df.columns) 
                                    if col in numeric_cols][:5]
                
                if important_numeric:
                    summary += "### Key Measurements\n"
                    stats_df = numeric_stats[important_numeric].round(2)
                    
                    # Format in a compact way
                    for col in important_numeric:
                        if col in stats_df.columns:
                            col_stats = stats_df[col]
                            # Only show if we have valid stats
                            if not col_stats.isna().all() and col_stats['count'] > 0:
                                summary += f"**{col}**: {col_stats['count']} values, "
                                summary += f"mean: {col_stats['mean']}, "
                                summary += f"range: {col_stats['min']} to {col_stats['max']}\n"
                    
                    summary += "\n"
            
            # 2. GEOGRAPHIC DISTRIBUTION (if available)
            if 'geo_loc_name' in df.columns:
                locations = df['geo_loc_name'].value_counts().head(3)
                if not locations.empty:
                    summary += "### Geographic Distribution\n"
                    loc_str = ", ".join([f"{loc} ({count})" for loc, count in locations.items() if pd.notna(loc)])
                    if loc_str:
                        summary += f"{loc_str}\n\n"
            
            # 3. CATEGORICAL DISTRIBUTIONS
            # Entity-specific categorical fields
            categorical_fields = []
            if entity_type == 'biosample':
                categorical_fields = ['env_medium', 'env_broad_scale', 'ecosystem_subtype', 'specific_ecosystem']
            elif entity_type == 'study':
                categorical_fields = ['study_category', 'principal_investigator_name']
            elif entity_type == 'data_object':
                categorical_fields = ['file_type_description', 'type']
                
            # Only show fields that exist in this dataframe
            existing_categorical = [f for f in categorical_fields if f in df.columns]
            
            if existing_categorical:
                summary += "### Category Distribution\n"
                for field in existing_categorical[:3]:  # Limit to top 3 fields
                    # Get value counts but handle dictionary values
                    try:
                        # Pre-process any dictionary values first
                        if df[field].apply(lambda x: isinstance(x, dict)).any():
                            logger.debug(f"Field '{field}' contains dictionary values, converting to readable form")
                            # Add debug sample of a raw dictionary for troubleshooting
                            try:
                                sample_dict = df[field].dropna().iloc[0]
                                if isinstance(sample_dict, dict):
                                    logger.debug(f"Dictionary sample for '{field}': {sample_dict}")
                            except (IndexError, KeyError, AttributeError):
                                pass
                                
                            # Apply the extract_term_value function to dictionary values
                            processed_values = df[field].apply(lambda x: self._extract_term_value(x) if isinstance(x, dict) else x)
                            top_values = processed_values.value_counts().head(3)
                        else:
                            top_values = df[field].value_counts().head(3)
                            
                        if not top_values.empty:
                            val_str = ", ".join([f"{val} ({count})" for val, count in top_values.items() if pd.notna(val)])
                            if val_str:
                                summary += f"**{field}**: {val_str}\n"
                    except Exception as e:
                        logger.error(f"Error processing field '{field}' for summary: {str(e)}")
                        # Try to continue with other fields
                        continue
                summary += "\n"
                
            # 4. DATA COMPLETENESS
            # Add a compact data completeness summary
            key_columns = self._get_important_columns(entity_type, df.columns)[:5]
            complete_data = []
            for col in key_columns:
                if col in df.columns:
                    n = df[col].notnull().sum()
                    if n > 0:
                        pct = int((n / total_records) * 100)
                        complete_data.append(f"{col}: {pct}%")
            
            if complete_data:
                summary += "### Data Completeness\n"
                summary += ", ".join(complete_data) + "\n\n"
            
            # 5. LLM INTERPRETATION
            # Add an LLM-generated scientific interpretation of the data
            try:
                # Only generate LLM interpretation if we have enough data
                if len(df) >= 5:
                    logger.info("Generating LLM interpretation for search results")
                    llm_interpretation = self._generate_llm_interpretation(df, entity_type)
                    if llm_interpretation:
                        summary += "### Interpretation\n"
                        summary += llm_interpretation + "\n"
            except Exception as e:
                logger.error(f"Error generating LLM interpretation: {str(e)}")
                # Continue without LLM interpretation if it fails
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            summary += f"Basic statistics for {total_records} {entity_type} records could not be generated due to an error."
        
        # Debug: Log size information
        token_estimate = len(summary) / 4  # Rough estimate: ~4 characters per token
        logger.info(f"Summary length: {len(summary)} chars, ~{token_estimate:.0f} tokens")
        
        return summary
    
    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process message for LLM Service interface.
        
        This method is required by the LLMServiceMixin but is not used directly
        in the NMDC Enhanced Service. It delegates to query handling methods.
        
        Args:
            message: User message
            chat_history: Chat history
            
        Returns:
            Response string
        """
        # This is a placeholder implementation to satisfy the interface
        # Our service handles messages through the execute method
        return "This method should not be called directly in NMDC Enhanced Service."
        
    def summarize(self, content: Union[pd.DataFrame, str], chat_history: List[Dict[str, Any]]) -> str:
        """Generate a summary of the given content.
        
        Args:
            content: DataFrame or string to summarize
            chat_history: Chat history
            
        Returns:
            Summary string
        """
        if isinstance(content, pd.DataFrame):
            entity_type = self._detect_entity_type(content)
            return self._generate_llm_interpretation(content, entity_type)
        elif isinstance(content, str):
            # For text content, wrap it in a simple prompt
            system_prompt = """You are a scientific assistant specializing in microbiome data analysis.
            Summarize the following NMDC data content, focusing on key findings and scientific implications.
            Highlight geographic and environmental patterns, and explain how these might impact microorganisms and their hosts."""
            
            response = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ])
            return response.strip()
        else:
            return "Unsupported content type for summarization"
            
    def _generate_llm_interpretation(self, df: pd.DataFrame, entity_type: str) -> str:
        """Generate an LLM interpretation of the dataset.
        
        Args:
            df: The DataFrame to interpret
            entity_type: Type of entity (biosample, study, etc.)
            
        Returns:
            LLM interpretation
        """
        # Generate statistical summaries for LLM context
        stats_summary = self._generate_stats_for_llm(df, entity_type)
        
        # Create prompt for LLM interpretation
        system_prompt = f"""You are a scientific assistant specializing in microbiome data analysis.
        
        The following contains statistical information about a dataset of {len(df)} {entity_type} records from the National Microbiome Data Collaborative (NMDC).
        
        Interpret this data, focusing on:
        1. Geographic patterns and their significance
        2. Environmental conditions represented in the dataset
        3. Scientific implications and potential research applications
        4. Any notable patterns or outliers in the measurements
        
        Use a scientific tone, but make your insights accessible. Avoid simply repeating the statistics - focus on their implications.
        """
        
        # Call LLM with statistics and prompt
        try:
            response = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": stats_summary}
            ])
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating LLM interpretation: {str(e)}")
            return "Error generating dataset interpretation."
            
    def _generate_stats_for_llm(self, df: pd.DataFrame, entity_type: str) -> str:
        """Generate statistical summaries formatted for LLM consumption.
        
        Args:
            df: The DataFrame to analyze
            entity_type: Type of entity (biosample, study, etc.)
            
        Returns:
            Formatted statistics string
        """
        stats = []
        
        # Add record count and basic info
        stats.append(f"Dataset: {len(df)} {entity_type} records")
        
        # 1. Geographic Distribution
        if 'geo_loc_name' in df.columns:
            locations = df['geo_loc_name'].value_counts().head(10)
            if not locations.empty:
                stats.append("\nGeographic Distribution:")
                for loc, count in locations.items():
                    if pd.notna(loc):
                        stats.append(f"- {loc}: {count} records ({(count/len(df))*100:.1f}%)")
        
        # 2. Environmental Categories
        env_columns = ['env_medium', 'env_broad_scale', 'env_local_scale', 
                       'ecosystem', 'ecosystem_category', 'ecosystem_subtype']
        
        for col in env_columns:
            if col in df.columns:
                # Handle dictionary values
                if df[col].apply(lambda x: isinstance(x, dict)).any():
                    values = df[col].apply(lambda x: self._extract_term_value(x) if isinstance(x, dict) else x)
                else:
                    values = df[col]
                    
                val_counts = values.value_counts().head(7)
                if not val_counts.empty:
                    stats.append(f"\n{col.replace('_', ' ').title()}:")
                    for val, count in val_counts.items():
                        if pd.notna(val):
                            stats.append(f"- {val}: {count} records ({(count/len(df))*100:.1f}%)")
        
        # 3. Numeric measurements
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            stats.append("\nKey Measurements:")
            for col in numeric_cols[:10]:  # Limit to top 10 numeric columns
                try:
                    stats_desc = df[col].describe()
                    stats.append(f"\n{col}:")
                    stats.append(f"- Count: {stats_desc['count']}")
                    stats.append(f"- Mean: {stats_desc['mean']:.2f}")
                    stats.append(f"- Min: {stats_desc['min']}")
                    stats.append(f"- Max: {stats_desc['max']}")
                    stats.append(f"- Standard Deviation: {stats_desc['std']:.2f}")
                except Exception as e:
                    logger.warning(f"Error calculating statistics for {col}: {str(e)}")
                    continue
        
        # Return formatted string
        return "\n".join(stats)