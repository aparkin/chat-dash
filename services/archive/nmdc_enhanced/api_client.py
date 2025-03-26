"""
API Client for NMDC API.

This module provides a client for interacting with the NMDC API endpoints,
handling requests, responses, and data formatting.
"""

import logging
import requests
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import asyncio
import inspect

# Configure logging
logger = logging.getLogger(__name__)

class NMDCApiClient:
    """Client for accessing NMDC API endpoints."""
    
    # API base URL
    API_BASE_URL = "https://api.microbiomedata.org"
    
    # Maximum number of concurrent requests
    MAX_CONCURRENT_REQUESTS = 5
    # Default per_page value for pagination
    DEFAULT_PER_PAGE = 1000
    
    def __init__(self, timeout: int = 30):
        """Initialize the NMDC API Client.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.session = requests.Session()
        
        # Add common headers
        self.session.headers.update({
            "Accept": "application/json"
        })
        
        logger.info("NMDC API Client initialized")
    
    async def get_api_schema(self) -> Dict[str, Any]:
        """Get schema information from the API summary endpoint.
        
        This method fetches the complete schema information including all entity types,
        their attributes, and attribute statistics (counts, types, etc.).
        
        Returns:
            Dictionary containing the complete schema information
        """
        # Check if we already have schema in cache
        if hasattr(self, '_schema_cache') and self._schema_cache is not None:
            logger.debug("Using cached schema data")
            return self._schema_cache
            
        # Try both possible API URLs
        urls = [
            "https://data.microbiomedata.org/api/summary",  # This is the known working URL, try it first
            f"{self.API_BASE_URL}/summary"
        ]
        
        for url in urls:
            try:
                logger.info(f"Fetching schema information from {url}")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                schema = response.json()
                
                logger.info(f"Successfully retrieved schema with {len(schema)} entity types from {url}")
                # Cache the schema
                self._schema_cache = schema
                return schema
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error getting schema from {url}: {str(e)}")
                continue
        
        # If we get here, both URLs failed
        logger.error("Failed to retrieve schema from any endpoint")
        
        # Return a minimal default schema to allow the system to function
        default_schema = self._get_default_schema()
        self._schema_cache = default_schema
        return default_schema
    
    def _get_default_schema(self) -> Dict[str, Any]:
        """Get a default minimal schema when API is unavailable.
        
        Returns:
            Dictionary with minimal schema information
        """
        return {
            "study": {"total": 0, "attributes": {}},
            "biosample": {"total": 0, "attributes": {}},
            "data_object": {"total": 0, "attributes": {}}
        }
    
    async def get_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a single entity by ID.
        
        Args:
            entity_type: Type of entity (e.g., study, biosample, data_object)
            entity_id: ID of the entity
            
        Returns:
            Entity data as dictionary or None if not found
        """
        # Convert entity_type to plural form for API endpoint
        entity_endpoint = self._get_entity_endpoint(entity_type)
        url = f"{self.API_BASE_URL}/{entity_endpoint}/{entity_id}"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Entity not found: {entity_type}/{entity_id}")
                return None
            logger.error(f"HTTP error getting entity {entity_type}/{entity_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error getting entity {entity_type}/{entity_id}: {str(e)}")
            raise
    
    async def search_entities(self, 
                        entity_type: str, 
                        conditions: List[Dict[str, Any]] = None,
                        page: int = 1, 
                        per_page: int = None,
                        sort_field: str = None,
                        sort_direction: str = "asc") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Search for entities matching given conditions.
        
        Args:
            entity_type: Type of entity to search for
            conditions: List of condition dictionaries with field, operator, and value
            page: Page number (for compatibility, not used with cursor pagination)
            per_page: Number of results per page
            sort_field: Field to sort by (not used currently)
            sort_direction: Sort direction (not used currently)
            
        Returns:
            Tuple of (DataFrame of results, metadata dictionary)
        """
        # Convert entity_type to plural form for API endpoint
        entity_endpoint = self._get_entity_endpoint(entity_type)
        if per_page is None:
            per_page = self.DEFAULT_PER_PAGE
        
        # Convert conditions to filter string format
        filter_str = self._conditions_to_filter_str(conditions)
        
        logger.info(f"Searching {entity_type} with filter: {filter_str}")
        
        # Use cursor pagination to get all results
        cursor = "*"
        all_results = []
        
        try:
            # First make a test request with a small limit to check if the endpoint works
            test_url = f"{self.API_BASE_URL}/{entity_endpoint}?per_page=1&cursor={cursor}"
            if filter_str:
                test_url += f"&filter={filter_str}"
                
            logger.info(f"Testing API endpoint with: {test_url}")
            test_response = self.session.get(test_url, timeout=self.timeout)
            test_response.raise_for_status()
            
            # If we got here, the endpoint works
            logger.info(f"API endpoint test successful, proceeding with full query")
            
            while True:
                url = f"{self.API_BASE_URL}/{entity_endpoint}?per_page={per_page}&cursor={cursor}"
                if filter_str:
                    url += f"&filter={filter_str}"
                
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                results = data.get("results", [])
                all_results.extend(results)
                
                # Get next cursor
                meta = data.get("meta", {})
                cursor = meta.get("next_cursor", "")
                
                # Log progress
                logger.info(f"Retrieved {len(results)} results, total so far: {len(all_results)}")
                
                if not cursor:
                    break
                    
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error searching {entity_type}: {str(e)}")
            if e.response.status_code == 422:
                logger.error(f"API rejected the request with 422 Unprocessable Entity. Filter: {filter_str}")
                # Add more detailed error info if available
                try:
                    error_detail = e.response.json()
                    logger.error(f"API error details: {error_detail}")
                except:
                    pass
            elif e.response.status_code == 500:
                logger.error(f"Server error (500) when accessing {entity_type}. This might be due to an invalid filter or server issues.")
                # Try one more time without the filter as a fallback
                if filter_str:
                    logger.info(f"Retrying request without filter as fallback")
                    try:
                        fallback_url = f"{self.API_BASE_URL}/{entity_endpoint}?per_page=10&cursor=*"
                        fallback_response = self.session.get(fallback_url, timeout=self.timeout)
                        fallback_response.raise_for_status()
                        fallback_data = fallback_response.json()
                        all_results = fallback_data.get("results", [])
                        # If this worked, let's return what we have with a warning
                        logger.warning(f"Fallback request without filter succeeded but may not match the original query criteria")
                    except Exception as fallback_error:
                        logger.error(f"Fallback request also failed: {str(fallback_error)}")
            raise
        except Exception as e:
            logger.error(f"Error searching {entity_type}: {str(e)}")
            raise
        
        # Convert results to DataFrame
        if all_results:
            # Debug: Check structure of first few results to ensure dictionaries are preserved
            for i, result in enumerate(all_results[:5]):
                for key, value in result.items():
                    if isinstance(value, dict):
                        logger.debug(f"Result {i}, key '{key}' is a dictionary with keys: {list(value.keys())}")
                    elif isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                        logger.warning(f"Result {i}, key '{key}' looks like a serialized JSON string: {value[:100]}")
                    elif isinstance(value, list) and len(value) > 0:
                        first_item = value[0]
                        if isinstance(first_item, dict):
                            logger.debug(f"Result {i}, key '{key}' is a list of dictionaries, first has keys: {list(first_item.keys())}")
            
            # Create DataFrame directly from the results list
            # This should preserve dictionaries as is without serialization
            df = pd.DataFrame(all_results)
            
            # Post-creation check to verify dictionary preservation
            for col in df.columns:
                # Check first non-null value if it exists
                sample_val = df[col].dropna().head(1)
                if not sample_val.empty:
                    val = sample_val.iloc[0]
                    if isinstance(val, dict):
                        logger.debug(f"Column '{col}' contains dictionary values after DataFrame creation")
                    elif isinstance(val, str) and val.startswith('{') and val.endswith('}'):
                        # This would indicate Pandas converted our dictionaries to strings
                        logger.warning(f"Column '{col}' contains string that looks like JSON after DataFrame creation: {val[:100]}")
                        # Try to convert back to dictionary if it's a simple case
                        try:
                            # Test if conversion would work
                            test_dict = json.loads(val)
                            if isinstance(test_dict, dict):
                                logger.info(f"Could parse JSON string in column '{col}' - this indicates premature serialization")
                                
                                # Attempt to fix the serialization in place
                                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and 
                                                      x.startswith('{') and x.endswith('}') else x)
                                logger.info(f"Fixed serialized dictionaries in column '{col}'")
                        except:
                            pass
        else:
            df = pd.DataFrame()
            
        # Create metadata
        metadata = {
            "total": len(all_results),
            "per_page": per_page
        }
        
        return df, metadata
    
    def _conditions_to_filter_str(self, conditions: List[Dict[str, Any]]) -> str:
        """Convert conditions list to filter string format.
        
        Args:
            conditions: List of condition dictionaries with field, operator, and value
            
        Returns:
            Filter string for API request
        """
        if not conditions:
            return ""
            
        filters = []
        for condition in conditions:
            field = condition.get("field", "")
            operator = condition.get("op", "")  # Changed from "operator" to "op"
            value = condition.get("value", "")
            
            # Skip empty values
            if not field:
                continue
                
            # Special field mappings
            if field == "env_medium" and operator == "search":
                # Use specific_ecosystem instead of env_medium for better results
                field = "specific_ecosystem"
                
            # Make sure value is properly formatted
            if value is not None and not isinstance(value, bool):
                # URL encode the value
                if isinstance(value, str):
                    # Remove any special characters that might cause issues
                    value = re.sub(r'[^\w\s]', '', value).strip()
            
            # Map our internal operators to API filter formats
            if operator == "search":
                if value:
                    filters.append(f"{field}.search:{value}")
            elif operator == "eq":
                if value is not None:
                    filters.append(f"{field}.eq:{value}")
            elif operator == "exists":
                # For exists operator, just specify the field
                filters.append(f"{field}")
            elif operator == "gt":
                if value is not None:
                    filters.append(f"{field}.gt:{value}")
            elif operator == "lt":
                if value is not None:
                    filters.append(f"{field}.lt:{value}")
            elif operator == "gte":
                if value is not None:
                    filters.append(f"{field}.gte:{value}")
            elif operator == "lte":
                if value is not None:
                    filters.append(f"{field}.lte:{value}")
            else:
                # Default to search if operator is not recognized
                if value:
                    filters.append(f"{field}.search:{value}")
        
        # Combine multiple filters with AND (comma-separated)
        return ",".join(filters)
    
    def _get_entity_endpoint(self, entity_type: str) -> str:
        """Convert entity type to plural endpoint name.
        
        Args:
            entity_type: Entity type name
            
        Returns:
            Plural form for endpoint
        """
        # Map entity types to their endpoint names
        entity_map = {
            "study": "studies",
            "biosample": "biosamples",
            "data_object": "data_objects"
        }
        
        # Return mapped endpoint or pluralize by adding 's'
        return entity_map.get(entity_type, f"{entity_type}s")
    
    async def get_related_entities(self, 
                            entity_type: str, 
                            entity_id: str, 
                            related_entity_type: str) -> pd.DataFrame:
        """Get related entities for a given entity.
        
        Args:
            entity_type: Type of the source entity
            entity_id: ID of the source entity
            related_entity_type: Type of related entities to retrieve
            
        Returns:
            DataFrame of related entities
        """
        # For now, handle specific relationships
        # For biosamples related to a study
        if entity_type == "study" and related_entity_type == "biosample":
            # Get biosamples with associated_studies containing this study ID
            filter_str = f"associated_studies.search:{entity_id}"
            df, _ = await self.search_entities("biosample", [{
                "field": "associated_studies", 
                "op": "search", 
                "value": entity_id
            }])
            return df
        
        # For data objects related to a biosample
        if entity_type == "biosample" and related_entity_type == "data_object":
            # Get data objects with associated_biosamples containing this biosample ID
            filter_str = f"associated_biosamples.search:{entity_id}"
            df, _ = await self.search_entities("data_object", [{
                "field": "associated_biosamples", 
                "op": "search", 
                "value": entity_id
            }])
            return df
            
        # If not a known relationship, return empty DataFrame
        logger.warning(f"Unknown relationship: {entity_type} to {related_entity_type}")
        return pd.DataFrame()
    
    async def get_environmental_data(self, conditions: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get environmental data matching given conditions.
        
        Args:
            conditions: List of condition dictionaries with field, operator, and value
            
        Returns:
            DataFrame of environmental data
        """
        # For now, get biosamples and extract environmental data
        biosamples_df, _ = await self.search_entities("biosample", conditions)
        
        if biosamples_df.empty:
            return pd.DataFrame()
            
        # Extract environmental fields
        env_fields = ["ecosystem", "ecosystem_category", "ecosystem_type", 
                     "ecosystem_subtype", "specific_ecosystem", "env_broad_scale", 
                     "env_local_scale", "env_medium"]
        
        # Filter to only include environmental fields that exist
        env_fields = [f for f in env_fields if f in biosamples_df.columns]
        
        if not env_fields:
            return pd.DataFrame()
            
        env_df = biosamples_df[env_fields].copy()
        
        # Count occurrences of each unique combination
        env_counts = env_df.groupby(env_fields).size().reset_index(name="count")
        
        return env_counts
    
    def _process_environmental_data(self, nodes: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
        """Process environmental data nodes and links into a structured DataFrame.
        
        Args:
            nodes: DataFrame of environmental nodes
            links: DataFrame of links between nodes
            
        Returns:
            Processed DataFrame of environmental data
        """
        if 'id' in nodes.columns and 'name' in nodes.columns:
            node_map = dict(zip(nodes['id'], nodes['name']))
            
            if not links.empty and 'source' in links.columns and 'target' in links.columns:
                links['source_name'] = links['source'].map(node_map)
                links['target_name'] = links['target'].map(node_map)
                
                result = links[['source_name', 'target_name', 'value']].copy()
                result.columns = ['parent', 'child', 'count']
                return result
        
        return pd.DataFrame()
    
    async def get_data_objects_by_study(self, study_id: str, data_object_type: Optional[str] = None) -> pd.DataFrame:
        """Get data objects for a specific study.
        
        Args:
            study_id: ID of the study
            data_object_type: Optional type of data objects to filter by
            
        Returns:
            DataFrame of data objects
        """
        # First get biosamples associated with this study
        biosamples_df = await self.get_related_entities("study", study_id, "biosample")
        
        if biosamples_df.empty:
            logger.warning(f"No biosamples found for study {study_id}")
            return pd.DataFrame()
        
        biosample_ids = biosamples_df['id'].tolist()
        
        # Get data objects for each biosample
        data_objects_dfs = []
        
        for biosample_id in biosample_ids:
            try:
                data_objects_df = await self.get_related_entities("biosample", biosample_id, "data_object")
                if not data_objects_df.empty:
                    # Add biosample_id column for reference
                    data_objects_df['biosample_id'] = biosample_id
                    
                    # Filter by data object type if specified
                    if data_object_type and 'type' in data_objects_df.columns:
                        data_objects_df = data_objects_df[
                            data_objects_df['type'].str.contains(data_object_type, case=False, na=False)
                        ]
                        
                    data_objects_dfs.append(data_objects_df)
            except Exception as e:
                logger.error(f"Error getting data objects for biosample {biosample_id}: {str(e)}")
        
        if data_objects_dfs:
            combined_df = pd.concat(data_objects_dfs, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['id'])
            return combined_df
        
        return pd.DataFrame()
    
    async def get_supported_entities(self) -> List[str]:
        """Get list of entity types supported by the current API.
        
        Returns:
            List of entity type names that are supported
        """
        try:
            # Get schema to determine supported entities
            schema = await self.get_api_schema()
            # Extract entity types with non-zero counts
            entities = [entity_type for entity_type, info in schema.items() 
                       if isinstance(info, dict) and 
                       info.get("total", 0) > 0]
            return entities
        except Exception as e:
            logger.error(f"Error getting supported entities: {str(e)}")
            # Return default list if we can't get from schema
            return ["study", "biosample", "data_object"]
    
    async def search_studies_post(self, study_ids: List[str], limit: int = 100, offset: int = 0) -> Tuple[pd.DataFrame, int]:
        """Search for studies using the POST /study/search endpoint.
        
        This method uses the POST API endpoint which may be more reliable for searching
        multiple studies by ID.
        
        Args:
            study_ids: List of study IDs to search for
            limit: Maximum number of results per page
            offset: Starting offset for pagination
            
        Returns:
            Tuple of (DataFrame with results, total count)
        """
        # Format study IDs properly - ensure they have the proper prefix format
        formatted_ids = []
        for study_id in study_ids:
            # If ID is in the format nmdcstyXXXXXXXXXX, convert to nmdc:sty-XX-XXXXXXXX
            if study_id.startswith('nmdcsty'):
                parts = study_id[7:]
                formatted_id = f"nmdc:sty-{parts[:2]}-{parts[2:]}"
                formatted_ids.append(formatted_id)
            elif not study_id.startswith('nmdc:sty-'):
                # If no prefix at all, add it
                formatted_id = f"nmdc:sty-{study_id}"
                formatted_ids.append(formatted_id)
            else:
                # Already in correct format
                formatted_ids.append(study_id)
        
        logger.info(f"Searching for {len(formatted_ids)} studies using POST endpoint")
        
        # Build conditions for each study ID
        conditions = []
        for study_id in formatted_ids:
            conditions.append({
                "field": "id",
                "op": "eq",
                "value": study_id
            })
        
        # Build the request body
        request_body = {
            "conditions": conditions
        }
        
        # First try the data.microbiomedata.org URL which we know works
        urls = [
            "https://data.microbiomedata.org/api/study/search",
            f"{self.API_BASE_URL}/study/search"
        ]
        
        for url in urls:
            try:
                full_url = f"{url}?offset={offset}&limit={limit}"
                logger.info(f"POST request to: {full_url}")
                logger.debug(f"Request body: {json.dumps(request_body)}")
                
                # Make the POST request
                response = self.session.post(
                    full_url,
                    json=request_body,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract results and count
                results = data.get("results", [])
                total = data.get("count", len(results))
                
                logger.info(f"Retrieved {len(results)} studies out of {total} total")
                
                if not results:
                    logger.warning("No studies found matching the provided IDs")
                    return pd.DataFrame(), 0
                
                # Convert to DataFrame
                df = pd.DataFrame(results)
                
                # Debug: Check for dictionary columns
                for i, result in enumerate(results[:3]):
                    for key, value in result.items():
                        if isinstance(value, dict):
                            logger.debug(f"Study result {i}, key '{key}' is a dictionary with keys: {list(value.keys())}")
                        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            logger.debug(f"Study result {i}, key '{key}' is a list of dictionaries")
                
                return df, total
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching studies from {url}: {str(e)}")
                if url == urls[-1]:  # If this was the last URL to try
                    logger.error("All study search endpoints failed")
                    return pd.DataFrame(), 0
        
        # This should not be reached, but just in case
        return pd.DataFrame(), 0
    
    async def get_studies_individually(self, study_ids: List[str]) -> pd.DataFrame:
        """Get multiple studies by retrieving them one by one.
        
        This is a fallback method when the POST /study/search endpoint fails.
        It retrieves each study individually using the GET /study/{study_id} endpoint.
        
        Args:
            study_ids: List of study IDs to retrieve
            
        Returns:
            DataFrame containing the study data
        """
        logger.info(f"Retrieving {len(study_ids)} studies individually")
        
        # Format study IDs properly - ensure they have the proper prefix format
        formatted_ids = []
        for study_id in study_ids:
            # Extract just the ID part from biosample's associated_studies field
            # If ID contains the full nmdc:sty-XX-XXXXXX format, use as is
            if isinstance(study_id, str) and ':' in study_id:
                formatted_id = study_id
                formatted_ids.append(formatted_id)
                logger.debug(f"Using study ID as is: {formatted_id}")
            # If ID is in the format nmdcstyXXXXXXXXXX, convert to nmdc:sty-XX-XXXXXXXX
            elif isinstance(study_id, str) and study_id.startswith('nmdcsty'):
                parts = study_id[7:]
                formatted_id = f"nmdc:sty-{parts[:2]}-{parts[2:]}"
                formatted_ids.append(formatted_id)
                logger.debug(f"Converted study ID from {study_id} to {formatted_id}")
            elif isinstance(study_id, str) and not study_id.startswith('nmdc:sty-'):
                # If no prefix at all, add it
                formatted_id = f"nmdc:sty-{study_id}"
                formatted_ids.append(formatted_id)
                logger.debug(f"Added prefix to study ID: {formatted_id}")
            else:
                # Already in correct format
                formatted_ids.append(study_id)
                logger.debug(f"Using study ID as is: {study_id}")
        
        # URLs to try - use the known working URL first
        base_url = "https://data.microbiomedata.org/api/study"
        
        # Test if this URL works
        if formatted_ids:
            test_id = formatted_ids[0]
            test_url = f"{base_url}/{test_id}"
            try:
                logger.info(f"Testing study endpoint with: {test_url}")
                response = self.session.get(test_url, timeout=self.timeout)
                response.raise_for_status()
                logger.info(f"Successfully tested study endpoint")
            except Exception as e:
                logger.error(f"Error with study endpoint {test_url}: {str(e)}")
                logger.warning("Continuing with retrieval attempts despite test failure")
        
        # Retrieve studies one by one
        studies = []
        for study_id in formatted_ids:
            study_url = f"{base_url}/{study_id}"
            try:
                logger.info(f"Fetching study: {study_url}")
                response = self.session.get(study_url, timeout=self.timeout)
                response.raise_for_status()
                study_data = response.json()
                studies.append(study_data)
                logger.debug(f"Successfully retrieved study: {study_id}")
            except Exception as e:
                logger.warning(f"Error retrieving study {study_id}: {str(e)}")
                
        if not studies:
            logger.warning("No studies could be retrieved")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(studies)
        logger.info(f"Successfully retrieved {len(df)} studies individually")
        return df 