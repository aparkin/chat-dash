"""
Query Builder for NMDC API.

This module provides functionality for constructing and validating queries
against the NMDC API.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import re
import logging
from .schema_manager import SchemaManager

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class QueryCondition:
    """Represents a single query condition."""
    field: str
    operator: str
    value: Any
    
    def to_filter_string(self) -> str:
        """Convert to NMDC filter string format."""
        value_str = str(self.value)
        
        # Quote string values with spaces if not already quoted
        if ' ' in value_str and not (value_str.startswith('"') and value_str.endswith('"')):
            value_str = f'"{value_str}"'
        
        # Format with colon for NMDC API
        if self.operator == '=':
            return f"{self.field}:{value_str}"
        else:
            return f"{self.field}{self.operator}{value_str}"

@dataclass
class NMDCQuery:
    """Represents a query against the NMDC API."""
    endpoint: str
    conditions: List[QueryCondition] = field(default_factory=list)
    per_page: int = 100
    cursor: str = "*"
    max_pages: int = 5
    
    def to_api_query(self) -> Dict[str, Any]:
        """Convert to NMDC API query format."""
        # Build filter string by joining conditions with 'and'
        filter_str = " and ".join(cond.to_filter_string() for cond in self.conditions)
        
        return {
            "endpoint": self.endpoint,
            "filter": filter_str,
            "per_page": self.per_page,
            "cursor": self.cursor,
            "max_pages": self.max_pages
        }

class QueryBuilder:
    """Builds and validates queries against NMDC API."""
    
    def __init__(self, schema_manager: SchemaManager):
        """Initialize the query builder.
        
        Args:
            schema_manager: SchemaManager instance for schema information
        """
        self.schema_manager = schema_manager
    
    def create_query(self, entity_type: str) -> NMDCQuery:
        """Create a new query for the specified entity type.
        
        Args:
            entity_type: Type of entity (e.g., 'biosample', 'study')
            
        Returns:
            NMDCQuery object
        """
        # Format endpoint with proper pluralization
        if entity_type.endswith('y'):
            # Words ending in 'y' typically change to 'ies' in plural
            endpoint = f"/{entity_type[:-1]}ies"
        else:
            # Default is to add 's'
            endpoint = f"/{entity_type}s"
        
        return NMDCQuery(endpoint=endpoint)
    
    def add_condition(self, query: NMDCQuery, field: str, operator: str, value: Any) -> NMDCQuery:
        """Add a condition to the query.
        
        Args:
            query: Existing query
            field: Field to filter on
            operator: Operator (=, >, <, >=, <=)
            value: Value to filter by
            
        Returns:
            Updated query
        """
        # Validate field against schema
        self._validate_field(query.endpoint, field)
        
        # Add condition
        condition = QueryCondition(field=field, operator=operator, value=value)
        query.conditions.append(condition)
        
        return query
    
    def set_pagination(self, query: NMDCQuery, per_page: int, max_pages: int = 5) -> NMDCQuery:
        """Set pagination parameters for the query.
        
        Args:
            query: Existing query
            per_page: Number of results per page
            max_pages: Maximum number of pages to fetch
            
        Returns:
            Updated query
        """
        query.per_page = per_page
        query.max_pages = max_pages
        
        return query
    
    def validate_query(self, query: NMDCQuery) -> Tuple[bool, Optional[str]]:
        """Validate the query against the schema.
        
        Args:
            query: Query to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        try:
            # Validate endpoint
            if not query.endpoint:
                return False, "No endpoint specified"
            
            # Get schema
            schema = self.schema_manager.get_schema()
            
            # Check if endpoint exists
            if query.endpoint not in schema.get('endpoints', {}):
                return False, f"Invalid endpoint: {query.endpoint}"
            
            # Validate conditions
            for condition in query.conditions:
                field = condition.field
                
                # Validate field against schema
                valid, error = self._validate_field(query.endpoint, field)
                if not valid:
                    return False, error
            
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            return False, str(e)
    
    def from_api_query(self, api_query: Dict[str, Any]) -> NMDCQuery:
        """Create a query from an NMDC API query object.
        
        Args:
            api_query: API query dictionary
            
        Returns:
            NMDCQuery object
        """
        try:
            # Extract endpoint
            endpoint = api_query.get('endpoint')
            if not endpoint:
                raise ValueError("Query must include 'endpoint'")
            
            # Create query
            query = NMDCQuery(endpoint=endpoint)
            
            # Set pagination
            query.per_page = api_query.get('per_page', 100)
            query.cursor = api_query.get('cursor', '*')
            query.max_pages = api_query.get('max_pages', 5)
            
            # Parse filter
            filter_str = api_query.get('filter', '')
            if filter_str:
                # Parse filter string into conditions
                conditions = self._parse_filter_string(filter_str)
                query.conditions.extend(conditions)
            
            return query
            
        except Exception as e:
            logger.error(f"Error creating query from API query: {str(e)}")
            raise
    
    def _validate_field(self, endpoint: str, field: str) -> Tuple[bool, Optional[str]]:
        """Validate a field against the schema.
        
        Args:
            endpoint: API endpoint
            field: Field to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Extract entity type from endpoint
        entity_type = endpoint.strip('/').rstrip('s')
        
        # Get searchable attributes
        attributes = self.schema_manager.get_searchable_attributes(entity_type)
        attribute_names = [attr.name for attr in attributes]
        
        # Check if field exists
        if field not in attribute_names:
            return False, f"Invalid field for {entity_type}: {field}"
        
        return True, None
    
    def _parse_filter_string(self, filter_str: str) -> List[QueryCondition]:
        """Parse a filter string into QueryCondition objects.
        
        Args:
            filter_str: Filter string (e.g., "field1=value1 and field2>value2")
            
        Returns:
            List of QueryCondition objects
        """
        conditions = []
        
        if not filter_str:
            return conditions
        
        # Split by 'and'
        parts = filter_str.split(' and ')
        
        for part in parts:
            part = part.strip()
            
            # Match different operator formats
            match = re.match(r'^([^:=<>]+)([:=<>]=?)(.+)$', part)
            if match:
                field = match.group(1).strip()
                operator = match.group(2).strip()
                value = match.group(3).strip()
                
                # Convert : to = for our internal representation
                if operator == ':':
                    operator = '='
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                
                conditions.append(QueryCondition(field=field, operator=operator, value=value))
        
        return conditions 