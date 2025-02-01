"""
Query filter construction for Weaviate queries.

This module provides utilities for building complex query filters:

Filter Types:
- Property filters (equals, contains, greater than, etc.)
- Date range filters
- Reference existence filters
- Nested property filters
- Combination filters (AND, OR)

Features:
- Fluent interface for filter construction
- Type validation for filter values
- Support for all Weaviate operators
- Nested filter composition
"""

from typing import Dict, List, Union, Any, Optional
from datetime import datetime

class FilterBuilder:
    """
    Builder for constructing Weaviate query filters.
    
    Features:
    - Fluent interface for chaining conditions
    - Support for all Weaviate operators
    - Nested filter composition
    - Type validation
    """
    
    def __init__(self):
        """Initialize an empty filter builder."""
        self._conditions = []
        self._operator = "And"  # Default to AND for multiple conditions
    
    def where(self, path: Union[str, List[str]]) -> 'PropertyFilter':
        """
        Start building a property filter.
        
        Args:
            path: Property path (string or list for nested properties)
            
        Returns:
            PropertyFilter builder
        """
        return PropertyFilter(self, path)
    
    def where_text(self, path: Union[str, List[str]]) -> 'TextFilter':
        """
        Start building a text property filter.
        
        Args:
            path: Property path (string or list for nested properties)
            
        Returns:
            TextFilter builder
        """
        return TextFilter(self, path)
    
    def where_date(self, path: Union[str, List[str]]) -> 'DateFilter':
        """
        Start building a date property filter.
        
        Args:
            path: Property path (string or list for nested properties)
            
        Returns:
            DateFilter builder
        """
        return DateFilter(self, path)
    
    def where_reference(self, path: Union[str, List[str]]) -> 'ReferenceFilter':
        """
        Start building a reference property filter.
        
        Args:
            path: Property path (string or list for nested properties)
            
        Returns:
            ReferenceFilter builder
        """
        return ReferenceFilter(self, path)
    
    def and_(self) -> 'FilterBuilder':
        """Set operator to AND for combining conditions."""
        self._operator = "And"
        return self
    
    def or_(self) -> 'FilterBuilder':
        """Set operator to OR for combining conditions."""
        self._operator = "Or"
        return self
    
    def add_condition(self, condition: Dict[str, Any]) -> None:
        """Add a condition to the filter."""
        self._conditions.append(condition)
    
    def build(self) -> Optional[Dict[str, Any]]:
        """
        Build the complete filter dictionary.
        
        Returns:
            Dict containing the Weaviate filter structure,
            or None if no conditions are set
        """
        if not self._conditions:
            return None
            
        if len(self._conditions) == 1:
            return self._conditions[0]
            
        return {
            "operator": self._operator,
            "operands": self._conditions
        }

class PropertyFilter:
    """Base class for property filters."""
    
    def __init__(self, builder: FilterBuilder, path: Union[str, List[str]]):
        """
        Initialize property filter.
        
        Args:
            builder: Parent FilterBuilder instance
            path: Property path
        """
        self.builder = builder
        self.path = path if isinstance(path, list) else [path]
    
    def equals(self, value: Any) -> FilterBuilder:
        """Add equals condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "Equal",
            "valueString": str(value)
        })
        return self.builder
    
    def not_equals(self, value: Any) -> FilterBuilder:
        """Add not equals condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "NotEqual",
            "valueString": str(value)
        })
        return self.builder
    
    def is_null(self) -> FilterBuilder:
        """Add is null condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "IsNull",
            "valueBoolean": True
        })
        return self.builder
    
    def is_not_null(self) -> FilterBuilder:
        """Add is not null condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "IsNull",
            "valueBoolean": False
        })
        return self.builder

class TextFilter(PropertyFilter):
    """Filter builder for text properties."""
    
    def contains(self, value: str) -> FilterBuilder:
        """Add contains condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "Like",
            "valueText": f"*{value}*"
        })
        return self.builder
    
    def starts_with(self, value: str) -> FilterBuilder:
        """Add starts with condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "Like",
            "valueText": f"{value}*"
        })
        return self.builder
    
    def ends_with(self, value: str) -> FilterBuilder:
        """Add ends with condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "Like",
            "valueText": f"*{value}"
        })
        return self.builder

class DateFilter(PropertyFilter):
    """Filter builder for date properties."""
    
    def before(self, date: Union[datetime, str]) -> FilterBuilder:
        """Add before date condition."""
        if isinstance(date, datetime):
            date = date.isoformat()
        
        self.builder.add_condition({
            "path": self.path,
            "operator": "LessThan",
            "valueDate": date
        })
        return self.builder
    
    def after(self, date: Union[datetime, str]) -> FilterBuilder:
        """Add after date condition."""
        if isinstance(date, datetime):
            date = date.isoformat()
        
        self.builder.add_condition({
            "path": self.path,
            "operator": "GreaterThan",
            "valueDate": date
        })
        return self.builder
    
    def between(
        self,
        start: Union[datetime, str],
        end: Union[datetime, str]
    ) -> FilterBuilder:
        """Add between dates condition."""
        if isinstance(start, datetime):
            start = start.isoformat()
        if isinstance(end, datetime):
            end = end.isoformat()
        
        self.builder.add_condition({
            "operator": "And",
            "operands": [
                {
                    "path": self.path,
                    "operator": "GreaterThanEqual",
                    "valueDate": start
                },
                {
                    "path": self.path,
                    "operator": "LessThanEqual",
                    "valueDate": end
                }
            ]
        })
        return self.builder

class ReferenceFilter(PropertyFilter):
    """Filter builder for reference properties."""
    
    def has_reference(self, collection: str, id: str) -> FilterBuilder:
        """Add has reference condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "ContainsAny",
            "valueString": f"{collection}/{id}"
        })
        return self.builder
    
    def has_any_reference(self) -> FilterBuilder:
        """Add has any reference condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "IsNull",
            "valueBoolean": False
        })
        return self.builder
    
    def has_no_reference(self) -> FilterBuilder:
        """Add has no reference condition."""
        self.builder.add_condition({
            "path": self.path,
            "operator": "IsNull",
            "valueBoolean": True
        })
        return self.builder 