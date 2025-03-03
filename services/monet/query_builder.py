"""
Query builder for MONet service.

This module handles query validation, building, and execution for the MONet service.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
from datetime import datetime

from .models import MONetConfig, GeoPoint, GeoBBox, QueryResult
from .data_manager import MONetDataManager

class MONetQueryBuilder:
    """Handles query validation and execution."""
    
    def __init__(self, data_manager: MONetDataManager):
        """Initialize query builder."""
        self.data_manager = data_manager
        
    def validate_query(self, query: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate query structure and parameters.
        
        Enforces strict rules about query format and allowed operations.
        
        Args:
            query: Query dictionary with filters and/or geographic constraints
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(query, dict):
            return False, "Query must be a dictionary"
            
        # Check that at least one of filters or geo constraints is present
        if not any(key in query for key in ['filters', 'geo_point', 'geo_bbox']):
            return False, "Query must contain at least one of: filters, geo_point, or geo_bbox"
            
        # Check for disallowed fields
        disallowed_fields = {'sort', 'limit', 'fields', '$or', '$and'}
        if any(field in query for field in disallowed_fields):
            return False, f"Query contains disallowed fields. Only 'filters', 'geo_point', and 'geo_bbox' are allowed."
            
        # Validate filters if present
        if 'filters' in query:
            if not isinstance(query['filters'], list):
                return False, "'filters' must be a list of filter groups"
                
            for filter_group in query['filters']:
                if not isinstance(filter_group, dict):
                    return False, "Each filter group must be a dictionary"
                    
                for column, conditions in filter_group.items():
                    # Validate column exists
                    if column not in self.data_manager.unified_df.columns:
                        return False, f"Column '{column}' does not exist in the dataset"
                        
                    if not isinstance(conditions, list):
                        return False, f"Conditions for column '{column}' must be a list"
                        
                    # Get column type and allowed operations
                    if column.endswith('_has_numeric_value'):
                        allowed_ops = {'>', '<', '>=', '<=', '==', 'range'}
                    elif pd.api.types.is_datetime64_any_dtype(self.data_manager.unified_df[column]):
                        allowed_ops = {'>', '<', '==', 'range'}
                    else:  # text
                        allowed_ops = {'contains', 'exact', 'starts_with'}
                        
                    for condition in conditions:
                        if not isinstance(condition, dict):
                            return False, "Each condition must be a dictionary"
                            
                        if not all(key in condition for key in ['operation', 'value']):
                            return False, "Each condition must have 'operation' and 'value' keys"
                            
                        op = condition['operation']
                        val = condition['value']
                        
                        # Validate operation
                        if op not in allowed_ops:
                            return False, f"Invalid operation '{op}' for column '{column}'. Allowed: {allowed_ops}"
                            
                        # Validate value type
                        if op == 'range':
                            if not isinstance(val, list) or len(val) != 2:
                                return False, f"Range operation requires [min, max] array for column '{column}'"
                            if column.endswith('_has_numeric_value'):
                                if not all(isinstance(v, (int, float)) for v in val):
                                    return False, f"Range values must be numbers for column '{column}'"
                        elif column.endswith('_has_numeric_value'):
                            if not isinstance(val, (int, float)):
                                return False, f"Value must be a number for column '{column}'"
                                
        # Validate geo_point if present
        if 'geo_point' in query:
            point = query['geo_point']
            if not isinstance(point, dict):
                return False, "geo_point must be a dictionary"
                
            if not all(key in point for key in ['latitude', 'longitude']):
                return False, "geo_point must contain latitude and longitude"
                
            try:
                lat = float(point['latitude'])
                lon = float(point['longitude'])
                if not (-90 <= lat <= 90):
                    return False, "Latitude must be between -90 and 90"
                if not (-180 <= lon <= 180):
                    return False, "Longitude must be between -180 and 180"
                if 'radius_km' in point:
                    radius = float(point['radius_km'])
                    if radius <= 0:
                        return False, "radius_km must be positive"
            except (ValueError, TypeError):
                return False, "Geographic coordinates must be valid numbers"
                
        # Validate geo_bbox if present
        if 'geo_bbox' in query:
            bbox = query['geo_bbox']
            if not isinstance(bbox, dict):
                return False, "geo_bbox must be a dictionary"
                
            required = ['min_lat', 'max_lat', 'min_lon', 'max_lon']
            if not all(key in bbox for key in required):
                return False, f"geo_bbox must contain: {', '.join(required)}"
                
            try:
                min_lat = float(bbox['min_lat'])
                max_lat = float(bbox['max_lat'])
                min_lon = float(bbox['min_lon'])
                max_lon = float(bbox['max_lon'])
                
                if not (-90 <= min_lat <= max_lat <= 90):
                    return False, "Invalid latitude range in bbox"
                if not (-180 <= min_lon <= max_lon <= 180):
                    return False, "Invalid longitude range in bbox"
            except (ValueError, TypeError):
                return False, "Geographic coordinates must be valid numbers"
                
        return True, "Query is valid"
    
    def execute_query(self, query: Dict[str, Any]) -> QueryResult:
        """Execute a validated query.
        
        Args:
            query: Validated query dictionary
            
        Returns:
            QueryResult containing DataFrame and metadata
        """
        # Get base DataFrame
        df = self.data_manager.unified_df
        
        # Apply geographic filtering if specified
        if 'geo_point' in query:
            point = GeoPoint(**query['geo_point'])
            df = self.data_manager.get_geo_filtered(point=point)
        elif 'geo_bbox' in query:
            bbox = GeoBBox(**query['geo_bbox'])
            df = self.data_manager.get_geo_filtered(bbox=bbox)
        
        # Apply column filters
        if 'filters' in query:
            df = self._apply_filters(df, query['filters'])
        
        # Generate query description
        description = self._generate_query_description(query)
        
        # Create metadata
        metadata = {
            'query': query,
            'execution_time': datetime.now().isoformat(),
            'total_rows': len(df),
            'columns': list(df.columns)
        }
        
        return QueryResult(
            dataframe=df,
            metadata=metadata,
            description=description
        )
    
    def _validate_filters(self, filters: List[Dict]) -> Tuple[bool, str]:
        """Validate filter specifications."""
        if not isinstance(filters, list):
            return False, "filters must be a list"
            
        for filter_dict in filters:
            if not isinstance(filter_dict, dict):
                return False, "Each filter must be a dictionary"
                
            for column, conditions in filter_dict.items():
                if not isinstance(conditions, list):
                    return False, f"Conditions for column {column} must be a list"
                    
                for condition in conditions:
                    if not isinstance(condition, dict):
                        return False, f"Each condition for column {column} must be a dictionary"
                    if 'operation' not in condition or 'value' not in condition:
                        return False, f"Each condition must have 'operation' and 'value' keys"
                        
        return True, "Filters are valid"
    
    def _validate_geo_point(self, point: Dict) -> Tuple[bool, str]:
        """Validate geographic point specification."""
        if not isinstance(point, dict):
            return False, "geo_point must be a dictionary"
            
        if not all(key in point for key in ['latitude', 'longitude']):
            return False, "geo_point must contain latitude and longitude"
            
        try:
            lat = float(point['latitude'])
            lon = float(point['longitude'])
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return False, "Invalid latitude/longitude values"
        except (ValueError, TypeError):
            return False, "latitude and longitude must be numbers"
            
        if 'radius_km' in point:
            try:
                radius = float(point['radius_km'])
                if radius <= 0:
                    return False, "radius_km must be positive"
            except (ValueError, TypeError):
                return False, "radius_km must be a number"
                
        return True, "Geographic point is valid"
    
    def _validate_geo_bbox(self, bbox: Dict) -> Tuple[bool, str]:
        """Validate geographic bounding box specification."""
        if not isinstance(bbox, dict):
            return False, "geo_bbox must be a dictionary"
            
        required = ['min_lat', 'max_lat', 'min_lon', 'max_lon']
        if not all(key in bbox for key in required):
            return False, f"geo_bbox must contain: {', '.join(required)}"
            
        try:
            min_lat = float(bbox['min_lat'])
            max_lat = float(bbox['max_lat'])
            min_lon = float(bbox['min_lon'])
            max_lon = float(bbox['max_lon'])
            
            if not (-90 <= min_lat <= max_lat <= 90):
                return False, "Invalid latitude range"
            if not (-180 <= min_lon <= max_lon <= 180):
                return False, "Invalid longitude range"
        except (ValueError, TypeError):
            return False, "All bbox coordinates must be numbers"
            
        return True, "Geographic bounding box is valid"
    
    def _apply_filters(self, df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
        """Apply filters to DataFrame."""
        if not filters:
            return df
            
        # Process each filter set (combined with OR)
        filtered_dfs = []
        
        for filter_set in filters:
            # Start with all rows
            mask = pd.Series(True, index=df.index)
            
            # Apply each column's conditions (combined with AND)
            for column, conditions in filter_set.items():
                for condition in conditions:
                    operation = condition['operation']
                    value = condition['value']
                    
                    # Skip if column doesn't exist
                    if column not in df.columns:
                        print(f"Warning: Column {column} not found")
                        continue
                    
                    # Apply operation based on column type
                    if pd.api.types.is_numeric_dtype(df[column]):
                        if operation == 'range':
                            if not isinstance(value, list) or len(value) != 2:
                                print(f"Warning: Invalid range value for {column}")
                                continue
                            mask &= (df[column] >= value[0]) & (df[column] <= value[1])
                        elif operation in ['>', '<', '>=', '<=', '==']:
                            mask &= eval(f"df[column] {operation} value")
                    elif pd.api.types.is_datetime64_any_dtype(df[column]):
                        try:
                            if operation == 'range':
                                start = pd.to_datetime(value[0])
                                end = pd.to_datetime(value[1])
                                mask &= (df[column] >= start) & (df[column] <= end)
                            else:
                                compare_date = pd.to_datetime(value)
                                mask &= eval(f"df[column] {operation} compare_date")
                        except Exception as e:
                            print(f"Warning: Error processing date filter for {column}: {e}")
                            continue
                    else:  # String/categorical
                        if operation == 'contains':
                            mask &= df[column].str.contains(str(value), case=False, na=False)
                        elif operation == 'exact':
                            mask &= df[column].str.lower() == str(value).lower()
                        elif operation == 'starts_with':
                            mask &= df[column].str.startswith(str(value), na=False)
            
            # Add filtered subset to results
            filtered_df = df[mask]
            if not filtered_df.empty:
                filtered_dfs.append(filtered_df)
        
        # Combine results (union of all filter sets)
        if not filtered_dfs:
            return pd.DataFrame(columns=df.columns)
        
        return pd.concat(filtered_dfs, ignore_index=True).drop_duplicates()
    
    def _generate_query_description(self, query: Dict) -> str:
        """Generate human-readable description of the query."""
        parts = []
        
        # Describe geographic constraints
        if 'geo_point' in query:
            point = query['geo_point']
            radius = point.get('radius_km', 0)
            parts.append(
                f"Locations within {radius}km of "
                f"({point['latitude']}, {point['longitude']})"
            )
        elif 'geo_bbox' in query:
            bbox = query['geo_bbox']
            parts.append(
                f"Locations within bounding box: "
                f"{bbox['min_lat']}-{bbox['max_lat']}°N, "
                f"{bbox['min_lon']}-{bbox['max_lon']}°E"
            )
        
        # Describe filters
        if 'filters' in query:
            for i, filter_set in enumerate(query['filters'], 1):
                conditions = []
                for column, specs in filter_set.items():
                    for spec in specs:
                        conditions.append(
                            f"{column} {spec['operation']} {spec['value']}"
                        )
                parts.append(f"Filter set {i}: " + " AND ".join(conditions))
        
        return " OR ".join(parts) if parts else "No filters applied" 