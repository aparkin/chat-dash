"""
DuckDB-based query engine for NMDC Enhanced service.

This module provides a reusable query engine pattern for services that use DataFrames
as their primary data source. It leverages DuckDB for SQL-like query capabilities.
"""

import duckdb
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Set
import logging
from datetime import datetime
import json
import traceback
import re
from .models import QueryResult

# Configure logging
logger = logging.getLogger(__name__)

class NMDCQueryEngine:
    """Query engine for NMDC Enhanced service using DuckDB.
    
    This engine provides:
    1. SQL-based querying of pandas DataFrames using DuckDB
    2. Query validation and error reporting
    3. Comprehensive statistics for query results
    4. Schema information for registered tables
    5. Query execution tracking and performance metrics
    """
    
    def __init__(self, memory_limit: Optional[str] = None):
        """Initialize the query engine.
        
        Args:
            memory_limit: Optional memory limit for DuckDB (e.g. '4GB')
        """
        self.conn = duckdb.connect(":memory:")
        if memory_limit:
            self.conn.execute(f"SET memory_limit='{memory_limit}'")
            
        self.registered_tables = {}
        self.table_schemas = {}
        self.query_history = []
        self.execution_stats = {}
    
    def register_dataframe(self, df: pd.DataFrame, name: str, replace: bool = True) -> bool:
        """Register a DataFrame as a DuckDB table.
        
        Args:
            df: DataFrame to register
            name: Name for the table
            replace: Whether to replace existing table with same name
            
        Returns:
            True if registration was successful
        """
        logger.info(f"Attempting to register DataFrame as '{name}'")
        
        # Validate DataFrame
        if df is None:
            logger.warning(f"Cannot register None as table '{name}'")
            return False
            
        if len(df) == 0:
            logger.warning(f"Cannot register empty DataFrame as table '{name}'")
            return False
            
        try:
            # Check if table already exists
            if name in self.registered_tables and not replace:
                logger.warning(f"Table '{name}' already exists and replace=False")
                return False
                
            # Clean up problematic column names if needed
            df_clean = df.copy()
            
            # Make column names safe for SQL
            original_columns = df_clean.columns.tolist()
            clean_columns = [col.replace(' ', '_').replace('-', '_') for col in original_columns]
            if original_columns != clean_columns:
                logger.info(f"Cleaning column names for table '{name}'")
                df_clean.columns = clean_columns
            
            # Handle problematic data types that might cause issues
            for col in df_clean.columns:
                # Check for mixed types in object columns that could cause issues
                if df_clean[col].dtype == 'object':
                    # Try to convert mixed types to string to avoid registration errors
                    try:
                        df_clean[col] = df_clean[col].astype(str)
                    except Exception as e:
                        logger.warning(f"Could not convert column '{col}' to string: {str(e)}")
                
            # Register DataFrame as table
            logger.info(f"Registering DataFrame with {len(df_clean)} rows as table '{name}'")
            self.conn.register(name, df_clean)
            self.registered_tables[name] = {
                "rows": len(df_clean),
                "columns": len(df_clean.columns),
                "registration_time": datetime.now().isoformat()
            }
            
            # Cache schema information
            self.table_schemas[name] = self._extract_schema(df_clean)
            
            logger.info(f"Successfully registered DataFrame as table '{name}' with {len(df_clean)} rows and {len(df_clean.columns)} columns")
            return True
            
        except Exception as e:
            logger.error(f"Error registering DataFrame as table '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try with a more defensive approach if original registration failed
            try:
                logger.info(f"Attempting fallback registration for table '{name}'")
                # Convert entire DataFrame to strings as a last resort
                str_df = df.astype(str)
                self.conn.register(name, str_df)
                self.registered_tables[name] = {
                    "rows": len(str_df),
                    "columns": len(str_df.columns),
                    "registration_time": datetime.now().isoformat(),
                    "fallback_registration": True
                }
                
                # Cache schema information
                self.table_schemas[name] = self._extract_schema(str_df)
                
                logger.info(f"Successfully registered DataFrame as table '{name}' using fallback method")
                return True
            except Exception as e2:
                logger.error(f"Fallback registration for table '{name}' also failed: {str(e2)}")
                logger.error(traceback.format_exc())
                return False
            
    def execute_query(self, query: str) -> Tuple[Optional[QueryResult], Optional[str]]:
        """Execute a SQL query against registered tables.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (QueryResult object, error message if any)
        """
        query_id = len(self.query_history) + 1
        start_time = datetime.now()
        
        try:
            logger.info(f"Executing query {query_id}: {query}")
            result_df = self.conn.execute(query).fetchdf()
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Record query execution
            query_record = {
                "query_id": query_id,
                "query": query,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "execution_time": execution_time,
                "success": True,
                "result_rows": len(result_df) if result_df is not None else 0,
                "result_columns": len(result_df.columns) if result_df is not None else 0
            }
            
            self.query_history.append(query_record)
            self.execution_stats[query_id] = query_record
            
            # Create QueryResult object
            query_result = QueryResult(
                dataframe=result_df,
                metadata={
                    'query': query,
                    'query_id': query_id,
                    'execution_time': execution_time,
                    'row_count': len(result_df),
                    'columns': list(result_df.columns),
                    'execution_stats': query_record
                },
                description=f"Query {query_id} executed at {start_time.isoformat()}, returned {len(result_df)} rows"
            )
            
            logger.info(f"Query {query_id} returned {len(result_df)} rows in {execution_time:.2f} seconds")
            return query_result, None
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            return None, str(e)
    
    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Validate a SQL query without executing it.
        
        Args:
            query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, error_message if not valid)
        """
        try:
            # Remove SQL comments before checking for operations
            query_no_comments = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
            query_no_comments = re.sub(r'/\*.*?\*/', '', query_no_comments, flags=re.DOTALL)
            query_lower = query_no_comments.lower().strip()

            # Check for unsafe operations
            unsafe_operations = {
                'create': 'CREATE operations are not allowed',
                'insert': 'INSERT operations are not allowed',
                'update': 'UPDATE operations are not allowed',
                'delete': 'DELETE operations are not allowed',
                'drop': 'DROP operations are not allowed',
                'alter': 'ALTER operations are not allowed',
                'truncate': 'TRUNCATE operations are not allowed'
            }

            for operation, message in unsafe_operations.items():
                if query_lower.startswith(operation) or f' {operation} ' in query_lower:
                    return False, message

            # Check for SELECT statement
            if not query_lower.startswith('select'):
                return False, "Only SELECT statements are allowed"

            # Check for incorrect table names
            if 'unified_table' in query_lower:
                return False, "Table 'unified_table' does not exist. Use 'unified' instead."

            # Try to prepare the query without executing
            self.conn.prepare(query)
            return True, None

        except Exception as e:
            error_message = str(e)
            logger.warning(f"Query validation failed: {error_message}")
            return False, error_message
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the schema for a registered table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            List of column information dictionaries
        """
        if table_name not in self.registered_tables:
            logger.warning(f"Attempted to get schema for non-existent table '{table_name}'")
            return []
            
        # Return cached schema if available
        if table_name in self.table_schemas:
            return self.table_schemas[table_name]
            
        try:
            schema_df = self.conn.execute(f"DESCRIBE SELECT * FROM {table_name}").fetchdf()
            schema = schema_df.to_dict('records')
            self.table_schemas[table_name] = schema
            return schema
        except Exception as e:
            logger.error(f"Error getting schema for table '{table_name}': {str(e)}")
            return []
    
    def _extract_schema(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract schema information from a DataFrame.
        
        Args:
            df: DataFrame to extract schema from
            
        Returns:
            List of column information dictionaries
        """
        schema = []
        
        for column in df.columns:
            col_type = str(df[column].dtype)
            col_info = {
                "name": column,
                "type": col_type,
                "nullable": df[column].isna().any(),
                "unique_values": df[column].nunique()
            }
            
            # Add numeric statistics if applicable
            if np.issubdtype(df[column].dtype, np.number):
                col_info.update({
                    "min": df[column].min() if not df.empty else None,
                    "max": df[column].max() if not df.empty else None,
                    "mean": df[column].mean() if not df.empty else None,
                    "median": df[column].median() if not df.empty else None,
                    "std": df[column].std() if not df.empty else None
                })
            
            # Add categorical statistics if applicable
            if df[column].nunique() < 20 or df[column].dtype == 'category':
                value_counts = df[column].value_counts().head(10).to_dict()
                col_info["value_distribution"] = value_counts
            
            schema.append(col_info)
            
        return schema
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive statistics for a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        if df is None or len(df) == 0:
            return {"error": "DataFrame is empty or None"}
            
        try:
            stats = {
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": {},
                "numeric_columns": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Overall memory usage
            stats["memory_usage"] = df.memory_usage(deep=True).sum()
            
            # First pass: identify and coerce numeric columns
            for column in df.columns:
                # Check for numeric suffix patterns
                is_numeric_suffix = any(column.lower().endswith(suffix) for suffix in ['_count', '_value', '_amount', '_concentration'])
                
                # Try to coerce to numeric if it has a numeric suffix or is already numeric
                if is_numeric_suffix or np.issubdtype(df[column].dtype, np.number):
                    try:
                        # Convert to numeric, coercing errors to NaN
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Could not coerce column {column} to numeric: {str(e)}")
            
            # Second pass: calculate statistics
            for column in df.columns:
                col_stats = {
                    "dtype": str(df[column].dtype),
                    "count": df[column].count(),
                    "null_count": df[column].isna().sum(),
                    "unique_count": df[column].nunique()
                }
                
                # Numeric column statistics
                if np.issubdtype(df[column].dtype, np.number):
                    non_null = df[column].dropna()
                    if len(non_null) > 0:
                        numeric_stats = {
                            "min": float(non_null.min()),
                            "max": float(non_null.max()),
                            "mean": float(non_null.mean()),
                            "median": float(non_null.median()),
                            "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                            "total": float(non_null.sum())  # Add total for count columns
                        }
                        
                        # Handle quartiles safely
                        try:
                            quartiles = non_null.quantile([0.25, 0.5, 0.75])
                            numeric_stats["quartiles"] = {str(k): float(v) for k, v in quartiles.to_dict().items()}
                        except Exception:
                            numeric_stats["quartiles"] = {}
                        
                        col_stats.update(numeric_stats)
                        stats["numeric_columns"][column] = numeric_stats
                
                # Categorical/string column statistics
                if df[column].dtype == 'object' or df[column].dtype == 'category' or df[column].nunique() < 20:
                    value_counts = df[column].value_counts().head(10).to_dict()
                    col_stats["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
                    
                    if df[column].dtype == 'object':
                        non_null = df[column].dropna()
                        if len(non_null) > 0:
                            length_series = non_null.astype(str).str.len()
                            col_stats["string_length"] = {
                                "min": int(length_series.min()),
                                "max": int(length_series.max()),
                                "mean": float(length_series.mean())
                            }
                
                # Datetime column statistics
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    non_null = df[column].dropna()
                    if len(non_null) > 0:
                        min_date = non_null.min()
                        max_date = non_null.max()
                        col_stats.update({
                            "min": min_date.isoformat(),
                            "max": max_date.isoformat(),
                            "range_days": (max_date - min_date).days
                        })
                
                stats["columns"][column] = col_stats
            
            # Check for geographic data
            geo_columns = self._identify_geographic_columns(df)
            if geo_columns:
                stats["geographic_data"] = geo_columns
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _identify_geographic_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify potential geographic columns in a DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary of geographic data information
        """
        geo_info = {}
        logger.info("Starting geographic column identification")
        
        try:
            # Look for common latitude/longitude column names
            lat_columns = [col for col in df.columns if col.lower() in ('lat', 'latitude', 'y')]
            lon_columns = [col for col in df.columns if col.lower() in ('lon', 'long', 'longitude', 'x')]
            
            logger.info(f"Found potential latitude columns: {lat_columns}")
            logger.info(f"Found potential longitude columns: {lon_columns}")
            
            if lat_columns and lon_columns:
                lat_col = lat_columns[0]
                lon_col = lon_columns[0]
                
                logger.info(f"Using {lat_col} as latitude and {lon_col} as longitude")
                
                # Check if columns contain numeric values in reasonable ranges
                is_lat_numeric = np.issubdtype(df[lat_col].dtype, np.number)
                is_lon_numeric = np.issubdtype(df[lon_col].dtype, np.number)
                
                logger.info(f"Latitude column is numeric: {is_lat_numeric}")
                logger.info(f"Longitude column is numeric: {is_lon_numeric}")
                
                if is_lat_numeric and is_lon_numeric:
                    # Get non-null values
                    lat_values = df[lat_col].dropna()
                    lon_values = df[lon_col].dropna()
                    
                    # Check if values exist after dropping NAs
                    lat_has_values = len(lat_values) > 0
                    lon_has_values = len(lon_values) > 0
                    
                    logger.info(f"Latitude column has values: {lat_has_values}")
                    logger.info(f"Longitude column has values: {lon_has_values}")
                    
                    if lat_has_values and lon_has_values:
                        # Get min/max values
                        lat_min = float(lat_values.min())
                        lat_max = float(lat_values.max())
                        lon_min = float(lon_values.min())
                        lon_max = float(lon_values.max())
                        
                        logger.info(f"Latitude range: {lat_min} to {lat_max}")
                        logger.info(f"Longitude range: {lon_min} to {lon_max}")
                        
                        # Check if values are in reasonable ranges for lat/lon
                        if lat_min >= -90 and lat_max <= 90 and lon_min >= -180 and lon_max <= 180:
                            logger.info("Coordinates are in valid geographic ranges")
                            
                            # Calculate coverage stats
                            lat_range = lat_max - lat_min
                            lon_range = lon_max - lon_min
                            
                            geo_info = {
                                "lat_column": lat_col,
                                "lon_column": lon_col,
                                "lat_range": {
                                    "min": lat_min,
                                    "max": lat_max,
                                    "span": lat_range
                                },
                                "lon_range": {
                                    "min": lon_min,
                                    "max": lon_max,
                                    "span": lon_range
                                },
                                "point_count": len(df[~df[lat_col].isna() & ~df[lon_col].isna()]),
                                "missing_count": len(df[df[lat_col].isna() | df[lon_col].isna()])
                            }
                            
                            # Add additional context if available
                            if "country" in df.columns:
                                country_counts = df["country"].value_counts().to_dict()
                                geo_info["countries"] = {str(k): int(v) for k, v in list(country_counts.items())[:10]}
                            
                            if "continent" in df.columns:
                                continent_counts = df["continent"].value_counts().to_dict()
                                geo_info["continents"] = {str(k): int(v) for k, v in continent_counts.items()}
                            
                            logger.info(f"Geographic data analysis complete: {len(geo_info)} attributes")
                        else:
                            logger.info("Coordinates outside expected ranges for latitude/longitude")
                    else:
                        logger.info("No valid values in coordinate columns after removing NAs")
                else:
                    logger.info("Coordinate columns are not numeric")
            else:
                logger.info("No matching latitude/longitude column pairs found")
                
            return geo_info
            
        except Exception as e:
            logger.error(f"Error identifying geographic columns: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
    
    def get_query_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the recent query history.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query execution records
        """
        return self.query_history[-limit:] if self.query_history else []
    
    def get_tables_info(self) -> Dict[str, Any]:
        """Get information about all registered tables.
        
        Returns:
            Dictionary with table information
        """
        return {
            "table_count": len(self.registered_tables),
            "tables": self.registered_tables,
            "schemas": self.table_schemas
        }
    
    def generate_sample_queries(self, table_name: str, count: int = 5) -> List[str]:
        """Generate sample queries for a table based on its schema.
        
        Args:
            table_name: Name of the table
            count: Number of sample queries to generate
            
        Returns:
            List of sample SQL queries
        """
        if table_name not in self.registered_tables:
            return []
            
        schema = self.get_table_schema(table_name)
        if not schema:
            return []
            
        sample_queries = []
        
        # Basic SELECT query
        sample_queries.append(f"SELECT * FROM {table_name} LIMIT 10")
        
        # Get column names
        column_names = [col["name"] for col in schema]
        
        # SELECT with specific columns
        if len(column_names) > 3:
            selected_columns = column_names[:3]
            sample_queries.append(f"SELECT {', '.join(selected_columns)} FROM {table_name} LIMIT 10")
        
        # Find numeric columns
        numeric_columns = [col["name"] for col in schema 
                          if col.get("type", "").startswith(("int", "float", "double"))]
        
        # Query with aggregation if numeric columns exist
        if numeric_columns and len(numeric_columns) > 0:
            agg_col = numeric_columns[0]
            sample_queries.append(f"SELECT MIN({agg_col}), MAX({agg_col}), AVG({agg_col}) FROM {table_name}")
        
        # Find potential categorical columns
        categorical_columns = [col["name"] for col in schema 
                             if col.get("unique_values", 1000) < 20 or "category" in col.get("type", "")]
        
        # Query with GROUP BY if categorical columns exist
        if categorical_columns and len(categorical_columns) > 0:
            group_col = categorical_columns[0]
            if numeric_columns and len(numeric_columns) > 0:
                agg_col = numeric_columns[0]
                sample_queries.append(
                    f"SELECT {group_col}, COUNT(*), AVG({agg_col}) FROM {table_name} GROUP BY {group_col} ORDER BY COUNT(*) DESC LIMIT 10"
                )
            else:
                sample_queries.append(
                    f"SELECT {group_col}, COUNT(*) FROM {table_name} GROUP BY {group_col} ORDER BY COUNT(*) DESC LIMIT 10"
                )
        
        # Return only the requested number of queries
        return sample_queries[:count]
        
    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None 