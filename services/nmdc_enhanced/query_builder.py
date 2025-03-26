"""
Query Builder for NMDC Enhanced service.

This module converts natural language questions into structured SQL queries 
and provides query validation and enhancement.
"""

import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
from datetime import datetime
import traceback

# Configure logging
logger = logging.getLogger(__name__)

class NMDCQueryBuilder:
    """Query builder for converting natural language to SQL.
    
    This class provides:
    1. Natural language to SQL conversion using LLMs
    2. Query validation against a schema
    3. Query enhancement suggestions
    4. Query tracking and management
    """
    
    def __init__(self, llm_client, config=None):
        """Initialize the query builder.
        
        Args:
            llm_client: LLM client for generating queries
            config: Configuration options
        """
        self.llm_client = llm_client
        self.config = config or {}
        self.query_store = {}
        self.last_query_id = None
        self.model_name = self.config.get('model_name', 'gpt-3.5-turbo')
        self.temperature = self.config.get('temperature', 0.2)
    
    def generate_query(self, natural_language_query: str, table_schemas: Dict[str, List[Dict[str, Any]]]) -> Tuple[str, List[str], Optional[str]]:
        """Generate a SQL query from natural language.
        
        Args:
            natural_language_query: Natural language question
            table_schemas: Dictionary of table schemas
            
        Returns:
            Tuple of (primary SQL query, list of alternative queries, error message if any)
        """
        try:
            # Prepare context information for the LLM
            schema_context = self._prepare_schema_context(table_schemas)
            
            # Create prompt for the LLM
            prompt = self._create_query_generation_prompt(natural_language_query, schema_context)
            
            # Get response from LLM
            llm_response = self.llm_client.chat_completion(
                [{"role": "system", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature
            )
            
            response_text = llm_response.choices[0].message.content
            
            # Parse the response to extract SQL queries
            main_query, alternative_queries = self._parse_query_response(response_text)
            
            if not main_query:
                return "", [], "Failed to generate a valid SQL query from the response"
                
            return main_query, alternative_queries, None
            
        except Exception as e:
            logger.error(f"Error generating query: {str(e)}")
            logger.error(traceback.format_exc())
            return "", [], str(e)
    
    def validate_and_improve_query(self, sql_query: str, table_schemas: Dict[str, List[Dict[str, Any]]]) -> Tuple[bool, str, List[str]]:
        """Validate a SQL query and suggest improvements.
        
        Args:
            sql_query: SQL query to validate
            table_schemas: Dictionary of table schemas
            
        Returns:
            Tuple of (is_valid, explanation, list of improved queries)
        """
        try:
            # Prepare context information for the LLM
            schema_context = self._prepare_schema_context(table_schemas)
            
            # Create prompt for validation
            prompt = self._create_query_validation_prompt(sql_query, schema_context)
            
            # Get response from LLM
            llm_response = self.llm_client.chat_completion(
                [{"role": "system", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature
            )
            
            response_text = llm_response.choices[0].message.content
            
            # Parse the validation response
            is_valid, explanation, improved_queries = self._parse_validation_response(response_text)
            
            return is_valid, explanation, improved_queries
            
        except Exception as e:
            logger.error(f"Error validating query: {str(e)}")
            logger.error(traceback.format_exc())
            return False, str(e), []
    
    def explain_query(self, sql_query: str, table_schemas: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate a natural language explanation of a SQL query.
        
        Args:
            sql_query: SQL query to explain
            table_schemas: Dictionary of table schemas
            
        Returns:
            Natural language explanation of the query
        """
        try:
            # Create prompt for explanation
            prompt = f"""
You are an expert database analyst helping explain SQL queries to scientists who may not be familiar with SQL syntax.

Given the following SQL query, provide a clear and concise explanation in natural language:

```sql
{sql_query}
```

Your explanation should:
1. Describe what the query is retrieving in simple terms
2. Explain any conditions or filters being applied
3. Describe any calculations or aggregations being performed
4. Avoid technical jargon unless necessary
5. Be understandable to someone with basic data analysis knowledge

Please format your response as a paragraph or short bulleted list depending on the query complexity.
"""
            
            # Get response from LLM
            llm_response = self.llm_client.chat_completion(
                [{"role": "system", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature
            )
            
            explanation = llm_response.choices[0].message.content.strip()
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining query: {str(e)}")
            return f"Error generating explanation: {str(e)}"
    
    def interpret_results(self, results_df: pd.DataFrame, query: str, context: Dict[str, Any] = None) -> str:
        """Generate an interpretation of query results.
        
        Args:
            results_df: DataFrame with query results
            query: The SQL query that generated the results
            context: Additional context information
            
        Returns:
            Natural language interpretation of the results
        """
        try:
            # Skip if DataFrame is empty
            if results_df is None or results_df.empty:
                return "The query returned no results to interpret."
            
            # Prepare context for the LLM
            df_info = self._prepare_dataframe_context(results_df)
            
            # Create prompt for interpretation
            prompt = f"""
You are a data analyst specializing in environmental and microbiome data from the National Microbiome Data Collaborative (NMDC).

I have executed the following SQL query:
```sql
{query}
```

The query returned {len(results_df)} rows and {len(results_df.columns)} columns.

Here is a summary of the result data:
{df_info}

Please provide a concise scientific interpretation of these results that would be helpful to a researcher. Your interpretation should:

1. Summarize the key findings in the data
2. Identify any obvious patterns, trends, or outliers
3. Put the results in scientific context related to environmental or microbiome research
4. If geographical data is present, comment on the spatial distribution
5. Suggest potential follow-up analyses or questions

Format your response as a professional scientific summary with 3-5 key points.
"""
            
            # Get response from LLM
            llm_response = self.llm_client.chat_completion(
                [{"role": "system", "content": prompt}],
                model=self.model_name,
                temperature=0.3
            )
            
            interpretation = llm_response.choices[0].message.content.strip()
            return interpretation
            
        except Exception as e:
            logger.error(f"Error interpreting results: {str(e)}")
            return f"Error generating interpretation: {str(e)}"
    
    def store_query(self, query: str, query_type: str, metadata: Dict[str, Any] = None) -> str:
        """Store a query and return a query ID.
        
        Args:
            query: The query to store
            query_type: Type of query (natural_language, sql, etc.)
            metadata: Additional metadata about the query
            
        Returns:
            Query ID
        """
        # Generate timestamp for query ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create query ID based on type
        if query_type == "natural_language":
            query_id = f"nmdc_enhanced_query_{timestamp}_orig"
        elif query_type == "sql":
            query_id = f"nmdc_enhanced_query_{timestamp}_sql"
        else:
            query_id = f"nmdc_enhanced_query_{timestamp}"
        
        # Store query information
        self.query_store[query_id] = {
            "query": query,
            "type": query_type,
            "timestamp": timestamp,
            "metadata": metadata or {}
        }
        
        self.last_query_id = query_id
        return query_id
    
    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored query by ID.
        
        Args:
            query_id: ID of the query to retrieve
            
        Returns:
            Query information or None if not found
        """
        return self.query_store.get(query_id)
    
    def _prepare_schema_context(self, table_schemas: Dict[str, List[Dict[str, Any]]]) -> str:
        """Prepare schema information for LLM prompts.
        
        Args:
            table_schemas: Dictionary of table schemas
            
        Returns:
            Formatted schema context
        """
        schema_text = []
        
        for table_name, schema in table_schemas.items():
            schema_text.append(f"Table: {table_name}")
            schema_text.append("Columns:")
            
            for column_info in schema:
                col_name = column_info.get("name", "")
                col_type = column_info.get("type", "")
                nullable = "NULL" if column_info.get("nullable", True) else "NOT NULL"
                
                schema_line = f"  - {col_name} ({col_type}, {nullable})"
                
                # Add additional information for important columns
                if "unique_values" in column_info:
                    unique_count = column_info["unique_values"]
                    schema_line += f", {unique_count} unique values"
                
                if "sample_values" in column_info and column_info["sample_values"]:
                    samples = column_info["sample_values"]
                    if len(samples) > 0:
                        sample_str = ", ".join([str(s) for s in samples[:3]])
                        schema_line += f", examples: {sample_str}"
                
                schema_text.append(schema_line)
            
            schema_text.append("")  # Empty line between tables
        
        return "\n".join(schema_text)
    
    def _prepare_dataframe_context(self, df: pd.DataFrame) -> str:
        """Prepare DataFrame summary for LLM prompts.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            DataFrame summary text
        """
        summary = []
        
        # Overall DataFrame info
        summary.append(f"Total rows: {len(df)}")
        summary.append(f"Columns: {', '.join(df.columns)}")
        summary.append("")
        
        # Sample data (first few rows)
        max_sample_rows = min(5, len(df))
        if max_sample_rows > 0:
            summary.append("Sample data (first few rows):")
            sample_df = df.head(max_sample_rows)
            sample_str = sample_df.to_string(index=False, max_cols=10)
            summary.append(sample_str)
            summary.append("")
        
        # Column summaries
        summary.append("Column summaries:")
        for column in df.columns[:10]:  # Limit to first 10 columns to avoid huge prompts
            col_data = df[column]
            
            # Basic info
            non_null_count = col_data.count()
            null_count = col_data.isna().sum()
            unique_count = col_data.nunique()
            
            summary.append(f"  - {column}: {non_null_count} non-null values, {null_count} nulls, {unique_count} unique values")
            
            # For numeric columns, add statistics
            if pd.api.types.is_numeric_dtype(col_data):
                if non_null_count > 0:
                    summary.append(f"    Min: {col_data.min()}, Max: {col_data.max()}, Mean: {col_data.mean():.2f}")
            
            # For categorical or string columns with few unique values, show distribution
            elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_categorical_dtype(col_data):
                if unique_count > 0 and unique_count <= 5:
                    value_counts = col_data.value_counts().head(5)
                    values_str = ", ".join([f"{v}: {c}" for v, c in value_counts.items()])
                    summary.append(f"    Top values: {values_str}")
        
        return "\n".join(summary)
    
    def _create_query_generation_prompt(self, natural_language_query: str, schema_context: str) -> str:
        """Create a prompt for generating SQL from natural language.
        
        Args:
            natural_language_query: Natural language query
            schema_context: Formatted schema information
            
        Returns:
            Prompt text
        """
        return f"""
You are an expert data analyst specializing in converting natural language questions into SQL queries for scientific databases. Your task is to generate a SQL query for the National Microbiome Data Collaborative (NMDC) database based on the user's question.

DATABASE SCHEMA AND STATISTICS:
{schema_context}

USER QUESTION:
{natural_language_query}

QUERY GUIDELINES:

1. Data Analysis Approach:
   - Use the provided statistics to inform your query design
   - Consider the full distribution of values when defining thresholds
   - Account for data quality (null values, unique counts, etc.)
   - Consider relationships between different measurements
   - Use domain knowledge about environmental and microbiome data

2. Technical Requirements:
   - Use DuckDB syntax (similar to PostgreSQL)
   - Quote column names containing special characters
   - Include clear column aliases
   - Handle NULL values appropriately
   - Use relevant JOINs if needed
   - Limit results to a reasonable number (e.g., LIMIT 100)

3. For Queries About Patterns or Unusual Values:
   - Use the provided statistics to understand typical ranges and distributions
   - Consider environmental context when determining what's "unusual"
   - Include relevant contextual columns to help interpret results
   - Consider multiple analytical approaches (e.g., absolute values, percentiles, ratios)

Provide your response in the following format:

QUERY:
```sql
Your SQL query here
```

EXPLANATION:
Brief explanation of:
- What the query does and why this approach was chosen
- How the statistics informed your query design
- What insights the results will provide
- Any assumptions or limitations

ALTERNATIVES:
```sql
Alternative query 1 here (optional)
```
Explanation of how this approach differs and what additional insights it might provide

```sql
Alternative query 2 here (optional)
```
Explanation of how this approach differs and what unique perspectives it offers

Always prioritize accuracy over complexity, and make reasonable assumptions when the natural language question is ambiguous."""
    
    def _create_query_validation_prompt(self, sql_query: str, schema_context: str) -> str:
        """Create a prompt for validating and improving a SQL query.
        
        Args:
            sql_query: SQL query to validate
            schema_context: Formatted schema information
            
        Returns:
            Prompt text
        """
        return f"""
You are an expert SQL developer and data scientist reviewing SQL queries for scientific databases. Please validate the following SQL query against the provided schema and suggest improvements.

DATABASE SCHEMA:
{schema_context}

SQL QUERY TO VALIDATE:
```sql
{sql_query}
```

Your tasks:
1. Check if the query is valid SQL syntax
2. Verify that all referenced tables and columns exist in the schema
3. Check for logical errors or inefficiencies
4. Suggest improvements for clarity, performance, or correctness
5. If the query has errors, provide corrected versions

Provide your response in the following format:

VALID: Yes/No

EXPLANATION:
Detailed explanation of whether the query is valid and any issues found

IMPROVEMENTS:
```sql
Improved version of the query
```
```sql
Alternative improved version (optional)
```

If the query is invalid and cannot be fixed with the given schema, explain why and provide the closest valid query you can create.
"""
    
    def _parse_query_response(self, response_text: str) -> Tuple[str, List[str]]:
        """Parse the LLM response to extract SQL queries.
        
        Args:
            response_text: LLM response text
            
        Returns:
            Tuple of (main query, list of alternative queries)
        """
        # Initialize return values
        main_query = ""
        alternative_queries = []
        
        # Look for the main query
        query_pattern = r"QUERY:\s*```sql\s*(.*?)```"
        query_match = re.search(query_pattern, response_text, re.DOTALL)
        if query_match:
            main_query = query_match.group(1).strip()
        
        # Look for alternative queries
        alt_pattern = r"ALTERNATIVES?:(?:.*?)```sql\s*(.*?)```"
        alt_matches = re.finditer(alt_pattern, response_text, re.DOTALL)
        
        for match in alt_matches:
            alt_query = match.group(1).strip()
            if alt_query and alt_query != main_query:
                alternative_queries.append(alt_query)
        
        # If no matches found using structured format, try simpler regex
        if not main_query:
            # Look for any SQL code blocks
            sql_blocks = re.finditer(r"```(?:sql)?\s*(.*?)```", response_text, re.DOTALL)
            for i, match in enumerate(sql_blocks):
                query = match.group(1).strip()
                if i == 0:
                    main_query = query
                else:
                    alternative_queries.append(query)
        
        return main_query, alternative_queries
    
    def _parse_validation_response(self, response_text: str) -> Tuple[bool, str, List[str]]:
        """Parse the validation response.
        
        Args:
            response_text: LLM validation response
            
        Returns:
            Tuple of (is_valid, explanation, improved_queries)
        """
        # Check if query is valid
        valid_pattern = r"VALID:\s*(Yes|No)"
        valid_match = re.search(valid_pattern, response_text, re.IGNORECASE)
        is_valid = False
        if valid_match:
            is_valid = valid_match.group(1).lower() == "yes"
        
        # Extract explanation
        explanation_pattern = r"EXPLANATION:\s*(.*?)(?:IMPROVEMENTS:|$)"
        explanation_match = re.search(explanation_pattern, response_text, re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else ""
        
        # Extract improved queries
        improved_queries = []
        improvements_pattern = r"IMPROVEMENTS:(?:.*?)```sql\s*(.*?)```"
        improvements_matches = re.finditer(improvements_pattern, response_text, re.DOTALL)
        
        for match in improvements_matches:
            improved_query = match.group(1).strip()
            if improved_query:
                improved_queries.append(improved_query)
        
        # If no structured improvements found, try simpler regex
        if not improved_queries and not is_valid:
            sql_blocks = re.finditer(r"```(?:sql)?\s*(.*?)```", response_text, re.DOTALL)
            for match in sql_blocks:
                query = match.group(1).strip()
                improved_queries.append(query)
        
        return is_valid, explanation, improved_queries 