"""
ChatDash.py
Version: 1.0.0
Last verified working: 2024-12-30

A Dash application that provides an interactive chat interface with dataset management.

Usage Instructions:
1. Starting the App:
   - Run the script: python ChatDash.py
   - Access via browser at http://localhost:8051

2. Dataset Management:
   - Upload files using drag-and-drop or file selector
   - Supported formats: CSV, TSV, ZIP (containing CSV/TSV)
   - Click '×' to delete datasets
   - Monitor memory usage in the status bar

3. Chat Interface:
   - Type messages in the text area
   - Press Enter to send (Shift+Enter for new line)
   - Click dataset names to auto-generate queries
   - Use "tell me about" queries for dataset exploration

4. Dataset Analysis:
   - Preview: Shows first few rows of data
   - Statistics: Displays detailed profiling report
   - Visualization: Interactive plots and charts
   - Database: SQL database structure and ERD
   - Weaviate: Vector database collections and relationships
     * Table Summary: Collection statistics and properties
     * ER Diagram: Visual representation of collection relationships

5. Database Connections:
   - SQL Database:
     * Select database from dropdown
     * View schema and relationships
     * Execute queries through chat
   - Weaviate Vector Database:
     * Real-time connection status
     * Collection availability monitoring
     * Detailed schema visualization

Known Limitations:
- Enter key to send messages is not currently supported due to Dash callback restrictions

Note: All callbacks must remain above the if __name__ == '__main__' block
"""

import dash
from dash import html, dcc, Input, Output, State, callback, ALL, dash_table
import dash_bootstrap_components as dbc
import openai
import os
import pandas as pd
import io
import base64
from datetime import datetime
import json
from ydata_profiling import ProfileReport
import warnings
import zipfile
import tempfile
from datetime import datetime
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import psutil
import sys
import plotly.express as px
import plotly.graph_objects as go
import re
from fuzzywuzzy import fuzz, process
from typing import Dict, List, Tuple, Optional, Union, Any
import colorsys
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import pdist
import sqlite3
from pathlib import Path
from database_manager import DatabaseManager
from dash_extensions import Mermaid
from dotenv import load_dotenv
from pathlib import Path
from weaviate_integration import WeaviateConnection
from weaviate_manager.query.manager import QueryManager
import traceback
from services import registry as service_registry
from services import ServiceMessage

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=Warning)

####################################
#
# Initialize Configuration
#
####################################

# Find the project root directory (where .env is located)
project_root = Path(__file__).parent
dotenv_path = project_root / '.env'

# Try to load from .env file
load_dotenv(dotenv_path=dotenv_path)

# OpenAI Settings
if True:  # Toggle for development environment
    OPENAI_BASE_URL = os.getenv('CBORG_BASE_URL', "https://api.cborg.lbl.gov")
    OPENAI_API_KEY = os.getenv('CBORG_API_KEY', '')  # Must be set in environment
else:  # Production environment
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Must be set in environment

# Configuration Constants
OPENAI_CONFIG = {
    'api_key': OPENAI_API_KEY,
    'base_url': OPENAI_BASE_URL
}

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# Available models
AVAILABLE_MODELS = [
    "lbl/cborg-chat:latest", "lbl/cborg-coder:latest", 
    "lbl/llama", "openai/gpt-4o", "openai/gpt-4o-mini",
    "openai/o1", "openai/o1-mini", "anthropic/claude-haiku",
    "anthropic/claude-sonnet", "anthropic/claude-opus", "google/gemini-pro",
    "google/gemini-flash", "aws/llama-3.1-405b", "aws/llama-3.1-70b",
    "aws/llama-3.1-8b", "aws/command-r-plus-v1", "aws/command-r-v1"
]

# Style Constants
CHAT_STYLES = {
    'user': {
        'backgroundColor': '#007bff',
        'color': 'white',
        'maxWidth': '75%',
        'marginLeft': 'auto'
    },
    'system': {
        'backgroundColor': '#dc3545',
        'color': 'white',
        'maxWidth': '75%'
    },
    'assistant': {
        'backgroundColor': '#f8f9fa',
        'maxWidth': '75%'
    },
    'service': {  # Add service-specific styling
        'backgroundColor': '#e9ecef',  # Light gray background
        'border': '1px solid #dee2e6',  # Subtle border
        'borderLeft': '4px solid #6c757d',  # Left accent border
        'maxWidth': '90%',  # Wider than other messages
        'fontFamily': 'monospace'  # Better for tables
    },
    'service_error': {  # Style for service error messages
        'backgroundColor': '#fff3f3',
        'border': '1px solid #dc3545',
        'borderLeft': '4px solid #dc3545',
        'maxWidth': '75%',
        'color': '#dc3545'
    }
}

####################################
#
# Initialize OpenAI client
#
####################################

client = openai.OpenAI(**OPENAI_CONFIG)
main_chat_client= client

####################################
#
# Initialize Dash app
#
####################################

app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://use.fontawesome.com/releases/v5.15.4/css/all.css'  # Add Font Awesome
])
app.config.suppress_callback_exceptions = True

####################################
#
# Help Message
#
####################################

help_message = """Here's what you can do with this chat interface:

🔍 **SQL Queries**
- First, select your database using the dropdown at the top of Data Management
- View database structure in the Database tab under Dataset Info
- Ask about your database: "Tell me about the connected database"
- Execute queries:
  - Simple: Type "execute\\." to run the last query
  - Specific: "execute query\\_20240315\\_123456\\_original" for a particular query
  - Note: Valid execution commands will run immediately
- Convert query results to dataset:
  - "convert query\\_20240315\\_123456\\_original to dataset"

📚 **Literature Search**
- Search for scientific literature using natural language:
  - "What is known about gene regulation?"
  - "Find papers about CRISPR"
  - "Search for articles related to metabolic pathways"
- Refine literature search results:
  - Use "refine lit_query_XXXXXXXX_XXXXXX with threshold 0.X"
  - Example: "refine lit_query_20250207_123456 with threshold 0.7"
  - Higher thresholds (0.7-0.9) give more relevant but fewer results
  - Lower thresholds (0.3-0.5) give more results but may be less relevant
- Convert literature results to dataset:
  - "convert lit_query_XXXXXXXX_XXXXXX to dataset"

📁 **Dataset Management**
- Add datasets by:
  - Dragging files onto the upload region
  - Clicking the upload region to browse files
  - Accepted formats: CSV, TSV, and ZIP files containing these types
- Datasets appear in the dataset browser:
  - Click a dataset to select it for analysis
  - Click the red X to delete a dataset
- View dataset statistics and profiles
- Combine SQL query results with uploaded data
- Get dataset information:
  - "Tell me about my datasets" - Summary of all available datasets
  - "Tell me about my dataset" - Information about currently selected dataset
  - "Tell me about dataset\\_name" - Information about a specific dataset

📊 **Data Visualization**
First, select a dataset by clicking its name in the dataset browser. Then use these commands:

1. Bubble Plots:
   ```
   plot [y_column] vs [x_column]
   ```
   Optional parameters:
   - size=[number or column_name]
   - color=[color_name or column_name]
   * For column-based color: numeric columns create continuous scales, categorical columns create discrete legends

2. Heatmaps:
   ```
   heatmap columns=[col1,col2,col3] 
   ```
   Required:
   - columns=[col1,col2,...] or columns=regex_pattern
   
   Optional:
   - rows=[row1,row2,...] or rows=regex_pattern fcol=filter_column
   - standardize=rows|columns
   - cluster=rows|columns|both
   - colormap=[valid_plotly_colormap]
   - transpose=true|false

   Note: When using regex patterns with rows, fcol must specify which column to filter on

3. Geographic Maps:
   ```
   map latitude=[lat_column] longitude=[lon_column]
   ```
   Required:
   - latitude=[column_name]
   - longitude=[column_name]
   
   Optional:
   - size=[number or column_name]
   - color=[color_name or column_name]
   * Numeric color columns show colorbar, categorical show legend

All visualizations feature:
- Pan: Click and drag
- Zoom: Mouse wheel or pinch gestures
- Reset: Double-click
- Export: Use modebar tools for PNG export
- Tooltips: Hover for detailed values

💡 **Tips**
- You must select a dataset before creating visualizations
- Double-click to reset any visualization view
- Use the modebar tools for additional controls
- Export high-quality images using the camera icon
- Create interactive plots with pan and zoom capabilities:
  - Bubble Plots: "Plot [column1] vs [column2]"
    * Optional: Add size=[value/column] color=[value/column]
    * Interact using mouse wheel to zoom, click and drag to pan
    * Double-click to reset view
  - Heatmaps: "Create heatmap columns=[col1,col2,col3]"
    * Options: rows=[...] standardize=rows/columns cluster=both/rows/columns
    * standardize normalizes data by row or column
    * cluster applies hierarchical clustering with optimal leaf ordering
  - Maps: "Create map latitude=[column1] longitude=[column2]"
    * Optional: Add size=[value/column] color=[value/column]
    * Pan by clicking and dragging
    * Zoom with mouse wheel or pinch gestures
- All visualizations feature:
  * Interactive tooltips showing data values
  * Export options for high-quality images
  * Modebar with zoom controls and other tools
  * Auto-scaling and responsive layout

💡 **Tips**
- Use natural language to ask questions about your data
- You may need to reference specific columns using \\`backticks\\`
- Click the dataset cards to switch between datasets
- Double-click on visualizations to reset the view
- Use the modebar tools for additional visualization controls
"""

####################################
#
# Index Search Classes
#
####################################

# The definition of the text indexer for databases so we can find values in databases   
class DatabaseTextSearch:
    """Text search across database content."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.table_docs = {}  # Store documents by table.column
        self.table_details = {}  # Store detailed value information
        self.index = None
        self.fitted = False
        
    def index_database(self, db_path, db_manager=None):
        """Index all text content in database tables.
        
        Args:
            db_path: Path to the database
            db_manager: Optional existing DatabaseManager instance
        """
        try:
            # Use provided DatabaseManager or create new one
            db = db_manager if db_manager is not None else DatabaseManager(db_path)
            
            # Get all tables using DatabaseManager's execute_query
            tables = db.execute_query("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
            """).fetchall()
            
            for (table_name,) in tables:
                # Get column info
                columns = db.execute_query(f"PRAGMA table_info('{table_name}')").fetchall()
                
                # Find text-like columns
                text_columns = [col[1] for col in columns 
                              if 'CHAR' in col[2].upper() 
                              or 'TEXT' in col[2].upper()]
                
                if not text_columns:
                    continue
                
                for column in text_columns:
                    # Get random sample to check if the column is primarily numeric
                    sample_query = f"""
                    SELECT "{column}"
                    FROM "{table_name}"
                    WHERE "{column}" IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT 100
                    """
                    sample_values = [row[0] for row in db.execute_query(sample_query).fetchall()]
                    
                    # Check if values are primarily numeric (must match whole value)
                    numeric_pattern = r'^-?\d*\.?\d+(?:[eE][-+]?\d+)?$'
                    numeric_count = sum(1 for v in sample_values 
                                     if re.match(numeric_pattern, str(v).strip()))
                    
                    # Skip if more than 95% of values are numeric
                    if len(sample_values) > 0 and numeric_count / len(sample_values) > 0.95:
                        continue
                    
                    # If we get here, get all values for the text column
                    key = f"{table_name}.{column}"
                    query = f"""
                    SELECT "{column}", COUNT(*) as count
                    FROM "{table_name}"
                    WHERE "{column}" IS NOT NULL
                    GROUP BY "{column}"
                    """
                    values = db.execute_query(query).fetchall()
                    
                    # Store column details
                    self.table_details[key] = {
                        'unique_values': [v[0] for v in values],
                        'value_counts': {v[0]: v[1] for v in values},
                        'total_unique': len(values)
                    }
                    
                    # Create searchable document
                    doc = f"Table {table_name} column {column} contains values: "
                    doc += ", ".join(str(v[0]) for v in values[:10])  # Sample values
                    
                    self.table_docs[key] = doc
            
            # Create search index
            self._create_index()
            
        except Exception as e:
            print(f"Error indexing database: {str(e)}")
            raise
    
    def _create_index(self):
        """Create TF-IDF index from documents."""
        if not self.table_docs:
            return
            
        all_docs = list(self.table_docs.values())
        self.doc_map = list(self.table_docs.keys())
        
        self.index = self.vectorizer.fit_transform(all_docs)
        self.fitted = True
    
    def search_text(self, query: str, threshold: float = 0.9, coverage = 0.0) -> list:
        """Search for text in database content."""
        if not self.fitted:
            return []
            
        try:
            search_term = query.lower().replace("'", "").replace('"', '').strip()
            results = []
            
            for table_col, details in self.table_details.items():
                table, column = table_col.split('.')
                matches = []
                
                for value in details['unique_values']:
                    if value is None:
                        continue
                        
                    str_value = str(value).lower()
                    ratio = fuzz.ratio(search_term, str_value)
                    
                    if ratio > threshold * 100 and len(str_value)/len(search_term) > coverage:
                        matches.append((value, ratio))
                
                if matches:
                    # Sort matches by similarity ratio
                    matches.sort(key=lambda x: x[1], reverse=True)
                    matched_values = [m[0] for m in matches]
                    
                    # Find or create table result
                    table_result = next((r for r in results if r['source_name'] == table), None)
                    if table_result is None:
                        table_result = {
                            'source_name': table,
                            'source_type': 'database',
                            'similarity': max(m[1] for m in matches) / 100,
                            'matched_text': f"Found matches in database table {table}",
                            'details': {}
                        }
                        results.append(table_result)
                    
                    # Add column details
                    table_result['details'][column] = {
                        'matches': matched_values,
                        'counts': {m: details['value_counts'].get(m, 0) for m in matched_values},
                        'similarities': {m: ratio for m, ratio in matches}
                    }
            
            return results
                    
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
        
# The definition of the text indexer for datasets so we can find values in datasets
class DatasetTextSearch:
    """Efficient text search across datasets.
    
    This class provides fuzzy text search capabilities across multiple datasets,
    using TF-IDF vectorization and cosine similarity for matching.
    
    Attributes:
        vectorizer: TfidfVectorizer instance for text processing
        dataset_docs: Dict mapping dataset names to searchable documents
        dataset_details: Dict storing column and value information
        index: Computed TF-IDF matrix
        fitted: Boolean indicating if index is ready
        
    Example:
        >>> searcher = DatasetTextSearch()
        >>> searcher.update_dataset("sales", sales_df)
        >>> results = searcher.search_text("high revenue")
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.dataset_docs = {}  # Store the raw documents
        self.dataset_details = {}  # Store detailed column/value information
        self.index = None
        self.fitted = False
        
    def _create_index(self, dataset_docs):
        """Create TF-IDF index from documents."""
        all_docs = []
        self.doc_map = []
        
        for dataset_name, docs in dataset_docs.items():
            all_docs.extend(docs)
            self.doc_map.extend([(dataset_name, i) for i in range(len(docs))])
        
        self.index = self.vectorizer.fit_transform(all_docs)
        self.fitted = True
        return self.index
    
    def update_dataset(self, name: str, df: pd.DataFrame):
        """Update the search index and store detailed value information."""
        # Store column names and metadata
        column_doc = f"Dataset: {name}\nColumns: {', '.join(df.columns)}"
        value_docs = []
        
        # Store detailed information about text columns
        self.dataset_details[name] = {}
        
        for col in df.select_dtypes(include=['object', 'string']).columns:
            unique_values = df[col].dropna().unique()
            n_unique = len(unique_values)
            value_counts = df[col].value_counts()
            
            # Store detailed column information
            self.dataset_details[name][col] = {
                'unique_values': list(unique_values),
                'value_counts': value_counts.to_dict(),
                'total_unique': n_unique
            }
            
            # Create document with column info and values
            value_doc = (
                f"Dataset {name} column {col} contains {n_unique} unique values including: "
                f"{', '.join(map(str, unique_values))}"
            )
            value_docs.append(value_doc)
            
            # Add frequency information
            if n_unique > 0:  # Changed from 10 to 0 to see all frequencies
                top_values = value_counts.head(10)
                freq_doc = (
                    f"Most frequent values in {col}: "
                    f"{', '.join(f'{v} ({c} times)' for v, c in top_values.items())}"
                )
                value_docs.append(freq_doc)
        
        # Update search index
        all_docs = [column_doc] + value_docs
        self.dataset_docs[name] = all_docs
        self._create_index(self.dataset_docs)
    
    def search_text(self, query: str, threshold: float = 0.6, coverage = 0.0) -> list:
        """Search for text and return detailed matches using fuzzy matching."""
        if not self.fitted:
            return []

        try:
            # Clean up the query minimally - just remove quotes and extra spaces
            search_term = query.lower().replace("'", "").replace('"', '').strip()
            
            results = []
            for dataset_name, columns in self.dataset_details.items():
                details = {}
                for col_name, col_info in columns.items():
                    # Use fuzzy matching to find similar strings
                    matches = []
                    for value in col_info['unique_values']:
                        str_value = str(value).lower()
                        # Calculate similarity ratio
                        ratio = fuzz.ratio(search_term, str_value)

                        if ratio > threshold * 100 and len(str_value)/len(search_term) > coverage:  # Convert threshold to percentage
                            matches.append((value, ratio))
                    
                    if matches:
                        # Sort matches by similarity ratio
                        matches.sort(key=lambda x: x[1], reverse=True)
                        matched_values = [m[0] for m in matches]
                        details[col_name] = {
                            'matches': matched_values,
                            'counts': {m: col_info['value_counts'].get(m, 0) for m in matched_values},
                            'similarities': {m: ratio for m, ratio in matches}
                        }
                
                if details:
                    results.append({
                        'source_name': dataset_name,     # Changed from 'dataset'
                        'source_type': 'dataset',        # Added this field
                        'similarity': max(max(d['similarities'].values()) for d in details.values()) / 100,
                        'matched_text': f"Found matches in dataset {dataset_name}",
                        'details': details
                    })
            
            return results
                
        except Exception as e:
            return []

# Initialize the text searchers for datasets and databases  
text_searcher = DatasetTextSearch()
text_searcher_db = DatabaseTextSearch()

####################################
#
# Chat Management Functions
#
####################################

def create_system_message(dataset_info: List[Dict[str, Any]], 
                         search_query: Optional[str] = None,
                         database_structure: Optional[Dict] = None,
                         weaviate_results: Optional[Dict] = None,
                         service_context: Optional[Dict] = None) -> str:
    """Create system message with context from datasets, database, and literature."""
    print("\n=== System Message Creation Debug ===")
    print(f"Database structure type: {type(database_structure)}")
    if database_structure:
        try:
            first_table = next(iter(database_structure.items()))
            print(f"First table entry: {first_table[0]} (type: {type(first_table[0])})")
            print(f"First table info: {first_table[1]}")
        except Exception as e:
            print(f"Error inspecting database structure: {str(e)}")

    base_message = "You are a data analysis assistant with access to:"

    # Track available data sources
    has_datasets = bool(dataset_info)
    has_database = bool(database_structure)
    has_literature = True  # We always have access to literature search through Weaviate
    has_literature_results = bool(weaviate_results and weaviate_results.get('unified_results'))

    # Add dataset information
    if has_datasets:
        base_message += "\n\nDatasets:"
        for ds in dataset_info:
            base_message += f"\n- {ds['name']}: {ds['rows']} rows, columns: {', '.join(ds['columns'])}"
            if ds.get('selected'):
                base_message += " (currently selected)"

    # Add database information
    if has_database:
        base_message += "\n\nConnected Database Tables:"
        for table, info in database_structure.items():
            # Add table info
            base_message += f"\n\n{table} ({info['row_count']} rows)"
            
            # Add column details
            base_message += "\nColumns:"
            for col in info['columns']:
                constraints = []
                if col['pk']: constraints.append("PRIMARY KEY")
                if col['notnull']: constraints.append("NOT NULL")
                if col['default']: constraints.append(f"DEFAULT {col['default']}")
                constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                base_message += f"\n  - {col['name']}: {col['type']}{constraint_str}"
            
            # Add foreign key relationships
            if info['foreign_keys']:
                base_message += "\nForeign Keys:"
                for fk in info['foreign_keys']:
                    base_message += f"\n  - {fk['from']} → {fk['table']}.{fk['to']}"
        
        # Enhanced SQL Guidelines
        base_message += "\n\nSQL Query Guidelines:"
        
        # 1. Safety and Compatibility
        base_message += "\n\n1. Query Safety and Compatibility:"
        base_message += "\n   - Use ONLY SQLite-compatible syntax"
        base_message += "\n   - SQLite limitations to be aware of:"
        base_message += "\n     * NO SIMILAR TO pattern matching (use LIKE or GLOB instead)"
        base_message += "\n     * NO FULL OUTER JOIN (use LEFT/RIGHT JOIN)"
        base_message += "\n     * NO WINDOW FUNCTIONS before SQLite 3.25"
        base_message += "\n     * NO stored procedures or functions"
        base_message += "\n   - NO database modifications (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE)"
        base_message += "\n   - NO destructive or resource-intensive operations"
        base_message += "\n   - Ensure proper table/column name quoting"
        base_message += "\n   - IMPORTANT: SQL queries can ONLY be run against the connected database tables listed above"
        base_message += "\n   - Datasets (if any are loaded) cannot be queried with SQL - they are separate from the database"
        
        # 2. User SQL Handling
        base_message += "\n\n2. When User Provides SQL Code:"
        base_message += "\n   - Validate for safety and SQLite compatibility"
        base_message += "\n   - If safe and valid:"
        base_message += "\n     * Use it as your primary (original) suggestion"
        base_message += "\n     * Explain what it does"
        base_message += "\n     * Suggest improvements as alternative queries"
        base_message += "\n   - If unsafe or invalid:"
        base_message += "\n     * Explain the specific issues"
        base_message += "\n     * Provide a safe alternative as primary suggestion"
        base_message += "\n     * Include the user's query as a comment for reference"
        
        # 3. Query Response Format
        base_message += "\n\n3. Query Response Format:"
        base_message += "\n   Always structure responses as follows:"
        base_message += "\n   a) Primary Query (Original):"
        base_message += "\n   ```sql"
        base_message += "\n   -- Purpose: Clear description of query goal"
        base_message += "\n   -- Tables: List of tables used"
        base_message += "\n   -- Assumptions: Any important assumptions"
        base_message += "\n   SELECT ... -- Your SQL here"
        base_message += "\n   ```"
        base_message += "\n   b) Alternative Queries (if relevant):"
        base_message += "\n   ```sql"
        base_message += "\n   -- Improvement: Explain how this improves on original"
        base_message += "\n   SELECT ... -- Alternative SQL"
        base_message += "\n   ```"
        base_message += "\n   Note: Query IDs will be added automatically by the system. Do not include them in your response."
        
        # 4. Query Best Practices
        base_message += "\n\n4. Query Best Practices:"
        base_message += "\n   - Use explicit column names instead of SELECT *"
        base_message += "\n   - Include appropriate WHERE clauses to limit results"
        base_message += "\n   - Use meaningful table aliases in JOINs"
        base_message += "\n   - Add comments for complex logic"
        base_message += "\n   - Consider performance with large tables"
        
        # 5. Execution Instructions
        base_message += "\n\n5. Query Execution:"
        base_message += "\n   Users can execute queries using:"
        base_message += '\n   - "execute." to run the primary (original) query'
        base_message += '\n   - "execute query_ID" to run a specific query'
        base_message += '\n   - "convert query_ID to dataset" to save results'

    # Add literature information
    if has_literature_results:
        try:
            results = weaviate_results.get('unified_results', [])
            if results:
                base_message += "\n\nLiterature Database Results:"
                base_message += f"\nFound {len(results)} relevant items:"
                # Show top 5 results with key information
                for result in results[:5]:
                    # Extract title and score safely
                    title = result.get('properties', {}).get('title', 'Untitled')
                    score = result.get('score', 0.0)
                    result_id = result.get('id', 'unknown')
                    # Truncate title if too long
                    if len(title) > 100:
                        title = title[:97] + "..."
                    base_message += f"\n- [{result_id}] {title} (Score: {score:.2f})"
                
                if len(results) > 5:
                    base_message += f"\n... and {len(results) - 5} more results"
                
                base_message += "\n\nPlease incorporate these findings in your response."
        except Exception as e:
            print(f"Error processing Weaviate results: {str(e)}")
            print(f"Result structure: {json.dumps(results[0] if results else {}, indent=2)}")
            base_message += "\n\nLiterature Database: Error processing results"
    elif weaviate_results is not None:  # We attempted a literature search but got no results
        base_message += "\n\nLiterature Database: No relevant results found for your query"

    # Add instructions for handling different types of queries
    base_message += "\n\nWhen responding to queries:"
    
    # Add knowledge retrieval instructions first
    base_message += "\n\nKnowledge Integration:"
    base_message += "\n1. First, provide relevant background knowledge from your training:"
    base_message += "\n   - Explain key concepts, terminology, and relationships"
    base_message += "\n   - Describe standard methodologies or approaches"
    base_message += "\n   - Highlight important considerations or limitations"
    base_message += "\n2. Then suggest how to explore this knowledge using available data sources:"
    base_message += "\n   - Identify relevant fields or patterns to search for"
    base_message += "\n   - Propose specific analyses or comparisons"
    base_message += "\n   - Structure suggestions to help formulate targeted queries"
    
    if has_datasets:
        base_message += "\n- If the query relates to the available datasets, analyze that data"
        base_message += "\n- If you find relevant information in the datasets, include it in your response"
        base_message += "\n- Remember: Datasets are separate from the database and cannot be queried using SQL"
        base_message += "\n- To analyze datasets, use the available visualization and analysis tools"
    
    if has_database:
        base_message += "\n- If you recognize a SQL query, analyze it and suggest improvements if needed"
        base_message += "\n- If you receive a natural language database question, propose an appropriate SQL query"
        base_message += "\n- DO NOT execute SQL queries directly - only suggest them for the user to execute"
        base_message += "\n- DO NOT claim to have run queries unless the user has explicitly executed them"
        base_message += "\n- Ensure all SQL queries are valid for SQLite and don't use extended features"
    else:
        base_message += "\n- The database is not connected - DO NOT suggest or reference SQL queries"
        base_message += "\n- Focus on other available data sources and general knowledge"
    
    if has_literature:
        base_message += "\n- You have access to a scientific literature database through Weaviate"
        base_message += "\n- For literature queries, use the available search functionality"
        base_message += "\n- When referencing literature results, use the [ID] format"
    
    base_message += "\n- You can combine available data sources with general knowledge"
    base_message += "\n- If no specific data is found, provide a helpful response using your general knowledge"
    base_message += "\n- NEVER suggest querying data sources that are not currently connected"
    base_message += "\n- NEVER claim to have executed queries or retrieved data unless explicitly done by the user"

    if service_context:
        base_message += "\n\nService Context:"
        for key, value in service_context.items():
            base_message += f"\n- {key}: {value}"

    return base_message

def create_chat_element(message: dict) -> dbc.Card:
    """
    Create a styled chat element based on message type.
    
    Args:
        message (dict): Message dictionary containing:
            - role (str): One of 'user', 'assistant', 'system', or 'service'
            - content (str): The message text
            - service (str, optional): Service name if message is from a service
            - type (str, optional): Message type for service messages
        
    Returns:
        dbc.Card: Styled card component
    """
    # Determine style based on message properties
    if 'service' in message:
        # Service message styling
        if message.get('type') == 'error':
            style = CHAT_STYLES['service_error']
        else:
            style = CHAT_STYLES['service']
            
        # Add service indicator if not an error
        if message.get('type') != 'error':
            header = html.Div(
                f"Service: {message['service']}",
                style={
                    'fontSize': '0.8em',
                    'color': '#6c757d',
                    'marginBottom': '8px'
                }
            )
        else:
            header = None
            
        # Always render service content as markdown
        content = dcc.Markdown(
            message['content'],
            style={'width': '100%'}  # Ensure tables can use full width
        )
        
        # Create card body with optional header
        card_body = [header, content] if header else [content]
        
    else:
        # Regular message styling
        style = CHAT_STYLES.get(message['role'], CHAT_STYLES['assistant'])
        content = dcc.Markdown(message['content']) if message['role'] == 'assistant' else message['content']
        card_body = [content]
    
    return dbc.Card(
        dbc.CardBody(card_body),
        className="mb-2 ml-auto" if message['role'] == 'user' else "mb-2",
        style=style
    )

####################################
#
# Database Management Functions
#
####################################

def get_database_files(data_dir='data') -> list:
    """Scan data directory for SQLite database files with validation.
    
    Args:
        data_dir: Path to directory containing databases, relative to app
        
    Returns:
        List of dicts with database options for dropdown
    """
    try:
        base_path = Path(__file__).parent / data_dir
        if not base_path.exists():
            print(f"Warning: Data directory {data_dir} not found")
            return []
            
        db_files = []
        for ext in ['.db', '.sqlite', '.sqlite3', '.db3']:
            try:
                # Check if files are actually SQLite databases
                for file in base_path.glob(f'**/*{ext}'):
                    try:
                        # Test if file is actually a SQLite database
                        with sqlite3.connect(file) as conn:
                            conn.execute("SELECT 1")
                        db_files.append(file)
                    except sqlite3.Error:
                        print(f"Warning: {file} has SQLite extension but is not a valid database")
            except Exception as e:
                print(f"Error scanning for {ext} files: {e}")
                
        return [{'label': db.name, 'value': str(db)} for db in db_files]
    except Exception as e:
        print(f"Error scanning database directory: {e}")
        return []

def generate_mermaid_erd(structure: dict) -> str:
    """Generate Mermaid ERD diagram from database structure."""
    print(f"\n=== Generating ERD for {len(structure)} tables ===")
    
    # Start with markdown code block like Weaviate
    mermaid_lines = []
    mermaid_lines.append("```mermaid")
    mermaid_lines.append("erDiagram")
    
    # Add tables with minimal attributes
    for table, info in structure.items():
        table_clean = table.replace(' ', '_').replace('-', '_')
        
        # Create list of column definitions with consistent indentation
        columns = []
        for col in info['columns']:
            col_name = col['name'].replace(' ', '_').replace('-', '_').replace('@', 'at_')
            # Clean up the type - remove parentheses and their contents
            col_type = col['type'].split('(')[0].upper()
            
            # Add indicators for special properties
            indicators = []
            if col.get('pk', 0) == 1:
                indicators.append("PK")
            if col.get('vectorize', False):
                indicators.append("V")  # V for vectorized
            if col.get('indexFilterable', False):
                indicators.append("F")  # F for filterable
            if col.get('indexSearchable', False):
                indicators.append("S")  # S for searchable
                
            if indicators:
                columns.append(f"        {col_type} {col_name} {' '.join(indicators)}")
            else:
                columns.append(f"        {col_type} {col_name}")
        
        # Add table definition with proper indentation
        mermaid_lines.append(f"    {table_clean} {{")
        mermaid_lines.extend(columns)
        mermaid_lines.append("    }")
    
    # Add relationships
    relationship_count = 0
    for table, info in structure.items():
        table_clean = table.replace(' ', '_').replace('-', '_')
        
        # Add foreign key relationships
        for fk in info.get('foreign_keys', []):
            ref_table = fk['table'].replace(' ', '_').replace('-', '_')
            mermaid_lines.append(f"    {table_clean} ||--o{{ {ref_table} : FK")
            relationship_count += 1
            
        # Add cross-references from schema
        for ref in info.get('references', []):
            ref_table = ref['target'].replace(' ', '_').replace('-', '_')
            mermaid_lines.append(f"    {table_clean} ||--o{{ {ref_table} : {ref['name']}")
            relationship_count += 1
    
    # Add closing markdown code block
    mermaid_lines.append("```")
    
    print(f"Generated ERD with {relationship_count} relationships")
    return "\n".join(mermaid_lines)

def validate_sql_query(sql: str, db_path: str) -> tuple[bool, str, dict]:
    """
    Validate SQL query for safety and correctness.
    
    Args:
        sql (str): SQL query to validate
        db_path (str): Path to SQLite database
        
    Returns:
        tuple[bool, str, dict]: (is_valid, error_message, metadata)
            - is_valid: True if query is safe and valid
            - error_message: Description of any issues found
            - metadata: Additional information about the query
    """
    try:
        # 1. Basic safety checks
        
        # Remove SQL comments before checking for write operations
        sql_no_comments = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)  # Remove single line comments
        sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)  # Remove multi-line comments
        sql_lower = sql_no_comments.lower().strip()
        
        # Check for write operations
        write_operations = ['insert', 'update', 'delete', 'drop', 'alter', 'create']
        for op in write_operations:
            if sql_lower.startswith(op) or f' {op} ' in sql_lower:
                return False, f"Write operation '{op}' is not allowed", {}
        
        # 2. Connect to database for deeper validation
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 3. Get schema information
        tables = {}
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table_name,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table_name})")
            tables[table_name] = {row[1]: row[2] for row in cursor.fetchall()}
        
        # 4. Explain query plan to validate syntax and references
        try:
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan = cursor.fetchall()
            
            # Extract referenced tables from plan
            referenced_tables = set()
            for row in plan:
                plan_detail = row[3].lower()
                for table in tables.keys():
                    if table.lower() in plan_detail:
                        referenced_tables.add(table)
            
            metadata = {
                'referenced_tables': list(referenced_tables),
                'schema': {t: list(cols.keys()) for t, cols in tables.items()},
                'plan': plan
            }
            
            return True, "", metadata
            
        except sqlite3.Error as e:
            return False, f"SQL syntax error: {str(e)}", {}
            
    except Exception as e:
        return False, f"Validation error: {str(e)}", {}
        
    finally:
        if 'conn' in locals():
            conn.close()

def execute_sql_query(query: str, db_path: str) -> Tuple[pd.DataFrame, str]:
    """Execute SQL query and return results with metadata."""
    try:
        # First validate the query
        is_valid, error_msg, metadata = validate_sql_query(query, db_path)
        if not is_valid:
            raise Exception(error_msg)
        
        db = DatabaseManager(db_path)
        
        # First, explain the query
        explain = db.execute_query(f"EXPLAIN QUERY PLAN {query}")
        plan = "\n".join([str(row) for row in explain])
        
        # Then execute it
        results = db.execute_query(query)
        df = pd.DataFrame(results.fetchall(), columns=[desc[0] for desc in results.description])
        
        # Format table with proper markdown code block
        preview = "\n\n```\n" + df.head().to_string() + "\n```\n\n"
        
        metadata = {
            'rows': len(df),
            'columns': list(df.columns),
            'execution_plan': plan
        }
        
        return df, metadata, preview  # Note: now returns 3 items
            
    except Exception as e:
        raise Exception(f"Query execution failed: {str(e)}")

def store_successful_query(query_id: str, sql: str, metadata: dict) -> dict:
    """Store a successful query execution and its results."""
    return {
        'query_id': query_id,
        'sql': sql,
        'metadata': {
            **metadata,
            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    }

####################################
#
# Layout Functions
#
####################################

def create_database_tab():
    """Create the database information tab layout."""
    return html.Div([
        dbc.Tabs([
            dbc.Tab(
                label="Table Summary",
                children=html.Div(id='database-summary'),
                tab_id="tab-summary"
            ),
            dbc.Tab(
                label="ER Diagram",
                children=html.Div([
                    html.Div(
                        id='database-erd',
                        style={
                            'width': '100%',
                            'height': '600px',
                            'overflow': 'auto',
                            'position': 'relative',
                            'backgroundColor': 'white',
                            'border': '1px solid #e0e0e0',
                            'borderRadius': '4px',
                            'padding': '15px'
                        }
                    )
                ]),
                tab_id="tab-erd"
            )
        ], id="database-view-tabs", active_tab="tab-summary")
    ])

def create_weaviate_tab():
    """Create the Weaviate information tab layout."""
    return html.Div([
        dbc.Tabs([
            dbc.Tab(
                label="Table Summary",
                children=html.Div(id='weaviate-summary'),
                tab_id="tab-weaviate-summary"
            ),
            dbc.Tab(
                label="ER Diagram",
                children=html.Div([
                    html.Div(
                        id='weaviate-erd',
                        style={
                            'width': '100%',
                            'height': '600px',
                            'overflow': 'auto',
                            'border': '1px solid #ddd',
                            'borderRadius': '5px',
                            'padding': '10px'
                        }
                    )
                ]),
                tab_id="tab-weaviate-erd"
            )
        ], id="weaviate-view-tabs")
    ])

def create_dataset_card(name: str, data: Dict[str, Any]) -> dbc.Card:
    """Create a card component for displaying dataset information."""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Div([  # Left side with checkbox and name
                    dbc.Checkbox(
                        id={'type': 'dataset-checkbox', 'index': name},
                        className='mr-2 d-inline-block',
                        style={'verticalAlign': 'middle'},
                        value=False,  # Initial value
                        persistence=False,  # Don't persist state
                        persistence_type='local'  # Only needed if persistence=True
                    ),
                    dbc.Button(
                        html.H6(name, className="d-inline"),
                        id={'type': 'dataset-card', 'index': name},
                        color="link",
                        className="p-0",
                        style={'textDecoration': 'none', 'width': '90%', 'textAlign': 'left'}
                    ),
                ], style={'display': 'inline-block', 'width': '90%'}),
                html.Div([  # Right side with delete button
                    dbc.Button(
                        "×",
                        id={'type': 'delete-dataset', 'index': name},
                        color="link",
                        style={'color': 'red'}
                    ),
                ], style={'display': 'inline-block', 'width': '10%', 'textAlign': 'right'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            html.Small(f"Rows: {data['metadata']['rows']}, Columns: {len(data['metadata']['columns'])}")
        ]),
        className="mb-2",
        id={'type': 'dataset-card-container', 'index': name}
    )

# Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Add this at the top of your layout
    dcc.Store(id='datasets-store', storage_type='memory', data={}),
    dcc.Store(id='selected-dataset-store', storage_type='memory'),  # NEW: Track selected dataset
    dcc.Store(id='chat-store', data=[]),
    dcc.Store(id='database-state', data={'connected': False, 'path': None}),
    dcc.Store(id='database-structure-store', data=None),
    dcc.Store(id='viz-state-store', data={
        'type': None,
        'params': {},
        'data': {}
    }),
    dcc.Store(id='successful-queries-store', storage_type='memory', data={}),
    # Add Weaviate state store
    dcc.Store(id='weaviate-state', data=None),
    dcc.Store(id='_weaviate-init', data=True),  # Hidden trigger for initial connection
    # Add interval for periodic status checks
    dcc.Interval(id='database-status-interval', interval=30000),  # 30 seconds
    dbc.Container([
        # Top Row with Dataset Browser and Info
        dbc.Row([
            # Data Management Column (20%)
            dbc.Col([
                html.H3("Data Management", className="mb-4"),
                # Add Database Selection
                html.Div([
                    html.Div([
                        dcc.Dropdown(
                            id='database-path-dropdown',
                            options=get_database_files(),
                            placeholder='Select a database',
                            style={'width': '60%'}  # Reduced width to make room for buttons
                        ),
                        html.Button(
                            'Connect', 
                            id='database-connect-button', 
                            n_clicks=0,
                            style={'marginLeft': '10px'}
                        ),
                        html.Button(
                            '🔄', 
                            id='refresh-database-list',
                            n_clicks=0,
                            title='Refresh database list',
                            style={
                                'marginLeft': '10px',
                                'fontSize': '20px',
                                'verticalAlign': 'middle',
                                'border': 'none',
                                'background': 'none',
                                'cursor': 'pointer'
                            }
                        ),
                    ], style={
                        'display': 'flex',  # Use flexbox
                        'alignItems': 'center',  # Vertically center items
                        'marginBottom': '10px'
                    }),
                    # Add Weaviate status indicators
                    dbc.Row([
                        dbc.Col([
                            html.Div(id='database-connection-status', className="mb-3"),
                            html.Div([
                                html.Span("Weaviate: ", className="mr-2"),
                                dbc.Tooltip(
                                    id='weaviate-connection-tooltip',
                                    target='weaviate-connection-icon',
                                    placement='top'
                                ),
                                html.I(id='weaviate-connection-icon', className="fas fa-circle", 
                                     style={'color': '#6c757d', 'marginRight': '5px'}),  # Neutral gray
                                dbc.Tooltip(
                                    id='weaviate-collections-tooltip',
                                    target='weaviate-collections-icon',
                                    placement='top'
                                ),
                                html.I(id='weaviate-collections-icon', className="fas fa-database",
                                     style={'color': '#6c757d'})  # Neutral gray
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '5px'})
                        ])
                    ])
                ]),               
                # Upload Component
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files'),
                        html.Div('Upload .csv, .tsv, or .zip files'),
                        dbc.Spinner(
                            html.Div(id="upload-loading", style={
                                'height': '20px',
                                'margin': '10px',
                                'color': '#666'
                            }),
                            color="primary",
                            size="sm",
                            spinner_style={'margin': '10px'}
                        )
                    ], style={
                        'textAlign': 'center',
                        'padding': '20px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'backgroundColor': '#fafafa',
                    }),
                    multiple=True,
                    accept='.csv, .tsv, .zip, .CSV, .TSV, .ZIP',
                    className="mb-3"
                ),
                # Dataset Browser
                html.Div([
                    html.H5(html.Div([
                        "Loaded Datasets ",
                        html.Span(id='dataset-count', className="text-muted"),
                    ])),
                    html.Div(id='memory-usage', className="small text-muted mb-2"),
                    html.Div([
                        html.Div(id='dataset-list', style={
                            'maxHeight': '300px',
                            'overflowY': 'auto',
                            'border': '1px solid #ddd',
                            'borderRadius': '5px',
                            'padding': '10px'
                        }),
                        dcc.Store(id='selected-datasets-store', data=[]),
                        dcc.Download(id='download-selected-datasets'),
                        html.Button(
                            'Download Selected Datasets',
                            id='download-button',
                            style={'display': 'none'},
                            className='mt-2 btn btn-primary'
                        )
                    ]),
                ]),
            ], width=3),  # Changed from 4 to 3 (25% -> ~20%)
            
            # Dataset Info Column (80%)
            dbc.Col([
                html.H3("Dataset Info", className="mb-4"),
                html.Div([
                    dbc.Tabs([
                        dbc.Tab(label="Preview", tab_id="tab-preview"),
                        dbc.Tab(label="Statistics", tab_id="tab-stats"),
                        dbc.Tab(label="Visualization", tab_id="tab-viz"),
                        dbc.Tab(label="Database", tab_id="tab-db", children=create_database_tab()),
                        dbc.Tab(label="Weaviate", tab_id="tab-weaviate", children=create_weaviate_tab())  # Add Weaviate tab
                    ],
                    id="dataset-tabs",
                    active_tab="tab-preview"
                    ),
                    html.Div(id="dataset-tab-content", className="p-3")
                ], style={
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'backgroundColor': '#f8f9fa',
                    'minHeight': '200px'
                })
            ], width=9)  # Changed from 8 to 9 (75% -> ~80%)
        ], className="mb-4"),
        
        # Bottom Row with Chat Interface (full width)
        dbc.Row([
            dbc.Col([
                html.H3("AI Chat Interface", className="mb-4"),
                dcc.Dropdown(
                    id='model-selector',
                    options=[{'label': model, 'value': model} for model in AVAILABLE_MODELS],
                    value='anthropic/claude-sonnet',
                    className="mb-3"
                ),
                html.Div(
                    id='chat-history',
                    style={
                        'height': '400px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'padding': '10px',
                        'marginBottom': '10px',
                        'backgroundColor': '#f8f9fa'
                    }
                ),
                dbc.Spinner(
                    html.Div(id='chat-loading-output'),
                    color="primary",
                    type="border",
                ),
                dbc.Row([
                    dbc.Col([
                        dcc.Textarea(
                            id='chat-input',
                            style={
                                'width': '100%',
                                'height': '100px',
                                'padding': '10px'
                            },
                            placeholder="Type your message here...",
                        ),
                    ], width=10),
                    dbc.Col([
                        dbc.Button("Send", id="send-button", color="primary", className="mt-4"),
                        dbc.Button("Help", id="help-button", color="info", className="mt-4 ml-2"),
                        dbc.Button("Download", id="download-chat-button", color="secondary", className="mt-4 ml-2"),  # Added this line
                        dcc.Download(id="download-chat"),  # Added this line - required by Dash for downloads
                    ], width=2),
                ], className="mb-3"),
            ], width=12)  # Full width
        ])
    ], fluid=True),
    
])

####################################
#
# Error Handling Functions
#
####################################

def handle_upload_error(filename: str, error: Exception) -> str:
    """
    Generate user-friendly error messages for dataset upload issues.
    """
    error_type = type(error).__name__

    if isinstance(error, pd.errors.EmptyDataError):
        return f"Error: {filename} is empty"
    elif isinstance(error, UnicodeDecodeError):
        return f"Error: {filename} has invalid encoding. Please ensure it's UTF-8"
    elif isinstance(error, pd.errors.ParserError):
        return f"Error: {filename} has invalid format. Please check the file structure"
    return f"Error processing {filename}: {str(error)}"

def validate_dataset(df: pd.DataFrame, filename: str) -> tuple[bool, str]:
    """
    Validate dataset structure and content.

    Performs checks for:
    - Empty dataframes
    - Duplicate column names
    - Duplicate index values

    Args:
        df (pd.DataFrame): DataFrame to validate
        filename (str): Name of the file being validated

    Returns:
        tuple[bool, str]: (is_valid, error_message)

    Example:
        >>> is_valid, msg = validate_dataset(df, "data.csv")
        >>> if not is_valid: print(msg)
    """

    if df.empty:
        return False, f"Error: {filename} is empty"
    if df.columns.duplicated().any():
        return False, f"Error: {filename} contains duplicate column names"
    if df.index.duplicated().any():
        return False, f"Error: {filename} contains duplicate index values"
    return True, ""

####################################
#
# Main Chat Input Callback
#
####################################

@callback(
    [Output('chat-history', 'children'),
     Output('chat-input', 'value', allow_duplicate=True),
     Output('chat-store', 'data'),
     Output('dataset-tabs', 'active_tab', allow_duplicate=True),  # Fix hyphen to underscore
     Output('viz-state-store', 'data'),
     Output('chat-loading-output', 'children', allow_duplicate=True),
     Output('successful-queries-store', 'data', allow_duplicate=True)],  # Add output for store
    [Input('send-button', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-store', 'data'),
     State('model-selector', 'value'),
     State('datasets-store', 'data'),
     State('selected-dataset-store', 'data'),
     State('database-state', 'data'),           
     State('database-structure-store', 'data'),
     State('successful-queries-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def handle_chat_message(n_clicks, input_value, chat_history, model, datasets, selected_dataset, database_state, database_structure_store, successful_queries):
    """Process chat messages and handle various command types."""
    try:
        if not input_value:
            return (dash.no_update,) * 7
            
        chat_history = chat_history or []
        current_message = {'role': 'user', 'content': input_value.strip()}

        # Handle help request
        if input_value.lower().strip() in ["help", "help me", "what can i do?", "what can i do", "what can you do?", "what can you do"]:
            chat_history.append(current_message)
            chat_history.append({
                'role': 'assistant',
                'content': help_message
            })
            return (
                create_chat_elements_batch(chat_history),
                '',
                chat_history,
                dash.no_update,
                dash.no_update,
                "",
                dash.no_update  # No store update needed
            )
        
        # Add service detection wrapper
        handlers = service_registry.detect_handlers(input_value)
        if len(handlers) > 1:
            # Multiple services detected - add warning message
            warning = ServiceMessage(
                service="system",
                content=f"Warning: Multiple services ({', '.join(name for name, _ in handlers)}) attempted to handle this message. This may indicate a service configuration issue.",
                message_type="warning"
            )
            chat_history.append(warning.to_chat_message())
            
        elif len(handlers) == 1:
            # Single service detected - execute it
            service_name, service = handlers[0]
            print(f"\n=== Service Execution Debug ===")
            print(f"Message: {input_value}")
            print(f"Executing service: {service_name}")
            
            try:
                # Add message to chat history first
                chat_history.append(current_message)
                
                # Parse request
                params = service.parse_request(input_value)
                
                # Create execution context
                context = {
                    'datasets_store': datasets,
                    'successful_queries_store': successful_queries,
                    'selected_dataset': selected_dataset,
                    'database_state': database_state,
                    'database_structure': database_structure_store,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Execute service
                response = service.execute(params, context)
                
                # Add service messages to chat
                for msg in response.messages:
                    chat_history.append(msg.to_chat_message())
                
                # Create system message with context for LLM
                if response.context:
                    system_message = create_system_message(
                        dataset_info=[{
                            'name': name,
                            'rows': len(pd.DataFrame(data['df'])),
                            'columns': list(pd.DataFrame(data['df']).columns),
                            'selected': name == selected_dataset
                        } for name, data in datasets.items()] if datasets else [],
                        database_structure=database_structure_store,
                        service_context=response.context.to_dict()
                    )
                else:
                    system_message = create_system_message(
                        dataset_info=[{
                            'name': name,
                            'rows': len(pd.DataFrame(data['df'])),
                            'columns': list(pd.DataFrame(data['df']).columns),
                            'selected': name == selected_dataset
                        } for name, data in datasets.items()] if datasets else [],
                        database_structure=database_structure_store
                    )
                
                # Get LLM response
                messages = [
                    {'role': 'system', 'content': system_message},
                    *[{'role': msg['role'], 'content': str(msg['content'])} for msg in chat_history]
                ]
                
                llm_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.4,
                    max_tokens=8192
                )
                
                # Add LLM response to chat
                chat_history.append({
                    'role': 'assistant',
                    'content': llm_response.choices[0].message.content
                })
                
                # Update stores if needed
                if response.store_updates:
                    successful_queries.update(response.store_updates)
                
                # Return with any state updates
                return (
                    create_chat_elements_batch(chat_history),
                    '',
                    chat_history,
                    response.state_updates.get('active_tab', dash.no_update),
                    response.state_updates.get('viz_state', dash.no_update),
                    "",
                    successful_queries
                )
                
            except Exception as e:
                print(f"Service execution error: {str(e)}")
                error_msg = ServiceMessage(
                    service=service_name,
                    content=f"Error executing service: {str(e)}",
                    message_type="error"
                )
                chat_history.append(error_msg.to_chat_message())
                return (
                    create_chat_elements_batch(chat_history),
                    '',
                    chat_history,
                    dash.no_update,
                    dash.no_update,
                    "",
                    successful_queries
                )

        # Skip old literature handling if service handled it
        literature_service_handled = any(
            handler[1].name == "literature" 
            for handler in service_registry.detect_handlers(input_value)
        )
        
        if not literature_service_handled:
            # Check for threshold refinement
            threshold = extract_threshold_from_message(input_value)
            query_id = extract_query_id_from_message(input_value) if threshold else None
            
            if threshold is not None and query_id is not None:
                print(f"\n=== Processing Threshold Refinement (Legacy) ===")
                print(f"Query ID: {query_id}")
                print(f"New threshold: {threshold}")
                
                
                chat_history.append(current_message)
                
                if query_id not in successful_queries:
                    chat_history.append({
                        'role': 'assistant',
                        'content': f"❌ Query {query_id} not found in history."
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        "",
                        successful_queries
                    )
                
                try:
                    stored_query = successful_queries[query_id]
                    original_query = stored_query['query']
                    
                    # Execute query with new threshold
                    weaviate_results = execute_weaviate_query(original_query, min_score=threshold)
                    if not weaviate_results or not weaviate_results.get('unified_results'):
                        raise Exception("No results found with new threshold")
                    
                    # Transform and store results
                    df = transform_weaviate_results(weaviate_results)
                    new_query_id = generate_literature_query_id(original_query, threshold)
                    
                    successful_queries[new_query_id] = {
                        'query': original_query,
                        'threshold': threshold,
                        'dataframe': df.to_dict('records'),
                        'metadata': {
                            'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'total_results': len(df),
                            'query_info': df.attrs.get('query_info', {}),
                            'summary': df.attrs.get('summary', {})
                        }
                    }
                    
                    # Format response with comparison
                    old_df = pd.DataFrame(stored_query['dataframe'])
                    response = f"""Refined search results:

    Previous results (threshold {stored_query['threshold']}): {len(old_df)} matches
    New results (threshold {threshold}): {len(df)} matches

    {format_literature_preview(df, new_query_id, threshold)}"""
                    
                    chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        "",
                        successful_queries
                    )
                    
                except Exception as e:
                    error_msg = f"Error refining query: {str(e)}"
                    print(error_msg)
                    chat_history.append({
                        'role': 'assistant',
                        'content': f"❌ {error_msg}"
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        "",
                        successful_queries
                    )
        

            
        chat_history.append(current_message)
        
        # Add visualization detection early in the flow
        viz_handler = VisualizationHandler()
        viz_type = viz_handler.detect_type(input_value)
        
        if viz_type:
            # Check for available datasets
            if not datasets:
                chat_history.append({
                    'role': 'assistant',
                    'content': f"Note: Detected a {viz_type} visualization request, but no datasets are currently loaded. Please load a dataset first."
                })
            elif not selected_dataset:
                chat_history.append({
                    'role': 'assistant',
                    'content': f"Note: Detected a {viz_type} visualization request, but no dataset is selected. Please select a dataset first."
                })
            else:
                # Process visualization request
                df = pd.DataFrame(datasets[selected_dataset]['df'])
                viz_state, error = viz_handler.process_request(input_value, df)
                
                if error:
                    chat_history.append({
                        'role': 'assistant',
                        'content': f"Visualization error: {error}"
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        "",
                        dash.no_update  # No store update needed
                    )
                
                chat_history.append({
                    'role': 'assistant',
                    'content': f"Creating {viz_type} visualization. Switching to visualization tab."
                })
                
                return (
                    create_chat_elements_batch(chat_history),
                    '',
                    chat_history,
                    'tab-viz',  # Switch to visualization tab
                    viz_state,  # Update visualization state
                    "",
                    successful_queries  # Add the store to all returns
                )
        
        # Create system message with all available context
        system_message = create_system_message(
            dataset_info=[{
                'name': name,
                'rows': len(pd.DataFrame(data['df'])),
                'columns': list(pd.DataFrame(data['df']).columns),
                'selected': name == selected_dataset
            } for name, data in datasets.items()] if datasets else [],
            database_structure=database_structure_store
        )
        
        # Smart context selection
        def get_relevant_context(current_msg: dict, history: list, max_context: int = 6) -> list:
            """Select relevant context messages, preserving order and relationships."""
            context = []
            # Always include the current message
            context.append(current_msg)
            
            # Look backwards through history for relevant messages
            for msg in reversed(history[:-1]):  # Exclude current message
                if len(context) >= max_context:
                    break
                    
                # Check relevance based on content
                content = msg['content'].lower()
                current_content = current_msg['content'].lower()
                
                # Always include the immediate previous message
                if len(context) == 1:
                    context.insert(0, msg)
                    continue
                
                # Include messages with SQL blocks
                if '```sql' in content:
                    context.insert(0, msg)
                    continue
                    
                # Include messages that seem related by content
                # Look for shared significant terms (excluding common words)
                current_terms = set(re.findall(r'\b\w+\b', current_content))
                msg_terms = set(re.findall(r'\b\w+\b', content))
                shared_terms = current_terms & msg_terms
                if len(shared_terms) >= 2:  # At least 2 significant shared terms
                    context.insert(0, msg)
                    
            return context
        
        # Handle search queries
        if not literature_service_handled and is_search_query(input_value):
            print("\n=== Processing Search Query ===")
            search_results, successful_queries = search_all_sources(input_value, successful_queries=successful_queries)
            
            # Add search results to chat history
            search_summary = format_search_results(search_results)
            chat_history.append({'role': 'assistant', 'content': search_summary})
            
            # Get LLM interpretation with relevant context
            context = get_relevant_context(current_message, chat_history)
            messages = [
                {'role': 'system', 'content': system_message},
                *[{'role': msg['role'], 'content': str(msg['content'])} for msg in context],
                {'role': 'assistant', 'content': search_summary},
                {'role': 'user', 'content': "Please analyze these results and suggest relevant SQL queries if appropriate."}
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=8192
            )
            
            ai_response = response.choices[0].message.content
            
            # Process SQL blocks if present
            if '```sql' in ai_response.lower():
                ai_response = add_query_ids_to_response(ai_response)
            
            chat_history.append({'role': 'assistant', 'content': ai_response})
            
        else:
            # Handle non-search queries
            context = get_relevant_context(current_message, chat_history)
            messages = [
                {'role': 'system', 'content': system_message},
                *[{'role': msg['role'], 'content': str(msg['content'])} for msg in context]
            ]
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.4,
                max_tokens=8192
            )
            
            ai_response = response.choices[0].message.content
            
            # Process SQL blocks if present
            if '```sql' in ai_response.lower():
                ai_response = add_query_ids_to_response(ai_response)
            
            chat_history.append({'role': 'assistant', 'content': ai_response})
        
        return (
            create_chat_elements_batch(chat_history),
            '',
            chat_history,
            dash.no_update,
            dash.no_update,
            "",
            successful_queries  # Add the store to all returns
        )
        
    except Exception as e:
        print(f"Error in handle_chat_message: {str(e)}")
        print(traceback.format_exc())
        return (dash.no_update,) * 7  # Updated for new output

####################################
#
# Weaviate Management Functions
#
####################################    

def get_weaviate_client():
    """Get or create Weaviate client instance."""
    try:
        connection = WeaviateConnection()
        with connection.get_client() as client:
            if not client:
                raise Exception("No Weaviate client available")
            return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
        return None

def execute_weaviate_query(query: str, min_score: float = 0.3) -> dict:
    """Execute a query through weaviate_manager.
    
    Args:
        query: Search query text
        min_score: Minimum score threshold
        
    Returns:
        Dict containing query results or empty dict if no results/error
    """
    print("\n=== Weaviate Query Debug ===")
    print(f"Query: '{query}'")
    print(f"Min score: {min_score}")
    
    try:
        connection = WeaviateConnection()
        with connection.get_client() as client:
            if not client:
                print("Error: No Weaviate client available")
                return {}
            
            print("Client connection successful")
            
            # Use the QueryManager from weaviate_manager
            query_manager = QueryManager(client)
            print("QueryManager initialized")
            
            print("Executing comprehensive search...")
            results = query_manager.comprehensive_search(
                query_text=query,
                search_type="hybrid",
                min_score=min_score,
                unify_results=True,  # Get unified article view
                verbose=True  # For debugging
            )
            
            print("\nSearch Results:")
            print(f"- Raw results: {bool(results)}")
            print(f"- Has unified_results: {bool(results and 'unified_results' in results)}")
            if results and 'unified_results' in results:
                print(f"- Number of unified results: {len(results['unified_results'])}")
                if results['unified_results']:
                    first_result = results['unified_results'][0]
                    print(f"- First result score: {first_result.get('score', 'N/A')}")
                    print(f"- First result collection: {first_result.get('collection', 'N/A')}")
            
            if not results or not results.get('unified_results'):
                print("No results found")
                return {}
                
            return results
            
    except Exception as e:
        print(f"Error in execute_weaviate_query: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        return {}

####################################
#
# Chat Management Functions
#
####################################

# Download chat history
@callback(
    Output("download-chat", "data"),
    Input("download-chat-button", "n_clicks"),
    State("chat-store", "data"),
    prevent_initial_call=True
)
def download_chat_history(n_clicks, chat_history):
    if not n_clicks or not chat_history:
        return None
        
    # Format chat history with timestamps and metadata
    formatted_history = []
    for msg in chat_history:
        entry = {
            'timestamp': msg.get('timestamp', datetime.now().isoformat()),
            'role': msg['role'],
            'content': msg['content'],
            'type': 'text'
        }
        
        # Special handling for SQL queries
        if msg['role'] == 'user' and '```sql' in msg['content']:
            entry['type'] = 'sql'
            entry['sql'] = msg['content'].split('```sql')[1].split('```')[0].strip()
            
        formatted_history.append(entry)
    
    # Create filename with timestamp
    filename = f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    return dict(
        content=json.dumps(formatted_history, indent=2),
        filename=filename,
        type='application/json'
    )

# Add a new callback for handling the API call
@callback(
    [Output('chat-history', 'children', allow_duplicate=True),
     Output('chat-store', 'data', allow_duplicate=True),
     Output('chat-input', 'value', allow_duplicate=True),
     Output('chat-loading-output', 'children', allow_duplicate=True)],
    [Input('chat-store', 'data')],
    [State('model-selector', 'value')],
    prevent_initial_call='initial_duplicate'
)
def process_api_call(chat_history, model):
    """Handle API calls to the AI model."""
    if not chat_history or len(chat_history) == 0:
        return dash.no_update, dash.no_update, dash.no_update, ""
    
    # Only process if the last message is from the user
    if chat_history[-1]['role'] != 'user':
        return dash.no_update, dash.no_update, dash.no_update, ""
    
    try:
        messages = [
            {'role': msg['role'], 'content': msg['content']}
            for msg in chat_history
        ]
        print(f"current model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        chat_history.append({
            'role': 'assistant',
            'content': ai_response
        })
        
        chat_elements = create_chat_elements_batch(chat_history)
        return chat_elements, chat_history, '', ""

    except Exception as e:

        error_message = f"Error: {str(e)}"
        chat_history.append({
            'role': 'system',
            'content': error_message
        })
        chat_elements = create_chat_elements_batch(chat_history)
        return chat_elements, chat_history, '', ""

# Optimize chat message processing
def create_chat_elements_batch(messages: list) -> list:
    """
    Create chat elements in a batch for better performance.
    
    Args:
        messages (list): List of chat messages
        
    Returns:
        list: List of chat element components
    """
    return [create_chat_element(msg) for msg in messages]

####################################
#
# Dataset Handling Functions
#
####################################

# MODIFIED: Upload handler with profile generation
@callback(
    [Output('datasets-store', 'data'),
     Output('dataset-list', 'children'),
     Output('upload-loading', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified'),
     State('datasets-store', 'data')],
    prevent_initial_call=True
)
def handle_dataset_upload(
    list_of_contents: Optional[List[str]], 
    list_of_names: List[str],
    list_of_dates: List[str], 
    existing_datasets: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Any], str]:
    """Process uploaded datasets, including zip files.
    
    Args:
        list_of_contents: Base64 encoded file contents
        list_of_names: Original filenames
        list_of_dates: File modification dates
        existing_datasets: Currently loaded datasets
        
    Returns:
        Tuple containing:
        - Updated datasets store
        - List of dataset card components
        - Upload status message
        
    Raises:
        pd.errors.EmptyDataError: If uploaded file contains no data
        UnicodeDecodeError: If file encoding is not supported
        pd.errors.ParserError: If file format is invalid
    """
    if list_of_contents is None:
        return dash.no_update, dash.no_update, ""

    datasets = existing_datasets or {}
    errors = []
    new_datasets = {}

    for content, name, date in zip(list_of_contents, list_of_names, list_of_dates):
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)

            # Handle zip files
            if name.lower().endswith('.zip'):
                try:
                    zip_buffer = io.BytesIO(decoded)
                    with zipfile.ZipFile(zip_buffer) as zip_file:
                        for zip_name in zip_file.namelist():
                            # Skip Mac OS metadata files
                            if '__MACOSX' in zip_name or zip_name.startswith('._'):
                                continue
                                
                            if zip_name.lower().endswith(('.csv', '.tsv')):
                                with zip_file.open(zip_name) as file:
                                    # Try different encodings
                                    content = None
                                    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                                    for encoding in encodings:
                                        try:
                                            content = file.read().decode(encoding)
                                            break
                                        except UnicodeDecodeError:
                                            continue
                                            
                                    if content is None:
                                        errors.append(f"Could not decode {zip_name} with any supported encoding")
                                        continue
                                        
                                    try:
                                        df = pd.read_csv(
                                            io.StringIO(content),
                                            index_col=0,
                                            low_memory=False,
                                            sep=',' if zip_name.lower().endswith('.csv') else '\t'
                                        )
                                        
                                        if df.empty:
                                            errors.append(f"Empty dataframe found in {zip_name}")
                                            continue
                                            
                                        clean_name = zip_name.rsplit('.', 1)[0]
                                        df = process_dataframe(df, zip_name)
                                        
                                        is_valid, error_msg = validate_dataset(df, zip_name)
                                        if not is_valid:
                                            errors.append(error_msg)
                                            continue

                                        df.columns = df.columns.str.replace(r'[.\[\]{}]', '_', regex=True)
                                        
                                        # Generate profile report during upload
                                        try:
                                            profile = ProfileReport(
                                                df,
                                                minimal=True,
                                                title=f"Profile Report for {clean_name}",
                                                html={'style': {'full_width': True}},
                                                progress_bar=False,
                                                correlations={'pearson': {'calculate': True}},
                                                missing_diagrams={'matrix': False},
                                                samples=None
                                            )
                                            profile_html = profile.to_html()
                                        except Exception as e:
                                            profile_html = None
                                        
                                        new_datasets[clean_name] = {
                                            'df': df.reset_index().to_dict('records'),
                                            'metadata': {
                                                'filename': zip_name,
                                                'source': f"Zip file: {name}",
                                                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                'rows': len(df),
                                                'columns': [df.index.name or 'index'] + list(df.columns)
                                            },
                                            'profile_report': profile_html
                                        }
                                        
                                        try:
                                            text_searcher.update_dataset(clean_name, df)
                                        except Exception as e:
                                            pass
                                            
                                    except pd.errors.EmptyDataError:
                                        errors.append(f"No data found in {zip_name}")
                                    except Exception as e:
                                        errors.append(f"Error processing {zip_name}: {str(e)}")
                                        
                except Exception as e:
                    errors.append(f"Error processing zip file {name}: {str(e)}")
                continue

            # Handle individual CSV/TSV files
            try:
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')),
                    index_col=0,
                    low_memory=False
                ) if name.lower().endswith('.csv') else pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')),
                    sep='\t',
                    index_col=0,
                    low_memory=False
                )
                
                clean_name = name.rsplit('.', 1)[0]
                df = process_dataframe(df, name)
                
                is_valid, error_msg = validate_dataset(df, name)
                if not is_valid:
                    errors.append(error_msg)
                    continue

                df.columns = df.columns.str.replace(r'[.\[\]{}]', '_', regex=True)
                
                # Generate profile report during upload
                try:
                    profile = ProfileReport(
                        df,
                        minimal=True,
                        title=f"Profile Report for {clean_name}",
                        html={'style': {'full_width': True}},
                        progress_bar=False,
                        correlations={'pearson': {'calculate': True}},
                        missing_diagrams={'matrix': False},
                        samples=None
                    )
                    profile_html = profile.to_html()
                except Exception as e:
                    profile_html = None
                
                new_datasets[clean_name] = {
                    'df': df.reset_index().to_dict('records'),
                    'metadata': {
                        'filename': name,
                        'source': f"Uploaded file: {name}",
                        'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'rows': len(df),
                        'columns': [df.index.name or 'index'] + list(df.columns)
                    },
                    'profile_report': profile_html
                }

                try:
                    text_searcher.update_dataset(clean_name, df)
                except Exception as e:
                    pass
                
            except Exception as e:
                errors.append(handle_upload_error(name, e))
                continue

        except Exception as e:
            errors.append(f"Unexpected error with {name}: {str(e)}")
            continue

    datasets.update(new_datasets)
    dataset_list = [create_dataset_card(name, data) for name, data in datasets.items()]
    
    error_message = html.Div([
        html.P("Errors occurred during upload:", className="text-danger") if errors else "",
        html.Ul([html.Li(error) for error in errors]) if errors else ""
    ]) if errors else ""
    
    return datasets, dataset_list if dataset_list else [], error_message

def process_dataframe(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Process dataframe to handle missing values and type inference.
    
    Performs the following operations:
    1. Replaces various missing value indicators with NaN
    2. Attempts to convert string columns to numeric where appropriate
    3. Handles basic data cleaning and standardization
    
    Args:
        df (pd.DataFrame): Input dataframe to process
        filename (str): Original filename for error reporting
        
    Returns:
        pd.DataFrame: Processed dataframe with standardized missing values
        and appropriate data types
        
    Raises:
        ValueError: If dataframe cannot be properly processed
        TypeError: If column type conversion fails
    """
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
        
        # Replace all missing value types with NaN
        df = df.replace(missing_values, np.nan)
        
        # Try to convert numeric columns
        for col in df.columns:
            try:
                # Only attempt numeric conversion if the column is string type
                if df[col].dtype == object:
                    # Check if column contains numbers (allowing for NaN)
                    non_nan = df[col].dropna()
                    if len(non_nan) > 0 and non_nan.astype(str).str.match(r'^-?\d*\.?\d+$').all():
                        df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                continue
                
        return df
        
    except Exception as e:
        return df

@callback(
    [Output('download-button', 'style', allow_duplicate=True),
     Output('selected-datasets-store', 'data', allow_duplicate=True)],
    [Input({'type': 'dataset-checkbox', 'index': ALL}, 'value')],
    [State({'type': 'dataset-checkbox', 'index': ALL}, 'id')],
    prevent_initial_call=True
)
def update_selected_datasets(checked_values, checkbox_ids):
    """Track selected datasets and show/hide download button."""
    
    # Handle case where no checkboxes exist
    if not checked_values:
        return {'display': 'none'}, []
    
    # Convert None to False and get selected datasets
    checked_values = [bool(v) for v in checked_values]
    
    selected = [
        checkbox_ids[i]['index'] 
        for i, checked in enumerate(checked_values) 
        if checked
    ]
    
    button_style = {
        'display': 'block' if any(checked_values) else 'none'
    }
    
    return button_style, selected

# Add callback for deleting datasets
@callback(
    [Output('datasets-store', 'data', allow_duplicate=True),
     Output('dataset-list', 'children', allow_duplicate=True),
     Output('selected-dataset-store', 'data', allow_duplicate=True),
     Output('dataset-tab-content', 'children', allow_duplicate=True)],
    [Input({'type': 'delete-dataset', 'index': ALL}, 'n_clicks')],
    [State('datasets-store', 'data'),
     State('selected-dataset-store', 'data')],
    prevent_initial_call=True
)
def delete_dataset(n_clicks, datasets, selected_dataset):
    """Handle dataset deletion and update all relevant components."""
    if not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    try:
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        trigger = ctx.triggered[0]
        prop_id = trigger['prop_id']
        
        # Use string parsing instead of JSON
        if '"index":"' in prop_id:
            dataset_name = prop_id.split('"index":"')[1].split('"')[0]
            
            if dataset_name in datasets:
                del datasets[dataset_name]
                
            dataset_list = [create_dataset_card(name, data) for name, data in datasets.items()]
            new_selected = None if selected_dataset == dataset_name else selected_dataset
            tab_content = "Please select a dataset" if selected_dataset == dataset_name else dash.no_update
            
            return datasets, dataset_list if dataset_list else [], new_selected, tab_content
        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Add callback for updating dataset info
@callback(
    Output('dataset-tab-content', 'children'),
    [Input('dataset-tabs', 'active_tab'),
     Input('selected-dataset-store', 'data')],
    [State('datasets-store', 'data')]
)
def render_tab_content(active_tab, selected_dataset, datasets):
    """Render content for dataset tabs."""
    
    if not datasets:
        return "No datasets selected"
        
    if not selected_dataset:
        return "Please select a dataset"
        
    if selected_dataset not in datasets:
        return f"Dataset '{selected_dataset}' not found"

    try:
        dataset = datasets[selected_dataset]
        df = pd.DataFrame(dataset['df'])

        if active_tab == "tab-preview":
            return [
                html.H4(selected_dataset),
                html.P(f"Source: {dataset['metadata']['source']}"),
                dash_table.DataTable(
                    data=df.head().to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={'fontWeight': 'bold'}
                )
            ]
        elif active_tab == "tab-stats":
            if dataset.get('profile_report'):
                return html.Iframe(
                    srcDoc=dataset['profile_report'],
                    style={'width': '100%', 'height': '800px', 'border': 'none'}
                )
            return html.Div([
                html.H4("Statistics Not Available"),
                html.P("Profile report was not generated during upload.")
            ])
        elif active_tab == "tab-viz":
            return html.Div([
                html.H4("Data Visualization", style={'color': 'blue'}),  # Make this visible
                html.Div("Debug: Viz Controls Container", style={'border': '1px solid green', 'padding': '10px', 'margin': '10px'}),
                html.Div(
                    id='viz-container',
                    children="Debug: Empty Viz Container",
                    style={'border': '1px solid red', 'padding': '10px', 'margin': '10px', 'minHeight': '200px'}
                ),
                html.Pre(id='viz-debug', children="Debug: Viz Debug Area")  # Add debug area
            ])
            
    except Exception as e:
        return html.Div([
            html.H4("Error"),
            html.Pre(str(e))
        ])

    return "Select a tab to view dataset information"

# Add a callback to handle dataset card highlighting
@callback(
    [Output({'type': 'dataset-card-container', 'index': ALL}, 'style')],
    [Input('selected-dataset-store', 'data')],
    [State('datasets-store', 'data')]
)
def highlight_selected_dataset(selected_dataset, datasets):
    """Update dataset card styling based on selection."""
    if not datasets:
        return [[{'cursor': 'pointer'}]]
        
    styles = []
    for name in datasets.keys():
        if name == selected_dataset:
            styles.append({
                'cursor': 'pointer',
                'backgroundColor': '#e9ecef',
                'border': '2px solid #007bff'
            })
        else:
            styles.append({'cursor': 'pointer'})
            
    return [styles]

@callback(
    Output('download-selected-datasets', 'data', allow_duplicate=True),
    Input('download-button', 'n_clicks'),
    [State('selected-datasets-store', 'data'),
     State('datasets-store', 'data')],
    prevent_initial_call=True
)
def download_selected_datasets(n_clicks, selected_datasets, all_datasets):
    """Create zip file with selected datasets and metadata."""
    if not n_clicks or not selected_datasets:
        return dash.no_update

    try:
        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip.name, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for dataset_name in selected_datasets:
                    if dataset_name not in all_datasets:
                        continue
                    
                    dataset = all_datasets[dataset_name]
                    
                    # Convert data to string before encoding
                    df = pd.DataFrame(dataset['df'])
                    tsv_data = df.to_csv(sep='\t', index=False, encoding='utf-8')
                    zf.writestr(f"{dataset_name}.tsv", tsv_data.encode('utf-8'))
                    
                    # Convert metadata to string before encoding
                    metadata = {
                        k: v for k, v in dataset['metadata'].items()
                        if k != 'profile_report'
                    }
                    metadata_str = json.dumps(metadata, indent=2, ensure_ascii=False)
                    zf.writestr(
                        f"{dataset_name}_metadata.json",
                        metadata_str.encode('utf-8')
                    )

        # Read zip file and clean up
        with open(temp_zip.name, 'rb') as f:
            content = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(temp_zip.name)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"selected_datasets_{timestamp}.zip"
        
        return dict(
            content=content,
            base64=True,
            filename=filename,
            type='application/zip'
        )
        
    except Exception as e:
        print(f"Error creating zip file: {str(e)}")
        return dash.no_update
    
# Add a callback to manage the selected dataset
@callback(
    [Output('selected-dataset-store', 'data'),
     Output('chat-history', 'children', allow_duplicate=True),
     Output('chat-store', 'data', allow_duplicate=True),
     Output('dataset-tabs', 'active_tab', allow_duplicate=True)],
    [Input({'type': 'dataset-card', 'index': ALL}, 'n_clicks')],
    [State('datasets-store', 'data'),
     State('chat-store', 'data')],
    prevent_initial_call=True
)
def handle_dataset_selection(n_clicks, datasets, chat_store):
    """Handle dataset selection and process any pending plot requests."""
    if not any(n_clicks):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    try:
        # Get the triggered component's ID
        triggered_id = ctx.triggered[0]['prop_id']
        
        # Extract dataset name from the triggered ID
        dataset_name = None
        if '"index":"' in triggered_id:
            dataset_name = triggered_id.split('"index":"')[1].split('"')[0]
        
        if not dataset_name or not datasets or dataset_name not in datasets:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        # Initialize chat history from store
        chat_history = chat_store if chat_store else []
        
        # Add dataset selection message
        df = pd.DataFrame(datasets[dataset_name]['df'])
        info_message = (
            f"Selected dataset: {dataset_name}\n"
            f"Number of rows: {len(df)}\n"
            f"Columns: {', '.join(df.columns)}"
        )
        chat_history.append({
            'role': 'assistant',
            'content': info_message,
            'selected_dataset': dataset_name
        })
        
        # Default behavior: just update dataset selection
        return dataset_name, create_chat_elements_batch(chat_history), chat_history, 'tab-preview'
        
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Add new callback to update dataset count and memory usage
@callback(
    [Output('dataset-count', 'children'),
     Output('memory-usage', 'children')],
    [Input('datasets-store', 'data')]
)
def update_dataset_stats(datasets):
    """Update dataset count and memory usage statistics."""
    if not datasets:
        return "(0)", "Memory usage: 0 MB"
    
    # Count datasets
    dataset_count = len(datasets)
    
    # Calculate memory usage
    try:
        # Calculate total memory usage
        total_size = 0
        dataset_sizes = {}  # Store individual dataset sizes for debugging
        for name, dataset in datasets.items():
            # Convert dictionary back to DataFrame for accurate memory calculation
            df = pd.DataFrame(dataset['df'])
            # Use pandas' memory_usage which accounts for actual data size
            memory_usage = df.memory_usage(deep=True).sum()
            total_size += memory_usage
            dataset_sizes[name] = memory_usage / (1024 * 1024)  # Convert to MB
            
            # Add size of profile report if it exists
            if dataset.get('profile_report'):
                profile_size = len(dataset['profile_report'].encode('utf-8'))
                total_size += profile_size
                dataset_sizes[name] += profile_size / (1024 * 1024)
        
        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        total_memory_mb = system_memory.total / (1024 * 1024)
        available_memory_mb = system_memory.available / (1024 * 1024)
        
        memory_text = (
            f"Memory usage: {total_size_mb:.1f} MB "
            f"({(total_size_mb/total_memory_mb*100):.4f}% of system memory) | "
            f"System memory available: {available_memory_mb:.1f} MB"
        )
        
    except Exception as e:
        memory_text = f"Error calculating memory usage: {str(e)}"
    
    return f"({dataset_count})", memory_text

####################################
#
# Query Management Functions
#
####################################

@callback(
    [Output('chat-history', 'children', allow_duplicate=True),
     Output('chat-input', 'value', allow_duplicate=True),
     Output('chat-store', 'data', allow_duplicate=True),
     Output('dataset-tabs', 'active_tab', allow_duplicate=True),
     Output('viz-state-store', 'data', allow_duplicate=True),
     Output('chat-loading-output', 'children', allow_duplicate=True),
     Output('successful-queries-store', 'data', allow_duplicate=True),
     Output('datasets-store', 'data', allow_duplicate=True),
     Output('dataset-list', 'children', allow_duplicate=True)],
    [Input('chat-input', 'value'),
     Input('send-button', 'n_clicks')],
    [State('chat-store', 'data'),
     State('database-state', 'data'),
     State('database-structure-store', 'data'),
     State('successful-queries-store', 'data'),
     State('datasets-store', 'data'),
     State('selected-dataset-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def execute_confirmed_query(input_value, n_clicks, chat_history, database_state, database_structure_store, successful_queries, datasets, selected_dataset):
    """Process chat commands related to SQL query execution and dataset conversion."""
    if not input_value:
        return (dash.no_update,) * 9
    
    # Initialize stores safely
    successful_queries = successful_queries or {}
    chat_history = chat_history or []
    datasets = datasets or {}
    
    # Check for dataset information request
    dataset_query = re.search(r'tell\s+me\s+about\s+my\s+(dataset|datasets)\b', input_value.lower())
    if dataset_query:
        chat_history.append({'role': 'user', 'content': input_value})
        
        if not datasets:
            chat_history.append({
                'role': 'assistant',
                'content': "No datasets are currently loaded. Please upload a dataset first."
            })
            return (
                create_chat_elements_batch(chat_history),
                '',
                chat_history,
                dash.no_update,
                dash.no_update,
                "",
                successful_queries,
                dash.no_update,
                dash.no_update
            )
            
        # Show all datasets unless specifically asking about single dataset and one is selected
        show_all = dataset_query.group(1) == 'datasets' or not selected_dataset
        
        if show_all:
            # Generate overview of all datasets
            overview = ["Here are the currently loaded datasets:"]
            for name, data in datasets.items():
                df = pd.DataFrame(data['df'])
                metadata = data['metadata']
                dataset_info = f"""
{name}{'  (Selected)' if name == selected_dataset else ''}
- Source: {metadata['source']}
- Upload time: {metadata['upload_time']}
- Rows: {len(df)}
- Columns: {', '.join(df.columns)}
"""
                overview.append(dataset_info)
            
            if selected_dataset:
                overview.append(f"\nCurrently selected dataset: {selected_dataset}")
            else:
                overview.append("\nNo dataset is currently selected. Click a dataset name to select it.")
                
            chat_history.append({
                'role': 'assistant',
                'content': '\n'.join(overview)
            })
        else:
            # Show detailed info for selected dataset
            df = pd.DataFrame(datasets[selected_dataset]['df'])
            metadata = datasets[selected_dataset]['metadata']
            summary = f"""Dataset: {selected_dataset}

Source: {metadata['source']}
Upload time: {metadata['upload_time']}
Rows: {len(df)}
Columns: {', '.join(df.columns)}

Preview:
```
{df.head().to_string()}
```

Data Types:
{df.dtypes.to_string()}

Summary Statistics:
{df.describe().to_string()}
"""
            chat_history.append({
                'role': 'assistant',
                'content': summary
            })
            
        return (
            create_chat_elements_batch(chat_history),
            '',
            chat_history,
            dash.no_update,
            dash.no_update,
            "",
            successful_queries,
            dash.no_update,
            dash.no_update
        )
        
    # Check for dataset conversion request
    convert_match = re.search(r'convert\s+((query|lit_query)_\d{8}_\d{6}(?:_original|_alt\d+)?)\s+to\s+dataset', input_value.lower().strip())
    if convert_match:
        query_id = convert_match.group(1)
        print(f"\nProcessing dataset conversion request for query: {query_id}")
        
        # Add command to chat history
        chat_history.append({
            'role': 'user',
            'content': input_value
        })
        
        # Check if query exists in store
        if query_id not in successful_queries:
            print(f"Error: Query {query_id} not found in store")
            chat_history.append({
                'role': 'assistant',
                'content': f"❌ Query {query_id} not found in history. Please execute the query first."
            })
            return (
                create_chat_elements_batch(chat_history),  # chat-history
                '',                                        # chat-input
                chat_history,                              # chat-store
                dash.no_update,                           # dataset-tabs
                dash.no_update,                           # viz-state-store
                "",                                       # chat-loading-output
                successful_queries,                       # successful-queries-store
                dash.no_update,                           # datasets-store
                dash.no_update                            # dataset-list
            )
            
        try:
            print(f"Converting query {query_id} to dataset...")
            stored_query = successful_queries[query_id]
            
            if query_id.startswith('lit_query_'):
                # Handle literature query conversion
                print("Processing literature query conversion")
                df = pd.DataFrame(stored_query['dataframe'])  # Ensure we have a DataFrame
                metadata = {
                    'filename': f"{query_id}.csv",
                    'source': f"Literature query: {stored_query['query']}",
                    'threshold': stored_query['threshold'],
                    'execution_time': stored_query['metadata']['execution_time'],
                    'query_info': stored_query['metadata']['query_info'],
                    'summary': stored_query['metadata']['summary'],
                    'rows': len(df),
                    'columns': [df.index.name or 'index'] + list(df.columns)
                }
            else:
                # Handle SQL query conversion
                print("Processing SQL query conversion")
                df, metadata, _ = execute_sql_query(stored_query['sql'], database_state['path'])
                metadata = {
                    'filename': f"{query_id}.csv",
                    'source': f"Database query: {query_id}",
                    'database': database_state['path'],
                    'sql': stored_query['sql'],
                    'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df),
                    'columns': [df.index.name or 'index'] + list(df.columns)
                }
            
            print(f"Query executed successfully: {len(df)} rows retrieved")
            
            # Create dataset name (avoid duplicates)
            base_name = f"{query_id}"
            dataset_name = base_name
            counter = 1
            while dataset_name in datasets:
                dataset_name = f"{base_name}_{counter}"
                counter += 1
            print(f"Using dataset name: {dataset_name}")
            
            # Generate profile report
            try:
                print("Generating profile report...")
                profile = ProfileReport(
                    df,
                    minimal=True,
                    title=f"Profile Report for {dataset_name}",
                    html={'style': {'full_width': True}},
                    progress_bar=False,
                    correlations={'pearson': {'calculate': True}},
                    missing_diagrams={'matrix': False},
                    samples=None
                )
                profile_html = profile.to_html()
                print("Profile report generated successfully")
            except Exception as e:
                print(f"Warning: Profile report generation failed: {str(e)}")
                profile_html = None
            
            # Create dataset with special metadata
            datasets[dataset_name] = {
                'df': df.reset_index().to_dict('records'),
                'metadata': metadata,
                'profile_report': profile_html
            }
            print(f"Dataset '{dataset_name}' created successfully")
            
            # Add text search indexing for the new dataset
            try:
                text_searcher.update_dataset(dataset_name, df)
                print(f"Text search index updated for dataset '{dataset_name}'")
            except Exception as e:
                print(f"Warning: Failed to update text search index: {str(e)}")
            
            # Create success message
            chat_history.append({
                'role': 'assistant',
                'content': f"✅ Query results converted to dataset '{dataset_name}'\n\n"
                          f"- Rows: {len(df)}\n"
                          f"- Columns: {', '.join(df.columns)}\n"
                          f"- Source: Query {query_id}"
            })
            
            # Create updated dataset list
            dataset_list = [create_dataset_card(name, data) for name, data in datasets.items()]
            print("Dataset list updated")
            
            return (
                create_chat_elements_batch(chat_history),  # chat-history
                '',                                        # chat-input
                chat_history,                              # chat-store
                dash.no_update,                           # dataset-tabs
                dash.no_update,                           # viz-state-store
                "",                                       # chat-loading-output
                successful_queries,                       # successful-queries-store
                datasets,                                 # datasets-store
                dataset_list                              # dataset-list
            )
            
        except Exception as e:
            print(f"Error: Dataset conversion failed: {str(e)}")
            chat_history.append({
                'role': 'system',
                'content': f"❌ Error converting query to dataset: {str(e)}"
            })
            return (
                create_chat_elements_batch(chat_history),  # chat-history
                '',                                        # chat-input
                chat_history,                              # chat-store
                dash.no_update,                           # dataset-tabs
                dash.no_update,                           # viz-state-store
                "",                                       # chat-loading-output
                successful_queries,                       # successful-queries-store
                dash.no_update,                           # datasets-store
                dash.no_update                            # dataset-list
            )
    
    # Handle query execution
    input_lower = input_value.lower().strip()
    
    # Check execution command type
    is_simple_command = (
        input_lower.startswith(('execute', 'run', 'query')) and
        len(input_lower.split()) == 1 and
        any(input_lower.endswith(char) for char in ['.', '!'])
    )
    
    query_match = re.search(r'^execute\s+query_\d{8}_\d{6}(_original|_alt\d+)\b', input_lower)
    is_query_reference = bool(query_match)
    
    if not (is_simple_command or is_query_reference):
        return (dash.no_update,) * 9
        
    print("\nProcessing query execution command...")
    print(f"- Simple command: {is_simple_command}")
    print(f"- Query reference: {is_query_reference}")
    
    # Add command to chat history
    chat_history.append({
        'role': 'user',
        'content': input_value
    })
    
    # Find the query to execute
    sql_query = None
    found_id = None
    
    if is_query_reference:
        target_query_id = input_lower[8:].strip()  # Remove 'execute ' prefix
        print(f"\nLooking for specific query: {target_query_id}")
        # Search for specific query ID
        for msg in reversed(chat_history):
            if msg['role'] == 'assistant' and '```sql' in msg['content'].lower():
                content = msg['content']
                for match in re.finditer(r'```sql\s*(.*?)```', content, re.DOTALL):
                    block = match.group(1).strip()
                    id_match = re.search(r'--\s*Query ID:\s*((query_\d{8}_\d{6})(_original|_alt\d+))\b', block)
                    if id_match and id_match.group(1) == target_query_id:
                        found_id = target_query_id
                        sql_query = '\n'.join(
                            line for line in block.split('\n')
                            if not line.strip().startswith('-- Query ID:')
                        ).strip()
                        print(f"Found matching query with ID: {found_id}")
                        break
                if sql_query:
                    break
    else:
        print("\nLooking for most recent original query...")
        # Find most recent original query
        for msg in reversed(chat_history):
            if msg['role'] == 'assistant' and '```sql' in msg['content'].lower():
                content = msg['content']
                for match in re.finditer(r'```sql\s*(.*?)```', content, re.DOTALL):
                    block = match.group(1).strip()
                    id_match = re.search(r'--\s*Query ID:\s*((query_\d{8}_\d{6})(_original))\b', block)
                    if id_match:
                        found_id = id_match.group(1)
                        sql_query = '\n'.join(
                            line for line in block.split('\n')
                            if not line.strip().startswith('-- Query ID:')
                        ).strip()
                        print(f"Found original query with ID: {found_id}")
                        break
                if sql_query:
                    break
    
    if not sql_query:
        print("No matching SQL query found")
        chat_history.append({
            'role': 'assistant',
            'content': "No matching SQL query found in chat history."
        })
        return (
            create_chat_elements_batch(chat_history),  # chat-history
            '',                                        # chat-input
            chat_history,                              # chat-store
            dash.no_update,                           # dataset-tabs
            dash.no_update,                           # viz-state-store
            "",                                       # chat-loading-output
            successful_queries,                       # successful-queries-store
            dash.no_update,                           # datasets-store
            dash.no_update                            # dataset-list
        )

    try:
        print(f"\nExecuting SQL query...")
        print(f"Query:\n{sql_query}")
        
        # Execute the query
        results, metadata, preview = execute_sql_query(sql_query, database_state['path'])
        print(f"Query executed successfully: {metadata['rows']} rows returned")
        
        # Store successful query
        successful_queries[found_id] = store_successful_query(
            query_id=found_id,
            sql=sql_query,
            metadata=metadata
        )
        print(f"Query stored with ID: {found_id}")
        
        # Format response
        response = f"""Query executed successfully!

Results preview:

Query ID: {found_id}
{preview}

Total rows: {metadata['rows']}

Execution plan:
{metadata['execution_plan']}

Would you like to save these results as a dataset?"""
        
        chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        return (
            create_chat_elements_batch(chat_history),  # chat-history
            '',                                        # chat-input
            chat_history,                              # chat-store
            dash.no_update,                           # dataset-tabs
            dash.no_update,                           # viz-state-store
            "",                                       # chat-loading-output
            successful_queries,                       # successful-queries-store
            dash.no_update,                           # datasets-store
            dash.no_update                            # dataset-list
        )
        
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        chat_history.append({
            'role': 'system',
            'content': f"Query execution failed: {str(e)}"
        })
        return (
            create_chat_elements_batch(chat_history),  # chat-history
            '',                                        # chat-input
            chat_history,                              # chat-store
            dash.no_update,                           # dataset-tabs
            dash.no_update,                           # viz-state-store
            "",                                       # chat-loading-output
            successful_queries,                       # successful-queries-store
            dash.no_update,                           # datasets-store
            dash.no_update                            # dataset-list
        )

    
####################################
#
# Help Message Callback
#
####################################

# Add a new callback to handle the help button
@callback(
    [Output('chat-input', 'value', allow_duplicate=True),
     Output('chat-store', 'data', allow_duplicate=True),
     Output('chat-history', 'children', allow_duplicate=True)],
    Input('help-button', 'n_clicks'),
    [State('chat-store', 'data')],
    prevent_initial_call=True
)
def show_help(n_clicks, chat_history):
    """
    Show help message in chat when help button is clicked.
    """
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Initialize chat history if empty
    chat_history = chat_history or []
    
    # Add help request to chat history
    chat_history.append({
        'role': 'user',
        'content': "What can I do with this chat interface?"
    })
    
    # Add help message response
    chat_history.append({
        'role': 'assistant',
        'content': help_message
    })
    
    return '', chat_history, create_chat_elements_batch(chat_history)

####################################
#
# Enter Key Handler
#
####################################

# Enter key handler - Known limitation: Enter key does not trigger message send
# This is due to Dash callback chain restrictions. Users must use the Send button.
app.clientside_callback(
    """
    function(n_clicks, value) {
        if (!window.enterListenerAdded) {
            const textarea = document.getElementById('chat-input');
            if (textarea) {
                textarea.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        const sendButton = document.getElementById('send-button');
                        if (sendButton) {
                            sendButton.click();
                        }
                    }
                });
                window.enterListenerAdded = true;
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-input', 'n_clicks', allow_duplicate=True),
    [Input('chat-input', 'n_clicks'),
     State('chat-input', 'value')],
    prevent_initial_call='initial_duplicate'
)

####################################
#
# Plot Management Functions
#
####################################

def is_plot_request(message: str) -> bool:
    """Check if the message is requesting a plot."""

    plot_keywords = ['plot', 'graph', 'visualize', 'show', 'display']
    message = message.lower()
    return any(keyword in message for keyword in plot_keywords)

def is_dataset_query(message: str) -> bool:
    """Check if the message is asking about a dataset."""
    query_patterns = [
        r'tell me about',
        r'describe',
        r'what is',
        r'explain',
        r'show me',
        r'information about'
    ]
    message = message.lower()
    return any(pattern in message for pattern in query_patterns)

class VisualizationType:
    """Base class for visualization types with standard interface."""
    def __init__(self):
        self.required_params = set()
        self.optional_params = set()
        
    def validate_params(self, params: dict, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate parameters against requirements and dataframe."""
        missing = self.required_params - set(params.keys())
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, ""
        
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract parameters from message. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement extract_params")
        
    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create the visualization figure. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_figure")

class BubblePlot(VisualizationType):
    def __init__(self):
        super().__init__()
        self.required_params = {'x_column', 'y_column'}
        self.optional_params = {'size', 'color'}
    
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract plot parameters supporting both x/y and vs syntaxes."""
        params = {}
        
        # Try x=, y= syntax first
        x_match = re.search(r'x=(\w+)', message)
        y_match = re.search(r'y=(\w+)', message)
        
        if x_match and y_match:
            x_col = x_match.group(1)
            y_col = y_match.group(1)
        else:
            # Try vs/versus/against syntax
            vs_match = re.search(r'plot\s+(\w+)\s+(?:vs|versus|against)\s+(\w+)', message)
            if not vs_match:
                return {}, "Could not parse plot parameters"
            y_col = vs_match.group(1)
            x_col = vs_match.group(2)
        
        if x_col not in df.columns or y_col not in df.columns:
            return {}, f"Column not found: {x_col if x_col not in df.columns else y_col}"
            
        params['x_column'] = x_col
        params['y_column'] = y_col
        
        # Handle optional parameters
        for param in ['color', 'size']:
            match = re.search(rf'{param}=(\w+)', message)  # Use raw string with f-string
            if match:
                col = match.group(1)
                if col not in df.columns:
                    return {}, f"{param.capitalize()} column not found: {col}"
                params[param] = col
                
        return params, None
    
    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create bubble plot figure."""
        try:
            # Process size parameter
            if params.get('size'):
                if params['size'] in df.columns:
                    # Column-based size
                    size_values = df[params['size']]
                    marker_size = 10 + 40 * (size_values - size_values.min()) / (size_values.max() - size_values.min())
                else:
                    # Static size - try to convert to number
                    try:
                        marker_size = float(params['size'])
                    except (ValueError, TypeError):
                        marker_size = 20  # Default if conversion fails
            else:
                marker_size = 20
            
            # Process color parameter
            color_values = None
            color_discrete = False
            colormap = 'viridis'  # Default colormap for numeric data
            
            if params.get('color'):
                if params['color'] in df.columns:
                    color_values = df[params['color']]
                    if pd.api.types.is_numeric_dtype(color_values):
                        color_discrete = False
                    else:
                        color_discrete = True
                        unique_values = color_values.nunique()
                        color_sequence = generate_colors(unique_values)
                else:
                    # Check if it's a valid color specification
                    if is_valid_color(params['color']):
                        color_values = params['color']
                    else:
                        color_values = 'blue'  # Default color
            
            # Create the plot
            fig = px.scatter(
                df,
                x=params['x_column'],
                y=params['y_column'],
                size=marker_size if isinstance(marker_size, pd.Series) else None,
                color=color_values if isinstance(color_values, pd.Series) else None,
                color_continuous_scale=None if color_discrete else colormap,
                color_discrete_sequence=color_sequence if color_discrete else None,
                title=f'Bubble Plot: {params["y_column"]} vs {params["x_column"]}'
            )
            
            # Update markers for static values
            if not isinstance(marker_size, pd.Series):
                fig.update_traces(marker=dict(size=marker_size))
            if not isinstance(color_values, pd.Series) and color_values is not None:
                fig.update_traces(marker=dict(color=color_values))
            
            # Improve layout
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    title=params['x_column']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    title=params['y_column']
                ),
                # Add configuration for better interactivity
                dragmode='pan',  # Enable panning by default
                modebar=dict(
                    orientation='h',  # Horizontal orientation
                    bgcolor='rgba(255,255,255,0.7)',
                    color='rgb(128,128,128)',
                    activecolor='rgb(50,50,50)'
                ),
                # Disable selection by default
                selectdirection=None,
                clickmode='event',  # Changed from 'event+select'
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating bubble plot: {str(e)}")

class HeatmapPlot(VisualizationType):
    """Heatmap plot implementation using existing logic."""
    def __init__(self):
        super().__init__()
        self.required_params = {'columns'}
        self.optional_params = {'rows', 'standardize', 'cluster', 'colormap', 'transpose', 'fcol'}  # Added fcol
    
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract heatmap parameters with regex support for rows/columns."""
        # Initialize params with default values (no default colormap)
        params = {
            'transpose': False,
            'standardize': None,
            'cluster': None,
            'colormap': None,
            'fcol': None  # Added fcol initialization
        }
        
        # Define valid parameter names
        valid_params = {'columns', 'rows', 'standardize', 'cluster', 'colormap', 'transpose', 'fcol'}  # Added fcol
        
        # Find all parameter assignments in the message
        param_matches = re.finditer(r'(\w+)=([^\s]+)', message)
        unknown_params = []
        
        for match in param_matches:
            param_name = match.group(1)
            if param_name not in valid_params:
                unknown_params.append(param_name)
        
        if unknown_params:
            return {}, f"Unknown parameter(s): {', '.join(unknown_params)}. Valid parameters are: {', '.join(sorted(valid_params))}"
        
        # Extract columns parameter with better error handling
        if 'columns=' in message:
            cols_match = re.search(r'columns=\[(.*?)\]', message)
            cols_regex_match = re.search(r'columns=(\S+)', message)
            
            if not (cols_match or cols_regex_match):
                return {}, "Malformed columns parameter. Use format: columns=[col1,col2,...] or columns=regex_pattern"
            
            if cols_match:
                cols = [c.strip() for c in cols_match.group(1).split(',')]
                if not cols:
                    return {}, "Empty column list provided"
                invalid_cols = [col for col in cols if col not in df.columns]
                if invalid_cols:
                    return {}, f"Column(s) not found in dataset: {', '.join(invalid_cols)}"
                params['columns'] = cols
            elif cols_regex_match:
                pattern = cols_regex_match.group(1)
                try:
                    regex = re.compile(pattern)
                    matching_cols = [col for col in df.columns if regex.search(col)]
                    if not matching_cols:
                        return {}, f"Column regex '{pattern}' matched no columns in dataset"
                    params['columns'] = matching_cols
                except re.error:
                    return {}, f"Invalid regex pattern for columns: '{pattern}'"
        else:
            params['columns'] = list(df.columns)  # Default to all columns
        
        # Extract rows parameter with regex and fcol support
        if 'rows=' in message:
            rows_match = re.search(r'rows=\[(.*?)\]', message)
            rows_regex_match = re.search(r'rows=(\S+)', message)
            
            if not (rows_match or rows_regex_match):
                return {}, "Malformed rows parameter. Use format: rows=[row1,row2,...] or rows=regex_pattern fcol=column_name"
            
            if rows_match:
                rows = [r.strip() for r in rows_match.group(1).split(',')]
                if not rows:
                    return {}, "Empty row list provided"
                invalid_rows = [row for row in rows if row not in df.columns]
                if invalid_rows:
                    return {}, f"Row(s) not found in dataset: {', '.join(invalid_rows)}"
                params['rows'] = rows
            else:  # Using regex pattern
                fcol_match = re.search(r'fcol=(\w+)', message)
                if not fcol_match:
                    return {}, "When using regex for rows, must specify fcol=column_name to filter on"
                
                fcol = fcol_match.group(1)
                if fcol not in df.columns:
                    return {}, f"Filter column '{fcol}' not found in dataset"
                
                pattern = rows_regex_match.group(1)
                try:
                    regex = re.compile(pattern)
                    # Filter the dataframe based on the regex pattern in fcol
                    filtered_indices = df[df[fcol].astype(str).str.match(regex)].index.tolist()
                    if not filtered_indices:
                        return {}, f"Row regex '{pattern}' matched no values in column '{fcol}'"
                    # Store the filtered indices and filter column instead of the DataFrame
                    params['row_indices'] = filtered_indices
                    params['fcol'] = fcol
                except re.error:
                    return {}, f"Invalid regex pattern for rows: '{pattern}'"
        
        # Standardize parameter - strict validation
        std_match = re.search(r'standardize=(\w+)', message)
        if std_match:
            std_value = std_match.group(1).lower()
            if std_value not in ['rows', 'columns']:
                return {}, f"Invalid value for standardize: '{std_value}'. Must be 'rows' or 'columns'"
            params['standardize'] = std_value
        
        # Cluster parameter - strict validation
        cluster_match = re.search(r'cluster=(\w+)', message)
        if cluster_match:
            cluster_value = cluster_match.group(1).lower()
            if cluster_value not in ['rows', 'columns', 'both']:
                return {}, f"Invalid value for cluster: '{cluster_value}'. Must be 'rows', 'columns', or 'both'"
            params['cluster'] = cluster_value
        
        # Colormap parameter - strict validation with sorted options
        colormap_match = re.search(r'colormap=(\w+)', message)
        if colormap_match:
            colormap = colormap_match.group(1)
            valid_colormaps = sorted(px.colors.named_colorscales())
            if colormap not in valid_colormaps:
                return {}, f"Invalid colormap: '{colormap}'. Valid options (alphabetically):\n{', '.join(valid_colormaps)}"
            params['colormap'] = colormap
        
        # Transpose parameter - strict validation
        transpose_match = re.search(r'transpose=(\w+)', message)
        if transpose_match:
            transpose_value = transpose_match.group(1).lower()
            if transpose_value not in ['true', 'false']:
                return {}, f"Invalid value for transpose: '{transpose_value}'. Must be 'true' or 'false'"
            params['transpose'] = transpose_value == 'true'
        elif 'transpose' in message.lower():
            params['transpose'] = True
        
        return params, None

    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create heatmap figure."""
        try:
            # Preprocess the data
            data, clustering_info = preprocess_heatmap_data(df, params)
            
            # Determine if we should center the colorscale
            center_scale = params.get('standardize') is not None
            if center_scale:
                max_abs = max(abs(data.values.min()), abs(data.values.max()))
                zmin, zmax = -max_abs, max_abs
                colorscale = 'RdBu_r'  # Red-Blue diverging colormap for centered data
            else:
                zmin, zmax = data.values.min(), data.values.max()
                colorscale = params.get('colormap', 'viridis')
            
            # Create the figure
            fig = go.Figure()
            
            # Add heatmap trace
            heatmap = go.Heatmap(
                z=data.values,
                x=clustering_info['col_labels'],
                y=clustering_info['row_labels'],
                colorscale=colorscale,
                zmid=0 if center_scale else None,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title='Standardized Value' if params.get('standardize') else 'Value',
                    titleside='right'
                ),
                hoverongaps=False,
                hovertemplate=(
                    'Row: %{y}<br>' +
                    'Column: %{x}<br>' +
                    ('Standardized Value: %{z:.2f}<br>' if params.get('standardize') else 'Value: %{z:.2f}<br>') +
                    '<extra></extra>'
                )
            )
            fig.add_trace(heatmap)
            
            # Update layout
            title_parts = []
            if params.get('standardize'):
                title_parts.append(f"Standardized by {params['standardize']}")
            if params.get('cluster'):
                title_parts.append(f"Clustered by {params['cluster']}")
            if params.get('transpose'):
                title_parts.append("Transposed")
            
            title = "Heatmap" + (f" ({', '.join(title_parts)})" if title_parts else "")
            
            fig.update_layout(
                title=title,
                xaxis=dict(
                    title='',
                    tickangle=45,
                    showgrid=False,
                    ticktext=clustering_info['col_labels'],
                    tickvals=list(range(len(clustering_info['col_labels'])))
                ),
                yaxis=dict(
                    title='',
                    showgrid=False,
                    side='left',
                    ticktext=clustering_info['row_labels'],
                    tickvals=list(range(len(clustering_info['row_labels'])))
                ),
                margin=dict(
                    l=100,
                    r=50,
                    t=50,
                    b=100
                ),
                # Add configuration for better interactivity
                dragmode='pan',  # Enable panning by default
                modebar=dict(
                    orientation='h',  # Horizontal orientation
                    bgcolor='rgba(255,255,255,0.7)',
                    color='rgb(128,128,128)',
                    activecolor='rgb(50,50,50)'
                ),
                # Disable selection by default
                selectdirection=None,
                clickmode='event',  # Changed from 'event+select'
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating heatmap: {str(e)}")

class GeoMap(VisualizationType):
    """Geographic map visualization implementation using existing logic."""
    def __init__(self):
        super().__init__()
        self.required_params = {'latitude', 'longitude'}
        self.optional_params = {'size', 'color'}
    
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract map parameters."""
        params = {}
        
        # Required parameters
        lat_match = re.search(r'latitude=(\w+)', message)
        lon_match = re.search(r'longitude=(\w+)', message)
        
        if not (lat_match and lon_match):
            return {}, "Map requires both latitude and longitude parameters"
        
        lat_col = lat_match.group(1)
        lon_col = lon_match.group(1)
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return {}, f"Column not found: {lat_col if lat_col not in df.columns else lon_col}"
        
        params['latitude'] = lat_col
        params['longitude'] = lon_col
        
        # Optional parameters
        for param in ['color', 'size']:
            match = re.search(rf'{param}=(\w+)', message)  # Use raw string with f-string
            if match:
                col = match.group(1)
                if col not in df.columns:
                    return {}, f"{param.capitalize()} column not found: {col}"
                params[param] = col
        
        return params, None

    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create geographic map visualization.
        
        Args:
            params: Dictionary containing:
                - latitude: Column name for latitude values
                - longitude: Column name for longitude values
                - size: Optional column name or value for point sizes
                - color: Optional column name or value for point colors
            df: DataFrame containing the data
            
        Returns:
            Plotly figure object
            
        Raises:
            Exception: If required parameters are missing or invalid
        """
        try:
            # Validate required columns
            for param in ['latitude', 'longitude']:
                if param not in params:
                    raise Exception(f"Missing required parameter: {param}")
                if params[param] not in df.columns:
                    raise Exception(f"Column not found: {params[param]}")
                    
            # Extract coordinates
            lat = df[params['latitude']]
            lon = df[params['longitude']]
            
            # Handle invalid coordinates
            valid_coords = (
                lat.between(-90, 90) & 
                lon.between(-180, 180) & 
                lat.notna() & 
                lon.notna()
            )
            
            if not valid_coords.any():
                raise Exception("No valid coordinates found in data")
                
            if (~valid_coords).any():
                print(f"Warning: {(~valid_coords).sum()} invalid coordinates removed")
                df = df[valid_coords].copy()
                lat = lat[valid_coords]
                lon = lon[valid_coords]
            
            # Process size parameter
            if params.get('size'):
                if params['size'] in df.columns:
                    # Column-based size
                    size_values = df[params['size']]
                    if not pd.to_numeric(size_values, errors='coerce').notna().all():
                        raise Exception(f"Size column '{params['size']}' must contain numeric values")
                    # Scale sizes for better visualization
                    size_min, size_max = 10, 50  # Reasonable size range
                    marker_size = size_min + (size_max - size_min) * (
                        (size_values - size_values.min()) / 
                        (size_values.max() - size_values.min())
                    )
                else:
                    # Static size - try to convert to number
                    try:
                        marker_size = float(params['size'])
                    except (ValueError, TypeError):
                        marker_size = 15  # Default if conversion fails
            else:
                marker_size = 15  # Default size
            
            # Process color parameter
            if params.get('color'):
                if params['color'] in df.columns:
                    color_values = df[params['color']]
                    if pd.api.types.is_numeric_dtype(color_values):
                        # Numeric color scale
                        marker_color = color_values
                        colorscale = 'viridis'
                    else:
                        # Categorical colors
                        unique_values = color_values.unique()
                        color_sequence = generate_colors(len(unique_values))
                        color_map = dict(zip(unique_values, color_sequence))
                        marker_color = [color_map[val] for val in color_values]
                        colorscale = None
                else:
                    # Static color
                    marker_color = params['color'] if is_valid_color(params['color']) else 'blue'
                    colorscale = None
            else:
                marker_color = 'blue'
                colorscale = None
            
            # Create the map
            fig = go.Figure()
            
            # Add scatter mapbox trace
            scatter_kwargs = {
                'lat': lat,
                'lon': lon,
                'mode': 'markers',
                'marker': {
                    'size': marker_size,
                    'color': marker_color,
                },
                'text': [
                    f"Latitude: {lat:.4f}<br>"
                    f"Longitude: {lon:.4f}<br>"
                    + (f"{params['size']}: {size}<br>" if params.get('size') in df.columns else "")
                    + (f"{params['color']}: {color}<br>" if params.get('color') in df.columns else "")
                    for lat, lon, size, color in zip(
                        lat, lon,
                        df[params['size']] if params.get('size') in df.columns else [None] * len(lat),
                        df[params['color']] if params.get('color') in df.columns else [None] * len(lat)
                    )
                ],
                'hoverinfo': 'text'
            }
            
            # Add colorscale if using numeric or categorical colors
            if params.get('color') in df.columns:
                if pd.api.types.is_numeric_dtype(df[params['color']]):
                    scatter_kwargs['marker']['colorscale'] = colorscale
                    scatter_kwargs['marker']['colorbar'] = dict(
                        title=params['color'],
                        titleside='right',
                        thickness=20,
                        len=0.9,
                        x=0.02,  # Move colorbar to the left side
                        xpad=0
                    )
                    scatter_kwargs['marker']['showscale'] = True
                else:
                    # For categorical data, create a discrete color scale
                    unique_values = sorted(df[params['color']].unique())
                    color_sequence = generate_colors(len(unique_values))
                    color_map = dict(zip(unique_values, color_sequence))
                    scatter_kwargs['marker']['color'] = [color_map[val] for val in df[params['color']]]
                    # Add a legend instead of colorbar for categorical data
                    scatter_kwargs['showlegend'] = True
                    scatter_kwargs['name'] = params['color']  # This will show in the legend
                    # Create separate traces for legend entries
                    for val, color in color_map.items():
                        fig.add_trace(go.Scattermapbox(
                            lat=[None],
                            lon=[None],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=str(val),
                            showlegend=True,
                            hoverinfo='skip'
                        ))
                
            fig.add_trace(go.Scattermapbox(**scatter_kwargs))
            
            # Update layout for mapbox
            center_lat = lat.mean()
            center_lon = lon.mean()
            
            # Calculate zoom based on coordinate spread
            lat_range = lat.max() - lat.min()
            lon_range = lon.max() - lon.min()
            zoom = 12 - max(lat_range, lon_range)  # Adjust zoom based on spread
            zoom = max(1, min(20, zoom))  # Ensure zoom is within valid range
            
            fig.update_layout(
                mapbox=dict(
                    style='carto-positron',  # Light, clean map style
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(
                    text="Geographic Distribution",
                    x=0.5,
                    xanchor='center'
                ),
                # Add configuration for better interactivity
                dragmode='pan',  # Enable panning by default
                modebar=dict(
                    orientation='h',  # Horizontal orientation
                    bgcolor='rgba(255,255,255,0.7)',
                    color='rgb(128,128,128)',
                    activecolor='rgb(50,50,50)'
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor='rgba(255,255,255,0.7)'
                ),
                # Disable selection by default
                selectdirection=None
            )
            
            # Add configuration for enhanced interactivity
            fig.update_layout(
                clickmode='event',  # Changed from 'event+select'
                hovermode='closest'
            )
            
            # Enable all interactions including scroll zoom
            fig.update_layout(
                mapbox_style="carto-positron",  # Ensure consistent style
                mapbox=dict(
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom,
                    pitch=0,  # Start with top-down view
                    bearing=0  # Start facing north
                )
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating map: {str(e)}")

class VisualizationHandler:
    def __init__(self):
        self.viz_types = {
            'bubble': BubblePlot(),
            'heatmap': HeatmapPlot(),
            'map': GeoMap()
        }
    
    def detect_type(self, message: str) -> str:
        """Detect the type of visualization requested."""
        message = message.lower().strip()
        
        if message.startswith('map') and 'latitude=' in message and 'longitude=' in message:
            return 'map'
        
        if message.startswith('plot'):
            if ('x=' in message and 'y=' in message) or \
               any(x in message for x in ['vs', 'versus', 'against']):
                return 'bubble'
        
        if message.startswith('heatmap'):
            return 'heatmap'
            
        return None
        
    def process_request(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Process visualization request."""
        viz_type = self.detect_type(message)
        if not viz_type:
            return None, "Could not determine visualization type"
            
        viz_processor = self.viz_types[viz_type]
        params, error = viz_processor.extract_params(message, df)
        if error:
            return None, error
            
        is_valid, error = viz_processor.validate_params(params, df)
        if not is_valid:
            return None, error
            
        return {
            'type': viz_type,
            'params': params,
            'df': df.to_dict('records')  # Include dataframe in state
        }, None

def generate_colors(n):
    """Generate n visually distinct colors.
    
    Args:
        n: Number of colors needed
        
    Returns:
        List of colors in RGB format suitable for plotly
    """
    colors = []
    for i in range(n):
        h = i / n  # Spread hues evenly
        s = 0.7    # Moderate saturation
        v = 0.9    # High value for visibility
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        
        # Convert to 0-255 range and then to plotly rgb string
        colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
    
    return colors

def preprocess_heatmap_data(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Prepare data for heatmap visualization."""
    # First filter the data if row_indices are provided
    if 'row_indices' in params:
        df = df.loc[params['row_indices']]
    
    # Rest of the function remains the same...
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
    from scipy.spatial.distance import pdist
    
    # Make a copy to avoid modifying original data
    if params['columns'] is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols].copy()
    else:
        data = df[params['columns']].copy()
    
    # Handle missing values more robustly
    # First check if we have any missing values
    if data.isna().any().any():
        # For each column, fill NaN with column median instead of mean
        # (median is more robust to outliers)
        for col in data.columns:
            data[col] = data[col].fillna(data[col].median())
            
        # If any NaNs remain (e.g., if a column is all NaN),
        # fill with 0 to ensure we can proceed
        data = data.fillna(0)
        
        print(f"Warning: Missing values were present and filled with median values")
    
    # Apply standardization before clustering
    if params['standardize'] == 'rows':
        # Check for zero variance rows to avoid division by zero
        row_std = data.std(axis=1)
        zero_var_rows = row_std == 0
        if zero_var_rows.any():
            print(f"Warning: {zero_var_rows.sum()} rows had zero variance and were not standardized")
            # Only standardize non-zero variance rows
            data.loc[~zero_var_rows] = ((data.loc[~zero_var_rows].T - data.loc[~zero_var_rows].mean(axis=1)) / 
                                      data.loc[~zero_var_rows].std(axis=1)).T
    elif params['standardize'] == 'columns':
        # Check for zero variance columns to avoid division by zero
        col_std = data.std()
        zero_var_cols = col_std == 0
        if zero_var_cols.any():
            print(f"Warning: {zero_var_cols.sum()} columns had zero variance and were not standardized")
            # Only standardize non-zero variance columns
            data.loc[:, ~zero_var_cols] = ((data.loc[:, ~zero_var_cols] - data.loc[:, ~zero_var_cols].mean()) / 
                                         data.loc[:, ~zero_var_cols].std())
    
    clustering_info = {
        'row_linkage': None,
        'col_linkage': None,
        'row_order': None,
        'col_order': None,
        'row_labels': data.index.tolist(),
        'col_labels': data.columns.tolist()
    }
    
    # Apply clustering if requested
    if params['cluster']:
        if params['cluster'] in ['rows', 'both']:
            # Calculate distance matrix for rows
            try:
                row_dist = pdist(data, metric='euclidean')
                # Perform hierarchical clustering
                row_linkage = linkage(row_dist, method='ward')
                # Apply optimal leaf ordering
                row_linkage = optimal_leaf_ordering(row_linkage, row_dist)
                row_order = leaves_list(row_linkage)
                data = data.iloc[row_order]
                clustering_info['row_linkage'] = row_linkage
                clustering_info['row_order'] = row_order
            except Exception as e:
                print(f"Warning: Row clustering failed: {str(e)}")
            
        if params['cluster'] in ['columns', 'both']:
            try:
                # Calculate distance matrix for columns
                col_dist = pdist(data.T, metric='euclidean')
                # Perform hierarchical clustering
                col_linkage = linkage(col_dist, method='ward')
                # Apply optimal leaf ordering
                col_linkage = optimal_leaf_ordering(col_linkage, col_dist)
                col_order = leaves_list(col_linkage)
                data = data.iloc[:, col_order]
                clustering_info['col_linkage'] = col_linkage
                clustering_info['col_order'] = col_order
            except Exception as e:
                print(f"Warning: Column clustering failed: {str(e)}")
    
    # Apply transpose if requested (after clustering)
    if params['transpose']:
        data = data.T
        clustering_info['row_linkage'], clustering_info['col_linkage'] = (
            clustering_info['col_linkage'], clustering_info['row_linkage'])
        clustering_info['row_order'], clustering_info['col_order'] = (
            clustering_info['col_order'], clustering_info['row_order'])
        clustering_info['row_labels'], clustering_info['col_labels'] = (
            clustering_info['col_labels'], clustering_info['row_labels'])
    
    return data, clustering_info

def is_valid_color(color_str: str) -> bool:
    """Check if a string is a valid color specification."""
    # Common CSS color names
    valid_colors =  {'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
    'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
    'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
    'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson',
    'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
    'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
    'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
    'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen',
    'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray',
    'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
    'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
    'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen',
    'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
    'lightslategray', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
    'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
    'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
    'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
    'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue',
    'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
    'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue',
    'slateblue', 'slategray', 'snow', 'springgreen', 'steelblue', 'tan', 'teal',
    'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
    'yellow', 'yellowgreen'}
    
    return (
        color_str.lower() in valid_colors or  # Named colors
        bool(re.match(r'^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', color_str)) or  # Hex colors
        bool(re.match(r'^rgb\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)$', color_str))  # RGB colors
    )

# Move all callbacks before main
@callback(
    [Output('viz-container', 'children'),
     Output('viz-debug', 'children')],
    Input('viz-state-store', 'data')
)
def update_visualization(viz_state):
    """Update visualization based on state store."""
    if not viz_state or not viz_state.get('type'):
        return html.Div("No visualization selected"), "Waiting for plot data..."
        
    try:
        # Convert the dictionary records back to DataFrame
        df = pd.DataFrame(viz_state['df'])
        
        # Get the visualization processor
        viz_handler = VisualizationHandler()
        viz_type = viz_state['type']
        
        if viz_type not in viz_handler.viz_types:
            return html.Div(f"Unknown visualization type: {viz_type}"), "Error: Invalid type"
        
        viz_processor = viz_handler.viz_types[viz_type]
        
        # Create the visualization
        figure = viz_processor.create_figure(viz_state['params'], df)
        
        return html.Div([
            dcc.Graph(
                figure=figure,
                style={'height': '700px'},
                config={
                    'scrollZoom': True,  # Enable scroll zoom
                    'modeBarButtonsToAdd': [
                        'drawclosedpath',  # Area selection
                        'eraseshape',      # Remove selections
                        'select2d',        # Box selection
                        'lasso2d',         # Lasso selection
                        'zoomIn2d',        # Zoom in button
                        'zoomOut2d',       # Zoom out button
                        'autoScale2d'      # Auto-scale
                    ],
                    'displaylogo': False,  # Remove plotly logo
                    'responsive': True,    # Make plot responsive
                    'showAxisDragHandles': True,  # Show axis drag handles
                    'showAxisRangeEntryBoxes': True,  # Show range entry boxes
                    'toImageButtonOptions': {
                        'format': 'png',  # Export format
                        'filename': 'plot',
                        'height': 700,
                        'width': 1200,
                        'scale': 2  # Increase export resolution
                    },
                    'doubleClick': 'reset+autosize',  # Double click to reset view
                    'displayModeBar': True,  # Always show mode bar
                    'modeBarButtonsToRemove': []  # Keep all default buttons
                }
            )
        ]), f"Created {viz_type} visualization"
        
    except Exception as e:
        print(f"Visualization error: {str(e)}")  # Add console logging
        return html.Div(f"Error creating visualization: {str(e)}"), str(e)

# Add new callback for refresh functionality
@callback(
    [Output('database-path-dropdown', 'options'),
     Output('database-path-dropdown', 'value')],  # Reset value on refresh
    Input('refresh-database-list', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_database_list(n_clicks):
    """Refresh the list of available databases."""
    return get_database_files(), None  # Reset dropdown selection

def get_database_structure(db_path: str) -> Dict[str, Any]:
    """Get structure of SQLite database including tables, columns, and relationships."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        structure = {}
        
        for table in tables:
            table_name = table[0]
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [{
                'name': str(col[1]),  # Ensure column name is string
                'type': str(col[2]),  # Ensure type is string
                'notnull': bool(col[3]),
                'pk': bool(col[4]),
                'default': col[5]
            } for col in cursor.fetchall()]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            
            # Get foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table_name});")
            foreign_keys = [{
                'from': str(fk[3]),  # Ensure key names are strings
                'to': str(fk[4]),
                'table': str(fk[2])
            } for fk in cursor.fetchall()]
            
            structure[table_name] = {
                'columns': columns,
                'row_count': row_count,
                'foreign_keys': foreign_keys
            }
            
        conn.close()
        print(f"\n=== Database Structure Debug ===")
        print(f"First table structure: {next(iter(structure.items()))}")
        return structure
        
    except Exception as e:
        print(f"Error getting database structure: {str(e)}")
        print(traceback.format_exc())
        return {}

# Update the database connection callback to use this
@callback(
    [Output('database-state', 'data'),
     Output('database-structure-store', 'data'),
     Output('database-connection-status', 'children')],
    [Input('database-connect-button', 'n_clicks')],
    [State('database-path-dropdown', 'value'),
     State('database-state', 'data')],
    prevent_initial_call=True
)
def connect_database(n_clicks, db_path, current_state):
    """Handle database connection attempts."""
    if not n_clicks or not db_path:
        return (
            {'connected': False, 'path': None},
            None,
            html.Div('Please select a database', style={'color': 'red'})
        )
    
    try:
        # Get detailed database structure
        print(f"\n=== Connecting to Database ===")
        print(f"Path: {db_path}")
        
        structure = get_database_structure(db_path)
        if not structure:
            raise Exception("Could not read database structure")
            
        # Initialize text search
        print(f"Indexing database: {db_path}")
        global text_searcher_db
        text_searcher_db = DatabaseTextSearch()
        text_searcher_db.index_database(db_path)
        print(f"Finished indexing database: {db_path}")
        
        # Store structure in global variable for access
        global current_database_structure
        current_database_structure = structure
        
        state = {
            'connected': True,
            'path': db_path,
            'has_text_search': True,
            'structure': structure  # Include structure in state
        }
        
        print(f"Database connected successfully with {len(structure)} tables")
        return (
            state,
            structure,
            html.Div('Connected successfully', style={'color': 'green'})
        )
        
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        print(traceback.format_exc())
        return (
            {'connected': False, 'path': None, 'has_text_search': False},
            None,
            html.Div(f'Connection failed: {str(e)}', style={'color': 'red'})
        )

# Create a global text searcher instance
_text_searcher = None

@callback(
    [Output('database-summary', 'children'),
     Output('database-erd', 'children'),
     Output('database-erd', 'style')],
    [Input('database-structure-store', 'data'),
     Input('database-view-tabs', 'active_tab')],
    [State('database-erd', 'children')],  # Add state to track existing ERD
    prevent_initial_call='initial_duplicate'
)
def update_database_views(structure_data, active_tab, existing_erd):
    """Update both table summary and ERD visualization."""
    # Base style for ERD container
    base_style = {
        'width': '100%',
        'height': '600px',
        'overflow': 'auto',
        'position': 'relative',
        'backgroundColor': 'white',
        'border': '1px solid #e0e0e0',
        'borderRadius': '4px',
        'padding': '15px'
    }
    
    # Update visibility based on active tab
    if active_tab == 'tab-erd':
        base_style.update({
            'display': 'block',
            'opacity': 1,
            'visibility': 'visible'
        })
    else:
        base_style.update({
            'display': 'none',
            'opacity': 0,
            'visibility': 'hidden'
        })
    
    if not structure_data:
        return "No database connected", None, base_style
    
    # Generate table summary
    table_rows = [
        "| Table | Rows | Columns | Foreign Keys |",
        "|-------|------|---------|--------------|"
    ]
    
    for table, info in structure_data.items():
        columns = len(info['columns'])
        rows = info['row_count']
        fks = len(info['foreign_keys'])
        table_rows.append(f"| {table} | {rows} | {columns} | {fks} |")
    
    summary = dcc.Markdown('\n'.join(table_rows))
    
    # If we're just switching tabs and already have an ERD, reuse it
    if active_tab == 'tab-erd' and existing_erd is not None:
        return summary, existing_erd, base_style
        
    try:
        # Only generate new ERD if we don't have one or structure changed
        if existing_erd is None or active_tab == 'tab-erd':
            print("\n=== Generating new ERD ===")
            # Generate and clean up ERD code
            mermaid_code = generate_mermaid_erd(structure_data)
            if mermaid_code.startswith('```mermaid\n'):
                mermaid_code = mermaid_code[len('```mermaid\n'):]
            if mermaid_code.endswith('```'):
                mermaid_code = mermaid_code[:-3]
            
            # Create ERD content
            erd = html.Div([
                Mermaid(
                    chart=mermaid_code,
                    config={
                        "securityLevel": "loose",
                        "theme": "default",
                        "themeVariables": {
                            "background": "#ffffff",
                            "primaryColor": "#e0e0e0",
                            "primaryBorderColor": "#555555",
                            "lineColor": "#555555",
                            "textColor": "#000000",
                            "entityBkgColor": "#ffffff",
                            "entityBorder": "#555555",
                            "labelBackground": "#ffffff",
                            "labelBorder": "#555555",
                            "nodeBkg": "#ffffff",
                            "classText": "#000000",
                            "mainBkg": "#ffffff",
                            "titleColor": "#000000",
                            "edgeLabelBackground": "#ffffff",
                            "clusterBkg": "#ffffff",
                            "defaultLinkColor": "#555555",
                            "tertiaryColor": "#ffffff",
                            "noteTextColor": "#000000",
                            "noteBkgColor": "#ffffff",
                            "noteBorderColor": "#555555",
                            "erd": {
                                "entityFill": "#ffffff",
                                "entityBorder": "#333333",
                                "attributeFill": "#ffffff",
                                "attributeBorder": "#333333",
                                "labelColor": "#000000",
                                "labelBackground": "#ffffff"
                            }
                        },
                        "er": {
                            "layoutDirection": "TB",
                            "entityPadding": 15,
                            "useMaxWidth": True
                        },
                        "maxZoom": 4,
                        "minZoom": 0.2,
                        "zoomScale": 0.5,
                        "pan": True,
                        "zoom": True,
                        "controlIcons": True
                    }
                )
            ])
        else:
            erd = existing_erd
            
        return summary, erd, base_style
        
    except Exception as e:
        print(f"Error generating ERD: {str(e)}")
        error_display = html.Div([
            html.P(f"Error generating ERD: {str(e)}", style={'color': 'red'})
        ])
        return summary, error_display, base_style

####################################
#
# Weaviate Management Functions
#
####################################

@callback(
    [Output('weaviate-summary', 'children'),
     Output('weaviate-erd', 'children'),
     Output('weaviate-erd', 'style')],
    [Input('weaviate-state', 'data'),
     Input('weaviate-view-tabs', 'active_tab')],
    prevent_initial_call='initial_duplicate'
)
def update_weaviate_views(weaviate_state, active_tab):
    """Update both table summary and ERD visualization for Weaviate."""
    # Base style for ERD container
    base_style = {
        'width': '100%',
        'height': '600px',
        'overflow': 'auto',
        'position': 'relative',
        'backgroundColor': 'white',
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'padding': '10px'
    }
    
    # Update visibility based on active tab
    if active_tab == 'tab-weaviate-erd':
        base_style.update({
            'display': 'block',
            'opacity': 1,
            'visibility': 'visible'
        })
    else:
        base_style.update({
            'display': 'none',
            'opacity': 0,
            'visibility': 'hidden'
        })

    if not weaviate_state or weaviate_state['connection']['status'] != 'connected':
        return "No Weaviate connection", None, base_style

    try:
        from weaviate_integration import WeaviateConnection
        from weaviate_manager.database.inspector import DatabaseInspector
        
        # Get schema info using context-managed client
        weaviate = WeaviateConnection()
        with weaviate.get_client() as client:
            inspector = DatabaseInspector(client)
            schema_info = inspector.get_schema_info()

            if not schema_info:
                return "No collections exist in Weaviate", None, base_style

            # Create styled HTML content with container for width control
            html_content = [
                html.Div([
                    html.H3("Collections Overview", 
                           className="mb-4",
                           style={'color': '#2c3e50', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Collection", style={'backgroundColor': '#e9ecef'}),
                            html.Th("Objects", style={'backgroundColor': '#e9ecef'}),
                            html.Th("Properties", style={'backgroundColor': '#e9ecef'}),
                            html.Th("Vectorized", style={'backgroundColor': '#e9ecef'}),
                            html.Th("References", style={'backgroundColor': '#e9ecef'})
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td(name),
                                html.Td(f"{client.collections.get(name).aggregate.over_all().total_count:,}"),
                                html.Td(str(len(info['properties']))),
                                html.Td(
                                    f"✓ ({sum(1 for p in info.get('properties', []) if p.get('vectorize'))})" 
                                    if any(p.get('vectorize') for p in info.get('properties', [])) 
                                    else "-"
                                ),
                                html.Td(str(len(info['references'])))
                            ], style={'backgroundColor': '#ffffff' if i % 2 == 0 else '#f8f9fa'})
                            for i, (name, info) in enumerate(schema_info.items())
                        ])
                    ], className="table table-bordered", style={'marginBottom': '2rem'})
                ], style={
                    'marginBottom': '3rem',
                    'width': '80%',
                    'marginLeft': 'auto',
                    'marginRight': 'auto'
                })
            ]

            # Generate and clean up ERD code
            mermaid_code = inspector.generate_mermaid_erd()
            if mermaid_code.startswith('```mermaid\n'):
                mermaid_code = mermaid_code[len('```mermaid\n'):]
            if mermaid_code.endswith('```'):
                mermaid_code = mermaid_code[:-3]

            # Create ERD content
            erd = html.Div([
                Mermaid(
                    chart=mermaid_code,
                    config={
                        "securityLevel": "loose",
                        "theme": "default",
                        "themeVariables": {
                            "background": "#ffffff",
                            "primaryColor": "#e0e0e0",
                            "primaryBorderColor": "#555555",
                            "lineColor": "#555555",
                            "textColor": "#000000",
                            # Add specific ERD theme variables
                            "entityBkgColor": "#ffffff",
                            "entityBorder": "#555555",
                            "labelBackground": "#ffffff",
                            "labelBorder": "#555555",
                            "nodeBkg": "#ffffff",
                            # Class colors
                            "classText": "#000000",
                            "mainBkg": "#ffffff",
                            "titleColor": "#000000",
                            # Relationship colors
                            "edgeLabelBackground": "#ffffff",
                            "clusterBkg": "#ffffff",
                            "defaultLinkColor": "#555555",
                            # Additional diagram elements
                            "tertiaryColor": "#ffffff",
                            "noteTextColor": "#000000",
                            "noteBkgColor": "#ffffff",
                            "noteBorderColor": "#555555",
                            # ERD-specific overrides
                            "erd": {
                                "entityFill": "#ffffff",
                                "entityBorder": "#333333",
                                "attributeFill": "#ffffff",
                                "attributeBorder": "#333333",
                                "labelColor": "#000000",
                                "labelBackground": "#ffffff"
                            }
                        },
                        "er": {
                            "layoutDirection": "TB",
                            "entityPadding": 15,
                            "useMaxWidth": True
                        },
                        "maxZoom": 4,
                        "minZoom": 0.2,
                        "zoomScale": 0.5,
                        "pan": True,
                        "zoom": True,
                        "controlIcons": True
                    }
                )
            ])

            return html_content, erd, base_style

    except Exception as e:
        error_msg = f"Error updating Weaviate views: {str(e)}"
        print(f"Error: {str(e)}")
        return error_msg, html.Div(error_msg, style={'color': 'red'}), base_style

def transform_weaviate_results(json_results: dict) -> pd.DataFrame:
    """Transform Weaviate JSON results into a unified DataFrame with consistent structure.
    
    Args:
        json_results: Dict containing:
            - query_info: Query parameters and metadata
            - raw_results: Direct hits from collections
            - unified_results: Unified Article records with cross-references
            
    Returns:
        pd.DataFrame with columns:
            - score: Search relevance score
            - id: Record identifier (uuid from raw_results, id from unified_results)
            - collection: Source collection name
            - source: Original collection for raw_results, source field for unified
            - {collection}_{property}: Dynamic property columns
            - cross_references: JSON string of cross-references (or None)
            
        DataFrame.attrs contains:
            - query_info: Original query parameters
            - summary: Collection counts and statistics
    """
    try:
        print("\n=== Processing Weaviate Results ===")
        
        # Initialize empty records list and property tracking
        records = []
        all_properties = {}  # {collection: {property_name: data_type}}
        
        # Process raw_results first to discover all possible properties
        raw_results = json_results.get('raw_results', {})
        print(f"\nProcessing raw results from {len(raw_results)} collections")
        
        for collection_name, collection_results in raw_results.items():
            if collection_name == 'Article':
                continue  # Skip Articles in raw_results as they'll be handled in unified
                
            print(f"\nProcessing collection: {collection_name}")
            print(f"Found {len(collection_results)} records")
            
            # Track properties for this collection
            all_properties[collection_name] = set()
            
            for record in collection_results:
                # Create base record
                transformed = {
                    'score': record.get('score', 0.0),
                    'id': str(record.get('uuid', '')),
                    'collection': collection_name,
                    'source': collection_name,
                    'cross_references': None  # Raw results don't have cross-references
                }
                
                # Process properties
                properties = record.get('properties', {})
                for prop_name, value in properties.items():
                    column_name = f"{collection_name}_{prop_name}"
                    transformed[column_name] = value
                    all_properties[collection_name].add(prop_name)
                
                records.append(transformed)
        
        # Process unified results (Articles)
        unified_results = json_results.get('unified_results', [])
        print(f"\nProcessing {len(unified_results)} unified Article results")
        
        if unified_results:
            all_properties['Article'] = set()
            
            for record in unified_results:
                # Create base record
                transformed = {
                    'score': record.get('score', 0.0),
                    'id': str(record.get('id', '')),
                    'collection': 'Article',
                    'source': record.get('source', 'Unknown')
                }
                
                # Process Article properties
                properties = record.get('properties', {})
                for prop_name, value in properties.items():
                    column_name = f"Article_{prop_name}"
                    transformed[column_name] = value
                    all_properties['Article'].add(prop_name)
                
                # Handle cross-references
                traced = record.get('traced_elements', {})
                if traced:
                    # Convert to simplified format for storage
                    refs = {}
                    for ref_collection, elements in traced.items():
                        if elements:  # Only store non-empty references
                            refs[ref_collection] = [
                                {
                                    'id': str(elem.get('id', '')),
                                    'score': elem.get('score', 0.0)
                                } for elem in elements
                            ]
                    transformed['cross_references'] = (
                        json.dumps(refs) if refs else None
                    )
                else:
                    transformed['cross_references'] = None
                
                records.append(transformed)
        
        # Create DataFrame
        if not records:
            print("No results found")
            empty_df = pd.DataFrame(columns=[
                'score', 'id', 'collection', 'source', 'cross_references'
            ])
            empty_df.attrs['query_info'] = json_results.get('query_info', {})
            empty_df.attrs['summary'] = {'total_results': 0}
            return empty_df
        
        # Create DataFrame and ensure all columns exist
        df = pd.DataFrame(records)
        
        # Create summary information
        summary = {
            'total_results': len(df),
            'collection_counts': df['collection'].value_counts().to_dict(),
            'score_range': {
                'min': df['score'].min(),
                'max': df['score'].max(),
                'mean': df['score'].mean()
            },
            'property_coverage': {
                collection: list(props) 
                for collection, props in all_properties.items()
            }
        }
        
        # Store metadata
        df.attrs['query_info'] = json_results.get('query_info', {})
        df.attrs['summary'] = summary
        
        # Sort by score descending
        df = df.sort_values('score', ascending=False)
        
        print("\n=== Results Summary ===")
        print(f"Total records: {len(df)}")
        print("\nBy collection:")
        for collection, count in summary['collection_counts'].items():
            print(f"- {collection}: {count} records")
        print(f"\nScore range: {summary['score_range']['min']:.3f} - {summary['score_range']['max']:.3f}")
        
        return df
        
    except Exception as e:
        print(f"Error transforming Weaviate results: {str(e)}")
        print(f"Error traceback: {traceback.format_exc()}")
        # Return empty DataFrame with error information
        empty_df = pd.DataFrame(columns=[
            'score', 'id', 'collection', 'source', 'cross_references'
        ])
        empty_df.attrs['query_info'] = json_results.get('query_info', {})
        empty_df.attrs['summary'] = {'error': str(e)}
        return empty_df
    
def format_weaviate_results_preview(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Generate a formatted preview of Weaviate search results for chat display.
    
    Args:
        df: DataFrame from transform_weaviate_results
        max_rows: Maximum number of rows to show per collection
    """
    try:
        # Get metadata from DataFrame attributes
        query_info = df.attrs.get('query_info', {})
        summary = df.attrs.get('summary', {})
        
        # Start building output
        sections = []
        
        # 1. Query Information
        sections.append("### Search Query Information")
        sections.append(f"- Query: {query_info.get('text', 'Not specified')}")
        sections.append(f"- Type: {query_info.get('type', 'Not specified')}")
        sections.append(f"- Minimum Score: {query_info.get('min_score', 'Not specified')}")
        sections.append("")
        
        # 2. Result Summary
        sections.append("### Results Summary")
        sections.append(f"Total Results: {summary.get('total_results', len(df))}")
        if 'collection_counts' in summary:
            sections.append("\nResults by Collection:")
            for collection, count in summary['collection_counts'].items():
                sections.append(f"- {collection}: {count}")
        sections.append("")
        
        # 3. Collection-specific previews
        sections.append("### Result Previews")
        
        for collection in df['collection'].unique():
            collection_df = df[df['collection'] == collection].head(max_rows)
            if len(collection_df) == 0:
                continue
                
            sections.append(f"\n#### {collection} Preview")
            
            # Build preview table based on collection type
            if collection == 'Article':
                sections.append("\n| Score | ID | Filename | Abstract Preview |")
                sections.append("|-------|-----|----------|-----------------|")
                for _, row in collection_df.iterrows():
                    # Get filename and abstract
                    filename = row.get('Article_filename', 'N/A')
                    abstract = row.get('Article_abstract', '')
                    # Create abstract preview
                    abstract_preview = abstract[:50] + "..." if abstract and len(abstract) > 50 else abstract or 'N/A'
                    # Format row with proper spacing
                    sections.append(
                        f"| {row['score']:.3f} | {row['id'][:8]}... | "
                        f"{filename} | {abstract_preview} |"
                    )
                sections.append("")  # Add spacing after table
                
            elif collection == 'Reference':
                sections.append("\n| Score | ID | Title |")
                sections.append("|-------|-----|-------|")
                for _, row in collection_df.iterrows():
                    title = row.get('Reference_title', 'N/A')
                    # Truncate long titles
                    title_preview = title[:50] + "..." if len(title) > 50 else title
                    sections.append(
                        f"| {row['score']:.3f} | {row['id'][:8]}... | {title_preview} |"
                    )
                sections.append("")  # Add spacing after table
                
            elif collection == 'NamedEntity':
                sections.append("\n| Score | ID | Name | Type |")
                sections.append("|-------|-----|------|------|")
                for _, row in collection_df.iterrows():
                    name = row.get('NamedEntity_name', 'N/A')
                    entity_type = row.get('NamedEntity_type', 'N/A')
                    sections.append(
                        f"| {row['score']:.3f} | {row['id'][:8]}... | {name} | {entity_type} |"
                    )
                sections.append("")  # Add spacing after table
            
            # Add note if there are more results
            total_count = len(df[df['collection'] == collection])
            if total_count > max_rows:
                sections.append(f"... and {total_count - max_rows} more {collection} results\n")
        
        # Add score distribution analysis
        sections.append("### Score Distribution")
        thresholds = [0.3, 0.5, 0.7, 0.9]
        sections.append("\n| Minimum Score | Results | By Collection |")
        sections.append("|---------------|----------|---------------|")
        
        for threshold in thresholds:
            filtered_df = df[df['score'] >= threshold]
            if len(filtered_df) > 0:
                collection_counts = filtered_df['collection'].value_counts()
                counts_str = ", ".join(
                    f"{col}: {count}" 
                    for col, count in collection_counts.items()
                )
                sections.append(
                    f"| {threshold:.1f} | {len(filtered_df)} | {counts_str} |"
                )
        
        return "\n".join(sections)
        
    except Exception as e:
        return f"Error formatting results preview: {str(e)}"

def format_results_preview(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Format a preview of the search results.
    
    Args:
        df: DataFrame with transformed Weaviate results
        max_rows: Maximum number of rows to show
        
    Returns:
        Markdown formatted string with results preview
    """
    if df.empty:
        return "No results found."
    
    # Create display DataFrame with property and reference counts in desired order
    display_df = pd.DataFrame({
        'Score': df['score'].round(3),
        'Collection': df['collection'],
        'ID': df['object_id'],
        'Properties': df['properties'].apply(len),
        'References': df['cross_references'].apply(len)
    })
    
    # Format as markdown table
    preview = ["### Search Results", ""]
    preview.append("| Score | Collection | ID | Properties | References |")
    preview.append("|--------|------------|-----|------------|------------|")
    
    # Add rows
    for _, row in display_df.head(max_rows).iterrows():
        preview.append(
            f"| {row['Score']} | {row['Collection']} | {row['ID']} | {row['Properties']} | {row['References']} |"
        )
    
    if len(df) > max_rows:
        preview.append(f"\n... and {len(df) - max_rows} more results.")
    
    return "\n".join(preview)

def generate_score_distribution(df: pd.DataFrame) -> str:
    """Generate a markdown summary of result score distribution.
    
    Args:
        df: DataFrame with transformed Weaviate results
        
    Returns:
        Markdown formatted string showing score distribution
    """
    if df.empty:
        return "No results found."
        
    # Get summary information
    summary = df.attrs.get('summary', {})
    total_matches = summary.get('total_matches', len(df))
    collection_counts = summary.get('collection_counts', df['collection'].value_counts().to_dict())
    
    # Calculate counts at different thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    threshold_results = []
    
    for threshold in thresholds:
        filtered_df = df[df['score'] >= threshold]
        if len(filtered_df) > 0:
            # Get counts by collection for this threshold
            counts = filtered_df['collection'].value_counts()
            details = ", ".join([f"{col}: {count}" for col, count in counts.items()])
            threshold_results.append({
                'threshold': threshold,
                'total': len(filtered_df),
                'details': details
            })
    
    # Build markdown output
    output = []
    
    # Add summary section
    output.append("### Search Results Summary")
    output.append(f"\nTotal Matches: {total_matches}")
    output.append("\nResults by Collection:")
    for collection, count in collection_counts.items():
        output.append(f"- {collection}: {count}")
    
    # Add threshold table
    output.append("\n### Score Distribution")
    output.append("\n| Minimum Score | Total Results | Details |")
    output.append("|--------------|---------------|----------|")
    for result in threshold_results:
        output.append(f"| {result['threshold']:.1f} | {result['total']} | {result['details']} |")
    
    # Add instructions
    output.append("\nTo view results, specify a minimum relevance score using: Use threshold X.X")
    output.append("\nFor example:")
    output.append("- Use threshold 0.3 for broader coverage")
    output.append("- Use threshold 0.7 for high-relevance results only")
    output.append("\nNote: This command specifically sets the literature search threshold.")
    
    return "\n".join(output)

def is_literature_query(message: str) -> Tuple[bool, Optional[str]]:
    """Detect if a message is requesting literature information.
    
    Returns:
        Tuple[bool, Optional[str]]: (is_literature_query, extracted_query)
    """
    patterns = [
        # Basic knowledge patterns
        r'(?:what is|what\'s) known about\s+(.+?)(?:\?|$)',
        
        # Direct literature search patterns
        r'(?:find|search for|look for|get)(?:\s+\w+)?\s+(?:papers?|articles?|literature|research)(?:\s+\w+)?\s+(?:about|on|related to|regarding|concerning)\s+(.+?)(?:\?|$)',
        
        # Research inquiry patterns
        r'tell me about the research (?:on|in|about)\s+(.+?)(?:\?|$)',
        r'what research exists (?:on|about)\s+(.+?)(?:\?|$)',
        r'what (?:papers|articles) discuss\s+(.+?)(?:\?|$)',
        
        # Biological entity patterns
        r'tell me about\s+([A-Z][a-z]+\s+[a-z]+)(?:\?|$)',
        r'(?:find|search for|tell me about)\s+(.+?(?:gene|protein|pathway|transposon|plasmid|enzyme|regulator))(?:\?|$)',
        r'what (?:is|are)\s+([A-Z][a-z]+\s+[a-z]+)(?:\?|$)',
        
        # Literature request patterns
        r'(?:show|give|get)(?:\s+\w+)?\s+(?:papers?|articles?|literature|research)\s+(?:about|on|for)\s+(.+?)(?:\?|$)',
        r'(?:papers?|articles?|literature|research)\s+(?:about|on|related to)\s+(.+?)(?:\?|$)',
        
        # Enzyme-specific patterns
        r'(?:find|tell me about|search for)\s+(.+?ase[s]?)(?:\?|$)',  # Match enzyme names ending in 'ase'
        r'(?:find|tell me about|search for)\s+(.+?(?:reductase|oxidase|synthase|kinase|phosphatase))(?:\?|$)'  # Common enzyme types
    ]
    
    print("\n=== Literature Query Detection Debug ===")
    print(f"Input message: '{message}'")
    normalized = message.lower().strip()
    print(f"Normalized message: '{normalized}'")
    print("\nTrying patterns:")
    
    for pattern in patterns:
        print(f"\nTrying pattern: {pattern}")
        match = re.search(pattern, normalized)
        if match:
            query = match.group(1).strip()
            print(f"Match found! Extracted query: '{query}'")
            return True, query
    
    print("No literature query patterns matched")
    return False, None

def extract_threshold_from_message(message: str) -> Optional[float]:
    """Extract threshold value from user message.
    
    Args:
        message: User's chat message
        
    Returns:
        Float threshold value or None if not found/invalid
        
    Examples:
        >>> extract_threshold_from_message("Use threshold 0.3")
        0.3
        >>> extract_threshold_from_message("Set cutoff to 0.7")
        0.7
    """
    patterns = [
        r"(?:use|set|apply|with)\s+(?:a\s+)?(?:threshold|cutoff|score|limit)\s+(?:of\s+)?(\d+\.?\d*)",
        r"threshold\s+(?:of\s+)?(\d+\.?\d*)",
        r"cutoff\s+(?:of\s+)?(\d+\.?\d*)"
    ]
    
    message = message.lower().strip()
    
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            try:
                threshold = float(match.group(1))
                if 0 <= threshold <= 1:
                    return threshold
            except ValueError:
                continue
    
    return None

####################################
#
# Weaviate Management Functions
#
####################################

@callback(
    [Output('weaviate-connection-icon', 'style'),
     Output('weaviate-collections-icon', 'style'),
     Output('weaviate-state', 'data', allow_duplicate=True)],
    [Input('_weaviate-init', 'data'),
     Input('database-connect-button', 'n_clicks')],
    [State('weaviate-state', 'data')],
    prevent_initial_call='initial_duplicate'
)
def update_weaviate_connection(init_trigger, n_clicks, current_state):
    """Update Weaviate connection status on app start and database connect."""
    try:
        from weaviate_integration import WeaviateConnection
        
        # Get current status
        weaviate = WeaviateConnection()
        status = weaviate.get_status()
        
        # Define status colors (colorblind-friendly)
        colors = {
            'connected': '#2ecc71',    # Teal for success
            'error': '#e67e22',        # Orange for error
            'disconnected': '#6c757d',  # Neutral gray
            'available': '#2ecc71',     # Teal for success
            'unavailable': '#6c757d',   # Neutral gray
            'warning': '#3498db'        # Light teal for warning
        }
        
        # Update connection icon style
        conn_style = {
            'color': colors[status['connection']['status']],
            'marginRight': '5px'
        }
        
        # Update collections icon style
        coll_style = {
            'color': colors[status['collections']['status']]
        }
        
        return conn_style, coll_style, status
        
    except Exception as e:
        print(f"Error updating Weaviate connection: {str(e)}")
        # Return neutral gray for both icons on error
        error_state = {
            'connection': {'status': 'error', 'message': str(e)},
            'collections': {'status': 'unavailable', 'message': 'No collections'}
        }
        return (
            {'color': '#6c757d', 'marginRight': '5px'},
            {'color': '#6c757d'},
            error_state
        )

@callback(
    [Output('weaviate-connection-tooltip', 'children'),
     Output('weaviate-collections-tooltip', 'children')],
    Input('weaviate-state', 'data')
)
def update_weaviate_tooltips(state):
    """Update tooltips with status messages."""
    if not state:
        return "Not initialized", "Not initialized"
        
    conn_msg = state['connection']['message']
    coll_msg = state['collections']['message']
    
    return conn_msg, coll_msg

def generate_query_id(is_original: bool = True, alt_number: Optional[int] = None) -> str:
    """Generate a unique query ID with timestamp.
    
    Args:
        is_original (bool): If True, generates ID for primary query
        alt_number (int, optional): For alternative queries, specifies which alternative (1,2,etc)
        
    Returns:
        str: Query ID in format query_YYYYMMDD_HHMMSS_(original|altN)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_original:
        suffix = '_original'
    else:
        if alt_number is None:
            alt_number = 1
        suffix = f'_alt{alt_number}'
        
    return f"query_{timestamp}{suffix}"

def generate_literature_query_id(query: str, threshold: float) -> str:
    """Generate a unique ID for literature queries.
    
    Args:
        query: The literature search query
        threshold: The relevance threshold used
        
    Returns:
        str: Query ID in format lit_query_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"lit_query_{timestamp}"

def add_query_ids_to_response(response: str) -> str:
    """Add unique Query IDs to SQL blocks in the response."""
    print("\n=== SQL Block Processing Debug ===")
    print(f"First 100 chars: {response[:100]}")
    
    if '```sql' not in response.lower():
        print("No SQL blocks detected, returning original response")
        return response
        
    try:
        # Process SQL blocks
        modified_response = []
        current_pos = 0
        
        # Use regex to find all SQL blocks
        sql_block_pattern = r'(```sql\s*\n?)(.*?)(\n?```)'
        matches = list(re.finditer(sql_block_pattern, response, re.DOTALL | re.IGNORECASE))
        
        print(f"Found {len(matches)} SQL blocks")
        
        # Process each block in order
        for i, match in enumerate(matches):
            start, end = match.span()
            modified_response.append(response[current_pos:start])
            
            sql_block = match.group(2).strip()
            # Skip if block already has a Query ID
            if re.search(r'^--\s*Query ID:', sql_block):
                modified_response.append(match.group(0))
            else:
                # First block gets _original, rest get _altN
                if i == 0:
                    query_id = generate_query_id(is_original=True)
                else:
                    query_id = generate_query_id(is_original=False, alt_number=i)
                
                # Add ID comment at the start of the block
                modified_block = f"{match.group(1)}-- Query ID: {query_id}\n{sql_block}{match.group(3)}"
                modified_response.append(modified_block)
            
            current_pos = end
        
        # Add any remaining text
        modified_response.append(response[current_pos:])
        
        result = ''.join(modified_response)
        print("\nProcessed response preview:")
        print(result[:200] + "..." if len(result) > 200 else result)
        return result
        
    except Exception as e:
        print(f"Error processing SQL blocks: {str(e)}")
        return response

def search_all_sources(query: str, threshold: float = 0.6, successful_queries: dict = None) -> dict:
    """Search across all available data sources for relevant information."""
    # Initialize or use provided store
    successful_queries = successful_queries or {}
    
    results = {
        'dataset_matches': [],
        'database_matches': [],
        'literature_matches': [],
        'metadata': {
            'sources_searched': [],
            'total_matches': 0,
            'coverage': {},
            'query': query,
            'threshold': threshold,
            'search_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }
    
    total_matches = 0
    
    # Check for literature query first
    is_lit, lit_query = is_literature_query(query)
    if is_lit:
        try:
            print("\n=== Executing Literature Search ===")
            print(f"Literature query: {lit_query}")
            weaviate_results = execute_weaviate_query(lit_query)
            if weaviate_results and weaviate_results.get('unified_results'):
                df = transform_weaviate_results(weaviate_results)
                query_id = generate_literature_query_id(lit_query, threshold)
                
                # Store the DataFrame as records in the store
                successful_queries[query_id] = {
                    'query': lit_query,
                    'threshold': threshold,
                    'dataframe': df.to_dict('records'),  # Convert to serializable format
                    'metadata': {
                        'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'total_results': len(df),
                        'query_info': df.attrs.get('query_info', {}),
                        'summary': df.attrs.get('summary', {})
                    }
                }
                
                results['literature_matches'] = {
                    'results': weaviate_results,
                    'preview': format_literature_preview(df, query_id, threshold),
                    'query_id': query_id
                }
                lit_match_count = len(weaviate_results.get('unified_results', []))
                total_matches += lit_match_count
                results['metadata']['sources_searched'].append('literature')
                results['metadata']['coverage']['literature'] = {
                    'total_matches': lit_match_count,
                    'total_articles': len(df) if df is not None else 0
                }
        except Exception as e:
            print(f"Literature search error: {str(e)}")
            results['metadata']['errors'] = results['metadata'].get('errors', [])
            results['metadata']['errors'].append(f"Literature search error: {str(e)}")

    # ... rest of existing code for dataset and database searches ...

    # Update total matches
    results['metadata']['total_matches'] = total_matches
    
    return results, successful_queries  # Return both results and updated store

def format_search_results(results: dict) -> str:
    """Format search results into a readable markdown summary."""
    # Early return if no sources were searched
    if not results['metadata']['sources_searched']:
        return "No data sources were available to search."

    # Calculate match counts
    dataset_matches = len(results.get('dataset_matches', []))
    database_matches = len(results.get('database_matches', []))
    literature_matches = 0
    if results.get('literature_matches'):
        literature_matches = results['literature_matches'].get('results', {}).get('unified_results', [])
        literature_matches = len(literature_matches)
    total_matches = dataset_matches + database_matches + literature_matches

    # Early return if no matches found
    if total_matches == 0:
        return "No matches found in available data sources."

    sections = []
    
    # Add header with search coverage info
    sections.append("### Search Results Summary")
    sources = results['metadata']['sources_searched']
    
    sections.append(f"Found {total_matches} total matches across available sources:")
    if 'literature' in sources and literature_matches > 0:
        sections.append(f"- Literature: {literature_matches} matches")
    if 'datasets' in sources and dataset_matches > 0:
        sections.append(f"- Datasets: {dataset_matches} matches")
    if 'database' in sources and database_matches > 0:
        sections.append(f"- Database: {database_matches} matches")
    sections.append("")
    
    # Format literature matches first if present
    if results.get('literature_matches'):
        sections.append("#### Literature Matches")
        sections.append(results['literature_matches']['preview'])
    
    # Format dataset matches
    if results['dataset_matches']:
        sections.append("#### Dataset Matches")
        for match in results['dataset_matches']:
            sections.append(f"\nDataset: **{match['source_name']}** (Score: {match['similarity']:.2f})")
            for col_name, details in match['details'].items():
                total_matches = len(details['matches'])
                sections.append(f"\nColumn `{col_name}`: {total_matches} matching values")
                
                # Show all matches with counts, sorted by similarity
                all_matches = sorted(
                    details['matches'],
                    key=lambda x: details['similarities'][x],
                    reverse=True
                )
                for value in all_matches:
                    count = details['counts'][value]
                    similarity = details['similarities'][value]
                    sections.append(f"- '{value}' ({count} occurrences, {similarity:.0f}% match)")
    
    # Format database matches
    if results['database_matches']:
        sections.append("\n#### Database Matches")
        for match in results['database_matches']:
            sections.append(f"\nTable: **{match['source_name']}** (Score: {match['similarity']:.2f})")
            for col_name, details in match['details'].items():
                total_matches = len(details['matches'])
                sections.append(f"\nColumn `{col_name}`: {total_matches} matching values")
                
                # Show all matches with counts, sorted by similarity
                all_matches = sorted(
                    details['matches'],
                    key=lambda x: details['similarities'][x],
                    reverse=True
                )
                for value in all_matches:
                    count = details['counts'][value]
                    similarity = details['similarities'][value]
                    sections.append(f"- '{value}' ({count} occurrences, {similarity:.0f}% match)")
    
    # Add error information if any
    if 'errors' in results['metadata']:
        sections.append("\n#### Search Errors")
        for error in results['metadata']['errors']:
            sections.append(f"- {error}")
    
    return "\n".join(sections)

def is_search_query(message: str) -> bool:
    """Detect if a message is a search query using various patterns."""
    # First check if it's a dataset/database info request
    if re.search(r'tell\s+me\s+about\s+(?:my\s+)?(?:database|datasets?)\b', message.lower()):
        return False
        
    search_patterns = [
        # Direct search commands
        r'^(?:find|search|look)\s+(?:for|up)?\s*(.+)$',
        # Question-based searches
        r'(?:where|which|what|how\s+many|who|when)\s+(?:are|is|was|were|have|has|had)?\s*(.+)$',
        # Show/list/get patterns
        r'(?:show|list|get|give)\s+(?:me)?\s+(?:all|any)?\s*(.+)$',
        # About/containing patterns
        r'(?:about|containing|related\s+to|with)\s*(.+)$',
        # Tell me patterns that imply search
        r'tell\s+me\s+(?:about|of|all)?\s+(.+)$'
    ]
    
    print("\n=== Search Pattern Detection Debug ===")
    print(f"Input message: '{message}'")
    normalized = message.lower().strip()
    print(f"Normalized message: '{normalized}'")
    print("\nTrying search patterns:")
    
    for pattern in search_patterns:
        print(f"\nPattern: {pattern}")
        if re.search(pattern, normalized):
            print("✓ Pattern matched!")
            return True
        print("✗ No match")
    
    return False

def format_literature_preview(df: pd.DataFrame, query_id: str, threshold: float) -> str:
    """Format literature results preview with query ID and conversion instructions.
    
    Args:
        df: DataFrame from transform_weaviate_results
        query_id: Unique identifier for this literature query
        threshold: Current relevance threshold
        
    Returns:
        str: Formatted preview with ID and instructions
    """
    # Get the standard preview
    preview = format_weaviate_results_preview(df)
    
    # Add DataFrame preview with ID
    preview_rows = min(5, len(df))  # Show up to 5 rows
    df_preview = df.head(preview_rows)
    
    # Select most relevant columns for preview
    preview_columns = ['score', 'collection']
    if 'Article_title' in df.columns:
        preview_columns.append('Article_title')
    elif 'Article_filename' in df.columns:
        preview_columns.append('Article_filename')
    if 'Article_abstract' in df.columns:
        preview_columns.append('Article_abstract')
    
    df_section = f"""

Results preview:

Query ID: {query_id}
```
{df_preview[preview_columns].to_string()}
```

Current threshold: {threshold}

Available actions:
1. Refine results with different threshold:
   refine {query_id} with threshold 0.X
   
2. Save results as dataset:
   convert {query_id} to dataset"""
    
    return preview + df_section

def extract_query_id_from_message(message: str) -> Optional[str]:
    """Extract literature query ID from refinement command.
    
    Args:
        message: User's chat message
        
    Returns:
        str: Query ID if found, None otherwise
        
    Examples:
        >>> extract_query_id_from_message("refine lit_query_20250207_123456 with threshold 0.7")
        'lit_query_20250207_123456'
    """
    patterns = [
        r"(?:refine|update|modify)\s+(lit_query_\d{8}_\d{6})",
        r"(lit_query_\d{8}_\d{6})\s+(?:with|using|at)\s+threshold"
    ]
    
    message = message.lower().strip()
    
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    
    return None

if __name__ == '__main__':
    # Start the app
    app.run_server(debug=True, host='0.0.0.0', port=8051)
