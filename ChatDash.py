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

# Find the project root directory (where .env is located)
project_root = Path(__file__).parent
dotenv_path = project_root / '.env'

# Try to load from .env file
load_dotenv(dotenv_path=dotenv_path)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=Warning)

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

📊 **Data Analysis & Visualization**
- Create plots: "Plot \\[column1\\] vs \\[column2\\]"
  - Optional: Add size=\\[value/column\\] color=\\[value/column\\]
- Create heatmaps: "Create heatmap columns=\\[col1,col2,col3\\]"
  - Options: rows=\\[...\\] standardize=rows/columns cluster=both/rows/columns
- Create maps: "Create map latitude=\\[column1\\] longitude=\\[column2\\]"
  - Optional: Add size=\\[value/column\\] color=\\[value/column\\]

💡 **Tips**
- Use natural language to ask questions about your data
- You may need to reference specific columns using \\`backticks\\`
- Click the dataset cards to switch between datasets
"""

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
    }
}

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

# Initialize OpenAI client
client = openai.OpenAI(**OPENAI_CONFIG)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    'https://use.fontawesome.com/releases/v5.15.4/css/all.css'  # Add Font Awesome
])
app.config.suppress_callback_exceptions = True

# Helper Functions
def create_system_message(dataset_info: List[Dict[str, Any]], 
                         search_query: Optional[str] = None,
                         database_structure: Optional[Dict] = None,
                         weaviate_results: Optional[Dict] = None) -> str:
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

    return base_message

def create_chat_element(message: dict) -> dbc.Card:
    """
    Create a styled chat element based on message type.
    
    Args:
        message (dict): Message dictionary containing:
            - role (str): One of 'user', 'assistant', or 'system'
            - content (str): The message text
        
    Returns:
        dbc.Card: Styled card component for the chat interface with:
            - Appropriate styling based on message role
            - Markdown rendering for assistant messages
            - Proper alignment and spacing
    """
    style = CHAT_STYLES.get(message['role'], CHAT_STYLES['assistant'])
    content = dcc.Markdown(message['content']) if message['role'] == 'assistant' else message['content']
    
    return dbc.Card(
        dbc.CardBody(content),
        className="mb-2 ml-auto" if message['role'] == 'user' else "mb-2",
        style=style
    )

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

# Add this callback to handle URL fragments
@app.callback(
    Output('dataset-tabs', 'active_tab'),
    Input('url', 'hash')
)
def handle_url_fragment(hash_value):
    return dash.no_update

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

# Add custom error handling utilities
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

# Callback for handling chat input
@callback(
    [Output('chat-history', 'children'),
     Output('chat-input', 'value', allow_duplicate=True),
     Output('chat-store', 'data'),
     Output('dataset-tabs', 'active_tab', allow_duplicate=True),  # Fix hyphen to underscore
     Output('viz-state-store', 'data'),
     Output('chat-loading-output', 'children', allow_duplicate=True)],
    [Input('send-button', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-store', 'data'),
     State('model-selector', 'value'),
     State('datasets-store', 'data'),
     State('selected-dataset-store', 'data'),
     State('database-state', 'data'),           
     State('database-structure-store', 'data'),
     State('successful-queries-store', 'data')],  # Add successful_queries store
    prevent_initial_call='initial_duplicate'
)
def handle_chat_message(n_clicks, input_value, chat_history, model, datasets, selected_dataset, database_state, database_structure_store, successful_queries):
    """Process chat messages and handle various command types."""
    try:
        if not input_value:
            return (dash.no_update,) * 6
            
        chat_history = chat_history or []
        current_message = {'role': 'user', 'content': input_value.strip()}
        chat_history.append(current_message)
        
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
        if is_search_query(input_value):
            print("\n=== Processing Search Query ===")
            search_results = search_all_sources(input_value)
            
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
            ""
        )
        
    except Exception as e:
        print(f"Error in handle_chat_message: {str(e)}")
        print(traceback.format_exc())
        return (dash.no_update,) * 6

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

# Optimize dataset search
def find_mentioned_dataset(input_text: str, dataset_names: list) -> str:
    """
    Efficiently search for mentioned dataset names in input text.
    
    Args:
        input_text (str): User's input message
        dataset_names (list): List of available dataset names
        
    Returns:
        str: Name of mentioned dataset or None
    """
    input_lower = input_text.lower()
    return next((name for name in dataset_names if name.lower() in input_lower), None)

# Add a new callback to handle the help button
@callback(
    [Output('chat-input', 'value', allow_duplicate=True),
     Output('send-button', 'n_clicks', allow_duplicate=True)],
    Input('help-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_help(n_clicks):
    """
    Inject a help query into the chat when the help button is clicked.
    
    Args:
        n_clicks (int): Number of times help button has been clicked
        
    Returns:
        tuple: (help_message, trigger_send)
    """
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    help_message = "What can I do with this chat interface?"
    return help_message, 1

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

@callback(
    [Output('chat-history', 'children', allow_duplicate=True),
     Output('chat-input', 'value', allow_duplicate=True),
     Output('chat-store', 'data', allow_duplicate=True),
     Output('dataset-tabs', 'active_tab', allow_duplicate=True),  # Fix hyphen to underscore
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
     State('datasets-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def execute_confirmed_query(input_value, n_clicks, chat_history, database_state, database_structure_store, successful_queries, datasets):
    """Process chat commands related to SQL query execution and dataset conversion.

    This function handles three main types of operations:
    1. SQL query execution (e.g., "execute." or "execute query_20240315_123456_original")
    2. Query-to-dataset conversion (e.g., "convert query_20240315_123456_original to dataset")
    3. Regular chat processing (all other inputs)

    Args:
        input_value (str): The user's chat input message
        n_clicks (int): Number of times send button clicked (for triggering)
        chat_history (list): List of chat message dictionaries with 'role' and 'content'
        database_state (dict): Current database connection state including path
        database_structure_store (dict): Database schema information
        successful_queries (dict): Store of previously executed queries with metadata
        datasets (dict): Currently loaded datasets in browser memory

    Returns:
        tuple: (
            chat_elements (list): Updated chat interface components
            input_value (str): Cleared input field
            chat_history (list): Updated chat history
            active_tab (str): Tab to display (or dash.no_update)
            viz_state (dict): Visualization state (or dash.no_update)
            loading_output (str): Loading indicator state
            successful_queries (dict): Updated query store
            datasets (dict): Updated dataset store
            dataset_list (list): Updated dataset card components
        )

    Query Storage Format:
        successful_queries = {
            'query_20240315_123456_original': {
                'sql': 'SELECT ...',
                'metadata': {...}
            }
        }

    Dataset Conversion:
        When converting a query to dataset, creates a new dataset with:
        - Data from fresh query execution
        - Metadata including query origin
        - Statistical profile report
        - Memory-efficient storage format
    """
    if not input_value:
        return (dash.no_update,) * 9
    
    # Initialize stores safely
    successful_queries = successful_queries or {}
    chat_history = chat_history or []
    datasets = datasets or {}
    
    #print(f"Current store state:")
    #print(f"- Successful queries: {len(successful_queries)} stored")
    #print(f"- Chat history: {len(chat_history)} messages")
    #print(f"- Datasets: {len(datasets)} loaded")
    
    # Check for dataset conversion request
    convert_match = re.search(r'convert\s+(query_\d{8}_\d{6}(?:_original|_alt\d+))\s+to\s+dataset', input_value.lower().strip())
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
            # Get stored query details
            stored_query = successful_queries[query_id]
            
            # Execute query to get fresh data
            print("Executing query to get fresh data...")
            df, metadata, _ = execute_sql_query(stored_query['sql'], database_state['path'])
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
                'metadata': {
                    'filename': f"{dataset_name}.csv",
                    'source': f"Database query: {query_id}",
                    'database': database_state['path'],
                    'sql': stored_query['sql'],
                    'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df),
                    'columns': [df.index.name or 'index'] + list(df.columns)
                },
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
        
        # Check for unprocessed plot request in the last message only
        last_message = chat_history[-2] if len(chat_history) >= 2 else None
        if last_message and last_message.get('plot_request') and not last_message.get('processed'):
            plot_request = last_message['plot_request']
            
            # Process the plot request
            available_columns = list(df.columns)
            params, error_msg = extract_plot_params(plot_request, available_columns)
            
            if error_msg:
                chat_history.append({
                    'role': 'assistant',
                    'content': f"{error_msg}\n\nAvailable columns are: {', '.join(available_columns)}"
                })
                # Mark the request as processed
                last_message['processed'] = True
                return dataset_name, create_chat_elements_batch(chat_history), chat_history, 'tab-preview'
            
            # Store plot parameters and mark as processed
            last_message['processed'] = True
            chat_history.append({
                'role': 'assistant',
                'content': (
                    f"I've set up a bubble plot with:\n"
                    f"- X: {params['x_column']}\n"
                    f"- Y: {params['y_column']}\n"
                    f"- Size: {params['size'] if params['size'] else 'default'}\n"
                    f"- Color: {params['color'] if params['color'] else 'default'}\n\n"
                    f"You can view and adjust the plot in the Visualization tab."
                ),
                'plot_params': params
            })
            
            return dataset_name, create_chat_elements_batch(chat_history), chat_history, 'tab-viz'
        
        # Default behavior: just update dataset selection
        return dataset_name, create_chat_elements_batch(chat_history), chat_history, 'tab-preview'
        
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

class BubblePlotParams:
    def __init__(self):
        self.x_column = None
        self.y_column = None
        self.size = None  # Could be column name, number, or None
        self.color = None  # Could be column name, color name, or None
        
    def validate(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate parameters against a dataframe."""
        try:
            # Check required parameters
            if not self.x_column or not self.y_column:
                return False, "X and Y columns are required"
                
            # Check if columns exist
            if self.x_column not in df.columns:
                return False, f"X column '{self.x_column}' not found in dataset"
            if self.y_column not in df.columns:
                return False, f"Y column '{self.y_column}' not found in dataset"
                
            # If size is a column name, check if it exists
            if isinstance(self.size, str) and self.size in df.columns:
                if not pd.to_numeric(df[self.size], errors='coerce').notna().all():
                    return False, f"Size column '{self.size}' must contain numeric values"
                    
            # If color is a column name, check if it exists
            if isinstance(self.color, str) and self.color in df.columns:
                pass  # We'll handle color mapping later
                
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

# Add a new callback to handle the plot validation
@callback(
    Output('plot-validation-output', 'children'),
    [Input('x-column', 'value'),
     Input('y-column', 'value'),
     Input('size-input', 'value'),
     Input('color-input', 'value')],
    [State('selected-dataset-store', 'data'),
     State('datasets-store', 'data')]
)
def validate_plot_params(x_col, y_col, size, color, selected_dataset, datasets):
    """Test the BubblePlotParams validation."""
    if not selected_dataset or not datasets or selected_dataset not in datasets:
        return "Please select a dataset"
        
    try:
        # Get the dataframe
        df = pd.DataFrame(datasets[selected_dataset]['df'])
        
        # Create and populate params
        params = BubblePlotParams()
        params.x_column = x_col
        params.y_column = y_col
        params.size = size
        params.color = color
        
        # Validate
        is_valid, message = params.validate(df)
        
        if is_valid:
            return html.Div("Parameters are valid!", style={'color': 'green'})
        else:
            return html.Div(f"Validation error: {message}", style={'color': 'red'})
            
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': 'red'})

# Add a new callback to handle the plot validation
@callback(
    Output('bubble-plot', 'figure'),
    [Input('x-column', 'value'),
     Input('y-column', 'value'),
     Input('size-input', 'value'),
     Input('color-input', 'value')],
    [State('selected-dataset-store', 'data'),
     State('datasets-store', 'data')]
)
def update_bubble_plot(x_col, y_col, size, color, selected_dataset, datasets):
    """Generate the bubble plot based on selected parameters."""
    if not selected_dataset or not datasets or selected_dataset not in datasets:
        return {}
        
    try:
        # Get the dataframe
        df = pd.DataFrame(datasets[selected_dataset]['df'])
        
        # Create and validate parameters
        params = BubblePlotParams()
        params.x_column = x_col
        params.y_column = y_col
        params.size = size
        params.color = color
        
        is_valid, message = params.validate(df)
        if not is_valid:
            return {}
            
        # Process size parameter
        if params.size is None:
            marker_size = 20  # Default size
        elif params.size in df.columns:
            marker_size = df[params.size]
        else:
            try:
                marker_size = float(params.size)
            except ValueError:
                marker_size = 20
                
        # Process color parameter
        if params.color is None:
            marker_color = 'blue'  # Default color
        elif params.color in df.columns:
            marker_color = df[params.color]
        else:
            marker_color = params.color
            
        # Create the plot
        fig = px.scatter(
            df,
            x=params.x_column,
            y=params.y_column,
            size=marker_size if isinstance(marker_size, pd.Series) else None,
            color=marker_color if isinstance(marker_color, pd.Series) else None,
            title=f'Bubble Plot: {params.y_column} vs {params.x_column}'
        )
        
        if not isinstance(marker_size, pd.Series):
            fig.update_traces(marker=dict(size=marker_size))
        if not isinstance(marker_color, pd.Series) and marker_color is not None:
            fig.update_traces(marker=dict(color=marker_color))
            
        return fig
        
    except Exception as e:
        return {}

def is_plot_request(message: str) -> bool:
    """Check if the message is requesting a plot."""

    plot_keywords = ['plot', 'graph', 'visualize', 'show', 'display']
    message = message.lower()
    return any(keyword in message for keyword in plot_keywords)

def extract_plot_params(message: str, available_columns: List[str]) -> Tuple[dict, Optional[str]]:
    """Extract plot parameters from message using fuzzy matching."""
    params = {'x_column': None, 'y_column': None, 'size': None, 'color': None}
    
    # Clean up message - remove punctuation from vs variations
    message = message.replace('vs.', 'vs').replace('versus.', 'versus')
    
    # Extract x and y columns
    if 'vs' in message:
        parts = message.split('vs')
    elif 'versus' in message:
        parts = message.split('versus')
    elif 'against' in message:
        parts = message.split('against')
    else:
        return params, "Could not find column comparison using 'vs', 'versus', or 'against'"
    
    # Clean and extract column names
    y_col = parts[0].replace('plot', '').strip()
    x_col = parts[1].split('size=')[0].split('color=')[0].strip()
    
    # Match column names
    if y_col.lower() not in [col.lower() for col in available_columns]:
        return params, f"Could not find Y column '{y_col}'. Available columns are: {', '.join(available_columns)}"
    if x_col.lower() not in [col.lower() for col in available_columns]:
        return params, f"Could not find X column '{x_col}'. Available columns are: {', '.join(available_columns)}"
    
    # Get original case for columns
    params['y_column'] = next(col for col in available_columns if col.lower() == y_col.lower())
    params['x_column'] = next(col for col in available_columns if col.lower() == x_col.lower())
    
    # Extract size parameter - allow numbers
    size_match = re.search(r'size\s*=\s*(\d+(?:\.\d+)?|\w+)', message)
    if size_match:
        params['size'] = size_match.group(1)
    
    # Extract color parameter - allow hex colors and RGB values
    color_match = re.search(r'color\s*=\s*(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|rgb\s*\([^)]+\)|\w+)', message)
    if color_match:
        params['color'] = color_match.group(1)
    
    return params, None

def extract_map_params(message: str, available_columns: List[str]) -> Tuple[dict, str]:
    """Extract map parameters from message."""
    params = {
        'latitude': None,
        'longitude': None,
        'size': None,
        'color': None
    }
    
    lat_match = re.search(r'latitude\s*=\s*(\w+)', message.lower())
    lon_match = re.search(r'longitude\s*=\s*(\w+)', message.lower())
    
    if not lat_match or not lon_match:
        return params, """To create a map, please specify both latitude and longitude columns:
map latitude=<column1> longitude=<column2>
Optional: Add size=<column/value> or color=<column/value>"""
    
    lat_col = lat_match.group(1)
    lon_col = lon_match.group(1)
    
    # Fuzzy match column names
    lat_col = process.extractOne(lat_col, available_columns)[0]
    lon_col = process.extractOne(lon_col, available_columns)[0]
    
    params['latitude'] = lat_col
    params['longitude'] = lon_col
    
    # Extract optional parameters
    size_match = re.search(r'size\s*=\s*(\w+|\d+\.?\d*)', message.lower())
    if size_match:
        size_val = size_match.group(1)
        try:
            size_val = float(size_val)
            params['size'] = size_val
        except ValueError:
            size_col = process.extractOne(size_val, available_columns)[0]
            params['size'] = size_col
    
    color_match = re.search(r'color\s*=\s*(\w+)', message.lower())
    if color_match:
        color_col = color_match.group(1)
        color_col = process.extractOne(color_col, available_columns)[0]
        params['color'] = color_col
    
    return params, None

def extract_heatmap_params(message: str, available_columns: List[str]) -> Tuple[dict, Optional[str]]:
    """Extract heatmap parameters from message.
    
    Example formats:
    - columns=[Be, B, Na, Mg] transpose
    - columns=[Be,B,Na,Mg] rows=[Fe,Cu] transpose=true
    - columns=[Be,B,Na,Mg] standardize=rows cluster=both colormap=RdBu
    """
    params = {
        'rows': None,
        'columns': None,
        'standardize': None,
        'transpose': False,
        'cluster': None,
        'colormap': 'viridis'
    }
    
    # Extract bracketed lists for rows and columns
    row_match = re.search(r'rows?\s*=\s*\[(.*?)\]', message)
    col_match = re.search(r'columns?\s*=\s*\[(.*?)\]', message)
    std_match = re.search(r'standardize=([^\s\]]+)', message)
    cluster_match = re.search(r'cluster=([^\s\]]+)', message)
    colormap_match = re.search(r'colou?rmap=([^\s\]]+)', message)
    
    # Check for transpose before processing columns
    params['transpose'] = 'transpose' in message.lower()
    if 'transpose=' in message.lower():
        transpose_match = re.search(r'transpose=(true|false)', message.lower())
        if transpose_match:
            params['transpose'] = transpose_match.group(1).lower() == 'true'
    
    # Process rows
    if row_match:
        rows = [col.strip() for col in row_match.group(1).split(',') if col.strip()]
        invalid_rows = [col for col in rows if col not in available_columns]
        if invalid_rows:
            return params, f"Invalid row columns: {', '.join(invalid_rows)}\nAvailable columns: {', '.join(available_columns)}"
        params['rows'] = rows
        
    # Process columns
    if col_match:
        cols = [col.strip() for col in col_match.group(1).split(',') if col.strip()]
        invalid_cols = [col for col in cols if col not in available_columns]
        if invalid_cols:
            valid_cols = [col for col in cols if col in available_columns]
            error_msg = (
                f"Found {len(invalid_cols)} invalid column names:\n"
                f"Invalid: {', '.join(invalid_cols)}\n"
                f"Valid: {', '.join(valid_cols)}"
            )
            return params, error_msg
        params['columns'] = cols
        
    # Process standardization
    if std_match:
        std_value = std_match.group(1).lower()
        if std_value not in ['rows', 'columns', 'none']:
            return params, "Standardization must be 'rows', 'columns', or 'none'"
        params['standardize'] = None if std_value == 'none' else std_value
        
    # Process clustering
    if cluster_match:
        cluster_value = cluster_match.group(1).lower()
        if cluster_value not in ['rows', 'columns', 'both', 'none']:
            return params, "Clustering must be 'rows', 'columns', 'both', or 'none'"
        params['cluster'] = None if cluster_value == 'none' else cluster_value
        
    # Process colormap
    if colormap_match:
        colormap = colormap_match.group(1).lower()
        valid_colormaps =px.colors.named_colorscales()
        if colormap not in valid_colormaps:
            return params, f"Invalid colormap. Choose from: {', '.join(valid_colormaps)}"
        params['colormap'] = colormap
    
    return params, None

def find_closest_column(query: str, available_columns: list) -> str:
    """
    Find the closest matching column name using fuzzy matching.
    
    Args:
        query (str): The search term
        available_columns (list): List of available column names
        
    Returns:
        str: Best matching column name or None
    """
    query = query.strip()
    
    # Direct match (case-insensitive)
    for col in available_columns:
        if col.lower() == query.lower():
            return col
    
    # Fuzzy matching
    matches = process.extract(
        query, 
        available_columns,
        scorer=fuzz.ratio,  # Use basic ratio matching
        limit=1
    )
    
    if matches and matches[0][1] >= 80:  # Require 80% similarity
        return matches[0][0]
    
    return None

def process_dataset_mention(message: str, datasets: dict) -> str:
    """Extract dataset name from message."""
    message = message.lower()
    for dataset_name in datasets.keys():
        if dataset_name.lower() in message:
            return dataset_name
    return None

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

class VisualizationHandler:
    """Handles different types of visualizations"""
    
    SUPPORTED_TYPES = {'bubble', 'heatmap', 'map', 'network'}  # Types we plan to support
    
    @staticmethod
    def detect_type(message: str) -> str:
        """Detect the type of visualization requested."""
        message = message.lower()

        # Check for heatmap/clustering request
        if any(keyword in message for keyword in ['heatmap', 'heat map', 'clustermap', 'clustering']):
            return 'heatmap'
        
        # Map detection
        if (message.startswith('map') and 
            'latitude=' in message and 
            'longitude=' in message):
            return 'map'
            
        # Bubble plot detection
        if ('plot' in message and 
            any(x in message for x in ['vs', 'versus', 'against'])):
            return 'bubble'
            
        return None

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
        
    if viz_state['type'] == 'bubble':
        try:
            df = pd.DataFrame(viz_state['data'])
            params = viz_state['params']
            debug_messages = []
            
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
                        debug_messages.append(f"Invalid size value '{params['size']}', using default size of 20")
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
                        debug_messages.append(
                            f"'{params['color']}' is neither a valid color specification nor a column name. "
                            "Using default color (blue). Valid options are:\n"
                            "- A column name from the dataset\n"
                            "- A hex color (e.g., #FF5733)\n"
                            "- An RGB color (e.g., rgb(100,150,200))\n"
                            "- A named color (e.g., red, blue, green)"
                        )
            
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
                )
            )
            
            # Add colorbar/legend title if using column for colors
            if isinstance(color_values, pd.Series):
                if color_discrete:
                    fig.update_layout(
                        showlegend=True,
                        legend_title_text=params['color']
                    )
                else:
                    fig.update_layout(
                        coloraxis_colorbar_title=params['color']
                    )
            
            debug_text = "\n".join(debug_messages) if debug_messages else "Plot created successfully"
            
            return (
                html.Div([
                    dcc.Graph(
                        id='bubble-plot',
                        figure=fig,
                        style={'height': '700px'}
                    )
                ]),
                debug_text
            )
            
        except Exception as e:
            error_msg = f"Error creating plot: {str(e)}"
            return html.Div(error_msg, style={'color': 'red'}), error_msg
    elif viz_state['type'] == 'map':
        
        try:
            df = pd.DataFrame(viz_state['data'])
            params = viz_state['params']
            
            # Handle size parameter
            marker_size = 10  # default size
            if params.get('size'):
                if isinstance(params['size'], (int, float)):
                    marker_size = params['size']
                else:
                    # Normalize column values to reasonable marker sizes (5-25)
                    size_values = df[params['size']]
                    marker_size = ((size_values - size_values.min()) / 
                                 (size_values.max() - size_values.min()) * 20 + 5)
            
            # Handle color parameter - mirror bubble plot approach
            marker_color = '#636EFA'  # default color
            colorscale = None
            showscale = False
            legend_traces = []  # Store legend traces for categorical data
            
            if params.get('color'):
                color_param = params['color'].lower()
                if is_valid_color(color_param):
                    marker_color = color_param
                else:
                    # Try to use as column name
                    try:
                        color_values = df[params['color']]
                        if pd.api.types.is_numeric_dtype(color_values):
                            marker_color = color_values
                            colorscale = 'Viridis'
                            showscale = True
                        else:
                            # Generate colors for categorical values
                            unique_values = color_values.unique()
                            color_map = generate_colors(len(unique_values))
                            value_to_color = dict(zip(unique_values, color_map))
                            marker_color = [value_to_color[val] for val in color_values]
                            
                            # Create legend traces
                            for value, color in value_to_color.items():
                                legend_traces.append(
                                    go.Scatter(
                                        x=[None],
                                        y=[None],
                                        mode='markers',
                                        marker=dict(size=10, color=color),
                                        showlegend=True,
                                        name=str(value)
                                    )
                                )
                    except KeyError:
                        print(f"Warning: Color parameter '{params['color']}' not found in columns, using default color")
            
            # Create the main figure with the map
            fig = go.Figure()
            
            # Add the main scattermapbox trace
            fig.add_trace(go.Scattermapbox(
                lat=df[params['latitude']],
                lon=df[params['longitude']],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=marker_color,
                    colorscale=colorscale,
                    showscale=showscale
                ),
                text=df.apply(lambda row: f"Lat: {row[params['latitude']]}<br>Lon: {row[params['longitude']]}" + 
                            (f"<br>Size: {row[params['size']]}" if params.get('size') and isinstance(params['size'], str) else "") +
                            (f"<br>Color: {row[params['color']]}" if params.get('color') else ""),
                            axis=1)
            ))
            
            # Add legend traces for categorical data
            for trace in legend_traces:
                fig.add_trace(trace)
            
            fig.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(
                        lat=df[params['latitude']].mean(),
                        lon=df[params['longitude']].mean()
                    ),
                    zoom=14
                ),
                margin={"r":0,"t":0,"l":0,"b":0},
                showlegend=bool(legend_traces)  # Show legend only if we have categorical data
            )
            
            return [
                html.Div([
                    dcc.Graph(
                        figure=fig,
                        style={'height': '60vh'},
                        config={
                            'scrollZoom': True,
                            'displayModeBar': True
                        }
                    )
                ]),
                f"Map created with {len(df)} points"
            ]
            
        except Exception as e:
            return [
                html.Div([
                    html.P(f"Error creating visualization: {str(e)}", 
                          style={'color': 'red'})
                ]),
                f"Error: {str(e)}"
            ]
    elif viz_state['type'] == 'heatmap':
        print(f"Creating heatmap visualization with parameters: {viz_state['params']}")
        
        try:
            df = pd.DataFrame(viz_state['data'])
            params = viz_state['params']
            
            # Preprocess the data
            data, clustering_info = preprocess_heatmap_data(df, params)
            
            # Create the figure
            fig = go.Figure()
            
            # Determine if we should center the colorscale
            center_scale = params['standardize'] is not None
            if center_scale:
                max_abs = max(abs(data.values.min()), abs(data.values.max()))
                zmin, zmax = -max_abs, max_abs
                colorscale = 'RdBu_r'  # Red-Blue diverging colormap for centered data
            else:
                zmin, zmax = data.values.min(), data.values.max()
                colorscale = params['colormap']
            
            # Add heatmap trace
            heatmap = go.Heatmap(
                z=data.values,
                x=clustering_info['col_labels'],
                y=clustering_info['row_labels'],  # Use clustering_info row labels
                colorscale=colorscale,
                zmid=0 if center_scale else None,
                zmin=zmin,
                zmax=zmax,
                colorbar=dict(
                    title='Standardized Value' if params['standardize'] else 'Value',
                    titleside='right'
                ),
                hoverongaps=False,
                hovertemplate=(
                    'Row: %{y}<br>' +
                    'Column: %{x}<br>' +
                    ('Standardized Value: %{z:.2f}<br>' if params['standardize'] else 'Value: %{z:.2f}<br>') +
                    '<extra></extra>'
                )
            )
            fig.add_trace(heatmap)
            
            # Update layout
            title_parts = []
            if params['standardize']:
                title_parts.append(f"Standardized by {params['standardize']}")
            if params['cluster']:
                title_parts.append(f"Clustered by {params['cluster']}")
            if params['transpose']:
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
                    ticktext=clustering_info['row_labels'],  # Use clustering_info row labels
                    tickvals=list(range(len(clustering_info['row_labels'])))
                ),
                margin=dict(
                    l=100,
                    r=50,
                    t=50,
                    b=100
                )
            )
            
            return [
                html.Div([
                    dcc.Graph(
                        figure=fig,
                        style={'height': '80vh'},
                        config={
                            'displayModeBar': True,
                            'toImageButtonOptions': {
                                'format': 'png',
                                'filename': 'heatmap',
                                'height': 800,
                                'width': 1200,
                                'scale': 2
                            }
                        }
                    )
                ]),
                f"Heatmap created with {data.shape[0]} rows and {data.shape[1]} columns"
            ]
            
        except Exception as e:
            return [
                html.Div([
                    html.P(f"Error creating visualization: {str(e)}", 
                          style={'color': 'red'})
                ]),
                f"Error: {str(e)}"
            ]
    
    return html.Div("Unsupported visualization type"), f"Type: {viz_state['type']}"

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
                        }
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
                        }
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

def test_literature_query_detection():
    """Test cases for literature query detection."""
    test_cases = [
        # Positive cases
        ("What is known about gene regulation?", True, "gene regulation"),
        ("Find papers about CRISPR", True, "CRISPR"),
        ("Search for articles related to metabolic pathways", True, "metabolic pathways"),
        ("Tell me about the research on bacterial growth", True, "bacterial growth"),
        ("What papers discuss protein folding", True, "protein folding"),
        ("Look for literature about DNA repair mechanisms", True, "DNA repair mechanisms"),
        ("Show me research on synthetic biology", True, "synthetic biology"),
        # New test cases for biological queries
        ("Tell me about b. subtilis", True, "tell me about b. subtilis"),
        ("What is Escherichia coli?", True, "escherichia coli"),
        ("Find me papers about transposons", True, "transposons"),
        ("Tell me about the lac operon", True, "lac operon"),
        ("What are plasmids?", True, "plasmids"),
        
        # Negative cases
        ("Plot temperature vs time", False, None),
        ("Create a heatmap", False, None),
        ("What is the average value?", False, None),
        ("Execute the query", False, None),
        ("Convert to dataset", False, None)
    ]
    
    results = []
    for message, expected_is_lit, expected_query in test_cases:
        is_lit, query = is_literature_query(message)
        passed = is_lit == expected_is_lit and (query == expected_query if expected_query else True)
        results.append({
            'message': message,
            'expected': (expected_is_lit, expected_query),
            'got': (is_lit, query),
            'passed': passed
        })
    
    return results

def test_threshold_extraction():
    """Test cases for threshold extraction."""
    test_cases = [
        ("Use threshold 0.3", 0.3),
        ("Set cutoff to 0.7", 0.7),
        ("Apply a threshold of 0.5", 0.5),
        ("threshold 0.1", 0.1),
        ("With score 0.8", 0.8),
        ("Use threshold 1.5", None),  # Invalid - above 1
        ("Set cutoff to -0.1", None),  # Invalid - below 0
        ("No threshold here", None),
        ("Use other settings", None)
    ]
    
    results = []
    for message, expected in test_cases:
        got = extract_threshold_from_message(message)
        passed = got == expected
        results.append({
            'message': message,
            'expected': expected,
            'got': got,
            'passed': passed
        })
    
    return results

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

def serialize_weaviate_object(obj):
    """Helper to convert Weaviate types to serializable format."""
    if hasattr(obj, '__class__') and '_WeaviateUUID' in obj.__class__.__name__:
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_weaviate_object(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_weaviate_object(item) for item in obj]
    return obj

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

def search_all_sources(query: str, threshold: float = 0.6) -> dict:
    """Search across all available data sources for relevant information."""
    results = {
        'dataset_matches': [],
        'database_matches': [],
        'literature_matches': [],  # Add literature matches
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
                results['literature_matches'] = {
                    'results': weaviate_results,
                    'preview': format_weaviate_results_preview(df),
                    'dataframe': df
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
    
    # Search datasets if available
    if text_searcher.fitted:
        try:
            dataset_results = text_searcher.search_text(query, threshold=threshold)
            if dataset_results:
                results['dataset_matches'] = dataset_results
                dataset_match_count = sum(len(match['details']) for match in dataset_results)
                total_matches += dataset_match_count
                results['metadata']['sources_searched'].append('datasets')
                results['metadata']['coverage']['datasets'] = {
                    'total_matches': dataset_match_count,
                    'matched_datasets': [r['source_name'] for r in dataset_results],
                    'total_values': sum(
                        sum(len(details['matches']) for details in match['details'].values())
                        for match in dataset_results
                    )
                }
        except Exception as e:
            print(f"Dataset search error: {str(e)}")
            results['metadata']['errors'] = results['metadata'].get('errors', [])
            results['metadata']['errors'].append(f"Dataset search error: {str(e)}")
    
    # Search database if available and initialized
    try:
        if text_searcher_db and text_searcher_db.fitted:
            db_results = text_searcher_db.search_text(query, threshold=threshold)
            if db_results:
                results['database_matches'] = db_results
                db_match_count = sum(len(match['details']) for match in db_results)
                total_matches += db_match_count
                results['metadata']['sources_searched'].append('database')
                results['metadata']['coverage']['database'] = {
                    'total_matches': db_match_count,
                    'matched_tables': list(set(r['source_name'] for r in db_results)),
                    'total_values': sum(
                        sum(len(details['matches']) for details in match['details'].values())
                        for match in db_results
                    )
                }
    except Exception as e:
        print(f"Database search error: {str(e)}")
        results['metadata']['errors'] = results['metadata'].get('errors', [])
        results['metadata']['errors'].append(f"Database search error: {str(e)}")
    
    # Update total matches
    results['metadata']['total_matches'] = total_matches
    
    return results

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

if __name__ == '__main__':
    # Start the app
    app.run_server(debug=True, host='0.0.0.0', port=8051)
