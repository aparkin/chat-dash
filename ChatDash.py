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
from dash import html, dcc, Input, Output, State, callback, ALL, dash_table, no_update
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
from services import initialize_index_search
from dash.exceptions import PreventUpdate
from services import PreviewIdentifier
from services import ServiceRegistry

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
if False:  # Toggle for development environment
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
    "lbl/cborg-chat:latest", "lbl/cborg-coder:latest", "lbl/cborg-deepthought:latest",
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
    },
    'chat_llm': {  # Add ChatLLM-specific styling
        'backgroundColor': '#f8f9fa',  # Light background
        'border': '1px solid #e9ecef',  # Subtle border
        'borderLeft': '4px solid #28a745',  # Green accent for main chat service
        'maxWidth': '75%',  # Standard width
        'padding': '15px',  # More padding for readability
        'fontFamily': 'system-ui',  # System font for better readability
        'lineHeight': '1.5',  # Improved line height
        'color': '#212529'  # Dark gray text for contrast
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

def get_base_help_message() -> str:
    """Get the base help message for UI and general features."""
    return """Here's what you can do with this chat interface:

📁 **Dataset Management**
- Add datasets by:
  - Dragging files onto the upload region
  - Clicking the upload region to browse files
  - Accepted formats: CSV, TSV, and ZIP files containing these types
- Datasets appear in the dataset browser:
  - Click a dataset to select it for analysis
  - Click the red X to delete a dataset
  - You can download datasets by checking the box next to the dataset and clicking the download button.
- View dataset statistics and profiles in the Statistics tab.

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
- Use natural language to ask questions about your data
- Click the dataset cards to switch between datasets
"""

def get_complete_help_message(service_registry: ServiceRegistry) -> str:
    """Get the complete help message including service documentation.
    
    Args:
        service_registry: The service registry containing all registered services
        
    Returns:
        str: Complete help message with UI help and service documentation
    """
    base_help = get_base_help_message()
    service_help = service_registry.get_help_text()
    
    if service_help:
        return f"{base_help}\n\n🔧 **Available Commands**\n{service_help}"
    return base_help

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
        self.current_db = None  # Track current database path
        
    def update_database(self, db_path, db_manager=None):
        """Update or create index for a database."""
        try:
            # Clear all previous data since we only support one database at a time
            self.table_docs = {}
            self.table_details = {}
            self.current_db = None
            self.fitted = False
            
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
                    key = f"{table_name}.{column}"  # No need for db_path in key anymore
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
            
            # Update current database path
            self.current_db = db_path
            
            # Create search index
            self._create_index()
            
            print(f"\n=== Database Indexing Summary ===")
            print(f"Database: {db_path}")
            print(f"Tables indexed: {len(tables)}")
            print(f"Total columns indexed: {len(self.table_docs)}")
            for key in self.table_docs.keys():
                print(f"- {key}: {self.table_details[key]['total_unique']} unique values")
            
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
            print("Warning: Database searcher not fitted")
            return []
            
        try:
            search_term = query.lower().replace("'", "").replace('"', '').strip()
            print(f"\n=== Database Search Debug ===")
            print(f"Query: '{search_term}'")
            print(f"Threshold: {threshold}")
            print(f"Coverage: {coverage}")
            print(f"Current DB: {self.current_db}")
            print(f"Total indexed columns: {len(self.table_docs)}")
            
            results = []
            
            for table_col, details in self.table_details.items():
                print(f"\nChecking {table_col}...")
                table, column = table_col.split('.')
                matches = []
                
                for value in details['unique_values']:
                    if value is None:
                        continue
                        
                    str_value = str(value).lower()
                    ratio = fuzz.ratio(search_term, str_value)
                    
                    if ratio > threshold * 100 and len(str_value)/len(search_term) > coverage:
                        matches.append((value, ratio))
                        print(f"- Match: '{value}' (score: {ratio})")
                
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
                        'similarities': {m: ratio for m, ratio in matches}  # Keep raw ratio (0-100)
                    }
            
            print(f"\nTotal results: {len(results)}")
            for result in results:
                print(f"\nTable: {result['source_name']}")
                for col, details in result['details'].items():
                    print(f"- Column {col}: {len(details['matches'])} matches")
            
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
                            'similarities': {m: ratio for m, ratio in matches}  # Keep raw ratio (0-100)
                        }
                
                if details:
                    results.append({
                        'source_name': dataset_name,     # Changed from 'dataset'
                        'source_type': 'dataset',        # Added this field
                        'similarity': max(max(d['similarities'].values()) for d in details.values()),
                        'matched_text': f"Found matches in dataset {dataset_name}",
                        'details': details
                    })
            
            return results
                
        except Exception as e:
            return []

# Initialize the text searchers for datasets and databases  
text_searcher = DatasetTextSearch()
text_searcher_db = DatabaseTextSearch()

# Initialize index search service with available searchers
initialize_index_search(text_searcher, text_searcher_db)

####################################
#
# Chat Management Functions
#
####################################

def create_system_message(dataset_info: List[Dict[str, Any]], 
                         search_query: Optional[str] = None,
                         database_structure: Optional[Dict] = None,
                         weaviate_results: Optional[Dict] = None) -> str:
    """Create system message with context from datasets, database, and literature."""

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

    base_message += "You are also involved in an interactive conversation with a user trying to analyze information in the various datasources to which you have access."
    base_message += "You also have accesses to a portion of the chat history as context."
    base_message += "You are able to use the chat history to inform your responses."
    base_message += """
    You should always summarize how much context from chat history you are able to see (number of each type of message)
    In the chat history, you will see messages from the user, other services, and yourself.
    If service messages immediately follow the last user message, you should use the service's response as context.
    You should not repeat that information but rather summarize its relationship to information you have about any datasets or databases that we mention above.
    If the last message is a user message, you should analyze the chat history and any information we have about datasets and the database to inform your response.
    In any case, in your response you should follow these guidelines:
    """
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

    base_message += "\n\nChat History Context:"
    base_message += "\n1. The most recent messages in the chat history contain important context:"
    base_message += "\n   - Pay special attention to service messages that immediately follow user messages"
    base_message += "\n   - Build upon service responses rather than repeating their content"
    base_message += "\n   - Focus on adding value through analysis and connections to other data"
    
    base_message += "\n\n2. When responding to follow-up questions:"
    base_message += "\n   - Check if there's a service message responding to the previous message"
    base_message += "\n   - If yes, use that response as context but don't repeat it"
    base_message += "\n   - If no, proceed with a direct response"
    base_message += "\n   - Connect new requests to previously discussed data or analyses"
    base_message += "\n   - Maintain continuity in multi-step analyses or explorations"
    
    base_message += "\n\n3. Service Message Handling:"
    base_message += "\n   - Service messages contain authoritative responses about queries, data, or operations"
    base_message += "\n   - When you see a service message following a user query:"
    base_message += "\n     * Use it as the primary response to that query"
    base_message += "\n     * Add analysis and insights to the service's results"
    base_message += "\n     * Suggest next steps based on the service's results"
    base_message += "\n     * DO NOT restate what the service has already said"
    base_message += "\n     * DO NOT suggest new queries when service has already returned results"
    base_message += "\n     * For search results, analyze the returned data rather than suggesting new searches"
    base_message += "\n     * CRITICAL: DO NOT suggest python code or any related code. The dataset service LLM will do that for you."
    base_message += "\n     * If you do notice an error in the code, call it out and explain why it is an error."
    
    base_message += "\n\n4. Progressive Analysis:"
    base_message += "\n   - Build upon previous analyses and visualizations"
    base_message += "\n   - Reference previous findings when suggesting new approaches"
    base_message += "\n   - Maintain context when refining or expanding previous queries"
    base_message += "\n   - Focus on adding new insights rather than repeating known information"
    base_message += "\n   - When search results are provided, analyze those results rather than suggesting new searches"
    # Add instructions for handling different types of queries
    base_message += "\n\nWhen responding to the user's message:"

    if has_datasets:
        base_message += "\n- If the query relates to the available datasets, suggest ways to analyze the data"
        base_message += "\n- If you find relevant information in the datasets, include it in your response"
        base_message += "\n- Remember: Datasets are separate from the database and cannot be queried using SQL"
        base_message += "\n- To analyze datasets, suggest the user use the available visualization and analysis tools with specific suggestions"
    
    if has_database:
        base_message += "\n- If you recognize a SQL query, analyze it and suggest improvements if needed"
        base_message += "\n- If you receive a natural language database question, propose an appropriate SQL query"
        base_message += "\n- DO NOT execute SQL queries directly - only suggest them for the user to execute"
        base_message += "\n- DO NOT claim to have run queries unless the user has explicitly executed them"
        base_message += "\n- Ensure all SQL queries are valid for SQLite and don't use extended features"
    else:
        base_message += "\n- DO NOT suggest or reference SQL queries"
        base_message += "\n- Focus on other available data sources and general knowledge"
    
    if has_literature:
        base_message += "\n- You have access to a scientific literature database through Weaviate"
        base_message += "\n- For literature queries, use the available search functionality"
        base_message += "\n- When referencing literature results, use the [ID] format"
    
    base_message += "\n- You can combine available data sources with general knowledge"
    base_message += "\n- If no specific data is found, provide a helpful response using your general knowledge"
    base_message += "\n- NEVER suggest querying data sources that are not currently connected"
    base_message += "\n- NEVER claim to have executed queries or retrieved data unless explicitly done by the user"

    if has_database:
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
        base_message += '\n   - "search." or "query."to run the primary (original) query'
        base_message += '\n   - "search|query query_ID" to run a specific query'
        base_message += '\n   - "convert query_ID to dataset" to save results'

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
        # Special handling for ChatLLM service
        if message['service'] == 'chat_llm':
            style = CHAT_STYLES['chat_llm']
        else:
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
     Output('dataset-tabs', 'active_tab', allow_duplicate=True),
     Output('viz-state-store', 'data'),
     Output('chat-loading-output', 'children', allow_duplicate=True),
     Output('successful-queries-store', 'data', allow_duplicate=True),
     Output('datasets-store', 'data', allow_duplicate=True),
     Output('dataset-list', 'children', allow_duplicate=True)],
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

    try:
        if not input_value:
            return (dash.no_update,) * 9  # Updated for all outputs
            
        # Initialize return values
        chat_input_value = dash.no_update
        active_tab = dash.no_update
        viz_state = dash.no_update
        chat_loading = ""
        dataset_list = dash.no_update
            
        chat_history = chat_history or []
        current_message = {'role': 'user', 'content': input_value.strip()}
        context = get_relevant_context(current_message, chat_history)

        # Handle help request
        if input_value.lower().strip() in ["help", "help me", "what can i do?", "what can i do", "what can you do?", "what can you do"]:
            chat_history.append(current_message)
            chat_history.append({
                'role': 'assistant',
                'content': get_complete_help_message(service_registry)
            })
            return (
                create_chat_elements_batch(chat_history),
                '',  # Clear input after help
                chat_history,
                dash.no_update,
                dash.no_update,
                "",
                dash.no_update,  # No store update needed
                dash.no_update,  # No datasets update needed
                dash.no_update   # No dataset list update needed
            )

        # Add message to chat history first
        chat_history.append(current_message)

        # Initialize store updates
        store_updates = {
            'successful_queries_store': successful_queries,
            'datasets_store': datasets
        }
        
        # Detect and execute service handlers
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
                # Parse request
                params = service.parse_request(input_value)
                
                # Create execution context
                context = {
                    'datasets_store': datasets,
                    'successful_queries_store': successful_queries,
                    'selected_dataset': selected_dataset,
                    'database_state': database_state,
                    'database_structure': database_structure_store,
                    'chat_history': chat_history,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Execute service
                response = service.execute(params, context)
                
                # Add service messages to chat
                for msg in response.messages:
                    chat_history.append(msg.to_chat_message())
                
                # Update stores immediately after service execution
                if response.store_updates:
                    if 'successful_queries_store' in response.store_updates:
                        successful_queries = {
                            **successful_queries,
                            **response.store_updates['successful_queries_store']
                        }
                    if 'datasets_store' in response.store_updates:
                        datasets = {
                            **datasets,
                            **response.store_updates['datasets_store']
                        }
                        # Update dataset list when datasets store changes
                        dataset_list = [create_dataset_card(name, data) for name, data in datasets.items()]
                
                # Update return values from service response
                chat_input_value = response.state_updates.get('chat_input', dash.no_update)
                active_tab = response.state_updates.get('active_tab', dash.no_update)
                viz_state = response.state_updates.get('viz_state', dash.no_update)

            except Exception as e:
                print(f"Service execution error in handle_chat_message: {str(e)}")

                error_msg = ServiceMessage(
                    service=service_name,
                    content=f"Error executing service: {str(e)}",
                    message_type="error"
                )
                
                chat_history.append(error_msg.to_chat_message())

        if False:
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

            # Use full chat history for context
            messages = get_context_messages(system_message, chat_history)
            
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
        
        # Test ChatLLM service in parallel
        try:
            chat_llm = service_registry.get_service("chat_llm")
            if chat_llm:
                llm_response = chat_llm.execute(
                    {'message': input_value},
                    {
                        'chat_history': chat_history,
                        'datasets_store': datasets,
                        'selected_dataset': selected_dataset,
                        'database_state': database_state,
                        'database_structure': database_structure_store,
                        'model': model  # Add the selected model to context
                    }
                )
                # Add response to chat history without test marker
                for msg in llm_response.messages:
                    chat_history.append(msg.to_chat_message())
        except Exception as e:
            print(f"ChatLLM service test error: {str(e)}")
            print(traceback.format_exc())
        
        return (
            create_chat_elements_batch(chat_history),
            chat_input_value,
            chat_history,
            active_tab,
            viz_state,
            chat_loading,
            successful_queries,
            datasets,
            dataset_list
        )

    except Exception as e:
        print(f"Error in handle_chat_message: {str(e)}")
        print(traceback.format_exc())
        return (dash.no_update,) * 9

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
                dcc.Graph(
                    id='viz-container',
                    figure={'data': [], 'layout': {'title': 'No visualization selected'}},
                    style={'height': '800px', 'width': '100%'},
                    config={
                        'scrollZoom': True,
                        'displaylogo': False,
                        'responsive': True,
                        'showAxisDragHandles': True,
                        'showAxisRangeEntryBoxes': True,
                        'doubleClick': 'reset+autosize',
                        'displayModeBar': True,
                        'modeBarButtonsToAdd': [
                            'pan', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
                            'autoScale2d', 'resetScale2d', 'drawline', 'drawopenpath',
                            'drawclosedpath', 'eraseshape'
                        ],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': 'plot',
                            'scale': 3
                        },
#                        'hovermode': 'closest',
                        'displayModeBar': 'hover'
                    }
                ),
                html.Pre(id='viz-debug', style={'display': 'none'})  # Keep debug area but hide it
            ], style={'height': '800px', 'width': '100%', 'position': 'relative'})
            
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

    # Temporarily disable query execution while testing service
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
                dash.no_update,  # Changed from '' to dash.no_update
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
            dash.no_update,  # Changed from '' to dash.no_update
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
                dash.no_update,  # Changed from '' to dash.no_update
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
                dash.no_update,  # Changed from '' to dash.no_update
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
                dash.no_update,  # Changed from '' to dash.no_update
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
        return (
            dash.no_update,                           # chat-history
            dash.no_update,  # Changed from '' to dash.no_update
            dash.no_update,                           # chat-store
            dash.no_update,                           # dataset-tabs
            dash.no_update,                           # viz-state-store
            "",                                       # chat-loading-output
            successful_queries,                       # successful-queries-store
            dash.no_update,                           # datasets-store
            dash.no_update                            # dataset-list
        )
        
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
            dash.no_update,  # Changed from '' to dash.no_update
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
            dash.no_update,  # Changed from '' to dash.no_update
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
            dash.no_update,  # Changed from '' to dash.no_update
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
        'content': get_complete_help_message(service_registry)
    })
    
    return dash.no_update, chat_history, create_chat_elements_batch(chat_history)

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

# Move all callbacks before main
@callback(
    [Output('viz-container', 'figure'),
     Output('viz-debug', 'children')],
    Input('viz-state-store', 'data')
)
def update_visualization(viz_state):
    """Update visualization when state changes."""
    if not viz_state:
        return {'data': [], 'layout': {'title': 'No visualization selected'}}, "No visualization state"
    
    if 'error' in viz_state:
        return {'data': [], 'layout': {'title': f"error creating visualization: {viz_state['error']}"}}, f"Error: {viz_state['error']}"
    
    if 'figure' not in viz_state:
        return {'data': [], 'layout': {'title': 'No figure in visualization state'}}, "No figure in state"
    
    return viz_state['figure'], f"Updated figure from state: {list(viz_state.keys())}"

# Add callback to handle figure state updates
@callback(
    Output('viz-state-store', 'data', allow_duplicate=True),
    [Input('viz-container', 'relayoutData'),
     Input('viz-container', 'figure')],
    State('viz-state-store', 'data'),
    prevent_initial_call=True
)
def update_figure_state(relayout_data, figure_data, viz_state):
    """Store figure view state when user interacts with the plot."""
    if not viz_state:
        raise PreventUpdate
        
    # Initialize view settings if not present
    if not viz_state.get('view_settings'):
        viz_state['view_settings'] = {}
    
    # Update from relayoutData
    if relayout_data:
        # Store relevant view settings (zoom, center, etc.)
        for key in relayout_data:
            if any(k in key for k in ['zoom', 'center', 'range', 'domain', 'shapes']):
                viz_state['view_settings'][key] = relayout_data[key]
    
    # Update from figure data
    if figure_data and 'layout' in figure_data:
        # Store shapes from the figure
        if 'shapes' in figure_data['layout']:
            viz_state['view_settings']['shapes'] = figure_data['layout']['shapes']
    
    return viz_state

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
            
        # Initialize or update text search
        print(f"Indexing database: {db_path}")
        global text_searcher_db
        if text_searcher_db is None:
            text_searcher_db = DatabaseTextSearch()
        text_searcher_db.update_database(db_path)
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
    """Generate a unique query ID with timestamp using PreviewIdentifier.
    
    Args:
        is_original (bool): If True, generates ID for primary query
        alt_number (int, optional): For alternative queries, specifies which alternative (1,2,etc)
        
    Returns:
        str: Query ID in format query_YYYYMMDD_HHMMSS_(orig|altN)
    """
    if is_original:
        return PreviewIdentifier.create_id(prefix="query")
    else:
        # For alternatives, we need to create based on the original ID
        original_id = PreviewIdentifier.create_id(prefix="query")
        # Then create alternatives from it
        for _ in range(alt_number or 1):
            original_id = PreviewIdentifier.create_id(previous_id=original_id)
        return original_id

def add_query_ids_to_response(response: str) -> str:
    """Add service-specific IDs to content blocks in LLM response.
    
    This function:
    1. Checks each registered service
    2. If the service finds its content type in the response,
       lets the service add its IDs to those blocks
    
    This allows each service to handle its own content type and ID format.
    """
    modified_response = response
    
    # Let each service process its content blocks
    for service_name, service in service_registry._services.items():
        if service.detect_content_blocks(modified_response):
            modified_response = service.add_ids_to_blocks(modified_response)
            
    return modified_response

def get_api_response(model: str, system_msg: str, messages: list) -> dict:
    """Get response from API with proper error handling.
    
    Args:
        model: Model to use for completion
        system_msg: System message with context
        messages: List of conversation messages
        
    Returns:
        Dict with response content and metadata
    """
    try:
        # Prepare messages with system context
        api_messages = [
            {'role': 'system', 'content': system_msg}
        ]
        api_messages.extend(messages)
        
        # Get completion
        response = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Extract response
        content = response.choices[0].message.content
        
        return {
            'role': 'assistant',
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"API call error: {str(e)}")
        return {
            'role': 'system',
            'content': f"Error calling API: {str(e)}",
            'type': 'error'
        }

def count_tokens(text: str) -> int:
    """Rough estimate of token count. Each word is approximately 1.3 tokens."""
    return int(len(text.split()) * 1.3)

def get_context_messages(system_message: str, chat_history: list, max_tokens: int = 8000) -> list:
    """Get context messages within token limit, prioritizing critical messages.
    
    Args:
        system_message: The system prompt
        chat_history: Full chat history
        max_tokens: Maximum tokens to allow
        
    Returns:
        List of messages within token limit, in chronological order
    """
    # First count system message tokens
    system_tokens = count_tokens(system_message)
    
    # Reserve tokens for system message and buffer
    available_tokens = max_tokens - system_tokens - 500  # 500 token buffer
    
    if available_tokens <= 0:
        print(f"Warning: System message is too long ({system_tokens} tokens)")
        return [{'role': 'system', 'content': system_message}]
    
    # Initialize with system message
    messages = [{'role': 'system', 'content': system_message}]
    token_count = system_tokens
    
    # Find most recent service message and user request
    recent_service_msg = None
    recent_user_msg = None
    
    for msg in reversed(chat_history):
        if not recent_service_msg and msg.get('service'):
            recent_service_msg = msg
            continue
        if not recent_user_msg and msg['role'] == 'user':
            recent_user_msg = msg
            if recent_service_msg:  # If we have both, stop looking
                break
    
    # Add recent service message if exists
    if recent_service_msg:
        service_tokens = count_tokens(recent_service_msg.get('content', ''))
        if token_count + service_tokens <= max_tokens - 500:
            messages.append({
                'role': 'assistant',
                'content': recent_service_msg.get('content', '')
            })
            token_count += service_tokens
            print(f"Added service message: {service_tokens} tokens")
    
    # Add user's request if exists
    if recent_user_msg:
        user_tokens = count_tokens(recent_user_msg.get('content', ''))
        if token_count + user_tokens <= max_tokens - 500:
            messages.append({
                'role': 'user',
                'content': recent_user_msg.get('content', '')
            })
            token_count += user_tokens
            print(f"Added user message: {user_tokens} tokens")
    
    # Add other recent context if space remains
    remaining_tokens = max_tokens - token_count - 500
    if remaining_tokens > 0:
        for msg in reversed(chat_history):
            # Skip messages we already added
            if msg == recent_service_msg or msg == recent_user_msg:
                continue
                
            msg_tokens = count_tokens(msg.get('content', ''))
            if token_count + msg_tokens > max_tokens - 500:
                break
                
            # Format service messages
            if 'service' in msg:
                content_parts = msg['content'].split('\n\n', 1)
                content = content_parts[1].strip() if len(content_parts) > 1 else msg['content']
                messages.append({
                    'role': 'assistant',
                    'content': content
                })
            else:
                messages.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', '')
                })
            token_count += msg_tokens
    
    # Debug output
    print(f"\n=== Context Window Stats ===")
    print(f"System message tokens: {system_tokens}")
    print(f"History tokens: {token_count - system_tokens}")
    print(f"Total tokens: {token_count}/{max_tokens}")
    print(f"Messages included: {len(messages)}")
    print(f"Message types: {[msg['role'] for msg in messages]}")
    
    return messages

if __name__ == '__main__':
    # Start the app
    app.run_server(debug=True, host='0.0.0.0', port=8051)

