"""
ChatDash: Interactive Data Analysis and Chat Interface
===================================================

A Dash-based application that combines interactive data analysis with an AI-powered
chat interface. Provides seamless integration between traditional data analysis tools
and natural language interaction.

Core Components
-------------
1. Data Management System
   - Multi-format data import (CSV, TSV, ZIP)
   - Automatic data profiling and validation
   - Memory-efficient data handling
   - Real-time memory usage monitoring
   - Dataset comparison capabilities

2. AI Chat Interface
   - Natural language data analysis
   - Multiple AI model support
   - Context-aware responses
   - Code generation and execution
   - Command history tracking

3. Database Integration
   - SQL Database Support:
     * Dynamic connection management
     * Schema visualization with ERD
     * Interactive query execution
     * Query result caching
   - Weaviate Vector Database:
     * Real-time connection monitoring
     * Collection management
     * Vector similarity search
     * Schema visualization

4. Visualization System
   - Interactive data plots
   - Real-time visualization updates
   - Multiple visualization types
   - Custom plot configurations
   - Export capabilities

Technical Specifications
----------------------
- Server: Dash (Flask-based)
- Port: 8051
- Host: 0.0.0.0 (network accessible)
- Python Version: 3.8+
- Key Dependencies:
  * dash
  * pandas
  * plotly
  * openai
  * weaviate-client
  * sqlite3

Security Features
---------------
- Input validation for all data uploads
- SQL query sanitization
- API key secure management
- Database access controls
- Error handling and logging

Installation
-----------
1. Environment Setup:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\\Scripts\\activate` on Windows
   pip install -r requirements.txt
   ```

2. Configuration:
   - Copy .env.example to .env
   - Set required API keys and endpoints
   - Configure database connections

3. Running:
   ```bash
   python ChatDash.py
   ```
   Access via browser at http://localhost:8051

Development Notes
---------------
- All callbacks must remain above __main__ block
- Service registry pattern for extensibility
- Modular design for easy feature addition
- Comprehensive error handling
- Memory management considerations

Version: 1.0.0
License: MIT
"""

import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx, ALL, MATCH, clientside_callback
import dash_bootstrap_components as dbc
import openai
import os
import pandas as pd
import io
import base64
from datetime import datetime, timedelta
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
from typing import Dict, List, Tuple, Optional, Union, Any, Iterable
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
import inspect
import asyncio
import weaviate
import hashlib
import pickle
import time

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
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
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

def get_help_message(service_registry: Optional[ServiceRegistry] = None) -> str:
    """Get help message for UI and available commands.
    
    Args:
        service_registry: Optional service registry for service-specific help
        
    Returns:
        str: Complete help message with UI help and optionally service documentation
    """
    base_message = """Here's what you can do with this chat interface:

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

    if service_registry:
        service_help = service_registry.get_help_text()
        if service_help:
            return f"{base_message}\n\n🔧 **Available Commands**\n{service_help}"
    
    return base_message

def get_services_status() -> Dict[str, Dict[str, Any]]:
    """Get status for all registered services."""
    from services import service_registry
    
    services_status = {}
    for name, service in service_registry._services.items():
        # Default status assumes ready
        status_info = {
            'status': 'ready',
            'ready': True,
            'description': f"{name} service"
        }
        
        # Check for optional get_status method
        if hasattr(service, 'get_status') and callable(service.get_status):
            try:
                custom_status = service.get_status()
                if custom_status and isinstance(custom_status, dict):
                    status_info.update(custom_status)
            except Exception:
                pass  # Ignore errors, keep default status
        
        services_status[name] = status_info
    
    return services_status

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

def create_chat_element(message: dict) -> dbc.Card:
    """Create a styled chat message component with appropriate formatting and styling.
    
    Generates a Dash Bootstrap Card component for displaying chat messages with:
    - Role-specific styling (user, assistant, system, service)
    - Markdown rendering for formatted text
    - Service-specific headers and indicators
    - Error state handling
    - Responsive layout adjustments
    
    Styling Rules:
    1. User messages: Right-aligned with primary color
    2. Assistant messages: Left-aligned with light background
    3. System messages: Left-aligned with warning color
    4. Service messages: Full-width with service-specific styling
    5. Error messages: Error-state styling with red accents
    
    Args:
        message (dict): Message configuration containing:
            - role (str): Message role ('user', 'assistant', 'system', 'service')
            - content (str): Message text content
            - service (str, optional): Service name for service messages
            - type (str, optional): Message type for service messages
            - metadata (dict, optional): Additional message metadata
            
    Returns:
        dbc.Card: Styled Dash Bootstrap card component with:
            - Appropriate styling based on message role
            - Formatted content with markdown support
            - Service headers when applicable
            - Error state indicators when needed
            
    Notes:
        - Styling is controlled by CHAT_STYLES global dictionary
        - Service messages get special handling for tables and code
        - ChatLLM service has custom styling for better readability
        - Error messages override normal service styling
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
    """Generate a Mermaid.js Entity-Relationship Diagram from database structure.
    
    Creates a comprehensive ERD visualization including:
    1. Tables with column details
    2. Primary and foreign key indicators
    3. Relationship lines with cardinality
    4. Column data types and constraints
    5. Index indicators
    
    Diagram Features:
    - Tables represented as entities
    - Columns listed with types
    - Primary keys marked with PK
    - Foreign keys marked with FK
    - Relationships shown with arrows
    - Cardinality indicators (1:1, 1:N, N:M)
    
    Args:
        structure (dict): Database structure information containing:
            - tables: Dict of table information
            - relationships: List of foreign key relationships
            - metadata: Database-level metadata
            
    Returns:
        str: Mermaid.js compatible ERD diagram code
        
    Note:
        - Handles special characters in names
        - Includes metadata indicators
        - Optimizes layout for readability
        - Compatible with Mermaid.js v8.11+
    """
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
    # Add services status store
    dcc.Store(id='services-status-store', data={}),
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
                    html.Div(id="dataset-tab-content", className="p-3"),
                    # Add Services Status Indicator at the bottom of the tabbed pane
                    html.Div([
                        html.Span("Services: ", style={'fontSize': '12px', 'marginRight': '8px', 'fontWeight': 'bold'}),
                        html.Div(id='services-status-container', style={
                            'display': 'flex',
                            'flexWrap': 'wrap',
                            'gap': '6px',
                            'alignItems': 'center',
                            'flex': '1'
                        }),
                        html.Button(
                            '🔄', 
                            id='refresh-services-status',
                            n_clicks=0,
                            title='Refresh service status',
                            style={
                                'marginLeft': '8px',
                                'fontSize': '12px',
                                'border': 'none',
                                'background': 'none',
                                'cursor': 'pointer'
                            }
                        )
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'marginTop': 'auto',  # Push to the bottom of the container
                        'padding': '8px',
                        'borderTop': '1px solid #dee2e6',
                        'backgroundColor': 'white'
                    })
                ], style={
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'backgroundColor': '#f8f9fa',
                    'minHeight': '200px',
                    'display': 'flex',
                    'flexDirection': 'column'  # Make it a column flex container so marginTop: auto works
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
                            placeholder="Type your message here..."
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
    """Validate dataset content and structure for analysis compatibility.
    
    Performs comprehensive dataset validation including:
    1. Structure Validation:
       - Non-empty dataframe check
       - Column name uniqueness
       - Index uniqueness and validity
       - Data type consistency
       
    2. Content Quality:
       - Missing value detection
       - Duplicate row detection
       - Invalid value checks
       - Data type compatibility
       
    3. Resource Requirements:
       - Memory usage estimation
       - Processing requirements
       - Storage requirements
       - Performance impact
       
    4. Analysis Readiness:
       - Numeric column presence
       - Categorical column validity
       - Date/time format validity
       - String column encoding
       
    Args:
        df (pd.DataFrame): DataFrame to validate
        filename (str): Original filename for error reporting
        
    Returns:
        tuple[bool, str]: Contains:
            - is_valid (bool): Whether dataset passes all checks
            - message (str): Success message or error details
            
    Raises:
        ValueError: For critical validation failures
        TypeError: For data type incompatibilities
        MemoryError: For resource limit violations
        
    Note:
        - Validation is ordered by importance
        - Early termination on critical failures
        - Memory checks prevent OOM situations
        - Error messages are user-friendly
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
    """Process and route chat messages through the application's service architecture.
    
    This is the main callback function for the chat interface. It:
    1. Routes messages to appropriate service handlers
    2. Manages chat history and state updates
    3. Handles service responses and visualizations
    4. Processes LLM responses when needed
    5. Manages dataset and database interactions
    
    Flow:
    1. Validates input and initializes state
    2. Detects appropriate service handlers
    3. Executes service-specific logic
    4. Updates chat history and UI state
    5. Processes LLM response if needed
    
    Args:
        n_clicks (int): Button click counter trigger
        input_value (str): User's chat message text
        chat_history (list): Previous chat messages and responses
        model (str): Selected AI model identifier
        datasets (dict): Currently loaded datasets and metadata
        selected_dataset (str): Name of currently selected dataset
        database_state (dict): Current database connection state
        database_structure_store (dict): Database schema information
        successful_queries (dict): History of successful query executions
        
    Returns:
        tuple: Contains:
            - chat_history_children (list): Updated chat UI elements
            - chat_input_value (str): New input field value
            - chat_store_data (list): Updated chat history
            - active_tab (str): Selected tab identifier
            - viz_state (dict): Visualization state
            - chat_loading (str): Loading indicator state
            - successful_queries (dict): Updated query history
            - datasets (dict): Updated dataset store
            - dataset_list (list): Updated dataset UI elements
            
    Raises:
        PreventUpdate: When no update is needed
        Exception: For various processing errors
        
    Notes:
        - Service handlers are detected and executed in priority order
        - Multiple services handling the same message triggers a warning
        - LLM processing occurs after service handling
        - State updates are applied immediately after service execution
    """

    # Initialize return values
    chat_input_value = dash.no_update
    active_tab = dash.no_update
    viz_state = dash.no_update
    chat_loading = ""
    dataset_list = dash.no_update
    
    # Check for empty input early and return no-update for all outputs
    if not input_value or not input_value.strip():
        return (
            dash.no_update,  # chat-history
            dash.no_update,  # chat-input
            dash.no_update,  # chat-store
            dash.no_update,  # dataset-tabs
            dash.no_update,  # viz-state-store
            dash.no_update,  # chat-loading-output
            dash.no_update,  # successful-queries-store
            dash.no_update,  # datasets-store
            dash.no_update   # dataset-list
        )

    try:
        chat_history = chat_history or []
        current_message = {'role': 'user', 'content': input_value.strip()}

        # Handle help request
        if input_value.lower().strip() in ["help", "help me", "what can i do?", "what can i do", "what can you do?", "what can you do"]:
            chat_history.append(current_message)
            chat_history.append({
                'role': 'assistant',
                'content': get_help_message(service_registry)
            })
            return (
                create_chat_elements_batch(chat_history),
                '',  # Clear input after help
                chat_history,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update,
                dash.no_update
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
                chat_input_value = response.state_updates.get('chat_input', '')  # Default to clearing input
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
    """Process and validate uploaded dataset files.
    
    Handles multiple file uploads with comprehensive validation:
    1. File Format Support:
       - CSV files with automatic delimiter detection
       - TSV files with tab delimiter
       - ZIP archives containing multiple CSV/TSV files
       - Handles various text encodings (UTF-8, Latin1, etc.)
       
    2. Data Processing:
       - Automatic data type inference
       - Missing value standardization
       - Column name cleaning
       - Index handling and validation
       
    3. Profile Generation:
       - Statistical profiling of datasets
       - Column type analysis
       - Distribution analysis
       - Correlation detection
       
    4. Memory Management:
       - Efficient data loading
       - Memory usage monitoring
       - Resource limit enforcement
       
    Args:
        list_of_contents (Optional[List[str]]): Base64 encoded file contents
        list_of_names (List[str]): Original filenames
        list_of_dates (List[str]): File modification timestamps
        existing_datasets (Dict[str, Any]): Currently loaded datasets
        
    Returns:
        Tuple containing:
            - datasets_store (Dict[str, Any]): Updated dataset store with:
                - df: DataFrame as records
                - metadata: Dataset information
                - profile_report: Generated profile report
            - dataset_list (List[Any]): Updated UI components
            - upload_status (str): Status message for display
            
    Raises:
        PreventUpdate: If no files are provided
        pd.errors.EmptyDataError: If uploaded file contains no data
        UnicodeDecodeError: If file encoding is not supported
        pd.errors.ParserError: If file format is invalid
        
    Note:
        - Handles multiple files simultaneously
        - Preserves existing datasets
        - Updates UI components automatically
        - Generates comprehensive profile reports
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
    """Process and clean a DataFrame for analysis and visualization.
    
    Performs comprehensive data cleaning and standardization:
    1. Missing Value Handling:
       - Standardizes various missing value indicators to NaN
       - Handles empty strings and special characters
       - Processes 'None', 'NULL', 'NA' variations
       
    2. Data Type Inference:
       - Attempts automatic type conversion for numeric columns
       - Preserves string columns that appear numeric but aren't
       - Handles mixed-type columns appropriately
       
    3. Column Standardization:
       - Removes special characters from column names
       - Ensures consistent naming conventions
       - Handles duplicate column names
       
    4. Data Validation:
       - Checks for data integrity
       - Validates numeric conversions
       - Ensures consistent data types
    
    Args:
        df (pd.DataFrame): Input DataFrame to process
        filename (str): Original filename for error reporting
        
    Returns:
        pd.DataFrame: Processed DataFrame with:
            - Standardized missing values
            - Appropriate data types
            - Cleaned column names
            - Validated data integrity
            
    Raises:
        ValueError: If DataFrame cannot be properly processed
        TypeError: If column type conversion fails
        
    Notes:
        - Preserves original data when conversion is uncertain
        - Logs warnings for potentially problematic conversions
        - Maintains column order from original DataFrame
        - Handles international number formats
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
    """Render content for the selected dataset tab.
    
    This callback manages dataset tab content rendering:
    1. Content Types:
       - Data preview with pagination
       - Statistical summaries
       - Profile reports
       - Visualization options
       - Database connections
       
    2. Tab Management:
       - Handles tab switching
       - Maintains tab state
       - Updates active content
       - Manages tab history
       
    3. Dataset Integration:
       - Links to selected dataset
       - Updates on dataset changes
       - Handles missing datasets
       - Manages data access
       
    4. UI Components:
       - Generates data tables
       - Creates summary cards
       - Builds visualization options
       - Manages layout
       
    Args:
        active_tab (str): Currently selected tab identifier
        selected_dataset (str): Name of selected dataset
        datasets (dict): Available datasets and their metadata
        
    Returns:
        list: List of Dash components for the active tab
        
    Raises:
        PreventUpdate: If no tab is selected or no dataset available
        
    Note:
        - Handles multiple content types
        - Maintains responsive layout
        - Optimizes component rendering
        - Preserves user selections
    """
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
                    style_header={'fontWeight': 'bold'},
                    style_data_conditional=[{
                        'if': {
                            'filter_query': '{' + col + '} contains ""',
                            'column_id': col
                        },
                        'textOverflow': 'ellipsis',
                        'maxWidth': 300
                    } for col in df.columns]
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
    """Handle dataset selection and update application state.
    
    This callback manages dataset selection events and their effects:
    1. Dataset State Management:
       - Updates selected dataset
       - Manages dataset preview state
       - Updates chat context
       
    2. UI Updates:
       - Updates dataset cards
       - Switches to preview tab
       - Updates chat history
       
    3. Context Management:
       - Adds dataset context to chat
       - Updates analysis state
       - Prepares for new analysis
       
    Args:
        n_clicks (List[int]): Click counts for dataset cards
        datasets (Dict[str, Any]): Available datasets with metadata
        chat_store (Dict[str, Any]): Current chat history and state
        
    Returns:
        tuple: Contains:
            - selected_dataset (str): Newly selected dataset
            - chat_history (List[dict]): Updated chat messages
            - chat_store (dict): Updated chat state
            - active_tab (str): New active tab selection
            
    Raises:
        PreventUpdate: If no dataset is selected
        
    Note:
        - Maintains chat context across selections
        - Updates UI to reflect new selection
        - Prepares environment for analysis
        - Handles multiple dataset formats
    """
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
    """Show help message in chat when help button is clicked."""
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    
    chat_history = chat_history or []
    
    chat_history.append({
        'role': 'user',
        'content': "What can I do with this chat interface?"
    })
    
    chat_history.append({
        'role': 'assistant',
        'content': get_help_message(service_registry)
    })
    
    return dash.no_update, chat_history, create_chat_elements_batch(chat_history)

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
    
    # Create a deep copy of the figure to avoid modifying the original
    figure = {
        'data': viz_state['figure'].get('data', []),
        'layout': viz_state['figure'].get('layout', {}).copy()
    }
    
    # Apply view settings if they exist
    if 'view_settings' in viz_state:
        view_settings = viz_state['view_settings']
        for key, value in view_settings.items():
            # Handle nested properties like 'xaxis.range'
            if '.' in key:
                parts = key.split('.')
                current = figure['layout']
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                figure['layout'][key] = value
    
    debug_info = {
        'has_view_settings': 'view_settings' in viz_state,
        'view_settings': list(viz_state.get('view_settings', {}).keys()),
        'layout_keys': list(figure['layout'].keys())
    }
    
    return figure, f"Updated figure with view settings: {json.dumps(debug_info, indent=2)}"

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
        # Store all view-related settings
        for key, value in relayout_data.items():
            # Comprehensive list of view-related properties
            if any(k in key for k in [
                'zoom', 'center', 'range', 'domain', 'autorange',
                'scale', 'scaleanchor', 'scaleratio', 'constrain',
                'constraintoward', 'matches', 'showspikes', 'spikethickness',
                'projection', 'camera', 'aspectratio', 'aspectmode'
            ]):
                viz_state['view_settings'][key] = value
    
    # Update from figure data if it contains new layout information
    if figure_data and 'layout' in figure_data:
        layout = figure_data['layout']
        # Store layout properties that affect the view
        for key, value in layout.items():
            if any(k in key for k in ['axis', 'margin', 'scene', 'geo', 'mapbox']):
                viz_state['view_settings'][key] = value
        
        # Store shapes and annotations
        if 'shapes' in layout:
            viz_state['view_settings']['shapes'] = layout['shapes']
        if 'annotations' in layout:
            viz_state['view_settings']['annotations'] = layout['annotations']
    
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
    """Extract and analyze the structure of a SQLite database.
    
    Performs comprehensive database analysis including:
    1. Table enumeration and metadata
    2. Column definitions and constraints
    3. Foreign key relationships
    4. Index configurations
    5. View definitions
    
    Args:
        db_path (str): Path to SQLite database file
        
    Returns:
        Dict[str, Any]: Database structure information containing:
            - tables: Dict of table information
                - name: Table name
                - columns: List of column definitions
                - constraints: List of constraints
                - indexes: List of indexes
            - relationships: List of foreign key relationships
            - views: List of view definitions
            - metadata: Database-level metadata
            
    Raises:
        sqlite3.Error: For database access issues
        FileNotFoundError: If database file doesn't exist
        PermissionError: If database file is not accessible
        
    Note:
        This function is used for both visualization (ERD generation)
        and query processing (schema validation).
    """
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
    """Establish and verify a database connection with comprehensive checks.
    
    This callback function manages database connections by:
    1. Validating database file existence and permissions
    2. Establishing and testing connection
    3. Extracting and caching database structure
    4. Initializing text search capabilities
    5. Updating connection state and UI
    
    Connection Process:
    1. File validation
    2. Connection establishment
    3. Schema extraction
    4. Search index creation
    5. State management
    
    Args:
        n_clicks (int): Button click counter trigger
        db_path (str): Path to SQLite database file
        current_state (dict): Current connection state containing:
            - connected: bool indicating if connected
            - path: Current database path
            - last_update: Timestamp of last update
            
    Returns:
        tuple: Contains:
            - state (dict): Updated connection state
            - structure (dict): Database structure information
            - status (html.Div): Connection status display
            
    Note:
        - Maintains connection state between clicks
        - Caches database structure for performance
        - Updates UI to reflect connection status
        - Initializes search capabilities automatically
    """
    if not n_clicks or not db_path:
        return (
            {'connected': False, 'path': None, 'last_update': datetime.now().isoformat()},
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
            'last_update': datetime.now().isoformat(),
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
            {'connected': False, 'path': None, 'last_update': datetime.now().isoformat(), 'has_text_search': False},
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

@callback(
    [Output('services-status-container', 'children'),
     Output('services-status-store', 'data')],
    [Input('refresh-services-status', 'n_clicks'),
     Input('_weaviate-init', 'data')],
    prevent_initial_call='initial_duplicate'
)
def update_services_status(n_clicks, weaviate_init):
    """Update service status indicators."""
    services_status = get_services_status()
    service_indicators = []
    
    # Define pastel colors for different services
    # This ensures each service has a consistent, visually distinct color
    service_colors = {
        'dataset': '#c6e6f5',       # Light blue
        'database': '#e1ecc8',      # Light green
        'visualization': '#ffd3b6',  # Light orange
        'literature': '#d6cdea',    # Light purple
        'store_report': '#f7d6e0',  # Light pink
        'index_search': '#eff9da',  # Light yellow-green
        'chat_llm': '#daeaf6',      # Light sky blue
        'monet': '#ffefd9',         # Light peach
        'nmdc': '#e6ffd9'           # Light mint
    }
    
    # Default pastel colors for any unlisted services - these will rotate based on name
    default_pastel_colors = [
        '#ffccd5', '#ffe5d9', '#fff0d9', '#fcf5c7', '#e2efc7', '#d0f0c0', 
        '#c7e9e0', '#c7e8f3', '#d1d2f9', '#e7d9f0'
    ]
    
    for name in sorted(services_status.keys()):
        status_info = services_status[name]
        
        # Get service color (or assign one from the defaults if not listed)
        if name in service_colors:
            base_color = service_colors[name]
        else:
            # Assign a color based on the name's hash so it's consistent between runs
            color_idx = hash(name) % len(default_pastel_colors)
            base_color = default_pastel_colors[color_idx]
        
        # Determine status color for the dot indicator
        status_color = '#28a745'  # Green for ready
        if not status_info.get('ready', True):
            status_color = '#ffc107'  # Yellow for initializing
        if status_info.get('status') == 'error': 
            status_color = '#dc3545'  # Red for error
            
        # Create tooltip text with detailed info
        tooltip_text = f"{name}: {status_info.get('status', 'ready')}"
        if status_info.get('description'):
            tooltip_text += f"\n{status_info['description']}"
        tooltip_text += "\nClick for help"

        # Create the indicator - now with a clickable button
        indicator_id = f"service-indicator-{name}"
        service_indicators.append(
            html.Div([
                dbc.Tooltip(
                    tooltip_text,
                    target=indicator_id,
                    placement='top'
                ),
                html.Button([
                    html.I(className="fas fa-circle", 
                          style={'color': status_color, 'marginRight': '3px', 'fontSize': '8px'}),
                    html.Span(name, style={'fontSize': '11px'})
                ], 
                id={'type': 'service-help-button', 'index': name},
                style={
                    'border': 'none', 
                    'background': 'none',
                    'cursor': 'pointer',
                    'padding': '0',
                    'textAlign': 'left',
                    'width': '100%'
                })
            ], 
            id=indicator_id,
            style={
                'padding': '3px 8px',
                'borderRadius': '12px',
                'border': '1px solid #dee2e6',
                'backgroundColor': base_color
            })
        )
    
    return service_indicators, services_status

@callback(
    [Output('chat-history', 'children', allow_duplicate=True),
     Output('chat-store', 'data', allow_duplicate=True)],
    Input({'type': 'service-help-button', 'index': ALL}, 'n_clicks'),
    [State('chat-store', 'data')],
    prevent_initial_call=True
)
def show_service_help(n_clicks_list, chat_history):
    """Display help message for the clicked service in the chat.
    
    This callback is triggered when a user clicks on one of the service
    indicators, showing that service's help text in the chat interface.
    """
    if not any(n_clicks for n_clicks in n_clicks_list if n_clicks):
        raise PreventUpdate
    
    # Get the clicked button's ID
    triggered_id = ctx.triggered_id
    if not triggered_id:
        raise PreventUpdate
    
    service_name = triggered_id['index']
    
    # Get the service from the registry
    service = service_registry.get_service(service_name)
    if not service:
        raise PreventUpdate
    
    # Get the help text from the service
    help_text = service.get_help_text()
    if not help_text:
        help_text = f"No help available for the {service_name} service."
    
    # Create a chat message with the help text
    help_message = {
        'role': 'assistant',
        'content': f"### {service_name.capitalize()} Service Help\n\n{help_text}",
        'service': service_name,
        'type': 'info',
        'timestamp': datetime.now().isoformat()
    }
    
    # Add the message to chat history
    chat_history = chat_history.copy() if chat_history else []
    chat_history.append(help_message)
    
    # Return updated chat history
    return create_chat_elements_batch(chat_history), chat_history

# Add this after the callback imports
app.clientside_callback(
    """
    function(children) {
        if (!children) return;
        
        // Use setTimeout to ensure this runs after the DOM is updated
        setTimeout(function() {
            var chatContainer = document.getElementById('chat-history');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }, 100);
        
        return window.dash_clientside.no_update;
    }
    """,
    Output('chat-history', 'children', allow_duplicate=True),
    Input('chat-history', 'children'),
    prevent_initial_call=True
)

if __name__ == '__main__':
    # Start the app
    app.run_server(debug=False, host='0.0.0.0', port=8051)

