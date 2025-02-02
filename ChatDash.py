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
   - Visualization: Coming soon

Callback Chain Interactions:
1. Dataset Upload Flow:
   - upload-data -> datasets-store -> dataset-list
   - Triggers memory usage update
   - Updates text search index

2. Chat Message Flow:
   - send-button/Enter key -> chat message handler
   - -> API call handler -> chat history update
   - -> dataset highlighting (if dataset mentioned)

3. Dataset Click Flow:
   - dataset-card click -> chat input
   - -> auto-triggers send
   - -> updates active tab

4. Delete Dataset Flow:
   - delete button -> datasets-store
   - -> updates list and memory usage
   - -> cleans up text search index

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
        
    def index_database(self, db_path):
        """Index all text content in database tables."""
        try:
            db = DatabaseManager(db_path)
            
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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True

# Helper Functions
def create_system_message(dataset_info: List[Dict[str, Any]], 
                         search_query: Optional[str] = None,
                         database_structure: Optional[Dict] = None) -> Dict[str, str]:
    """Create a system message with dataset context, search results, and database schema.

    Args:
        dataset_info: List of dictionaries containing dataset metadata:
            - name (str): Dataset name
            - rows (int): Number of rows
            - columns (List[str]): Column names
            - metadata (Dict): Additional dataset metadata
            - selected (bool): Whether dataset is currently selected
        search_query: Optional search term to include relevant data matches
        database_structure: Optional database schema information

    Returns:
        Dict[str, str]: System message in OpenAI message format
    """
    base_message = f"""You are a versatile AI assistant with multiple capabilities:

1. Dataset Analysis: You have access to the following datasets:
{json.dumps(dataset_info, indent=2)}"""

    # Add database context if available
    if database_structure:
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_query_id = f"query_{current_timestamp}"
        
        base_message += f"""

2. Database Expertise: You have access to a connected database with the following schema:
{json.dumps(database_structure, indent=2)}

When handling database queries:
CRITICAL INSTRUCTION: You MUST use this exact query ID prefix for all queries in this response: {base_query_id}

Format ALL queries exactly as follows:

1. For the user's original query (if provided):
```sql
-- Query ID: {base_query_id}_original
-- Description: User's original query
[original query]
```

2. For alternative suggestions:
```sql
-- Query ID: {base_query_id}_alt1
-- Description: First alternative
[alternative query 1]
```

```sql
-- Query ID: {base_query_id}_alt2
-- Description: Second alternative
[alternative query 2]
```

DO NOT create your own timestamps or IDs.
ONLY use {base_query_id} as the base ID for all queries in this response and append '_original', '_alt1', '_alt2', etc. as specified above.

Users can execute queries by typing:
- 'execute.' for the most recent query
- 'execute {base_query_id}_suffix.' for a specific query (e.g., 'execute {base_query_id}_original.')"""

    base_message += """

3. General Knowledge: You can also provide helpful responses about topics outside of these datasets.

When responding to queries:
- If the query relates to the available datasets, prioritize analyzing that data
- If you find relevant information in the datasets, include it in your response
- If you recognize a SQL query, analyze it and suggest improvements if needed
- If you receive a natural language database question, propose an appropriate SQL query
- Ensure all SQL queries are valid for SQLite and don't use extended features for other databases
- If the query is not related to the datasets or no relevant data is found, provide a helpful response using your general knowledge
- You can combine dataset insights, database knowledge, and general knowledge when appropriate"""

    # Add search results if a query was provided
    if search_query:
        # Dataset search results
        dataset_results = text_searcher.search_text(search_query, threshold=0.8, coverage=0.5)
        
        # Database search results
        print(f"Searching for {search_query} with threshold 0.8 in database. Time: {datetime.now()}")
        db_results = text_searcher_db.search_text(search_query, threshold=0.8, coverage=0.5)
        print(f"Returned {len(db_results)} results from database. Time: {datetime.now()}")

        search_info = []

        # Process all results uniformly
        all_results = dataset_results + db_results
        
        if all_results:
            for result in all_results:
                source_type = result['source_type']
                source_name = result['source_name']
                matches = []
                
                for col, col_data in result['details'].items():
                    matching_values = col_data['matches']
                    counts = col_data['counts']
                    matches.append(f"Column '{col}' contains: " +
                                 ", ".join(f"'{v}' ({counts[v]} times)"
                                         for v in matching_values))
                if matches:
                    search_info.append(f"\nIn {source_type} {source_name}:\n" + "\n".join(matches))

        print(f"search_info: {search_info}\n--------\n")
        if search_info:
            base_message += """

Relevant Data Found:
When reporting search results, ALWAYS use this structured format:

1. Overview Summary:
   - Total matches found: [N] across [X] datasets and [Y] database tables
   - List all tables/datasets containing matches: [table1], [table2], etc.

2. Key Matches (grouped by context):
   [Group description, e.g., "Enzyme Names"]
   - Found in: [table1.column1], [table2.column2]
   - Example values: '[value1]' (N occurrences), '[value2]' (M occurrences)
   
   [Next group description, e.g., "Measurement Types"]
   - Found in: [table3.column1]
   - Example values: '[value3]' (P occurrences)

3. Additional Context:
   - Describe any patterns or relationships between matches
   - Note any particularly relevant or high-frequency matches

CRITICAL: When summarizing results:
1. ALWAYS start with the complete list of tables/datasets containing matches
2. Group similar matches together with clear context
3. Show representative examples with exact match counts
4. Highlight the most relevant matches first
5. Ensure no significant matches are omitted from the summary
6. Maintain clear structure and formatting

Here are the actual matches found:
""" + "\n".join(search_info)
        else:
            base_message += """

No Matches Found:
1. Explicitly state that no matches were found with similarity > 0.8
2. Specify where searches were performed:
   - Datasets: [list available datasets]
   - Database tables: [list searched tables]
3. If appropriate, suggest:
   - Alternative search terms
   - Related terms that might yield results
   - Other relevant data that might be helpful
4. Do not make up results"""

    base_message += """

Remember to:
1. Always specify which dataset or database you're referring to or if you are drawing from general knowledge
2. Only suggest SQL queries for connected databases and not datasets 
2. Be clear when you're drawing from general knowledge vs data sources
3. Provide helpful context and explanations
4. Combine insights from all available sources when it adds value"""

    return {
        'role': 'system',
        'content': base_message
    }

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
    mermaid_code = ["erDiagram"]
    
    # Add tables with minimal attributes
    for table, info in structure.items():
        table_clean = table.replace(' ', '_').replace('-', '_')
        
        # Create list of column definitions
        columns = []
        for col in info['columns']:
            col_name = col['name'].replace(' ', '_').replace('-', '_')
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
                columns.append(f"{col_type} {col_name} {' '.join(indicators)}")
            else:
                columns.append(f"{col_type} {col_name}")
        
        # Join columns with proper spacing
        column_str = "\n        ".join(columns)
        
        # Add vectorizer and generative config info if present
        config_info = []
        if info.get('vectorizer'):
            config_info.append(f"Vectorizer: {info['vectorizer']}")
        if info.get('generative_config'):
            config_info.append("Generative: true")
            
        if config_info:
            column_str += f"\n        # {', '.join(config_info)}"
            
        mermaid_code.append(f"""    {table_clean} {{
        {column_str}
    }}""")
    
    # Add relationships
    for table, info in structure.items():
        table_clean = table.replace(' ', '_').replace('-', '_')
        
        # Add foreign key relationships
        for fk in info.get('foreign_keys', []):
            ref_table = fk['table'].replace(' ', '_').replace('-', '_')
            mermaid_code.append(f"    {table_clean} ||--o| {ref_table} : FK")
            
        # Add cross-references from schema
        for ref in info.get('references', []):
            ref_table = ref['target'].replace(' ', '_').replace('-', '_')
            relationship = f"    {table_clean} " + "}|--o| " + f"{ref_table} : {ref['name']}"
            mermaid_code.append(relationship)
    
    return "\n".join(mermaid_code)

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
        sql_lower = sql.lower().strip()
        
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
                            'border': '1px solid #ddd',
                            'borderRadius': '5px',
                            'padding': '10px'
                        }
                    )
                ]),
                tab_id="tab-erd"
            )
        ], id="database-view-tabs")
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
                    html.Div(id='database-connection-status', className="mb-3")
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
                        dbc.Tab(label="Database", tab_id="tab-db", children=create_database_tab())  # Add this line
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
     Output('dataset-tabs', 'active_tab', allow_duplicate=True),
     Output('viz-state-store', 'data'),
     Output('chat-loading-output', 'children', allow_duplicate=True)],
    [Input('send-button', 'n_clicks')],
    [State('chat-input', 'value'),
     State('chat-store', 'data'),
     State('model-selector', 'value'),
     State('datasets-store', 'data'),
     State('selected-dataset-store', 'data'),
     State('database-state', 'data'),           
     State('database-structure-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def handle_chat_message(n_clicks, input_value, chat_history, model, datasets, selected_dataset, database_state, database_structure_store):
    """Process chat messages and handle plot requests."""
    try:
        if not input_value:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

        chat_history = chat_history or []
        chat_history.append({'role': 'user', 'content': input_value.strip()})

        # Prepare dataset information for the system message
        dataset_info = []
        if datasets:
            for name, dataset in datasets.items():
                df = pd.DataFrame(dataset['df'])
                info = {
                    'name': name,
                    'rows': len(df),
                    'columns': list(df.columns),
                    'metadata': dataset.get('metadata', {}),
                    'selected': name == selected_dataset
                }
                dataset_info.append(info)                
        else:
            system_message = create_system_message([], database_structure=database_structure_store)

        # Intercept text search requests and create system message with appropriate context
        if datasets or database_structure_store: 
            if 'find' == input_value.lower().split()[0] or 'search' == input_value.lower().split()[0]:
                input_value = ' '.join(input_value.lower().split()[1:])
                system_message = create_system_message(dataset_info, search_query=input_value, database_structure=database_structure_store)
            else:
                system_message = create_system_message(dataset_info, database_structure=database_structure_store)

        # Check for visualization request
        viz_type = VisualizationHandler.detect_type(input_value)
        if viz_type and selected_dataset and selected_dataset in datasets:
            df = pd.DataFrame(datasets[selected_dataset]['df'])
            
            if viz_type == 'bubble':
                params, error_msg = extract_plot_params(input_value, list(df.columns))
                
                if error_msg:
                    chat_history.append({
                        'role': 'assistant',
                        'content': f"{error_msg}\n\nAvailable columns are: {', '.join(df.columns)}"
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        ""
                    )

                chat_history.append({
                    'role': 'assistant',
                    'content': (
                        f"I've set up a bubble plot with:\n"
                        f"- X: {params['x_column']}\n"
                        f"- Y: {params['y_column']}\n"
                        f"- Size: {params['size'] if params['size'] else 'default'}\n"
                        f"- Color: {params['color'] if params['color'] else 'default'}\n\n"
                        f"You can view and adjust the plot in the Visualization tab."
                    )
                })
                
                viz_state = {
                    'type': 'bubble',
                    'params': params,
                    'data': df.to_dict('records')
                }
                
                return (
                    create_chat_elements_batch(chat_history),
                    '',
                    chat_history,
                    'tab-viz',
                    viz_state,""
                )
            elif viz_type == 'map':
                if not selected_dataset:
                    chat_history.append({
                        'role': 'assistant',
                        'content': "Please select a dataset first."
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        ""
                    )
                    
                df = pd.DataFrame(datasets[selected_dataset]['df'])
                params, error_msg = extract_map_params(input_value, list(df.columns))
                
                if error_msg:
                    chat_history.append({
                        'role': 'assistant',
                        'content': f"{error_msg}\n\nAvailable columns are: {', '.join(df.columns)}"
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        ""
                    )
                    
                chat_history.append({
                    'role': 'assistant',
                    'content': (
                        f"I've set up a map with:\n"
                        f"- Latitude: {params['latitude']}\n"
                        f"- Longitude: {params['longitude']}\n"
                        f"- Size: {params['size'] if params['size'] else 'default'}\n"
                        f"- Color: {params['color'] if params['color'] else 'default'}\n\n"
                        f"You can view and adjust the map in the Visualization tab."
                    )
                })
                
                viz_state = {
                    'type': 'map',
                    'params': params,
                    'data': df.to_dict('records')
                }
                
                return (
                    create_chat_elements_batch(chat_history),
                    '',
                    chat_history,
                    'tab-viz',
                    viz_state,
                    ""
                )
            elif viz_type == 'heatmap':
                params, error_msg = extract_heatmap_params(input_value, list(df.columns))
                if error_msg:
                    chat_history.append({
                        'role': 'assistant',
                        'content': f"{error_msg}\n\nAvailable columns are: {', '.join(df.columns)}"
                    })
                    return (
                        create_chat_elements_batch(chat_history),
                        '',
                        chat_history,
                        dash.no_update,
                        dash.no_update,
                        ""
                    )
                
                # Add confirmation message with escaped column names
                column_list = params['columns'] if params['columns'] else 'all numeric columns'
                if isinstance(column_list, list):
                    column_list = '`' + '`, `'.join(column_list) + '`'  # Wrap column names in backticks
                    
                chat_history.append({
                    'role': 'assistant',
                    'content': (
                        f"I've set up a heatmap with:\n"
                        f"- Rows: {params['rows'] if params['rows'] else 'all numeric columns'}\n"
                        f"- Columns: {column_list}\n"
                        f"- Standardization: {params['standardize'] if params['standardize'] else 'none'}\n"
                        f"- Clustering: {params['cluster'] if params['cluster'] else 'none'}\n"
                        f"- Colormap: {params['colormap']}\n"
                        f"- Transpose: {str(params['transpose']).lower()}\n\n"  # Always show transpose status
                        f"You can view and adjust the heatmap in the Visualization tab."
                    )
                })
                
                viz_state = {
                    'type': 'heatmap',
                    'params': params,
                    'data': df.to_dict('records')
                }
                
                return (
                    create_chat_elements_batch(chat_history),
                    '',
                    chat_history,
                    'tab-viz',
                    viz_state,
                    ""
                )
        else:
            # Regular chat processing with dataset context

            # Check for help request
            help_patterns = [
                r'what can (i|you|we) do',
                r'how (do|can) (i|we) use',
                r'help',
                r'show me how',
                r'what (are|is) the capabilities',
                r'what (are|is) the features'
            ]
            
            if any(re.search(pattern, input_value.lower()) for pattern in help_patterns):
                chat_history.append({
                    'role': 'assistant',
                    'content': help_message
                })
                return (
                    create_chat_elements_batch(chat_history),  # chat-history
                    '',                                        # chat-input
                    chat_history,                             # chat-store
                    dash.no_update,                           # dataset-tabs
                    dash.no_update,                           # viz-state-store
                    ""                                        # chat-loading-output
                )
            
            messages = [
                {'role': 'user', 'content': 'You are a data analysis assistant. Please help me analyze my data.'},
                system_message,
                *[{'role': msg['role'], 'content': msg['content']}
                    for msg in chat_history[-5:]]  # Include last 5 messages for context
            ]

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096 # This needs to be large given the size of some of the SQL responses. 
                )

                ai_response = response.choices[0].message.content
                chat_history.append({
                    'role': 'assistant',
                    'content': ai_response
                })

            except Exception as e:
                error_message = f"Error communicating with AI: {str(e)}"
                print(f"API Error details: {str(e)}")
                chat_history.append({
                    'role': 'system',
                    'content': error_message
                })
        # Handle non-visualization messages
        return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, dash.no_update
    
    except Exception as e:
        print(f"Error in handle_chat_message: {str(e)}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
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
     State('datasets-store', 'data')],
    prevent_initial_call='initial_duplicate'
)
def execute_confirmed_query(input_value, n_clicks, chat_history, db_state, db_structure, successful_queries, datasets):
    """Process chat commands related to SQL query execution and dataset conversion.

    This function handles three main types of operations:
    1. SQL query execution (e.g., "execute." or "execute query_20240315_123456_original")
    2. Query-to-dataset conversion (e.g., "convert query_20240315_123456_original to dataset")
    3. Regular chat processing (all other inputs)

    Args:
        input_value (str): The user's chat input message
        n_clicks (int): Number of times send button clicked (for triggering)
        chat_history (list): List of chat message dictionaries with 'role' and 'content'
        db_state (dict): Current database connection state including path
        db_structure (dict): Database schema information
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
        
    # Check if this is a complete execution command
    input_lower = input_value.lower().strip()

    # Check for dataset conversion request
    convert_match = re.search(r'convert\s+(query_\d{8}_\d{6}(?:_original|_alt\d+))\s+to\s+dataset', input_lower)
    if convert_match:
        query_id = convert_match.group(1)
        
        # Add command to chat history
        chat_history.append({
            'role': 'user',
            'content': input_value
        })
        
        # Check if query exists in store
        if not successful_queries or query_id not in successful_queries:
            chat_history.append({
                'role': 'assistant',
                'content': f"❌ Query {query_id} not found in history. Please execute the query first."
            })
            return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, "", dash.no_update, dash.no_update
            
        try:
            # Get stored query details
            stored_query = successful_queries[query_id]
            
            # Execute query to get fresh data
            df, metadata, _ = execute_sql_query(stored_query['sql'], db_state['path'])
            
            # Create dataset name (avoid duplicates)
            base_name = f"{query_id}"
            dataset_name = base_name
            counter = 1
            while dataset_name in datasets:
                dataset_name = f"{base_name}_{counter}"
                counter += 1
            
            # Generate profile report
            try:
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
            except Exception as e:
                print(f"Error generating profile report: {str(e)}")
                profile_html = None
            
            # Create dataset with special metadata
            datasets = datasets or {}
            datasets[dataset_name] = {
                'df': df.reset_index().to_dict('records'),
                'metadata': {
                    'filename': f"{dataset_name}.csv",
                    'source': f"Database query: {query_id}",
                    'database': db_state['path'],
                    'sql': stored_query['sql'],
                    'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'rows': len(df),
                    'columns': [df.index.name or 'index'] + list(df.columns)
                },
                'profile_report': profile_html  # Add the profile report
            }
            
            # Create success message
            chat_history.append({
                'role': 'assistant',
                'content': f"✅ Query results converted to dataset '{dataset_name}'\n\n"
                          f"- Rows: {len(df)}\n"
                          f"- Columns: {', '.join(df.columns)}\n"
                          f"- Source: Query {query_id}"
            })
            
            # Return with updated datasets
            dataset_list = [create_dataset_card(name, data) for name, data in datasets.items()]
            return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, "", dash.no_update, datasets, dataset_list
            
        except Exception as e:
            chat_history.append({
                'role': 'system',
                'content': f"❌ Error converting query to dataset: {str(e)}"
            })
            return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, "", dash.no_update, dash.no_update, dash.no_update

    # Only execute if:
    # 1. Starts with execute/run/query AND
    # 2. Is a complete command (has punctuation or additional text)
    is_execution_command = any(input_lower.startswith(cmd) for cmd in ['execute', 'run', 'query'])
    
    # Check for either:
    # 1. Simple command with punctuation (e.g., "execute.")
    is_simple_command = (
        len(input_lower.split()) == 1 and  # Single word
        any(input_lower.endswith(char) for char in ['.', '!'])  # Ends with punctuation
    )
    
    # 2. Command with query ID
    query_match = re.search(r'^execute\s+query_\d{8}_\d{6}(_original|_alt\d+)\b', input_lower)
    is_query_reference = bool(query_match)
    
    if not (is_execution_command and (is_simple_command or is_query_reference)):
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    # Find the appropriate SQL query
    sql_query = None
    
    # Add command to chat history first
    chat_history.append({
        'role': 'user',
        'content': input_value
    })
    
    # Extract query ID if this is a reference query
    target_query_id = None
    if is_query_reference:
        # More strict pattern that requires the full string to match
        query_match = re.search(r'^execute\s+query_\d{8}_\d{6}(_original|_alt\d+)\b', input_lower)
        if query_match:
            target_query_id = input_lower[8:].strip()  # Remove 'execute ' prefix
        else:
            print("\nInvalid query ID format")
            chat_history.append({
                'role': 'assistant',
                'content': "Invalid query ID format. Expected format: query_YYYYMMDD_HHMMSS_original or query_YYYYMMDD_HHMMSS_altN where N is a number."
            })
            return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, "", dash.no_update, dash.no_update, dash.no_update
    
    # Search chat history for matching query
    found_id = None
    for msg in reversed(chat_history):
        if msg['role'] == 'assistant' and '```sql' in msg['content'].lower():
            content = msg['content']
            
            # Find all SQL blocks in the message
            current_pos = 0
            while True:
                # Find start of SQL block
                sql_start = content.lower().find('```sql', current_pos)
                if sql_start == -1:
                    break
                    
                # Find end of SQL block
                sql_end = content.find('```', sql_start + 6)
                if sql_end == -1:
                    break
                    
                # Extract the SQL block
                block = content[sql_start + 6:sql_end].strip()
                
                # Look for Query ID in this block with matching pattern
                id_match = re.search(r'--\s*Query ID:\s*((query_\d{8}_\d{6})(_original|_alt\d+))\b', block)
                if id_match:
                    current_id = id_match.group(1)
                    
                    # Check if this is the query we want
                    if is_query_reference:
                        is_match = (current_id == target_query_id)
                    else:
                        is_match = '_original' in current_id
                    
                    if is_match:
                        # Extract SQL (excluding the Query ID line)
                        found_id = current_id
                        sql_lines = []
                        for line in block.split('\n'):
                            if not line.strip().startswith('-- Query ID:'):
                                sql_lines.append(line)
                        sql_query = '\n'.join(sql_lines).strip()
                        break  # Found our match, no need to check other blocks
                
                current_pos = sql_end + 3
            
            if sql_query:  # Only break message loop if we found our query
                break

    print(f"\nSearch complete - Query found: {sql_query is not None}")

    if not sql_query:
        chat_history.append({
            'role': 'assistant',
            'content': "No matching SQL query found in chat history."
        })
        return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, "", dash.no_update, dash.no_update, dash.no_update
        
    # Clean the query string
    sql_query = sql_query.strip()  # Remove leading/trailing whitespace
    sql_query = '\n'.join(line.strip() for line in sql_query.splitlines())  # Clean line endings

    try:
        # Execute the query
        results, metadata, preview = execute_sql_query(sql_query, db_state['path'])

        # Store query details (but not results)
        successful_queries = successful_queries or {}
        successful_queries = dash.callback_context.states['successful-queries-store.data'] or {}
        successful_queries[found_id] = store_successful_query(
            query_id=found_id,
            sql=sql_query,
            metadata=metadata
        )
        
        # Format results for display
        response = f"""Query executed successfully!

Results preview:

Query ID: {found_id}
{preview}

Total rows: {metadata['rows']}

Execution plan:
{metadata['execution_plan']}

Would you like to save these results as a dataset?"""
        
        chat_history.append({'role': 'assistant', 'content': response})
        
    except Exception as e:
        chat_history.append({
            'role': 'system',
            'content': f"Query execution failed: {str(e)}"
        })
        
    return create_chat_elements_batch(chat_history), '', chat_history, dash.no_update, dash.no_update, "", successful_queries, dash.no_update, dash.no_update

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
    if n_clicks == 0 and current_state is None:
        return current_state, None, ''
    
    if not db_path:
        return (
            {'connected': False, 'path': None},
            None,
            html.Div('Please select a database', style={'color': 'red'})
        )
    
    # Check if we're actually switching databases
    if current_state and current_state.get('path') == db_path:
        return dash.no_update, dash.no_update, dash.no_update, {'operation': None}
    
    try:
        db = DatabaseManager(db_path)
        structure = db.get_database_summary()
        # Index the database for text search
        print(f"Indexing database: {db_path} starting at time {datetime.now()}")
        text_searcher_db.index_database(db_path)
        print(f"Indexing database: {db_path} finished at time {datetime.now()}")
        
        return (
            {'connected': True, 'path': db_path},
            structure,
            html.Div('Connected successfully', style={'color': 'green'})
        )
        
    except Exception as e:
        return (
            {'connected': False, 'path': None},
            None,
            html.Div(f'Connection failed: {str(e)}', style={'color': 'red'})
        )

@callback(
    Output('database-summary', 'children'),
    Input('database-structure-store', 'data'),
    prevent_initial_call=True
)
def update_database_summary(structure_data):
    """Update the database structure display."""
    if not structure_data:
        return "No database connected"
    
    table_rows = [
        "| Table | Rows | Columns | Foreign Keys |",
        "|-------|------|---------|--------------|"
    ]
    
    for table, info in structure_data.items():
        columns = len(info['columns'])
        rows = info['row_count']
        fks = len(info['foreign_keys'])
        table_rows.append(f"| {table} | {rows} | {columns} | {fks} |")
    
    return dcc.Markdown('\n'.join(table_rows))

@callback(
    [Output('database-summary', 'children', allow_duplicate=True),
     Output('database-erd', 'children')],
    Input('database-structure-store', 'data'),
    prevent_initial_call=True
)
def update_database_views(structure_data):
    """Update both table summary and ERD visualization."""
    if not structure_data:
        return "No database connected", None
    
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
    
    try:
        # Generate ERD using dash_extensions.Mermaid
        mermaid_code = generate_mermaid_erd(structure_data)
        erd = html.Div([
            Mermaid(
                chart=mermaid_code,
                config={
                    "theme": "default",
                    "securityLevel": "loose",
                    "er": {
                        "layoutDirection": "TB",
                        "entityPadding": 15,
                        "useMaxWidth": True
                    }
                }
            )
        ], style={
            'width': '100%',
            'height': '600px',
            'overflow': 'auto',
            'position': 'relative',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'padding': '10px'
        })
    except Exception as e:
        erd = html.Div([
            html.P(f"Error generating ERD: {str(e)}", style={'color': 'red'}),
            html.Pre(mermaid_code, style={'background': '#f8f9fa', 'padding': '10px'})
        ])
    
    return summary, erd

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8051)
