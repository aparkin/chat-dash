# NMDC Enhanced Service

This service provides enhanced capabilities for interacting with the NMDC (National Microbiome Data Collaborative) API, including schema introspection, advanced query building, and natural language processing.

## Components

### 1. Schema Manager

The `SchemaManager` provides functionality for fetching and processing NMDC schema information. It caches schema data to minimize API calls and offers methods to retrieve metadata about entities and their attributes.

**Key features:**
- Fetches NMDC schema, statistics, and summary information
- Provides metadata about entities and their attributes
- Identifies searchable attributes for each entity type
- Caches data to minimize API calls

**Example usage:**
```python
from schema_manager import SchemaManager

# Create schema manager
schema_manager = SchemaManager()

# Get schema
schema = schema_manager.get_schema()

# Get statistics
stats = schema_manager.get_stats()

# Get summary
summary = schema_manager.get_summary()

# Get metadata for a specific entity
biosample_metadata = schema_manager.get_entity_metadata("biosample")

# Get searchable attributes for an entity
biosample_attrs = schema_manager.get_searchable_attributes("biosample")
```

### 2. API Client

The `APIClient` executes queries against the NMDC API and processes the responses. It supports both the Runtime API and Data API, with intelligent fallback mechanisms.

**Key features:**
- Supports both Runtime API and Data API endpoints
- Handles different endpoint naming conventions
- Implements intelligent fallbacks when endpoints aren't available
- Uses appropriate HTTP methods (GET vs POST) based on endpoint requirements
- Processes responses into pandas DataFrames for easy data manipulation
- Handles pagination

**Example usage:**
```python
from api_client import APIClient

# Create API client
api_client = APIClient()

# Execute a query against the Runtime API
api_query = {
    "endpoint": "/studies",
    "filter": "ecosystem_type:Marine",
    "per_page": 10
}
results_df, metadata = api_client.execute_query(api_query)

# Get entity by ID
study = api_client.get_entity_by_id("study", "nmdc:sty-11-33fbta56")

# Use Data API specifically
data_api_client = APIClient(prefer_data_api=True)
env_query = {"endpoint": "/environment/sankey"}
env_data, env_metadata = data_api_client.execute_query(env_query)
```

### 3. Query Builder

The `QueryBuilder` constructs and validates queries against the NMDC API, providing a programmatic way to build complex queries.

**Key features:**
- Creates queries for different entity types
- Adds conditions with various operators
- Sets pagination parameters
- Validates queries against the schema
- Parses API query objects into structured query representations

**Example usage:**
```python
from schema_manager import SchemaManager
from query_builder import QueryBuilder

# Create dependencies
schema_manager = SchemaManager()
query_builder = QueryBuilder(schema_manager)

# Create a new query
query = query_builder.create_query("biosample")

# Add conditions
query = query_builder.add_condition(query, "ecosystem", "=", "Terrestrial")
query = query_builder.add_condition(query, "depth", ">", 10)

# Set pagination
query = query_builder.set_pagination(query, per_page=25, max_pages=2)

# Validate query
is_valid, error = query_builder.validate_query(query)

# Convert to API query format
api_query = query.to_api_query()
```

### 4. Natural Language Processor

The `NLProcessor` provides functionality for understanding and processing natural language queries about NMDC data, extracting structured query information that can be used with the Query Builder.

**Key features:**
- Extracts entity types from natural language queries
- Identifies filtering conditions with field, operator, and value
- Handles common synonyms and phrasings
- Supports complex queries with multiple conditions
- Provides query suggestions and insights
- Calculates confidence scores for parsed queries

**Example usage:**
```python
from schema_manager import SchemaManager
from nl_processor import NLProcessor
from query_builder import QueryBuilder
from api_client import APIClient

# Create dependencies
schema_manager = SchemaManager()
nl_processor = NLProcessor(schema_manager)
query_builder = QueryBuilder(schema_manager)
api_client = APIClient()

# Process a natural language query
query_result = nl_processor.process_query("Find studies with ecosystem type Marine by Kelly Wrighton")

if query_result.success:
    # Create a query using the extracted information
    query = query_builder.create_query(query_result.entity_type)
    
    # Add conditions from the NL query result
    for condition in query_result.conditions:
        query = query_builder.add_condition(
            query, 
            condition['field'],
            condition['operator'],
            condition['value']
        )
    
    # Set pagination
    query = query_builder.set_pagination(query, per_page=10)
    
    # Convert to API query and execute
    api_query = query.to_api_query()
    results, metadata = api_client.execute_query(api_query)
    
    # Display results
    print(f"Found {metadata['rows']} {query_result.entity_type}s")
    
# Get insights about the query
insights = nl_processor.extract_query_insights(query_result)
```

## Upcoming Components

### 1. Service API

The service will expose a REST API for interacting with the enhanced NMDC functionality.

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages:
  - requests
  - pandas
  - numpy
  - transformers (for NLP functionality, optional)

### Installation
```bash
pip install -r requirements.txt
```

## Development

### Running Tests
```bash
python test_schema_manager.py
python test_query.py
python test_nl_processor.py
```

## License

[MIT License](LICENSE) 