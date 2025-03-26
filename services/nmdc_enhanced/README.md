# NMDC Enhanced Service

The NMDC Enhanced Service provides advanced data discovery and integration capabilities for the National Microbiome Data Collaborative (NMDC) database.

## Features

- Natural language query processing
- Advanced data integration
- Rich metadata and context
- SQL-based querying capabilities

## Commands

### Query Pattern:
1. **Build**: `nmdc_enhanced: [your question]` to formulate a sample query
2. **Execute**: `nmdc_enhanced.query [query_id]` to run a previously built query

### Available Commands:
- `nmdc_enhanced.help` - Show help information
- `nmdc_enhanced.about` - Show service information
- `nmdc_enhanced.entities` - List available entity types
- `convert [result_id] to dataset` - Convert query results to a dataset

### Example Queries:
- `nmdc_enhanced: find soil samples from Washington state`
- `nmdc_enhanced: samples with carbon measurements from forest environments`
- `nmdc_enhanced: soil samples from the Wrighton lab`
- `nmdc_enhanced: find studies about soil microbiology`
- `nmdc_enhanced: show biosamples from forest environments`
- `nmdc_enhanced: get data objects with metagenome sequencing`

## Result Information

Results include:
- Sample metadata (ID, collection date, location)
- Associated study information
- Physical measurements when available
- Counts of related data objects by type

## Components

### 1. Data Manager

The `NMDCEnhancedDataManager` handles data loading, processing, and statistical analysis. It loads data from the NMDC API upfront, processes it into unified dataframes, and calculates statistics to support rich context for queries.

**Key features:**
- Loads all NMDC studies and biosamples data upfront
- Processes and cleans data for consistent access
- Builds a unified dataframe linking studies to biosamples
- Calculates statistics and metadata for context
- Provides geographic coverage analysis
- Handles data refreshing based on configurable cache expiry

**Example usage:**
```python
from services.nmdc_enhanced import NMDCEnhancedDataManager, NMDCEnhancedConfig

# Create configuration
config = NMDCEnhancedConfig()

# Create data manager
data_manager = NMDCEnhancedDataManager(config)

# Access data
studies_df = data_manager.studies_df
biosamples_df = data_manager.biosamples_df
unified_df = data_manager.unified_df

# Get context information
context = data_manager.get_dataframe_context()
```

### 2. Models

The `models.py` file defines data structures used throughout the service.

#### NMDCEnhancedConfig

Configuration class for the NMDC Enhanced service.

**Key attributes:**
- `name`: Service name
- `api_base_url`: Base URL for the NMDC API
- `model_name`: LLM model name for query processing
- `temperature`: Temperature setting for LLM
- `cache_expiry_hours`: How often to refresh data
- `default_preview_rows`: Default number of rows to show in previews

#### QueryResult

Container for query execution results, storing both the full dataframe and a preview.

**Key features:**
- Stores the full dataframe result
- Maintains metadata about the query
- Includes a description of the results
- Provides a `to_preview` method for generating compact previews

**Example usage:**
```python
from services.nmdc_enhanced import QueryResult
import pandas as pd

# Create a result
result_df = pd.DataFrame(...)
result = QueryResult(
    result_df,
    description="Query results for forest soil samples",
    metadata={"query_text": "Find forest soil samples", "row_count": len(result_df)}
)

# Generate a preview
preview = result.to_preview(max_rows=10)
```

### 3. Service

The `NMDCEnhancedService` class integrates with ChatDash to provide a natural language interface to NMDC data.

**Key features:**
- Natural language query processing using LLMs
- Data integration across studies and biosamples
- Preview generation for query results
- Dataset conversion capabilities
- Contextual information about available data

**Example commands:**
- `nmdc_enhanced.help` - Show help information
- `nmdc_enhanced.about` - Get information about available data
- `nmdc_enhanced.entities` - List available entity types
- `nmdc_enhanced: find soil samples from Washington state` - Build a query
- `nmdc_enhanced.query [query_id]` - Execute a previously built query
- `convert [result_id] to dataset` - Convert query results to a dataset

## Implementation Design

The NMDC Enhanced Service follows the MONet pattern with these key design principles:

1. **Upfront data loading** - All data is loaded into memory at startup for immediate access
2. **Rich metadata** - Statistics and context are calculated and stored for use in queries
3. **Unified dataframe** - Studies and biosamples are merged into a single dataframe for easy querying
4. **LLM-powered queries** - Natural language queries are transformed into structured query plans
5. **Efficient storage** - Query results are stored with both preview and full data

This pattern ensures fast query execution while maintaining the ability to process complex natural language queries.

## Installation & Setup

### Prerequisites
- Python 3.8+
- Required packages:
  - pandas
  - numpy
  - requests
  - openai (or equivalent for LLM access)

## License

[MIT License](LICENSE) 