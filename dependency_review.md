# Dependency Review

This document analyzes the dependencies used in the ChatDash application and identifies opportunities for optimization.

## Core Dependencies in Use

Based on an analysis of the imports in the main application files, these are the primary dependencies actually being used:

### Web Framework and UI
- `dash`
- `dash-bootstrap-components`
- `dash-extensions` (for Mermaid diagrams)
- `dash_cytoscape`

### Data Processing
- `pandas`
- `numpy`
- `scipy` (for hierarchical clustering)
- `sklearn` (for TF-IDF and cosine similarity)

### Visualization
- `plotly.express`
- `plotly.graph_objects`
- `ydata-profiling` (for automatic data profiling)

### API Clients
- `openai` (for LLM integration)
- `requests` (for API calls)
- `weaviate-client` (for vector database)

### Storage
- `sqlite3` (for relational database)

### Utilities
- `python-dotenv` (for environment variables)
- `fuzzywuzzy` (for string matching)
- `tqdm` (for progress bars)

## Dependencies That May Be Unnecessary

Based on the requirements.txt analysis and code inspection, these dependencies may not be actively used or could be optional:

### Potentially Unused
- `aiodns`, `aiofiles`, `aiohappyeyeballs` - May not be needed if async operations are minimal
- `affine` - Used for geographic raster data, appears in requirements but no clear imports
- `branca` - Dependency of folium, may not be directly used
- `cattrs`, `cligj` - No direct imports found
- `contourpy` - Dependency for matplotlib, may not be direct import
- `EditorConfig` - Development tool, not a runtime dependency
- `grobid-client-python` - Referenced in documentation but may be an optional component
- `grpcio`, `grpcio-health-checking`, `grpcio-tools` - No direct gRPC usage identified
- `httpcore`, `httpx` - More modern HTTP clients but requests is mainly used

### Development Dependencies (Move to dev-requirements.txt)
- `coverage`
- `pytest`, `pytest-asyncio`, `pytest-cov`, `pytest-mock`
- `black` (if used for formatting)

### Geographic Dependencies (Consider Optional Installation)
- `geopandas`
- `folium`
- `contextily`
- `pyproj`
- `rasterio`
- Geographic related packages (`py3dep`, `pydaymet`, `pygeohydro`, etc.) are likely only used in specific modules

## Recommendations

### 1. Split Requirements Files

Create separate requirements files:
- `requirements.txt` - Core dependencies needed for the main application
- `requirements-dev.txt` - Development dependencies (testing, linting)
- `requirements-geo.txt` - Geographic data processing dependencies
- `requirements-full.txt` - All dependencies

### 2. Dependencies to Remove or Make Optional

The following dependencies could be removed or made optional:
- `aiodns`, `aiofiles`, `aiohappyeyeballs` - Unless async HTTP is critical
- `grpcio` and related packages - Unless gRPC is used somewhere
- Geographic packages if not essential to core functionality
- Development packages from main requirements

### 3. Version Pinning Strategy

- Keep exact versions (`==`) for core dependencies to ensure reproducibility
- Consider flexible versions (`>=`) for development dependencies
- Group packages by functionality in the requirements files with comments

## Proposed requirements.txt Structure

```
# Core Web Framework
dash==2.18.2
dash-bootstrap-components==1.6.0
dash-core-components==2.0.0
dash-extensions==1.0.19
dash-html-components==2.0.0
dash-table==5.0.0
dash_cytoscape==0.2.0
Flask==3.0.3
Flask-Caching==2.3.0
Werkzeug==3.0.6

# Data Processing
pandas==2.2.3
numpy==2.0.2
scipy==1.13.1
scikit-learn==1.6.0

# Visualization
plotly==5.24.1
ydata-profiling==4.12.1

# LLM Integration
openai==1.58.1
tiktoken==0.8.0

# Vector Database
weaviate-client==4.10.4

# Utilities
python-dotenv==1.0.1
fuzzywuzzy==0.18.0
tqdm==4.67.1
psutil==6.1.1
Levenshtein==0.26.1
requests==2.32.3
requests-cache==1.2.1

# Optional features can be installed with:
# pip install -r requirements-geo.txt (for geographic features)
# pip install -r requirements-dev.txt (for development)
```

## Next Steps

1. Test the application with reduced dependencies to ensure functionality
2. Move development dependencies to a separate file
3. Make specialized dependencies optional with clear documentation
4. Update installation instructions to reflect the new requirements structure 