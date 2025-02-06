# ChatDash

ChatDash is an integrated system for interactive data exploration and analysis through a chat interface. It provides a unified frontend for querying and analyzing data from multiple sources, including vector databases and SQL databases. The system is designed to prototype natural language driven data integration, harmonization, visualization and analysis with provenance tracking.

## System Overview

```mermaid
graph TD
    A[ChatDash Frontend] --> B[Weaviate Vector DB]
    A --> C[SQLite Databases]
    A --> D[Future Data Sources]
    B --> E[Literature Database]
    C --> F[Converted MySQL DBs]
    D --> G[Government Data - Coming Soon]
```

## Using ChatDash

### Starting the Application
1. Launch the application:
```bash
python ChatDash.py
```

2. Access the interface in your browser:
```
http://0.0.0.0:8051
```

### Key Features

#### Database Connection
- Use the dropdown and connect button in the upper right
- Database indexing status is shown via an indicator
- Click on database table to view schema (tabular and graphical form)
- Weaviate connection status indicators:
  - Connection status (circle icon): Shows if Weaviate is connected
    - Green: Connected successfully
    - Orange: Connection error
    - Gray: Disconnected
  - Collection status (database icon): Shows literature collection availability
    - Green: Collections available
    - Orange: Missing or misconfigured collections
    - Gray: Collections unavailable
  - Hover over icons for detailed status messages

#### Data Management
- Drop or upload TSV/ZIP archives using the drop pane
- Data files appear as tablets on the left side
- Supported formats: CSV, TSV, and ZIP (containing CSV/TSV)
- Click '×' to delete datasets
- Monitor memory usage in status bar
- Click a tablet to:
  - Select it for chat interaction
  - View data snippet in preview pane
  - See statistical summary in statistics page
- Dataset cards auto-trigger relevant queries when clicked

#### Chat Interface
- Select your preferred LLM using the dropdown
- Type messages in the text area
- Press Enter to send (Shift+Enter for new line)
- Real-time response with data previews and visualizations
- Dataset names are clickable to auto-generate queries

#### Chat Commands and Interactions

##### Data Exploration
- "Tell me about my datasets" - Get a summary of all available datasets
- "Tell me about my dataset" - Information about currently selected dataset
- "Tell me about dataset_name" - Get information about a specific dataset
- "Tell me about the connected database" - Explore database structure and contents
- Use natural language to ask questions about your data
- Reference specific columns using `backticks` when needed

##### SQL Queries and Database Operations
- First, select your database using the dropdown at the top of Data Management
- View database structure in the Database tab under Dataset Info
- Write natural language queries that get converted to SQL
- Execute queries in multiple ways:
  - Simple: Type "execute." to run the last query
  - Specific: "execute query_20240315_123456_original" for a particular query
  - Note: Valid execution commands run immediately
- Convert query results to datasets:
  - "convert query_20240315_123456_original to dataset"
- Combine SQL query results with uploaded data

##### Visualization Commands
- Basic plots: "Plot [column1] vs [column2]"
- Enhanced plots: Add "size=[value/column] color=[value/column]"
- Heatmaps: "Create heatmap columns=[col1,col2,col3]"
  - Options: rows=[...] standardize=rows/columns cluster=both/rows/columns
- Maps: "Create map latitude=[column1] longitude=[column2]"
  - Options: size=[value/column] color=[value/column]

#### Visualization Features
Currently supported visualization types:

##### Bubble Plots
- 2D scatter plots with size and color dimensions
- Customizable axes and labels
- Interactive tooltips and zooming
- Adjustable plot parameters in Visualization tab

##### Maps
- Geographical data visualization
- Support for latitude/longitude plotting
- Color coding by data values
- Interactive pan and zoom

##### Heatmaps
- Correlation analysis
- Matrix-style data visualization
- Custom color scales
- Interactive value inspection

All visualizations feature:
- Interactive controls
- Export capabilities
- Real-time updates
- Responsive design

#### Data Analysis Features
- Preview: First few rows of datasets
- Statistics: Detailed profiling reports
- Visualization: Interactive plots and charts
- SQL Query: Natural language to SQL conversion

## Components

### 1. ChatDash Frontend (`ChatDash.py`)
A Plotly Dash-based web application running on port 8051 that provides:
- Interactive chat interface
- Dataset management and visualization
- Database connection and querying
- Real-time data analysis

### 2. Data Sources

#### Literature Database (Weaviate)
- Managed through `weaviate_manager` CLI tool
- Processes academic papers using GROBID pipeline
- Supports semantic search and complex queries
- [See detailed documentation](CreatingTheLiteratureDatabase.md)

#### SQLite Databases
- Store in `data/` directory
- Convert from MySQL using provided `mysql2sqlite` tool
- Support standard SQL queries and joins

#### Government Database (Coming Soon)
- Integration planned for government data sources
- Details and implementation to be determined

## Getting Started

### Prerequisites
- Python 3.7+
- Weaviate instance for vector database
- SQLite for relational databases

### Environment Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with required credentials:
```env
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com  # or your custom endpoint
```

### Database Setup

#### Literature Database
Follow the [Literature Database Setup Guide](CreatingTheLiteratureDatabase.md) for:
- Setting up GROBID for PDF processing
- Processing academic papers
- Importing to Weaviate

#### SQLite Databases
1. Place SQLite databases in the `data/` directory
2. To convert MySQL databases:
```bash
# Convert MySQL dump to SQLite
./data/mysql2sqlite dump_mysql.sql > output.sql
# Import into SQLite
sqlite3 data/your_database.db < output.sql
```

## Usage

### Using weaviate_manager
The `weaviate_manager` package provides both a CLI and programmatic interface for managing and querying the literature database.

#### CLI Usage
```bash
# Show database information
python -m weaviate_manager.cli --show info

# Perform a search
python -m weaviate_manager.cli --query "your query" --search-type hybrid
```

#### Programmatic Usage
```python
from weaviate_manager.query import QueryManager

# Initialize query manager
query_manager = QueryManager(client)

# Perform a search
results = query_manager.comprehensive_search(
    query_text="your query",
    search_type="hybrid",
    limit=10
)
```

## Contributing
Guidelines for contributing to be added.

## License
License information to be added.