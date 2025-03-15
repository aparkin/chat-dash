# USGS Services for ChatDash

This package provides services for accessing and analyzing USGS data through ChatDash, with a particular focus on water quality monitoring data.

## Services

### Water Quality Service (`water_quality.py`)

Core functionality for accessing USGS water quality data:

- Parameter management and caching
- Site location search
- Data retrieval and analysis
- Summary generation

### USGS Service (`service.py`)

Chat interface services:

1. **USGSService**: General USGS data catalog access
2. **USGSWaterService**: Interactive water quality data queries

## Usage

### Water Quality Queries

The service supports both structured commands and natural language queries:

```
# Direct Commands
tell me about usgs_water
usgs_water.parameters related to nutrients
find sites near Sacramento
get temperature data for site 11447650

# Natural Language
What's the water temperature in the Sacramento River?
Show me dissolved oxygen levels near Freeport
Find monitoring sites around Davis
```

### Parameter Groups

The service organizes water quality parameters into groups:

1. Physical
   - Temperature
   - Specific conductance
   - pH
   - Dissolved oxygen
   
2. Nutrients
   - Nitrate
   - Phosphate
   - Total nitrogen
   - Total phosphorus
   
3. Metals
   - Iron
   - Copper
   - Lead
   - Zinc
   
4. Organic
   - Dissolved organic carbon
   - Total organic carbon

### Location Search

Sites can be found using:
- Named locations (cities, landmarks)
- Geographic coordinates
- Bounding boxes
- Site IDs

## Data Coverage

- **Temporal**: Historical and real-time data
- **Geographic**: Nationwide USGS monitoring network
- **Parameters**: 3000+ water quality parameters
- **Quality**: USGS certified measurements

## Implementation Details

### Core Classes

1. `WaterQualityService`
   - Parameter cache management
   - Site location search
   - Data retrieval
   
2. `USGSWaterService`
   - Natural language processing
   - Command parsing
   - Response formatting

### Data Structures

1. `GeoBounds`
   - Geographic bounding box
   - Coordinate validation
   
2. `SiteLocation`
   - Site coordinates
   - Search radius

## Best Practices

1. **Parameter Selection**
   - Use parameter codes for precise queries
   - Group related parameters for comprehensive analysis
   
2. **Data Retrieval**
   - Consider date ranges to manage data volume
   - Check data availability before detailed queries
   
3. **Location Search**
   - Start with broad area searches
   - Refine based on site types and parameters

## Future Improvements

1. **Enhanced Natural Language Processing**
   - Better location extraction
   - Complex query understanding
   - Context-aware responses

2. **Data Analysis**
   - Trend detection
   - Statistical summaries
   - Visualization options

3. **Integration**
   - Cross-parameter correlations
   - Multiple site comparisons
   - External data sources

## Dependencies

- pandas: Data manipulation
- geopandas: Geographic data handling
- shapely: Geometric operations
- requests: API communication
- asyncio: Asynchronous operations

## Contributing

When adding new features:

1. Update parameter cache as needed
2. Add appropriate tests
3. Document new functionality
4. Follow existing code patterns

## Testing

Run tests with:
```bash
python -m pytest tests/usgs/
```

Key test areas:
- Parameter search
- Site location
- Data retrieval
- Command parsing 