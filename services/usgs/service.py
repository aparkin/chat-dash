"""USGS services for ChatDash."""

from typing import Dict, List, Optional, Any, Tuple
import json
import re
from datetime import datetime, timedelta
import pandas as pd
import logging
from collections import defaultdict
from enum import Enum, auto
from ydata_profiling import ProfileReport

from services.base import ChatService, ServiceMessage, ServiceResponse, MessageType, PreviewIdentifier
from services.llm_service import LLMServiceMixin
from .water_quality import WaterQualityService, GeoBounds, SiteLocation

# Configure logging
logger = logging.getLogger(__name__)

class RequestType(Enum):
    """Types of requests this service can handle."""
    WATER_QUALITY_QUERY = auto()
    SITE_SEARCH = auto()
    PARAMETER_SEARCH = auto()
    SERVICE_INFO = auto()
    SERVICE_TEST = auto()
    NATURAL_QUERY = auto()
    QUERY_SEARCH = auto()
    DATASET_CONVERSION = auto()

class USGSWaterService(ChatService, LLMServiceMixin):
    """Service for USGS Water Quality data access and querying.
    
    This service provides:
    1. Natural language queries for water quality data
    2. Parameter search and metadata retrieval
    3. Geographic site location search
    4. Time-series data retrieval and analysis
    5. Dataset conversion capabilities
    
    Command Patterns:
    - Natural language: "usgs_water: find water temperature in Sacramento"
    - Parameter search: "usgs_water.parameters temperature"
    - Service info: "tell me about usgs_water"
    - Query search: "usgs_water.search [query_id]"
    - Dataset conversion: "convert [query_id] to dataset"
    
    Implementation Notes:
    - Uses USGS Water Quality Portal API for data retrieval
    - Implements caching for improved performance
    - Provides comprehensive metadata and statistics
    - Supports dataset conversion for further analysis
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating new USGSWaterService instance")
            cls._instance = super(USGSWaterService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("Initializing USGSWaterService (chat interface)...")
            ChatService.__init__(self, "usgs_water")
            LLMServiceMixin.__init__(self, "usgs_water")
            self.water_quality = WaterQualityService()
            self.parameter_metadata = self._initialize_parameter_metadata()
            
            # Register our prefix for query IDs
            try:
                PreviewIdentifier.register_prefix("usgs_water_query")
            except ValueError:
                # Prefix already registered, which is fine
                pass
                
            # Query execution patterns
            self.execution_patterns = [
                r'^usgs_water\.search\s+(?:usgs_water_query_)?\d{8}_\d{6}(?:_orig|_alt\d+)\b',
                r'^usgs_water\.parameters\s+.*',
                r'^usgs_water\.search\.?$',
                r'tell\s+me\s+about\s+usgs_water\b',
                r'^convert\s+usgs_water_query_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b'
            ]
            
            # Compile patterns
            self.execution_res = [re.compile(p, re.IGNORECASE) for p in self.execution_patterns]
            
            # Query block pattern
            self.query_block_re = re.compile(r'```usgs_water\s*(.*?)```', re.DOTALL)
            
            self._initialized = True
            logger.info("USGSWaterService initialization complete!")

    def _initialize_parameter_metadata(self) -> Dict[str, Any]:
        """Initialize and structure parameter metadata for enhanced querying."""
        try:
            # Get parameters from water quality service
            params = self.water_quality.parameters
            if not params or not params.get('parameters'):
                logger.error("No parameter data available from water quality service")
                return {}

            metadata = {
                'by_group': defaultdict(list),
                'by_name': {},
                'by_code': {},  # Separate dictionary for code lookups
                'common_ranges': {},
                'related_parameters': defaultdict(set),
                'measurement_units': defaultdict(set),
                'counts': {
                    'total_raw_parameters': len(params['parameters']),
                    'unique_names': 0,
                    'unique_codes': 0,
                    'by_group': defaultdict(int),
                    'by_provider': params.get('metadata', {}).get('providers', {}),
                    'provider_overlap': params.get('metadata', {}).get('provider_overlap', {
                        'single_provider': 0,
                        'two_providers': 0,
                        'all_providers': 0
                    })
                }
            }

            # Track unique names and codes
            unique_names = set()
            unique_codes = set()

            # Also track provider overlap if not provided
            if not metadata['counts']['provider_overlap']:
                provider_counts = defaultdict(int)
                for param in params['parameters']:
                    providers = param.get('providers', '').split()
                    provider_counts[len(providers)] += 1
                
                metadata['counts']['provider_overlap'] = {
                    'single_provider': provider_counts[1],
                    'two_providers': provider_counts[2],
                    'all_providers': provider_counts[3]
                }

            # Process each parameter
            for param in params['parameters']:
                group = param['parameter_group']
                name = param['parameter_nm'].split(',')[0].strip()  # Get clean name
                code = param.get('parameter_cd', '')  # Make parameter_cd optional
                unit = param.get('unit', '')
                description = param.get('description', '')

                # Track unique identifiers
                if name:
                    unique_names.add(name)
                if code:  # Only track non-empty codes
                    unique_codes.add(code)

                # Group organization
                param_info = {
                    'name': name,
                    'unit': unit,
                    'description': description
                }
                if code:  # Only include code if it exists
                    param_info['code'] = code
                metadata['by_group'][group].append(param_info)
                metadata['counts']['by_group'][group] += 1

                # Name lookup - only add if we don't have this name yet or if this is a better entry
                if name and (name not in metadata['by_name'] or (code and not metadata['by_name'][name].get('code'))):
                    metadata['by_name'][name] = {
                        'group': group,
                        'unit': unit,
                        'description': description
                    }
                    if code:  # Only include code if it exists
                        metadata['by_name'][name]['code'] = code

                # Code lookup - separate from name lookup
                if code:  # Only add if we have a valid code
                    metadata['by_code'][code] = {
                        'name': name,
                        'group': group,
                        'unit': unit,
                        'description': description
                    }

                # Unit tracking
                if unit:
                    metadata['measurement_units'][group].add(unit)

                # Build parameter relationships based on descriptions
                if code and description:  # Only build relationships if we have both code and description
                    desc_words = set(re.findall(r'\w+', description.lower()))
                    for other_param in params['parameters']:
                        other_code = other_param.get('parameter_cd', '')
                        if other_code and other_code != code:  # Only compare if other param has a code
                            other_desc = set(re.findall(r'\w+', other_param.get('description', '').lower()))
                            # If descriptions share significant words, consider them related
                            if len(desc_words & other_desc) >= 3:
                                metadata['related_parameters'][code].add(other_code)

            # Update final counts
            metadata['counts']['unique_names'] = len(unique_names)
            metadata['counts']['unique_codes'] = len(unique_codes)

            # Add common value ranges for well-known parameters
            metadata['common_ranges'] = {
                '00010': {'min': -5, 'max': 40, 'unit': '°C'},  # Water temperature
                '00300': {'min': 0, 'max': 20, 'unit': 'mg/L'},  # Dissolved oxygen
                '00400': {'min': 0, 'max': 14, 'unit': 'std'},   # pH
                '00095': {'min': 0, 'max': 5000, 'unit': 'µS/cm'},  # Specific conductance
                '63680': {'min': 0, 'max': 1000, 'unit': 'NTU'}  # Turbidity
            }

            # Log detailed parameter statistics
            logger.info("Parameter Statistics:")
            logger.info(f"  - Total raw parameters from API: {metadata['counts']['total_raw_parameters']}")
            logger.info(f"  - Unique parameter names: {metadata['counts']['unique_names']}")
            logger.info(f"  - Unique parameter codes: {metadata['counts']['unique_codes']}")
            logger.info("  - Parameters by group: " + ", ".join(f"{k}: {v}" for k, v in metadata['counts']['by_group'].items()))
            logger.info("  - Parameters by provider: " + ", ".join(f"{k}: {v}" for k, v in metadata['counts']['by_provider'].items()))
            logger.info("  - Provider overlap: " + 
                      f"Single provider: {metadata['counts']['provider_overlap']['single_provider']}, " +
                      f"Two providers: {metadata['counts']['provider_overlap']['two_providers']}, " +
                      f"All providers: {metadata['counts']['provider_overlap']['all_providers']}")
            
            return metadata

        except Exception as e:
            logger.error(f"Error initializing parameter metadata: {str(e)}", exc_info=True)
            return {}

    def _analyze_results(self, data: pd.DataFrame, query: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis of query results."""
        try:
            analysis = {
                'coverage': {
                    'temporal': {
                        'start': data['Date'].min(),
                        'end': data['Date'].max(),
                        'gaps': []
                    },
                    'spatial': {
                        'sites': data['Site ID'].nunique(),
                        'bounds': {
                            'min_lat': data['Latitude'].min(),
                            'max_lat': data['Latitude'].max(),
                            'min_lon': data['Longitude'].min(),
                            'max_lon': data['Longitude'].max()
                        }
                    }
                },
                'parameters': {},
                'recommendations': [],
                'additional_measurements': {
                    'available': [],
                    'by_site': defaultdict(list),
                    'by_date': defaultdict(list)
                }
            }

            # Analyze temporal coverage
            dates = pd.to_datetime(data['Date'])
            date_gaps = []
            if len(dates) > 1:
                date_diffs = dates.sort_values().diff()
                significant_gaps = date_diffs[date_diffs > pd.Timedelta(days=30)]
                if not significant_gaps.empty:
                    for gap_start in significant_gaps.index:
                        gap_end = dates[dates > gap_start].min()
                        date_gaps.append({
                            'start': gap_start.strftime('%Y-%m-%d'),
                            'end': gap_end.strftime('%Y-%m-%d'),
                            'duration_days': (gap_end - gap_start).days
                        })
            analysis['coverage']['temporal']['gaps'] = date_gaps

            # Get requested parameters
            requested_params = set(query.get('parameters', []))

            # Analyze all parameters (including additional ones returned)
            for param in [col for col in data.columns if col not in ['Site ID', 'Site Name', 'Latitude', 'Longitude', 'Date']]:
                param_data = data[param].dropna()
                if not param_data.empty:
                    # Basic statistics
                    stats = {
                        'count': len(param_data),
                        'coverage': len(param_data) / len(data) * 100,
                        'statistics': {
                            'min': float(param_data.min()),
                            'max': float(param_data.max()),
                            'mean': float(param_data.mean()),
                            'median': float(param_data.median())
                        }
                    }
                    
                    # Track if this was a requested parameter or additional
                    stats['was_requested'] = param in requested_params
                    
                    # Add to parameters analysis
                    analysis['parameters'][param] = stats
                    
                    # If this is an additional parameter, analyze its distribution
                    if not stats['was_requested']:
                        analysis['additional_measurements']['available'].append({
                            'parameter': param,
                            'coverage': stats['coverage'],
                            'sites': data[~data[param].isna()]['Site ID'].nunique()
                        })
                        
                        # Track by site
                        for site_id in data[~data[param].isna()]['Site ID'].unique():
                            analysis['additional_measurements']['by_site'][site_id].append(param)
                        
                        # Track by date (monthly buckets)
                        site_dates = data[~data[param].isna()]['Date']
                        for date in pd.to_datetime(site_dates):
                            month_key = date.strftime('%Y-%m')
                            if param not in analysis['additional_measurements']['by_date'][month_key]:
                                analysis['additional_measurements']['by_date'][month_key].append(param)

                    # Check for related parameters not in query
                    if param in self.parameter_metadata.get('related_parameters', {}):
                        related = self.parameter_metadata['related_parameters'][param]
                        missing_related = [p for p in related if p not in data.columns]
                        if missing_related:
                            analysis['recommendations'].append({
                                'type': 'related_parameters',
                                'parameter': param,
                                'suggestions': missing_related[:3]  # Limit to top 3
                            })

            # Analyze additional measurements
            if analysis['additional_measurements']['available']:
                # Sort by coverage
                analysis['additional_measurements']['available'].sort(key=lambda x: x['coverage'], reverse=True)
                
                # Add recommendations based on additional measurements
                high_coverage_params = [
                    p for p in analysis['additional_measurements']['available']
                    if p['coverage'] > 50 and p['sites'] >= analysis['coverage']['spatial']['sites'] * 0.5
                ]
                if high_coverage_params:
                    analysis['recommendations'].append({
                        'type': 'additional_parameters',
                        'message': "Consider including these commonly available parameters in future queries:",
                        'parameters': high_coverage_params[:5]  # Top 5 by coverage
                    })
                
                # Look for temporal patterns
                monthly_patterns = defaultdict(int)
                for month, params in analysis['additional_measurements']['by_date'].items():
                    for param in params:
                        monthly_patterns[param] += 1
                
                # Identify consistently measured parameters
                consistent_params = [
                    param for param, count in monthly_patterns.items()
                    if count >= len(analysis['additional_measurements']['by_date']) * 0.8
                ]
                if consistent_params:
                    analysis['recommendations'].append({
                        'type': 'consistent_parameters',
                        'message': "These parameters are consistently measured at these sites:",
                        'parameters': consistent_params[:5]  # Top 5 most consistent
                    })

            # Generate recommendations
            if date_gaps:
                analysis['recommendations'].append({
                    'type': 'temporal_coverage',
                    'message': f"Found {len(date_gaps)} significant gaps in temporal coverage"
                })

            # Check spatial coverage
            if analysis['coverage']['spatial']['sites'] < 3:
                analysis['recommendations'].append({
                    'type': 'spatial_coverage',
                    'message': "Limited spatial coverage. Consider expanding search area."
                })

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}", exc_info=True)
            return {}

    def can_handle(self, message: str) -> bool:
        """Check if this service can handle the message."""
        message = message.strip().lower()
        
        # Check direct commands
        if message in ["tell me about usgs_water", "test usgs_water"]:
            return True
            
        # Check prefixed commands
        prefixes = ["usgs_water:", "usgs_water."]
        if any(message.startswith(prefix) for prefix in prefixes):
            return True
            
        # Check for code blocks
        if '```usgs_water' in message:
            return True
            
        # Check for dataset conversion
        if re.match(r'^convert\s+usgs_water_query_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b', message):
            return True
            
        return False

    def parse_request(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Parse incoming message to determine request type and parameters."""
        message = message.strip()
        message_lower = message.lower()
        
        # Check for service info
        if message_lower == "tell me about usgs_water":
            return RequestType.SERVICE_INFO, {}
            
        # Check for parameter search
        if message_lower.startswith("usgs_water.parameters"):
            return RequestType.PARAMETER_SEARCH, {
                "query": message[len("usgs_water.parameters"):].strip()
            }
            
        # Check for search execution
        if match := re.match(r'^usgs_water\.search\s+(?:usgs_water_query_)?(\d{8}_\d{6}(?:_orig|_alt\d+))\b', message_lower):
            query_id = match.group(1)
            if not query_id.startswith('usgs_water_query_'):
                query_id = f"usgs_water_query_{query_id}"
            return RequestType.QUERY_SEARCH, {"query_id": query_id}
            
        # Check for simple search (no ID)
        if message_lower.rstrip('.') == 'usgs_water.search':
            return RequestType.QUERY_SEARCH, {"query_id": None}
            
        # Check for natural language query
        if message_lower.startswith("usgs_water:"):
            return RequestType.NATURAL_QUERY, {
                "query": message[11:].strip()  # Remove "usgs_water: "
            }
            
        # Check for dataset conversion
        if match := re.match(r'^convert\s+(usgs_water_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset\b', message_lower):
            return RequestType.DATASET_CONVERSION, {"query_id": match.group(1)}
            
        # Check for code blocks
        if '```usgs_water' in message:
            # Extract query from code block
            match = self.query_block_re.search(message)
            if match:
                try:
                    query_json = match.group(1).strip()
                    # Parse and validate query
                    query = json.loads(query_json)
                    return RequestType.WATER_QUALITY_QUERY, query
                except json.JSONDecodeError:
                    pass
            
        # Default to natural query
        return RequestType.NATURAL_QUERY, {"query": message}

    def execute(self, request: Tuple[str, Dict[str, Any]], context: Dict[str, Any]) -> ServiceResponse:
        """Execute a parsed request."""
        try:
            request_type, params = request
            
            if request_type == RequestType.SERVICE_INFO:
                return self._handle_service_info(params.get("query", ""), context)
                
            elif request_type == RequestType.NATURAL_QUERY:
                return self._handle_natural_query(params["query"], context)
                
            elif request_type == RequestType.PARAMETER_SEARCH:
                return self._handle_parameter_search(params["query"], context)
                
            elif request_type == RequestType.QUERY_SEARCH:
                return self._handle_query_search(params["query_id"], context)
                
            elif request_type == RequestType.DATASET_CONVERSION:
                return self._handle_dataset_conversion(params["query_id"], context)
                
            elif request_type == RequestType.WATER_QUALITY_QUERY:
                return self._handle_water_quality_query(params, context)
                
            else:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=f"Unknown request type: {request_type}",
                        message_type=MessageType.ERROR
                    )]
                )
                
        except Exception as e:
            logger.error(f"Error executing request: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error executing request: {str(e)}",
                    message_type=MessageType.ERROR
                )]
            )

    def get_llm_prompt_addition(self) -> str:
        """Get additional context for LLM prompts."""
        return """
        The USGS Water Quality Portal provides access to water quality monitoring data across the United States.
        
        Key capabilities:
        1. Search for monitoring sites by:
           - Geographic coordinates with radius
           - Bounding box (min/max lat/lon)
           - Site IDs
           
        2. Retrieve water quality data with:
           - Multiple water quality parameters (specified by characteristic name)
           - Date range filtering
           - Site-specific or area-based queries
           
        Requirements:
        - Location searches must use a minimum 20-mile radius
        - All queries must include a start_date
        - Parameters should be specified by their characteristic names (e.g., "Temperature, water" not "00010")
        - Coordinates must be in decimal degrees
        
        Example query formats:
        1. Point search:
        ```usgs_water
        {
            "coordinates": [latitude, longitude],
            "radius": radius_in_miles,  # minimum 20
            "parameters": [
                // Physical parameters
                "Temperature, water",  // Water temperature in degrees Celsius
                "Specific conductance",  // Conductivity in µS/cm
                // Chemical parameters
                "Dissolved oxygen (DO)",  // DO in mg/L
                "pH"  // pH in standard units
            ],
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"  // Optional
        }
        ```
        
        2. Area search (for water bodies):
        ```usgs_water
        {
            "bounds": {
                "min_lat": float,  // Southern boundary
                "max_lat": float,  // Northern boundary
                "min_lon": float,  // Western boundary
                "max_lon": float   // Eastern boundary
            },
            "parameters": [
                // Physical parameters
                "Temperature, water",  // Water temperature in degrees Celsius
                "Specific conductance",  // Conductivity in µS/cm
                // Chemical parameters
                "Dissolved oxygen (DO)",  // DO in mg/L
                "pH"  // pH in standard units
            ],
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"  // Optional
        }
        ```
        
        3. Site-specific search:
        ```usgs_water
        {
            "sites": ["USGS-12345678", "USGS-87654321"],
            "parameters": [
                // Physical parameters
                "Temperature, water",  // Water temperature in degrees Celsius
                "Specific conductance",  // Conductivity in µS/cm
            ],
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"  # optional
        }
        ```
        """

    def _handle_natural_query(self, query: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle a natural language query."""
        try:
            # Validate parameter metadata
            if not self.parameter_metadata or not any(self.parameter_metadata.values()):
                logger.error("Parameter metadata not properly initialized")
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="Unable to process natural language query: Parameter metadata not available. Please try again later or use direct query format.",
                        message_type=MessageType.ERROR
                    )]
                )

            try:
                # Create parameter context with full metadata, emphasizing characteristic names
                param_context = {
                    'total_parameters': len(self.parameter_metadata.get('by_name', {})),
                    'groups': {},
                    'common_parameters': {}
                }
                
                # Build detailed group information
                for group, params in self.parameter_metadata.get('by_group', {}).items():
                    param_context['groups'][group] = {
                        'count': len(params),
                        'parameters': [
                            {
                                'name': p['name'],  # Using 'name' instead of 'parameter_nm'
                                'unit': p.get('unit', ''),
                                'description': p.get('description', ''),
                                'providers': p.get('providers', '').split() if isinstance(p.get('providers', ''), str) else []
                            }
                            for p in params[:10]  # Limit to 10 examples per group
                        ]
                    }
                
                # Add common parameters with typical ranges and units
                common_params = {
                    'Temperature, water': {'min': -5, 'max': 40, 'unit': '°C'},
                    'Dissolved oxygen (DO)': {'min': 0, 'max': 20, 'unit': 'mg/L'},
                    'pH': {'min': 0, 'max': 14, 'unit': 'std'},
                    'Specific conductance': {'min': 0, 'max': 5000, 'unit': 'µS/cm'},
                    'Turbidity': {'min': 0, 'max': 1000, 'unit': 'NTU'}
                }
                
                param_context['common_parameters'] = {
                    name: {
                        'range': ranges,
                        'description': next(
                            (p.get('description', '') for p in self.parameter_metadata.get('by_group', {}).get(
                                next((g for g, params in self.parameter_metadata.get('by_group', {}).items()
                                     for p in params if p['name'] == name), 'Physical'
                            ), [])
                            ),
                            ''
                        )
                    }
                    for name, ranges in common_params.items()
                }
                
                if not param_context['groups']:
                    raise ValueError("No parameter groups available")
                
            except Exception as e:
                logger.error(f"Error creating parameter context: {str(e)}")
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="Unable to process natural language query: Error accessing parameter information. Please try again later or use direct query format.",
                        message_type=MessageType.ERROR
                    )]
                )

            # Define system prompt for natural language query handling
            system_prompt = """You are an expert at converting natural language queries about water quality into structured USGS Water Quality Portal API queries.

Important Requirements:
1. ALWAYS use characteristic names (e.g., "Temperature, water") instead of parameter codes
2. Choose appropriate search type based on the query context:
   - Point-radius for specific locations (cities, monitoring stations)
   - Bounding box for water bodies (rivers, lakes, bays) or regions
3. Point searches must use minimum 20-mile radius
4. All queries must include a start_date
5. Coordinates must be in decimal degrees

Query Format for POINT searches (use for specific locations):
```usgs_water
{
    "coordinates": [latitude, longitude],  // For point searches
    "radius": radius_in_miles,  // Minimum 20
    "parameters": [
        // Physical parameters
        "Temperature, water",  // Water temperature in degrees Celsius
        "Specific conductance",  // Conductivity in µS/cm
        // Chemical parameters
        "Dissolved oxygen (DO)",  // DO in mg/L
        "pH"  // pH in standard units
    ],
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD"  // Optional
}
```

Query Format for AREA searches (use for water bodies or regions):
```usgs_water
{
    "bounds": {
        "min_lat": float,  // Southern boundary
        "max_lat": float,  // Northern boundary
        "min_lon": float,  // Western boundary
        "max_lon": float   // Eastern boundary
    },
    "parameters": [
        // Physical parameters
        "Temperature, water",  // Water temperature in degrees Celsius
        "Specific conductance",  // Conductivity in µS/cm
        // Chemical parameters
        "Dissolved oxygen (DO)",  // DO in mg/L
        "pH"  // pH in standard units
    ],
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD"  // Optional
}
```

Examples of when to use each:
1. Point-radius: 'water quality in Seattle' or 'measurements near station USGS-12345'
2. Bounding box: 'Sacramento River water quality' or 'Lake Tahoe measurements'

Parameter Selection Guidelines:
1. Group parameters by type (physical, chemical, biological, etc.)
2. Include descriptive comments for each parameter with units
3. Consider seasonal relevance and measurement frequencies
4. Use parameter relationships to suggest complementary measurements
5. Consider typical value ranges when relevant

When constructing queries:
1. Use parameter groups to ensure comprehensive coverage
2. Include related parameters that might be relevant
3. Set appropriate temporal ranges based on context
4. Consider seasonal patterns for environmental parameters

IMPORTANT FORMAT REQUIREMENTS:
1. ALWAYS wrap query suggestions in ```usgs_water blocks
2. Provide 2-3 complementary query suggestions that approach the question from different angles
3. Each suggestion should have a clear explanation of its focus and why it's relevant
4. Format each suggestion block as:
   💡 SUGGESTION [focus/approach]:
   [explanation of approach and parameter choices]
   ```usgs_water
   [query JSON with parameter explanations]
   ```
5. ALWAYS include descriptive comments for parameters explaining:
   - What the parameter measures
   - Units of measurement
   - Typical value ranges if known
   - Seasonal or environmental relevance."""

            # Create messages for LLM
            messages = [{
                'role': 'user',
                'content': f"""Query Context:
Parameter Groups and Examples:
{json.dumps(param_context, indent=2)}

Natural Language Query: "{query}"

IMPORTANT: Your response MUST:
1. Choose appropriate search type (point-radius vs bounding box) based on the query
2. Document ALL parameter codes with their names and units in comments
3. Group parameters by type (physical, chemical, etc.)
4. Provide 2-3 complementary query suggestions
5. Explain the focus and relevance of each suggestion

For each suggestion:
1. Explain its specific focus and approach
2. Include relevant parameters from appropriate groups
3. Consider temporal and spatial aspects
4. Add helpful context about the parameters chosen"""
            }]
            
            # Add chat history if available
            if context.get('chat_history'):
                messages.append({
                    'role': 'user',
                    'content': "\nPrevious conversation:\n" + "\n".join(
                        f"{msg['role']}: {msg['content']}" for msg in context['chat_history']
                    )
                })
            
            # Get LLM response
            logger.info(f"Sending natural language query to LLM: '{query}'")
            try:
                response = self._call_llm(messages=messages, system_prompt=system_prompt)
                logger.info("Successfully received response from LLM")
            except Exception as llm_error:
                logger.error(f"LLM call failed: {str(llm_error)}", exc_info=True)
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=f"Error processing natural language query with LLM: {str(llm_error)}. This may be due to model compatibility issues. Please try again or use direct query format.",
                        message_type=MessageType.ERROR
                    )]
                )
            
            # Add query IDs to code blocks
            response_with_ids = self.add_ids_to_blocks(response)
            
            # Create service response
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=response_with_ids,
                        message_type=MessageType.SUGGESTION
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error handling natural query: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Error processing natural language query. Please try again or use direct query format.",
                    message_type=MessageType.ERROR
                )]
            )

    def _handle_water_quality_query(self, query: Dict, context: Dict[str, Any]) -> ServiceResponse:
        """Handle a water quality data query."""
        try:
            # Store context for use in _call_llm
            self.context = context  # Add this line to set context
            
            # Extract query parameters and normalize location format
            location = None
            if 'coordinates' in query and 'radius' in query:
                location = {
                    'lat': query['coordinates'][0],
                    'lon': query['coordinates'][1],
                    'radius': query['radius']
                }
            elif 'bounds' in query:
                location = query['bounds']
            elif 'sites' in query:
                location = {'sites': query['sites']}
            
            parameters = query.get('parameters', [])
            start_date = query.get('start_date')
            end_date = query.get('end_date')
            
            if not location:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No location specification in query",
                        message_type=MessageType.ERROR
                    )]
                )
            
            # Generate query ID
            query_id = PreviewIdentifier.create_id(prefix="usgs_water_query")
            
            # Get raw data from water quality service
            logger.info("Retrieving raw data from API...")
            raw_data, api_metadata = self.water_quality.get_data(
                location=location,
                parameters=parameters,
                start_date=start_date,
                end_date=end_date
            )
            
            if raw_data.empty:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No data found for the specified query",
                        message_type=MessageType.WARNING
                    )]
                )

            # Log data structure for debugging
            logger.info(f"Raw data columns: {raw_data.columns.tolist()}")
            logger.info(f"Total records: {len(raw_data)}")
            
            # Verify essential columns are present
            required_columns = [
                'Date', 'Site ID', 'Organization ID', 'Organization Name',
                'Latitude', 'Longitude'
            ]
            
            missing_columns = [col for col in required_columns if col not in raw_data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=f"Error: Missing required columns in API response: {', '.join(missing_columns)}",
                        message_type=MessageType.ERROR
                    )]
                )

            # Create unified DataFrame with proper column names and types
            unified_df = raw_data.copy()
            
            # Ensure datetime type for Date column
            unified_df['Date'] = pd.to_datetime(unified_df['Date'])
            
            # Ensure numeric types for coordinates
            unified_df['Latitude'] = pd.to_numeric(unified_df['Latitude'], errors='coerce')
            unified_df['Longitude'] = pd.to_numeric(unified_df['Longitude'], errors='coerce')
            
            # Get statistics from api_metadata
            stats = api_metadata.get('characteristics_stats', {})
            if not stats:
                logger.error("No statistics found in API metadata")
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="Error: No statistics available for analysis",
                        message_type=MessageType.ERROR
                    )]
                )

            # Store the full result
            if 'successful_queries_store' in context:
                context['successful_queries_store'][query_id] = {
                    'dataframe': unified_df.to_dict('records'),
                    'query': query,  # Preserve the original query
                    'metadata': {
                        'execution_time': datetime.now().isoformat(),
                        'total_records': len(unified_df),
                        'unique_sites': stats['overall']['unique_sites'],
                        'date_range': stats['overall']['date_range'],
                        'geographic_bounds': stats['overall']['geographic_bounds'],
                        'characteristics': {
                            'requested': stats['requested'],
                            'additional': stats['additional']
                        },
                        'api_info': api_metadata
                    }
                }

            # Create preview with key information
            desired_preview_cols = ['Date', 'Site ID', 'Site Name', 'Organization Name']
            # Filter to only columns that exist in the DataFrame
            preview_cols = [col for col in desired_preview_cols if col in unified_df.columns]
            
            # Add requested characteristic columns that exist in the DataFrame
            if stats['requested']:
                for char in stats['requested'].keys():
                    if char in unified_df.columns:
                        preview_cols.append(char)
            
            # Add coordinates if they exist
            if 'Latitude' in unified_df.columns and 'Longitude' in unified_df.columns:
                preview_cols.extend(['Latitude', 'Longitude'])
            
            # Get first few rows
            preview_df = unified_df.head(5)[preview_cols]
            
            # Format the preview message (this is the RESULT)
            result_message = f"""Query Results (ID: {query_id}):
- Total Records: {len(unified_df):,}
- Time Range: {stats['overall']['date_range']['start']} to {stats['overall']['date_range']['end']} ({stats['overall']['date_range']['days']} days)
- Unique Sites: {stats['overall']['unique_sites']}
- Geographic Coverage: {stats['overall']['geographic_bounds']['min_lat']:.4f}°N to {stats['overall']['geographic_bounds']['max_lat']:.4f}°N, 
                      {stats['overall']['geographic_bounds']['min_lon']:.4f}°W to {stats['overall']['geographic_bounds']['max_lon']:.4f}°W

Characteristics Summary:"""

            # Add requested characteristics first
            if stats['requested']:
                result_message += "\n\nRequested Characteristics:"
                for char, char_stats in stats['requested'].items():
                    result_message += f"\n- {char}:"
                    result_message += f"\n  • {char_stats['count']:,} measurements from {char_stats['sites']['count']} sites"
                    result_message += f"\n  • Range: {char_stats['min']:.2f} to {char_stats['max']:.2f} {char_stats['units'][0] if char_stats['units'] else ''}"
                    result_message += f"\n  • Mean: {char_stats['mean']:.2f} ± {char_stats['std']:.2f}"

            # Add additional characteristics if any
            if stats['additional']:
                result_message += "\n\nAdditional Characteristics Found:"
                for char, char_stats in stats['additional'].items():
                    result_message += f"\n- {char}:"
                    result_message += f"\n  • {char_stats['count']:,} measurements from {char_stats['sites']['count']} sites"
                    result_message += f"\n  • Range: {char_stats['min']:.2f} to {char_stats['max']:.2f} {char_stats['units'][0] if char_stats['units'] else ''}"
                    result_message += f"\n  • Mean: {char_stats['mean']:.2f} ± {char_stats['std']:.2f}"

            result_message += "\n\nData Preview (first 5 rows with measurements):\n```\n"
            result_message += preview_df.to_string()
            result_message += "\n```\n\nTo convert these results to a dataset, use: `convert " + query_id + " to dataset`"

            # Generate summary using LLM
            summary = self.summarize(unified_df, context.get('chat_history', []), {
                'requested_params': parameters,  # explicitly pass requested parameters
                'query': query,
                'summary_stats': stats,  # explicitly pass summary_stats
                'preview': preview_df,
                'api_metadata': api_metadata
            })
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=result_message,
                        message_type=MessageType.RESULT
                    ),
                    ServiceMessage(
                        service=self.name,
                        content=summary,
                        message_type=MessageType.SUMMARY
                    )
                ],
                store_updates={'successful_queries_store': {query_id: {
                    'dataframe': unified_df.to_dict('records'),
                    'query': query,  # Preserve the original query
                    'metadata': context['successful_queries_store'][query_id]['metadata']
                }}}
            )
            
        except Exception as e:
            logger.error(f"Error handling water quality query: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error processing query: {str(e)}",
                    message_type=MessageType.ERROR
                )]
            )

    def summarize(self, data: pd.DataFrame, chat_history: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        """Generate a summary of the query results."""
        try:
            # Get requested parameters and statistics from context
            query = context.get('query', {})
            requested_params = set(query.get('parameters', []))
            stats = context.get('summary_stats', {})  # Get stats directly from context
            messages = context.get('api_metadata', {}).get('messages', [])
            
            if not stats:
                logger.warning("No statistics found in context")
                return "Error: No statistics available for analysis"
            
            # Create summary statistics structure with enhanced context
            summary_stats = {
                'query_info': {
                    'parameters': list(requested_params),
                    'location': query.get('coordinates') or query.get('bounds') or query.get('sites', []),
                    'time_range': {
                        'start': query.get('start_date'),
                        'end': query.get('end_date')
                    }
                },
                'data_coverage': {
                    'total_measurements': stats['overall']['total_records'],
                    'unique_sites': stats['overall']['unique_sites'],
                    'date_range': stats['overall']['date_range'],
                    'geographic_bounds': stats['overall']['geographic_bounds']
                },
                'parameter_analysis': {
                    'requested': {
                        name: {
                            'measurements': info['count'],
                            'sites': info['sites']['count'],
                            'statistics': {
                                'min': info['min'],
                                'max': info['max'],
                                'mean': info['mean'],
                                'std': info['std']
                            },
                            'units': info['units'][0] if info['units'] else None
                        }
                        for name, info in stats['requested'].items()
                    },
                    'additional': {
                        name: {
                            'measurements': info['count'],
                            'sites': info['sites']['count'],
                            'statistics': {
                                'min': info['min'],
                                'max': info['max'],
                                'mean': info['mean'],
                                'std': info['std']
                            },
                            'units': info['units'][0] if info['units'] else None
                        }
                        for name, info in stats['additional'].items()
                    }
                },
                'quality_indicators': {
                    'messages': messages,
                    'coverage_completeness': {
                        param: stats['requested'][param]['count'] / stats['overall']['total_records'] * 100
                        for param in stats['requested']
                    }
                }
            }
            
            # Create system prompt for summary generation
            system_prompt = """You are a water quality scientist analyzing USGS monitoring data. Create a focused analytical summary that:

1. Evaluates Data Quality and Coverage:
   - Analyze temporal and spatial coverage comprehensiveness
   - Assess measurement frequency and distribution
   - Identify any significant data gaps or limitations
   - Evaluate the reliability of the measurements

2. Analyze Parameter Relationships and Patterns:
   For Requested Parameters:
   - Compare values against typical ranges for the water body type
   - Identify potential correlations between parameters
   - Flag any unusual patterns or potential anomalies
   - Assess the completeness of critical parameter combinations

   For Additional Parameters Found:
   - Explain their relevance to the requested parameters
   - Highlight complementary measurements that enhance understanding
   - Note parameters that help explain observed patterns

3. Provide Scientific Context:
   - Interpret the results in the context of water quality standards
   - Consider seasonal and geographic influences
   - Identify potential environmental processes or conditions
   - Note any anthropogenic influences suggested by the data

4. Make Evidence-Based Recommendations:
   - Suggest additional parameters that would enhance the analysis
   - Identify specific areas needing additional sampling
   - Recommend complementary measurements for validation
   - Propose focused areas for further investigation

Keep the summary scientifically rigorous but accessible. Focus on patterns and relationships that advance understanding of the water system's behavior."""
            
            # Create messages for LLM with enhanced context
            messages = [{
                'role': 'user',
                'content': f"""Analyze this water quality dataset with the following context:

Query Parameters:
- Requested: {', '.join(requested_params) if requested_params else 'None specified'}
- Time Range: {query.get('start_date')} to {query.get('end_date', 'present')}
- Location: {json.dumps(query.get('coordinates') or query.get('bounds') or query.get('sites', []), indent=2)}

Data Statistics:
{json.dumps(summary_stats, indent=2)}

Focus your analysis on:
1. Data quality and coverage assessment
2. Parameter relationships and patterns
3. Environmental context and implications
4. Evidence-based recommendations for future measurements"""
            }]
            
            # Add chat history if available
            if chat_history:
                messages.append({
                    'role': 'user',
                    'content': "\nPrevious conversation context:\n" + "\n".join(
                        f"{msg['role']}: {msg['content']}" for msg in chat_history[-3:]  # Only last 3 messages
                    )
                })
            
            # Get LLM response
            return self._call_llm(messages=messages, system_prompt=system_prompt)
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return "Error generating summary"

    def _handle_query_search(self, query_id: Optional[str], context: Dict[str, Any]) -> ServiceResponse:
        """Handle a query search request."""
        try:
            store = context.get('successful_queries_store', {})
            
            # First, check the store directly
            if query_id and query_id in store:
                stored_query = store[query_id]['query']
                return self._handle_water_quality_query(stored_query, context)
            
            # Fallback to chat history parsing if not found in store
            chat_history = context.get('chat_history', [])
            if not chat_history:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No chat history available. Please create a new query using: `usgs_water: your question`",
                        message_type=MessageType.ERROR
                    )]
                )

            most_recent_orig = None

            for msg in reversed(chat_history):
                if msg.get('role') == 'assistant' and '```usgs_water' in msg.get('content', ''):
                    content = msg.get('content', '')
                    blocks = re.finditer(r'```usgs_water\s*(.*?)```', content, re.DOTALL)
                    for block_match in blocks:
                        block_content = block_match.group(1).strip()
                        id_match = re.search(r'--\s*Query ID:\s*(usgs_water_query_\d{8}_\d{6}(?:_orig|_alt\d+))\b', block_content)
                        if not id_match:
                            continue
                        block_id = id_match.group(1)
                        if query_id and block_id == query_id:
                            query_text = block_content.split('--')[0].strip()
                            clean_query = re.sub(r'//.*$', '', query_text, flags=re.MULTILINE)
                            query = json.loads(clean_query)
                            return self._handle_water_quality_query(query, context)
                        elif not query_id and '_orig' in block_id:
                            query_text = block_content.split('--')[0].strip()
                            clean_query = re.sub(r'//.*$', '', query_text, flags=re.MULTILINE)
                            query = json.loads(clean_query)
                            most_recent_orig = query
                            break

            if not query_id and most_recent_orig:
                return self._handle_water_quality_query(most_recent_orig, context)

            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Query ID '{query_id}' not found in store or chat history.",
                    message_type=MessageType.ERROR
                )]
            )

        except Exception as e:
            logger.error(f"Error handling query search: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error executing query: {str(e)}",
                    message_type=MessageType.ERROR
                )]
            )

    def _handle_dataset_conversion(self, query_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle dataset conversion request."""
        try:
            # Get the query result
            store = context.get('successful_queries_store', {})
            if query_id not in store:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=f"Query ID {query_id} not found in query store",
                        message_type=MessageType.ERROR
                    )]
                )
            
            stored_data = store[query_id]
            
            # Convert stored records back to DataFrame
            data = pd.DataFrame.from_records(stored_data['dataframe'])
            metadata = stored_data['metadata']
            
            if data is None or data.empty:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No data available for conversion",
                        message_type=MessageType.ERROR
                    )]
                )
            
            # Create dataset ID
            dataset_id = query_id  # Use the original query ID directly
            
            # Generate profile report
            profile = ProfileReport(data, title=f"USGS Water Quality Dataset {dataset_id}", minimal=True)
            profile_html = profile.to_html()

            # Create dataset entry explicitly at conversion time
            dataset_entry = {
                'df': data.to_dict('records'),  # Changed from 'dataframe' to 'df'
                'metadata': {
                    'source': f"USGS Water Quality Query {query_id}",
                    'creation_time': datetime.now().isoformat(),
                    'query': stored_data['query'],
                    'rows': len(data),
                    'columns': list(data.columns),
                    'characteristics': metadata.get('characteristics', {}),
                    'geographic_bounds': metadata.get('geographic_bounds', {}),
                    'date_range': metadata.get('date_range', {}),
                    'unique_sites': metadata.get('unique_sites', 0),
                    'selectable': True,
                    'profile_report': profile_html
                }
            }
            
            # Debug logging statement
            logger.debug(f"Dataset {dataset_id} created with {len(data)} rows, {len(data.columns)} columns")
            
            # Format response message
            response = f"""Created dataset {dataset_id}

Dataset Summary:
- Total Records: {len(data):,}
- Time Range: {metadata['date_range']['start']} to {metadata['date_range']['end']}
- Unique Sites: {metadata['unique_sites']}
- Geographic Coverage: {metadata['geographic_bounds']['min_lat']:.4f}°N to {metadata['geographic_bounds']['max_lat']:.4f}°N, 
                      {metadata['geographic_bounds']['min_lon']:.4f}°W to {metadata['geographic_bounds']['max_lon']:.4f}°W
- Total Columns: {len(data.columns)}"""
            
            datasets = context.get('datasets_store', {})

            # Update dataset store explicitly
            datasets[dataset_id] = dataset_entry

            # Return the entire updated dataset store
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=response,
                        message_type=MessageType.INFO
                    )
                ],
                store_updates={
                    'datasets_store': datasets,
                    'successful_queries_store': {k: v for k, v in store.items() if k != query_id}
                }
            )
            
        except Exception as e:
            logger.error(f"Error converting to dataset: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error converting to dataset: {str(e)}",
                    message_type=MessageType.ERROR
                )]
            )

    def add_ids_to_blocks(self, text: str) -> str:
        """Add query IDs to USGS water query blocks in text."""
        if not text or '```usgs_water' not in text:
            return text
            
        # Track if we've marked a block as primary
        has_primary = False
        # Track alternative numbers used
        alt_numbers = set()
        # Store processed blocks to avoid duplicates
        processed_blocks = set()
        
        def replace_block(match) -> str:
            nonlocal has_primary, alt_numbers
            block = match.group(1).strip()
            
            # Skip if block is empty or just whitespace
            if not block or block.isspace():
                return match.group(0)
            
            # Skip if block already has an ID
            if re.search(r'--\s*Query ID:\s*usgs_water_query_\d{8}_\d{6}(?:_orig|_alt\d+)\b', block):
                # Extract existing alt number if present
                if '_alt' in block:
                    try:
                        alt_num = int(re.search(r'_alt(\d+)', block).group(1))
                        alt_numbers.add(alt_num)
                    except (AttributeError, ValueError):
                        pass
                return match.group(0)
            
            try:
                # Store original block with comments
                original_block = block
                
                # Remove comments only for JSON parsing
                clean_block = re.sub(r'//.*$', '', block, flags=re.MULTILINE)
                # Parse the cleaned block
                query = json.loads(clean_block)
                if not isinstance(query, dict):
                    return match.group(0)
                
                # Skip if we've already processed this exact block
                block_hash = json.dumps(query, sort_keys=True)
                if block_hash in processed_blocks:
                    return ""  # Remove duplicate blocks
                processed_blocks.add(block_hash)
                
                # Generate new query ID
                query_id = PreviewIdentifier.create_id(prefix="usgs_water_query")
                
                # Add suffix based on whether this is primary or alternative
                if not has_primary:
                    query_id = query_id.replace('_orig', '_orig')  # Ensure primary query
                    has_primary = True
                else:
                    # Find the next available alternative number
                    alt_num = 1
                    while alt_num in alt_numbers:
                        alt_num += 1
                    alt_numbers.add(alt_num)
                    query_id = query_id.replace('_orig', f'_alt{alt_num}')
                
                # Format block with ID, preserving original formatting and comments
                return f"```usgs_water\n{original_block}\n\n-- Query ID: {query_id}\n```"
            except json.JSONDecodeError:
                return match.group(0)
        
        # Process all blocks
        processed_text = self.query_block_re.sub(replace_block, text)
        
        # Remove any empty lines created by removing duplicates
        return '\n'.join(line for line in processed_text.split('\n') if line.strip())

    def get_help_text(self) -> str:
        """Get help text for service commands."""
        return (
            "\n🌊 **USGS Water Quality Service**\n\n"
            "The USGS Water Quality service provides access to nationwide water quality monitoring data "
            "through natural language queries and direct commands.\n\n"
            "**Commands:**\n"
            "- **Service Information:**\n"
            "  `tell me about usgs_water`\n\n"
            "- **Parameter Search:**\n"
            "  `usgs_water.parameters <search term>`\n"
            "  Example: `usgs_water.parameters temperature`\n\n"
            "- **Natural Language Queries:**\n"
            "  `usgs_water: <your question>`\n"
            "  Example: `usgs_water: what's the water temperature in the Sacramento River?`\n\n"
            "- **Query Management:**\n"
            "  - View last query: `usgs_water.search`\n"
            "  - View specific query: `usgs_water.search <query_id>`\n"
            "  - Convert to dataset: `convert <query_id> to dataset`\n\n"
            "**Data Coverage:**\n"
            "- Nationwide USGS monitoring network\n"
            "- 3000+ water quality parameters\n"
            "- Historical and real-time data\n"
            "- USGS certified measurements\n\n"
            "**Tips:**\n"
            "- Include specific locations for better results\n"
            "- Specify time periods when relevant\n"
            "- Use parameter names from `usgs_water.parameters` for precision\n"
            "- Convert queries to datasets for further analysis\n"
        )

    def process_message(self, message: ServiceMessage) -> ServiceResponse:
        """Process an incoming message."""
        try:
            if message.message_type == MessageType.QUERY:
                # Parse and execute
                request = self.parse_request(message.content)
                return self.execute(request, {'chat_history': []})
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="Unsupported message type",
                    message_type=MessageType.ERROR
                )]
            )
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error processing request: {str(e)}",
                    message_type=MessageType.ERROR
                )]
            )

    def _handle_parameter_search(self, query: str, context: Dict[str, Any]) -> ServiceResponse:
        try:
            water_quality_service = WaterQualityService()
            if not query.strip():
                # Provide categorical summary if no query specified
                param_summary = water_quality_service.create_llm_context()
                summary_message = "Available parameter categories and their significance:\n"
                for category, details in param_summary['parameters']['categories'].items():
                    examples = ', '.join(details['examples'])
                    summary_message += f"\n{category} ({details['count']} parameters): Examples include {examples}."
                return ServiceResponse(messages=[ServiceMessage(
                    service=self.name,
                    content=summary_message,
                    message_type=MessageType.INFO
                )])

            matching_params = water_quality_service.find_parameters(query)
            if not matching_params:
                return ServiceResponse(messages=[ServiceMessage(
                    service=self.name,
                    content=f"No parameters found matching '{query}'.",
                    message_type=MessageType.INFO
                )])

            # Prepare LLM summarization
            system_prompt = """You are an expert on USGS water quality parameters. Provide a concise summary explaining the relevance and significance of the parameters matching the user's query."""
            param_list = "\n".join([f"- {param['parameter_nm']} (Code: {param.get('parameter_cd', 'N/A')})" for param in matching_params])
            messages = [{
                'role': 'user',
                'content': f"Parameters matching '{query}':\n{param_list}\n\nExplain their significance."
            }]

            interpretation = self._call_llm(messages=messages, system_prompt=system_prompt)

            return ServiceResponse(messages=[ServiceMessage(
                service=self.name,
                content=interpretation,
                message_type=MessageType.INFO
            )])

        except Exception as e:
            logger.error(f"Error searching parameters: {str(e)}")
            return ServiceResponse(messages=[ServiceMessage(
                service=self.name,
                content=f"Error searching parameters: {str(e)}",
                message_type=MessageType.ERROR
            )])

    def _handle_service_info(self, query: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle service information request."""
        try:
            # Get service information from water quality service
            service_info = json.loads(self.water_quality.get_service_info())
            
            # Create system prompt for service summary
            system_prompt = """You are an expert on the USGS Water Quality Portal. Create a clear, informative summary that:

1. Explains the service's core capabilities:
   - Types of water quality data available
   - Geographic coverage
   - Temporal coverage
   - Parameter categories and measurements

2. Highlights key features:
   - Search capabilities (location, parameter, time-based)
   - Data formats and access methods
   - Real-time vs historical data
   - Quality control and data validation

3. Provides practical usage guidance:
   - Common use cases and examples
   - Best practices for queries
   - Important limitations or constraints
   - Tips for getting the best results

Keep the summary informative but concise, focusing on what would be most useful for someone starting to work with the service."""
            
            # Create messages for LLM
            messages = [{
                'role': 'user',
                'content': f"Service Information:\n{json.dumps(service_info, indent=2)}\n\nPlease create a user-friendly summary of the USGS Water Quality Portal service."
            }]
            
            # Get LLM response
            summary = self._call_llm(messages=messages, system_prompt=system_prompt)
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=summary,
                        message_type=MessageType.INFO
                    )
                ]
            )
            
        except Exception as e:
            logger.error(f"Error handling service info request: {str(e)}", exc_info=True)
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error retrieving service information: {str(e)}",
                    message_type=MessageType.ERROR
                )]
            ) 
