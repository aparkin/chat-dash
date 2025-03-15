"""Direct API implementation for accessing the USGS Water Quality Portal."""

import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import json
import os
import io
import re
import zipfile
import time
import numpy as np

# Configure logging with a more concise format
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message).200s'  # Truncate messages to 200 chars
)
logger = logging.getLogger(__name__)

# Reduce logging level for some messages
logger.setLevel(logging.WARNING)  # Only show WARNING and above by default

# Column definitions for data processing
NUMERIC_COLUMNS = {
    # Result values
    'ResultMeasureValue': float,
    'ResultDepthHeightMeasure/MeasureValue': float,
    'DetectionQuantitationLimitMeasure/MeasureValue': float,
    'ResultDepthAltitudeReferencePointMeasure/MeasureValue': float,
    'ActivityDepthHeightMeasure/MeasureValue': float,
    'SampleDepthHeightMeasure/MeasureValue': float,
    
    # Location coordinates
    'ActivityLocation/LatitudeMeasure': float,
    'ActivityLocation/LongitudeMeasure': float,
    'HorizontalAccuracyMeasure/MeasureValue': float,
    'VerticalMeasure/MeasureValue': float,
    'VerticalAccuracyMeasure/MeasureValue': float,
    
    # Flow and measurement values
    'FlowRateMeasure/MeasureValue': float,
    'VelocityMeasure/MeasureValue': float,
    'HeightMeasure/MeasureValue': float,
    'WaterLevelMeasure/MeasureValue': float
}

# Date columns that should be converted to datetime
DATE_COLUMNS = [
    'ActivityStartDate',
    'ActivityEndDate',
    'ResultAnalyticalMethod/MethodModificationDate',
    'AnalysisStartDate',
    'AnalysisEndDate',
    'SamplePreparationStartDate'
]

# Columns that form the unique index for pivoting
INDEX_COLUMNS = [
    'ActivityStartDate',
    'OrganizationIdentifier',
    'OrganizationFormalName',
    'MonitoringLocationIdentifier'
]

# Activity columns to pivot alongside the main value
ACTIVITY_COLUMNS = [
    'ResultMeasure/MeasureUnitCode',
    'MeasureQualifierCode',
    'ResultStatusIdentifier',
    'ResultCommentText',
    'ResultAnalyticalMethod/MethodIdentifier',
    'ResultAnalyticalMethod/MethodName',
    'ResultDepthHeightMeasure/MeasureValue',
    'ResultDepthHeightMeasure/MeasureUnitCode',
    'ResultDepthAltitudeReferencePointText',
    'ActivityCommentText',
    'ResultValueTypeName',
    'StatisticalBaseCode',
    'ProviderName'
]

class WaterQualityAPI:
    """Direct implementation of the Water Quality Portal API."""
    
    _instance = None
    
    BASE_URL = "https://www.waterqualitydata.us/data"
    STATION_URL = "https://www.waterqualitydata.us/Station/search"
    PARAM_URL = "https://www.waterqualitydata.us/Codes/Characteristicname"
    
    def __new__(cls):
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(WaterQualityAPI, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the API client."""
        # Only initialize once
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.session = requests.Session()
        self._parameters = None
        self._load_parameters()
        self._initialized = True
    
    def _make_request(self, url: str, params: Dict, timeout: int = 300) -> Optional[pd.DataFrame]:
        """Make a request to the API and return results as a DataFrame."""
        try:
            # Create ordered parameter list to match working URL pattern
            query_params = []
            
            # 1. Location parameters first
            if 'within' in params:
                query_params.extend([
                    ('within', str(params['within'])),
                    ('lat', str(params['lat'])),
                    ('long', str(params['long']))
                ])
            elif 'bBox' in params:
                query_params.append(('bBox', params['bBox']))
            elif 'siteid' in params:
                query_params.append(('siteid', params['siteid']))
            
            # 2. Date parameters
            if 'startDateLo' in params:
                query_params.append(('startDateLo', params['startDateLo']))
            if 'startDateHi' in params:
                query_params.append(('startDateHi', params['startDateHi']))
            
            # 3. Characteristic name (if provided)
            if 'characteristicName' in params:
                # Split on semicolon and add each characteristic separately
                chars = params['characteristicName'].split(';')
                for char in chars:
                    if char.strip():  # Only add non-empty characteristics
                        query_params.append(('characteristicName', char.strip()))
            
            # 4. Format parameters
            mime_type = params.get('mimeType', 'csv')
            query_params.append(('mimeType', mime_type))
            
            if url == self.PARAM_URL:
                # For parameters endpoint, we need to request JSON
                mime_type = 'json'
                query_params = [('mimeType', 'json')]
                headers = {'Accept': 'application/json'}
            else:
                # For data endpoints, use CSV with zip
                query_params.append(('zip', params.get('zip', 'yes')))
                headers = {'Accept': 'text/csv'}
                
                # 5. Data profile
                if 'dataProfile' in params:
                    query_params.append(('dataProfile', params['dataProfile']))
                
                # 6. Providers (must be last)
                if 'providers' in params:
                    providers = params['providers'].split(',') if isinstance(params['providers'], str) else params['providers']
                    for provider in providers:
                        query_params.append(('providers', provider.strip()))
            
            # Log the full URL and parameters
            full_url = f"{url}?{'&'.join(f'{k}={requests.utils.quote(str(v))}' for k,v in query_params)}"
            logger.info(f"Making request to: {full_url}")
            logger.info(f"Request parameters: {json.dumps(dict(query_params), indent=2)}")
            
            # Make request with properly encoded parameters
            response = self.session.get(url, params=query_params, headers=headers, timeout=timeout)
            
            if response.status_code != 200:
                logger.error(f"Request failed with status {response.status_code}")
                logger.error(f"Full URL that failed: {response.url}")
                logger.error(f"Response content: {response.text[:1000]}")  # Log first 1000 chars of error
                return None
            
            # Log successful response info
            logger.info(f"Request successful: {response.url}")
            logger.info(f"Response content type: {response.headers.get('content-type', 'unknown')}")
            
            # Handle response based on content type
            if url == self.PARAM_URL:
                # Handle JSON response for parameters
                data = response.json()
                if not isinstance(data, dict) or 'codes' not in data:
                    logger.error(f"Unexpected response format from parameters endpoint")
                    return None
                    
                # Convert JSON to DataFrame - use actual response format
                parameters = []
                for code in data['codes']:
                    name = code.get('value', '')
                    if not name:  # Skip entries without a name
                        continue
                        
                    # Extract parameter code if present in the name
                    param_code = ''
                    if '(' in name and ')' in name:
                        # Try to extract code from parentheses
                        code_match = re.search(r'\((\d+)\)', name)
                        if code_match:
                            param_code = code_match.group(1)
                    
                    parameters.append({
                        'parameter_cd': param_code,
                        'parameter_nm': name,
                        'providers': code.get('providers', '')  # Providers might be string or list
                    })
                return pd.DataFrame(parameters)
            else:
                # Handle CSV response for data
                content_type = response.headers.get('content-type', '')
                if 'zip' in content_type or params.get('zip') == 'yes':
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        csv_filename = zf.namelist()[0]
                        with zf.open(csv_filename) as f:
                            df = pd.read_csv(f, low_memory=False)
                else:
                    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), low_memory=False)
                
                # Add diagnostic logging for columns
                logger.info(f"Total columns: {len(df.columns)}")
                geo_cols = [col for col in df.columns if 'Latitude' in col or 'Longitude' in col]
                logger.info(f"Geographic columns found: {geo_cols}")
                return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text[:1000]}")
            return None
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return None
    
    def _load_parameters(self) -> None:
        """Load parameter metadata from the API."""
        try:
            # Try to load from cache first
            cache_file = os.path.join(os.path.dirname(__file__), 'parameter_cache.json')
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cached = json.load(f)
                        if cached.get('timestamp'):
                            # Check if cache is less than 24 hours old
                            cache_time = datetime.fromisoformat(cached['timestamp'])
                            if (datetime.now() - cache_time).days < 1:
                                self._parameters = cached
                                logger.info("Loaded parameters from cache")
                                return
                except Exception as e:
                    logger.warning(f"Error reading parameter cache: {str(e)}")
            
            # Get parameters from API
            params = {
                'mimeType': 'json'  # Parameters endpoint requires JSON
            }
            
            response = self._make_request(self.PARAM_URL, params)
            
            if response is None:
                logger.error("No valid response from API")
                return
            
            try:
                parameters = []
                groups = {}
                
                # Process each parameter from the response DataFrame
                for _, row in response.iterrows():
                    try:
                        # Extract values with safe access
                        name = str(row.get('parameter_nm', ''))
                        if not name or name.lower() == 'nan':
                            continue
                            
                        param = {
                            'parameter_cd': str(row.get('parameter_cd', '')),
                            'parameter_nm': name,
                            'parameter_group': self._categorize_parameter(name),
                            'providers': row.get('providers', '')
                        }
                        
                        # Clean up values
                        param = {k: v if v.lower() != 'nan' else '' for k, v in param.items()}
                        
                        # Only add if we have the minimum required fields
                        if param['parameter_nm']:
                            parameters.append(param)
                            
                            # Track group statistics
                            group = param['parameter_group']
                            if group not in groups:
                                groups[group] = {
                                    'count': 0,
                                    'examples': []
                                }
                            groups[group]['count'] += 1
                            if len(groups[group]['examples']) < 3:
                                groups[group]['examples'].append(param['parameter_nm'])
                    
                    except Exception as e:
                        logger.debug(f"Skipped parameter entry due to: {str(e)}")
                        continue
                
                if not parameters:
                    logger.error("No valid parameters found in API response")
                    return
                
                # Track provider statistics
                provider_stats = {}
                provider_overlap = {
                    'single_provider': 0,
                    'two_providers': 0,
                    'all_providers': 0
                }
                
                for param in parameters:
                    # Count by provider - handle both string and list formats
                    providers = param['providers']
                    if isinstance(providers, str):
                        providers = providers.split()
                    
                    for provider in providers:
                        if provider not in provider_stats:
                            provider_stats[provider] = 0
                        provider_stats[provider] += 1
                    
                    # Track overlap
                    provider_count = len(providers)
                    if provider_count == 1:
                        provider_overlap['single_provider'] += 1
                    elif provider_count == 2:
                        provider_overlap['two_providers'] += 1
                    elif provider_count == 3:
                        provider_overlap['all_providers'] += 1
                
                self._parameters = {
                    'parameters': parameters,
                    'metadata': {
                        'total_count': len(parameters),
                        'groups': groups,
                        'providers': provider_stats,
                        'provider_overlap': provider_overlap
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"Loaded {len(parameters)} unique parameters across {len(groups)} groups")
                logger.info("Provider coverage: " + ", ".join(f"{k}: {v}" for k, v in provider_stats.items()))
                logger.info("Provider overlap: " + 
                          f"{provider_overlap['single_provider']} parameters from single provider, " +
                          f"{provider_overlap['two_providers']} from two providers, " +
                          f"{provider_overlap['all_providers']} from all three providers")
                
                # Cache the results
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(self._parameters, f)
                    logger.info("Cached parameters to file")
                except Exception as e:
                    logger.warning(f"Error caching parameters: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error processing parameters: {str(e)}", exc_info=True)
                return
            
        except Exception as e:
            logger.error(f"Error in _load_parameters: {str(e)}", exc_info=True)
    
    def _categorize_parameter(self, name: str) -> str:
        """Categorize a parameter based on its name.
        
        Args:
            name: Parameter name
            
        Returns:
            Category name
        """
        if not name:
            return 'Other'
            
        name_lower = name.lower()
        
        # Define category keywords with more comprehensive matches
        categories = {
            'Physical': [
                'temperature', 'temp.', 'conductivity', 'specific conductance',
                'ph', 'turbidity', 'dissolved oxygen', 'pressure', 'depth',
                'flow', 'discharge', 'velocity', 'elevation', 'stage height',
                'transparency', 'color', 'density', 'salinity', 'hardness'
            ],
            'Nutrients': [
                'nitrogen', 'phosphorus', 'nitrate', 'nitrite', 'phosphate',
                'ammonia', 'ammonium', 'orthophosphate', 'total n', 'total p',
                'kjeldahl', 'nutrient'
            ],
            'Metals': [
                'iron', 'copper', 'lead', 'zinc', 'mercury', 'cadmium', 'arsenic',
                'chromium', 'nickel', 'aluminum', 'manganese', 'silver', 'selenium',
                'metal', 'antimony', 'beryllium', 'cobalt', 'molybdenum', 'thallium',
                'uranium', 'vanadium'
            ],
            'Organic': [
                'carbon', 'organic', 'bod', 'cod', 'doc', 'toc',
                'oil', 'grease', 'petroleum', 'benzene', 'pcb',
                'pesticide', 'herbicide', 'volatile', 'voc'
            ],
            'Biological': [
                'algae', 'chlorophyll', 'bacteria', 'coliform', 'e. coli',
                'enterococcus', 'plankton', 'biomass', 'biological',
                'microbiological', 'fecal'
            ],
            'Ions': [
                'chloride', 'sulfate', 'fluoride', 'bromide', 'calcium',
                'magnesium', 'sodium', 'potassium', 'bicarbonate', 'carbonate',
                'alkalinity', 'silica'
            ],
            'Sediment': [
                'sediment', 'suspended solids', 'tss', 'suspended',
                'bed material', 'particle size', 'silt', 'clay', 'sand'
            ],
            'Isotopes': [
                'isotope', 'deuterium', 'tritium', 'radioactive',
                'radon', 'uranium', 'radium', 'strontium'
            ]
        }
        
        # Check each category
        for category, keywords in categories.items():
            if any(keyword in name_lower for keyword in keywords):
                return category
        
        # Special handling for combined terms
        if any(metal in name_lower for metal in categories['Metals']):
            if any(form in name_lower for form in ['dissolved', 'total']):
                return 'Metals'
        
        if any(nutrient in name_lower for nutrient in categories['Nutrients']):
            if any(form in name_lower for form in ['dissolved', 'total', 'organic']):
                return 'Nutrients'
        
        return 'Other'
    
    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """Get parameter metadata."""
        if self._parameters is None:
            self._load_parameters()
        return self._parameters
    
    def get_sites_by_bbox(self, min_lon: float, min_lat: float, 
                         max_lon: float, max_lat: float) -> pd.DataFrame:
        """Get monitoring sites within a bounding box.
        
        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            
        Returns:
            DataFrame with site information
        """
        params = {
            'mimeType': 'csv',
            'zip': 'no',
            'bBox': f"{min_lon},{min_lat},{max_lon},{max_lat}"
        }
        
        response = self._make_request(self.STATION_URL, params)
        
        if response is None:
            logger.error("No valid response from API")
            return pd.DataFrame()
            
        try:
            df = pd.read_csv(response)
            if not df.empty:
                df = df.rename(columns={
                    'MonitoringLocationIdentifier': 'Site ID',
                    'MonitoringLocationName': 'Site Name',
                    'LatitudeMeasure': 'Latitude',
                    'LongitudeMeasure': 'Longitude'
                })
            return df
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return pd.DataFrame()
    
    def get_sites_by_point(self, lat: float, lon: float, radius: float) -> pd.DataFrame:
        """Get monitoring sites near a point.
        
        Args:
            lat: Latitude
            lon: Longitude
            radius: Search radius in miles
            
        Returns:
            DataFrame with site information
        """
        logger.info(f"Searching for sites near ({lat}, {lon}) within {radius} miles")
        
        params = {
            'mimeType': 'csv',
            'zip': 'no',
            'lat': lat,
            'long': lon,
            'within': f"{radius}"  # API expects miles
        }
        
        response = self._make_request(self.STATION_URL, params)
        
        if response is None:
            logger.error("No valid response from API")
            return pd.DataFrame()
            
        try:
            # Response is already a DataFrame from _make_request
            df = response
            if not df.empty:
                logger.info(f"Found {len(df)} sites in radius")
                df = df.rename(columns={
                    'MonitoringLocationIdentifier': 'Site ID',
                    'MonitoringLocationName': 'Site Name',
                    'LatitudeMeasure': 'Latitude',
                    'LongitudeMeasure': 'Longitude'
                })
                # Log count of sites found instead of all IDs
                if not df.empty:
                    logger.debug(f"Found {len(df)} sites in the specified area")
            else:
                logger.warning(f"No sites found within {radius} miles of ({lat}, {lon})")
            return df
        except Exception as e:
            logger.error(f"Error processing site data: {str(e)}")
            return pd.DataFrame()
    
    def _normalize_location_input(self, location_input: Union[Dict, List, Tuple]) -> Dict:
        """
        Normalize different location input formats to a standard format.
        Returns a dict with one of:
        - 'points': List of (lat, lon) tuples
        - 'radius': Dict with center (lat, lon) and radius
        - 'bbox': Dict with min/max lat/lon
        """
        if isinstance(location_input, (list, tuple)):
            if len(location_input) == 2:  # Single lat/lon point
                return {'points': [(location_input[0], location_input[1])]}
            elif len(location_input) == 4:  # Bounding box
                return {'bbox': {
                    'min_lon': location_input[0],
                    'min_lat': location_input[1],
                    'max_lon': location_input[2],
                    'max_lat': location_input[3]
                }}
        elif isinstance(location_input, dict):
            if 'lat' in location_input and ('lon' in location_input or 'long' in location_input):
                # Point + radius
                longitude = location_input.get('long', location_input.get('lon'))
                return {
                    'radius': {
                        'lat': location_input['lat'],
                        'long': longitude,  # Always use 'long' internally
                        'radius': location_input.get('radius', 20)  # Default 20 mile radius
                    }
                }
            elif all(k in location_input for k in ['min_lon', 'min_lat', 'max_lon', 'max_lat']):
                return {'bbox': location_input}
            elif 'coordinates' in location_input:
                return {
                    'radius': {
                        'lat': location_input['coordinates'][0],
                        'long': location_input['coordinates'][1],
                        'radius': location_input.get('radius', 20)  # Default 20 mile radius
                    }
                }
            elif 'bounds' in location_input:
                return {'bbox': location_input['bounds']}
            elif 'sites' in location_input:
                return {'sites': location_input['sites']}
        
        raise ValueError(f"Invalid location input format: {location_input}")

    def _get_sites_for_location(self, location: Dict) -> pd.DataFrame:
        """Get sites for normalized location input.
        
        Args:
            location: Normalized location dictionary from _normalize_location_input
            
        Returns:
            DataFrame with site information
        """
        try:
            if 'bbox' in location:
                bbox = location['bbox']
                return self.get_sites_by_bbox(
                    min_lon=bbox['min_lon'],
                    min_lat=bbox['min_lat'],
                    max_lon=bbox['max_lon'],
                    max_lat=bbox['max_lat']
                )
            elif 'radius' in location:
                rad = location['radius']
                return self.get_sites_by_point(
                    lat=rad['lat'],
                    lon=rad['long'],
                    radius=rad['radius']
                )
            elif 'points' in location:
                # For multiple points, combine results
                all_sites = []
                for lat, lon in location['points']:
                    sites = self.get_sites_by_point(lat, lon, radius=20)  # Default 20-mile radius
                    if sites is not None and not sites.empty:
                        all_sites.append(sites)
                return pd.concat(all_sites).drop_duplicates() if all_sites else pd.DataFrame()
            elif 'sites' in location:
                # If we already have site IDs, return a DataFrame with those IDs
                return pd.DataFrame({'Site ID': location['sites']})
            
            logger.warning(f"Unsupported location format: {location}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error getting sites for location: {str(e)}")
            return pd.DataFrame()

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for each characteristic in the pivoted data.
        
        Args:
            df: Pivoted DataFrame with characteristics as columns
            
        Returns:
            Dictionary of statistics by characteristic
        """
        stats = {}
        
        # Get all value columns (exclude unit columns)
        value_cols = [col for col in df.columns if col.endswith('_ResultMeasureValue')]
        
        for col in value_cols:
            characteristic = col.replace('_ResultMeasureValue', '')
            values = df[col].dropna()
            
            if len(values) > 0:
                unit_col = f"{characteristic}_ResultMeasure/MeasureUnitCode"
                units = df[unit_col].dropna().unique()[0] if unit_col in df.columns else 'Unknown'
                
                stats[characteristic] = {
                    'count': len(values),
                    'sites': len(df[df[col].notna()]['Site ID'].unique()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'units': units
                }
        
        return stats

    def _clean_numeric_data(self, series: pd.Series) -> pd.Series:
        """Clean numeric data by handling common issues and special values.
        
        Handles:
        - Inequality expressions (e.g., "<0.5", ">10")
        - Non-determined values ("ND", "NA", etc.)
        - Mixed numeric/text values
        """
        # If input is a single value, convert to series
        if not isinstance(series, pd.Series):
            series = pd.Series([series])
        
        # If already numeric, just return as is
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        # Create a copy to avoid modifying the original
        cleaned = series.copy()
        
        # Handle inequality expressions
        cleaned = cleaned.apply(lambda x: str(x).strip() if pd.notna(x) else x)
        # Extract numbers from inequality expressions
        cleaned = cleaned.apply(lambda x: str(x)[1:] if pd.notna(x) and str(x).startswith(('<', '>')) else x)
        
        # Replace non-determined values with NaN
        non_determined = ['ND', 'NA', 'N/A', 'NOT DETERMINED', 'UNKNOWN', '--']
        cleaned = cleaned.replace(non_determined, np.nan)
        
        # Remove any remaining non-numeric characters except decimal points and minus signs
        cleaned = cleaned.apply(lambda x: re.sub(r'[^\d.-]', '', str(x)) if pd.notna(x) else x)
        
        # Convert to numeric, coercing errors to NaN
        result = pd.to_numeric(cleaned, errors='coerce')
        
        # If input was a single value, return the first element
        if len(result) == 1:
            return result.iloc[0]
        return result

    def _clean_and_pivot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and pivot raw data into a unified structure."""
        logger.info("Cleaning and pivoting data...")
        
        # Check if data is already in pivoted format
        metadata_cols = ['Date', 'Site ID', 'Organization ID', 'Organization Name', 'Latitude', 'Longitude']
        if all(col in df.columns for col in metadata_cols):
            logger.info("Data is already in pivoted format")
            return df
        
        # Define column groups for raw data
        index_columns = [
            'OrganizationIdentifier',
            'OrganizationFormalName',
            'ActivityStartDate',
            'ActivityLocation/LatitudeMeasure',
            'ActivityLocation/LongitudeMeasure',
            'MonitoringLocationIdentifier'
        ]
        
        activity_columns = [
            'CharacteristicName',
            'ResultMeasureValue',
            'ResultMeasure/MeasureUnitCode'
        ]
        
        # Ensure all required columns exist
        missing_cols = [col for col in index_columns + activity_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        try:
            # Clean numeric columns
            logger.info("Cleaning numeric data...")
            df['ResultMeasureValue'] = self._clean_numeric_data(df['ResultMeasureValue'])
            df['ActivityLocation/LatitudeMeasure'] = self._clean_numeric_data(df['ActivityLocation/LatitudeMeasure'])
            df['ActivityLocation/LongitudeMeasure'] = self._clean_numeric_data(df['ActivityLocation/LongitudeMeasure'])
            
            # Convert date column
            df['ActivityStartDate'] = pd.to_datetime(df['ActivityStartDate'])
            
            # Group and aggregate
            logger.info("Grouping and aggregating data...")
            grouped_df = df.groupby(index_columns + ['CharacteristicName'], as_index=False).agg({
                'ResultMeasureValue': 'mean',
                'ResultMeasure/MeasureUnitCode': 'first'
            })
            
            # Check if we have any data after grouping
            if grouped_df.empty:
                logger.warning("No data available after grouping")
                return pd.DataFrame()
            
            # Pivot the DataFrame
            logger.info("Creating pivot table...")
            try:
                pivoted = grouped_df.pivot(
                    index=index_columns,
                    columns='CharacteristicName',
                    values=['ResultMeasureValue', 'ResultMeasure/MeasureUnitCode']
                )
            except ValueError as e:
                logger.error(f"Error during pivot operation: {str(e)}")
                return pd.DataFrame()
            
            # Reset index to flatten
            pivoted = pivoted.reset_index()
            
            # Clean up column names
            logger.info("Cleaning up column names...")
            pivoted.columns = [
                x[0] if x[1] == '' else 
                x[1] if 'Value' in x[0] else 
                x[1] + '_units' 
                for x in pivoted.columns.values
            ]
            
            # Rename columns for consistency with rest of codebase
            column_mapping = {
                'ActivityStartDate': 'Date',
                'MonitoringLocationIdentifier': 'Site ID',
                'OrganizationIdentifier': 'Organization ID',
                'OrganizationFormalName': 'Organization Name',
                'ActivityLocation/LatitudeMeasure': 'Latitude',
                'ActivityLocation/LongitudeMeasure': 'Longitude'
            }
            pivoted = pivoted.rename(columns=column_mapping)
            
            # Identify and coerce numeric columns
            logger.info("Coercing numeric columns...")
            # Find all unit columns
            unit_cols = [col for col in pivoted.columns if col.endswith('_units')]
            # Get corresponding measure columns
            measure_cols = [col.rsplit('_', 1)[0] for col in unit_cols]
            
            # Create ordered column list
            column_order = metadata_cols + [y for x in unit_cols for y in [x.rsplit('_', 1)[0], x]]
            
            # Convert measure columns to float
            logger.info(f"Converting measure columns to float: {measure_cols}")
            pivoted[measure_cols] = pivoted[measure_cols].astype('float')
            
            # Reorder columns
            pivoted = pivoted[column_order]
            
            # Sort by date and site
            pivoted = pivoted.sort_values(['Date', 'Site ID'])
            
            logger.info(f"Created pivoted structure with {len(pivoted)} rows")
            return pivoted
            
        except Exception as e:
            logger.error(f"Error in _clean_and_pivot_data: {str(e)}")
            return pd.DataFrame()

    def get_data(self, 
                 location_input: Union[Dict, List, Tuple],
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 characteristics: Optional[List[str]] = None,
                 max_sites: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """Get data from Water Quality Portal API."""
        # Initialize metadata
        api_metadata = {
            'query_info': {
                'location': location_input,
                'date_range': {'start': start_date, 'end': end_date},
                'characteristics_requested': characteristics,
                'max_sites': max_sites
            },
            'errors': [],
            'sites_found': 0,
            'sites_queried': 0,
            'messages': []
        }
        
        try:
            # Normalize location input
            location = self._normalize_location_input(location_input)
            logger.info(f"Normalized location: {location}")
            
            # Get sites for location
            sites_df = self._get_sites_for_location(location)
            if sites_df is None or sites_df.empty:
                msg = "No sites found for the specified location"
                logger.warning(msg)
                api_metadata['messages'].append(msg)
                return pd.DataFrame(), api_metadata
            
            api_metadata['sites_found'] = len(sites_df)
            api_metadata['sites_queried'] = min(len(sites_df), max_sites)
            
            # Prepare parameters for data query
            params = {
                "mimeType": "csv",
                "zip": "yes",
                "providers": ["NWIS", "STORET"]
            }
            
            # Add spatial parameters
            if 'bbox' in location:
                bbox = location['bbox']
                params['bBox'] = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
            elif 'radius' in location:
                rad = location['radius']
                params.update({
                    'lat': rad['lat'],
                    'long': rad['long'],
                    'within': rad['radius']
                })
            elif 'points' in location:
                point = location['points'][0]
                params.update({
                    'lat': point[0],
                    'long': point[1],
                    'within': 1
                })
            
            # Add date parameters
            if start_date:
                params['startDateLo'] = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m-%d-%Y')
            if end_date:
                params['startDateHi'] = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m-%d-%Y')
            
            # Make requests for both profiles without filtering by characteristics
            all_dfs = []
            profiles = ['resultPhysChem', 'biological']
            
            for profile in profiles:
                params['dataProfile'] = profile
                logger.info(f"\nTrying {profile} profile with parameters:")
                logger.info(f"Location: {json.dumps({k:v for k,v in params.items() if k in ['lat', 'long', 'within', 'bBox']})}")
                logger.info(f"Date range: {json.dumps({k:v for k,v in params.items() if k in ['startDateLo', 'startDateHi']})}")
                
                df = self._make_request(f"{self.BASE_URL}/Result/search", params)
                if df is not None and not df.empty:
                    all_dfs.append(df)
                    api_metadata['messages'].append(f"Retrieved {len(df)} records from {profile} profile")
                else:
                    logger.warning(f"No data returned for {profile} profile")
            
            if not all_dfs:
                msg = "No data returned from either profile"
                logger.warning(msg)
                api_metadata['messages'].append(msg)
                return pd.DataFrame(), api_metadata
            
            # Combine raw results
            raw_df = pd.concat(all_dfs, ignore_index=True)
            msg = f"Combined {len(raw_df)} total records from all profiles"
            logger.info(msg)
            api_metadata['messages'].append(msg)
            
            # Log available characteristics in raw data
            available_chars = raw_df['CharacteristicName'].unique()
            logger.info(f"Available characteristics in raw data: {available_chars}")
            
            # Clean and pivot the combined data
            result_df = self._clean_and_pivot_data(raw_df)
            if result_df.empty:
                msg = "No data available after cleaning and pivoting"
                logger.warning(msg)
                api_metadata['messages'].append(msg)
                return pd.DataFrame(), api_metadata
            
            # Calculate statistics on full dataset
            full_stats = self._calculate_comprehensive_statistics(raw_df, result_df, characteristics)
            api_metadata['full_dataset_stats'] = full_stats
            
            # If characteristics were requested, filter the data
            filtered_df = result_df
            if characteristics:
                logger.info(f"Filtering for requested characteristics: {characteristics}")
                # Create filter mask for requested characteristics
                char_mask = pd.Series(False, index=result_df.index)
                for char in characteristics:
                    # Look for columns matching this characteristic
                    char_cols = [col for col in result_df.columns if char in col and not col.endswith('_units')]
                    if char_cols:
                        char_mask |= result_df[char_cols].notna().any(axis=1)
                
                # Apply filter
                filtered_df = result_df[char_mask]
                logger.info(f"After filtering: {len(filtered_df)} records")
            
            # Calculate statistics on filtered dataset
            stats = self._calculate_comprehensive_statistics(raw_df, filtered_df, characteristics)
            
            # Update metadata
            api_metadata.update({
                'total_records': len(filtered_df),
                'characteristics_stats': stats,
                'date_range': stats.get('overall', {}).get('date_range', {}),
                'geographic_bounds': stats.get('overall', {}).get('geographic_bounds', {})
            })
            
            return filtered_df, api_metadata
            
        except Exception as e:
            logger.error(f"Error in get_data: {str(e)}", exc_info=True)
            api_metadata['errors'].append(str(e))
            return pd.DataFrame(), api_metadata
            
    def _find_temporal_gaps(self, dates: pd.Series, min_gap_days: int = 30) -> List[Dict[str, str]]:
        """Find significant gaps in temporal coverage."""
        if len(dates) < 2:
            return []
            
        gaps = []
        date_diffs = dates.sort_values().diff()
        significant_gaps = date_diffs[date_diffs > pd.Timedelta(days=min_gap_days)]
        
        for gap_start in significant_gaps.index:
            gap_end = dates[dates > gap_start].min()
            gaps.append({
                'start': gap_start.strftime('%Y-%m-%d'),
                'end': gap_end.strftime('%Y-%m-%d'),
                'duration_days': (gap_end - gap_start).days
            })
            
        return gaps

    def _create_pivoted_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pivoted structure from raw data with proper handling of duplicates."""
        logger.info("Creating pivoted structure...")
        
        # 1. Standardize date format and create measurement ID
        df['date'] = pd.to_datetime(df['ActivityStartDate'])
        
        # 2. Define core columns we want to preserve
        metadata_cols = [
            'OrganizationIdentifier',
            'OrganizationFormalName',
            'MonitoringLocationIdentifier',
            'ActivityLocation/LatitudeMeasure',
            'ActivityLocation/LongitudeMeasure'
        ]
        
        # 3. First aggregate measurements at site/date/characteristic level
        agg_functions = {
            'ResultMeasureValue': {
                'value': 'mean',
                'count': 'count',
                'std': 'std',
                'min': 'min',
                'max': 'max'
            },
            'ResultMeasure/MeasureUnitCode': lambda x: x.value_counts().index[0],  # Most common unit
            'ActivityLocation/LatitudeMeasure': 'first',
            'ActivityLocation/LongitudeMeasure': 'first',
            'OrganizationIdentifier': 'first',
            'OrganizationFormalName': 'first'
        }
        
        # Group by date, site, and characteristic
        grouped = df.groupby([
            'date',
            'MonitoringLocationIdentifier',
            'CharacteristicName'
        ]).agg(agg_functions)
        
        # Flatten multi-level columns
        grouped.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col 
                          for col in grouped.columns]
        
        # Reset index to prepare for pivot
        grouped = grouped.reset_index()
        
        # 4. Create the pivoted structure
        # First pivot values
        values_pivot = grouped.pivot(
            index=['date', 'MonitoringLocationIdentifier'],
            columns='CharacteristicName',
            values=['ResultMeasureValue_value', 'ResultMeasureValue_count', 
                    'ResultMeasureValue_std', 'ResultMeasureValue_min', 
                    'ResultMeasureValue_max']
        )
        
        # Flatten the multi-level column names
        values_pivot.columns = [f"{col[1]}_{col[0]}" for col in values_pivot.columns]
        
        # Pivot units separately
        units_pivot = grouped.pivot(
            index=['date', 'MonitoringLocationIdentifier'],
            columns='CharacteristicName',
            values='ResultMeasure/MeasureUnitCode'
        )
        units_pivot.columns = [f"{col}_unit" for col in units_pivot.columns]
        
        # 5. Combine pivoted data with metadata
        metadata = grouped.groupby(['date', 'MonitoringLocationIdentifier']).agg({
            'OrganizationIdentifier': 'first',
            'OrganizationFormalName': 'first',
            'ActivityLocation/LatitudeMeasure': 'first',
            'ActivityLocation/LongitudeMeasure': 'first'
        })
        
        # 6. Join all components
        result = pd.concat([metadata, values_pivot, units_pivot], axis=1)
        result = result.reset_index()
        
        # 7. Rename columns for clarity
        result = result.rename(columns={
            'MonitoringLocationIdentifier': 'Site ID',
            'OrganizationIdentifier': 'Organization ID',
            'OrganizationFormalName': 'Organization Name',
            'ActivityLocation/LatitudeMeasure': 'Latitude',
            'ActivityLocation/LongitudeMeasure': 'Longitude'
        })
        
        # 8. Sort by date and site
        result = result.sort_values(['date', 'Site ID'])
        
        # Move metadata columns to front
        metadata_cols = ['date', 'Site ID', 'Organization ID', 'Organization Name', 
                        'Latitude', 'Longitude']
        other_cols = [col for col in result.columns if col not in metadata_cols]
        result = result[metadata_cols + other_cols]
        
        logger.info(f"Created pivoted structure with {len(result)} rows")
        return result

    def _calculate_comprehensive_statistics(self, raw_df: pd.DataFrame, pivoted_df: pd.DataFrame, requested_characteristics: Optional[List[str]] = None) -> Dict:
        """Calculate comprehensive statistics using pandas describe() for both raw and pivoted data.
        
        Args:
            raw_df: Raw DataFrame before pivoting
            pivoted_df: Pivoted DataFrame with characteristics as columns
            requested_characteristics: List of originally requested characteristics
            
        Returns:
            Dictionary containing statistics for requested and additional characteristics
        """
        stats = {
            'requested': {},
            'additional': {},
            'overall': {
                'total_records': len(pivoted_df),
                'unique_sites': pivoted_df['Site ID'].nunique(),
                'date_range': {
                    'start': pivoted_df['Date'].min().strftime('%Y-%m-%d'),
                    'end': pivoted_df['Date'].max().strftime('%Y-%m-%d'),
                    'days': (pivoted_df['Date'].max() - pivoted_df['Date'].min()).days
                },
                'geographic_bounds': {
                    'min_lat': float(pivoted_df['Latitude'].min()),
                    'max_lat': float(pivoted_df['Latitude'].max()),
                    'min_lon': float(pivoted_df['Longitude'].min()),
                    'max_lon': float(pivoted_df['Longitude'].max())
                }
            }
        }
        
        # Get all value columns (exclude metadata and unit columns)
        value_cols = [col for col in pivoted_df.columns 
                     if not col.endswith('_units') 
                     and col not in ['Date', 'Site ID', 'Organization ID', 'Organization Name', 'Latitude', 'Longitude']]
        
        # Get available characteristics from raw data
        available_chars = raw_df['CharacteristicName'].unique()
        stats['overall']['available_characteristics'] = list(available_chars)
        
        # Process each value column
        for col in value_cols:
            # Skip if column is empty or non-numeric
            if pivoted_df[col].dtype not in ['int64', 'float64'] or pivoted_df[col].isna().all():
                continue
                
            try:
                # Calculate statistics for this column
                col_stats = self._calculate_column_statistics(pivoted_df, col)
                
                # Determine if this was a requested characteristic
                if requested_characteristics:
                    if any(req.lower() in col.lower() for req in requested_characteristics):
                        stats['requested'][col] = col_stats
                    else:
                        stats['additional'][col] = col_stats
                else:
                    stats['additional'][col] = col_stats
                    
            except Exception as e:
                logger.debug(f"Could not calculate statistics for {col}: {str(e)}")
                continue
        
        # Add summary information about requested characteristics
        if requested_characteristics:
            found_chars = set(stats['requested'].keys())
            requested_set = set(requested_characteristics)
            missing = requested_set - {c.split('_')[0] for c in found_chars}
            if missing:
                stats['overall']['missing_characteristics'] = list(missing)
        
        return stats
        
    def _calculate_column_statistics(self, df: pd.DataFrame, col: str) -> Dict:
        """Calculate detailed statistics for a single column.
        
        Args:
            df: DataFrame containing the column
            col: Column name to calculate statistics for
            
        Returns:
            Dictionary of statistics for the column
        """
        try:
            # Get the corresponding units column
            units_col = f"{col}_units"
            units = df[units_col].dropna().unique().tolist() if units_col in df.columns else ['Unknown']
            
            # Get the statistics using describe
            desc_stats = df[col].describe()
            
            # Calculate temporal coverage
            valid_dates = df[df[col].notna()]['Date']
            temporal_coverage = {
                'start': valid_dates.min().strftime('%Y-%m-%d'),
                'end': valid_dates.max().strftime('%Y-%m-%d'),
                'days': (valid_dates.max() - valid_dates.min()).days
            }
            
            # Calculate site coverage
            sites_with_data = df[df[col].notna()]['Site ID'].unique()
            
            # Calculate quartiles safely
            quartiles = desc_stats.to_dict()
            q1 = float(quartiles.get('25%', quartiles.get('25.0%', quartiles.get('0.25', 0))))
            q3 = float(quartiles.get('75%', quartiles.get('75.0%', quartiles.get('0.75', 0))))
            
            # Calculate IQR and outlier bounds
            iqr = q3 - q1
            outlier_bounds = {
                'lower': q1 - 1.5 * iqr,
                'upper': q3 + 1.5 * iqr
            }
            
            # Find potential outliers
            potential_outliers = df[
                (df[col] < outlier_bounds['lower']) | 
                (df[col] > outlier_bounds['upper'])
            ][col]
            
            # Create statistics dictionary with safe value extraction
            stats = {
                'count': int(desc_stats['count']),
                'mean': float(desc_stats['mean']),
                'std': float(desc_stats['std']),
                'min': float(desc_stats['min']),
                'max': float(desc_stats['max']),
                'median': float(quartiles.get('50%', quartiles.get('50.0%', quartiles.get('0.5', desc_stats['mean'])))),
                'quartiles': {
                    'q1': q1,
                    'q3': q3
                },
                'units': units,
                'sites': {
                    'count': len(sites_with_data),
                    'ids': sites_with_data.tolist()
                },
                'temporal_coverage': temporal_coverage,
                'outliers': {
                    'bounds': outlier_bounds,
                    'count': len(potential_outliers),
                    'values': potential_outliers.tolist() if len(potential_outliers) < 10 else potential_outliers.describe().to_dict()
                }
            }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating statistics for column {col}: {str(e)}")
            # Return minimal statistics if calculation fails
            return {
                'count': len(df[df[col].notna()]),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'units': ['Unknown'],
                'error': str(e)
            } 