"""USGS water quality service implementation."""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
from pygeohydro import NWIS, WaterQuality
import os
import json
import time
from contextlib import contextmanager
import warnings
import logging
import math
import numpy as np
from .direct_api import WaterQualityAPI

# Configure logging with a more concise format
logging.basicConfig(
    level=logging.WARNING,  # Reduce overall logging level
    format='%(levelname)s: %(message).200s'  # Truncate messages to 200 chars
)
logger = logging.getLogger(__name__)

# Suppress ResourceWarnings about unclosed sockets
warnings.filterwarnings("ignore", category=ResourceWarning)

from math import radians, cos, sin, asin, sqrt
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in miles."""
    R = 3959.87433  # Earth's radius in miles

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

class ConnectionManager:
    """Manages NWIS connections with retries and timeouts."""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    @staticmethod
    @contextmanager
    def retry_context():
        """Context manager for retrying operations."""
        retries = 0
        while retries < ConnectionManager.MAX_RETRIES:
            try:
                yield
                break
            except Exception as e:
                retries += 1
                if retries == ConnectionManager.MAX_RETRIES:
                    raise e
                logger.warning(f"Attempt {retries} failed: {str(e)}. Retrying in {ConnectionManager.RETRY_DELAY}s...")
                time.sleep(ConnectionManager.RETRY_DELAY)

class GeoBounds:
    """Geographic bounding box."""
    
    def __init__(self, min_lat: float, max_lat: float, min_lon: float, max_lon: float):
        """Initialize bounds.
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude
            min_lon: Minimum longitude
            max_lon: Maximum longitude
        """
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon
        
    @staticmethod
    def format_coord(coord: float) -> float:
        """Format coordinate to 7 decimal places for USGS API compatibility."""
        return round(float(coord), 7)
        
    def to_bbox(self) -> List[float]:
        """Convert to bounding box format.
        
        Returns:
            List of [min_lon, min_lat, max_lon, max_lat]
        """
        return [
            self.format_coord(self.min_lon),
            self.format_coord(self.min_lat),
            self.format_coord(self.max_lon),
            self.format_coord(self.max_lat)
        ]
        
    def to_bbox_string(self) -> str:
        """Convert to USGS API compatible bounding box string."""
        bbox = self.to_bbox()
        return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

class SiteLocation:
    """Site location with optional radius."""
    
    def __init__(self, site_id: Optional[str] = None, lat: Optional[float] = None, 
                 lon: Optional[float] = None, radius_km: Optional[float] = None):
        """Initialize location.
        
        Args:
            site_id: USGS site ID
            lat: Latitude
            lon: Longitude
            radius_km: Search radius in kilometers
        """
        self.site_id = site_id
        self.lat = lat
        self.lon = lon
        self.radius_km = radius_km

class WaterQualityService:
    """Service for accessing USGS water quality data."""
    
    _instance = None
    
    def __new__(cls):
        """Ensure only one instance is created."""
        if cls._instance is None:
            cls._instance = super(WaterQualityService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize service."""
        # Only initialize once
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing WaterQualityService (data backend)...")
        self.direct_api = WaterQualityAPI()  # Direct API implementation
        self._parameters = None  # We'll load this on demand
        self._initialized = True

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        """Get parameter metadata.
        
        Returns:
            Dictionary containing parameter information or None if not available
        """
        if self._parameters is None:
            try:
                self._parameters = self.direct_api.parameters
                if self._parameters:
                    self.logger.info(f"Loaded {self._parameters['metadata']['total_count']} parameters")
                else:
                    self.logger.warning("No parameters available from API")
            except Exception as e:
                self.logger.error(f"Error loading parameters: {str(e)}")
                return None
        return self._parameters

    def get_data(self,
                location: Union[Dict[str, float], List[str]],
                parameters: List[str],
                start_date: str,
                end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Get water quality data for a location or sites.
        
        Args:
            location: Either:
                - Dictionary with coordinates/radius: {"coordinates": [lat, lon], "radius": radius_miles}
                - Dictionary with bounds: {"bounds": {"min_lat": float, "max_lat": float, "min_lon": float, "max_lon": float}}
                - List of site IDs (legacy support)
            parameters: List of parameter names/characteristics
            start_date: Start date in YYYY-MM-DD format
            end_date: Optional end date in YYYY-MM-DD format
            
        Returns:
            Tuple of (DataFrame with data, metadata dictionary)
        """
        try:
            logger.info(f"Getting data for location: {location}")
            logger.info(f"Parameters: {parameters}")
            logger.info(f"Date range: {start_date} to {end_date}")
            
            # Initialize direct API if not already done
            if not hasattr(self, 'direct_api'):
                self.direct_api = WaterQualityAPI()
            
            # Get data using direct API - pass location directly since normalization happens in direct_api
            data, metadata = self.direct_api.get_data(
                location_input=location,  # Pass location directly
                start_date=start_date,
                end_date=end_date,
                characteristics=parameters
            )
            
            if data.empty:
                logger.warning("No data returned from direct API")
            else:
                logger.info(f"Retrieved {len(data)} records")
                logger.info(f"Columns: {data.columns.tolist()}")
            
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error getting data: {str(e)}", exc_info=True)
            return pd.DataFrame(), {}

    def find_sites_by_location(self, location: Union[GeoBounds, SiteLocation]) -> pd.DataFrame:
        """Find monitoring sites by location."""
        if isinstance(location, GeoBounds):
            return self.direct_api.get_sites_by_bbox(
                min_lon=location.min_lon,
                min_lat=location.min_lat,
                max_lon=location.max_lon,
                max_lat=location.max_lat
            )
        elif isinstance(location, SiteLocation):
            return self.direct_api.get_sites_by_point(
                lat=location.lat,
                lon=location.lon,
                radius=location.radius_km * 0.621371  # Convert km to miles
            )
        else:
            raise ValueError("Invalid location type")

    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get statistics about available parameters."""
        return {
            'total': len(self.parameters['parameters']),
            'groups': self.parameters['metadata']['groups'],
            'last_updated': self.parameters['last_updated']
        }
        
    def get_site_statistics(self) -> Dict[str, Any]:
        """Get statistics about available sites."""
        return self.site_cache
        
    def get_parameters_by_group(self, group: Optional[str] = None) -> List[Dict[str, str]]:
        """Get parameters for a group.
        
        Args:
            group: Parameter group name (optional)
            
        Returns:
            List of parameter dictionaries
        """
        if not self.parameters:
            self._initialize_parameter_cache()
            
        if group:
            return self.parameter_groups.get(group, [])
        return self.parameters['parameters']
        
    def find_parameters(self, search: str) -> List[Dict[str, str]]:
        """Find parameters by name or description.
        
        Args:
            search: Search string
            
        Returns:
            List of matching parameters
        """
        if not self.parameters:
            self._initialize_parameter_cache()
            
        search = search.lower()
        return [p for p in self.parameters['parameters']
                if search in p['parameter_nm'].lower() or 
                   search in p.get('description', '').lower()]
        
    def format_coord(coord: Union[float, str]) -> str:
        """Format coordinate to 6 decimal places and ensure it's a string."""
        return f"{float(coord):.6f}"

    def get_area_data(self, center_site_id: str, radius_miles: float, parameters: List[str],
                      start_date: str, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get data for sites within a radius using the direct API."""
        try:
            # First get the center site's info
            center_info = self.find_sites_by_location(SiteLocation(None, None))
            center_info = center_info[center_info['Site ID'] == center_site_id]
            
            if center_info.empty:
                logger.error(f"Could not find center site {center_site_id}")
                return pd.DataFrame(), {}
            
            # Create location input for direct API
            location_input = {
                'lat': float(center_info.iloc[0]['Latitude']),
                'lon': float(center_info.iloc[0]['Longitude']),
                'radius': radius_miles
            }
            
            # Get data using direct API
            data, metadata = self.direct_api.get_data(
                location_input=location_input,
                start_date=start_date,
                end_date=end_date,
                characteristics=parameters
            )
            
            if data.empty:
                logger.warning("No data found for the given parameters")
                return pd.DataFrame(), metadata
            
            # Add distance from center calculations
            if 'Latitude' in data.columns and 'Longitude' in data.columns:
                data['DistanceFromCenter'] = data.apply(
                    lambda row: haversine_distance(
                        float(center_info.iloc[0]['Latitude']),
                        float(center_info.iloc[0]['Longitude']),
                        float(row['Latitude']),
                        float(row['Longitude'])
                    ),
                    axis=1
                )
                
            return data, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving area data: {str(e)}")
            return pd.DataFrame(), {}

    def _format_column_name(self, param_code: str, param_info: Optional[Dict[str, str]] = None) -> str:
        """Create standardized column names.
        
        Args:
            param_code: Parameter code
            param_info: Optional dictionary with parameter name and unit
            
        Returns:
            Formatted column name
        """
        if param_info is None:
            return f"Parameter {param_code}"
            
        # Get base name - take first part before comma if exists
        name = param_info['name'].split(',')[0].strip()
        
        # Clean and standardize unit
        unit = param_info.get('unit', '').strip()
        if unit:
            # Standardize common units
            unit_map = {
                'degrees Celsius': '°C',
                'milligrams per liter': 'mg/L',
                'standard units': '',  # pH doesn't need units in header
                'nephelometric turbidity units': 'NTU',
                'microsiemens per centimeter': 'µS/cm',
                'feet': 'ft',
                'meters': 'm'
            }
            for old_unit, new_unit in unit_map.items():
                if old_unit in unit.lower():
                    unit = new_unit
                    break
            
            # Add unit if we have one
            if unit:
                return f"{name} ({unit})"
        
        return name

    def _find_alternative_parameters(self, site_params: List[str], requested_param: str) -> List[str]:
        """Find alternative parameters in the same group.
        
        Args:
            site_params: List of parameters available at the site
            requested_param: Parameter code we're looking for alternatives for
            
        Returns:
            List of alternative parameter codes
        """
        if not self.parameters:
            return []
            
        # Find the group of the requested parameter
        req_param_info = next(
            (p for p in self.parameters['parameters'] 
             if p['parameter_cd'] == requested_param),
            None
        )
        if not req_param_info:
            return []
            
        req_group = req_param_info['parameter_group']
        
        # Find parameters in the same group that are available at the site
        alternatives = []
        for param in site_params:
            param_info = next(
                (p for p in self.parameters['parameters'] 
                 if p['parameter_cd'] == param),
                None
            )
            if param_info and param_info['parameter_group'] == req_group:
                alternatives.append(param)
        
        return alternatives

    def _clean_measurement_values(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Clean measurement values, converting strings to floats and handling special cases.
        
        Args:
            df: DataFrame containing measurements
            column: Column name to clean
            
        Returns:
            Series with cleaned numeric values
        """
        def convert_value(val):
            if pd.isna(val):
                return np.nan
            if isinstance(val, (int, float)):
                return float(val)
            
            # Convert to string and clean
            val_str = str(val).strip().upper()
            
            # Handle special cases
            if val_str in ('', 'NONE', 'NULL', 'ND', 'N/D', 'NA', 'EQP', '-'):
                return np.nan
            
            # Try to convert string number to float
            try:
                # Remove any trailing units or qualifiers
                val_str = val_str.split()[0]
                return float(val_str)
            except ValueError:
                logger.debug(f"Could not convert value '{val}' to numeric in column {column}")
                return np.nan

        return df[column].apply(convert_value)

    def get_site_data(self, sites: List[str], parameters: List[str],
                      start_date: str, end_date: Optional[str] = None,
                      include_alternatives: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get data for sites and parameters.
        
        Args:
            sites: List of site IDs
            parameters: List of parameter codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_alternatives: Whether to include alternative parameters when requested ones aren't available
            
        Returns:
            Tuple of (DataFrame of site data with metadata, Dict of parameter availability info)
        """
        # Get data for requested parameters first
        result, param_availability = self._get_site_data_basic(
            sites, parameters, start_date, end_date)
        
        # Check if we need to look for alternatives
        if include_alternatives:
            missing_data = all(
                not param_availability['parameters'][param]['has_data']
                for param in parameters
            )
            
            if missing_data:
                logger.info("No data found for requested parameters. Looking for alternatives...")
                
                # Find alternative parameters for each site
                alt_params = set()
                param_alternatives = {param: [] for param in parameters}
                
                for site in sites:
                    if site in param_availability['sites']:
                        site_params = param_availability['sites'][site]['available_parameters']
                        
                        # Find alternatives for each requested parameter
                        for param in parameters:
                            alts = self._find_alternative_parameters(site_params, param)
                            param_alternatives[param].extend(alts)
                            alt_params.update(alts)
                
                if alt_params:
                    logger.info(f"Found {len(alt_params)} alternative parameters: {alt_params}")
                    
                    # Get data for alternative parameters
                    alt_result, alt_availability = self._get_site_data_basic(
                        sites, list(alt_params), start_date, end_date)
                    
                    # Add alternative parameter information to availability info
                    param_availability['alternative_parameters'] = param_alternatives
                    param_availability['parameters'].update(alt_availability['parameters'])
                    
                    # Merge alternative data with original result
                    if not alt_result.empty:
                        # Ensure we have all columns from both DataFrames
                        for col in alt_result.columns:
                            if col not in result.columns:
                                result[col] = None
                        
                        # Update with alternative data
                        result.update(alt_result)
                        
                        logger.info("Successfully added alternative parameter data")
                else:
                    logger.warning("No suitable alternative parameters found")
        
        return result, param_availability

    def _get_site_data_basic(self, sites: List[str], parameters: List[str],
                            start_date: str, end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Get basic site data for a batch of sites and parameters."""
        try:
            nwis = NWIS()
            
            # Initialize parameter availability tracking with enhanced stats
            parameter_availability = {
                'parameters': {
                    param: {
                        'has_data': False,
                        'sites': [],
                        'measurements': 0,
                        'stats': {
                            'min': None,
                            'max': None,
                            'mean': None,
                            'missing_count': 0,
                            'missing_percent': 0
                        }
                    } for param in parameters
                },
                'sites': {
                    site: {
                        'available_parameters': [],
                        'has_data': False,
                        'total_measurements': 0,
                        'data_quality': 'unknown'
                    } for site in sites
                },
                'total_sites': len(sites),
                'sites_with_data': set()
            }

            # Get site metadata first
            site_info = None
            try:
                site_info = nwis.get_info([{"site": site, "siteStatus": "all"} for site in sites])
                if not site_info.empty:
                    for _, site_row in site_info.iterrows():
                        self.logger.info(f"Site {site_row['site_no']}: {site_row['station_nm']}")
            except Exception as e:
                self.logger.warning(f"Could not get site info: {str(e)}")

            # Get parameter catalog for all sites in one batch
            param_services = {}  # Will store best service for each parameter
            try:
                url = "https://waterservices.usgs.gov/nwis/site"
                site_list = ','.join(sites)
                payload = {
                    "format": "rdb",
                    "sites": site_list,
                    "seriesCatalogOutput": "true",
                    "siteStatus": "all"
                }
                catalog_df = NWIS.retrieve_rdb(url, [payload])
                
                if not catalog_df.empty:
                    # Process catalog to determine best service for each parameter
                    for param in parameters:
                        param_data = catalog_df[catalog_df['parm_cd'] == param]
                        if not param_data.empty:
                            # Check each service type in order of preference
                            service_found = False
                            for service_type in ['uv', 'dv', 'qw']:
                                service_mask = param_data['data_type_cd'].str.lower().str.contains(
                                    '|'.join([service_type, 'unit' if service_type == 'uv' else '',
                                             'instantaneous' if service_type == 'uv' else '',
                                             'daily' if service_type == 'dv' else '']),
                                    na=False
                                )
                                if any(service_mask):
                                    param_services[param] = service_type
                                    service_found = True
                                    self.logger.info(f"Parameter {param} will use {service_type} service")
                                    break
                        
                            if not service_found:
                                self.logger.warning(f"No suitable service found for parameter {param}, will try all")
                                param_services[param] = 'all'
                        else:
                            self.logger.warning(f"Parameter {param} not found in catalog, will try all services")
                            param_services[param] = 'all'
                else:
                    self.logger.warning("Empty parameter catalog, will try all services for all parameters")
                    param_services = {param: 'all' for param in parameters}
                
            except Exception as e:
                self.logger.warning(f"Error getting parameter catalog: {str(e)}, will try all services")
                param_services = {param: 'all' for param in parameters}

            # Function to split date range into chunks
            def get_date_chunks(start: str, end: str, chunk_days: int = 100) -> List[Tuple[str, str]]:
                start_date = datetime.strptime(start, "%Y-%m-%d")
                end_date = datetime.strptime(end or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")
                chunks = []
                current = start_date
                while current < end_date:
                    chunk_end = min(current + timedelta(days=chunk_days), end_date)
                    chunks.append((
                        current.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d")
                    ))
                    current = chunk_end + timedelta(days=1)
                return chunks

            # Get data in chunks
            date_chunks = get_date_chunks(start_date, end_date)
            all_data = []
            
            # Define quality thresholds
            QUALITY_THRESHOLDS = {
                'Temperature (00010)': {'min': -5, 'max': 40},  # °C
                'Discharge (00060)': {'min': -1e6, 'max': 1e6},  # ft³/s
                'Specific conductance (00095)': {'min': 0, 'max': 5000},  # µS/cm
                'Dissolved oxygen (00300)': {'min': 0, 'max': 20},  # mg/L
                'pH (00400)': {'min': 0, 'max': 14},  # standard units
                'Turbidity (63680)': {'min': 0, 'max': 1000}  # NTU
            }

            for chunk_start, chunk_end in date_chunks:
                self.logger.info(f"\nRetrieving data for period {chunk_start} to {chunk_end}")
                
                # Group parameters by service
                service_params = {'uv': [], 'dv': [], 'qw': [], 'all': []}
                for param, service in param_services.items():
                    if service == 'all':
                        service_params['all'].append(param)
                    else:
                        service_params[service].append(param)
                
                # Handle parameters that need to try all services
                if service_params['all']:
                    for param in service_params['all']:
                        for service in ['uv', 'dv', 'qw']:
                            service_params[service].append(param)
                
                # Make requests for each service type
                service_data = {'uv': [], 'dv': [], 'qw': []}
                for service_type in ['uv', 'dv', 'qw']:
                    if not service_params[service_type]:
                        continue
                        
                    try:
                        url = f"https://waterservices.usgs.gov/nwis/{service_type}/"
                        payload = {
                            "format": "rdb",
                            "sites": ','.join(sites),
                            "startDT": chunk_start,
                            "endDT": chunk_end,
                            "parameterCd": ','.join(service_params[service_type])
                        }
                        
                        if service_type == 'qw':
                            payload.update({
                                "sampleMedia": "water",
                                "sorted": "no"
                            })
                        
                        self.logger.info(f"Requesting {service_type} data for parameters: {service_params[service_type]}")
                        
                        try:
                            data = NWIS.retrieve_rdb(url, [payload])
                            if data is not None and not data.empty:
                                # Verify we got actual data columns
                                value_columns = []
                                for param in service_params[service_type]:
                                    # Look for columns containing the parameter code
                                    param_cols = [col for col in data.columns if param in col]
                                    # Filter out qualification columns
                                    value_cols = [col for col in param_cols if not col.endswith('_cd')]
                                    if value_cols:
                                        value_columns.extend(value_cols)
                                
                                if value_columns:
                                    service_data[service_type].append(data)
                                    self.logger.info(f"Retrieved {len(data)} records with value columns: {value_columns}")
                                else:
                                    self.logger.warning(f"No value columns found in {service_type} response")
                        
                        except IndexError as ie:
                            self.logger.warning(f"Index error in {service_type} data retrieval: {str(ie)}")
                            continue
                        except Exception as e:
                            self.logger.warning(f"Error retrieving {service_type} data: {str(e)}")
                            continue
                        
                    except Exception as e:
                        self.logger.warning(f"Error setting up {service_type} request: {str(e)}")
                        continue

                # Process each service type's data
                for service_type, data_list in service_data.items():
                    if not data_list:
                        continue
                        
                    try:
                        data = pd.concat(data_list, ignore_index=True) if len(data_list) > 1 else data_list[0]
                        
                        # Convert datetime
                        if service_type == 'qw':
                            date_col = next((col for col in data.columns if 'sample' in col.lower() and 'date' in col.lower()), None)
                            if date_col:
                                data['datetime'] = pd.to_datetime(data[date_col])
                        else:
                            data['datetime'] = pd.to_datetime(data['datetime'])
                        
                        # Find and process value columns
                        for param in parameters:
                            param_info = self._get_parameter_info_by_code(param)
                            param_name = param_info['parameter_nm'] if param_info else f"Parameter {param}"
                            param_unit = param_info.get('unit', '') if param_info else ''
                            
                            # Format column name with units
                            col_name = self._format_column_name(param, {
                                'name': param_name,
                                'unit': param_unit
                            })
                            
                            # Find value column
                            value_col = None
                            if service_type == 'qw':
                                value_cols = [c for c in data.columns if 'result' in c.lower() and 'va' in c.lower()]
                                if value_cols:
                                    value_col = value_cols[0]
                            else:
                                # Look for exact parameter code match first
                                value_cols = [c for c in data.columns if param in c and not c.endswith('_cd')]
                                if value_cols:
                                    # Prefer columns without additional qualifiers
                                    base_cols = [c for c in value_cols if c == param]
                                    value_col = base_cols[0] if base_cols else value_cols[0]
                            
                            if value_col:
                                # Convert to numeric and apply quality filters
                                data[col_name] = pd.to_numeric(data[value_col], errors='coerce')
                                
                                # Apply quality thresholds
                                if col_name in QUALITY_THRESHOLDS:
                                    thresholds = QUALITY_THRESHOLDS[col_name]
                                    mask = (data[col_name] >= thresholds['min']) & (data[col_name] <= thresholds['max'])
                                    invalid_count = (~mask).sum()
                                    if invalid_count > 0:
                                        self.logger.warning(
                                            f"Found {invalid_count} values outside valid range "
                                            f"[{thresholds['min']}, {thresholds['max']}] for {col_name}"
                                        )
                                    data.loc[~mask, col_name] = np.nan
                                
                                # Update parameter availability stats
                                for site in sites:
                                    site_data = data[data['site_no'] == site][col_name]
                                    valid_count = site_data.notna().sum()
                                    
                                    if valid_count > 0:
                                        # Update parameter stats
                                        param_stats = parameter_availability['parameters'][param]['stats']
                                        if param_stats['min'] is None or site_data.min() < param_stats['min']:
                                            param_stats['min'] = float(site_data.min())
                                        if param_stats['max'] is None or site_data.max() > param_stats['max']:
                                            param_stats['max'] = float(site_data.max())
                                        
                                        current_mean = param_stats['mean']
                                        current_count = parameter_availability['parameters'][param]['measurements']
                                        new_values = site_data.notna().sum()
                                        new_mean = site_data.mean()
                                        
                                        # Update running mean
                                        if current_mean is None:
                                            param_stats['mean'] = float(new_mean)
                                        else:
                                            param_stats['mean'] = float(
                                                (current_mean * current_count + new_mean * new_values) /
                                                (current_count + new_values)
                                            )
                                        
                                        # Update counts
                                        parameter_availability['parameters'][param]['has_data'] = True
                                        if site not in parameter_availability['parameters'][param]['sites']:
                                            parameter_availability['parameters'][param]['sites'].append(site)
                                        parameter_availability['parameters'][param]['measurements'] += valid_count
                                        
                                        # Update site stats
                                        parameter_availability['sites'][site]['has_data'] = True
                                        parameter_availability['sites'][site]['total_measurements'] += valid_count
                                        if param not in parameter_availability['sites'][site]['available_parameters']:
                                            parameter_availability['sites'][site]['available_parameters'].append(param)
                                        parameter_availability['sites_with_data'].add(site)
                                        
                                        # Calculate missing value stats
                                        total_possible = len(site_data)
                                        missing_count = total_possible - valid_count
                                        param_stats['missing_count'] += missing_count
                                        param_stats['missing_percent'] = (
                                            param_stats['missing_count'] /
                                            parameter_availability['parameters'][param]['measurements'] * 100
                                        )
                        
                        all_data.append(data)
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing {service_type} data: {str(e)}")
                        continue

            if not all_data:
                self.logger.warning(f"No data found for sites {sites} and parameters {parameters}")
                return pd.DataFrame(), parameter_availability

            # Combine all data and create final dataset
            try:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Create result DataFrame with consistent column naming
                result_columns = ['datetime', 'site_no']
                param_columns = [
                    self._format_column_name(param, {
                        'name': self._get_parameter_info_by_code(param)['parameter_nm'] if self._get_parameter_info_by_code(param) else f"Parameter {param}",
                        'unit': self._get_parameter_info_by_code(param).get('unit', '') if self._get_parameter_info_by_code(param) else ''
                    })
                    for param in parameters
                ]
                
                # Ensure all parameter columns exist
                for col in param_columns:
                    if col not in combined_data.columns:
                        combined_data[col] = np.nan
                
                result = combined_data[result_columns + param_columns].copy()
                
                # Rename columns to standard format
                result = result.rename(columns={
                    'site_no': 'Site ID',
                    'datetime': 'Date'
                })
                
                # Add site metadata if available
                if site_info is not None and not site_info.empty:
                    result = result.merge(
                        site_info[['site_no', 'station_nm', 'dec_lat_va', 'dec_long_va']].rename(columns={
                            'site_no': 'Site ID',
                            'station_nm': 'Site Name',
                            'dec_lat_va': 'Latitude',
                            'dec_long_va': 'Longitude'
                        }),
                        on='Site ID',
                        how='left'
                    )
                
                # Sort by date and site
                result = result.sort_values(['Date', 'Site ID'])
                
                # Assess data quality for each site
                for site in sites:
                    site_data = result[result['Site ID'] == site]
                    if not site_data.empty:
                        missing_rates = []
                        for param in parameters:
                            if param in parameter_availability['sites'][site]['available_parameters']:
                                col_name = next(col for col in param_columns if param in col)
                                missing_rate = site_data[col_name].isna().mean() * 100
                                missing_rates.append(missing_rate)
                        
                        if missing_rates:
                            avg_missing_rate = sum(missing_rates) / len(missing_rates)
                            if avg_missing_rate < 5:
                                parameter_availability['sites'][site]['data_quality'] = 'excellent'
                            elif avg_missing_rate < 20:
                                parameter_availability['sites'][site]['data_quality'] = 'good'
                            elif avg_missing_rate < 50:
                                parameter_availability['sites'][site]['data_quality'] = 'fair'
                            else:
                                parameter_availability['sites'][site]['data_quality'] = 'poor'
                
                return result, parameter_availability
                
            except Exception as e:
                self.logger.error(f"Error creating final dataset: {str(e)}")
                return pd.DataFrame(), parameter_availability
            
        except Exception as e:
            self.logger.error(f"Error retrieving data: {str(e)}")
            return pd.DataFrame(), parameter_availability

    def get_site_summary(self, sites: List[str]) -> pd.DataFrame:
        """Get summary of available data for sites.
        
        Args:
            sites: List of site IDs
            
        Returns:
            DataFrame summarizing available data
        """
        results = []
        for site_id in sites:
            try:
                # Get site info
                site_info = self.nwis.get_info([{"site": site_id, "siteStatus": "all"}])
                if not site_info.empty:
                    info = site_info.iloc[0]
                    
                    # Get available parameters
                    url = "https://waterservices.usgs.gov/nwis/site"
                    payload = {
                        "format": "rdb",
                        "sites": site_id,
                        "seriesCatalogOutput": "true",
                        "siteStatus": "all"
                    }
                    
                    catalog = NWIS.retrieve_rdb(url, [payload])
                    param_count = len(catalog['parm_cd'].unique()) if not catalog.empty else 0
                    
                    # Get date range
                    if not catalog.empty:
                        begin_dates = pd.to_datetime(catalog['begin_date'])
                        end_dates = pd.to_datetime(catalog['end_date'])
                        begin_date = begin_dates.min().strftime('%Y-%m-%d')
                        end_date = end_dates.max().strftime('%Y-%m-%d')
                    else:
                        begin_date = None
                        end_date = None
                    
                    results.append({
                        'site_no': info['site_no'],
                        'station_nm': info['station_nm'],
                        'dec_lat_va': info['dec_lat_va'],
                        'dec_long_va': info['dec_long_va'],
                        'site_tp_cd': info['site_tp_cd'],
                        'site_tp_desc': info.get('site_tp_desc', ''),
                        'parameter_count': param_count,
                        'begin_date': begin_date,
                        'end_date': end_date
                    })
            except Exception as e:
                logger.error(f"Error getting summary for site {site_id}: {str(e)}")
                
        return pd.DataFrame(results)
        
    def create_llm_context(self) -> Dict[str, Any]:
        """Create compact context for LLM from cached data."""
        if not self.parameters:
            return {}
            
        # Get top parameters by usage
        top_params = []
        for param in self.parameters['parameters'][:30]:  # Top 30 parameters
            top_params.append({
                'code': param.get('parameter_cd', ''),
                'name': param['parameter_nm'],
                'description': param.get('description', ''),
                'unit': param.get('unit', ''),
                'group': param['parameter_group']
            })
        
        # Organize parameters by category
        categories = {}
        for param in self.parameters['parameters']:
            group = param['parameter_group']
            if group not in categories:
                categories[group] = {'count': 0, 'examples': []}
            categories[group]['count'] += 1
            if len(categories[group]['examples']) < 3:
                categories[group]['examples'].append(param['parameter_nm'])
        
        return {
            'summary': {
                'total_parameters': len(self.parameters['parameters']),
                'last_updated': self.parameters.get('last_updated', datetime.now().strftime('%Y-%m-%d'))
            },
            'parameters': {
                'common': top_params,
                'categories': categories
            },
            'capabilities': {
                'search': ['location', 'parameter', 'site'],
                'data': ['temporal', 'spatial'],
                'formats': ['RDB', 'DataFrame']
            }
        }
        
    def get_service_info(self) -> str:
        """Get formatted service information.
        
        Returns:
            Formatted string describing the service
        """
        # Core service description
        core_info = {
            "overview": {
                "description": "The USGS Water Quality Portal provides access to water-resources data collected at approximately 1.9 million sites across the United States and its territories.",
                "coverage": "All 50 States, District of Columbia, Puerto Rico, Virgin Islands, Guam, American Samoa, and Northern Mariana Islands",
                "mission": "The USGS investigates the occurrence, quantity, quality, distribution, and movement of surface and underground waters and disseminates the data to the public, State and local governments, public and private utilities, and other Federal agencies."
            },
            "data_types": {
                "physical": ["Temperature", "Specific conductance", "Turbidity", "Flow"],
                "chemical": ["pH", "Dissolved oxygen", "Nutrients", "Major ions", "Metals"],
                "biological": ["Bacteria", "Algae", "Chlorophyll"],
                "sediment": ["Suspended solids", "Bed material", "Particle size"]
            },
            "search_capabilities": {
                "location_based": [
                    "Point-radius search (minimum 20-mile radius)",
                    "Bounding box for water bodies",
                    "Site-specific queries"
                ],
                "temporal": [
                    "Historical data archives",
                    "Date range filtering",
                    "Time series data"
                ],
                "parameter_based": [
                    "Single or multiple parameters",
                    "Parameter groups",
                    "Quality indicators"
                ]
            },
            "data_formats": [
                "RDB (USGS Relational Database)",
                "CSV",
                "Structured DataFrame"
            ]
        }

        if self.parameters:
            param_groups = {}
            for param in self.parameters['parameters']:
                group = param['parameter_group']
                if group not in param_groups:
                    param_groups[group] = {'count': 0, 'examples': []}
                param_groups[group]['count'] += 1
                if len(param_groups[group]['examples']) < 3:
                    param_groups[group]['examples'].append(param['parameter_nm'])
            
            core_info["available_parameters"] = {
                "total_count": len(self.parameters['parameters']),
                "groups": param_groups
            }

        return json.dumps(core_info, indent=2)
        
    def get_parameter_info(self, parameter_query: str) -> str:
        """Get formatted parameter information based on a query.
        
        Args:
            parameter_query: Search string for parameters
            
        Returns:
            Formatted string describing matching parameters
        """
        matching_params = self.find_parameters(parameter_query)
        if not matching_params:
            return f"No parameters found matching '{parameter_query}'."
            
        # Group parameters by category
        params_by_group = {}
        for param in matching_params:
            group = param['parameter_group']
            if group not in params_by_group:
                params_by_group[group] = []
            params_by_group[group].append(param)
            
        # Create summary
        info = [
            f"Found {len(matching_params)} parameters matching '{parameter_query}':",
            ""  # Empty line for spacing
        ]
        
        for group, params in params_by_group.items():
            info.append(f"{group} Parameters:")
            for param in params:
                param_info = [f"- {param['parameter_nm']} (Code: {param['parameter_cd']})"]
                if param.get('description'):
                    param_info.append(f"  Description: {param['description']}")
                if param.get('unit'):
                    param_info.append(f"  Unit: {param['unit']}")
                info.extend(param_info)
            info.append("")  # Empty line between groups
            
        return "\n".join(info)
        
    def test_data_retrieval(self, site_id: str = "11181300", parameter: str = "00010", 
                           start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
        """Test basic data retrieval functionality.
        
        Args:
            site_id: Site ID to test
            parameter: Parameter code to test
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with test results
        """
        logger.info(f"Testing data retrieval for site {site_id}, parameter {parameter}")
        
        # Test direct NWIS call first
        try:
            # Try instantiating NWIS with different configurations
            nwis = NWIS()
            
            # Test using get_data method
            logger.info("Testing NWIS.get_data method...")
            try:
                data = nwis.get_data(
                    sites=[site_id],
                    start_date=start_date,
                    end_date=end_date,
                    parameterCd=[parameter],
                    service="dv"
                )
                logger.info(f"get_data result shape: {data.shape if data is not None else 'None'}")
                logger.info(f"get_data columns: {data.columns.tolist() if data is not None else 'None'}")
                if len(data) > 0:
                    logger.info(f"First row of get_data:\n{data.iloc[0]}")
                return data
            except Exception as e:
                logger.error(f"Error with get_data: {str(e)}")
            
            # If get_data fails, try direct retrieve_rdb
            logger.info("\nTesting direct retrieve_rdb...")
            url = "https://waterservices.usgs.gov/nwis/dv/"
            payload = {
                "format": "rdb",
                "sites": site_id,
                "startDT": start_date,
                "endDT": end_date,
                "parameterCd": parameter,
                "siteStatus": "all"
            }
            logger.info(f"Making direct NWIS call with payload: {payload}")
            
            data = NWIS.retrieve_rdb(url, [payload])
            logger.info(f"Direct NWIS call result shape: {data.shape if data is not None else 'None'}")
            logger.info(f"Direct NWIS call columns: {data.columns.tolist() if data is not None else 'None'}")
            if len(data) > 0:
                logger.info(f"First row of direct call:\n{data.iloc[0]}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error in test retrieval: {str(e)}")
            return pd.DataFrame()

    def _get_parameter_info_by_code(self, param_code: str) -> Optional[Dict[str, str]]:
        """Get parameter information by code.
        
        Args:
            param_code: USGS parameter code
            
        Returns:
            Dictionary with parameter information or None if not found
        """
        if not self.parameters:
            return None
        
        param_info = next(
            (p for p in self.parameters['parameters'] 
             if p['parameter_cd'] == param_code),
            None
        )
        return param_info

    def _get_descriptive_column_name(self, param_code: str) -> str:
        """Get a human-readable column name for a parameter.
        
        Args:
            param_code: USGS parameter code
            
        Returns:
            Descriptive column name with units if available
        """
        param_info = self._get_parameter_info_by_code(param_code)
        if not param_info:
            return f"Parameter {param_code}"
        
        name = param_info['parameter_nm'].split(',')[0].strip()
        unit = param_info.get('unit', '').strip()
        
        # Standardize common units
        unit_map = {
            'degrees celsius': '°C',
            'milligrams per liter': 'mg/L',
            'standard units': '',  # pH doesn't need units in header
            'nephelometric turbidity units': 'NTU',
            'microsiemens per centimeter': 'µS/cm',
            'feet': 'ft',
            'meters': 'm',
            'cubic feet per second': 'cfs'
        }
        
        if unit.lower() in unit_map:
            unit = unit_map[unit.lower()]
        
        return f"{name} ({unit})" if unit else name

    def find_parameters_by_description(self, description: str) -> List[Dict[str, Any]]:
        """Find parameters matching a natural language description.
        
        Args:
            description: Natural language description of desired parameters
            
        Returns:
            List of matching parameter dictionaries with codes and descriptions
        """
        description = description.lower()
        matches = []
        
        if not self.parameters:
            return matches
        
        # Define common parameter mappings
        common_terms = {
            'temperature': ['00010', '00011'],
            'water temperature': ['00010'],
            'air temperature': ['00020'],
            'flow': ['00060', '00061'],
            'discharge': ['00060', '00061'],
            'streamflow': ['00060'],
            'ph': ['00400'],
            'conductivity': ['00095'],
            'dissolved oxygen': ['00300'],
            'turbidity': ['63680'],
            'nitrate': ['00618'],
            'phosphate': ['00660'],
            'chlorophyll': ['32209', '32210'],
            'salinity': ['00480'],
        }
        
        # Check for exact matches in common terms
        for term, codes in common_terms.items():
            if term in description:
                for code in codes:
                    param_info = self._get_parameter_info_by_code(code)
                    if param_info:
                        matches.append(param_info)
        
        # If no common term matches, do a broader search
        if not matches:
            for param in self.parameters['parameters']:
                if (description in param['parameter_nm'].lower() or
                    description in param.get('description', '').lower()):
                    matches.append(param)
        
        return matches

    def _convert_characteristic_to_param_code(self, characteristic: str) -> Optional[str]:
        """Convert a characteristic name to USGS parameter code.
        
        Args:
            characteristic: Characteristic name
            
        Returns:
            USGS parameter code if found, None otherwise
        """
        if not self.parameters:
            return None
        
        # Clean up the characteristic name
        char_lower = characteristic.lower().strip()
        
        # First try exact matches
        for param in self.parameters['parameters']:
            name_lower = param['parameter_nm'].lower()
            if char_lower == name_lower.split(',')[0].strip():
                return param['parameter_cd']
        
        # Then try partial matches
        for param in self.parameters['parameters']:
            name_lower = param['parameter_nm'].lower()
            if char_lower in name_lower:
                return param['parameter_cd']
        
        return None

    def _convert_param_code_to_characteristic(self, param_code: str) -> Optional[str]:
        """Convert a USGS parameter code to characteristic name.
        
        Args:
            param_code: USGS parameter code
            
        Returns:
            Characteristic name if found, None otherwise
        """
        param_info = self._get_parameter_info_by_code(param_code)
        if param_info:
            # Take the first part of the name before any commas
            return param_info['parameter_nm'].split(',')[0].strip()
        return None 