"""
Data management for MONet service.

This module handles data loading, caching, and analysis for the MONet service.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from urllib.parse import urljoin
import json
from functools import lru_cache
from tqdm import tqdm

from .models import MONetConfig, GeoPoint, GeoBBox

class MONetDataManager:
    """Manages data loading and analysis for MONet service."""
    
    def __init__(self, config: MONetConfig):
        """Initialize data manager with configuration."""
        self.config = config
        self._unified_df: Optional[pd.DataFrame] = None
        self._cache_timestamp: Optional[datetime] = None
        self._stats: Optional[Dict[str, Any]] = None
        self._column_descriptions: Optional[Dict[str, str]] = None
        self._geographic_coverage: Optional[Dict[str, Any]] = None
        
        # Load data and calculate statistics immediately
        print("Initializing MONet database...")
        self._unified_df = self._build_unified_dataframe()
        self._calculate_all_statistics()
        print("Initialization complete.")
        
    def _calculate_all_statistics(self) -> None:
        """Calculate and store all statistics about the DataFrame."""
        if self._unified_df is None or self._unified_df.empty:
            print("Warning: No data available for statistics calculation")
            return
            
        df = self._unified_df
        
        # Calculate basic statistics
        self._stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': {},
            'categorical_columns': {},
            'temporal_columns': {}
        }
        
        # Process each column
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    col_stats = df[col].describe()
                    self._stats['numeric_columns'][col] = {
                        'mean': float(col_stats['mean']),
                        'std': float(col_stats['std']),
                        'min': float(col_stats['min']),
                        'max': float(col_stats['max'])
                    }
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    self._stats['temporal_columns'][col] = {
                        'min': df[col].min().isoformat(),
                        'max': df[col].max().isoformat()
                    }
                else:
                    unique_vals = df[col].nunique()
                    self._stats['categorical_columns'][col] = {
                        'unique_values': unique_vals,
                        'top_values': df[col].value_counts().head(5).to_dict()
                    }
            except Exception as e:
                print(f"Error calculating statistics for column {col}: {str(e)}")
                continue
        
        # Calculate geographic coverage
        lat_col = 'lat_lon_latitude'
        lon_col = 'lat_lon_longitude'
        if lat_col in df.columns and lon_col in df.columns:
            lat_stats = df[lat_col].describe()
            lon_stats = df[lon_col].describe()
            self._geographic_coverage = {
                'latitude_range': {
                    'min': lat_stats['min'],
                    'max': lat_stats['max'],
                    'mean': lat_stats['mean']
                },
                'longitude_range': {
                    'min': lon_stats['min'],
                    'max': lon_stats['max'],
                    'mean': lon_stats['mean']
                },
                'total_locations': len(df.dropna(subset=[lat_col, lon_col]))
            }
        
        # Update column descriptions with unit information
        self._column_descriptions = self._get_base_column_descriptions()
        for col in df.columns:
            if col.endswith('_has_numeric_value'):
                unit_col = col.replace('_has_numeric_value', '_has_unit')
                if unit_col in df.columns:
                    units = df[unit_col].unique()
                    units = [u for u in units if pd.notna(u)]
                    if units and col in self._column_descriptions:
                        self._column_descriptions[col] += f" (units: {', '.join(units)})"
        
    def _get_base_column_descriptions(self) -> Dict[str, str]:
        """Get base descriptions for DataFrame columns without unit information."""
        return {
            # Core measurements
            'total_carbon_has_numeric_value': 'Total carbon content in soil sample',
            'total_organic_carbon_has_numeric_value': 'Organic carbon content in soil sample',
            'total_nitrogen_has_numeric_value': 'Total nitrogen content in soil sample',
            'total_kjeldahl_nitrogen_has_numeric_value': 'Total Kjeldahl nitrogen (organic nitrogen + ammonia)',
            'total_sulfur_has_numeric_value': 'Total sulfur content in soil sample',
            
            # Nutrient measurements
            'nh4_n_has_numeric_value': 'Ammonium nitrogen content',
            'no3_n_has_numeric_value': 'Nitrate nitrogen content',
            'sulfate_has_numeric_value': 'Sulfate content',
            
            # Element measurements
            'calcium_has_numeric_value': 'Calcium content in soil',
            'magnesium_has_numeric_value': 'Magnesium content in soil',
            'potassium_has_numeric_value': 'Potassium content in soil',
            'sodium_has_numeric_value': 'Sodium content in soil',
            'iron_has_numeric_value': 'Iron content in soil',
            'manganate_has_numeric_value': 'Manganese content in soil',
            'zinc_has_numeric_value': 'Zinc content in soil',
            'copper_has_numeric_value': 'Copper content in soil',
            'boron_has_numeric_value': 'Boron content in soil',
            'phosporous_has_numeric_value': 'Phosphorous content in soil',
            
            # Soil properties
            'bulk_density_has_numeric_value': 'Soil bulk density',
            'gwc_percent_has_numeric_value': 'Gravimetric water content',
            'ph': 'Soil pH',
            'cation_exchange_capacity_has_numeric_value': 'Cation exchange capacity',
            'total_bases_has_numeric_value': 'Total bases in soil',
            
            # Microbial properties
            'mbc_has_numeric_value': 'Microbial biomass carbon',
            'mbn_has_numeric_value': 'Microbial biomass nitrogen',
            'respiration_rate_per_day_has_numeric_value': 'Soil microbial respiration rate',
            
            # Location information
            'lat_lon_latitude': 'Sample location latitude',
            'lat_lon_longitude': 'Sample location longitude',
            'elev_has_value_unit_value': 'Sample elevation above sea level',
            'geo_loc_name': 'Geographic location name',
            
            # Sample information
            'sample_id': 'Unique identifier for the sample',
            'sample_name': 'Name of the sample',
            'subsample_name': 'Name of the subsample',
            'collection_date': 'Date when the sample was collected',
            'sample_type': 'Type of sample (e.g., soil, sediment)',
            'core_section': 'Section of the core sample',
            
            # Project metadata
            'project_id': 'Unique identifier for the project',
            'study_id': 'Unique identifier for the study',
            'proposal_id': 'Unique identifier for the proposal',
            'proposal_title': 'Title of the research proposal',
            'principal_investigator': 'Lead researcher conducting the study',
            'collaborating_institution': 'Institution collaborating on the study',
            'project_start': 'Start date of the project',
            'project_end': 'End date of the project'
        }
    
    def get_dataframe_context(self) -> Dict[str, Any]:
        """Get context information about the DataFrame."""
        return {
            'column_descriptions': self._column_descriptions,
            'statistics': self._stats,
            'geographic_coverage': self._geographic_coverage
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """Get cached statistics about the DataFrame."""
        return self._stats
    
    def _get_geographic_coverage(self) -> Dict[str, Any]:
        """Get cached geographic coverage information."""
        return self._geographic_coverage
    
    def _get_column_descriptions(self) -> Dict[str, str]:
        """Get cached column descriptions."""
        return self._column_descriptions
    
    @property
    def unified_df(self) -> pd.DataFrame:
        """Get unified DataFrame, refreshing if needed."""
        if self._unified_df is None:
            self._unified_df = self._build_unified_dataframe()
        return self._unified_df
    
    def _build_unified_dataframe(self) -> pd.DataFrame:
        """Build unified DataFrame from all data sources."""
        try:
            # Fetch all data first
            studies = self._fetch_dataset('study')
            sampling_activities = self._fetch_dataset('samplingactivities')
            samples = self._fetch_dataset('samples')
            processed_data = self._fetch_dataset('processeddata')

            # Handle samples response format
            if isinstance(samples, dict) and 'samples' in samples:
                samples = samples['samples']
            elif isinstance(samples, list):
                pass
            else:
                print("Warning: Unexpected samples format")
                samples = []

            if not all([studies, sampling_activities, samples, processed_data]):
                print("Warning: Some datasets are empty")
                return pd.DataFrame()

            # Collect all DataFrames to concatenate at the end
            df_list = []
                
            # Process studies with progress bar
            for study in tqdm(studies, desc="Building unified DataFrame"):
                # Create study DataFrame
                study_df = pd.DataFrame([self._flatten_dict(study)])
                study_df.rename(columns={'id': 'study_id'}, inplace=True)
                
                # Find related sampling activities
                for activity in sampling_activities:
                    if activity['study_id'] != study['id']:
                        continue
                        
                    # Create activity DataFrame
                    activity_df = pd.DataFrame([self._flatten_dict(activity)])
                    activity_df.rename(columns={
                        'id': 'samplingactivity_id',
                        'type': 'samplingactivity_type'
                    }, inplace=True)
                    activity_df.drop(columns=['study_id', 'sample_id', 'sample_name'], errors='ignore', inplace=True)
                    
                    # Find related samples
                    for sample in samples:
                        if sample['id'] not in activity['sample_id']:
                            continue
                            
                        # Create sample DataFrame
                        sample_df = pd.DataFrame([self._flatten_dict(sample)])
                        sample_df.rename(columns={'id': 'sample_id', 'type': 'sample_type'}, inplace=True)
                        
                        # Find related processed data
                        if 'unique_ID' in sample and 'sampling_set' in sample:
                            subsamples = [
                                d for d in processed_data 
                                if (str(d.get('proposal_id')) == str(sample['unique_ID']) and 
                                    str(d.get('sampling_set')) == str(sample['sampling_set']))
                            ]
                            
                            # Group by subsample name
                            for subsample_name in list({m['sample_name'] for m in subsamples}):
                                measurements = [m for m in subsamples if m['sample_name'] == subsample_name]
                                
                                if measurements:
                                    # Create measurements DataFrame
                                    measurements_df = pd.DataFrame([self._flatten_dict(m) for m in measurements])
                                    measurements_df.rename(columns={
                                        'id': 'processeddata_id',
                                        'sample_name': 'subsample_name'
                                    }, inplace=True)
                                    measurements_df.drop(columns=['sampling_set'], errors='ignore', inplace=True)
                                    
                                    # Combine all DataFrames
                                    combined_df = pd.concat([
                                        study_df.copy(),
                                        activity_df.copy(),
                                        sample_df.copy()
                                    ], axis=1)
                                    
                                    # Repeat combined_df to match measurements
                                    if len(measurements_df) > 1:
                                        combined_df = pd.concat([combined_df] * len(measurements_df), ignore_index=True)
                                    
                                    # Add measurements
                                    result_df = pd.concat([combined_df, measurements_df], axis=1)
                                    df_list.append(result_df)
            
            # Combine all results
            if not df_list:
                return pd.DataFrame()
                
            final_df = pd.concat(df_list, ignore_index=True, sort=False)
            
            # Drop known problematic columns
            final_df.drop(['lat_lon_has_raw_value', 'elev_has_raw_value'], 
                         axis=1, errors='ignore', inplace=True)
            final_df.drop(['rms_id','rms_has_unit','rms_has_numeric_value',
                          'percent_mz_assigned_id','percent_mz_assigned_has_unit',
                          'percent_mz_assigned_has_numeric_value'],
                          axis=1, errors='ignore', inplace=True)
            
            # Apply type coercion
            type_list = {
                col: 'float' if any(x in col for x in ['value', 'latitude', 'longitude', 'average']) 
                              or col in ['ph', 'aq', 'rep']
                       else 'int' if col == 'filesize'
                       else 'datetime' if col in ['project_start', 'project_end']
                       else 'str'
                for col in final_df.columns
            }
            
            final_df = self._coerce_column_types(final_df, type_list)
            
            return final_df
                
        except Exception as e:
            print(f"Error building unified DataFrame: {str(e)}")
            return pd.DataFrame()

    def _fetch_dataset(self, name: str) -> List[Dict]:
        """Fetch a specific dataset with proper error handling.
        
        Different endpoints have different response formats:
        - samples: Returns {'samples': [list of dicts]}
        - others: Returns [list of dicts] or {'data': [list of dicts]}
        """
        url = f"{self.config.base_url}/{name}"
        headers = {'accept': 'application/json'}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                
                # Special case for samples endpoint
                if name == 'samples':
                    if isinstance(data, dict) and 'samples' in data:
                        return data['samples']
                    else:
                        raise ValueError(f"Unexpected samples response format: missing 'samples' key")
                
                # Standard handling for other endpoints
                if isinstance(data, dict) and 'data' in data:
                    return data['data']
                elif isinstance(data, list):
                    return data
                else:
                    raise ValueError(f"Unexpected response format from {name}")
                    
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to fetch {name} after {max_retries} attempts: {str(e)}")
                continue
            except ValueError as e:
                raise Exception(f"Error parsing {name} response: {str(e)}")
            except Exception as e:
                raise Exception(f"Unexpected error fetching {name}: {str(e)}")

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _coerce_column_types(self, df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
        """Coerce DataFrame columns to specified types with error handling."""
        df = df.copy()
        for column, col_type in column_types.items():
            if column in df.columns:
                try:
                    if col_type == 'datetime':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif col_type == 'boolean':
                        df[column] = df[column].astype('bool')
                    elif col_type == 'category':
                        df[column] = df[column].astype('category')
                    elif col_type == 'float':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif col_type == 'int':
                        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
                    else:
                        df[column] = df[column].astype(col_type)
                except Exception as e:
                    print(f"Error converting column {column} to {col_type}: {e}")
        return df
    
    @lru_cache(maxsize=1000)
    def get_geo_filtered(self, 
                        point: Optional[GeoPoint] = None,
                        bbox: Optional[GeoBBox] = None) -> pd.DataFrame:
        """Get geo-filtered data with caching."""
        if not point and not bbox:
            return self.unified_df
            
        df = self.unified_df
        lat_col = 'lat_lon_latitude'
        lon_col = 'lat_lon_longitude'
        
        if point:
            # Convert radius to approximate degrees
            radius_deg = point.radius_km / 111.0 if point.radius_km else 0
            mask = (
                (df[lat_col] >= point.latitude - radius_deg) &
                (df[lat_col] <= point.latitude + radius_deg) &
                (df[lon_col] >= point.longitude - radius_deg) &
                (df[lon_col] <= point.longitude + radius_deg)
            )
            return df[mask]
            
        if bbox:
            mask = (
                (df[lat_col] >= bbox.min_lat) &
                (df[lat_col] <= bbox.max_lat) &
                (df[lon_col] >= bbox.min_lon) &
                (df[lon_col] <= bbox.max_lon)
            )
            return df[mask] 