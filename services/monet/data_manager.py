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
import time

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
        lat_col = 'latitude'
        lon_col = 'longitude'
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
        
    def _get_base_column_descriptions(self) -> Dict[str, str]:
        """Get base descriptions for DataFrame columns without unit information."""
        return {
            # Project metadata
            'title': 'Title of the project',
            'project_type': 'Type of the project',
            'active': 'Project active status',
            'accepted': 'Project acceptance status',
            'award_doi': 'DOI of the project award',
            'id': 'Unique identifier for the project',
            'uuid': 'UUID of the project',
            'uri': 'URI of the project',
            'abstract': 'Abstract of the project',
            'started_date': 'Start date of the project',
            'closed_date': 'Closed date of the project',
            'current_status': 'Current status of the project',
            'project_members': 'Members involved in the project',

            # Sampling information
            'sampling_set': 'Identifier for the sampling set',
            'study_id': 'Unique identifier for the study',
            'sample_type': 'Type of sample (e.g., soil, sediment)',
            'collection_date': 'Date when the sample was collected',
            'geolocation': 'Geographic location name',
            'latitude': 'Sample location latitude',
            'longitude': 'Sample location longitude',
            'elevation.value': 'Sample elevation above sea level',
            'elevation.unit': 'Unit of sample elevation',
            'neon_domain': 'NEON domain of the sample location',
            'crop_rotation': 'Crop rotation status',
            'cur_land_use': 'Current land use at the sample location',
            'cur_vegetation': 'Current vegetation at the sample location',

            # Soil metadata
            'core_collector': 'Person or entity that collected the core sample',
            'fao_class': 'FAO soil classification',
            'infiltration_notes': 'Notes on soil infiltration',
            'link_climate_info': 'Link to climate information',
            'drainage_class': 'Drainage class of the soil',
            'water_content_meth': 'Method used to measure water content',
            'soil_type': 'Type of soil',
            'soil_type_meth': 'Method used to classify soil type',
            'previous_land_use': 'Previous land use at the sample location',
            'agrochem_addition': 'Agrochemical addition status',
            'tillage': 'Tillage practices',
            'previous_land_use_meth': 'Method used to determine previous land use',
            'land_use': 'Current land use',
            'ecoregion': 'Ecoregion of the sample location',
            'vegetation': 'Vegetation at the sample location',

            # Sample information
            'id_sample': 'Unique identifier for the sample',
            'core_section': 'Section of the core sample',
            'density': 'Soil bulk density',
            'carbon': 'Total carbon content in soil sample',
            'nitrogen': 'Total nitrogen content in soil sample',
            'kj_nitro': 'Total Kjeldahl nitrogen (organic nitrogen + ammonia)',
            'sulfur': 'Total sulfur content in soil sample',
            'enzyme': 'Enzyme activity in soil sample',
            'sample_name': 'Name of the sample',
            'mz_percent': 'Mass-to-charge ratio percentage',
            'rms': 'Root mean square value',
            'hc_ratio': 'Hydrogen to carbon ratio',
            'oc_ratio': 'Oxygen to carbon ratio',
            'c_ratio': 'Carbon ratio',
            'dbe_average': 'Average double bond equivalent',
            'water_content': 'Gravimetric water content',

            # Nutrient measurements
            'sulfate': 'Sulfate content',
            'boron': 'Boron content in soil',
            'zinc': 'Zinc content in soil',
            'manganate': 'Manganese content in soil',
            'copper': 'Copper content in soil',
            'iron': 'Iron content in soil',
            'calcium': 'Calcium content in soil',
            'magnesium': 'Magnesium content in soil',
            'sodium': 'Sodium content in soil',
            'potassium': 'Potassium content in soil',
            'total_bases': 'Total bases in soil',
            'cation_exchange_capacity': 'Cation exchange capacity',

            # Microbial properties
            'mbc': 'Microbial biomass carbon',
            'mbn': 'Microbial biomass nitrogen',
            'nh4n': 'Ammonium nitrogen content',
            'no3n': 'Nitrate nitrogen content',
            'phosphorus': 'Phosphorus content in soil',
            'extraction': 'Extraction method used',
            'ph': 'Soil pH',
            'rate': 'Respiration rate of microbes in soil',

            # Soil texture
            'sand': 'Sand content in soil',
            'silt': 'Silt content in soil',
            'clay': 'Clay content in soil',
            'type': 'Type of soil texture',

            # Organic content
            'toc': 'Total organic carbon content',
            'tn': 'Total nitrogen content'
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
            # Fetch study data
            print("Fetching study data...")
            url = 'https://sc-data.emsl.pnnl.gov/study'
            headers = {'accept': 'application/json'}

            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                emsl_study = response.json()
            else:
                print(f"Request failed with status code {response.status_code}")
            df_study=pd.json_normalize(emsl_study)

            # Fetch study metadata
            print("Fetching study metadata...")
            study_meta=[]
            for sid in df_study['id']:
                url = f'https://sc-data.emsl.pnnl.gov/study/{sid}'
                headers = {
                    'accept': 'application/json',
                }
                response = requests.get(url, headers=headers)
                study_meta.append(response.json())
            df_study_meta=pd.json_normalize(study_meta)

            # Fetch samples
            print("Fetching samples...")
            samples=self.get_all_data('https://sc-data.emsl.pnnl.gov/sample',params={'per_page':100},accessor='samples')
            df_samples = pd.json_normalize(samples)
            df_samples.columns = [c.split('.')[1] if 'soil_metadata' in c else c for c in df_samples.columns] 

            # Fetch analysis metadata
            print("Fetching analysis metadata...")
            response = requests.get('https://sc-data.emsl.pnnl.gov/elastic/metadata', headers={'accept': 'application/json'})
            analysis_metadata=response.json()
            df_analysis=pd.json_normalize(analysis_metadata)
            df_analysis.columns=[c.split('.')[-1] for c in df_analysis.columns]

            analyses=["Bulk_Density","Elemental_Analysis","Enzyme","FTICR","Gravimetric_Water_Content","Ion_Analysis","Microbial_Biomass","Nitrogen_Extraction","Phosphorus_Extraction","pH","Respiration","Texture","TOC_TN"]
            print("Fetching measurements...")   
            measurements={}
            for a in analyses:
                measurements[a]=pd.json_normalize(self.get_all_data(f'https://sc-data.emsl.pnnl.gov/elastic/{a}',params={'per_page':100,'data':'true'}))    
            
            # Ensure consistent column types for proposal_id
            print("Ensuring consistent column types for proposal_id...")
            df_study['id'] = df_study['id'].astype(str)
            df_study_meta['id'] = df_study_meta['id'].astype(str)
            df_samples['proposal_id'] = df_samples['proposal_id'].astype(str)
            df_samples['sampling_set'] = df_samples['sampling_set'].astype(str)
            df_analysis['proposal_id'] = df_analysis['proposal_id'].astype(str)
            df_analysis['sampling_set'] = df_analysis['sampling_set'].astype(str)
            for measurement_type, measurement_df in measurements.items():
                measurement_df['proposal_id'] = measurement_df['proposal_id'].astype(str)
                measurement_df['sampling_set'] = measurement_df['sampling_set'].astype(str)

            print("Merging dataframes...")
            # Merge studies with study metadata
            merged_studies_df = pd.merge(df_study, df_study_meta, on='id', how='left',suffixes=('', '_meta'))

            # Merge samples with studies
            merged_samples_df = pd.merge(df_samples, merged_studies_df, left_on='proposal_id', right_on='id', how='left',suffixes=('_sample', ''))

            # Merge analysis metadata with samples
            merged_analysis_df = pd.merge(df_analysis, merged_samples_df, on=['proposal_id', 'sampling_set'], how='left',suffixes=('_analysis', ''))

            # Generate a list of unique core sections from all measurement DataFrames
            unique_core_sections = pd.concat([df[['core_section']] for df in measurements.values()]).drop_duplicates().reset_index(drop=True)

            # Expand merged_analysis_df for each core section
            expanded_master_df = merged_analysis_df.merge(unique_core_sections, how='cross')

            # Sequentially merge each measurement DataFrame
            for measurement_type, measurement_df in measurements.items():
                expanded_master_df = pd.merge(expanded_master_df, measurement_df, on=['proposal_id', 'sampling_set', 'core_section'], how='left')

            print("Cleaning up columns...")
            expanded_master_df=expanded_master_df[[c for c in expanded_master_df if not c.endswith('_meta') and not c.endswith('_analysis')]]
            study_cols=[c for c in expanded_master_df if c in df_study]
            study_meta_cols=[d for d in [c for c in expanded_master_df if c in df_study_meta] if d not in study_cols]
            study_order=study_cols+study_meta_cols
            sample_order=[c for c in expanded_master_df if c in df_samples and c not in study_order and c != 'proposal_id']
            analysis_order=[c for c in expanded_master_df if c in df_analysis and c not in study_order+sample_order and c!='proposal_id']
            measurement_order=[c for c in expanded_master_df if c not in study_order+sample_order+analysis_order and c!='proposal_id']
            expanded_master_df=expanded_master_df[study_order+sample_order+analysis_order+measurement_order]
            
            expanded_master_df['latitude']=expanded_master_df['latitude'].astype('float64')
            expanded_master_df['longitude']=expanded_master_df['longitude'].astype('float64')

            # Convert columns to datetime, handling missing values and different formats
            expanded_master_df['started_date'] = pd.to_datetime(expanded_master_df['started_date'], errors='coerce')
            expanded_master_df['closed_date'] = pd.to_datetime(expanded_master_df['closed_date'], errors='coerce')
            expanded_master_df['collection_date'] = pd.to_datetime(expanded_master_df['collection_date'], errors='coerce')

            expanded_master_df['project_members']=expanded_master_df['project_members'].astype(str)
            return expanded_master_df
                
        except Exception as e:
            print(f"Error building unified DataFrame: {str(e)}")
            return pd.DataFrame()
        
    def get_all_data(self,base_url, params={'per_page':20},accessor='data'):
        all_data = []
        page = 1
        while True:
            params['page']=page
            response = requests.get(base_url, params=params,headers={'accept': 'application/json'})
            if response.status_code == 200:
                data = response.json()
                if not data.get(accessor, []):
                    break
                all_data.extend(data[accessor])
                page += 1
                time.sleep(1)  # Respect rate limits
            else:
                break
        return all_data

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
        lat_col = 'latitude'
        lon_col = 'longitude'
        
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