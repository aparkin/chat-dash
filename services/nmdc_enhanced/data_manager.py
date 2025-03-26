"""
Data management for NMDC Enhanced service.

This module handles data loading, caching, and analysis for the NMDC Enhanced service,
implementing an upfront loading pattern similar to the MONet service.
"""

from typing import Dict, Any, Optional, List, Tuple, Union, Set
import pandas as pd
import numpy as np
import requests
import json
import asyncio
import logging
from datetime import datetime, timedelta
from functools import lru_cache
import time
import traceback

# Add imports for file operations
import os
import pathlib
import threading

from .models import NMDCEnhancedConfig

# Configure logging
logger = logging.getLogger(__name__)

class NMDCEnhancedDataManager:
    """Manages data loading and analysis for NMDC Enhanced service."""
    
    def __init__(self, config: NMDCEnhancedConfig, load_data: bool = True):
        """Initialize data manager with configuration.
        
        Args:
            config: Configuration for the NMDC Enhanced service.
            load_data: Whether to load data from the NMDC API.
        """
        self.config = config
        
        # DataFrame storage
        self._studies_df: Optional[pd.DataFrame] = None
        self._biosamples_df: Optional[pd.DataFrame] = None
        self._unified_df: Optional[pd.DataFrame] = None
        
        # Metadata storage
        self._cache_timestamp: Optional[datetime] = None
        self._stats: Optional[Dict[str, Any]] = None
        self._column_descriptions: Optional[Dict[str, str]] = None
        self._geographic_coverage: Optional[Dict[str, Any]] = None
        self._ecosystem_summaries: Optional[Dict[str, Any]] = None
        
        # Loading flag to prevent double initialization
        self._loading_in_progress = False
        
        # Thread safety for cache operations
        self._cache_lock = threading.Lock()
        
        # Cache directory setup
        self._cache_dir = self._get_cache_dir()
        
        if load_data:
            # First try to load from cache
            loaded_from_cache = self._load_from_cache()
            
            if loaded_from_cache and self._has_valid_data():
                logger.info("Successfully loaded data from cache")
                # Start background refresh if cache is valid
                self.start_background_refresh()
            else:
                # If cache load failed or data is invalid, do blocking load
                logger.info("Cache not available or invalid, loading data from API...")
                logger.info("Initializing NMDC Enhanced Data Manager...")
                self._load_all_data()
        
        logger.info("NMDC Enhanced Data Manager initialization complete.")
    
    def _get_cache_dir(self) -> str:
        """Get the cache directory path, creating it if it doesn't exist."""
        # Get the directory of the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create cache directory path
        cache_dir = os.path.join(current_dir, "cache")
        
        # Create the directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        return cache_dir
    
    def _cache_is_valid(self) -> bool:
        """Check if the cache is valid and not expired."""
        # Path to metadata file
        metadata_path = os.path.join(self._cache_dir, "metadata.json")
        
        # Check if metadata file exists
        if not os.path.exists(metadata_path):
            logger.info("Cache metadata file not found")
            return False
        
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Check timestamp
            cache_time = datetime.fromisoformat(metadata.get('timestamp', '2000-01-01'))
            max_age_hours = self.config.cache_expiry_hours
            
            # Calculate expiration
            expiration_time = cache_time + timedelta(hours=max_age_hours)
            now = datetime.now()
            
            # Check if cache has expired
            if now > expiration_time:
                logger.info(f"Cache is older than {max_age_hours} hours (created {cache_time.isoformat()})")
                # Don't return False here - we'll still use it but trigger a refresh
            
            # Check if all required files exist
            required_files = ["studies.parquet", "biosamples.parquet", "unified.parquet"]
            for file in required_files:
                file_path = os.path.join(self._cache_dir, file)
                if not os.path.exists(file_path):
                    logger.info(f"Required cache file missing: {file}")
                    return False
            
            # Validate file sizes as a basic integrity check
            for file in required_files:
                file_path = os.path.join(self._cache_dir, file)
                if os.path.getsize(file_path) < 100:  # Arbitrary minimum size
                    logger.info(f"Cache file too small (possibly corrupted): {file}")
                    return False
            
            # Cache appears valid
            logger.info(f"Found valid cache from {cache_time.isoformat()}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating cache: {str(e)}")
            return False
    
    def _load_from_cache(self) -> bool:
        """Load DataFrames from cache if available and valid.
        
        Returns:
            True if successfully loaded from cache, False otherwise
        """
        # Check if cache is valid
        if not self._cache_is_valid():
            return False
        
        try:
            logger.info("Loading data from cache...")
            
            # Load metadata
            metadata_path = os.path.join(self._cache_dir, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Set cache timestamp
            self._cache_timestamp = datetime.fromisoformat(metadata.get('timestamp', '2000-01-01'))
            
            # Load DataFrames
            studies_path = os.path.join(self._cache_dir, "studies.parquet")
            biosamples_path = os.path.join(self._cache_dir, "biosamples.parquet")
            unified_path = os.path.join(self._cache_dir, "unified.parquet")
            
            self._studies_df = pd.read_parquet(studies_path)
            logger.info(f"Loaded studies DataFrame from cache: {len(self._studies_df)} rows")
            
            self._biosamples_df = pd.read_parquet(biosamples_path)
            logger.info(f"Loaded biosamples DataFrame from cache: {len(self._biosamples_df)} rows")
            
            self._unified_df = pd.read_parquet(unified_path)
            logger.info(f"Loaded unified DataFrame from cache: {len(self._unified_df)} rows")
            
            # Load statistics if available
            stats_path = os.path.join(self._cache_dir, "statistics.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    self._stats = json.load(f)
                logger.info("Loaded statistics from cache")
            else:
                # Calculate statistics from loaded data
                logger.info("Calculating statistics from cached data...")
                self._calculate_all_statistics()
            
            logger.info("Successfully loaded all data from cache")
            return True
            
        except Exception as e:
            logger.error(f"Error loading from cache: {str(e)}")
            logger.error(traceback.format_exc())
            # Reset to None in case of partial loading
            self._studies_df = None
            self._biosamples_df = None
            self._unified_df = None
            self._stats = None
            return False
    
    def _save_to_cache(self) -> bool:
        """Save current DataFrames to cache.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            logger.info("Saving data to cache...")
            
            # Check if we have data to save
            if self._studies_df is None or self._biosamples_df is None or self._unified_df is None:
                logger.warning("Cannot save to cache: One or more DataFrames are None")
                return False
            
            # Clean column names by removing 'annotations.' prefix from all DataFrames
            def clean_columns(df):
                if df is not None:
                    df.columns = [col.replace('annotations.', '') if col.startswith('annotations.') else col 
                                for col in df.columns]
            
            # Clean column names on all DataFrames
            clean_columns(self._studies_df)
            clean_columns(self._biosamples_df)
            clean_columns(self._unified_df)
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'studies_count': len(self._studies_df) if self._studies_df is not None else 0,
                'biosamples_count': len(self._biosamples_df) if self._biosamples_df is not None else 0,
                'unified_count': len(self._unified_df) if self._unified_df is not None else 0,
                'version': '1.0'
            }
            
            # Save DataFrames
            studies_path = os.path.join(self._cache_dir, "studies.parquet")
            biosamples_path = os.path.join(self._cache_dir, "biosamples.parquet")
            unified_path = os.path.join(self._cache_dir, "unified.parquet")
            
            # Save DataFrames
            self._prepare_df_for_parquet(self._studies_df).to_parquet(studies_path, index=False)
            logger.info(f"Saved studies DataFrame to cache: {len(self._studies_df)} rows")
            
            self._prepare_df_for_parquet(self._biosamples_df).to_parquet(biosamples_path, index=False)
            logger.info(f"Saved biosamples DataFrame to cache: {len(self._biosamples_df)} rows")
            
            self._prepare_df_for_parquet(self._unified_df).to_parquet(unified_path, index=False)
            logger.info(f"Saved unified DataFrame to cache: {len(self._unified_df)} rows")
            
            # Save metadata
            metadata_path = os.path.join(self._cache_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save statistics if available
            if self._stats:
                stats_path = os.path.join(self._cache_dir, "statistics.json")
                with open(stats_path, 'w') as f:
                    json.dump(self._stats, f, indent=2)
                logger.info("Saved statistics to cache")
            
            logger.info("Successfully saved all data to cache")
            return True
            
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _prepare_df_for_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Parquet serialization.
        
        Args:
            df: Original DataFrame
            
        Returns:
            DataFrame prepared for Parquet
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # First pass: identify and coerce numeric columns
        for col in df_copy.columns:
            try:
                # Check for numeric suffix patterns and measurement field prefixes
                is_numeric_pattern = any(col.lower().endswith(suffix) for suffix in [
                    '_count', '_value', '_amount', '_concentration', '_ph', '_temperature',
                    '_depth', '_elevation', '_length', '_width', '_height', '_size',
                    '_rate', '_percentage', '_ratio', '_index'
                ]) or any(col.lower().startswith(prefix) for prefix in [
                    'tot_', 'total_', 'avg_', 'mean_', 'min_', 'max_', 'std_',
                    'concentration_', 'temp_', 'temperature_', 'depth_', 'height_',
                    'width_', 'length_', 'volume_', 'mass_', 'weight_', 'density_',
                    'pressure_', 'flow_', 'speed_', 'velocity_', 'acceleration_',
                    'force_', 'energy_', 'power_', 'frequency_', 'wavelength_',
                    'intensity_', 'conductivity_', 'resistivity_', 'capacitance_',
                    'inductance_', 'voltage_', 'current_', 'charge_', 'field_',
                    'flux_', 'luminosity_', 'illuminance_', 'irradiance_',
                    'radiance_', 'exposure_', 'dose_', 'activity_', 'count_',
                    'abundance_', 'biomass_', 'density_', 'richness_', 'diversity_'
                ])
                
                # Try to coerce to numeric if it has a numeric pattern or is already numeric
                if is_numeric_pattern or pd.api.types.is_numeric_dtype(df_copy[col]):
                    logger.info(f"Attempting to coerce column {col} to numeric")
                    # Convert to numeric, coercing errors to NaN
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    
                    # Log success or partial success
                    non_null_count = df_copy[col].notna().sum()
                    total_count = len(df_copy[col])
                    success_rate = (non_null_count / total_count) * 100 if total_count > 0 else 0
                    logger.info(f"Column {col} numeric conversion: {success_rate:.1f}% successful values")
            except Exception as e:
                logger.warning(f"Could not coerce column {col} to numeric: {str(e)}")
        
        # Second pass: handle remaining columns and complex types
        for col in df_copy.columns:
            try:
                # Get a sample of non-null values
                sample_vals = df_copy[col].dropna().head(5)
                
                # Skip if empty
                if sample_vals.empty:
                    continue
                
                # Check if any value is a complex type
                complex_types = [val for val in sample_vals if isinstance(val, (list, dict, pd.DataFrame))]
                
                if complex_types:
                    logger.info(f"Converting complex column for Parquet storage: {col}")
                    # Convert to string representation
                    df_copy[col] = df_copy[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, (list, dict)) 
                                else str(x) if isinstance(x, pd.DataFrame)
                                else x
                    )
                
                # Ensure datetime columns are datetime
                if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
                
                # Convert any remaining object columns to string
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].astype(str)
                    
            except Exception as e:
                logger.warning(f"Error preparing column '{col}' for Parquet: {str(e)}")
                # If we can't handle the column, convert it to string
                df_copy[col] = df_copy[col].astype(str)
        
        return df_copy
    
    def _restore_from_parquet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Restore complex types from Parquet-loaded DataFrame.
        
        Args:
            df: DataFrame loaded from Parquet
            
        Returns:
            DataFrame with complex types restored
        """
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Look for columns that might contain serialized JSON
        for col in df_copy.columns:
            # Check first few non-null values
            sample_vals = df_copy[col].dropna().head(5).tolist()
            
            # Skip if empty
            if not sample_vals:
                continue
                
            # Check if values look like JSON strings
            json_candidates = [
                val for val in sample_vals 
                if isinstance(val, str) and val.startswith('{') and val.endswith('}')
            ]
            
            # If we have potential JSON strings, try to parse them
            if json_candidates and len(json_candidates) / len(sample_vals) > 0.5:  # More than half are JSON-like
                try:
                    logger.info(f"Attempting to restore complex objects in column: {col}")
                    df_copy[col] = df_copy[col].apply(
                        lambda x: json.loads(x) if isinstance(x, str) and x.startswith('{') and x.endswith('}')
                                else x
                    )
                except:
                    # If parsing fails, leave as is
                    logger.warning(f"Failed to parse JSON in column {col}, leaving as strings")
        
        return df_copy
    
    def refresh_data_async(self) -> None:
        """Start an asynchronous refresh of the data from API."""
        # Create a thread to refresh data
        thread = threading.Thread(target=self._refresh_data_thread)
        thread.daemon = True  # Thread will exit when main program exits
        thread.start()
        logger.info("Started asynchronous data refresh thread")
    
    def _refresh_data_thread(self) -> None:
        """Background thread function to refresh data."""
        logger.info("Background data refresh started")
        
        try:
            # Create temporary DataFrames
            temp_studies_df = self._load_studies()
            temp_biosamples_df = self._load_biosamples()
            
            # Build unified DataFrame
            temp_unified_df = None
            if temp_studies_df is not None and temp_biosamples_df is not None:
                # Create copies to avoid modifying originals
                studies_copy = temp_studies_df.copy()
                biosamples_copy = temp_biosamples_df.copy()
                
                # Process environmental columns and merge
                try:
                    logger.info("Building unified DataFrame in background refresh")
                    # This code replicates the core of _build_unified_dataframe
                    # Process environmental columns
                    env_columns = ['env_broad_scale', 'env_local_scale', 'env_medium']
                    for env_col in env_columns:
                        if env_col in biosamples_copy.columns:
                            col_id = f"{env_col}_id"
                            col_label = f"{env_col}_label"
                            col_url = f"{env_col}_url"
                            
                            biosamples_copy[col_id] = biosamples_copy[env_col].apply(
                                lambda x: x.get('id') if isinstance(x, dict) and x is not None else None
                            )
                            biosamples_copy[col_label] = biosamples_copy[env_col].apply(
                                lambda x: x.get('label') if isinstance(x, dict) and x is not None else None
                            )
                            biosamples_copy[col_url] = biosamples_copy[env_col].apply(
                                lambda x: x.get('url') if isinstance(x, dict) and x is not None else None
                            )
                    
                    # Handle principal_investigator
                    if 'principal_investigator' in studies_copy.columns:
                        studies_copy['principal_investigator_name'] = studies_copy['principal_investigator'].apply(
                            lambda x: x.get('name') if isinstance(x, dict) and x is not None else None
                        )
                    
                    # Rename study columns to avoid conflicts
                    study_cols = [col for col in studies_copy.columns if col != 'id']
                    studies_rename = {col: f"study_{col}" for col in study_cols}
                    studies_copy = studies_copy.rename(columns=studies_rename)
                    
                    # Rename id column for merging
                    studies_copy = studies_copy.rename(columns={'id': 'study_id'})
                    
                    # Merge DataFrames
                    if 'study_id' in biosamples_copy.columns:
                        temp_unified_df = biosamples_copy.merge(
                            studies_copy, 
                            on='study_id', 
                            how='left',
                            suffixes=('', '_study')
                        )
                        logger.info(f"Unified DataFrame created in background with {len(temp_unified_df)} rows")
                    else:
                        logger.error("Cannot merge DataFrames: study_id not found in biosamples")
                        temp_unified_df = biosamples_copy.copy()
                except Exception as e:
                    logger.error(f"Error building unified DataFrame in background: {str(e)}")
                    logger.error(traceback.format_exc())
                    temp_unified_df = biosamples_copy.copy()
            
            # If we have all the data, update the class variables with a lock
            if temp_studies_df is not None and temp_biosamples_df is not None and temp_unified_df is not None:
                logger.info("Updating DataFrames with fresh data from background thread")
                # Update the main DataFrames
                self._studies_df = temp_studies_df
                self._biosamples_df = temp_biosamples_df
                self._unified_df = temp_unified_df
                
                # Update timestamp
                self._cache_timestamp = datetime.now()
                
                # Recalculate statistics
                self._calculate_all_statistics()
                
                # Save to cache
                self._save_to_cache()
                
                logger.info("Background data refresh completed successfully")
            else:
                logger.error("Background refresh failed: One or more DataFrames are None")
                
        except Exception as e:
            logger.error(f"Error in background data refresh: {str(e)}")
            logger.error(traceback.format_exc())
            
        logger.info("Background data refresh thread finished")
    
    def _load_all_data(self) -> None:
        """Load all data from NMDC API endpoints."""
        # Prevent double initialization
        if self._loading_in_progress:
            logger.info("Data loading already in progress, skipping duplicate load")
            return
        
        self._loading_in_progress = True
        
        try:
            # Load studies first
            logger.info("Loading studies data...")
            self._studies_df = self._load_studies()
            if self._studies_df is not None:
                # Extract fields from studies
                self._studies_df = self._extract_study_fields(self._studies_df)
                logger.info(f"Successfully loaded {len(self._studies_df)} studies")
            else:
                logger.warning("Failed to load studies data, using empty DataFrame")
                self._studies_df = pd.DataFrame()
            
            # Then load biosamples
            logger.info("Loading biosamples data...")
            self._biosamples_df = self._load_biosamples()
            if self._biosamples_df is not None:
                # Extract fields from biosamples
                self._biosamples_df = self._extract_biosample_fields(self._biosamples_df)
                logger.info(f"Successfully loaded {len(self._biosamples_df)} biosamples")
            else:
                logger.warning("Failed to load biosamples data, using empty DataFrame")
                self._biosamples_df = pd.DataFrame()
            
            # Create unified dataframe
            logger.info("Creating unified dataframe...")
            self._unified_df = self._build_unified_dataframe()  # Store the result
            if self._unified_df is not None:
                logger.info(f"Unified dataframe created with {len(self._unified_df)} rows")
            else:
                logger.warning("Failed to build unified DataFrame, using empty DataFrame")
                self._unified_df = pd.DataFrame()
            
            # Calculate statistics
            self._calculate_all_statistics()
            
            # Set cache timestamp
            self._cache_timestamp = datetime.now()
            
            # Save data to cache after successful load
            logger.info("Saving data to cache...")
            if self._save_to_cache():
                logger.info("Successfully saved all data to cache")
            else:
                logger.warning("Failed to save data to cache")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            logger.error(traceback.format_exc())
            # Initialize empty dataframes to avoid NoneType errors
            self._studies_df = pd.DataFrame()
            self._biosamples_df = pd.DataFrame()
            self._unified_df = pd.DataFrame()
        finally:
            # Reset loading flag when done, whether successful or not
            self._loading_in_progress = False
    
    def _load_studies(self) -> pd.DataFrame:
        """Load all studies from NMDC API.
        
        Returns:
            DataFrame containing study data
        """
        results = []
        offset = 0
        limit = 500  # Start with 500, will reduce dynamically if needed
        min_batch_size = 50  # Don't go below this batch size
        
        logger.info("Fetching studies from NMDC API...")
        
        # Build request body - directly matching the curl example
        request_body = {
            "conditions": [],
            "data_object_filter": []
        }
        
        # Try directly with the known working endpoint
        url = f"{self.config.api_base_url}/study/search"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        
        try:
            # Start paginating immediately without trying to get total count
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            while True:  # We'll break when we get no more results
                logger.info(f"Fetching studies batch at offset {offset}, batch size {limit}")
                params = {"offset": offset, "limit": limit}
                
                try:
                    response = requests.post(
                        url,
                        params=params,
                        headers=headers,
                        json=request_body,
                        timeout=60  # Maintain 60 second timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract results based on response format
                    if isinstance(data, dict) and "results" in data:
                        batch_results = data.get("results", [])
                    elif isinstance(data, list):
                        batch_results = data
                    else:
                        logger.warning(f"Unexpected response format at offset {offset}")
                        batch_results = []
                    
                    if not batch_results:
                        logger.info(f"No more results at offset {offset}")
                        break
                    
                    # Add results to our collection
                    results.extend(batch_results)
                    logger.info(f"Fetched {len(batch_results)} studies at offset {offset}, total collected: {len(results)}")
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Move to next page
                    offset += limit
                    
                except requests.exceptions.HTTPError as e:
                    consecutive_failures += 1
                    status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else 'unknown'
                    
                    if status_code == 502 or status_code == 504:  # Bad Gateway or Gateway Timeout
                        logger.warning(f"Server error (status {status_code}) at offset {offset} with batch size {limit}")
                        
                        # Reduce batch size if possible
                        if limit > min_batch_size:
                            old_limit = limit
                            limit = max(min_batch_size, limit // 2)
                            logger.info(f"Reducing batch size from {old_limit} to {limit} and retrying")
                            # Don't increment offset - we'll retry the same batch with a smaller limit
                            
                            # Add backoff delay proportional to failure count
                            backoff_time = 1 * (2 ** (consecutive_failures - 1))  # 1, 2, 4, 8... seconds
                            logger.info(f"Backing off for {backoff_time} seconds before retry")
                            time.sleep(backoff_time)
                            
                            # Reset consecutive failures if we're adjusting strategy
                            if consecutive_failures >= max_consecutive_failures:
                                consecutive_failures = 0
                                
                            continue  # Try again with reduced batch size
                        else:
                            logger.error(f"Server error at minimum batch size ({min_batch_size}), skipping offset {offset}")
                            # Skip this problematic section by moving the offset forward
                            offset += min_batch_size
                    
                    # For other HTTP errors, log and continue
                    logger.error(f"HTTP error at offset {offset}: {str(e)}")
                    
                    # If we've had too many consecutive failures, move forward to avoid getting stuck
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures, skipping ahead from offset {offset}")
                        offset += limit
                        consecutive_failures = 0
                    
                except Exception as e:
                    logger.error(f"Error fetching studies at offset {offset}: {str(e)}")
                    consecutive_failures += 1
                    
                    # If we've had too many consecutive failures, move forward to avoid getting stuck
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures, skipping ahead from offset {offset}")
                        offset += limit
                        consecutive_failures = 0
                    
                    # Add exponential backoff delay
                    backoff_time = 1 * (2 ** (consecutive_failures - 1))
                    logger.info(f"Backing off for {backoff_time} seconds before retry")
                    time.sleep(backoff_time)
                
                # Small delay between successful requests to avoid overwhelming the API
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in studies loading process: {str(e)}")
            logger.error(traceback.format_exc())
        
        if not results:
            logger.warning("No studies were fetched, returning empty DataFrame")
            return pd.DataFrame()
        
        # Convert to DataFrame
        logger.info(f"Converting {len(results)} studies to DataFrame")
        df = pd.json_normalize(results)
        
        # Clean up DataFrame
        df = self._clean_dataframe(df)
        return df
    
    def _load_biosamples(self) -> pd.DataFrame:
        """Load all biosamples from NMDC API.
        
        Returns:
            DataFrame containing biosample data
        """
        results = []
        offset = 0
        limit = 500  # Start with 500, will reduce dynamically if needed
        min_batch_size = 50  # Don't go below this batch size
        
        logger.info("Fetching biosamples from NMDC API...")
        
        # Build request body - directly matching the curl example
        request_body = {
            "conditions": [],
            "data_object_filter": []
        }
        
        # Try directly with the known working endpoint
        url = f"{self.config.api_base_url}/biosample/search"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        
        try:
            # Start paginating immediately without trying to get total count
            consecutive_failures = 0
            max_consecutive_failures = 3
            
            while True:  # We'll break when we get no more results
                logger.info(f"Fetching biosamples batch at offset {offset}, batch size {limit}")
                params = {"offset": offset, "limit": limit}
                
                try:
                    response = requests.post(
                        url,
                        params=params,
                        headers=headers,
                        json=request_body,
                        timeout=60  # Maintain 60 second timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    # Extract results based on response format
                    if isinstance(data, dict) and "results" in data:
                        batch_results = data.get("results", [])
                    elif isinstance(data, list):
                        batch_results = data
                    else:
                        logger.warning(f"Unexpected response format at offset {offset}")
                        batch_results = []
                    
                    if not batch_results:
                        logger.info(f"No more results at offset {offset}")
                        break
                    
                    # Add results to our collection
                    results.extend(batch_results)
                    logger.info(f"Fetched {len(batch_results)} biosamples at offset {offset}, total collected: {len(results)}")
                    
                    # Reset failure counter on success
                    consecutive_failures = 0
                    
                    # Move to next page
                    offset += limit
                    
                except requests.exceptions.HTTPError as e:
                    consecutive_failures += 1
                    status_code = e.response.status_code if hasattr(e, 'response') and hasattr(e.response, 'status_code') else 'unknown'
                    
                    if status_code == 502 or status_code == 504:  # Bad Gateway or Gateway Timeout
                        logger.warning(f"Server error (status {status_code}) at offset {offset} with batch size {limit}")
                        
                        # Reduce batch size if possible
                        if limit > min_batch_size:
                            old_limit = limit
                            limit = max(min_batch_size, limit // 2)
                            logger.info(f"Reducing batch size from {old_limit} to {limit} and retrying")
                            # Don't increment offset - we'll retry the same batch with a smaller limit
                            
                            # Add backoff delay proportional to failure count
                            backoff_time = 1 * (2 ** (consecutive_failures - 1))  # 1, 2, 4, 8... seconds
                            logger.info(f"Backing off for {backoff_time} seconds before retry")
                            time.sleep(backoff_time)
                            
                            # Reset consecutive failures if we're adjusting strategy
                            if consecutive_failures >= max_consecutive_failures:
                                consecutive_failures = 0
                                
                            continue  # Try again with reduced batch size
                        else:
                            logger.error(f"Server error at minimum batch size ({min_batch_size}), skipping offset {offset}")
                            # Skip this problematic section by moving the offset forward
                            offset += min_batch_size
                    
                    # For other HTTP errors, log and continue
                    logger.error(f"HTTP error at offset {offset}: {str(e)}")
                    
                    # If we've had too many consecutive failures, move forward to avoid getting stuck
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures, skipping ahead from offset {offset}")
                        offset += limit
                        consecutive_failures = 0
                    
                except Exception as e:
                    logger.error(f"Error fetching biosamples at offset {offset}: {str(e)}")
                    consecutive_failures += 1
                    
                    # If we've had too many consecutive failures, move forward to avoid getting stuck
                    if consecutive_failures >= max_consecutive_failures:
                        logger.warning(f"Too many consecutive failures, skipping ahead from offset {offset}")
                        offset += limit
                        consecutive_failures = 0
                    
                    # Add exponential backoff delay
                    backoff_time = 1 * (2 ** (consecutive_failures - 1))
                    logger.info(f"Backing off for {backoff_time} seconds before retry")
                    time.sleep(backoff_time)
                
                # Small delay between successful requests to avoid overwhelming the API
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in biosamples loading process: {str(e)}")
            logger.error(traceback.format_exc())
        
        if not results:
            logger.warning("No biosamples were fetched, returning empty DataFrame")
            return pd.DataFrame()
        
        # Convert to DataFrame
        logger.info(f"Converting {len(results)} biosamples to DataFrame")
        df = pd.json_normalize(results)
        
        # Clean up DataFrame
        df = self._clean_dataframe(df)
        return df
    
    def _build_unified_dataframe(self) -> pd.DataFrame:
        """Build a unified DataFrame from studies and biosamples.
        
        Returns:
            Unified DataFrame with studies and biosamples data
        """
        if self._studies_df is None or self._biosamples_df is None:
            logger.error("Cannot build unified DataFrame: studies or biosamples data not loaded")
            return pd.DataFrame()
        
        try:
            logger.info("Building unified DataFrame...")
            
            # Create copies to avoid modifying originals
            studies_df = self._studies_df.copy()
            biosamples_df = self._biosamples_df.copy()
            
            # Check if required columns exist
            if 'id' not in studies_df.columns:
                logger.error("Cannot build unified DataFrame: 'id' column missing from studies")
                return pd.DataFrame()
            
            if 'study_id' not in biosamples_df.columns:
                logger.error("Cannot build unified DataFrame: 'study_id' column missing from biosamples")
                return pd.DataFrame()
            
            # Rename all study columns to prevent conflicts with biosamples columns
            studies_cols = studies_df.columns.tolist()
            rename_map = {}
            
            # Rename all study columns except 'id' which will be used for merging
            for col in studies_cols:
                if col != 'id':
                    new_name = f"study_{col}"
                    rename_map[col] = new_name
                    logger.info(f"Renaming study column '{col}' to '{new_name}' to avoid conflicts")
            
            if rename_map:
                studies_df = studies_df.rename(columns=rename_map)
            
            # Rename the studies id column to match biosamples study_id for merging
            studies_df = studies_df.rename(columns={'id': 'study_id'})
            
            # Perform an outer merge to keep all biosamples and studies
            merged_df = pd.merge(
                biosamples_df, 
                studies_df, 
                on='study_id', 
                how='left',
                suffixes=('', '_study')  # This will handle any remaining conflicts
            )
            
            # Log merge results
            logger.info(f"Merged {len(biosamples_df)} biosamples with {len(studies_df)} studies into {len(merged_df)} records")
            
            # Clean up any problematic columns
            for col in merged_df.columns:
                try:
                    # Convert any DataFrame objects to strings
                    if merged_df[col].apply(lambda x: isinstance(x, pd.DataFrame)).any():
                        logger.warning(f"Converting DataFrame objects in column '{col}' to strings")
                        merged_df[col] = merged_df[col].apply(lambda x: str(x) if isinstance(x, pd.DataFrame) else x)
                    
                    # Convert any lists to strings
                    if merged_df[col].apply(lambda x: isinstance(x, list)).any():
                        logger.warning(f"Converting lists in column '{col}' to strings")
                        merged_df[col] = merged_df[col].apply(lambda x: ', '.join(str(item) for item in x) if isinstance(x, list) else x)
                    
                    # Convert any dictionaries to strings
                    if merged_df[col].apply(lambda x: isinstance(x, dict)).any():
                        logger.warning(f"Converting dictionaries in column '{col}' to strings")
                        merged_df[col] = merged_df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
                    
                    # Handle numeric columns, including quantity fields
                    if col.endswith('_unit'):
                        # Skip unit columns - keep as strings
                        continue
                        
                    # Try to convert to numeric if the column appears to contain numbers
                    sample = merged_df[col].dropna().iloc[0] if not merged_df[col].empty else None
                    if sample is not None and isinstance(sample, (int, float, str)):
                        try:
                            numeric_col = pd.to_numeric(merged_df[col], errors='coerce')
                            # Only update if we successfully converted some values
                            if not numeric_col.isna().all():
                                merged_df[col] = numeric_col
                                logger.info(f"Converted column '{col}' to numeric")
                        except:
                            pass
                        
                except Exception as e:
                    logger.warning(f"Error processing column '{col}': {str(e)}")
                    # If we can't process the column, leave it as is
                    continue
            
            logger.info(f"Built unified DataFrame with {len(merged_df)} rows and {len(merged_df.columns)} columns")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error building unified DataFrame: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _extract_biosample_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant fields from biosample data according to specifications.
        
        Args:
            df: Biosample DataFrame with raw JSON data
            
        Returns:
            Processed DataFrame with extracted fields
        """
        # Remove omics_processing
        if 'omics_processing' in df.columns:
            df = df.drop(columns=['omics_processing'])
        
        # Process alternate_identifiers (if exists)
        if 'alternate_identifiers' in df.columns:
            df['alternate_identifiers'] = df['alternate_identifiers'].apply(
                lambda x: ', '.join(x) if isinstance(x, list) else x
            )
        
        # Define measurement fields that should be numeric
        measurement_fields = {
            'tot_carb', 'tot_org_carb', 'tot_nitro', 'tot_phosp', 'depth',
            'elevation', 'temperature', 'ph', 'pressure', 'humidity',
            'wind_speed', 'precipitation', 'salinity'
        }
        
        # Process annotations dictionary
        if 'annotations' in df.columns:
            # Extract all fields from annotations
            for row_idx, row in df.iterrows():
                annotations = row.get('annotations', {})
                if isinstance(annotations, dict):
                    for key, value in annotations.items():
                        # Handle structured quantity fields
                        if isinstance(value, dict) and value.get('type') == 'nmdc:QuantityValue':
                            # Extract numeric value
                            numeric_value = value.get('has_numeric_value')
                            if numeric_value is not None:
                                # Create column if it doesn't exist
                                if key not in df.columns:
                                    df[key] = None
                                # Convert to numeric if it's a measurement field
                                if any(measure in key.lower() for measure in measurement_fields):
                                    try:
                                        numeric_value = pd.to_numeric(numeric_value)
                                    except:
                                        logger.warning(f"Could not convert {key} value to numeric: {numeric_value}")
                                df.at[row_idx, key] = numeric_value
                                
                                # Store unit in separate column
                                unit = value.get('has_unit')
                                if unit is not None:
                                    unit_col = f"{key}_unit"
                                    if unit_col not in df.columns:
                                        df[unit_col] = None
                                    df.at[row_idx, unit_col] = unit
                            else:
                                # For non-structured fields
                                if key not in df.columns:
                                    df[key] = None
                                # Try numeric conversion for measurement fields
                                if any(measure in key.lower() for measure in measurement_fields):
                                    try:
                                        if isinstance(value, str):
                                            value = pd.to_numeric(value)
                                    except:
                                        logger.warning(f"Could not convert {key} value to numeric: {value}")
                                df.at[row_idx, key] = value
        
            # Drop the original annotations column
            df = df.drop(columns=['annotations'])
        
        # Process env fields that contain dictionaries
        env_fields = ['env_broad_scale', 'env_local_scale', 'env_medium']
        for env_field in env_fields:
            if env_field in df.columns:
                # Extract id and label from env dictionaries
                id_col = f"{env_field}_id"
                label_col = f"{env_field}_label"
                url_col = f"{env_field}_url"
                
                if id_col not in df.columns:
                    df[id_col] = None
                if label_col not in df.columns:
                    df[label_col] = None
                if url_col not in df.columns:
                    df[url_col] = None
                    
                for row_idx, row in df.iterrows():
                    env_data = row.get(env_field, {})
                    if isinstance(env_data, dict):
                        if 'id' in env_data:
                            df.at[row_idx, id_col] = env_data['id']
                        if 'label' in env_data:
                            df.at[row_idx, label_col] = env_data['label']
                        if 'url' in env_data:
                            df.at[row_idx, url_col] = env_data['url']
                
                # Drop the original env field column
                df = df.drop(columns=[env_field])
        
        # Process any remaining dictionary columns
        for col in df.columns:
            sample_val = None
            # Find first non-null value to check type
            for val in df[col].dropna():
                sample_val = val
                break
            
            if isinstance(sample_val, dict):
                # Flatten this dictionary into additional columns
                for key in sample_val.keys():
                    new_col = f"{col}_{key}"
                    df[new_col] = df[col].apply(
                        lambda x: x.get(key) if isinstance(x, dict) else None
                    )
                # Drop the original column
                df = df.drop(columns=[col])
        
        # Process lists and convert to strings
        for col in df.columns:
            sample_val = None
            # Find first non-null value to check type
            for val in df[col].dropna():
                sample_val = val
                break
            
            if isinstance(sample_val, list):
                df[col] = df[col].apply(
                    lambda x: ', '.join(str(item) for item in x) if isinstance(x, list) else x
                )
        
        return df
    
    def _extract_study_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract relevant fields from study data according to specifications.
        
        Args:
            df: Study DataFrame with raw JSON data
            
        Returns:
            Processed DataFrame with extracted fields
        """
        # First, remove all columns starting with doi_map
        df = df.loc[:, ~df.columns.str.startswith('doi_map')]
        
        # Remove omics_processing_counts
        if 'omics_processing_counts' in df.columns:
            df = df.drop(columns=['omics_processing_counts'])
        
        # Process omics_counts
        if 'omics_counts' in df.columns:
            # First identify all types
            omics_types = set()
            for row_idx, row in df.iterrows():
                counts_list = row.get('omics_counts', [])
                if isinstance(counts_list, list):
                    for count_item in counts_list:
                        if isinstance(count_item, dict) and 'type' in count_item:
                            omics_types.add(count_item['type'])
            
            # Initialize columns for each type with 0
            for omics_type in omics_types:
                type_col = f"omics_{omics_type}_count"
                df[type_col] = 0
            
            # Update counts
            for row_idx, row in df.iterrows():
                counts_list = row.get('omics_counts', [])
                if isinstance(counts_list, list):
                    for count_item in counts_list:
                        if isinstance(count_item, dict) and 'type' in count_item and 'count' in count_item:
                            omics_type = count_item['type']
                            count = count_item['count']
                            type_col = f"omics_{omics_type}_count"
                            df.at[row_idx, type_col] = count
            
            # Drop the original omics_counts column
            df = df.drop(columns=['omics_counts'])
        
        # Process annotations dictionary
        if 'annotations' in df.columns:
            # Extract fields from annotations
            annotation_fields = [
                'type', 'title', 'jgi_portal_study_identifiers', 
                'pricipal_investigator_image_url', 'ecosystem',
                'ecosystem_type', 'ecosystem_subtype', 'ecosystem_category',
                'specific_ecosystem', 'gnps_task_identifiers', 'notes',
                'insdc_bioproject_identifiers'
            ]
            
            for row_idx, row in df.iterrows():
                annotations = row.get('annotations', {})
                if isinstance(annotations, dict):
                    for field in annotation_fields:
                        if field in annotations:
                            # Create column if it doesn't exist, using the field directly (no prefix)
                            if field not in df.columns:
                                df[field] = None
                            df.at[row_idx, field] = annotations[field]
            
            # Drop the original annotations column
            df = df.drop(columns=['annotations'])
        
        # Process principal_investigator dictionary
        if 'principal_investigator' in df.columns:
            pi_fields = ['name', 'email', 'orcid', 'profile_image_url']
            
            for row_idx, row in df.iterrows():
                pi = row.get('principal_investigator', {})
                if isinstance(pi, dict):
                    for field in pi_fields:
                        if field in pi:
                            # Create column if it doesn't exist
                            if field not in df.columns:
                                df[field] = None
                            df.at[row_idx, field] = pi[field]
            
            # Drop the original principal_investigator column
            df = df.drop(columns=['principal_investigator'])
        
        # Process lists and convert to strings
        for col in df.columns:
            sample_val = None
            # Find first non-null value to check type
            for val in df[col].dropna():
                sample_val = val
                break
            
            if isinstance(sample_val, list):
                df[col] = df[col].apply(
                    lambda x: ', '.join(str(item) for item in x) if isinstance(x, list) else x
                )
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame by:
        - Removing duplicates
        - Converting dates to datetime
        - Handling nested structures
        - Coercing numeric columns
        """
        logger.info(f"Cleaning DataFrame with shape {df.shape}")
        
        # First pass: identify and coerce numeric columns
        numeric_suffixes = [
            '_count', '_value', '_amount', '_concentration', '_ph', '_temperature',
            '_depth', '_elevation', '_length', '_width', '_height', '_size',
            '_rate', '_percentage', '_ratio', '_index'
        ]
        
        for col in df.columns:
            try:
                # Check for numeric suffix patterns
                is_numeric_suffix = any(col.lower().endswith(suffix) for suffix in numeric_suffixes)
                
                # Try to coerce to numeric if it has a numeric suffix or is already numeric
                if is_numeric_suffix or pd.api.types.is_numeric_dtype(df[col]):
                    logger.info(f"Attempting to coerce column {col} to numeric")
                    # Convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Log success or partial success
                    non_null_count = df[col].notna().sum()
                    total_count = len(df[col])
                    success_rate = (non_null_count / total_count) * 100 if total_count > 0 else 0
                    logger.info(f"Column {col} numeric conversion: {success_rate:.1f}% successful values")
            except Exception as e:
                logger.warning(f"Could not coerce column {col} to numeric: {str(e)}")
        
        # Special handling for environmental term columns that are dictionaries
        env_cols = [col for col in df.columns if col.startswith('env_') and col.endswith(('_id', '_label', '_url'))]
        if env_cols:
            logger.info(f"Found {len(env_cols)} environment columns to verify: {env_cols}")
            
            for col in env_cols:
                # Check if any values are still dictionaries
                sample = df[col].dropna().head(1).iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, dict):
                    logger.warning(f"Column {col} still contains dictionary values, converting to strings")
                    # Convert dictionaries to string values
                    df[col] = df[col].apply(lambda x: x.get('id' if col.endswith('_id') else 
                                                 'label' if col.endswith('_label') else 
                                                 'url') if isinstance(x, dict) else x)
        
        # Handle principal investigator names
        pi_cols = [col for col in df.columns if 'principal_investigator_name' in col]
        for col in pi_cols:
            sample = df[col].dropna().head(1).iloc[0] if not df[col].dropna().empty else None
            if isinstance(sample, dict):
                logger.warning(f"Column {col} contains dictionary values, extracting name")
                df[col] = df[col].apply(lambda x: x.get('name') if isinstance(x, dict) else x)
        
        return df
    
    def _clean_column_name(self, column_name: str) -> str:
        """Clean column name for consistency.
        
        Args:
            column_name: Original column name
            
        Returns:
            Cleaned column name
        """
        # Replace dots with underscores
        cleaned = column_name.replace('.', '_')
        # Remove any invalid characters
        cleaned = ''.join(c if c.isalnum() or c == '_' else '_' for c in cleaned)
        return cleaned
    
    def _calculate_all_statistics(self) -> None:
        """Calculate statistics for all dataframes."""
        # Initialize column descriptions
        self._column_descriptions = self._get_base_column_descriptions()
        
        # Calculate statistics for unified DataFrame
        if self._unified_df is not None and not self._unified_df.empty:
            self._calculate_unified_statistics()
            # Calculate ecosystem summaries
            self._calculate_ecosystem_summaries()
        else:
            logger.warning("No unified data available for statistics calculation")
            # Initialize empty statistics
            self._stats = {
                'total_rows': 0,
                'total_columns': 0,
                'numeric_columns': {},
                'categorical_columns': {},
                'temporal_columns': {}
            }
            self._geographic_coverage = None
            self._ecosystem_summaries = {}
    
    def _calculate_unified_statistics(self) -> None:
        """Calculate statistics for the unified DataFrame."""
        df = self._unified_df
        
        # Calculate basic statistics
        self._stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': {},
            'categorical_columns': {},
            'temporal_columns': {}
        }
        
        # List of problematic columns we know about and need special handling
        known_problematic_cols = [
            'env_broad_scale_id', 'env_local_scale_id', 'env_medium_id',
            'study_principal_investigator_name', 'principal_investigator_name'
        ]
        
        logger.info(f"Special handling for known problematic columns: {known_problematic_cols}")
        
        # First, ensure problematic columns are correctly formatted
        for col in known_problematic_cols:
            if col in df.columns:
                try:
                    # Get a sample to check type
                    sample_series = df[col].dropna().head(1)
                    if sample_series.empty:
                        logger.info(f"Column {col} is empty or all null - skipping special handling")
                        continue
                    
                    sample = sample_series.iloc[0]
                    
                    # Check if it needs conversion
                    if isinstance(sample, (dict, pd.DataFrame)):
                        logger.warning(f"Column {col} contains complex objects - converting to strings")
                        
                        if isinstance(sample, dict):
                            # Extract the appropriate value based on column name
                            if col.endswith('_id'):
                                df[col] = df[col].apply(lambda x: x.get('id') if isinstance(x, dict) and x is not None else x)
                            elif col.endswith('_name'):
                                df[col] = df[col].apply(lambda x: x.get('name') if isinstance(x, dict) and x is not None else x)
                            elif col.endswith('_label'):
                                df[col] = df[col].apply(lambda x: x.get('label') if isinstance(x, dict) and x is not None else x)
                            else:
                                # Default to string representation
                                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
                        
                        elif isinstance(sample, pd.DataFrame):
                            # Convert DataFrames to string representation
                            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, pd.DataFrame) else x)
                    
                    # Double-check the conversion worked
                    sample_after = df[col].dropna().head(1).iloc[0] if not df[col].dropna().empty else None
                    logger.info(f"Column {col} after conversion: sample value type is {type(sample_after).__name__}")
                    
                except Exception as e:
                    logger.error(f"Error handling known problematic column {col}: {str(e)}")
        
        # Now process each column for statistics
        for col in df.columns:
            try:
                # Skip empty columns
                if df[col].empty or df[col].isnull().all():
                    logger.info(f"Skipping empty column '{col}'")
                    continue
                
                # Special handling for problematic columns to avoid truth value ambiguity
                if col in known_problematic_cols:
                    logger.info(f"Using special statistics calculation for column '{col}'")
                    
                    # Get non-null count safely
                    non_null_count = int(df[col].notna().sum())
                    null_count = int(df[col].isna().sum())
                    
                    # For these columns, we only store basic information
                    self._stats['categorical_columns'][col] = {
                        'non_null_count': non_null_count,
                        'null_count': null_count,
                        'special_handling': True
                    }
                    
                    # Try to get unique counts if possible
                    try:
                        unique_count = int(df[col].nunique())
                        self._stats['categorical_columns'][col]['unique_values'] = unique_count
                    except Exception as e:
                        logger.warning(f"Could not calculate unique values for {col}: {str(e)}")
                    
                    continue
                
                # Get the first non-null value to determine type
                sample_value = None
                sample_series = df[col].dropna().head(1)
                if not sample_series.empty:
                    sample_value = sample_series.iloc[0]
                
                # Skip if no sample value available
                if sample_value is None:
                    logger.info(f"Skipping column '{col}' with no non-null values")
                    continue
                
                # Verify sample value is not a complex type
                if isinstance(sample_value, (pd.DataFrame, dict, list)):
                    logger.warning(f"Column '{col}' contains complex type {type(sample_value).__name__}")
                    # Store basic info only
                    self._stats['categorical_columns'][col] = {
                        'complex_type': type(sample_value).__name__,
                        'non_null_count': int(df[col].notna().sum()),
                        'null_count': int(df[col].isna().sum())
                    }
                    continue
                
                # For numeric columns (not boolean)
                if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                    try:
                        # Calculate standard statistics safely
                        stats = df[col].describe()
                        self._stats['numeric_columns'][col] = {
                            'mean': float(stats['mean']) if not pd.isna(stats['mean']) else 0,
                            'std': float(stats['std']) if not pd.isna(stats['std']) else 0,
                            'min': float(stats['min']) if not pd.isna(stats['min']) else 0,
                            'max': float(stats['max']) if not pd.isna(stats['max']) else 0,
                            'non_null_count': int(stats['count']),
                            'null_count': int(len(df) - stats['count'])
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating numeric statistics for '{col}': {str(e)}")
                        # Fallback to categorical stats
                        self._stats['categorical_columns'][col] = {
                            'error': str(e),
                            'non_null_count': int(df[col].notna().sum()),
                            'null_count': int(df[col].isna().sum())
                        }
                # For datetime columns
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    try:
                        min_val = df[col].min()
                        max_val = df[col].max()
                        self._stats['temporal_columns'][col] = {
                            'min': min_val.isoformat() if not pd.isna(min_val) else None,
                            'max': max_val.isoformat() if not pd.isna(max_val) else None,
                            'non_null_count': int(df[col].notna().sum()),
                            'null_count': int(df[col].isna().sum())
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating temporal statistics for '{col}': {str(e)}")
                        self._stats['categorical_columns'][col] = {
                            'error': str(e),
                            'non_null_count': int(df[col].notna().sum()),
                            'null_count': int(df[col].isna().sum())
                        }
                # For categorical or string columns
                else:
                    try:
                        non_null_count = int(df[col].notna().sum())
                        
                        # Try to calculate unique counts safely
                        try:
                            unique_count = int(df[col].nunique())
                        except (TypeError, ValueError):
                            logger.warning(f"Cannot calculate unique values for column '{col}' - using estimate")
                            # Estimate unique count by sampling
                            sample_size = min(1000, len(df))
                            if len(df) > 0:
                                sample_unique = df[col].sample(sample_size).nunique() 
                                unique_count = int(sample_unique)
                            else:
                                unique_count = 0
                        
                        # Only try to get value counts for columns with reasonable cardinality
                        top_values = {}
                        if unique_count > 0 and unique_count < 1000:
                            try:
                                # Only get top 5 values to avoid excessive memory usage
                                value_counts = df[col].value_counts().head(5)
                                top_values = {str(k): int(v) for k, v in value_counts.items()}
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Cannot calculate value counts for column '{col}': {str(e)}")
                        
                        self._stats['categorical_columns'][col] = {
                            'unique_values': unique_count,
                            'top_values': top_values,
                            'non_null_count': non_null_count,
                            'null_count': int(len(df) - non_null_count)
                        }
                    except Exception as e:
                        logger.warning(f"Error calculating categorical statistics for '{col}': {str(e)}")
                        self._stats['categorical_columns'][col] = {
                            'error': str(e),
                            'non_null_count': int(df[col].notna().sum()) if hasattr(df[col], 'notna') else 0,
                            'null_count': int(df[col].isna().sum()) if hasattr(df[col], 'isna') else 0
                        }
                
            except Exception as e:
                logger.error(f"Error calculating statistics for column {col}: {str(e)}")
                # Include the column with basic error info
                self._stats['categorical_columns'][col] = {
                    'error': str(e),
                    'non_null_count': int(df[col].notna().sum()) if hasattr(df[col], 'notna') else 0,
                    'null_count': int(df[col].isna().sum()) if hasattr(df[col], 'isna') else 0
                }
        
        # Calculate geographic coverage if lat/lon columns exist
        self._calculate_geographic_coverage(df)
    
    def _calculate_geographic_coverage(self, df: pd.DataFrame) -> None:
        """Calculate geographic coverage for the DataFrame."""
        # Find latitude/longitude columns
        lat_cols = [col for col in df.columns if 'latitude' in col.lower()]
        lon_cols = [col for col in df.columns if 'longitude' in col.lower()]
        
        lat_col = lat_cols[0] if lat_cols else None
        lon_col = lon_cols[0] if lon_cols else None
        
        if lat_col and lon_col:
            try:
                logger.info(f"Calculating geographic coverage using columns: {lat_col}, {lon_col}")
                
                # Ensure we're working with a copy to avoid modifying the original
                calc_df = df.copy()
                
                # Convert any problematic values to numeric
                try:
                    calc_df[lat_col] = pd.to_numeric(calc_df[lat_col], errors='coerce')
                    calc_df[lon_col] = pd.to_numeric(calc_df[lon_col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Error converting lat/lon to numeric: {str(e)}")
                
                # Only use rows with valid lat/lon values
                valid_coords_mask = calc_df[lat_col].notna() & calc_df[lon_col].notna()
                valid_coords_count = valid_coords_mask.sum()
                
                logger.info(f"Found {valid_coords_count} valid geographic coordinates")
                
                if valid_coords_count > 0:
                    # Get the latitude/longitude values
                    lat_values = calc_df.loc[valid_coords_mask, lat_col]
                    lon_values = calc_df.loc[valid_coords_mask, lon_col]
                    
                    # Check for valid numeric range for lat/lon
                    # Latitude: -90 to 90
                    # Longitude: -180 to 180
                    valid_lat_mask = (lat_values >= -90) & (lat_values <= 90)
                    valid_lon_mask = (lon_values >= -180) & (lon_values <= 180)
                    
                    # Combine masks
                    valid_range_mask = valid_lat_mask & valid_lon_mask
                    
                    if valid_range_mask.sum() > 0:
                        # Filter to valid range values
                        lat_values = lat_values[valid_range_mask]
                        lon_values = lon_values[valid_range_mask]
                        
                        logger.info(f"Using {len(lat_values)} coordinates in valid range for geographic coverage")
                        
                        # Calculate statistics
                        self._geographic_coverage = {
                            'latitude_range': {
                                'min': float(lat_values.min()),
                                'max': float(lat_values.max()),
                                'mean': float(lat_values.mean())
                            },
                            'longitude_range': {
                                'min': float(lon_values.min()),
                                'max': float(lon_values.max()),
                                'mean': float(lon_values.mean())
                            },
                            'total_locations': int(len(lat_values))
                        }
                    else:
                        logger.warning("No coordinates in valid range (-90 to 90 for lat, -180 to 180 for lon)")
                        self._geographic_coverage = None
                else:
                    logger.warning("No valid lat/lon coordinates found after conversion")
                    self._geographic_coverage = None
            except Exception as e:
                logger.error(f"Error calculating geographic coverage: {str(e)}")
                logger.error(traceback.format_exc())
                self._geographic_coverage = None
        else:
            logger.info("No latitude/longitude columns found for geographic coverage")
            self._geographic_coverage = None
    
    def _calculate_ecosystem_summaries(self) -> None:
        """Calculate summaries of ecosystem data."""
        if self._unified_df is None or self._unified_df.empty:
            logger.warning("Cannot calculate ecosystem summaries: DataFrame is empty or None")
            self._ecosystem_summaries = {}
            return
        
        try:
            df = self._unified_df
            summaries = {}
            
            # Look for ecosystem columns
            ecosystem_columns = [
                col for col in df.columns 
                if any(term in col.lower() for term in ['ecosystem', 'env_', 'biome', 'habitat'])
            ]
            
            logger.info(f"Found {len(ecosystem_columns)} ecosystem-related columns")
            
            # Process each column
            for col in ecosystem_columns:
                try:
                    # Skip columns with all null values
                    if df[col].isnull().all():
                        continue
                    
                    # Get value counts
                    value_counts = df[col].value_counts().head(10)  # Top 10 values
                    
                    # Only include if we have some values
                    if len(value_counts) > 0:
                        summaries[col] = {
                            'top_values': {str(k): int(v) for k, v in value_counts.items()},
                            'unique_count': int(df[col].nunique()),
                            'non_null_count': int(df[col].notna().sum())
                        }
                except Exception as e:
                    logger.warning(f"Error calculating summary for column {col}: {str(e)}")
            
            self._ecosystem_summaries = summaries
            logger.info(f"Calculated ecosystem summaries for {len(summaries)} columns")
            
        except Exception as e:
            logger.error(f"Error calculating ecosystem summaries: {str(e)}")
            logger.error(traceback.format_exc())
            self._ecosystem_summaries = {}
    
    def _get_base_column_descriptions(self) -> Dict[str, str]:
        """Get base descriptions for DataFrame columns."""
        return {
            # Study fields
            'id': 'Unique identifier for the entity',
            'name': 'Name of the study or biosample',
            'title': 'Title of the study',
            'description': 'Description of the study or biosample',
            'study_id': 'Reference to the study ID',
            'study_category': 'Category or type of the study',
            'study_description': 'Description of the study',
            'principal_investigator': 'Principal investigator of the study',
            'doi': 'Digital Object Identifier',
            'gold_study_id': 'GOLD database study identifier',
            'principal_investigator_name': 'Name of the principal investigator',
            
            # Biosample fields
            'biosample_id': 'Identifier for the biosample',
            'collection_date': 'Date when the sample was collected',
            'depth': 'Sampling depth',
            'ecosystem': 'Type of ecosystem',
            'ecosystem_category': 'Category of ecosystem',
            'ecosystem_type': 'Type of ecosystem',
            'ecosystem_subtype': 'Subtype of ecosystem',
            'env_broad_scale': 'Broad-scale environmental context',
            'env_local_scale': 'Local-scale environmental context',
            'env_medium': 'Environmental medium from which the sample was obtained',
            
            # Location fields
            'latitude': 'Latitude coordinate of the sampling location',
            'longitude': 'Longitude coordinate of the sampling location',
            'location': 'Location description',
            'country': 'Country where the sample was collected',
            'geolocation': 'Geographic location',
            
            # Environmental parameters
            'temperature': 'Temperature at the sampling location',
            'ph': 'pH value of the sample',
            'salinity': 'Salinity of the sample',
            'pressure': 'Pressure at the sampling location',
            'humidity': 'Humidity at the sampling location',
            
            # Sample characteristics
            'sample_collection_site': 'Site where the sample was collected',
            'sample_material_processing': 'Material processing method',
            'sample_type': 'Type of the sample',
            'scientific_name': 'Scientific name of the organism',
            'taxonomy': 'Taxonomic classification',
            'ncbi_taxonomy_name': 'NCBI taxonomy name',
            'feature_type': 'Type of feature',
            
            # Metagenomic properties
            'gold_biosample_id': 'GOLD database biosample identifier',
            'insdc_biosample_identifiers': 'INSDC biosample identifiers',
            'omics_type': 'Type of omics data',
            'analysis_type': 'Type of analysis performed',
            'part_of': 'Parent entity',
            'has_part': 'Child entities',
            
            # Data processing
            'processing_institution': 'Institution where data was processed',
            'seq_meth': 'Sequencing method',
            'omics_processing_id': 'Identifier for omics processing',
            'insdc_experiment_identifiers': 'INSDC experiment identifiers'
        }
    
    def get_dataframe_context(self) -> Dict[str, Any]:
        """Get context information about the DataFrame."""
        # Initialize empty context with default values
        context = {
            'column_descriptions': {},
            'statistics': {'numeric_columns': {}},
            'geographic_coverage': {},
            'studies_count': 0,
            'biosamples_count': 0,
            'unified_rows': 0
        }
        
        try:
            # Calculate statistics if not already done
            if not hasattr(self, '_stats') or not self._stats:
                self._calculate_unified_statistics()
            
            # Get base column descriptions if not already done
            if not hasattr(self, '_column_descriptions') or not self._column_descriptions:
                self._column_descriptions = self._get_base_column_descriptions()
            
            # Update context with actual values if available
            if hasattr(self, '_column_descriptions') and self._column_descriptions:
                context['column_descriptions'] = self._column_descriptions
            
            if hasattr(self, '_stats') and self._stats:
                context['statistics'] = self._stats
            
            if hasattr(self, '_geographic_coverage') and self._geographic_coverage:
                context['geographic_coverage'] = self._geographic_coverage
            
            # Update counts
            if hasattr(self, '_studies_df') and self._studies_df is not None:
                context['studies_count'] = len(self._studies_df)
            
            if hasattr(self, '_biosamples_df') and self._biosamples_df is not None:
                context['biosamples_count'] = len(self._biosamples_df)
            
            if hasattr(self, '_unified_df') and self._unified_df is not None:
                context['unified_rows'] = len(self._unified_df)
            
        except Exception as e:
            logger.error(f"Error getting DataFrame context: {str(e)}")
            logger.error(traceback.format_exc())
        
        return context
    
    @property
    def unified_df(self) -> pd.DataFrame:
        """Get unified DataFrame, refreshing if needed."""
        if self._unified_df is None and not self._loading_in_progress:
            # Only load if not already loaded and not currently loading
            logger.info("unified_df accessed but not loaded, triggering data load")
            self._load_all_data()
        # Use explicit None check instead of 'or' to avoid DataFrame truth value ambiguity
        return self._unified_df if self._unified_df is not None else pd.DataFrame()
    
    @property
    def studies_df(self) -> pd.DataFrame:
        """Get studies DataFrame, refreshing if needed."""
        if self._studies_df is None and not self._loading_in_progress:
            # Only load if not already loaded and not currently loading
            logger.info("studies_df accessed but not loaded, triggering studies load")
            self._studies_df = self._load_studies()
        # Use explicit None check instead of 'or' to avoid DataFrame truth value ambiguity
        return self._studies_df if self._studies_df is not None else pd.DataFrame()
    
    @property
    def biosamples_df(self) -> pd.DataFrame:
        """Get biosamples DataFrame, refreshing if needed."""
        if self._biosamples_df is None and not self._loading_in_progress:
            # Only load if not already loaded and not currently loading
            logger.info("biosamples_df accessed but not loaded, triggering biosamples load")
            self._biosamples_df = self._load_biosamples()
        # Use explicit None check instead of 'or' to avoid DataFrame truth value ambiguity
        return self._biosamples_df if self._biosamples_df is not None else pd.DataFrame()
    
    def _make_api_request(self, endpoint_path: str, method: str = "GET", params: dict = None, 
                          json_data: dict = None, use_fallback: bool = True) -> dict:
        """Make a request to the NMDC API with support for fallback URLs and retries.
        
        Args:
            endpoint_path: API endpoint path (e.g., "/study/search")
            method: HTTP method (GET or POST)
            params: URL parameters
            json_data: JSON data for POST requests
            use_fallback: Whether to try fallback URL if primary fails
            
        Returns:
            API response as dictionary
            
        Raises:
            Exception if all requests fail
        """
        urls_to_try = [f"{self.config.api_base_url}{endpoint_path}"]
        
        # Add fallback URL if enabled
        if use_fallback:
            urls_to_try.append(f"{self.config.api_fallback_url}{endpoint_path}")
        
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        timeout = self.config.request_timeout
        
        last_error = None
        for url in urls_to_try:
            # Try the request with retries
            for retry in range(self.config.max_retries):
                try:
                    logger.info(f"Making {method} request to {url} (Attempt {retry+1}/{self.config.max_retries})")
                    
                    if method.upper() == "GET":
                        response = requests.get(url, params=params, headers=headers, timeout=timeout)
                    else:  # POST
                        response = requests.post(url, params=params, json=json_data, headers=headers, timeout=timeout)
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    # If we get here, the request was successful
                    logger.info(f"Request to {url} successful")
                    return data
                
                except Exception as e:
                    last_error = e
                    logger.warning(f"Request to {url} failed (Attempt {retry+1}): {str(e)}")
                    
                    # Wait before retrying, but not on the last retry
                    if retry < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (retry + 1))  # Exponential backoff
        
        # If we get here, all URLs and retries failed
        error_msg = f"All API requests to {endpoint_path} failed after {len(urls_to_try)} URLs and {self.config.max_retries} retries per URL"
        logger.error(error_msg)
        if last_error:
            logger.error(f"Last error: {str(last_error)}")
        raise Exception(error_msg)

    def process(self) -> None:
        """Process all data sources and create the unified DataFrame."""
        start_time = time.time()
        logger.info("Processing all data sources...")
        
        try:
            # Load data from sources
            self._studies_df = self._load_studies()
            self._biosamples_df = self._load_biosamples()
            
            # Build the unified DataFrame and store the result
            self._unified_df = self._build_unified_dataframe()
            
            # Make sure the unified DataFrame is cleaned
            if self._unified_df is not None and not self._unified_df.empty:
                self._unified_df = self._clean_dataframe(self._unified_df)
                logger.info(f"Final unified DataFrame has {len(self._unified_df)} rows and {len(self._unified_df.columns)} columns")
                
                # Calculate statistics on the unified DataFrame
                self._calculate_unified_statistics()
                
                # Calculate ecosystem summaries
                self._calculate_ecosystem_summaries()
                
                # Save to cache
                self._save_to_cache()
            else:
                logger.warning("Unified DataFrame is empty after processing")
                self._stats = {'error': 'No data available after processing'}
            
            # Log completion time
            elapsed_time = time.time() - start_time
            logger.info(f"Data processing completed in {elapsed_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            logger.error(traceback.format_exc())

    def start_background_refresh(self) -> None:
        """Start a background thread to refresh data from API."""
        if self._loading_in_progress:
            logger.warning("Data refresh already in progress, skipping")
            return

        # Check if cache is older than 24 hours
        current_time = datetime.now()
        cache_age_hours = 0
        
        if hasattr(self, '_cache_timestamp') and self._cache_timestamp:
            cache_age = current_time - self._cache_timestamp
            cache_age_hours = cache_age.total_seconds() / 3600
            
            if cache_age_hours < 24:
                logger.info(f"Skipping background refresh - cache is only {cache_age_hours:.1f} hours old (< 24 hours)")
                return
        
        logger.info(f"Initiating background refresh - cache is {cache_age_hours:.1f} hours old")

        def refresh_data():
            try:
                self._loading_in_progress = True
                logger.info("Starting background data refresh...")
                
                # Load fresh data into temporary manager
                temp_manager = NMDCEnhancedDataManager(self.config, load_data=False)
                temp_manager._load_all_data()
                
                # Verify the new data is valid
                if (temp_manager._studies_df is not None and 
                    temp_manager._biosamples_df is not None and 
                    temp_manager._unified_df is not None and
                    not temp_manager._unified_df.empty):
                    
                    # Log the data sizes for verification
                    logger.info(
                        f"New data loaded - Studies: {len(temp_manager._studies_df):,} rows, "
                        f"Biosamples: {len(temp_manager._biosamples_df):,} rows, "
                        f"Unified: {len(temp_manager._unified_df):,} rows"
                    )
                    
                    # Atomic replacement of dataframes with lock
                    with self._cache_lock:
                        # Store old dataframe sizes for logging
                        old_sizes = {
                            'studies': len(self._studies_df) if self._studies_df is not None else 0,
                            'biosamples': len(self._biosamples_df) if self._biosamples_df is not None else 0,
                            'unified': len(self._unified_df) if self._unified_df is not None else 0
                        }
                        
                        # Create new references to the temporary dataframes
                        new_studies_df = temp_manager._studies_df
                        new_biosamples_df = temp_manager._biosamples_df
                        new_unified_df = temp_manager._unified_df
                        new_stats = temp_manager._stats
                        new_geographic_coverage = temp_manager._geographic_coverage
                        new_ecosystem_summaries = temp_manager._ecosystem_summaries
                        
                        # Atomic swap of references
                        self._studies_df = new_studies_df
                        self._biosamples_df = new_biosamples_df
                        self._unified_df = new_unified_df
                        self._stats = new_stats
                        self._geographic_coverage = new_geographic_coverage
                        self._ecosystem_summaries = new_ecosystem_summaries
                        
                        # Update cache timestamp
                        self._cache_timestamp = datetime.now()
                        
                        # Save to cache while still holding the lock
                        self._save_to_cache()
                    
                    # Log the changes
                    logger.info(
                        f"Background refresh complete - Changes in rows: "
                        f"Studies: {old_sizes['studies']:,} → {len(new_studies_df):,}, "
                        f"Biosamples: {old_sizes['biosamples']:,} → {len(new_biosamples_df):,}, "
                        f"Unified: {old_sizes['unified']:,} → {len(new_unified_df):,}"
                    )
                else:
                    logger.warning(
                        "Background refresh failed - received incomplete or empty data. "
                        "Keeping existing data."
                    )
            except Exception as e:
                logger.error(f"Error in background data refresh: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                self._loading_in_progress = False

        # Start background thread
        thread = threading.Thread(target=refresh_data)
        thread.daemon = True
        thread.start()
        logger.info("Started background data refresh thread")

    def _has_valid_data(self) -> bool:
        """Check if we have valid data loaded."""
        return (
            self._studies_df is not None and not self._studies_df.empty and
            self._biosamples_df is not None and not self._biosamples_df.empty and
            self._unified_df is not None and not self._unified_df.empty
        )