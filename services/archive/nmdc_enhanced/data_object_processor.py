"""
Data Object Processor for NMDC API.

This module provides functionality for retrieving and processing data objects
from the NMDC API.
"""

import logging
import pandas as pd
import tempfile
import os
import requests
from typing import Dict, Any, List, Optional, Tuple, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

class DataObjectProcessor:
    """Processes data objects from NMDC API."""
    
    def __init__(self, api_client):
        """Initialize the data object processor.
        
        Args:
            api_client: NMDCApiClient instance for API access
        """
        self.api_client = api_client
        self.temp_dir = tempfile.mkdtemp(prefix="nmdc_data_")
        logger.info(f"DataObjectProcessor initialized with temp directory: {self.temp_dir}")
    
    async def get_data_object(self, data_object_id: str) -> Dict[str, Any]:
        """Get a data object by ID.
        
        Args:
            data_object_id: ID of the data object
            
        Returns:
            Dictionary with data object metadata
        """
        return await self.api_client.get_entity("data_object", data_object_id)
    
    async def get_data_objects_for_biosample(self, biosample_id: str) -> pd.DataFrame:
        """Get data objects for a biosample.
        
        Args:
            biosample_id: ID of the biosample
            
        Returns:
            DataFrame with data objects
        """
        return await self.api_client.get_related_entities("biosample", biosample_id, "data_object")
    
    async def get_data_objects_for_study(self, study_id: str) -> pd.DataFrame:
        """Get data objects for a study.
        
        Args:
            study_id: ID of the study
            
        Returns:
            DataFrame with data objects
        """
        return await self.api_client.get_data_objects_by_study(study_id)
    
    async def download_file(self, url: str, filename: Optional[str] = None) -> str:
        """Download a file from a URL.
        
        Args:
            url: URL of the file to download
            filename: Optional filename to save as
            
        Returns:
            Path to the downloaded file
        """
        try:
            # Create a filename if not provided
            if not filename:
                filename = url.split('/')[-1]
                if not filename:
                    filename = f"data_{hash(url) % 10000}.dat"
            
            # Ensure filename is safe
            filename = os.path.basename(filename)
            
            # Create full path
            file_path = os.path.join(self.temp_dir, filename)
            
            # Download the file
            logger.info(f"Downloading file from {url} to {file_path}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return file_path
        
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {str(e)}")
            raise
    
    async def process_taxonomic_data(self, file_path: str) -> pd.DataFrame:
        """Process a taxonomic data file.
        
        Args:
            file_path: Path to the taxonomic data file
            
        Returns:
            DataFrame with processed taxonomic data
        """
        try:
            logger.info(f"Processing taxonomic data file: {file_path}")
            
            # Determine file format based on extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.tsv' or extension == '.txt':
                # Try to read as TSV
                return pd.read_csv(file_path, sep='\t', comment='#')
            
            elif extension == '.csv':
                # Read as CSV
                return pd.read_csv(file_path, comment='#')
            
            elif extension == '.biom' or extension == '.hdf5' or extension == '.h5':
                # For BIOM format, we would need biom package
                logger.warning(f"BIOM format not directly supported: {file_path}")
                # In a real implementation, you would use the biom package
                return pd.DataFrame()
            
            else:
                # Try to infer format from content
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                if '\t' in first_line:
                    return pd.read_csv(file_path, sep='\t', comment='#')
                elif ',' in first_line:
                    return pd.read_csv(file_path, comment='#')
                else:
                    logger.warning(f"Could not determine format for file: {file_path}")
                    return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error processing taxonomic data file: {str(e)}")
            return pd.DataFrame()
    
    async def process_functional_data(self, file_path: str) -> pd.DataFrame:
        """Process a functional annotation file.
        
        Args:
            file_path: Path to the functional annotation file
            
        Returns:
            DataFrame with processed functional data
        """
        try:
            logger.info(f"Processing functional data file: {file_path}")
            
            # Determine file format based on extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.tsv' or extension == '.txt':
                # Try to read as TSV
                return pd.read_csv(file_path, sep='\t', comment='#')
            
            elif extension == '.csv':
                # Read as CSV
                return pd.read_csv(file_path, comment='#')
            
            else:
                # Try to infer format from content
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                if '\t' in first_line:
                    return pd.read_csv(file_path, sep='\t', comment='#')
                elif ',' in first_line:
                    return pd.read_csv(file_path, comment='#')
                else:
                    logger.warning(f"Could not determine format for file: {file_path}")
                    return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error processing functional data file: {str(e)}")
            return pd.DataFrame()
    
    async def process_file(self, file_path: str) -> pd.DataFrame:
        """Process a generic data file.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            DataFrame with processed data
        """
        try:
            logger.info(f"Processing generic data file: {file_path}")
            
            # Determine file format based on extension
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension in ['.tsv', '.txt', '.tab']:
                # Try to read as TSV
                return pd.read_csv(file_path, sep='\t', comment='#')
            
            elif extension == '.csv':
                # Read as CSV
                return pd.read_csv(file_path, comment='#')
            
            elif extension in ['.xlsx', '.xls']:
                # Read Excel
                return pd.read_excel(file_path)
            
            elif extension == '.json':
                # Read JSON
                return pd.read_json(file_path)
            
            elif extension in ['.parquet', '.pq']:
                # Read Parquet
                return pd.read_parquet(file_path)
            
            else:
                # Try to infer format from content
                with open(file_path, 'r') as f:
                    first_line = f.readline().strip()
                
                if '\t' in first_line:
                    return pd.read_csv(file_path, sep='\t', comment='#')
                elif ',' in first_line:
                    return pd.read_csv(file_path, comment='#')
                else:
                    logger.warning(f"Could not determine format for file: {file_path}")
                    return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Error processing data file: {str(e)}")
            return pd.DataFrame()
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    def __del__(self):
        """Destructor to clean up resources."""
        self.cleanup() 