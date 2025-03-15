"""
Data manager for UniProt service.

This module provides the UniProtDataManager class which handles:
1. Communication with the UniProt REST API
2. Caching of API responses
3. Data transformation from API responses to pandas DataFrames
4. Rate limiting to comply with API usage policies

The data manager is the central component for data retrieval and manipulation.
"""

import requests
import pandas as pd
import json
import io
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from .models import UniProtConfig, ProteinEntry, QueryResult

class UniProtDataManager:
    """Manager for UniProt API interactions.
    
    Handles:
    - API request construction and execution
    - Response parsing and conversion
    - Caching for performance
    - Rate limiting for API compliance
    
    All data retrieval should go through this manager
    to ensure consistent handling and caching.
    """
    
    def __init__(self, config: UniProtConfig):
        """Initialize with configuration.
        
        Args:
            config: UniProt service configuration
        """
        self.config = config
        self.cache = {}
        self.cache_expiry = {}
        self.last_request_time = 0
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def search_proteins(self, 
                       query: str, 
                       fields: List[str], 
                       format: str = "json", 
                       size: int = 25) -> pd.DataFrame:
        """Search for proteins using UniProt search syntax.
        
        Args:
            query: UniProt query syntax string
            fields: List of fields to retrieve
            format: Response format (json or tsv)
            size: Maximum number of results
            
        Returns:
            DataFrame containing protein records
            
        Raises:
            ValueError: For unsupported formats
            requests.HTTPError: For API errors
        """
        # Check cache first
        cache_key = f"search:{query}:{','.join(fields)}:{format}:{size}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            self.logger.info(f"Cache hit for query: {query}")
            return self.cache[cache_key]
        
        # Log the query for debugging
        self.logger.info(f"Processing query: {query}")
        
        # Check if the query contains a reviewed filter
        reviewed_filter_present = "reviewed:true" in query.lower() or "reviewed:false" in query.lower()
        self.logger.debug(f"Query contains reviewed filter: {reviewed_filter_present}")
        
        # Prepare request parameters
        params = {
            "query": query,
            "fields": ",".join(fields),
            "format": format,
            "size": size
        }
        
        # Respect rate limits
        self._enforce_rate_limit()
        
        # Execute API request
        url = f"{self.config.base_url}{self.config.search_endpoint}"
        self.logger.info(f"Executing UniProt search: {url} with params {params}")
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"UniProt API error: {e}")
            raise
        
        # Process response based on format
        if format == "json":
            data = response.json()
            # Transform JSON to DataFrame
            df = self._json_to_dataframe(data)
            
            # Post-process to ensure reviewed status matches query if specified
            if "reviewed:true" in query.lower() and not df.empty:
                original_count = len(df)
                df = df[df['reviewed'] == True]
                filtered_count = len(df)
                if filtered_count < original_count:
                    self.logger.warning(f"Filtered out {original_count - filtered_count} results that were not reviewed despite 'reviewed:true' in query")
            elif "reviewed:false" in query.lower() and not df.empty:
                original_count = len(df)
                df = df[df['reviewed'] == False]
                filtered_count = len(df)
                if filtered_count < original_count:
                    self.logger.warning(f"Filtered out {original_count - filtered_count} results that were reviewed despite 'reviewed:false' in query")
                
        elif format == "tsv":
            # Parse TSV directly to DataFrame
            df = pd.read_csv(io.StringIO(response.text), sep='\t')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Update cache
        self.cache[cache_key] = df
        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.config.cache_expiry_hours)
        
        return df
    
    def get_protein_by_id(self, protein_id: str, format: str = "json") -> Dict[str, Any]:
        """Retrieve a specific protein by its UniProt ID/accession.
        
        Args:
            protein_id: UniProt accession or entry name
            format: Response format (json or text)
            
        Returns:
            Dictionary with protein data or text in requested format
            
        Raises:
            requests.HTTPError: For API errors or not found proteins
        """
        # Check cache first
        cache_key = f"protein:{protein_id}:{format}"
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            self.logger.info(f"Cache hit for protein ID: {protein_id}")
            return self.cache[cache_key]
        
        # Respect rate limits
        self._enforce_rate_limit()
        
        # Execute API request
        url = f"{self.config.base_url}/uniprotkb/{protein_id}"
        headers = {"Accept": "application/json"} if format == "json" else {}
        
        self.logger.info(f"Retrieving protein {protein_id} from {url}")
        
        try:
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
        except requests.HTTPError as e:
            self.logger.error(f"Error retrieving protein {protein_id}: {e}")
            raise
        
        # Process response based on format
        if format == "json":
            result = response.json()
        else:
            result = response.text
        
        # Update cache
        self.cache[cache_key] = result
        self.cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.config.cache_expiry_hours)
        
        return result
    
    def get_protein_entry(self, protein_id: str) -> ProteinEntry:
        """Get structured protein entry from ID.
        
        Retrieves protein data and converts to ProteinEntry model.
        
        Args:
            protein_id: UniProt accession or entry name
            
        Returns:
            ProteinEntry object with protein data
            
        Raises:
            requests.HTTPError: For API errors or not found proteins
        """
        protein_data = self.get_protein_by_id(protein_id)
        
        # Extract relevant fields (this will need adjustment based on actual API response)
        entry = ProteinEntry(
            accession=protein_data.get("accession", [protein_id])[0],
            entry_name=protein_data.get("id", protein_id),
            protein_name=protein_data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown"),
            gene_names=[gene.get("value") for gene in protein_data.get("genes", []) if "value" in gene],
            organism=protein_data.get("organism", {}).get("scientificName", None),
            length=protein_data.get("sequence", {}).get("length", None),
            sequence=protein_data.get("sequence", {}).get("value", None),
            reviewed=protein_data.get("entryType", None) == "UniProtKB/Swiss-Prot"
        )
        
        # Add additional annotations if available
        if "features" in protein_data:
            entry.annotations["features"] = protein_data["features"]
        
        return entry
    
    def _json_to_dataframe(self, json_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert JSON response to DataFrame.
        
        Args:
            json_data: JSON response from UniProt API
            
        Returns:
            DataFrame with protein records
        """
        # Check if results exist
        if 'results' not in json_data or not json_data['results']:
            self.logger.warning("No results found in JSON response")
            return pd.DataFrame()
            
        # Extract results
        results = json_data['results']
        self.logger.info(f"Processing {len(results)} results from UniProt API")
        
        # Initialize data lists
        data = []
        reviewed_count = 0
        
        # Process each result
        for result in results:
            # Extract entry type for determining review status
            entry_type = result.get('entryType', '')
            # Check if the entry is reviewed (SwissProt)
            is_reviewed = "reviewed" in entry_type.lower() or "swiss-prot" in entry_type.lower()
            
            # Debug log for entry type and review status
            self.logger.debug(f"Entry Type: {entry_type}, Reviewed: {is_reviewed}")
            
            if is_reviewed:
                reviewed_count += 1
                
            # Extract basic fields
            record = {
                'accession': result.get('primaryAccession', ''),
                'id': result.get('uniProtkbId', ''),
                'reviewed': is_reviewed,
                'entry_type': entry_type
            }
            
            # Extract protein name
            if 'proteinDescription' in result:
                protein_desc = result['proteinDescription']
                if 'recommendedName' in protein_desc:
                    record['protein_name'] = protein_desc['recommendedName'].get('fullName', {}).get('value', '')
                elif 'submittedName' in protein_desc and len(protein_desc['submittedName']) > 0:
                    record['protein_name'] = protein_desc['submittedName'][0].get('fullName', {}).get('value', '')
                else:
                    record['protein_name'] = ''
            else:
                record['protein_name'] = ''
                
            # Extract organism
            if 'organism' in result:
                record['organism'] = result['organism'].get('scientificName', '')
                record['taxonomy_id'] = result['organism'].get('taxonId', 0)
            else:
                record['organism'] = ''
                record['taxonomy_id'] = 0
                
            # Extract gene names
            gene_names = []
            for gene in result.get('genes', []):
                if 'geneName' in gene:
                    gene_names.append(gene['geneName'].get('value', ''))
            record['gene_names'] = ', '.join(gene_names) if gene_names else ''
            
            # Extract sequence
            if 'sequence' in result:
                record['sequence'] = result['sequence'].get('value', '')
                record['length'] = result['sequence'].get('length', 0)
                record['mass'] = result['sequence'].get('molWeight', 0)
            else:
                record['sequence'] = ''
                record['length'] = 0
                record['mass'] = 0
                
            # Add to data list
            data.append(record)
            
        # Log summary of reviewed entries
        if data:
            reviewed_percentage = (reviewed_count / len(data)) * 100
            self.logger.info(f"Total results: {len(data)}, Reviewed entries: {reviewed_count} ({reviewed_percentage:.1f}%)")
        
        # Convert to DataFrame
        return pd.DataFrame(data)
    
    def _enforce_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # If less than our rate limit, wait
        if elapsed < (1.0 / self.config.rate_limit_per_second):
            wait_time = (1.0 / self.config.rate_limit_per_second) - elapsed
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            
        self.last_request_time = time.time() 