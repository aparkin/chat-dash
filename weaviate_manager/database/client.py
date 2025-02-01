"""
Weaviate client management module.

Provides functions for creating and managing Weaviate client connections.
"""

import logging
from typing import Optional, Dict, Any

import weaviate
from weaviate.config import AdditionalConfig
from weaviate.embedded import EmbeddedOptions

from ..config.settings import (
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_GRPC_HOST,
    WEAVIATE_GRPC_PORT,
    WEAVIATE_SECURE,
    REQUEST_TIMEOUT,
    VECTORIZER_TIMEOUT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL
)

def get_client() -> weaviate.Client:
    """Get configured Weaviate client with proper context management."""
    client = weaviate.connect_to_custom(
        http_host=WEAVIATE_HOST,
        http_port=WEAVIATE_HTTP_PORT,
        http_secure=WEAVIATE_SECURE,
        grpc_host=WEAVIATE_GRPC_HOST,
        grpc_port=WEAVIATE_GRPC_PORT,
        grpc_secure=WEAVIATE_SECURE,
        headers={
            "X-OpenAI-Api-Key": OPENAI_API_KEY,
        },
        additional_config=AdditionalConfig(
            timeout_config=REQUEST_TIMEOUT,
            timeout_vectorizer=VECTORIZER_TIMEOUT
        ),
        skip_init_checks=True  # Skip OpenID config check
    )
    return ClientContextManager(client)

class ClientContextManager:
    """Context manager wrapper for Weaviate client."""
    
    def __init__(self, client: weaviate.Client):
        self.client = client
        
    def __enter__(self):
        return self.client
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            self.client.close()
            
    def close(self):
        """Explicitly close the client connection."""
        if self.client:
            self.client.close()

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a collection including object count, properties, and references."""
        try:
            # Get collection schema to check references
            schema = self.client.schema.get(collection_name)
            
            # Count reference properties from schema
            reference_count = sum(
                1 for prop in schema.get('properties', [])
                if prop.get('dataType', [None])[0] in ['object[]', 'object']  # Reference types
            )
            
            # Get total object count
            total = self.client.collections.get(collection_name).objects.aggregate.over_all()
            
            # Get text property count
            text_props = sum(
                1 for prop in schema.get('properties', [])
                if prop.get('dataType', [None])[0] == 'text'
            )
            
            # Calculate appropriate sample size:
            # - For collections with <= 20 objects: use all objects
            # - For collections with > 20 objects: use 20% (min 20, max 100)
            if total <= 20:
                sample_size = total
            else:
                sample_size = min(max(20, int(total * 0.2)), 100)
            
            # Get sample of objects for property analysis
            sample = list(self.client.collections.get(collection_name)
                        .objects.fetch_objects(limit=sample_size)
                        .objects)
            
            return {
                'total': total,
                'text_properties': text_props,
                'references': reference_count,
                'sample': sample,
                'sample_size': sample_size
            }
            
        except Exception as e:
            logging.error(f"Error getting stats for {collection_name}: {str(e)}")
            return {
                'total': 0,
                'text_properties': 0,
                'references': 0,
                'sample': [],
                'sample_size': 0
            }

def check_connection(client: Optional[weaviate.Client] = None) -> bool:
    """Check if connection to Weaviate is working.
    
    Args:
        client: Optional client instance. If not provided, creates new client.
        
    Returns:
        bool: True if connection is working, False otherwise
    """
    try:
        if client is None:
            client = get_client()
            
        # Check if we can get meta information
        meta = client.get_meta()
        version = meta.get('version', 'unknown')
        
        logging.debug(f"Connected to Weaviate version: {version}")
        return True
        
    except Exception as e:
        logging.debug(f"Connection check failed: {str(e)}")
        return False 