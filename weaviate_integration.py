"""
Weaviate integration module for ChatDash.

This module provides a singleton class for managing Weaviate connections
and checking literature collection availability.
"""

import os
import logging
from typing import Dict, Optional, Tuple
import weaviate
from weaviate.config import AdditionalConfig, Timeout
from dotenv import load_dotenv
from pathlib import Path
from contextlib import contextmanager

# Load environment variables
project_root = Path(__file__).parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

class WeaviateConnection:
    """Singleton class for managing Weaviate connection."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WeaviateConnection, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Get OpenAI configuration from environment
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
        
        # Get Weaviate configuration
        self.weaviate_host = "weaviate.kbase.us"
        self.weaviate_http_port = 443
        self.weaviate_grpc_host = "weaviate-grpc.kbase.us"
        self.weaviate_grpc_port = 443
        
    def _create_client(self):
        """Create a new Weaviate client."""
        return weaviate.connect_to_custom(
            http_host=self.weaviate_host,
            http_port=self.weaviate_http_port,
            http_secure=True,
            grpc_host=self.weaviate_grpc_host,
            grpc_port=self.weaviate_grpc_port,
            grpc_secure=True,
            headers={
                "X-OpenAI-Api-Key": self.openai_api_key,
            },
            additional_config=AdditionalConfig(
                timeout_config=120,
                timeout_vectorizer=120
            ),
            skip_init_checks=True
        )
        
    @contextmanager
    def get_client(self):
        """Context manager for Weaviate client connection."""
        client = None
        try:
            client = self._create_client()
            yield client
        finally:
            if client:
                client.close()
        
    def connect(self) -> Tuple[bool, str]:
        """Establish connection to Weaviate."""
        try:
            if not self.openai_api_key:
                return False, "OpenAI API key not found in environment"
                
            with self.get_client() as client:
                # Just test the connection
                return True, "Connected"
            
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            return False, str(e)
            
    def check_literature_collections(self) -> Tuple[bool, str]:
        """Check if required literature collections exist."""
        try:
            # Get managed collections from settings
            from weaviate_manager.config.settings import MANAGED_COLLECTIONS
            
            with self.get_client() as client:
                try:
                    # Get all collections with full configuration
                    collections = client.collections.list_all(simple=False)
                    existing_collections = set(collections.keys())
                    
                    # Check if all required collections exist
                    missing = [col for col in MANAGED_COLLECTIONS if col not in existing_collections]
                    
                    if missing:
                        return False, f"Missing collections: {', '.join(missing)}"
                        
                    # Verify each collection has the expected configuration
                    for collection_name in MANAGED_COLLECTIONS:
                        if collection_name not in collections:
                            continue
                            
                        collection = client.collections.get(collection_name)
                        config = collection.config.get()
                        
                        # Check if collection has vectorizer configuration
                        if not hasattr(config, 'vectorizer'):
                            return False, f"Collection {collection_name} missing vectorizer configuration"
                        
                        # Check if collection has required properties
                        if not hasattr(config, 'properties') or not config.properties:
                            return False, f"Collection {collection_name} has no properties"
                    
                    return True, "Literature collections available"
                    
                except Exception as e:
                    self.logger.error(f"Error listing collections: {str(e)}")
                    return False, f"Error listing collections: {str(e)}"
            
        except Exception as e:
            self.logger.error(f"Collection check error: {str(e)}")
            return False, str(e)
            
    def get_status(self) -> Dict[str, Dict[str, str]]:
        """Get current connection and collection status."""
        connected, conn_msg = self.connect()
        collections_ok, coll_msg = self.check_literature_collections() if connected else (False, "Not connected")
        
        return {
            'connection': {
                'status': 'connected' if connected else 'error',
                'message': conn_msg
            },
            'collections': {
                'status': 'available' if collections_ok else 'error',
                'message': coll_msg
            }
        } 