"""
Database schema inspection utilities.

This module provides tools for inspecting and validating the Weaviate database schema,
particularly focusing on property and reference validation.
"""

import logging
from typing import Dict, List, Optional
import weaviate
from weaviate.collections.classes.config import Property, ReferenceProperty
from weaviate.classes.query import QueryReference

class DatabaseInspector:
    """
    Inspects and validates database schema.
    
    This class provides methods to:
    1. Get schema information
    2. Validate property existence
    3. Check reference configurations
    """
    
    def __init__(self, client: weaviate.WeaviateClient):
        """Initialize with Weaviate client."""
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    def get_schema_info(self) -> Dict:
        """Get schema information for all collections."""
        try:
            # Get all collections using collections API with full config
            collections = self.client.collections.list_all(simple=False)
            schema_info = {}
            
            for collection_name, collection_config in collections.items():
                # Get collection config
                collection = self.client.collections.get(collection_name)
                config = collection.config.get()
                
                # Extract property info
                properties = []
                references = []
                
                # Process properties from config
                for prop in config.properties:
                    if isinstance(prop, ReferenceProperty):
                        # This is a reference property
                        targets = prop.target_collections
                        if isinstance(targets, str):
                            targets = [targets]
                        for target in targets:
                            references.append({
                                'name': prop.name,
                                'target': target,
                                'description': getattr(prop, 'description', '')
                            })
                    else:
                        # This is a regular property
                        properties.append({
                            'name': prop.name,
                            'type': prop.data_type.value if hasattr(prop.data_type, 'value') else str(prop.data_type),
                            'description': getattr(prop, 'description', '')
                        })
                
                # Also check collection level references
                if hasattr(collection_config, 'references'):
                    for ref in collection_config.references:
                        targets = ref.target_collections
                        if isinstance(targets, str):
                            targets = [targets]
                        for target in targets:
                            references.append({
                                'name': ref.name,
                                'target': target,
                                'description': getattr(ref, 'description', '')
                            })
                
                schema_info[collection_name] = {
                    'properties': properties,
                    'references': references,
                    'vectorizer': getattr(config, 'vectorizer', None),
                    'vector_index_config': getattr(config, 'vector_index_config', None)
                }
                
                # Debug log the schema info
                self.logger.debug(f"\nSchema info for {collection_name}:")
                self.logger.debug(f"Properties: {properties}")
                self.logger.debug(f"References: {references}")
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Error getting schema info: {str(e)}", exc_info=True)
            return {}
    
    def validate_property(self, collection: str, property_name: str) -> bool:
        """
        Check if a property exists in a collection's schema.
        
        Args:
            collection: Collection name
            property_name: Property to check
            
        Returns:
            bool: True if property exists
        """
        try:
            # Get collection directly
            collection_obj = self.client.collections.get(collection)
            config = collection_obj.config.get()
            
            # Check property existence
            return any(prop.name == property_name for prop in config.properties)
            
        except Exception as e:
            self.logger.error(f"Error validating property {property_name} in {collection}: {str(e)}")
            return False
    
    def get_reference_info(self, collection: str, reference_name: str) -> Optional[Dict]:
        """
        Get information about a reference property.
        
        Args:
            collection: Collection name
            reference_name: Reference property name
            
        Returns:
            Dict with reference info or None if not found
        """
        try:
            # Get collection directly
            collection_obj = self.client.collections.get(collection)
            config = collection_obj.config.get()
            
            # Find reference property
            for prop in config.properties:
                if isinstance(prop, ReferenceProperty) and prop.name == reference_name:
                    return {
                        'name': prop.name,
                        'target': prop.target_collection,
                        'description': getattr(prop, 'description', '')
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting reference info for {reference_name} in {collection}: {str(e)}")
            return None 