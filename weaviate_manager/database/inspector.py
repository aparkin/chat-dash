"""
Database inspection utilities for Weaviate.

Provides comprehensive introspection capabilities for database state,
schema validation, and collection statistics.
"""

import logging
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table

import weaviate
from weaviate.classes.config import Property, DataType, Configure, Tokenization, ReferenceProperty
from weaviate.config import AdditionalConfig, Timeout

from ..config.settings import MANAGED_COLLECTIONS

class DatabaseInspector:
    """Handles all database introspection and statistics."""
    
    def __init__(self, client: weaviate.Client):
        """Initialize inspector with client.
        
        Args:
            client: Connected Weaviate client
        """
        self.client = client
        self.console = Console()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get current database state including collections and counts.
        
        Returns:
            Dict containing:
                collections: Dict of collection info
                total_objects: Total object count
        """
        try:
            info = {
                'collections': {},
                'total_objects': 0
            }
            
            # Get all collections
            collections = self.client.collections.list_all(simple=False)
            if not collections:
                return info
                
            for collection_name, collection_config in collections.items():
                if collection_name in MANAGED_COLLECTIONS:
                    collection = self.client.collections.get(collection_name)
                    count_response = collection.aggregate.over_all(total_count=True)
                    object_count = count_response.total_count
                    
                    info['collections'][collection_name] = {
                        'object_count': object_count,
                        'properties': [
                            {
                                'name': prop.name,
                                'data_type': prop.data_type.value if hasattr(prop.data_type, 'value') else prop.data_type,
                                'description': getattr(prop, 'description', None)
                            }
                            for prop in collection_config.properties
                        ],
                        'vectorizer': collection_config.vectorizer
                    }
                    
                    info['total_objects'] += object_count
            
            return info
            
        except Exception as e:
            logging.error(f"Error getting database info: {str(e)}")
            raise
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get detailed schema information including cross-references."""
        try:
            # Get all collections using collections API
            collections = self.client.collections.list_all(simple=False)
            schema_info = {}
            
            for collection_name, collection_config in collections.items():
                if collection_name in MANAGED_COLLECTIONS:
                    # Get properties from collection config
                    properties = collection_config.properties
                    
                    # Separate regular and reference properties
                    regular_props = []
                    ref_props = []
                    
                    # Process each property
                    for prop in properties:
                        if not hasattr(prop, 'target_collections'):
                            regular_props.append({
                                'name': prop.name,
                                'type': prop.data_type.value if hasattr(prop.data_type, 'value') else prop.data_type,
                                'description': getattr(prop, 'description', ''),
                                'vectorize': getattr(prop, 'vectorizer_config', None) is not None,
                                'tokenization': getattr(prop, 'tokenization', None),
                                'indexFilterable': getattr(prop, 'index_filterable', False),
                                'indexSearchable': getattr(prop, 'index_searchable', False)
                            })
                        else:
                            # Handle multiple target collections if present
                            targets = prop.target_collections
                            if isinstance(targets, str):
                                targets = [targets]
                            for target in targets:
                                ref_props.append({
                                    'name': prop.name,
                                    'target': target,
                                    'description': getattr(prop, 'description', '')
                                })
                    
                    # Also check collection level references
                    if hasattr(collection_config, 'references'):
                        for ref in collection_config.references:
                            # Handle multiple target collections if present
                            targets = ref.target_collections
                            if isinstance(targets, str):
                                targets = [targets]
                            for target in targets:
                                ref_props.append({
                                    'name': ref.name,
                                    'target': target,
                                    'description': getattr(ref, 'description', '')
                                })
                    
                    schema_info[collection_name] = {
                        'description': collection_config.description,
                        'vectorizer': getattr(collection_config, 'vectorizer', None),
                        'vectorizer_config': getattr(collection_config, 'vectorizer_config', None),
                        'generative_config': getattr(collection_config, 'generative_config', None),
                        'properties': regular_props,
                        'references': ref_props
                    }
            
            return schema_info
            
        except Exception as e:
            logging.error(f"Error getting schema info: {str(e)}")
            return {}
    
    def get_collection_stats(self, collection_name: str, sample_size: int = 20) -> Dict[str, Any]:
        """Get statistics for a collection including object count, properties, and references."""
        try:
            logging.debug(f"\nGetting stats for collection: {collection_name}")
            
            # Get collection configuration
            collection = self.client.collections.get(collection_name)
            if not collection:
                raise ValueError(f"Collection {collection_name} not found")
            
            config = collection.config
            logging.debug(f"Got collection config for {collection_name}")
            
            # Get properties from collection config
            properties = config.properties
            logging.debug(f"Raw properties for {collection_name}: {properties}")
            
            # Count outgoing references
            reference_count = 0
            
            # Check property-level references
            for prop in properties:
                if hasattr(prop, 'data_type') and isinstance(prop.data_type, list):
                    reference_count += 1
            
            stats = {
                'object_count': collection.objects.total,
                'property_count': len(properties),
                'reference_count': reference_count,
                'properties': [
                    {
                        'name': prop.name,
                        'type': prop.data_type[0] if isinstance(prop.data_type, list) else prop.data_type,
                        'description': getattr(prop, 'description', '')
                    }
                    for prop in properties
                ]
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting collection stats: {str(e)}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("Detailed error:", exc_info=True)
            return {}

    def _analyze_text_properties_dict(self, objects: List[Dict], text_properties: List[Dict]) -> Dict[str, Any]:
        """Analyze text properties from sample objects using property dictionaries."""
        stats = {}
        for prop in text_properties:
            prop_name = prop.get('name')
            if prop_name:
                values = [obj.get(prop_name) for obj in objects if obj.get(prop_name)]
                if values:
                    stats[prop_name] = {
                        "avg_length": sum(len(v) for v in values) / len(values),
                        "sample_count": len(objects),
                        "non_empty_count": len(values)
                    }
        return stats
    
    def _analyze_references_dict(self, collection: Any, references: List[Dict]) -> Dict[str, Any]:
        """Analyze cross-references for a collection using reference dictionaries."""
        stats = {}
        for ref in references:
            ref_name = ref.get('name')
            if ref_name:
                try:
                    # Get count of objects with references
                    count = collection.aggregate.over_all().with_reference_count(ref_name).do()
                    stats[ref_name] = {
                        "target_collection": ref.get('target_collections', ['unknown'])[0],
                        "reference_count": count.reference_count
                    }
                except Exception as e:
                    logging.error(f"Error analyzing references for {ref_name}: {str(e)}")
        return stats

    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all collections."""
        stats = {}
        for collection_name in MANAGED_COLLECTIONS:
            stats[collection_name] = self.get_collection_stats(collection_name)
        return stats

    def display_stats_rich(self) -> None:
        """Display detailed statistics using rich tables."""
        stats = self.get_stats()
        
        # Create summary table
        summary = Table(title="Database Statistics")
        summary.add_column("Collection", style="cyan")
        summary.add_column("Objects", style="green")
        summary.add_column("Text Properties", style="yellow")
        summary.add_column("References", style="magenta")
        
        total_objects = 0
        
        for collection_name, collection_stats in stats.items():
            total_objects += collection_stats['total']
            summary.add_row(
                collection_name,
                str(collection_stats['total']),
                str(collection_stats['text_properties']),
                str(collection_stats['references'])
            )
        
        self.console.print(summary)
        self.console.print(f"\n[bold green]Total Objects: {total_objects}[/bold green]")

    def display_stats_log(self) -> None:
        """Display detailed statistics using logging."""
        stats = self.get_all_stats()
        
        logging.info("\nDatabase Statistics:")
        for collection_name, collection_stats in stats.items():
            logging.info(f"\n{collection_name}:")
            logging.info(f"  Objects: {collection_stats['object_count']:,}")
            
            if collection_stats['property_count'] > 0:
                logging.info("  Properties:")
                for prop_name, prop_stats in collection_stats.items():
                    if prop_name.startswith('property_count'):
                        logging.info(f"    {prop_name}:")
                        logging.info(f"      Count: {prop_stats}")
                    
            if collection_stats['reference_count'] > 0:
                logging.info("  References:")
                for ref_name, ref_stats in collection_stats.items():
                    if ref_name.startswith('reference_count'):
                        logging.info(f"    {ref_name} → {ref_stats['target_collection']}: "
                                   f"{ref_stats['reference_count']:,} references")

    def display_schema_summary(self) -> None:
        """Display a rich formatted summary of the schema."""
        schema_info = self.get_schema_info()
        
        # Create main collections table
        collections_table = Table(title="Collections Overview")
        collections_table.add_column("Collection", style="cyan")
        collections_table.add_column("Description", style="green")
        collections_table.add_column("Properties", style="yellow")
        collections_table.add_column("References", style="magenta")
        
        for name, info in schema_info.items():
            collections_table.add_row(
                name,
                info['description'],
                str(len(info['properties'])),
                str(len(info['references']))
            )
            
        self.console.print(collections_table)
        self.console.print()
        
        # Create detailed tables for each collection
        for name, info in schema_info.items():
            self.console.print(f"[cyan]Collection: {name}[/cyan]")
            
            if info['properties']:
                props_table = Table(title="Properties")
                props_table.add_column("Name", style="yellow")
                props_table.add_column("Type", style="green")
                props_table.add_column("Vectorized", style="cyan")
                props_table.add_column("Description")
                
                for prop in info['properties']:
                    props_table.add_row(
                        prop['name'],
                        prop['type'],
                        "✓" if prop.get('vectorize') else "-",
                        prop['description']
                    )
                self.console.print(props_table)
            
            if info['references']:
                refs_table = Table(title="References")
                refs_table.add_column("Name", style="magenta")
                refs_table.add_column("Target", style="cyan")
                refs_table.add_column("Description")
                
                for ref in info['references']:
                    refs_table.add_row(
                        ref['name'],
                        ref['target'],
                        ref['description']
                    )
                self.console.print(refs_table)
            
            self.console.print()
            
    def generate_mermaid_erd(self) -> str:
        """Generate a Mermaid ERD diagram of the schema."""
        schema_info = self.get_schema_info()
        
        # Start Mermaid ERD
        mermaid = ["```mermaid", "erDiagram"]
        
        # Add entities and their properties
        for name, info in schema_info.items():
            # Clean name for Mermaid compatibility
            clean_name = name.replace(" ", "_")
            
            # Start entity definition
            mermaid.append(f"    {clean_name} {{")
            
            # Add properties
            for prop in info['properties']:
                prop_type = prop['type'].lower()
                mermaid.append(f"        {prop_type} {prop['name']}")
                
            mermaid.append("    }")
            
            # Add relationships
            for ref in info['references']:
                target = ref['target'].replace(" ", "_")
                # Use standard ERD notation: one-to-many relationship
                mermaid.append(f"    {clean_name} ||--o{{ {target} : {ref['name']}")
        
        mermaid.append("```")
        return "\n".join(mermaid)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the database contents."""
        stats = {}
        
        try:
            # Get collection statistics
            for collection_name in MANAGED_COLLECTIONS:
                # Get all collections config using collections API
                collections = self.client.collections.list_all(simple=False)
                collection_config = collections.get(collection_name)
                if not collection_config:
                    continue
                
                # Get collection object for querying
                collection = self.client.collections.get(collection_name)
                
                # Get total object count
                count_response = collection.aggregate.over_all(total_count=True)
                logging.debug(f"Count response for {collection_name}: {count_response}")
                total = count_response.total_count
                logging.debug(f"Total objects in {collection_name}: {total}")
                
                # Get text property count
                text_props = sum(
                    1 for prop in collection_config.properties
                    if hasattr(prop, 'data_type') and prop.data_type == 'text'
                )
                logging.debug(f"Found {text_props} text properties")
                
                # Count references
                reference_count = 0
                
                # Check property-level references
                for prop in collection_config.properties:
                    if hasattr(prop, 'target_collections'):
                        targets = prop.target_collections
                        if isinstance(targets, str):
                            reference_count += 1
                        else:
                            reference_count += len(targets)
                
                # Check collection-level references
                if hasattr(collection_config, 'references'):
                    for ref in collection_config.references:
                        targets = ref.target_collections
                        if isinstance(targets, str):
                            reference_count += 1
                        else:
                            reference_count += len(targets)
                
                stats[collection_name] = {
                    'total': total,
                    'text_properties': text_props,
                    'references': reference_count
                }
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting stats: {str(e)}")
            return {}
    
    def validate_schema_consistency(self) -> bool:
        """Validate actual schema matches expected configuration.
        
        Returns:
            bool: True if schema is consistent
        """
        try:
            schema_info = self.get_schema_info()
            
            # Validate all managed collections exist
            for collection in MANAGED_COLLECTIONS:
                if collection not in schema_info:
                    logging.error(f"Missing collection: {collection}")
                    return False
            
            # Validate each collection has required properties
            for collection_name, config in schema_info.items():
                if not self._validate_collection_properties(collection_name, config['properties']):
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error validating schema: {str(e)}")
            return False
    
    def _validate_collection_properties(self, collection_name: str, 
                                     properties: List[Dict[str, Any]]) -> bool:
        """Validate properties for a collection.
        
        Args:
            collection_name: Name of collection
            properties: List of property configurations
            
        Returns:
            bool: True if properties are valid
        """
        
        # Add more specific validation as needed
        # TODO: Add more specific validation as needed
        return True
    
    def display_stats(self) -> None:
        """Display detailed statistics about the database contents."""
        stats = self.get_stats()
        
        logging.info("\nDatabase Statistics:")
        for collection_name, collection_stats in stats.items():
            logging.info(f"\n{collection_name}:")
            logging.info(f"  Objects: {collection_stats['total']:,}")
            
            if "text_properties" in collection_stats:
                logging.info("  Text Properties:")
                for prop_name, prop_stats in collection_stats["text_properties"].items():
                    logging.info(f"    {prop_name}:")
                    logging.info(f"      Average Length: {prop_stats['avg_length']:.1f} chars")
                    logging.info(f"      Non-empty: {prop_stats['non_empty_count']}/{prop_stats['sample_count']}")
                    
            if "references" in collection_stats:
                logging.info("  References:")
                for ref_name, ref_stats in collection_stats["references"].items():
                    logging.info(f"    {ref_name} → {ref_stats['target_collection']}: "
                               f"{ref_stats['reference_count']:,} references")
                               
        # Show example queries
        logging.info("\nExample Queries:")
        logging.info("  1. Get articles with abstracts:")
        logging.info('     client.query.get("Article").with_additional("abstract").with_limit(5).do()')
        logging.info("  2. Get authors with their articles:")
        logging.info('     client.query.get("Author").with_references("articles").with_limit(5).do()')
        logging.info("  3. Get named entities with high scores:")
        logging.info('     client.query.get("NamedEntity").with_references("article_scores").with_limit(5).do()')

    def display_database_info(self) -> None:
        """Display formatted database information."""
        info = self.get_database_info()
        
        if not info['collections']:
            self.console.print("[yellow]No collections found in database[/yellow]")
            return
        
        # Create main overview table
        overview = Table(title="Database Overview")
        overview.add_column("Collection", style="cyan")
        overview.add_column("Objects", style="green")
        overview.add_column("Properties", style="yellow")
        overview.add_column("Vectorizer", style="magenta")
        
        for name, collection_info in info['collections'].items():
            overview.add_row(
                name,
                str(collection_info['object_count']),
                str(len(collection_info['properties'])),
                str(collection_info.get('vectorizer', 'None'))
            )
        
        self.console.print(overview)
        self.console.print(f"\n[bold green]Total Objects: {info['total_objects']:,}[/bold green]")
        
        # Show detailed information for each collection
        for name, collection_info in info['collections'].items():
            self.console.print(f"\n[cyan]Collection: {name}[/cyan]")
            
            if collection_info['properties']:
                props_table = Table(title="Properties")
                props_table.add_column("Name", style="yellow")
                props_table.add_column("Type", style="green")
                props_table.add_column("Description")
                
                for prop in collection_info['properties']:
                    props_table.add_row(
                        prop['name'],
                        prop['data_type'],
                        prop.get('description', '')
                    )
                self.console.print(props_table)
            
            self.console.print() 