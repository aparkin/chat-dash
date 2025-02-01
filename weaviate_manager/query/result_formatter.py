"""
Result formatting for search results.

This module provides the core formatter for converting raw Weaviate results
into a standardized format for display and analysis.
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from datetime import datetime

from .result_model import (
    QueryInfo, ResultSummary, ItemProperties,
    CollectionItem, StandardResults
)

# Define collection properties for each collection type
COLLECTION_PROPERTIES = {
    'Article': [
        'title', 'abstract', 'content', 'filename', 'publication_date',
        'journal', 'volume', 'issue', 'pages', 'doi', 'pmid', 'pmc'
    ],
    'Author': [
        'name', 'email', 'affiliation', 'orcid'
    ],
    'Reference': [
        'title', 'authors', 'journal', 'volume', 'issue', 'pages',
        'publication_date', 'doi', 'pmid', 'pmc', 'raw_reference'
    ],
    'NamedEntity': [
        'text', 'type', 'start', 'end', 'normalized'
    ]
}

class ResultFormatter:
    """Formats raw search results into standardized format."""
    
    def __init__(self):
        """Initialize the formatter."""
        self.logger = logging.getLogger(__name__)
        
    def format_results(self, results: Dict[str, Any]) -> StandardResults:
        """Format search results for display."""
        # Extract query info
        query_info = QueryInfo(
            text=results['query_info']['text'],
            type=results['query_info']['type'],
            parameters={
                k: v for k, v in results['query_info']['parameters'].items()
                if v is not None
            },
            timestamp=datetime.now()
        )
        
        # Check if results are unified
        is_unified = bool(results.get('unified_results'))
        query_info.parameters['unify_results'] = is_unified
        
        # Generate summary
        collection_results = results.get('raw_results', {})
        collection_counts = {
            name: len(items)
            for name, items in collection_results.items()
        }
        total_matches = sum(collection_counts.values())
        
        # Format collection results
        formatted_collections = {}
        for collection, items in collection_results.items():
            formatted_items = []
            for item in items:
                formatted_item = self._format_collection_item(item, collection)
                formatted_items.append(formatted_item)
            formatted_collections[collection] = formatted_items
        
        # Create StandardResults object with initial values
        standard_results = StandardResults(
            query_info=query_info,
            summary=ResultSummary(
                total_matches=total_matches,
                collection_counts=collection_counts,
                unified_articles=None  # Will update after filtering
            ),
            collection_results=formatted_collections,
            unified_results=None
        )
        
        # Add unified results if present, with duplicate detection
        if is_unified and 'unified_results' in results:
            seen_ids = set()
            unified_items = []
            
            # First pass to collect unique items
            for item in results['unified_results']:
                formatted_item = self._format_unified_item(item)
                if formatted_item.id not in seen_ids:
                    unified_items.append(formatted_item)
                    seen_ids.add(formatted_item.id)
                else:
                    self.logger.warning(f"Duplicate unified result ID found: {formatted_item.id}")
            
            # Update StandardResults with filtered unified results
            standard_results.unified_results = unified_items
            standard_results.summary.unified_articles = len(unified_items)
        
        return standard_results
        
    def _format_collection_item(self, item: Dict, collection: str) -> CollectionItem:
        """Format a single collection item."""
        # Extract metadata values
        score = item.get('score', 0.0)
        metadata = item.get('metadata', {})
        certainty = metadata.get('certainty') if metadata.get('certainty', 0.0) > 0 else None
        explanation = metadata.get('explain_score') if metadata.get('explain_score') else None
        
        # Extract item's properties
        properties = {}
        if isinstance(item.get('properties'), dict):
            properties = item['properties']
        elif hasattr(item.get('properties', {}), '__dict__'):
            properties = {
                k: v for k, v in item['properties'].__dict__.items()
                if not k.startswith('_')
            }
        
        # Create item properties
        item_props = ItemProperties(values=properties)
        
        # Extract cross references
        cross_refs = self._extract_cross_references(item)
        
        # Create collection item with only meaningful values
        collection_item_args = {
            'id': str(item.get('uuid', item.get('id', ''))),
            'score': score,
            'properties': item_props,
        }
        
        # Only add optional fields if they have meaningful values
        if cross_refs:
            collection_item_args['cross_references'] = cross_refs
        if certainty is not None:
            collection_item_args['certainty'] = certainty
        if explanation:
            collection_item_args['score_explanation'] = explanation
        
        return CollectionItem(**collection_item_args)
        
    def _format_unified_item(self, item: Dict) -> CollectionItem:
        """Format a single unified item."""
        # Extract metadata values
        score = item.get('score', 0.0)
        metadata = item.get('metadata', {})
        certainty = metadata.get('certainty') if metadata.get('certainty', 0.0) > 0 else None
        explanation = metadata.get('explain_score') if metadata.get('explain_score') else None
        
        # Create item properties
        item_props = ItemProperties(values=item.get('properties', {}))
        
        # Extract cross references from traced elements
        cross_refs = {}
        for collection, elements in item.get('traced_elements', {}).items():
            if elements:
                cross_refs[collection] = [str(element.get('id', '')) for element in elements]
        
        # Create collection item with only meaningful values
        collection_item_args = {
            'id': str(item.get('id', '')),
            'score': score,
            'properties': item_props,
        }
        
        # Only add optional fields if they have meaningful values
        if cross_refs:
            collection_item_args['cross_references'] = cross_refs
        if certainty is not None:
            collection_item_args['certainty'] = certainty
        if explanation:
            collection_item_args['score_explanation'] = explanation
        
        return CollectionItem(**collection_item_args)
        
    def _extract_metadata_values(self, item: Dict) -> Tuple[float, float, str]:
        """Extract score, certainty and explanation from item metadata."""
        metadata = {}
        if isinstance(item.get('metadata'), dict):
            metadata = item['metadata']
        elif hasattr(item.get('metadata', {}), '__dict__'):
            metadata = {
                k: v for k, v in item['metadata'].__dict__.items()
                if not k.startswith('_')
            }
            
        # Get search type
        search_type = metadata.get('search_type', 'unknown')
        
        # Extract score - use original scores without normalization
        score = 0.0
        explanation = metadata.get('explain_score', '')
        
        # For hybrid search, try to extract original score from explanation
        if search_type == 'hybrid' and explanation:
            import re
            original_score_match = re.search(r'original score ([\d.]+)', explanation)
            if original_score_match:
                score = float(original_score_match.group(1))
            else:
                score = float(metadata.get('score', 0.0))
        else:
            score = float(metadata.get('score', 0.0))
        
        # Calculate certainty - only for semantic search where it's meaningful
        certainty = None
        if search_type == 'semantic' and 'distance' in metadata:
            certainty = 1 - float(metadata['distance'])
        
        # Get or format explanation
        if not explanation:
            if search_type == 'semantic':
                explanation = f"Semantic distance: {metadata.get('distance', 'N/A')}"
            elif search_type == 'keyword':
                explanation = f"BM25 score: {score:.3f}"
            elif search_type == 'hybrid':
                explanation = f"Hybrid score: {score:.3f}"
                if 'alpha' in metadata:
                    explanation += f" (alpha: {metadata['alpha']:.2f})"
                if 'distance' in metadata and 'score' in metadata:
                    explanation += f"\nComponents: distance={metadata['distance']}, BM25={metadata['score']}"
            else:
                explanation = f"Score: {score:.3f}"
                
        return score, certainty, explanation
        
    def _extract_cross_references(self, item: Dict) -> Dict[str, List[str]]:
        """Extract cross-references from an item."""
        refs = {}
        
        # Handle references from raw Weaviate response
        if 'references' in item:
            for ref_name, ref_list in item['references'].items():
                if isinstance(ref_list, list):
                    refs[ref_name] = [str(ref['uuid']) for ref in ref_list if 'uuid' in ref]
        
        # Handle references in _additional field
        additional = item.get('_additional', {})
        for key, value in additional.items():
            if key.endswith('_refs') and isinstance(value, list):
                collection_name = key[:-5]  # Remove '_refs' suffix
                refs[collection_name] = [str(ref_id) for ref_id in value if ref_id]
        
        # Handle references in properties
        properties = {}
        if isinstance(item.get('properties'), dict):
            properties = item['properties']
        elif hasattr(item.get('properties', {}), '__dict__'):
            properties = {
                k: v for k, v in item['properties'].__dict__.items()
                if not k.startswith('_')
            }
            
        # Look for reference properties (lists of IDs or objects)
        for key, value in properties.items():
            if isinstance(value, list) and value:
                # Try to determine if this is a reference list
                if all(isinstance(v, str) for v in value):
                    # List of IDs
                    refs[key] = value
                elif all(isinstance(v, dict) for v in value):
                    # List of objects, extract IDs
                    refs[key] = [
                        str(v.get('uuid', v.get('id')))
                        for v in value
                        if 'uuid' in v or 'id' in v
                    ]
                elif all(hasattr(v, '__class__') and v.__class__.__name__ == '_WeaviateUUIDInt' for v in value):
                    # List of Weaviate UUIDs
                    refs[key] = [str(v) for v in value]
                    
        # Handle traced_elements if present (for unified results)
        if 'traced_elements' in item:
            traced = item['traced_elements']
            if isinstance(traced, dict):
                for collection, elements in traced.items():
                    if isinstance(elements, list):
                        refs[collection] = [
                            str(e.get('uuid', e.get('id'))) if isinstance(e, dict)
                            else str(e) if hasattr(e, '__class__') and e.__class__.__name__ == '_WeaviateUUIDInt'
                            else str(e)
                            for e in elements
                        ]
                    
        return refs 