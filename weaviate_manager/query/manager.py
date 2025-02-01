"""
Query management for the Weaviate Database Manager.

This module provides a flexible interface for querying the scientific literature
database using various search strategies.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import weaviate
from weaviate.collections import Collection
from weaviate.collections.classes.grpc import MetadataQuery, QueryReference, HybridFusion
from weaviate.collections.classes.filters import Filter

from ..config.settings import (
    DEFAULT_LIMIT,
    DEFAULT_CERTAINTY,
    VECTOR_SEARCH_LIMIT,
    DEFAULT_ALPHA,
    SEARCHABLE_COLLECTIONS
)
from .models import SearchResult, QueryParameters, EntityNetwork
from .inspector import DatabaseInspector
from .exceptions import DataIntegrityError

# Collection property definitions
COLLECTION_PROPERTIES = {
    'Article': [
        'title', 'authors', 'affiliations', 'funding_info', 'abstract',
        'introduction', 'methods', 'results', 'discussion', 'figures',
        'tables', 'publication_info', 'acknowledgements'
    ],
    'Author': ['canonical_name', 'email'],
    'Reference': [
        'title', 'journal', 'volume', 'pages', 'publication_date',
        'raw_reference'
    ],
    'NamedEntity': ['name', 'type', 'source'],
    'CitationContext': ['text', 'section', 'sentiment'],
    'NERArticleScore': ['score', 'confidence', 'method'],
    'NameVariant': ['variant', 'source', 'confidence']
}

class QueryManager:
    """
    Manages database queries and result processing.
    """
    
    def __init__(self, client: weaviate.WeaviateClient):
        """Initialize with Weaviate client."""
        self.client = client
        self.inspector = DatabaseInspector(client)
        self.schema_info = self.inspector.get_schema_info()
        self.logger = logging.getLogger(__name__)
        
        # Debug log schema info
        self.logger.debug("=== Schema Info ===")
        self.logger.debug(f"Schema info: {self.schema_info}")
        
        # Log only relevant references for each collection
        for collection_name, info in self.schema_info.items():
            if 'references' in info:
                self.logger.debug(f"{collection_name} references: {info['references']}")
        
        self.logger.debug("=== End Schema Info ===")
        
        # Cache valid references per collection to avoid repeated lookups
        self.valid_references = {}
        for collection_name, info in self.schema_info.items():
            self.valid_references[collection_name] = {
                ref['name']: ref['target'] 
                for ref in info.get('references', [])
            }
            self.logger.debug(f"Valid references for {collection_name}: {self.valid_references[collection_name]}")
    
    def _query_collection(
        self,
        collection_name: str,
        query: str,
        search_type: str = "hybrid",
        min_score: float = 0.0,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_LIMIT,
        include_vectors: bool = False
    ) -> List[Dict]:
        """Execute a search query on a specific collection."""
        try:
            self.logger.debug(f"=== Query Collection Debug ===")
            self.logger.debug(f"Collection: {collection_name}")
            self.logger.debug(f"Search type: {search_type}")
            self.logger.debug(f"Min score: {min_score}")
            self.logger.debug(f"Alpha: {alpha}")
            
            collection = self.client.collections.get(collection_name)
            
            # Get valid properties for this collection
            properties = []
            if collection_name in self.schema_info:
                properties = [p['name'] for p in self.schema_info[collection_name].get('properties', [])]
            
            # Configure base query parameters
            query_params = {
                "limit": limit,
                "return_metadata": ["score", "certainty", "explain_score", "distance"]
            }
            
            # Only add return_properties if we found properties in the schema
            if properties:
                query_params["return_properties"] = properties
            
            # Add references if they exist for this collection
            if collection_name in self.valid_references and self.valid_references[collection_name]:
                references = []
                for ref_name, target in self.valid_references[collection_name].items():
                    references.append(QueryReference(link_on=ref_name))
                if references:
                    query_params["return_references"] = references

            # Execute search based on type
            if search_type == "hybrid":
                self.logger.debug(f"Executing hybrid search with alpha={alpha}")
                results = collection.query.hybrid(
                    query=query,
                    alpha=alpha,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    **query_params
                ).objects
            elif search_type == "keyword":
                self.logger.debug("Executing keyword (BM25) search")
                results = collection.query.bm25(
                    query=query,
                    **query_params
                ).objects
            else:  # semantic
                self.logger.debug("Executing semantic (vector) search")
                results = collection.query.near_text(
                    query=query,
                    certainty=min_score if min_score > 0 else None,
                    **query_params
                ).objects
            
            # Post-process results to ensure consistent scoring
            processed_results = []
            for r in results:
                try:
                    # Get the appropriate score based on search type
                    score = 0.0
                    certainty = 0.0
                    raw_score = 0.0
                    explain_score = getattr(r.metadata, 'explain_score', '')
                    
                    if search_type == "semantic":
                        # For semantic search, score is 1 - distance
                        distance = getattr(r.metadata, 'distance', 0)
                        score = 1 - distance
                        raw_score = score  # For semantic, raw = final score
                        certainty = score  # For semantic, certainty equals score
                        self.logger.debug(f"\nSemantic result:")
                        self.logger.debug(f"Distance: {distance}")
                        self.logger.debug(f"Score: {score}")
                    elif search_type == "keyword":
                        # For keyword search, use BM25 score
                        score = getattr(r.metadata, 'score', 0.0)
                        raw_score = score  # For keyword, raw = final score
                        # Normalize BM25 score to certainty
                        certainty = min(score / (1 + abs(score)), 1.0)
                        self.logger.debug(f"\nKeyword result:")
                        self.logger.debug(f"Raw score: {score}")
                        self.logger.debug(f"Certainty: {certainty}")
                    else:  # hybrid
                        # For hybrid, extract original score from explain_score
                        raw_score = getattr(r.metadata, 'score', 0.0)
                        
                        # Extract original score if available
                        if explain_score:
                            import re
                            original_score_match = re.search(r'original score ([\d.]+)', explain_score)
                            if original_score_match:
                                score = float(original_score_match.group(1))
                            else:
                                score = raw_score
                        else:
                            score = raw_score
                            
                        # Calculate certainty from original score
                        certainty = min(score / (1 + abs(score)), 1.0)
                        
                        self.logger.debug(f"\nHybrid result:")
                        self.logger.debug(f"Raw normalized score: {raw_score}")
                        self.logger.debug(f"Original score: {score}")
                        self.logger.debug(f"Certainty: {certainty}")
                        self.logger.debug(f"Explain score: {explain_score}")
                    
                    # Only include results meeting the minimum score/certainty threshold
                    if score >= min_score:
                        result = {
                            'uuid': r.uuid,
                            'score': score,
                            'properties': r.properties,
                            'metadata': {
                                'score': score,
                                'raw_score': raw_score,
                                'certainty': certainty,
                                'explain_score': explain_score,
                                'search_type': search_type,
                                'distance': getattr(r.metadata, 'distance', None)
                            }
                        }
                        
                        processed_results.append(result)
                        
                except Exception as e:
                    self.logger.error(f"Error processing result: {str(e)}")
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("Detailed error:", exc_info=True)
                    continue
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Query error: {str(e)}")
            raise

    def comprehensive_search(
        self,
        query_text: str,
        search_type: str = "hybrid",
        min_score: float = 0.0,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_LIMIT,
        unify_results: bool = False,
        collections: Optional[List[str]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Perform a comprehensive search across collections."""
        try:
            self.logger.debug(f"Starting comprehensive search across collections: {collections or SEARCHABLE_COLLECTIONS}")
            
            # Query each collection
            results = {}
            seen_articles = set()  # Track seen article IDs
            
            # First process Article collection if present
            if 'Article' in (collections or SEARCHABLE_COLLECTIONS):
                article_results = self._query_collection(
                    collection_name='Article',
                    query=query_text,
                    search_type=search_type,
                    min_score=min_score,
                    alpha=alpha,
                    limit=limit,
                    include_vectors=search_type == "vector"
                )
                results['Article'] = article_results
                seen_articles.update(r['uuid'] for r in article_results)
                self.logger.debug(f"Found {len(results['Article'])} direct article hits")
            
            # Process other collections
            for collection_name in (collections or SEARCHABLE_COLLECTIONS):
                if collection_name == 'Article':
                    continue  # Already processed
                    
                collection_results = self._query_collection(
                    collection_name=collection_name,
                    query=query_text,
                    search_type=search_type,
                    min_score=min_score,
                    alpha=alpha,
                    limit=limit,
                    include_vectors=search_type == "vector"
                )
                
                if collection_results:
                    results[collection_name] = collection_results
                    self.logger.debug(f"Found {len(results[collection_name])} results for {collection_name}")
            
            response = {
                'raw_results': results,
                'query_info': {
                    'text': query_text,
                    'type': search_type,
                    'min_score': min_score,
                    'alpha': alpha
                }
            }
            
            # Unify results if requested
            if unify_results:
                unified = self._unify_results_on_articles(results)
                # Sort unified results by score in descending order
                unified.sort(key=lambda x: x['score'], reverse=True)
                response['unified_results'] = unified
                self.logger.debug(f"Unified results count: {len(unified)}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Comprehensive search error: {str(e)}", exc_info=True)
            raise

    def _has_reference_property(self, collection: str, reference: str) -> bool:
        """Check if a reference exists in the schema.
        
        Args:
            collection: Collection name
            reference: Reference property name
            
        Returns:
            bool: True if reference exists in schema
        """
        # Use cached valid references
        if collection not in self.valid_references:
            return False
        return reference in self.valid_references[collection]
    
    def _verify_object_exists(self, collection: str, uuid: str) -> bool:
        """Verify an object exists in the database.
        
        This should ALWAYS return True in normal operation since we use
        deterministic UUIDs. If it returns False, this indicates a serious
        data integrity issue that should be investigated.
        
        Args:
            collection: Collection name
            uuid: Object UUID
            
        Returns:
            bool: True if object exists
            
        Raises:
            DataIntegrityError: If object doesn't exist (indicates serious issue)
        """
        try:
            result = self.client.collections.get(collection).query.fetch_object_by_id(
                uuid=uuid,
                include_vector=False
            )
            if not result:
                self.logger.error(f"Data integrity error: {collection} object with UUID {uuid} not found")
                raise DataIntegrityError(f"{collection} object with UUID {uuid} not found")
            return True
        except Exception as e:
            self.logger.error(f"Error verifying {collection} object {uuid}: {str(e)}")
            raise
    
    def search_articles(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
        filters: Optional[Dict] = None,
        include_references: bool = False,
        include_authors: bool = False,
        include_entities: bool = False,
        min_certainty: float = DEFAULT_CERTAINTY,
        alpha: Optional[float] = None
    ) -> SearchResult:
        """Search for articles using specified strategy."""
        self.logger.debug(f"Starting {search_type} search with query: {query}")
        self.logger.debug(f"Search parameters: {locals()}")

        try:
            collection = self.client.collections.get("Article")
            
            # Build reference configuration if needed
            references = []
            if include_authors:
                references.append(
                    QueryReference(
                        link_on="authors",  # Schema name
                        return_properties=["canonical_name", "email", "affiliations"]
                    )
                )
            if include_entities:
                references.append(
                    QueryReference(
                        link_on="named_entities",  # Schema name
                        return_properties=["name", "type", "description"]
                    )
                )
            if include_references:
                references.append(
                    QueryReference(
                        link_on="references",  # Schema name
                        return_properties=["title", "journal", "volume", "pages", "publication_date", "raw_reference"]
                    )
                )
            
            # Configure metadata query with consistent settings
            metadata_query = MetadataQuery(
                score=True,  # Always get scores
                distance=True,  # Get distances for vector/semantic search
                explain_score=True,  # Get explanations for all types
                certainty=True,  # Get certainty scores
                creation_time=True
            )
            
            # Get properties from schema
            article_properties = []
            if 'Article' in self.schema_info:
                article_properties = [p['name'] for p in self.schema_info['Article'].get('properties', [])]
            
            # Configure base query parameters
            query_params = {
                "limit": limit,
                "offset": offset,
                "return_metadata": metadata_query,
                "return_references": references if references else None,
                "return_properties": article_properties
            }
            
            # Add filters if provided
            if filters:
                query_params["filters"] = Filter.from_dict(filters)
            
            # Execute search based on type
            if search_type == "semantic":
                self.logger.debug("Performing semantic search")
                result = collection.query.near_text(
                    query=query,
                    certainty=min_certainty,
                    **query_params
                )
            elif search_type == "keyword":
                self.logger.debug("Performing keyword search")
                result = collection.query.bm25(
                    query=query,
                    **query_params
                )
            else:  # hybrid
                self.logger.debug(f"Performing hybrid search with alpha={alpha if alpha is not None else DEFAULT_ALPHA}")
                result = collection.query.hybrid(
                    query=query,
                    alpha=alpha if alpha is not None else DEFAULT_ALPHA,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    **query_params
                )
            
            # Process results with consistent scoring
            items = []
            for article in result.objects:
                # Calculate appropriate score based on search type
                if search_type == "semantic":
                    score = 1 - getattr(article.metadata, 'distance', 0) if hasattr(article.metadata, 'distance') else 0.0
                else:
                    score = getattr(article.metadata, 'score', 0.0)
                
                # Only include results meeting the minimum score/certainty threshold
                if score >= min_certainty:
                    item = {
                        'id': article.uuid,
                        'properties': article.properties if hasattr(article, 'properties') else {},
                        'metadata': {
                            'score': score,
                            'certainty': getattr(article.metadata, 'certainty', None),
                            'explain_score': getattr(article.metadata, 'explain_score', None)
                        }
                    }
                    
                    # Process cross-references
                    if include_authors and hasattr(article, 'references'):
                        self.logger.debug(f"Processing authors for article {article.uuid}")
                        item['authors'] = [
                            {
                                'id': ref.uuid,
                                'properties': ref.properties if hasattr(ref, 'properties') else {}
                            }
                            for ref in article.references.get('authors', [])
                        ]
                    
                    if include_entities and hasattr(article, 'references'):
                        self.logger.debug(f"Processing entities for article {article.uuid}")
                        item['entities'] = [
                            {
                                'id': ref.uuid,
                                'properties': ref.properties if hasattr(ref, 'properties') else {}
                            }
                            for ref in article.references.get('named_entities', [])  # Schema name
                        ]
                    
                    if include_references and hasattr(article, 'references'):
                        self.logger.debug(f"Processing references for article {article.uuid}")
                        item['references'] = [
                            {
                                'id': ref.uuid,
                                'properties': ref.properties if hasattr(ref, 'properties') else {}
                            }
                            for ref in article.references.get('references', [])  # Schema name
                        ]
                    
                    items.append(item)
                    
                    # Log scoring details for debugging
                    self.logger.debug(f"Article {article.uuid} score: {score:.3f}")
                    if hasattr(article.metadata, 'explain_score'):
                        self.logger.debug(f"Score explanation: {article.metadata.explain_score}")
            
            # Create search result
            search_result = SearchResult(
                query={
                    "text": query,
                    "type": search_type,
                    "limit": limit,
                    "offset": offset,
                    "filters": filters,
                    "include_references": include_references,
                    "include_authors": include_authors,
                    "include_entities": include_entities,
                    "min_certainty": min_certainty,
                    "alpha": alpha if alpha is not None else DEFAULT_ALPHA
                },
                total_results=len(items),
                items=items
            )
            
            return search_result
            
        except Exception as e:
            self.logger.error(f"Error performing {search_type} search: {str(e)}", exc_info=True)
            raise
    
    def explore_references(
        self,
        article_id: str,
        direction: str = "outgoing",
        depth: int = 1,
        limit: int = DEFAULT_LIMIT
    ) -> SearchResult:
        """
        Explore citation network from an article.
        
        Args:
            article_id: UUID of the starting article
            direction: "outgoing" for references, "incoming" for citations
            depth: How many levels to traverse
            limit: Maximum results per level
            
        Returns:
            SearchResult containing citation network
        """
        try:
            collection = self.client.collections.get("Article")
            
            # Start with the source article
            article = collection.query.with_id(article_id).with_additional(["id"]).do()
            
            if not article or "data" not in article:
                raise ValueError(f"Article not found: {article_id}")
            
            references = []
            current_level = [article_id]
            
            for level in range(depth):
                next_level = []
                for current_id in current_level:
                    if direction == "outgoing":
                        refs = self._get_article_references(current_id, limit)
                    else:
                        refs = self._get_article_citations(current_id, limit)
                    
                    references.extend(refs)
                    next_level.extend([ref["id"] for ref in refs])
                
                current_level = next_level
                if not current_level:
                    break
            
            return SearchResult(
                query={
                    "article_id": article_id,
                    "direction": direction,
                    "depth": depth,
                    "limit": limit
                },
                total_results=len(references),
                items=references
            )
            
        except Exception as e:
            self.logger.error(f"Reference exploration error: {str(e)}")
            raise
    
    def find_related_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        min_score: float = 0.5,
        limit: int = DEFAULT_LIMIT
    ) -> SearchResult:
        """
        Find named entities related to a query.
        
        Args:
            query: Search text
            entity_type: Optional filter for entity type
            min_score: Minimum relevance score
            limit: Maximum number of results
            
        Returns:
            SearchResult containing related entities
        """
        try:
            collection = self.client.collections.get("NamedEntity")
            search_query = collection.query.near_text(
                query,
                certainty=min_score
            )
            
            if entity_type:
                search_query = search_query.with_where({
                    "path": ["type"],
                    "operator": "Equal",
                    "valueString": entity_type
                })
            
            result = search_query.with_limit(limit).with_additional(["certainty"]).do()
            
            items = []
            if result and "data" in result:
                for item in result["data"]["Get"]["NamedEntity"]:
                    items.append({
                        "id": item["_additional"]["id"],
                        "name": item["name"],
                        "type": item["type"],
                        "certainty": item["_additional"]["certainty"]
                    })
            
            return SearchResult(
                query={
                    "text": query,
                    "entity_type": entity_type,
                    "min_score": min_score,
                    "limit": limit
                },
                total_results=len(items),
                items=items
            )
            
        except Exception as e:
            self.logger.error(f"Entity search error: {str(e)}")
            raise
    
    def _apply_filters(self, query: Any, filters: Dict) -> Any:
        """
        Apply filters to a query.
        
        Args:
            query: Weaviate query builder object
            filters: Dictionary of filters to apply
            
        Returns:
            Modified query with filters applied
            
        The filters should be constructed using the FilterBuilder class
        and passed as a dictionary matching Weaviate's where filter format.
        """
        if not filters:
            return query
            
        return query.with_where(filters)
    
    def _process_results(
        self,
        raw_results: Dict[str, Any],
        include_references: bool = False,
        include_authors: bool = False,
        include_entities: bool = False
    ) -> List[Dict[str, Any]]:
        """Process raw results into standardized format."""
        items = []
        if raw_results and "data" in raw_results:
            for item in raw_results["data"]["Get"]["Article"]:
                processed = {
                    "id": item["_additional"]["id"],
                    "filename": item["filename"],
                    "abstract": item.get("abstract", ""),
                    "title": item.get("title", ""),
                    "publication_info": item.get("publication_info", ""),
                    "certainty": item["_additional"]["certainty"]
                }
                
                if include_references and "references" in item:
                    processed["references"] = item["references"]
                if include_authors and "authors" in item:
                    processed["authors"] = item["authors"]
                if include_entities and "named_entities" in item:
                    processed["entities"] = item["named_entities"]
                    
                items.append(processed)
        
        return items
        
    def _get_article_references(self, article_id: str, limit: int) -> List[Dict]:
        """Get outgoing references from an article."""
        try:
            # First verify article exists (will raise if not)
            self._verify_object_exists("Article", article_id)
            
            # Get the article with reference information
            article = self.client.collections.get("Article").query.fetch_object_by_id(
                uuid=article_id,
                return_references=QueryReference(
                    link_on="references",
                    target_collection="Reference",
                    return_properties=["title", "journal", "volume", "pages", "publication_date"]
                )
            )
            
            # Extract references
            references = []
            if hasattr(article, 'references') and 'references' in article.references:
                references = [
                    {
                        'id': ref.uuid,
                        'properties': ref.properties if hasattr(ref, 'properties') else {}
                    }
                    for ref in article.references['references'][:limit]
                ]
            
            return references
            
        except Exception as e:
            self.logger.error(f"Error retrieving references for article {article_id}: {str(e)}", exc_info=True)
            return []

    def _get_article_citations(self, article_id: str, limit: int) -> List[Dict]:
        """Get incoming citations to an article."""
        try:
            # First verify article exists (will raise if not)
            self._verify_object_exists("Article", article_id)
            
            # Query articles citing this one
            result = self.client.collections.get("Article").query.fetch_objects(
                filters=Filter.by_ref("references").by_id().equal(article_id),
                limit=limit,
                return_properties=["title", "publication_info"]
            )
            
            # Process results
            citations = []
            if result.objects:
                citations = [
                    {
                        'id': article.uuid,
                        'properties': article.properties if hasattr(article, 'properties') else {}
                    }
                    for article in result.objects
                ]
            
            return citations
            
        except Exception as e:
            self.logger.error(f"Error retrieving citations for article {article_id}: {str(e)}", exc_info=True)
            return []

    def search_across_collections(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        search_type: str = "hybrid",
        limit: int = DEFAULT_LIMIT,
        min_certainty: float = DEFAULT_CERTAINTY
    ) -> Dict[str, SearchResult]:
        """Search across multiple collections simultaneously."""
        try:
            if not collections:
                collections = SEARCHABLE_COLLECTIONS
            
            results = {}
            for collection_name in collections:
                collection = self.client.collections.get(collection_name)
                
                # Configure base query parameters
                query_params = {
                    "limit": limit,
                    "return_metadata": ["score", "certainty"]
                }
                
                # Execute search based on type
                if search_type == "semantic":
                    result = collection.query.near_text(
                        concepts=[query],
                        certainty=min_certainty,
                        **query_params
                    )
                elif search_type == "keyword":
                    result = collection.query.bm25(
                        query=query,
                        **query_params
                    )
                else:  # hybrid
                    result = collection.query.hybrid(
                        query=query,
                        alpha=DEFAULT_ALPHA,
                        fusion_type=HybridFusion.RELATIVE_SCORE,  # Use enum value
                        **query_params
                    )
                
                # Process results
                items = []
                for obj in result.objects:
                    item = {
                        'id': obj.uuid,
                        'properties': obj.properties,
                        'metadata': {
                            'score': getattr(obj.metadata, 'score', None),
                            'certainty': getattr(obj.metadata, 'certainty', None)
                        }
                    }
                    items.append(item)
                
                results[collection_name] = SearchResult(
                    query={
                        "text": query,
                        "type": search_type,
                        "collection": collection_name,
                        "limit": limit
                    },
                    total_results=len(items),
                    items=items,
                    collection=collection_name
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error performing cross-collection search: {str(e)}", exc_info=True)
            raise

    def get_entity_network(
        self,
        entity_id: str,
        depth: int = 1,
        include_articles: bool = True,
        include_authors: bool = True
    ) -> Dict[str, Any]:
        """
        Get the network of relationships for a named entity.
        
        Args:
            entity_id: UUID of the named entity
            depth: How many levels of relationships to traverse
            include_articles: Include related articles
            include_authors: Include related authors
            
        Returns:
            Dict containing entity details and its relationship network
        """
        try:
            entity = self.client.collections.get("NamedEntity").query.with_id(entity_id).with_additional(["id"]).do()
            
            if not entity or "data" not in entity:
                raise ValueError(f"Entity not found: {entity_id}")
            
            network = {
                "entity": entity["data"]["Get"]["NamedEntity"][0],
                "articles": [],
                "authors": [],
                "related_entities": []
            }
            
            # Get articles mentioning this entity
            if include_articles:
                articles = self.client.collections.get("Article").query.with_additional(["id"]).with_where({
                    "path": ["named_entities", "NamedEntity", "_additional", "id"],
                    "operator": "Equal",
                    "valueString": entity_id
                }).do()
                
                if articles and "data" in articles:
                    network["articles"] = articles["data"]["Get"]["Article"]
            
            # Get authors of those articles
            if include_authors and network["articles"]:
                author_ids = set()
                for article in network["articles"]:
                    if "authors" in article:
                        author_ids.update(a["_additional"]["id"] for a in article["authors"])
                
                if author_ids:
                    authors = self.client.collections.get("Author").query.with_additional(["id"]).with_where({
                        "path": ["_additional", "id"],
                        "operator": "ContainsAny",
                        "valueString": list(author_ids)
                    }).do()
                    
                    if authors and "data" in authors:
                        network["authors"] = authors["data"]["Get"]["Author"]
            
            # Get co-occurring entities if depth > 1
            if depth > 1 and network["articles"]:
                article_ids = [a["_additional"]["id"] for a in network["articles"]]
                related = self.client.collections.get("NamedEntity").query.with_additional(["id"]).with_where({
                    "path": ["mentionedIn", "Article", "_additional", "id"],
                    "operator": "ContainsAny",
                    "valueString": article_ids
                }).do()
                
                if related and "data" in related:
                    network["related_entities"] = [
                        e for e in related["data"]["Get"]["NamedEntity"]
                        if e["_additional"]["id"] != entity_id
                    ]
            
            return network
            
        except Exception as e:
            self.logger.error(f"Entity network error: {str(e)}")
            raise

    def find_connecting_path(
        self,
        start_id: str,
        end_id: str,
        max_depth: int = 3,
        connection_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find paths connecting two items in the database.
        
        Args:
            start_id: UUID of starting item
            end_id: UUID of ending item
            max_depth: Maximum path length to consider
            connection_types: Types of connections to traverse (e.g., ["authors", "references", "named_entities"])
            
        Returns:
            List of paths connecting the items, each path being a list of nodes
        """
        try:
            if not connection_types:
                connection_types = ["authors", "references", "named_entities"]
            
            # Build GraphQL query for path finding
            path_query = f"""
            {{
                Get {{
                    Article(
                        where: {{
                            path: ["_additional", "id"],
                            operator: Equal,
                            valueString: "{start_id}"
                        }}
                    ) {{
                        _additional {{
                            id
                            path(
                                to: {{
                                    id: "{end_id}"
                                }}
                                maxDepth: {max_depth}
                            ) {{
                                path {{
                                    id
                                    className
                                    properties
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            """
            
            result = self.client.query.raw(path_query)
            
            paths = []
            if result and "data" in result:
                article_data = result["data"]["Get"]["Article"]
                if article_data and "_additional" in article_data[0]:
                    path_data = article_data[0]["_additional"].get("path", [])
                    for path in path_data:
                        if "path" in path:
                            paths.append(path["path"])
            
            return paths
            
        except Exception as e:
            self.logger.error(f"Path finding error: {str(e)}")
            raise

    def _get_score(self, obj: Any, search_type: str) -> float:
        """Extract and normalize score from result object based on search type."""
        self.logger.debug(f"\n=== Score Extraction Debug ===")
        self.logger.debug(f"Search type: {search_type}")
        
        # Log all available attributes
        self.logger.debug("Available object attributes:")
        for attr in dir(obj):
            if not attr.startswith('_'):
                self.logger.debug(f"- {attr}: {getattr(obj, attr)}")
        
        # Log metadata details if available
        if hasattr(obj, 'metadata'):
            self.logger.debug("\nMetadata attributes:")
            for attr in dir(obj.metadata):
                if not attr.startswith('_'):
                    self.logger.debug(f"- {attr}: {getattr(obj.metadata, attr)}")
        else:
            self.logger.debug("No metadata found on object")
        
        # Extract score based on search type
        score = 0.0
        if search_type == "semantic":
            if hasattr(obj.metadata, 'distance'):
                score = 1 - obj.metadata.distance
                self.logger.debug(f"\nSemantic score calculation:")
                self.logger.debug(f"- Distance: {obj.metadata.distance}")
                self.logger.debug(f"- Calculated score: {score}")
            else:
                self.logger.debug("No distance found for semantic search")
        elif search_type in ["keyword", "hybrid"]:
            # Try metadata.score first
            if hasattr(obj.metadata, 'score'):
                score = obj.metadata.score
                self.logger.debug(f"\nScore found in metadata: {score}")
            # Then try direct score attribute
            elif hasattr(obj, 'score'):
                score = obj.score
                self.logger.debug(f"\nScore found directly on object: {score}")
            else:
                self.logger.debug("No score found in metadata or direct attributes")
        
        self.logger.debug(f"\nFinal score: {score}")
        self.logger.debug("=== End Score Extraction ===\n")
        return score
    
    def _unify_results_on_articles(self, results: Dict[str, List]) -> List[Dict]:
        """Unify results from different collections around articles."""
        try:
            # Track all article IDs we need to include and their scores
            article_ids = {}  # Dict to track article IDs and their best scores
            
            # Get search type from first result's metadata
            search_type = None
            for collection, items in results.items():
                if items:
                    search_type = items[0].get('metadata', {}).get('search_type', 'unknown')
                    break
            
            # 1. Add direct article hits (these take precedence)
            if 'Article' in results and results['Article']:
                for article in results['Article']:
                    # Get score based on search type
                    score = article.get('score', 0.0)
                    if search_type == 'semantic' and 'distance' in article.get('metadata', {}):
                        score = 1 - float(article['metadata']['distance'])
                    
                    article_ids[article['uuid']] = {
                        'score': score,
                        'source': 'Article',
                        'search_type': search_type
                    }
                    self.logger.debug(f"Added direct article hit: {article['uuid']} with score {score}")
            
            # 2. Add articles from references (if not already added with a better score)
            if 'Reference' in results and results['Reference']:
                for ref in results['Reference']:
                    # Get articles citing this reference
                    ref_with_citations = self.client.collections.get("Reference").query.fetch_object_by_id(
                        uuid=ref['uuid'],
                        return_references=[
                            QueryReference(link_on="cited_in")
                        ]
                    )
                    
                    if hasattr(ref_with_citations, 'references') and 'cited_in' in ref_with_citations.references:
                        citing_articles = ref_with_citations.references['cited_in'].objects
                        for citing_article in citing_articles:
                            # Get score based on search type
                            score = ref.get('score', 0.0)
                            if search_type == 'semantic' and 'distance' in ref.get('metadata', {}):
                                score = 1 - float(ref['metadata']['distance'])
                            
                            # Only add if not present or if this score is better
                            current_score = article_ids.get(citing_article.uuid, {}).get('score', 0.0)
                            if citing_article.uuid not in article_ids or score > current_score:
                                article_ids[citing_article.uuid] = {
                                    'score': score,
                                    'source': 'Reference',
                                    'search_type': search_type
                                }
                                self.logger.debug(f"Added/Updated article {citing_article.uuid} from reference with score {score}")
            
            # 3. Add articles from named entities (if not already added with a better score)
            if 'NamedEntity' in results and results['NamedEntity']:
                for entity in results['NamedEntity']:
                    # Get articles containing this entity
                    entity_with_articles = self.client.collections.get("NamedEntity").query.fetch_object_by_id(
                        uuid=entity['uuid'],
                        return_references=[
                            QueryReference(link_on="found_in")
                        ]
                    )
                    
                    if hasattr(entity_with_articles, 'references') and 'found_in' in entity_with_articles.references:
                        containing_articles = entity_with_articles.references['found_in'].objects
                        for containing_article in containing_articles:
                            # Get score based on search type
                            score = entity.get('score', 0.0)
                            if search_type == 'semantic' and 'distance' in entity.get('metadata', {}):
                                score = 1 - float(entity['metadata']['distance'])
                            
                            # Only add if not present or if this score is better
                            current_score = article_ids.get(containing_article.uuid, {}).get('score', 0.0)
                            if containing_article.uuid not in article_ids or score > current_score:
                                article_ids[containing_article.uuid] = {
                                    'score': score,
                                    'source': 'NamedEntity',
                                    'search_type': search_type
                                }
                                self.logger.debug(f"Added/Updated article {containing_article.uuid} from entity with score {score}")
            
            self.logger.debug(f"Total unique articles to process: {len(article_ids)}")
            
            # Now build unified results for each article
            unified_results = []
            processed_articles = set()  # Track processed articles to avoid duplicates
            
            for article_id, score_info in article_ids.items():
                # Skip if already processed
                if article_id in processed_articles:
                    continue
                    
                # Get full article details with all references
                article = self.client.collections.get("Article").query.fetch_object_by_id(
                    uuid=article_id,
                    return_references=[
                        QueryReference(link_on="authors"),
                        QueryReference(link_on="references"),
                        QueryReference(link_on="named_entities")
                    ]
                )
                
                if not article:
                    continue
                
                processed_articles.add(article_id)
                
                # Get article properties from all sources
                article_props = {}
                
                # 1. First get properties from the article object itself
                if hasattr(article, 'properties'):
                    if isinstance(article.properties, dict):
                        article_props.update(article.properties)
                    else:
                        article_props.update({
                            k: v for k, v in article.properties.__dict__.items() 
                            if not k.startswith('_')
                        })
                
                # 2. Then try schema properties
                if 'Article' in self.schema_info:
                    schema_props = [p['name'] for p in self.schema_info['Article'].get('properties', [])]
                    for prop in schema_props:
                        if hasattr(article.properties, prop):
                            article_props[prop] = getattr(article.properties, prop)
                
                # 3. Finally try to get properties from raw results if this was a direct hit
                if score_info['source'] == 'Article' and 'Article' in results and results['Article']:
                    raw_article = next((
                        a for a in results['Article'] 
                        if a['uuid'] == article_id
                    ), None)
                    if raw_article:
                        if isinstance(raw_article.get('properties'), dict):
                            article_props.update(raw_article['properties'])
                        elif hasattr(raw_article.get('properties', {}), '__dict__'):
                            article_props.update({
                                k: v for k, v in raw_article['properties'].__dict__.items() 
                                if not k.startswith('_')
                            })
                
                # Create base result with all properties
                unified_result = {
                    'id': article.uuid,
                    'score': score_info['score'],
                    'source': score_info['source'],
                    'properties': article_props,
                    'metadata': {
                        'score': score_info['score'],
                        'search_type': score_info['search_type']
                    },
                    'traced_elements': {
                        'Reference': [],
                        'Author': [],
                        'NamedEntity': []
                    }
                }
                
                # Track processed elements to avoid duplicates
                processed_elements = {
                    'Reference': set(),
                    'Author': set(),
                    'NamedEntity': set()
                }
                
                # Add matching references
                references = getattr(article, 'references', None)
                if references and isinstance(references, dict):
                    # Add references
                    if 'references' in references:
                        for ref in references['references'].objects:
                            # Skip if already processed
                            if ref.uuid in processed_elements['Reference']:
                                continue
                                
                            # Check if this reference was in our search results
                            matching_ref = next((r for r in results.get('Reference', []) if r['uuid'] == ref.uuid), None)
                            if matching_ref:
                                # Get reference properties from schema
                                ref_props = {}
                                if 'Reference' in self.schema_info:
                                    schema_props = [p['name'] for p in self.schema_info['Reference'].get('properties', [])]
                                    for prop in schema_props:
                                        if hasattr(ref.properties, prop):
                                            ref_props[prop] = getattr(ref.properties, prop)
                                
                                unified_result['traced_elements']['Reference'].append({
                                    'id': ref.uuid,
                                    'score': matching_ref['score'],
                                    'properties': ref_props
                                })
                                processed_elements['Reference'].add(ref.uuid)
                    
                    # Add authors
                    if 'authors' in references:
                        for author in references['authors'].objects:
                            # Skip if already processed
                            if author.uuid in processed_elements['Author']:
                                continue
                                
                            # Check if this author was in our search results
                            matching_author = next((a for a in results.get('Author', []) if a['uuid'] == author.uuid), None)
                            if matching_author:
                                # Get author properties from schema
                                author_props = {}
                                if 'Author' in self.schema_info:
                                    schema_props = [p['name'] for p in self.schema_info['Author'].get('properties', [])]
                                    for prop in schema_props:
                                        if hasattr(author.properties, prop):
                                            author_props[prop] = getattr(author.properties, prop)
                                
                                unified_result['traced_elements']['Author'].append({
                                    'id': author.uuid,
                                    'score': matching_author['score'],
                                    'properties': author_props
                                })
                                processed_elements['Author'].add(author.uuid)
                    
                    # Add named entities
                    if 'named_entities' in references:
                        for entity in references['named_entities'].objects:
                            # Skip if already processed
                            if entity.uuid in processed_elements['NamedEntity']:
                                continue
                                
                            # Check if this entity was in our search results
                            matching_entity = next((e for e in results.get('NamedEntity', []) if e['uuid'] == entity.uuid), None)
                            if matching_entity:
                                # Get entity properties from schema
                                entity_props = {}
                                if 'NamedEntity' in self.schema_info:
                                    schema_props = [p['name'] for p in self.schema_info['NamedEntity'].get('properties', [])]
                                    for prop in schema_props:
                                        if hasattr(entity.properties, prop):
                                            entity_props[prop] = getattr(entity.properties, prop)
                                
                                unified_result['traced_elements']['NamedEntity'].append({
                                    'id': entity.uuid,
                                    'score': matching_entity['score'],
                                    'properties': entity_props
                                })
                                processed_elements['NamedEntity'].add(entity.uuid)
                
                unified_results.append(unified_result)
            
            return unified_results
            
        except Exception as e:
            self.logger.error(f"Error unifying results: {str(e)}", exc_info=True)
            return []

    def _trace_back_to_articles(self, item: Dict, collection_name: str) -> List[str]:
        """Trace back to articles using cross-references."""
        article_ids = []

        if collection_name == 'Reference':
            # Use cited_in cross-reference
            article_ids = self._get_articles_by_reference(item['id'])
        elif collection_name == 'Author':
            # Use primary_articles cross-reference
            article_ids = self._get_articles_by_author(item['id'])
        elif collection_name == 'NameVariant':
            # Trace to Author and then to Article
            author_ids = self._get_authors_by_name_variant(item['id'])
            for author_id in author_ids:
                article_ids.extend(self._get_articles_by_author(author_id))
        elif collection_name == 'NamedEntity':
            # Use found_in cross-reference
            article_ids = self._get_articles_by_entity(item['id'])

        return article_ids

    def _get_articles_by_reference(self, reference_id: str) -> List[str]:
        """Get articles citing a specific reference."""
        try:
            # First run diagnostics
            self._inspect_reference(reference_id)
            
            # First get the reference object
            reference_collection = self.client.collections.get("Reference")
            reference = reference_collection.query.fetch_object_by_id(
                uuid=reference_id,
                return_references=QueryReference(
                    link_on="cited_in",
                    reference_property="cited_in"
                )
            )
            
            if not reference:
                self.logger.warning(f"Reference not found: {reference_id}")
                return []
            
            # Extract article IDs from the cross-references
            article_ids = []
            if hasattr(reference, 'references'):
                self.logger.debug(f"Available references: {list(reference.references.keys())}")
                if 'cited_in' in reference.references:
                    article_ids = [
                        article.uuid 
                        for article in reference.references['cited_in']
                    ]
                    self.logger.debug(f"Found {len(article_ids)} articles citing reference {reference_id}")
                    if article_ids:
                        self.logger.debug(f"First citing article ID: {article_ids[0]}")
                else:
                    self.logger.debug(f"No 'cited_in' reference found in reference object")
            else:
                self.logger.debug(f"Reference object has no references attribute")
            
            return article_ids
            
        except Exception as e:
            self.logger.error(f"Error retrieving articles for reference {reference_id}: {str(e)}", exc_info=True)
            return []

    def _get_authors_by_name_variant(self, name_variant_id: str) -> List[str]:
        """Get authors by a specific name variant."""
        try:
            collection = self.client.collections.get("Author")
            
            # Query authors using proper v4 filter syntax
            result = collection.query.fetch_objects(
                filters=QueryReference(
                    link_on="hasNameVariant",
                    reference_property="hasNameVariant"
                ).by_id().equal(name_variant_id),
                return_metadata=["score", "certainty"]
            )
            
            if not result.objects:
                self.logger.warning(f"No authors found for name variant: {name_variant_id}")
                return []
            
            return [obj.uuid for obj in result.objects]
            
        except Exception as e:
            self.logger.error(f"Error retrieving authors for name variant {name_variant_id}: {str(e)}", exc_info=True)
            return []

    def _get_articles_by_author(self, author_id: str) -> List[str]:
        """Get articles by a specific author."""
        try:
            # First verify author exists (will raise if not)
            self._verify_object_exists("Author", author_id)
            
            # Get the author object with references
            author = self.client.collections.get("Author").query.fetch_object_by_id(
                uuid=author_id,
                return_references=[
                    QueryReference(
                        link_on="primary_articles",
                        target_collection="Article",
                        return_properties=["_id"]
                    )
                ]
            )
            
            # Extract article IDs from the cross-references
            article_ids = []
            if hasattr(author, 'references') and 'primary_articles' in author.references:
                article_ids = [article.uuid for article in author.references['primary_articles']]
                self.logger.debug(f"Found {len(article_ids)} articles by author {author_id}")
            
            return article_ids
            
        except Exception as e:
            self.logger.error(f"Error retrieving articles for author {author_id}: {str(e)}")
            return []

    def _get_articles_by_entity(self, entity_id: str) -> List[str]:
        """Get articles mentioning a specific entity."""
        try:
            # First verify entity exists (will raise if not)
            self._verify_object_exists("NamedEntity", entity_id)
            
            # Get the entity object with references
            entity = self.client.collections.get("NamedEntity").query.fetch_object_by_id(
                uuid=entity_id,
                return_references=[
                    QueryReference(
                        link_on="found_in",
                        target_collection="Article",
                        return_properties=["_id"]
                    )
                ]
            )
            
            # Extract article IDs from the cross-references
            article_ids = []
            if hasattr(entity, 'references') and 'found_in' in entity.references:
                article_ids = [article.uuid for article in entity.references['found_in']]
                self.logger.debug(f"Found {len(article_ids)} articles containing entity {entity_id}")
            
            return article_ids
            
        except Exception as e:
            self.logger.error(f"Error retrieving articles for entity {entity_id}: {str(e)}")
            return []

    def _get_article_by_id(self, article_id: str, include_references: bool = False) -> Optional[Dict]:
        """Get full article details by ID."""
        try:
            # First verify article exists (will raise if not)
            self._verify_object_exists("Article", article_id)
            
            # Define base properties to return
            return_properties = [
                "filename", "affiliations", "funding_info",
                "abstract", "introduction", "methods", 
                "results", "discussion", "figures",
                "tables", "publication_info", "acknowledgements"
            ]
            
            # Define references to include if requested
            references = []
            if include_references:
                # Check each reference type in schema
                if self._has_reference_property("Article", "authors"):
                    references.append(
                        QueryReference(
                            link_on="authors",
                            target_collection="Author",
                            return_properties=["canonical_name", "email"]
                        )
                    )
                if self._has_reference_property("Article", "references"):
                    references.append(
                        QueryReference(
                            link_on="references",
                            target_collection="Reference",
                            return_properties=["title", "journal", "volume", "pages", "publication_date"]
                        )
                    )
                if self._has_reference_property("Article", "entities"):
                    references.append(
                        QueryReference(
                            link_on="entities",
                            target_collection="NamedEntity",
                            return_properties=["name", "type"]
                        )
                    )
            
            # Fetch article with validated properties and references
            article = self.client.collections.get("Article").query.fetch_object_by_id(
                uuid=article_id,
                properties=return_properties,
                return_references=references if references else None
            )
            
            # Initialize result with empty properties if none exist
            result = {
                'properties': article.properties if hasattr(article, 'properties') else {},
                'authors': [],
                'references': [],
                'entities': []
            }
            
            # Process references if requested and they exist
            if include_references and hasattr(article, 'references'):
                # Add authors if reference exists and has values
                if 'authors' in article.references:
                    result['authors'] = [
                        {'id': author.uuid, 'properties': author.properties if hasattr(author, 'properties') else {}}
                        for author in article.references['authors']
                    ]
                
                # Add references if reference exists and has values
                if 'references' in article.references:
                    result['references'] = [
                        {'id': ref.uuid, 'properties': ref.properties if hasattr(ref, 'properties') else {}}
                        for ref in article.references['references']
                    ]
                
                # Add entities if reference exists and has values
                if 'entities' in article.references:
                    result['entities'] = [
                        {'id': entity.uuid, 'properties': entity.properties if hasattr(entity, 'properties') else {}}
                        for entity in article.references['entities']
                    ]
            
            return result
            
        except DataIntegrityError:
            raise  # Re-raise data integrity errors
        except Exception as e:
            self.logger.error(f"Error retrieving article {article_id}: {str(e)}")
            raise

    def _inspect_reference(self, reference_id: str) -> None:
        """Diagnostic method to inspect a reference and its cross-references."""
        try:
            reference_collection = self.client.collections.get("Reference")
            
            # First, get the reference object with all its properties
            reference = reference_collection.query.fetch_object_by_id(
                uuid=reference_id,
                return_properties=["title", "journal", "volume", "pages", "publication_date", "raw_reference"]
            )
            
            if reference:
                self.logger.debug(f"\n=== Reference Object Inspection ===")
                self.logger.debug(f"Reference ID: {reference_id}")
                self.logger.debug(f"Properties: {reference.properties}")
                
                # Now try to get its cross-references
                ref_with_citations = reference_collection.query.fetch_object_by_id(
                    uuid=reference_id,
                    return_references=[
                        QueryReference(
                            link_on="cited_in",
                            reference_property="cited_in"
                        )
                    ]
                )
                
                if hasattr(ref_with_citations, 'references'):
                    self.logger.debug(f"\nCross-references available: {list(ref_with_citations.references.keys())}")
                    if 'cited_in' in ref_with_citations.references:
                        self.logger.debug(f"Number of citing articles: {len(ref_with_citations.references['cited_in'])}")
                        for article in ref_with_citations.references['cited_in']:
                            self.logger.debug(f"Citing article: {article.uuid} - {article.properties.get('title', 'No title')}")
                    else:
                        self.logger.debug("No 'cited_in' cross-references found")
                else:
                    self.logger.debug("No cross-references attribute found")
                
                self.logger.debug("=== End Inspection ===\n")
            else:
                self.logger.debug(f"Reference {reference_id} not found in database")
        except Exception as e:
            self.logger.error(f"Error inspecting reference {reference_id}: {str(e)}", exc_info=True)

    def _inspect_entity(self, entity_id: str) -> None:
        """Diagnostic method to inspect a named entity and its cross-references."""
        try:
            entity_collection = self.client.collections.get("NamedEntity")
            
            # First, get the entity object with all its properties
            entity = entity_collection.query.fetch_object_by_id(
                uuid=entity_id,
                return_properties=["name", "type"]
            )
            
            if entity:
                self.logger.debug(f"\n=== Named Entity Object Inspection ===")
                self.logger.debug(f"Entity ID: {entity_id}")
                self.logger.debug(f"Properties: {entity.properties}")
                
                # Now try to get its cross-references
                entity_with_articles = entity_collection.query.fetch_object_by_id(
                    uuid=entity_id,
                    return_references=[
                        QueryReference(
                            link_on="found_in",
                            reference_property="found_in"
                        )
                    ]
                )
                
                if hasattr(entity_with_articles, 'references'):
                    self.logger.debug(f"\nCross-references available: {list(entity_with_articles.references.keys())}")
                    if 'found_in' in entity_with_articles.references:
                        self.logger.debug(f"Number of containing articles: {len(entity_with_articles.references['found_in'])}")
                        for article in entity_with_articles.references['found_in']:
                            self.logger.debug(f"Article: {article.uuid} - {article.properties.get('title', 'No title')}")
                    else:
                        self.logger.debug("No 'found_in' cross-references found")
                else:
                    self.logger.debug("No cross-references attribute found")
                
                self.logger.debug("=== End Inspection ===\n")
            else:
                self.logger.debug(f"Entity {entity_id} not found in database")
        except Exception as e:
            self.logger.error(f"Error inspecting entity {entity_id}: {str(e)}", exc_info=True)

    def _filter_results(self, results: Dict[str, Any], min_score: float = 0.0) -> Dict[str, Any]:
        """Filter results based on score threshold."""
        search_type = results.get('query_info', {}).get('type', 'unknown')
        self.logger.debug(f"\n=== Filtering Results ===")
        self.logger.debug(f"Search type: {search_type}")
        self.logger.debug(f"Min score: {min_score}")
        
        filtered_results = {}
        for collection, items in results.get('raw_results', {}).items():
            self.logger.debug(f"\nProcessing collection: {collection}")
            filtered_items = []
            for item in items:
                metadata = item.get('metadata', {})
                self.logger.debug(f"\nItem {item.get('uuid', 'unknown')}:")
                self.logger.debug(f"Raw metadata score: {metadata.get('score')}")
                self.logger.debug(f"Raw explain_score: {metadata.get('explain_score')}")
                
                score = float(metadata.get('score', 0.0))
                
                if search_type == 'hybrid' and metadata.get('explain_score'):
                    import re
                    original_score_match = re.search(r'original score ([\d.]+)', metadata['explain_score'])
                    if original_score_match:
                        score = float(original_score_match.group(1))
                        self.logger.debug(f"Extracted original score: {score}")
                
                self.logger.debug(f"Final score for comparison: {score}")
                self.logger.debug(f"Keep item? {score >= min_score}")
                
                if score >= min_score:
                    filtered_items.append(item)
                    
            if filtered_items:
                filtered_results[collection] = filtered_items
                self.logger.debug(f"Kept {len(filtered_items)}/{len(items)} items for {collection}")
            else:
                print(f"No items kept for {collection}")
        
        if 'unified_results' in results:
            print("\nProcessing unified results:")
            unified_filtered = []
            for item in results['unified_results']:
                metadata = item.get('metadata', {})
                print(f"\nUnified item {item.get('id', 'unknown')}:")
                print(f"Raw metadata score: {metadata.get('score')}")
                print(f"Raw explain_score: {metadata.get('explain_score')}")
                
                score = float(metadata.get('score', 0.0))
                
                if search_type == 'hybrid' and metadata.get('explain_score'):
                    import re
                    original_score_match = re.search(r'original score ([\d.]+)', metadata['explain_score'])
                    if original_score_match:
                        score = float(original_score_match.group(1))
                        print(f"Extracted original score: {score}")
                
                print(f"Final score for comparison: {score}")
                print(f"Keep item? {score >= min_score}")
                
                if score >= min_score:
                    unified_filtered.append(item)
            
            if unified_filtered:
                filtered_results['unified_results'] = unified_filtered
                print(f"Kept {len(unified_filtered)}/{len(results['unified_results'])} unified results")
            else:
                print("No unified results kept")
        
        print("\n=== End Filtering ===\n")
        return filtered_results 