"""
Database importer for Weaviate.

This module provides functionality for importing scientific literature data into Weaviate,
with robust handling of:
- Batch processing with configurable sizes per entity type
- Text compression and token management for large text fields
- Error handling and retry logic for failed imports
- Progress tracking and detailed statistics
- UUID consistency across imports
- Cross-reference management between entities

Token Management Features:
- Per-field token counting and compression
- Total token limit enforcement across all fields
- Intelligent compression preserving important content
- Configurable safety margins for token limits
- Progressive compression based on content importance

Compression Strategy:
1. Individual field compression when exceeding per-field limits
2. Total token calculation across all fields
3. Additional compression if total exceeds safe limit
4. Content preservation based on:
   - Scientific term importance
   - Position in text (favoring first/last paragraphs)
   - Numerical content density
   - Citation presence

The import process is designed to be resilient and informative, with comprehensive
logging and progress reporting at each stage.
"""

import logging
from typing import Dict, Set, Optional, List, Any, Iterator, Tuple
from pathlib import Path
import tiktoken
import json
import random
from collections import defaultdict
import time

import weaviate
from ..data.loader import LiteratureDataManager
from ..data.models import Article, Author, Reference, NamedEntity
from .client import get_client
from ..config.settings import (
    BATCH_SIZES, 
    MANAGED_COLLECTIONS,
    MAX_STRING_LENGTH,
    DEFAULT_MAX_TOKENS,
    WORDS_PER_TOKEN,
    VECTORIZER_MODEL,
    TEXT_REDUCTION_RATIOS,
    IMPORTANCE_WEIGHTS,
    IMPORTANCE_KEY_TERMS,
    IMPORTANCE_COMPARISON_TERMS,
    MODEL_TOKEN_LIMITS,
    TOKEN_SAFETY_MARGIN
)
from weaviate.util import generate_uuid5
from tqdm import tqdm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from ..utils import log_progress
from weaviate.collections.classes.grpc import QueryReference
from weaviate.classes.query import Filter

# Configure logging
logger = logging.getLogger('weaviate_manager.importer')

class WeaviateImporter:
    """
    Manages the import of scientific literature data into Weaviate.
    
    This class implements a robust import pipeline with:
    - Intelligent text compression to meet token limits while preserving key content
    - Batch processing with collection-specific batch sizes
    - Comprehensive error handling and retry mechanisms
    - Detailed progress tracking and statistics
    - UUID consistency management
    - Cross-reference integrity maintenance
    
    Token Management:
    - Uses cl100k_base tokenizer (same as text-embedding-3-large)
    - Enforces per-field and total token limits
    - Applies progressive compression based on content importance
    - Maintains configurable safety margins
    - Provides detailed token statistics and logging
    
    Compression Features:
    - Intelligent content selection based on scientific importance
    - Position-aware preservation (favors start/end)
    - Numerical content density consideration
    - Citation preservation
    - Configurable compression ratios
    
    The import process occurs in two phases:
    1. Primary Entity Import:
       - Articles with full text and metadata
       - Authors with canonical names
       - References with citation information
       - Named Entities from text analysis
       - Citation Contexts from references
       - NER Scores for entity recognition
       - Name Variants for author disambiguation
       
    2. Cross-Reference Creation:
       - Article-Author relationships
       - Article-Reference citations
       - Article-NamedEntity occurrences
       - Reference-CitationContext links
       - Author-NameVariant connections
       - NamedEntity-NERArticleScore associations
    
    The class provides detailed logging and statistics for monitoring the import
    process and diagnosing any issues that arise.
    """
    
    def __init__(self, data_manager: LiteratureDataManager, client: weaviate.Client, suppress_initial_logging: bool = False):
        """Initialize the importer with a data manager and client."""
        self.data_manager = data_manager
        self.suppress_initial_logging = suppress_initial_logging
        
        # Validate client connection before proceeding
        try:
            meta = client.get_meta()
            if not self.suppress_initial_logging:
                logger.info(f"Connected to Weaviate version: {meta.get('version', 'unknown')}")
            self.client = client
        except Exception as e:
            error_msg = f"Failed to validate Weaviate connection: {str(e)}"
            if hasattr(e, 'response'):
                error_msg += f"\nResponse: {e.response.content if hasattr(e.response, 'content') else e.response}"
            raise ConnectionError(error_msg) from e
        
        # Initialize tokenizer for text processing
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Track created UUIDs for cross-referencing
        self.created_uuids = {
            "Article": {},
            "Author": {},
            "Reference": {},
            "NamedEntity": {},
            "CitationContext": {},
            "NERArticleScore": {},
            "NameVariant": {}
        }
        
        # Map collection names to data manager attributes
        self.DATA_MANAGER_ATTRS = {
            "Article": "articles",
            "Author": "authors",
            "Reference": "references",
            "NamedEntity": "ner_objects",
            "CitationContext": "citation_contexts",
            "NERArticleScore": "ner_scores",
            "NameVariant": "name_variants"
        }
        
        # Track import statistics with all required fields
        self.import_stats = {
            collection: {
                "total": len(getattr(self.data_manager, self.DATA_MANAGER_ATTRS[collection], {})),
                "created": 0,
                "failed": 0,
                "skipped": 0,
                "retried": 0,
                "references_created": 0
            }
            for collection in MANAGED_COLLECTIONS
        }
        
        # Track compression statistics
        self.compression_stats = {
            "total_compressed": 0,
            "total_tokens_before": 0,
            "total_tokens_after": 0,
            "compression_ratios": []
        }
        
        # Initialize connection state
        self._connection_verified = True
        self._last_connection_check = time.time()

    def _verify_connection(self) -> bool:
        """Verify connection is still valid, with caching to prevent too frequent checks."""
        current_time = time.time()
        # Only check every 30 seconds unless forced
        if current_time - self._last_connection_check < 30 and self._connection_verified:
            return True
            
        try:
            self.client.get_meta()
            self._connection_verified = True
            self._last_connection_check = current_time
            return True
        except Exception as e:
            self._connection_verified = False
            logger.error(f"Lost connection to Weaviate: {str(e)}")
            return False
            
    def __enter__(self):
        """Enhanced context manager setup with connection validation."""
        try:
            if not self.suppress_initial_logging:
                logger.info("Initializing importer...")
                
            # Verify connection is still valid
            if not self._verify_connection():
                raise ConnectionError("Lost connection to Weaviate")
                
            return self
            
        except Exception as e:
            logger.error(f"Failed to initialize importer: {str(e)}")
            self._cleanup()
            raise
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Enhanced cleanup with proper error handling."""
        try:
            if exc_type is not None:
                logger.error(f"Error during import: {str(exc_val)}")
                self._cleanup()
                return False
                
            if not self.suppress_initial_logging:
                self._log_import_summary()
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return False
            
    def _cleanup(self):
        """Cleanup resources and log final state."""
        try:
            # Log final stats if available
            if hasattr(self, 'import_stats') and not self.suppress_initial_logging:
                logger.info("Import statistics at cleanup:")
                for collection, stats in self.import_stats.items():
                    if stats['total'] > 0:
                        success_rate = (stats['created'] / stats['total']) * 100
                        logger.info(f"{collection}: {stats['created']}/{stats['total']} created ({success_rate:.1f}%)")
                        if stats['failed'] > 0:
                            logger.info(f"  Failed: {stats['failed']}")
                        if stats['retried'] > 0:
                            logger.info(f"  Retried: {stats['retried']}")
                            
            # Clear any partial state
            self.created_uuids = {k: {} for k in self.created_uuids.keys()}
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            
    def _log_progress(self, collection: str, current: int, total: int, operation: str = "Processing"):
        """Enhanced progress logging with error state tracking."""
        if self.suppress_initial_logging:
            return
            
        progress = (current / total) * 100 if total > 0 else 0
        status = f"{operation} {collection}: {current}/{total} ({progress:.1f}%)"
        
        # Preserve rich progress bar if it exists
        if hasattr(self, '_progress'):
            self._progress.update(self._task_id, completed=current, description=status)
        else:
            logger.info(status)

    def _validate_cross_reference_coherence(self) -> Tuple[bool, List[str]]:
        """
        Validate that all cross-references point to valid objects of the correct type.
        Returns (is_valid, list_of_errors).
        """
        errors = []
        
        # Check Article references
        for article_id, article in self.data_manager.articles.items():
            # Check author references
            for author_uuid in article.authors:
                if not any(author.uuid == author_uuid for author in self.data_manager.authors.values()):
                    errors.append(f"Article {article_id} references non-existent Author UUID: {author_uuid}")
            
            # Check reference references
            for ref_uuid in article.references:
                if not any(ref.uuid == ref_uuid for ref in self.data_manager.references.values()):
                    errors.append(f"Article {article_id} references non-existent Reference UUID: {ref_uuid}")
                
            # Check named entity references
            for entity_uuid in article.named_entities:
                if not any(entity.uuid == entity_uuid for entity in self.data_manager.ner_objects.values()):
                    errors.append(f"Article {article_id} references non-existent NamedEntity UUID: {entity_uuid}")
        
        # Check Author references
        for author_id, author in self.data_manager.authors.items():
            # Check article references
            for article_uuid in author.articles:
                if not any(article.uuid == article_uuid for article in self.data_manager.articles.values()):
                    errors.append(f"Author {author_id} references non-existent Article UUID: {article_uuid}")
        
        # Check Reference references
        for ref_id, ref in self.data_manager.references.items():
            # Check citing article references
            for article_uuid in ref.citing_articles:
                if not any(article.uuid == article_uuid for article in self.data_manager.articles.values()):
                    errors.append(f"Reference {ref_id} references non-existent Article UUID: {article_uuid}")
        
        # Check CitationContext direct references
        for ctx_id, ctx in self.data_manager.citation_contexts.items():
            if not any(article.uuid == ctx.article_uuid for article in self.data_manager.articles.values()):
                errors.append(f"CitationContext {ctx_id} references non-existent Article UUID: {ctx.article_uuid}")
            if not any(ref.uuid == ctx.reference_uuid for ref in self.data_manager.references.values()):
                errors.append(f"CitationContext {ctx_id} references non-existent Reference UUID: {ctx.reference_uuid}")
        
        # Check NERArticleScore direct references
        for score_id, score in self.data_manager.ner_scores.items():
            if not any(article.uuid == score.article_uuid for article in self.data_manager.articles.values()):
                errors.append(f"NERArticleScore {score_id} references non-existent Article UUID: {score.article_uuid}")
            if not any(entity.uuid == score.entity_uuid for entity in self.data_manager.ner_objects.values()):
                errors.append(f"NERArticleScore {score_id} references non-existent NamedEntity UUID: {score.entity_uuid}")
        
        # Check NameVariant direct references
        for variant_id, variant in self.data_manager.name_variants.items():
            if not any(author.uuid == variant.author_uuid for author in self.data_manager.authors.values()):
                errors.append(f"NameVariant {variant_id} references non-existent Author UUID: {variant.author_uuid}")
        
        return len(errors) == 0, errors

    def import_data(self) -> bool:
        """
        Import data in phases while preserving text compression.
        
        Phases:
        1. Create all primary objects with compressed text
        2. Verify object creation
        3. Add cross-references
        
        Returns:
            bool: True if import successful, False otherwise
        """
        logger = logging.getLogger('weaviate_manager.importer')
        logger.info("Beginning phased import process...")
        
        try:
            # First validate cross-reference coherence
            logger.info("Validating cross-reference coherence...")
            is_valid, errors = self._validate_cross_reference_coherence()
            if not is_valid:
                logger.error("Cross-reference validation failed:")
                for error in errors:
                    logger.error(f"  {error}")
                return False
            logger.info("✓ Cross-references validated successfully")
            
            # Phase 1: Create Primary Objects
            logger.info("\nPhase 1: Creating primary objects...")
            
            # Process collections in dependency order
            collection_order = [
                "Article",      # Base content
                "Author",       # Independent entities
                "Reference",    # Independent entities
                "NamedEntity",  # Independent entities
                "CitationContext",  # Depends on Article/Reference
                "NERArticleScore",  # Depends on Article/NamedEntity
                "NameVariant"       # Depends on Author
            ]
            
            # First verify all collections exist
            logger.info("Verifying collections exist...")
            existing_collections = self.client.collections.list_all(simple=True)
            for collection_name in collection_order:
                if collection_name not in existing_collections:
                    logger.error(f"Collection {collection_name} does not exist")
                    return False
                logger.info(f"✓ Found collection: {collection_name}")
            
            # Create progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=None if self.suppress_initial_logging else None
            ) as progress:
                # Phase 1 progress
                phase1_total = sum(len(getattr(self.data_manager, self.DATA_MANAGER_ATTRS[col], {}))
                                 for col in collection_order)
                phase1_task = progress.add_task("Creating primary objects...", total=phase1_total)
                
                for collection_name in collection_order:
                    logger.info(f"\nProcessing {collection_name} objects...")
                    
                    # Get data manager attribute for this collection
                    data_attr = getattr(self.data_manager, self.DATA_MANAGER_ATTRS[collection_name])
                    if not data_attr:
                        logger.info(f"No data for collection {collection_name}, skipping...")
                        continue
                        
                    collection = self.client.collections.get(collection_name)
                    
                    # Get collection schema to validate properties
                    schema = collection.config.get()
                    if not schema:
                        logger.error(f"Could not get schema for collection {collection_name}")
                        return False
                        
                    logger.debug(f"Collection {collection_name} schema: {schema}")
                    
                    # Use configured batch size for better performance
                    batch_size = BATCH_SIZES.get(collection_name, BATCH_SIZES["default"])
                    logger.info(f"Using batch size {batch_size} for {collection_name}")
                    
                    # Process in batches
                    items = list(data_attr.items())
                    collection_task = progress.add_task(f"Processing {collection_name}...", total=len(items))
                    
                    for i in range(0, len(items), batch_size):
                        batch_items = items[i:i + batch_size]
                        successful_uuids = set()
                        
                        # Verify connection before each batch
                        if not self._verify_connection():
                            raise ConnectionError("Lost connection during batch processing")
                        
                        try:
                            with collection.batch.dynamic() as batch:
                                for entity_id, entity in batch_items:
                                    try:
                                        # Prepare properties with text compression
                                        properties = self._prepare_basic_properties(collection_name, entity)
                                        
                                        # Validate properties against schema
                                        logger.debug(f"Validating properties for {collection_name} {entity_id}:")
                                        logger.debug(f"Properties: {self._truncate_for_logging(json.dumps(properties, indent=2))}")
                                        
                                        # Generate consistent UUID
                                        uuid = entity.uuid
                                        
                                        # Log what we're about to add
                                        logger.debug(f"Adding {collection_name} {entity_id} with UUID {uuid}")
                                        
                                        # Add to batch without cross-references
                                        result = batch.add_object(
                                            properties=properties,
                                            uuid=uuid
                                        )
                                        
                                        if hasattr(result, 'errors') and result.errors:
                                            logger.error(f"Failed to create {collection_name} object {entity_id}:")
                                            for error in result.errors:
                                                logger.error(f"  {error}")
                                            self.import_stats[collection_name]["failed"] += 1
                                        else:
                                            self.created_uuids[collection_name][entity_id] = uuid
                                            successful_uuids.add(uuid)
                                            self.import_stats[collection_name]["created"] += 1
                                            logger.debug(f"Created {collection_name} {entity_id}")
                                        
                                    except Exception as e:
                                        error_msg = str(e)
                                        if hasattr(e, 'response'):
                                            resp = e.response
                                            if hasattr(resp, 'content'):
                                                error_msg = f"{error_msg}\nResponse: {resp.content}"
                                        logger.error(f"Failed to create {collection_name} object {entity_id}: {str(e)}")
                                        self.import_stats[collection_name]["failed"] += 1
                                        continue
                        
                        except Exception as e:
                            logger.error(f"Batch operation failed: {str(e)}")
                            if hasattr(e, 'response'):
                                logger.error(f"Response content: {e.response.content}")
                            # Don't fail immediately, try next batch
                            continue
                        
                        # Verify batch after creation
                        time.sleep(1)  # Wait for consistency
                        if not self._verify_batch_uuids(collection, successful_uuids):
                            logger.error(f"Batch verification failed for {collection_name}")
                            # If single object failed, try next one instead of failing completely
                            if batch_size == 1:
                                continue
                            return False
                        
                        # Update progress
                        progress.update(phase1_task, advance=len(batch_items))
                        progress.update(collection_task, advance=len(batch_items))
                    
                    logger.info(f"✓ Created {self.import_stats[collection_name]['created']} {collection_name} objects")
                
                # Phase 2: Verify Object Creation
                logger.info("\nPhase 2: Verifying object creation...")
                verify_task = progress.add_task("Verifying objects...", total=len(collection_order))
                
                for collection_name in collection_order:
                    collection = self.client.collections.get(collection_name)
                    expected = self.import_stats[collection_name]["created"]
                    if expected == 0:
                        progress.update(verify_task, advance=1)
                        continue
                        
                    actual = collection.aggregate.over_all().total_count
                    
                    if actual < expected:
                        raise ValueError(
                            f"Object verification failed for {collection_name}: "
                            f"Expected {expected}, found {actual}"
                        )
                    logger.info(f"✓ Verified {actual} objects in {collection_name}")
                    progress.update(verify_task, advance=1)
                
                # Phase 3: Add Cross-References
                logger.info("\nPhase 3: Adding cross-references...")
                
                # Process collections in same order for references
                for collection_name in collection_order:
                    logger.info(f"\nAdding references for {collection_name}...")
                    
                    # Get data manager attribute for this collection
                    data_attr = getattr(self.data_manager, self.DATA_MANAGER_ATTRS[collection_name])
                    if not data_attr:
                        continue
                        
                    collection = self.client.collections.get(collection_name)
                    
                    # Group all references by type for the entire collection
                    reference_groups = defaultdict(list)
                    
                    # First pass: collect all references by type
                    for entity_id, entity in data_attr.items():
                        if entity_id not in self.created_uuids[collection_name]:
                            continue
                            
                        uuid = self.created_uuids[collection_name][entity_id]
                        
                        # Collect references based on collection type
                        if collection_name == "Article":
                            if entity.authors:
                                reference_groups["authors"].append((uuid, [str(author_uuid) for author_uuid in entity.authors]))
                            if entity.references:
                                reference_groups["references"].append((uuid, [str(ref_uuid) for ref_uuid in entity.references]))
                        elif collection_name == "Author":
                            if entity.articles:
                                reference_groups["primary_articles"].append((uuid, [str(article_uuid) for article_uuid in entity.articles]))
                            if entity.authored_references:
                                reference_groups["authored_references"].append((uuid, [str(ref_uuid) for ref_uuid in entity.authored_references]))
                            if entity.name_variants:
                                reference_groups["name_variants"].append((uuid, [str(variant_uuid) for variant_uuid in entity.name_variants]))
                        elif collection_name == "Reference":
                            if entity.authors:
                                reference_groups["authors"].append((uuid, [str(author_uuid) for author_uuid in entity.authors]))
                            if entity.citing_articles:
                                reference_groups["cited_in"].append((uuid, [str(article_uuid) for article_uuid in entity.citing_articles]))
                            if entity.citation_contexts:
                                reference_groups["citation_contexts"].append((uuid, [str(context_uuid) for context_uuid in entity.citation_contexts]))
                        elif collection_name == "NamedEntity":
                            if entity.articles:
                                reference_groups["found_in"].append((uuid, list(str(article_uuid) for article_uuid in entity.articles)))
                            if entity.ner_scores:
                                reference_groups["article_scores"].append((uuid, list(str(score_uuid) for score_uuid in entity.ner_scores)))
                        elif collection_name == "CitationContext":
                            if entity.article_uuid:
                                reference_groups["article"].append((uuid, [str(entity.article_uuid)]))
                            if entity.reference_uuid:
                                reference_groups["reference"].append((uuid, [str(entity.reference_uuid)]))
                        elif collection_name == "NERArticleScore":
                            if entity.article_uuid:
                                reference_groups["article"].append((uuid, [str(entity.article_uuid)]))
                            if entity.entity_uuid:
                                reference_groups["entity"].append((uuid, [str(entity.entity_uuid)]))
                        elif collection_name == "NameVariant":
                            if entity.author_uuid:
                                reference_groups["author"].append((uuid, [str(entity.author_uuid)]))
                    
                    # Second pass: add references in batches by type
                    batch_size = 100  # Larger batch size for references
                    
                    # Calculate total references for this collection
                    total_refs_for_collection = sum(len(refs) for refs in reference_groups.values())
                    collection_task = progress.add_task(
                        f"Adding {collection_name} references...",
                        total=total_refs_for_collection
                    )
                    
                    for ref_type, refs in reference_groups.items():
                        logger.info(f"Adding {len(refs)} {ref_type} references for {collection_name}...")
                        
                        # Create progress task for this reference type
                        ref_task = progress.add_task(
                            f"Adding {collection_name} {ref_type} references...",
                            total=len(refs)
                        )
                        
                        # Process references in batches
                        for i in range(0, len(refs), batch_size):
                            batch_refs = refs[i:i + batch_size]
                            try:
                                with collection.batch.dynamic() as batch:
                                    for from_uuid, to_uuids in batch_refs:
                                        batch.add_reference(
                                            from_uuid=from_uuid,
                                            from_property=ref_type,
                                            to=to_uuids
                                        )
                            except Exception as e:
                                logger.error(f"Failed adding batch of {ref_type} references for {collection_name}: {str(e)}")
                                if hasattr(e, 'response'):
                                    logger.error(f"Response content: {e.response.content}")
                                continue
                            
                            # Update progress using rich progress bar
                            progress.update(ref_task, advance=len(batch_refs))
                            progress.update(collection_task, advance=len(batch_refs))
                    
                    logger.info(f"Completed adding references for {collection_name}")
            
            # Log final statistics
            logger.info("\nImport completed successfully")
            for collection_name in collection_order:
                stats = self.import_stats[collection_name]
                if stats["total"] > 0:
                    logger.info(f"\n{collection_name} Statistics:")
                    logger.info(f"  Created: {stats['created']}/{stats['total']} ({(stats['created']/stats['total'])*100:.1f}%)")
                    if stats["failed"] > 0:
                        logger.info(f"  Failed: {stats['failed']}")
                    if stats["references_created"] > 0:
                        logger.info(f"  References Created: {stats['references_created']}")
            
            if self.compression_stats["total_compressed"] > 0:
                avg_ratio = sum(self.compression_stats["compression_ratios"]) / len(self.compression_stats["compression_ratios"])
                logger.info(f"\nCompression Statistics:")
                logger.info(f"  Total Compressed: {self.compression_stats['total_compressed']}")
                logger.info(f"  Average Compression Ratio: {avg_ratio:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response content: {e.response.content}")
            return False

    def _get_data_manager_attr(self, collection_name: str) -> str:
        """Map collection names to data manager attributes."""
        mapping = {
            "Article": "articles",
            "Author": "authors",
            "Reference": "references",
            "NamedEntity": "ner_objects",
            "CitationContext": "citation_contexts",
            "NERArticleScore": "ner_scores",
            "NameVariant": "name_variants"
        }
        return mapping.get(collection_name, "")

    def _verify_entity_creation(self) -> bool:
        """Verify that we have created the necessary entities before adding references."""
        expected_counts = {
            # Primary entities
            "Article": len(self.data_manager.articles),
            "Author": len(self.data_manager.authors),
            "Reference": len(self.data_manager.references),
            "NamedEntity": len(self.data_manager.ner_objects),
            # Relationship entities
            "NameVariant": len(self.data_manager.name_variants),
            "CitationContext": len(self.data_manager.citation_contexts),
            "NERArticleScore": len(self.data_manager.ner_scores)
        }
        
        for collection, expected in expected_counts.items():
            created = len(self.created_uuids[collection])
            if created == 0 and expected > 0:
                logging.error(f"No {collection} entities were created (expected {expected})")
                return False
            if created < expected:
                logging.warning(f"Only created {created}/{expected} {collection} entities")
                
        return True

    def _create_entities(self, progress, task) -> bool:
        """Create all entities with their basic properties using batch processing."""
        try:
            # Define collection creation order with batch sizes
            collections_to_create = [
                # Primary entities must be created first and successfully
                ("Article", self.data_manager.articles, BATCH_SIZES["Article"]),
                ("Author", self.data_manager.authors, BATCH_SIZES["Author"]),
                ("Reference", self.data_manager.references, BATCH_SIZES["Reference"]),
                ("NamedEntity", self.data_manager.ner_objects, BATCH_SIZES["NamedEntity"]),
                # Relationship entities depend on primary entities
                ("NameVariant", self.data_manager.name_variants, BATCH_SIZES["NameVariant"]),
                ("CitationContext", self.data_manager.citation_contexts, BATCH_SIZES["CitationContext"]),
                ("NERArticleScore", self.data_manager.ner_scores, BATCH_SIZES["NERArticleScore"])
            ]
            
            logger = logging.getLogger('weaviate_manager.importer')
            
            # Create each collection's entities in batches
            for collection_name, entities, batch_size in collections_to_create:
                if not entities:
                    logger.info(f"No {collection_name} entities to create")
                    continue
                    
                collection = self.client.collections.get(collection_name)
                total_entities = len(entities)
                created_count = 0
                failed_count = 0
                
                # Process entities in batches
                current_batch = []
                for entity_id, entity in entities.items():
                    # Verify entity has required UUID
                    if not hasattr(entity, 'uuid'):
                        logger.error(f"{collection_name} {entity_id} missing UUID")
                        failed_count += 1
                        continue
                        
                    try:
                        current_batch.append((entity_id, entity))
                        
                        # Process batch if it reaches the size limit
                        if len(current_batch) >= batch_size:
                            success = self._process_batch(collection, current_batch)
                            if success:
                                created_count += len(current_batch)
                            else:
                                # For primary entities, we must succeed
                                if collection_name in ["Article", "Author", "Reference", "NamedEntity"]:
                                    logger.error(f"Critical failure creating {collection_name} batch - stopping import")
                                    return False
                                failed_count += len(current_batch)
                            progress.advance(task, len(current_batch))
                            current_batch = []
                            
                    except Exception as e:
                        logger.error(f"Error preparing {collection_name} {entity_id}: {str(e)}")
                        failed_count += 1
                        progress.advance(task)
                        continue
                
                # Process any remaining entities in the final batch
                if current_batch:
                    success = self._process_batch(collection, current_batch)
                    if success:
                        created_count += len(current_batch)
                    else:
                        if collection_name in ["Article", "Author", "Reference", "NamedEntity"]:
                            logger.error(f"Critical failure creating final {collection_name} batch - stopping import")
                            return False
                        failed_count += len(current_batch)
                    progress.advance(task, len(current_batch))
                
                # Verify creation counts match expectations
                logger.info(f"\n{collection_name} Creation Summary:")
                logger.info(f"Expected: {total_entities}")
                logger.info(f"Created:  {created_count}")
                logger.info(f"Failed:   {failed_count}")
                
                # For primary entities, we must have created all expected entities
                if collection_name in ["Article", "Author", "Reference", "NamedEntity"]:
                    if created_count < total_entities:
                        logger.error(f"Failed to create all {collection_name} entities - stopping import")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating entities: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _process_batch(self, collection: weaviate.collections.Collection, batch: List[Tuple[str, Any]], batch_size: int = 10) -> bool:
        """Process a batch of entities with enhanced error handling and verification."""
        try:
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Verify connection before processing batch
                    if not self._verify_connection():
                        raise ConnectionError("Lost connection during batch processing")
                        
                    # Use legacy batch API for Weaviate 1.28.3
                    with self.client.batch as batch_processor:
                        batch_processor.batch_size = batch_size
                        failed_objects = []
                        successful_uuids = set()
                        
                        # Clear any previously tracked UUIDs for this batch
                        for entity_id, entity in batch:
                            if entity_id in self.created_uuids[collection.name]:
                                del self.created_uuids[collection.name][entity_id]
                        
                        for entity_id, entity in batch:
                            try:
                                # Verify entity has required UUID
                                if not hasattr(entity, 'uuid'):
                                    error_msg = f"Entity {entity_id} missing UUID"
                                    logger.error(error_msg)
                                    failed_objects.append((entity_id, [error_msg]))
                                    continue

                                # Prepare basic properties without references
                                properties = self._prepare_basic_properties(collection.name, entity)
                                logger.debug(f"Adding {collection.name} {entity_id} to batch with properties: {properties}")

                                # Use legacy batch add_data_object
                                result = batch_processor.add_data_object(
                                    data_object=properties,
                                    class_name=collection.name,
                                    uuid=entity.uuid
                                )
                                
                                # Track the UUID for verification
                                self.created_uuids[collection.name][entity_id] = entity.uuid
                                successful_uuids.add(entity.uuid)
                                self.import_stats[collection.name]["created"] += 1
                                logger.debug(f"Created {collection.name} {entity_id}")
                                    
                            except Exception as e:
                                error_msg = str(e)
                                if hasattr(e, 'response'):
                                    resp = e.response
                                    if hasattr(resp, 'content'):
                                        error_msg = f"{error_msg}\nResponse: {resp.content}"
                                logger.error(f"Failed to create {collection.name} {entity_id}:\n{error_msg}")
                                failed_objects.append((entity_id, [error_msg]))
                                self.import_stats[collection.name]["failed"] += 1
                
                        # Wait for consistency
                        time.sleep(1)
                        
                        # Verify the batch
                        if successful_uuids:
                            if not self._verify_batch_uuids(collection, successful_uuids):
                                if retry_count < max_retries - 1:
                                    logger.warning(f"Batch verification failed, retrying (attempt {retry_count + 2}/{max_retries})")
                                    retry_count += 1
                                    self.import_stats[collection.name]["retried"] += len(successful_uuids)
                                    # Clean up tracking on retry
                                    for entity_id, _ in batch:
                                        if entity_id in self.created_uuids[collection.name]:
                                            del self.created_uuids[collection.name][entity_id]
                                    continue
                                else:
                                    logger.error("Batch verification failed after all retries")
                                    return False
                
                        if failed_objects:
                            error_msg = f"\nBatch processing failed for {collection.name}:"
                            error_msg += f"\nFailed: {len(failed_objects)}/{len(batch)} objects"
                            for entity_id, errors in failed_objects:
                                error_msg += f"\n  {entity_id}:"
                                for error in errors:
                                    error_msg += f"\n    {error}"
                            logger.error(error_msg)
                            return False
                        
                        return True
                        
                except Exception as e:
                    error_msg = str(e)
                    if hasattr(e, 'response'):
                        resp = e.response
                        if hasattr(resp, 'content'):
                            error_msg = f"{error_msg}\nResponse: {resp.content}"
                    logger.error(f"Error processing batch: {error_msg}")
                    
                    if retry_count < max_retries - 1:
                        retry_count += 1
                        logger.info(f"Retrying batch for {collection.name} (attempt {retry_count + 1}/{max_retries})")
                        # Clean up tracking on retry
                        for entity_id, _ in batch:
                            if entity_id in self.created_uuids[collection.name]:
                                del self.created_uuids[collection.name][entity_id]
                        continue
                    
                    return False
                    
            return False
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return False
            
    def _verify_batch_uuids(self, collection: weaviate.collections.Collection, expected_uuids: Set[str]) -> bool:
        """Verify that all expected UUIDs exist in the collection with pagination."""
        try:
            # Build filters for UUIDs using proper Filter class
            uuid_filters = [Filter.by_id().equal(uuid) for uuid in expected_uuids]
            if not uuid_filters:
                logger.debug("No UUIDs to verify")
                return True
                
            filters = Filter.any_of(uuid_filters)
            
            # Log what we're looking for
            logger.debug(f"Verifying {len(expected_uuids)} UUIDs")
            
            # Query for objects with pagination
            found_uuids = set()
            offset = 0
            limit = 25  # Weaviate's default limit
            
            while True:
                # Query with current offset and filter for only our expected UUIDs
                result = collection.query.fetch_objects(
                    limit=limit,
                    offset=offset,
                    include_vector=False,
                    filters=filters
                )

                if not result or not hasattr(result, 'objects') or not result.objects:
                    break

                # Process this batch of results
                current_batch = result.objects
                logger.debug(f"Found {len(current_batch)} objects in response (offset={offset})")
                
                for obj in current_batch:
                    if hasattr(obj, 'uuid'):
                        # Convert Weaviate UUID to string for comparison
                        found_uuids.add(str(obj.uuid))
                    else:
                        logger.error(f"Object missing uuid attribute: {obj}")
                
                # If we got fewer results than limit, we're done
                if len(current_batch) < limit:
                    break
                    
                # Move to next page
                offset += limit

            # Convert expected UUIDs to strings if they aren't already
            expected_uuids_str = {str(uuid) for uuid in expected_uuids}
            
            # Check for missing UUIDs
            missing_uuids = expected_uuids_str - found_uuids
            if missing_uuids:
                logger.error(f"Missing UUIDs: {missing_uuids}")
                # Log a few examples for debugging
                for uuid in list(missing_uuids)[:3]:
                    logger.debug(f"Example missing UUID: {uuid}")
                return False

            # Since we filtered the query, we shouldn't need to check for unexpected UUIDs
            # as only objects matching our filter would be returned

            logger.info(f"Successfully verified {len(found_uuids)} UUIDs")
            return True

        except Exception as e:
            logger.error(f"Error verifying batch: {str(e)}")
            if hasattr(e, 'response'):
                logger.error(f"Response content: {e.response.content}")
            return False

    def _truncate_for_logging(self, obj: Any, max_length: int = 20) -> str:
        """Truncate object representation for logging.
        
        Args:
            obj: Object to truncate
            max_length: Maximum length of the string representation
            
        Returns:
            str: Truncated string representation
        """
        s = str(obj)
        if len(s) <= max_length:
            return s
        return f"{s[:max_length]}... [truncated, total length: {len(s)}]"

    def _get_safe_token_limit(self) -> int:
        """
        Get the safe token limit for individual properties.
        
        Calculates a safe token limit using:
        - Base token limit from model configuration
        - Safety margin to prevent overflows
        - Rounding to ensure integer values
        
        Returns:
            int: Maximum safe token count for a single property
        """
        return int(DEFAULT_MAX_TOKENS * TOKEN_SAFETY_MARGIN)

    def _calculate_compression_ratio(self, current_tokens: int, safe_limit: int) -> float:
        """
        Calculate required compression ratio to fit within safe limit.
        
        The compression ratio is calculated as:
        ratio = safe_limit / current_tokens
        
        A ratio of:
        - 1.0 means no compression needed
        - <1.0 means compression required (e.g., 0.5 = reduce by half)
        - >1.0 is not possible (would mean expansion)
        
        Args:
            current_tokens: Current number of tokens in text
            safe_limit: Maximum safe token limit
            
        Returns:
            float: Compression ratio to apply (1.0 if no compression needed)
        """
        if current_tokens <= safe_limit:
            return 1.0
        return safe_limit / current_tokens

    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score for text content.
        
        Scores text based on:
        - Presence of key scientific terms
        - Presence of comparison terms
        - Density of numerical content
        - Presence of citations
        - Position in text (favors first and last paragraphs)
        
        Returns:
            float: Importance score (higher is more important)
        """
        score = 1.0  # Base score
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.strip().lower()
        
        # Score key terms
        for term in IMPORTANCE_KEY_TERMS:
            if term in text_lower:
                score += IMPORTANCE_WEIGHTS["key_terms"]
        
        # Score comparison terms
        for term in IMPORTANCE_COMPARISON_TERMS:
            if term in text_lower:
                score += IMPORTANCE_WEIGHTS["comparisons"]
        
        # Score numerical content
        num_count = sum(1 for c in text if c.isdigit())
        score += (num_count / len(text)) * IMPORTANCE_WEIGHTS["numbers"] * 100
        
        # Score citations (rough heuristic for parenthetical citations)
        citation_count = text.count("(") + text.count("[")
        score += citation_count * IMPORTANCE_WEIGHTS["citations"]
        
        return score

    def _process_text(self, text: str, field_name: str, force_ratio: Optional[float] = None) -> str:
        """
        Process text ensuring it fits within safe token limits.
        
        This method handles text compression in multiple stages:
        1. Count tokens in input text
        2. Return as-is if under limit and no force_ratio
        3. Calculate required compression ratio
        4. Apply intelligent compression preserving important content
        5. Force truncate if still over limit
        
        The compression process preserves:
        - Important scientific content
        - Document structure
        - Key information based on position
        
        Args:
            text: Text content to process
            field_name: Name of the field being processed
            force_ratio: Optional ratio to force compression regardless of current size
            
        Returns:
            str: Processed text guaranteed to be under token limit
            
        Side Effects:
            - Updates compression statistics
            - Logs compression details
        """
        if not text:
            return ""
        
        safe_limit = self._get_safe_token_limit()
        tokens = self._tokenizer.encode(text)
        token_count = len(tokens)
        
        logger.debug(f"Processing {field_name}: {token_count} tokens (limit: {safe_limit})")
        
        # If we're under the limit and no force_ratio, return as is
        if token_count <= safe_limit and not force_ratio:
            return text
            
        # Calculate required compression ratio
        compression_ratio = force_ratio if force_ratio is not None else self._calculate_compression_ratio(token_count, safe_limit)
        target_tokens = int(token_count * compression_ratio) if force_ratio else min(token_count, safe_limit)
        
        logger.debug(f"Compressing {field_name} with ratio {compression_ratio:.2f} "
                    f"({token_count} -> {target_tokens} tokens)")
        
        # First try intelligent compression
        compressed = self._reduce_text_to_token_limit(
            text=text,
            target_tokens=target_tokens,
            compression_ratio=compression_ratio,
            field_name=field_name
        )
        
        # Verify and force truncate if still over limit
        final_tokens = len(self._tokenizer.encode(compressed))
        if final_tokens > target_tokens:
            logger.warning(f"{field_name} still over limit after compression, truncating "
                          f"({final_tokens} -> {target_tokens})")
            compressed = self._truncate_to_tokens(compressed, target_tokens)
            final_tokens = len(self._tokenizer.encode(compressed))
            logger.debug(f"{field_name} final token count after truncation: {final_tokens}")
            
        # Update compression statistics
        self.compression_stats["total_compressed"] += 1
        self.compression_stats["total_tokens_before"] += token_count
        self.compression_stats["total_tokens_after"] += final_tokens
        self.compression_stats["compression_ratios"].append(final_tokens / token_count)
        
        return compressed

    def _reduce_text_to_token_limit(
        self, 
        text: str, 
        target_tokens: int,
        compression_ratio: float,
        field_name: str
    ) -> str:
        """
        Intelligently reduce text using calculated compression ratio.
        
        This method implements a sophisticated compression strategy:
        1. Split text into paragraphs
        2. Score each paragraph for importance
        3. Apply position-based weighting
        4. Select and compress paragraphs to meet target
        
        Importance Scoring:
        - Scientific term presence
        - Comparison term presence
        - Numerical content density
        - Citation presence
        - Position in text
        
        Position Weighting:
        - First paragraph gets 2.0x weight
        - Last paragraph gets 1.5x weight
        - Middle paragraphs weighted by content
        
        Args:
            text: Text to reduce
            target_tokens: Target token count
            compression_ratio: Required compression ratio
            field_name: Name of field being processed
            
        Returns:
            str: Compressed text meeting token limit while preserving important content
        """
        current_tokens = self._estimate_tokens(text)
        if current_tokens <= target_tokens:
            return text
        
        # Split into sections
        paragraphs = text.split('\n\n')
        if not paragraphs:
            return text
        
        # Score and select paragraphs
        scored_paragraphs = []
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            
            # Calculate importance score
            base_score = self._calculate_importance_score(para)
            
            # Position bias: favor first and last paragraphs
            position_score = 0
            if i == 0:  # First paragraph
                position_score = 2.0
            elif i == len(paragraphs) - 1:  # Last paragraph
                position_score = 1.5
            
            final_score = base_score + position_score
            para_tokens = self._estimate_tokens(para)
            scored_paragraphs.append((final_score, para, para_tokens))
        
        # Sort by importance score
        scored_paragraphs.sort(reverse=True)
        
        # Calculate target length for each paragraph based on compression ratio
        reduced_text = []
        remaining_tokens = target_tokens
        
        # Always try to include first paragraph
        if scored_paragraphs:
            first_para = scored_paragraphs[0][1]
            first_tokens = self._estimate_tokens(first_para)
            first_target = int(first_tokens * compression_ratio)
            if first_target <= remaining_tokens:
                if first_tokens > first_target:
                    first_para = self._truncate_to_tokens(first_para, first_target)
                reduced_text.append(first_para)
                remaining_tokens -= self._estimate_tokens(first_para)
                scored_paragraphs = scored_paragraphs[1:]
        
        # Process remaining paragraphs
        for score, para, para_tokens in scored_paragraphs:
            # Calculate target tokens for this paragraph based on importance
            importance_factor = min(1.2, score / scored_paragraphs[0][0])  # Cap at 120%
            para_target = int(para_tokens * compression_ratio * importance_factor)
            
            if para_target <= remaining_tokens:
                if para_tokens > para_target:
                    para = self._truncate_to_tokens(para, para_target)
                reduced_text.append(para)
                remaining_tokens -= self._estimate_tokens(para)
            else:
                # Try to fit a smaller portion if high importance
                if score > 1.5 and remaining_tokens > 50:
                    truncated = self._truncate_to_tokens(para, remaining_tokens)
                    reduced_text.append(truncated)
                break
        
        return '\n\n'.join(reduced_text)

    def _truncate_to_tokens(self, text: str, max_tokens: int, from_end: bool = False) -> str:
        """Truncate text to fit within token limit."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
            
        if from_end:
            truncated_tokens = tokens[-max_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]
            
        return self._tokenizer.decode(truncated_tokens)

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Uses the cl100k_base tokenizer (same as text-embedding-3-large model)
        for accurate token counting.
        
        Args:
            text: Text to count tokens in
        
        Returns:
            int: Estimated number of tokens
        """
        if not text:
            return 0
        return len(self._tokenizer.encode(text))

    @staticmethod
    def _batch_iterator(items: Iterator[Tuple[str, Any]], batch_size: int) -> Iterator[List[Tuple[str, Any]]]:
        """
        Create batches from an iterator.
        
        Args:
            items: Iterator of items to batch
            batch_size: Size of each batch
            
        Returns:
            Iterator[List]: Batches of items
        """
        batch = []
        for item in items:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        
    def _log_import_summary(self) -> None:
        """Log detailed summary of the import process at debug level."""
        if self.suppress_initial_logging:
            return
            
        logging.debug("\nImport Summary:")
        
        # Log compression statistics if any compression occurred
        if self.compression_stats["total_compressed"] > 0:
            avg_ratio = sum(self.compression_stats["compression_ratios"]) / len(self.compression_stats["compression_ratios"])
            logging.debug(f"\nCompression Statistics:")
            logging.debug(f"Total fields compressed: {self.compression_stats['total_compressed']}")
            logging.debug(f"Average compression ratio: {avg_ratio:.2f}")
            logging.debug(f"Total tokens before: {self.compression_stats['total_tokens_before']}")
            logging.debug(f"Total tokens after: {self.compression_stats['total_tokens_after']}")
        
        # Log import statistics for each collection
        for collection in MANAGED_COLLECTIONS:
            stats = self.import_stats[collection]
            if stats["total"] > 0:
                logging.debug(f"\n{collection}:")
                logging.debug(f"Total: {stats['total']}")
                logging.debug(f"Created: {stats['created']} ({stats['created']/stats['total']:.1%})")
                if stats["failed"] > 0:
                    logging.debug(f"Failed: {stats['failed']} ({stats['failed']/stats['total']:.1%})")
                if stats["references_created"] > 0:
                    logging.debug(f"References Created: {stats['references_created']}")
        
        # Log data manager statistics
        logging.debug("\nData Manager Statistics:")
        logging.debug(f"  Articles: {len(self.data_manager.articles)}")
        logging.debug(f"  Authors: {len(self.data_manager.authors)}")
        logging.debug(f"  References: {len(self.data_manager.references)}")
        logging.debug(f"  NER Objects: {len(self.data_manager.ner_objects)}")
        logging.debug(f"  Citation Contexts: {len(self.data_manager.citation_contexts)}")
        logging.debug(f"  Name Variants: {len(self.data_manager.name_variants)}")
        logging.debug(f"  NER Scores: {len(self.data_manager.ner_scores)}")

    def _log_entity_import_summary(self, entity_type: str) -> None:
        """
        Log detailed summary for a specific entity type's import process.
        
        Logs the following metrics:
        - Total entities processed
        - Success rate (percentage created)
        - Failure rate with reasons
        - Skip rate with reasons
        - Retry attempts if any
        - Reference creation count if applicable
        
        Args:
            entity_type: Type of entity to summarize (e.g., "Article", "Author")
        
        Side Effects:
            - Writes detailed logs at INFO level
        """
        stats = self.import_stats[entity_type]
        logging.info(f"\n{entity_type} Import Summary:")
        logging.info(f"Total: {stats['total']}")
        logging.info(f"Created: {stats['created']} ({stats['created']/stats['total']:.1%})")
        logging.info(f"Failed: {stats['failed']} ({stats['failed']/stats['total']:.1%})")
        logging.info(f"Skipped: {stats['skipped']} ({stats['skipped']/stats['total']:.1%})")
        if stats['retried'] > 0:
            logging.info(f"Retried: {stats['retried']}")
        if stats['references_created'] > 0:
            logging.info(f"References Created: {stats['references_created']}")

    def _log_failed_object(self, failed) -> None:
        """
        Log detailed information about a failed import object.
        
        Logs the following details:
        - Object UUID if available
        - Error messages from Weaviate
        - Error paths indicating problematic properties
        - Full properties of failed object (for debugging)
        
        Args:
            failed: Failed object from Weaviate batch operation
        
        Side Effects:
            - Writes error logs with detailed failure information
        """
        logging.error("\nFailed object details:")
        if hasattr(failed, 'id'):
            logging.error(f"UUID: {failed.id}")
        if hasattr(failed, 'errors'):
            for error in failed.errors:
                if hasattr(error, 'message'):
                    logging.error(f"Error message: {error.message}")
                if hasattr(error, 'path'):
                    logging.error(f"Error path: {error.path}")
        if hasattr(failed, 'properties'):
            logging.error(f"Failed properties: {json.dumps(failed.properties, indent=2)}")

    def _log_subset_extraction(self, subset_data: Dict[str, Set[str]]) -> None:
        """Log the results of subset extraction."""
        logging.info("\nExtracted connected subset:")
        for collection, subset in subset_data.items():
            total = len(self.created_uuids.get(collection, {}))
            logging.info(f"  {collection}: {len(subset)}/{total}")
            
            # Add debug logging for NER Scores
            if collection == "NERArticleScore":
                logging.debug(f"NER Score UUIDs in subset: {list(subset)[:5]}")  # Log first 5 for debugging
                logging.debug(f"Total NER Score UUIDs: {list(self.created_uuids[collection].keys())[:5]}")  # Log first 5 for comparison

    def _import_articles(self) -> bool:
        """Import Article entities with their properties."""
        try:
            collection = self.client.collections.get("Article")
            logger = logging.getLogger('weaviate_manager.importer')
            logger.info("Beginning article import with compression...")
            
            batch_errors = []
            with collection.batch.dynamic() as batch:
                for article_id, article in self.data_manager.articles.items():
                    try:
                        # Log article being processed
                        logger.info(f"\nProcessing article: {article.filename}")
                        
                        try:
                            # Create object with properties
                            properties = self._prepare_basic_properties("Article", article)
                            
                            # Verify properties before adding to batch
                            for key, value in properties.items():
                                if isinstance(value, str) and len(value) > MAX_STRING_LENGTH:
                                    logger.warning(f"Property {key} exceeds MAX_STRING_LENGTH, truncating")
                                    properties[key] = value[:MAX_STRING_LENGTH]
                            
                        except Exception as e:
                            logger.error(f"Failed to prepare properties for article {article_id}: {str(e)}")
                            import traceback
                            logger.error(f"Property preparation traceback: {traceback.format_exc()}")
                            self.import_stats["Article"]["failed"] += 1
                            continue
                        
                        try:
                            # Add to batch with UUID
                            result = batch.add_object(
                                properties=properties,
                                uuid=article.uuid
                            )
                            
                            # Check for batch operation errors
                            if hasattr(result, 'errors') and result.errors:
                                error_msg = f"Batch operation failed for article {article_id}:"
                                for error in result.errors:
                                    error_msg += f"\n  - {error}"
                                logger.error(error_msg)
                                batch_errors.append((article_id, error_msg))
                                self.import_stats["Article"]["failed"] += 1
                                continue
                                
                            # Track created UUID
                            self.created_uuids["Article"][article_id] = article.uuid
                            self.import_stats["Article"]["created"] += 1
                            
                            # Log success
                            logger.debug(f"Successfully added article {article_id} to batch")
                            
                        except Exception as e:
                            error_msg = f"Failed to add article {article_id} to batch: {str(e)}"
                            if hasattr(e, 'response'):
                                resp = e.response
                                if hasattr(resp, 'content'):
                                    error_msg += f"\nResponse content: {resp.content}"
                            logger.error(error_msg)
                            import traceback
                            logger.error(f"Batch addition traceback: {traceback.format_exc()}")
                            batch_errors.append((article_id, error_msg))
                            self.import_stats["Article"]["failed"] += 1
                            continue
                        
                    except Exception as e:
                        logger.error(f"Failed to process article {article_id}: {str(e)}")
                        import traceback
                        logger.error(f"Article processing traceback: {traceback.format_exc()}")
                        self.import_stats["Article"]["failed"] += 1
            
            # Log batch operation summary
            if batch_errors:
                logger.error("\nBatch import errors summary:")
                for article_id, error in batch_errors:
                    logger.error(f"Article {article_id}:")
                    logger.error(f"  {error}")
                return False
            
            # Log overall compression statistics
            if self.compression_stats["total_compressed"] > 0:
                logger.info("\nOverall compression statistics:")
                logger.info(f"Total fields compressed: {self.compression_stats['total_compressed']}")
                logger.info(f"Total tokens before: {self.compression_stats['total_tokens_before']}")
                logger.info(f"Total tokens after: {self.compression_stats['total_tokens_after']}")
                if self.compression_stats["compression_ratios"]:
                    avg_ratio = sum(self.compression_stats["compression_ratios"]) / len(self.compression_stats["compression_ratios"])
                    logger.info(f"Average compression ratio: {avg_ratio:.2%}")
            else:
                logger.info("\nNo compression was needed for any articles")
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing articles: {str(e)}")
            import traceback
            logger.error(f"Import error traceback: {traceback.format_exc()}")
            return False

    def _import_authors(self) -> bool:
        """Import Author entities with their properties."""
        try:
            collection = self.client.collections.get("Author")
            
            with collection.batch.dynamic() as batch:
                for author_id, author in self.data_manager.authors.items():
                    try:
                        # Create object with properties
                        properties = {
                            "canonical_name": author.canonical_name,
                            "email": author.email
                        }
                        
                        # Add to batch with UUID
                        batch.add_object(
                            properties=properties,
                            uuid=author.uuid
                        )
                        
                        # Track created UUID
                        self.created_uuids["Author"][author_id] = author.uuid
                        self.import_stats["Author"]["created"] += 1
                        
                    except Exception as e:
                        logging.error(f"Failed to import author {author_id}: {str(e)}")
                        self.import_stats["Author"]["failed"] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error importing authors: {str(e)}")
            return False

    def _import_references(self) -> bool:
        """Import Reference entities with their properties."""
        try:
            collection = self.client.collections.get("Reference")
            
            with collection.batch.dynamic() as batch:
                for ref_id, ref in self.data_manager.references.items():
                    try:
                        # Create object with properties
                        properties = {
                            "title": ref.title,
                            "journal": ref.journal,
                            "volume": ref.volume,
                            "pages": ref.pages,
                            "publication_date": ref.publication_date,
                            "raw_reference": ref.raw_reference
                        }
                        
                        # Add to batch with UUID
                        batch.add_object(
                            properties=properties,
                            uuid=ref.uuid
                        )
                        
                        # Track created UUID
                        self.created_uuids["Reference"][ref_id] = ref.uuid
                        self.import_stats["Reference"]["created"] += 1
                        
                    except Exception as e:
                        logging.error(f"Failed to import reference {ref_id}: {str(e)}")
                        self.import_stats["Reference"]["failed"] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error importing references: {str(e)}")
            return False

    def _import_entities(self) -> bool:
        """Import NamedEntity entities with their properties."""
        try:
            collection = self.client.collections.get("NamedEntity")
            
            with collection.batch.dynamic() as batch:
                for entity_id, entity in self.data_manager.ner_objects.items():
                    try:
                        # Create object with properties
                        properties = {
                            "name": entity.name,
                            "type": entity.type
                        }
                        
                        # Add to batch with UUID
                        batch.add_object(
                            properties=properties,
                            uuid=entity.uuid
                        )
                        
                        # Track created UUID
                        self.created_uuids["NamedEntity"][entity_id] = entity.uuid
                        self.import_stats["NamedEntity"]["created"] += 1
                        
                    except Exception as e:
                        logging.error(f"Failed to import entity {entity_id}: {str(e)}")
                        self.import_stats["NamedEntity"]["failed"] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error importing entities: {str(e)}")
            return False

    def _import_contexts(self) -> bool:
        """Import CitationContext entities with their properties."""
        try:
            collection = self.client.collections.get("CitationContext")
            
            with collection.batch.dynamic() as batch:
                for ctx_id, ctx in self.data_manager.citation_contexts.items():
                    try:
                        # Create object with properties
                        properties = {
                            "section": ctx.section,
                            "local_ref_id": ctx.local_ref_id
                        }
                        
                        # Add to batch with UUID
                        batch.add_object(
                            properties=properties,
                            uuid=ctx.uuid
                        )
                        
                        # Track created UUID
                        self.created_uuids["CitationContext"][ctx_id] = ctx.uuid
                        self.import_stats["CitationContext"]["created"] += 1
                        
                    except Exception as e:
                        logging.error(f"Failed to import citation context {ctx_id}: {str(e)}")
                        self.import_stats["CitationContext"]["failed"] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error importing citation contexts: {str(e)}")
            return False

    def _import_scores(self) -> bool:
        """Import NERArticleScore entities with their properties."""
        try:
            collection = self.client.collections.get("NERArticleScore")
            
            with collection.batch.dynamic() as batch:
                for score_id, score in self.data_manager.ner_scores.items():
                    try:
                        # Create object with properties
                        properties = {
                            "score": score.score
                        }
                        
                        # Add to batch with UUID
                        batch.add_object(
                            properties=properties,
                            uuid=score.uuid
                        )
                        
                        # Track created UUID
                        self.created_uuids["NERArticleScore"][score_id] = score.uuid
                        self.import_stats["NERArticleScore"]["created"] += 1
                        
                    except Exception as e:
                        logging.error(f"Failed to import NER score {score_id}: {str(e)}")
                        self.import_stats["NERArticleScore"]["failed"] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error importing NER scores: {str(e)}")
            return False

    def _import_variants(self) -> bool:
        """Import NameVariant entities with their properties."""
        try:
            collection = self.client.collections.get("NameVariant")
            
            with collection.batch.dynamic() as batch:
                for variant_id, variant in self.data_manager.name_variants.items():
                    try:
                        # Create object with properties
                        properties = {
                            "name": variant.name
                        }
                        
                        # Add to batch with UUID
                        batch.add_object(
                            properties=properties,
                            uuid=variant.uuid
                        )
                        
                        # Track created UUID
                        self.created_uuids["NameVariant"][variant_id] = variant.uuid
                        self.import_stats["NameVariant"]["created"] += 1
                        
                    except Exception as e:
                        logging.error(f"Failed to import name variant {variant_id}: {str(e)}")
                        self.import_stats["NameVariant"]["failed"] += 1
            
            return True
            
        except Exception as e:
            logging.error(f"Error importing name variants: {str(e)}")
            return False

    def _format_reference(self, collection_name: str, uuid: str) -> str:
        """Format a reference UUID into Weaviate's expected format."""
        # Just return the UUID string - Weaviate's batch API expects simple UUIDs
        return str(uuid)

    def _add_article_references(self, batch, uuid, article) -> bool:
        """Add all relationships for Article entities."""
        try:
            # Add author references
            if article.authors:
                try:
                    # Pass list of UUIDs directly
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="authors",
                        to=[str(author_uuid) for author_uuid in article.authors]
                    )
                except Exception as e:
                    logger.error(f"Failed adding author references for article: {str(e)}")
            
            # Add reference references
            if article.references:
                try:
                    # Pass list of UUIDs directly
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="references",
                        to=[str(ref_uuid) for ref_uuid in article.references]
                    )
                except Exception as e:
                    logger.error(f"Failed adding reference citations for article: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding article references: {str(e)}")
            return False

    def _add_author_references(self, batch, uuid, author) -> bool:
        """Add all relationships for Author entities."""
        try:
            # Add article references
            if author.articles:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="primary_articles",
                        to=[str(article_uuid) for article_uuid in author.articles]
                    )
                except Exception as e:
                    logger.error(f"Failed adding article references for author: {str(e)}")
            
            # Add reference references
            if author.authored_references:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="authored_references",
                        to=[str(ref_uuid) for ref_uuid in author.authored_references]
                    )
                except Exception as e:
                    logger.error(f"Failed adding reference references for author: {str(e)}")
            
            # Add name variant references
            if author.name_variants:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="name_variants",
                        to=[str(variant_uuid) for variant_uuid in author.name_variants]
                    )
                except Exception as e:
                    logger.error(f"Failed adding name variant references for author: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding author references: {str(e)}")
            return False

    def _add_reference_references(self, batch, uuid, reference) -> bool:
        """Add all relationships for Reference entities."""
        try:
            # Add author references
            if reference.authors:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="authors",
                        to=[str(author_uuid) for author_uuid in reference.authors]
                    )
                except Exception as e:
                    logger.error(f"Failed adding author references for reference: {str(e)}")
            
            # Add citing article references
            if reference.citing_articles:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="cited_in",
                        to=[str(article_uuid) for article_uuid in reference.citing_articles]
                    )
                except Exception as e:
                    logger.error(f"Failed adding citing article references for reference: {str(e)}")
            
            # Add citation context references
            if reference.citation_contexts:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="citation_contexts",
                        to=[str(context_uuid) for context_uuid in reference.citation_contexts]
                    )
                except Exception as e:
                    logger.error(f"Failed adding citation context references for reference: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding reference references: {str(e)}")
            return False

    def _add_entity_references(self, batch, uuid, entity) -> bool:
        """Add all relationships for NamedEntity entities."""
        try:
            # Add article references
            if entity.articles:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="found_in",
                        to=list(entity.articles)
                    )
                except Exception as e:
                    logger.error(f"Failed adding article references for entity: {str(e)}")
            
            # Add score references
            if entity.ner_scores:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="article_scores",
                        to=list(entity.ner_scores)
                    )
                except Exception as e:
                    logger.error(f"Failed adding score references for entity: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding entity references: {str(e)}")
            return False

    def _add_citation_context_references(self, batch, uuid, context) -> bool:
        """Add all relationships for CitationContext entities."""
        try:
            # Add article reference
            if context.article_uuid:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="article",
                        to=context.article_uuid
                    )
                except Exception as e:
                    logger.error(f"Failed adding article reference for context: {str(e)}")
            
            # Add reference reference
            if context.reference_uuid:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="reference",
                        to=context.reference_uuid
                    )
                except Exception as e:
                    logger.error(f"Failed adding reference reference for context: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding citation context references: {str(e)}")
            return False

    def _add_score_references(self, batch, uuid, score) -> bool:
        """Add all relationships for NERArticleScore entities."""
        try:
            # Add entity reference
            if score.entity_uuid:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="entity",
                        to=score.entity_uuid
                    )
                except Exception as e:
                    logger.error(f"Failed adding entity reference for score: {str(e)}")
            
            # Add article reference
            if score.article_uuid:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="article",
                        to=score.article_uuid
                    )
                except Exception as e:
                    logger.error(f"Failed adding article reference for score: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding NER score references: {str(e)}")
            return False

    def _add_name_variant_references(self, batch, uuid, variant) -> bool:
        """Add all relationships for NameVariant entities."""
        try:
            # Add author reference
            if variant.author_uuid:
                try:
                    batch.add_reference(
                        from_uuid=uuid,
                        from_property="author",
                        to=variant.author_uuid
                    )
                except Exception as e:
                    logger.error(f"Failed adding author reference for variant: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding name variant references: {str(e)}")
            return False

    def _verify_cross_references(self) -> bool:
        """Verify cross-references were correctly established."""
        try:
            # Verify each collection has expected data
            for collection_name in ["Article", "Author", "Reference", "NamedEntity", "CitationContext", "NERArticleScore", "NameVariant"]:
                collection = self.client.collections.get(collection_name)
                objects = collection.query.fetch_objects(limit=1, include_vector=False)
                if not objects or not objects.objects:
                    logging.error(f"❌ No {collection_name} objects found")
                    return False

            # Verify cross-references are accessible
            article_collection = self.client.collections.get("Article")
            articles = article_collection.query.fetch_objects(limit=1, include_vector=False)
            if articles and articles.objects:
                article = article_collection.query.fetch_object_by_id(
                    uuid=articles.objects[0].uuid,
                    return_references=[
                        QueryReference(link_on="authors", return_properties=["canonical_name"]),
                        QueryReference(link_on="references", return_properties=["title"]),
                        QueryReference(link_on="named_entities", return_properties=["name"]),
                        QueryReference(link_on="citation_contexts", return_properties=["section"]),
                        QueryReference(link_on="ner_scores", return_properties=["score"]),
                    ]
                )
                
                # Verify each reference type exists
                for ref_type in ["authors", "references", "named_entities", "citation_contexts", "ner_scores"]:
                    if ref_type not in article.references:
                        logging.error(f"❌ Missing {ref_type} references in sample article")
                        return False

            logging.info("\n✓ All cross-reference verifications passed!")
            return True
            
        except Exception as e:
            logging.error(f"Error verifying data: {str(e)}")
            return False

    def verify_imported_data(self) -> bool:
        """
        Verify that all data was imported correctly and cross-references are accessible.
        This is a comprehensive verification that includes cross-reference checks.
        """
        try:
            logging.info("\nVerifying collections...")
            # Verify each collection has expected data and log counts
            for collection_name in ["Article", "Author", "Reference", "NamedEntity", "CitationContext", "NERArticleScore", "NameVariant"]:
                collection = self.client.collections.get(collection_name)
                objects = collection.query.fetch_objects(limit=None, include_vector=False)
                if not objects or not objects.objects:
                    logging.error(f"❌ No {collection_name} objects found")
                    return False
                logging.info(f"✓ {collection_name}: {len(objects.objects):,} objects")

            logging.info("\nVerifying cross-references...")
            if not self._verify_cross_references():
                return False

            logging.info("\n✓ All data verified successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error verifying data: {str(e)}")
            return False

    def _prepare_basic_properties(self, collection_name: str, entity: Any) -> Dict[str, Any]:
        """Prepare properties ensuring each text field is under token limit."""
        try:
            properties = {}
            safe_limit = self._get_safe_token_limit()
            logger.info(f"\nPreparing properties for {collection_name} with safe token limit {safe_limit}")
            
            # Process each property based on collection type
            if collection_name == "Article":
                # First calculate total tokens before compression
                raw_properties = {
                    "filename": entity.filename,
                    "affiliations": entity.affiliations,
                    "funding_info": entity.funding_info,
                    "abstract": entity.abstract,
                    "introduction": entity.introduction,
                    "methods": entity.methods,
                    "results": entity.results,
                    "discussion": entity.discussion,
                    "figures": entity.figures,
                    "tables": entity.tables,
                    "publication_info": entity.publication_info,
                    "acknowledgements": entity.acknowledgements
                }
                
                # Log token counts for each field before compression
                total_tokens = 0
                for field_name, value in raw_properties.items():
                    if isinstance(value, str):
                        tokens = self._estimate_tokens(value)
                        total_tokens += tokens
                        logger.info(f"Field {field_name}: {tokens} tokens")
                
                logger.info(f"Total tokens before compression: {total_tokens}")
                
                # Now process each field
                properties = {
                    field_name: self._process_text(value, field_name) 
                    for field_name, value in raw_properties.items()
                }
                
                # Log token counts after compression
                final_total = 0
                for field_name, value in properties.items():
                    if isinstance(value, str):
                        tokens = self._estimate_tokens(value)
                        final_total += tokens
                        logger.info(f"Field {field_name} after compression: {tokens} tokens")
                
                logger.info(f"Total tokens after compression: {final_total}")
                
            elif collection_name == "Author":
                properties = {
                    "canonical_name": self._process_text(entity.canonical_name, "canonical_name"),
                    "email": self._process_text(entity.email, "email")
                }
            elif collection_name == "Reference":
                properties = {
                    "title": self._process_text(entity.title, "title"),
                    "journal": self._process_text(entity.journal, "journal"),
                    "volume": self._process_text(entity.volume, "volume"),
                    "pages": self._process_text(entity.pages, "pages"),
                    "publication_date": self._process_text(entity.publication_date, "publication_date"),
                    "raw_reference": self._process_text(entity.raw_reference, "raw_reference")
                }
            elif collection_name == "NamedEntity":
                properties = {
                    "name": self._process_text(entity.name, "name"),
                    "type": self._process_text(entity.type, "type")
                }
            elif collection_name == "NameVariant":
                properties = {
                    "name": self._process_text(entity.name, "name")
                }
            elif collection_name == "CitationContext":
                properties = {
                    "section": self._process_text(entity.section, "section"),
                    "local_ref_id": self._process_text(entity.local_ref_id, "local_ref_id")
                }
            elif collection_name == "NERArticleScore":
                properties = {
                    "score": entity.score  # Numeric field, no text processing needed
                }
            else:
                raise ValueError(f"Unknown collection: {collection_name}")
            
            # Verify all text fields are under limit
            total_tokens = 0
            for field_name, value in properties.items():
                if isinstance(value, str):
                    tokens = self._estimate_tokens(value)
                    total_tokens += tokens
                    if tokens > safe_limit:
                        logger.error(f"Field {field_name} exceeds safe limit: {tokens} > {safe_limit}")
                        raise ValueError(
                            f"Field {field_name} in {collection_name} still exceeds token limit "
                            f"after processing: {tokens} > {safe_limit}"
                        )
            
            logger.info(f"Final total tokens for all fields: {total_tokens}")
            if total_tokens > safe_limit:
                logger.error(f"Total tokens {total_tokens} exceeds safe limit {safe_limit}")
                # We need to reduce further
                compression_ratio = safe_limit / total_tokens
                logger.info(f"Applying additional compression with ratio {compression_ratio:.2f}")
                
                # Reprocess all text fields with the new ratio
                for field_name, value in properties.items():
                    if isinstance(value, str):
                        properties[field_name] = self._process_text(
                            value, 
                            field_name,
                            force_ratio=compression_ratio
                        )
                
                # Verify final total
                final_total = sum(
                    self._estimate_tokens(value) 
                    for value in properties.values() 
                    if isinstance(value, str)
                )
                logger.info(f"Final total tokens after additional compression: {final_total}")
            
            return properties
            
        except Exception as e:
            logger.error(f"Error preparing properties for {collection_name}: {str(e)}")
            raise