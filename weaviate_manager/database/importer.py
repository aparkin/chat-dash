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
    MODEL_TOKEN_LIMITS
)
from weaviate.util import generate_uuid5
from tqdm import tqdm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from ..utils import log_progress
from weaviate.collections.classes.grpc import QueryReference

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
        self.client = client
        self.suppress_initial_logging = suppress_initial_logging
        
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
        DATA_MANAGER_ATTRS = {
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
                "total": len(getattr(self.data_manager, DATA_MANAGER_ATTRS[collection], {})),
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

    def import_data(self) -> bool:
        """Import all data into Weaviate with progress tracking."""
        try:
            logging.info("\nStarting data import...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=None,  # This ensures output goes to stdout
                transient=True  # This ensures the progress bar is cleared when done
            ) as progress:
                # Create task for overall progress
                task = progress.add_task(
                    "[cyan]Importing data...", 
                    total=12  # Total steps including cross-references
                )
                
                # Phase 1: Primary entity import
                for collection_name in ["Article", "Author", "Reference", "NamedEntity"]:
                    progress.update(task, description=f"[cyan]Importing {collection_name}...")
                    if not self._import_collection_method(collection_name):
                        logging.error(f"\n❌ Failed to import {collection_name} collection")
                        return False
                    logging.info(f"✓ Imported {len(self.created_uuids[collection_name]):,} {collection_name} objects")
                    progress.advance(task)
                
                # Phase 2: Secondary entity import
                for collection_name in ["CitationContext", "NERArticleScore", "NameVariant"]:
                    if collection_name in self.data_manager.collections:
                        progress.update(task, description=f"[cyan]Importing {collection_name}...")
                        if not self._import_collection_method(collection_name):
                            logging.error(f"\n❌ Failed to import {collection_name} collection")
                            return False
                        logging.info(f"✓ Imported {len(self.created_uuids[collection_name]):,} {collection_name} objects")
                        progress.advance(task)
                
                # Phase 3: Add cross-references
                progress.update(task, description="[cyan]Adding cross-references...")
                if not self._add_cross_references():
                    logging.error("\n❌ Failed to add cross-references")
                    return False
                logging.info("✓ Added cross-references")
                progress.advance(task)
                
                # Complete the progress bar
                progress.update(task, completed=12)
            
            # Phase 4: Verify data and cross-references (outside progress bar)
            logging.info("\nVerifying imported data and cross-references...")
            if not self.verify_imported_data():
                logging.error("❌ Data verification failed")
                return False
            
            # Print final success message
            logging.info("\n✓ Import completed successfully!")
            return True
            
        except Exception as e:
            logging.error(f"Error during import: {str(e)}")
            return False

    def _import_collection_method(self, collection_name: str) -> bool:
        """Map collection names to their import methods."""
        method_map = {
            "Article": self._import_articles,
            "Author": self._import_authors,
            "Reference": self._import_references,
            "NamedEntity": self._import_entities,
            "CitationContext": self._import_contexts,
            "NERArticleScore": self._import_scores,
            "NameVariant": self._import_variants
        }
        
        if collection_name not in method_map:
            logging.error(f"❌ No import method found for collection {collection_name}")
            return False
            
        return method_map[collection_name]()

    def __enter__(self):
        """Set up the importer."""
        try:
            logging.info("Initializing importer...")
            self.created_uuids = defaultdict(dict)
            return self
        except Exception as e:
            logging.error(f"❌ Error initializing importer: {str(e)}")
            raise
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up after import."""
        if exc_type is not None:
            logging.error(f"\n❌ Error during import: {str(exc_val)}")
            return False
        return True

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

    def _process_batch(self, collection, batch) -> bool:
        """Process a batch of entities for creation."""
        logger = logging.getLogger('weaviate_manager.importer')
        try:
            with collection.batch.dynamic() as batch_processor:
                failed_objects = []
                for entity_id, entity in batch:
                    try:
                        # Verify entity has required UUID
                        if not hasattr(entity, 'uuid'):
                            error_msg = f"Entity {entity_id} missing UUID"
                            logger.error(error_msg)
                            failed_objects.append((entity_id, [error_msg]))
                            continue

                        # Use the entity's pre-created UUID
                        result = batch_processor.add_object(
                            properties=self._prepare_basic_properties(collection.name, entity),
                            uuid=entity.uuid
                        )
                        
                        # Check if the object was actually created
                        if result and hasattr(result, 'errors') and result.errors:
                            failed_objects.append((entity_id, result.errors))
                            self.import_stats[collection.name]["failed"] += 1
                            logger.error(f"Failed to create {collection.name} {entity_id}:")
                            for error in result.errors:
                                logger.error(f"  {error}")
                        else:
                            # Track the UUID for verification
                            self.created_uuids[collection.name][entity_id] = entity.uuid
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
            
            if failed_objects:
                error_msg = f"\nBatch processing failed for {collection.name}:"
                error_msg += f"\nFailed: {len(failed_objects)}/{len(batch)} objects"
                for entity_id, errors in failed_objects:
                    error_msg += f"\n  {entity_id}:"
                    for error in errors:
                        error_msg += f"\n    {error}"
                logger.error(error_msg)
                return False  # For stricter error handling, fail if any object failed
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return False

    def _prepare_basic_properties(self, entity_type: str, entity) -> Dict[str, Any]:
        """Prepare basic properties, applying compression only when needed."""
        logger = logging.getLogger('weaviate_manager.importer')
        properties = {}
        
        # For Articles, handle text field compression specially
        if entity_type == "Article":
            # Define compressible fields in order of importance (most important first)
            COMPRESSIBLE_FIELDS = [
                "abstract",
                "introduction",
                "methods",
                "results", 
                "discussion",
                "figures",
                "tables"
            ]
            
            # Define non-compressible fields that should be included
            NON_COMPRESSIBLE_FIELDS = [
                "filename",
                "affiliations",
                "funding_info",
                "publication_info",
                "acknowledgements",
                "keywords",
                "doi",
                "journal",
                "title",
                "publication_date",
                "authors_string",
                "corresponding_authors",
                "corresponding_emails"
            ]
            
            # Step 1: Calculate tokens for all fields
            field_tokens = {}
            field_values = {}
            uncompressible_total = 0
            compressible_total = 0
            
            # Calculate tokens for non-compressible fields
            for field in NON_COMPRESSIBLE_FIELDS:
                if hasattr(entity, field):
                    value = getattr(entity, field)
                    if value is not None:
                        properties[field] = value
                        if isinstance(value, str):
                            tokens = self._estimate_tokens(str(value))
                            field_tokens[field] = tokens
                            uncompressible_total += tokens
            
            # Calculate tokens for compressible fields
            for field in COMPRESSIBLE_FIELDS:
                if hasattr(entity, field):
                    value = getattr(entity, field)
                    if isinstance(value, str) and value.strip():
                        tokens = self._estimate_tokens(str(value))
                        field_tokens[field] = tokens
                        field_values[field] = value
                        compressible_total += tokens
            
            # Log initial token counts - only if exceeds limit
            total_tokens = uncompressible_total + compressible_total
            if total_tokens > DEFAULT_MAX_TOKENS:
                logger.info(f"\nCompressing article: {entity.filename} ({total_tokens:,} tokens)")
            
            # If total is under limit, no compression needed
            if total_tokens <= DEFAULT_MAX_TOKENS:
                properties.update(field_values)
                return properties
            
            # Calculate required compression
            available_tokens = DEFAULT_MAX_TOKENS - uncompressible_total
            if available_tokens <= 0:
                raise ValueError(
                    f"Non-compressible fields alone exceed token limit: {uncompressible_total:,} > {DEFAULT_MAX_TOKENS:,}"
                )
            
            # Compress fields
            compressed_total = 0
            for field in COMPRESSIBLE_FIELDS:
                if field not in field_values:
                    continue
                    
                original_tokens = field_tokens[field]
                # Calculate target tokens, ensuring important fields get minimum representation
                min_tokens = 100 if field in ["abstract", "introduction"] else 50
                target_tokens = max(
                    min_tokens,
                    min(
                        original_tokens,  # Don't expand
                        int(original_tokens * (available_tokens / compressible_total))  # Compress proportionally
                    )
                )
                
                # Ensure we don't exceed remaining tokens
                remaining_tokens = available_tokens - compressed_total
                if target_tokens > remaining_tokens:
                    target_tokens = remaining_tokens
                
                if target_tokens <= 0:
                    continue
                
                # Compress the field
                compressed = self._reduce_text_to_token_limit(
                    text=field_values[field],
                    target_tokens=target_tokens,
                    field_name=field
                )
                properties[field] = compressed
                compressed_total += self._estimate_tokens(compressed)
                
                if compressed_total >= available_tokens:
                    break
            
            # Log only final result if compression was needed
            final_total = sum(self._estimate_tokens(str(v)) for v in properties.values() if v is not None)
            if final_total > DEFAULT_MAX_TOKENS:
                raise ValueError(
                    f"Final token count {final_total:,} exceeds limit {DEFAULT_MAX_TOKENS:,}"
                )
            
            # Update compression statistics silently
            self.compression_stats["total_compressed"] += 1
            self.compression_stats["total_tokens_before"] += compressible_total
            self.compression_stats["total_tokens_after"] += compressed_total
            self.compression_stats["compression_ratios"].append(compressed_total/compressible_total)
            
        else:
            # For non-Article entities, copy properties including relationship sets
            for field, value in vars(entity).items():
                if field == 'uuid' or field == 'id':  # Only skip uuid/id fields
                    continue
                # Include relationship sets but convert to list for JSON serialization
                if isinstance(value, set):
                    properties[field] = list(value)
                else:
                    properties[field] = value
        
        return properties
        
    def _add_cross_references(self) -> bool:
        """Add all cross-references between entities."""
        try:
            # Add article references
            self._add_article_references()
            
            # Add author references
            self._add_author_references()
            
            # Add reference references
            self._add_reference_references()
            
            # Add entity references
            self._add_entity_references()
            
            # Add context references
            self._add_citation_context_references()
            
            # Add score references
            self._add_score_references()
            
            # Add variant references
            self._add_name_variant_references()
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding cross-references: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            return False

    def _process_text(self, text: str, field_name: str) -> str:
        """
        Process text fields for import, applying compression if needed.
        
        Features:
        - Token counting and estimation
        - Intelligent text compression
        - Length validation
        - Statistics tracking
        """
        logger = logging.getLogger('weaviate_manager.importer')
        
        if not text:
            return ""
            
        # Enforce MAX_STRING_LENGTH
        if len(text) > MAX_STRING_LENGTH:
            text = text[:MAX_STRING_LENGTH]
            logger.warning(f"Truncated {field_name} to {MAX_STRING_LENGTH} characters")
                
        # Count tokens
        tokens = self._tokenizer.encode(text)
        token_count = len(tokens)
        
        # Only compress if needed
        if token_count > DEFAULT_MAX_TOKENS:
            logger.info(f"Compressing {field_name} ({token_count} tokens > {DEFAULT_MAX_TOKENS} limit)...")
            compressed = self._reduce_text_to_token_limit(
                text=text,
                target_tokens=DEFAULT_MAX_TOKENS,
                field_name=field_name
            )
            final_tokens = len(self._tokenizer.encode(compressed))
            
            # Force truncate if compression didn't meet token limit
            if final_tokens > DEFAULT_MAX_TOKENS:
                logger.warning(
                    f"Compression insufficient for {field_name}. "
                    f"Tokens: {final_tokens} > {DEFAULT_MAX_TOKENS}. "
                    f"Force truncating..."
                )
                compressed = self._truncate_to_tokens(compressed, DEFAULT_MAX_TOKENS)
                final_tokens = len(self._tokenizer.encode(compressed))
            
            # Update compression statistics
            self.compression_stats["total_compressed"] += 1
            self.compression_stats["total_tokens_before"] += token_count
            self.compression_stats["total_tokens_after"] += final_tokens
            ratio = final_tokens / token_count
            self.compression_stats["compression_ratios"].append(ratio)
            
            logger.info(
                f"Compressed {field_name}: {token_count:,} → {final_tokens:,} tokens "
                f"({ratio:.1%} of original)"
            )
                
            return compressed
            
        return text

    def _reduce_text_to_token_limit(self, text: str, target_tokens: int, field_name: str) -> str:
        """Intelligently reduce text to fit within token limit while preserving key content."""
        logger = logging.getLogger('weaviate_manager.importer')
        current_tokens = self._estimate_tokens(text)
        
        if current_tokens <= target_tokens:
            return text
            
        # Split into sections
        paragraphs = text.split('\n\n')
        if not paragraphs:
            return text
            
        # Score paragraphs based on importance
        scored_paragraphs = []
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Calculate importance score
            score = 1  # Base score
            
            # Higher score for paragraphs with key terms
            score += sum(2 for term in IMPORTANCE_KEY_TERMS if term.lower() in para.lower())
            
            # Higher score for paragraphs with numbers/measurements
            score += sum(1 for char in para if char.isdigit())
            
            # Higher score for paragraphs with citations
            score += para.count('(') + para.count('[')
            
            # Higher score for paragraphs with comparison terms
            score += sum(1 for term in IMPORTANCE_COMPARISON_TERMS if term.lower() in para.lower())
            
            # Store score and token count
            para_tokens = self._estimate_tokens(para)
            scored_paragraphs.append((score, para, para_tokens))
            
        # Sort by importance score
        scored_paragraphs.sort(reverse=True)
        
        # Always include first paragraph if it fits
        reduced_text = []
        current_total = 0
        
        if scored_paragraphs:
            first_para = scored_paragraphs[0][1]
            first_tokens = self._estimate_tokens(first_para)
            if first_tokens <= target_tokens:
                reduced_text.append(first_para)
                current_total = first_tokens
                scored_paragraphs = scored_paragraphs[1:]
        
        # Add highest scoring paragraphs that fit
        for score, para, para_tokens in scored_paragraphs:
            if current_total + para_tokens <= target_tokens:
                reduced_text.append(para)
                current_total += para_tokens
            else:
                # Try to fit a truncated version if we have room
                remaining = target_tokens - current_total
                if remaining > 50:  # Only if we have reasonable space
                    truncated = self._truncate_to_tokens(para, remaining)
                    reduced_text.append(truncated)
                    current_total += remaining
                break
        
        if not reduced_text:  # If nothing fit, at least keep some of the first paragraph
            first_para = paragraphs[0]
            reduced_text = [self._truncate_to_tokens(first_para, target_tokens)]
        
        final = '\n\n'.join(reduced_text)
        final_tokens = self._estimate_tokens(final)
        
        # Remove debug logging of individual field compression
        return final

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
                if stats["skipped"] > 0:
                    logging.debug(f"Skipped: {stats['skipped']} ({stats['skipped']/stats['total']:.1%})")
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

    def _add_article_references(self) -> bool:
        """Add all relationships for Article entities."""
        try:
            collection = self.client.collections.get("Article")
            
            with collection.batch.dynamic() as batch:
                for article_id, article in self.data_manager.articles.items():
                    if article_id not in self.created_uuids["Article"]:
                        continue
                        
                    article_uuid = article.uuid
                    
                    # Add author references
                    if article.authors:
                        try:
                            batch.add_reference(
                                from_uuid=article_uuid,
                                from_property="authors",
                                to=list(article.authors)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding author references for article {article_id}: {str(e)}")
                    
                    # Add reference references
                    if article.references:
                        try:
                            batch.add_reference(
                                from_uuid=article_uuid,
                                from_property="references",
                                to=list(article.references)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding reference citations for article {article_id}: {str(e)}")
                    
                    # Add named entity references
                    if article.named_entities:
                        try:
                            batch.add_reference(
                                from_uuid=article_uuid,
                                from_property="named_entities",
                                to=list(article.named_entities)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding named entity references for article {article_id}: {str(e)}")
                    
                    # Add citation context references
                    if article.citation_contexts:
                        try:
                            batch.add_reference(
                                from_uuid=article_uuid,
                                from_property="citation_contexts",
                                to=list(article.citation_contexts)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding citation context references for article {article_id}: {str(e)}")
                    
                    # Add NER score references
                    if article.ner_scores:
                        try:
                            batch.add_reference(
                                from_uuid=article_uuid,
                                from_property="ner_scores",
                                to=list(article.ner_scores)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding NER score references for article {article_id}: {str(e)}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding article references: {str(e)}")
            return False

    def _add_author_references(self) -> bool:
        """Add all relationships for Author entities."""
        try:
            collection = self.client.collections.get("Author")
            total_authors = len(self.data_manager.authors)
            authors_processed = 0
            
            with collection.batch.dynamic() as batch:
                for author_id, author in self.data_manager.authors.items():
                    if author_id not in self.created_uuids["Author"]:
                        continue
                        
                    author_uuid = author.uuid
                    
                    # Add article references
                    if author.articles:
                        try:
                            batch.add_reference(
                                from_uuid=author_uuid,
                                from_property="primary_articles",
                                to=list(author.articles)
                            )
                            authors_processed += 1
                            if authors_processed % 50000 == 0:  # Log every 50k
                                logging.info(f"Author relationships: {authors_processed:,}/{total_authors:,}")
                        except Exception as e:
                            logging.error(f"Failed adding article references for author {author_id}: {str(e)}")
                    
                    # Add reference references
                    if author.authored_references:
                        try:
                            batch.add_reference(
                                from_uuid=author_uuid,
                                from_property="authored_references",
                                to=list(author.authored_references)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding reference references for author {author_id}: {str(e)}")
                    
                    # Add name variant references
                    if author.name_variants:
                        try:
                            batch.add_reference(
                                from_uuid=author_uuid,
                                from_property="name_variants",
                                to=list(author.name_variants)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding name variant references for author {author_id}: {str(e)}")
                        
            return True
            
        except Exception as e:
            logging.error(f"Error adding author references: {str(e)}")
            return False

    def _add_reference_references(self) -> bool:
        """Add all relationships for Reference entities."""
        try:
            collection = self.client.collections.get("Reference")
            total_refs = len(self.data_manager.references)
            refs_processed = 0
            
            with collection.batch.dynamic() as batch:
                for ref_id, ref in self.data_manager.references.items():
                    try:
                        if ref_id not in self.created_uuids["Reference"]:
                            continue
                            
                        ref_uuid = ref.uuid
                        # Add author references
                        if ref.authors:
                            try:
                                batch.add_reference(
                                    from_uuid=ref_uuid,
                                    from_property="authors",
                                    to=list(ref.authors)
                                )
                                refs_processed += 1
                                if refs_processed % 50000 == 0:  # Log every 50k
                                    logging.info(f"Reference relationships: {refs_processed:,}/{total_refs:,}")
                            except Exception as e:
                                logging.error(f"Failed adding author references for reference {ref_id}: {str(e)}")
                        
                        # Add citing article references
                        if ref.citing_articles:
                            try:
                                batch.add_reference(
                                    from_uuid=ref_uuid,
                                    from_property="cited_in",
                                    to=list(ref.citing_articles)
                                )
                            except Exception as e:
                                logging.error(f"Failed adding citing article references for reference {ref_id}: {str(e)}")
                                
                        # Add citation context references
                        if ref.citation_contexts:
                            try:
                                batch.add_reference(
                                    from_uuid=ref_uuid,
                                    from_property="citation_contexts",
                                    to=list(ref.citation_contexts)
                                )
                            except Exception as e:
                                logging.error(f"Failed adding citation context references for reference {ref_id}: {str(e)}")

                    except Exception as e:
                        logging.error(f"Failed processing reference {ref_id}: {str(e)}")
                        continue

            return True
            
        except Exception as e:
            logging.error(f"Error adding reference references: {str(e)}")
            return False

    def _add_entity_references(self) -> bool:
        """Add all relationships for NamedEntity entities."""
        try:
            collection = self.client.collections.get("NamedEntity")
            
            with collection.batch.dynamic() as batch:
                for entity_id, entity in self.data_manager.ner_objects.items():
                    if entity_id not in self.created_uuids["NamedEntity"]:
                        continue
                        
                    entity_uuid = entity.uuid
                    
                    # Add article references
                    if entity.articles:
                        try:
                            batch.add_reference(
                                from_uuid=entity_uuid,
                                from_property="found_in",
                                to=list(entity.articles)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding article references for entity {entity_id}: {str(e)}")
                    
                    # Add score references
                    if entity.ner_scores:
                        try:
                            batch.add_reference(
                                from_uuid=entity_uuid,
                                from_property="article_scores",
                                to=list(entity.ner_scores)
                            )
                        except Exception as e:
                            logging.error(f"Failed adding score references for entity {entity_id}: {str(e)}")
                        
            return True
            
        except Exception as e:
            logging.error(f"Error adding entity references: {str(e)}")
            return False
            
    def _add_citation_context_references(self) -> bool:
        """Add all relationships for CitationContext entities."""
        try:
            collection = self.client.collections.get("CitationContext")
            
            with collection.batch.dynamic() as batch:
                for ctx_id, ctx in self.data_manager.citation_contexts.items():
                    try:
                        if ctx_id not in self.created_uuids["CitationContext"]:
                            continue
                            
                        ctx_uuid = ctx.uuid
                        
                        # Add article reference
                        if ctx.article_uuid:
                            try:
                                batch.add_reference(
                                    from_uuid=ctx_uuid,
                                    from_property="article",
                                    to=ctx.article_uuid
                                )
                            except Exception as e:
                                logging.error(f"Failed adding article reference for context {ctx_id}: {str(e)}")
                        
                        # Add reference reference
                        if ctx.reference_uuid:
                            try:
                                batch.add_reference(
                                    from_uuid=ctx_uuid,
                                    from_property="reference",
                                    to=ctx.reference_uuid
                                )
                            except Exception as e:
                                logging.error(f"Failed adding reference reference for context {ctx_id}: {str(e)}")

                    except Exception as e:
                        logging.error(f"Failed processing citation context {ctx_id}: {str(e)}")
                        continue

            return True
            
        except Exception as e:
            logging.error(f"Error adding citation context references: {str(e)}")
            return False
            
    def _add_score_references(self) -> bool:
        """Add all relationships for NERArticleScore entities."""
        try:
            collection = self.client.collections.get("NERArticleScore")
            
            with collection.batch.dynamic() as batch:
                for score_id, score in self.data_manager.ner_scores.items():
                    try:
                        if score_id not in self.created_uuids["NERArticleScore"]:
                            continue
                            
                        score_uuid = score.uuid
                        
                        # Add entity reference
                        if score.entity_uuid:
                            try:
                                batch.add_reference(
                                    from_uuid=score_uuid,
                                    from_property="entity",
                                    to=score.entity_uuid
                                )
                            except Exception as e:
                                logging.error(f"Failed adding entity reference for score {score_id}: {str(e)}")
                        
                        # Add article reference
                        if score.article_uuid:
                            try:
                                batch.add_reference(
                                    from_uuid=score_uuid,
                                    from_property="article",
                                    to=score.article_uuid
                                )
                            except Exception as e:
                                logging.error(f"Failed adding article reference for score {score_id}: {str(e)}")

                    except Exception as e:
                        logging.error(f"Failed processing NER score {score_id}: {str(e)}")
                        continue

            return True
            
        except Exception as e:
            logging.error(f"Error adding NER score references: {str(e)}")
            return False
            
    def _add_name_variant_references(self) -> bool:
        """Add all relationships for NameVariant entities."""
        try:
            collection = self.client.collections.get("NameVariant")
            
            with collection.batch.dynamic() as batch:
                for variant_id, variant in self.data_manager.name_variants.items():
                    try:
                        if variant_id not in self.created_uuids["NameVariant"]:
                            continue
                            
                        variant_uuid = variant.uuid
                        
                        # Add author reference
                        if variant.author_uuid:
                            try:
                                batch.add_reference(
                                    from_uuid=variant_uuid,
                                    from_property="author",
                                    to=variant.author_uuid
                                )
                            except Exception as e:
                                logging.error(f"Failed adding author reference for variant {variant_id}: {str(e)}")

                    except Exception as e:
                        logging.error(f"Failed processing name variant {variant_id}: {str(e)}")
                        continue

            return True
            
        except Exception as e:
            logging.error(f"Error adding name variant references: {str(e)}")
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
                if not article.references:
                    logging.error("❌ No references found in sample article")
                    return False
                    
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