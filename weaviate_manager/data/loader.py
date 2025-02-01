"""
Data loader for scientific literature database.

This module provides functionality for loading, processing, and managing scientific literature data.
It handles articles, authors, references, and named entities, establishing and validating relationships
between them. The loader uses a two-pass approach to ensure data integrity and proper relationship
establishment.

Key Features:
- Two-pass loading process for proper UUID generation and relationship building
- Comprehensive validation of entity relationships
- Support for extracting connected subsets of the data
- Rich statistics generation and analysis tools
- Flexible logging with file and console output
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import random
import weaviate
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from weaviate.util import generate_uuid5

# Configure logging
# File handler - detailed logging
file_handler = logging.FileHandler('weaviate_import.log', mode='w')
file_handler.setLevel(logging.DEBUG)  # Always capture everything in the file
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler - only show warnings and errors by default
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Default to only showing warnings/errors
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)  # Allow all levels to be logged to file
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

def set_console_log_level(verbose: bool) -> None:
    """Set console logging level based on verbose flag."""
    if verbose:
        console_handler.setLevel(logging.INFO)  # Show info and above when verbose
        logging.info("Verbose logging enabled")
    else:
        console_handler.setLevel(logging.WARNING)  # Only show warnings/errors
        
# Remove any existing console handlers to avoid duplicates
for handler in root_logger.handlers[:]:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        root_logger.removeHandler(handler)

from .models import (
    Article, Author, NamedEntity, Reference,
    NERArticleScore, NameVariant, CitationContext
)

@dataclass
class Relationships:
    """
    Container for relationship data between entities.
    
    Attributes:
        name_variants: List of dictionaries containing author name variant relationships
        citation_contexts: List of dictionaries containing citation context relationships
        ner_scores: List of dictionaries containing named entity recognition scores
    """
    name_variants: List[Dict]
    citation_contexts: List[Dict]
    ner_scores: List[Dict]

class LiteratureDataManager:
    """
    Manages loading and processing of scientific literature data.
    
    This class handles the complete lifecycle of scientific literature data, from loading
    raw JSON files to establishing and validating relationships between entities. It uses
    a two-pass approach to ensure proper data loading and relationship establishment.
    
    Loading Process:
    1. First Pass:
       - Loads primary entities (Articles, Authors, References, Named Entities)
       - Generates and stores UUIDs for each entity
       - Builds ID to UUID mappings for later reference
       
    2. Second Pass:
       - Creates relationship objects (name variants, citation contexts, NER scores)
       - Updates cross-references using UUID mappings
       - Validates all relationships for consistency and completeness
    
    Attributes:
        data_dir: Path to directory containing input JSON files
        articles: Dictionary mapping article IDs to Article objects
        ner_objects: Dictionary mapping NER terms to NamedEntity objects
        authors: Dictionary mapping author IDs to Author objects
        references: Dictionary mapping reference IDs to Reference objects
        ner_scores: Dictionary mapping score keys to NERArticleScore objects
        name_variants: Dictionary mapping variant keys to NameVariant objects
        citation_contexts: Dictionary mapping context keys to CitationContext objects
        uuid_maps: Dictionary of dictionaries mapping original IDs to UUIDs for each entity type
    """
    
    @property
    def collections(self) -> Dict[str, Dict]:
        """Map collection names to their corresponding data dictionaries."""
        return {
            "Article": self.articles,
            "Author": self.authors,
            "Reference": self.references,
            "NamedEntity": self.ner_objects,
            "CitationContext": self.citation_contexts,
            "NERArticleScore": self.ner_scores,
            "NameVariant": self.name_variants
        }
    
    def __init__(self, data_dir: Path):
        """
        Initialize the LiteratureDataManager.
        
        Args:
            data_dir: Path to directory containing the input JSON files. The directory
                     must contain the following files:
                     - processed_articles.json: Article data
                     - unified_authors.json: Author data
                     - unified_references.json: Reference data
                     - unified_ner_objects.json: Named entity data
        
        Raises:
            FileNotFoundError: If data_dir does not exist
        """
        self.data_dir = Path(data_dir)
        
        # Primary data collections
        self.articles: Dict[str, Article] = {}
        self.ner_objects: Dict[str, NamedEntity] = {}
        self.authors: Dict[str, Author] = {}
        self.references: Dict[str, Reference] = {}
        
        # Relationship collections
        self.ner_scores: Dict[str, NERArticleScore] = {}
        self.name_variants: Dict[str, NameVariant] = {}
        self.citation_contexts: Dict[str, CitationContext] = {}
        
        # UUID mappings (original_id -> uuid)
        self.uuid_maps = {
            "articles": {},      # filename -> uuid
            "authors": {},       # canonical_name -> uuid
            "references": {},    # ref_id -> uuid
            "ner_objects": {},   # term -> uuid
            "citation_contexts": {},
            "name_variants": {},
            "ner_scores": {}     # score_key -> uuid
        }
        
        # Temporary storage for relationship data
        self._temp_ner_scores = {}
        self._temp_author_variants = {}
        self._temp_reference_occurrences = {}
        self._temp_article_authors = {}
        self._temp_reference_authors = {}

    def load_data(self) -> bool:
        """
        Load and process all data files in a two-pass approach.
        
        This method orchestrates the complete data loading process:
        1. First pass loads primary entities and generates UUIDs
        2. Second pass establishes relationships between entities
        3. Final validation ensures data integrity
        
        Returns:
            bool: True if loading was successful, False if any errors occurred
            
        Raises:
            FileNotFoundError: If any required input files are missing
            JSONDecodeError: If any input files contain invalid JSON
        """
        try:
            # First pass: Load primary entities and generate UUIDs
            self._load_primary_entities()
            
            # Second pass: Establish relationships using UUIDs
            self._establish_relationships()
            
            # Validate the loaded data
            return self._validate_relationships()
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            return False
    
    def _load_primary_entities(self) -> None:
        """
        Execute first pass: Load primary entities and generate their UUIDs.
        
        The loading order is critical to maintain proper relationships:
        1. Articles (base entities with no initial cross-references)
        2. NER Objects (establishes NER scores with articles)
        3. Authors (establishes author-article relationships)
        4. References (establishes reference-author and reference-article relationships)
        
        Each entity is loaded from its corresponding JSON file and assigned a UUID.
        Original IDs are mapped to UUIDs for later relationship establishment.
        
        Raises:
            FileNotFoundError: If any required input files are missing
            JSONDecodeError: If any input files contain invalid JSON
        """
        self._load_articles()
        self._load_ner_objects()
        self._load_authors()
        self._load_references()
        
        logging.info("Completed first pass: Primary entities loaded")
        logging.info(f"Generated UUIDs for: {len(self.articles)} articles, "
                    f"{len(self.ner_objects)} NER objects, "
                    f"{len(self.authors)} authors, "
                    f"{len(self.references)} references")
    
    def _load_articles(self) -> None:
        """Load articles and generate their UUIDs."""
        articles_file = self.data_dir / "processed_articles.json"
        if not articles_file.exists():
            raise FileNotFoundError(f"Articles file not found: {articles_file}")
            
        with articles_file.open() as f:
            raw_articles = json.load(f)
            
        for filename, data in raw_articles.items():
            article = Article(
                filename=filename,
                affiliations=data.get("affiliations", ""),
                funding_info=data.get("funding_info", ""),
                abstract=data.get("abstract", ""),
                introduction=data.get("introduction", ""),
                methods=data.get("methods", ""),
                results=data.get("results", ""),
                discussion=data.get("discussion", ""),
                figures=data.get("figures", ""),
                tables=data.get("tables", ""),
                publication_info=data.get("publication_info", ""),
                acknowledgements=" ".join(data.get("acknowledgements", []))
            )
            # Generate deterministic UUID based on filename
            article.uuid = generate_uuid5(filename, "Article")
            
            self.articles[filename] = article
            self.uuid_maps["articles"][filename] = article.uuid
            
        logging.info(f"Loaded {len(self.articles)} articles")
    
    def _load_ner_objects(self) -> None:
        """
        Load named entities and generate their UUIDs.
        Phase 1: Only loads core properties and assigns UUIDs.
        Relationship data is stored temporarily for phase 2.
        """
        ner_file = self.data_dir / "unified_ner_objects.json"
        if not ner_file.exists():
            raise FileNotFoundError(f"NER objects file not found: {ner_file}")
            
        with ner_file.open() as f:
            raw_ner = json.load(f)
            
        logging.debug(f"\nLoading NER data from {len(raw_ner)} terms")
        for term, data in raw_ner.items():
            # Create NER object with only core properties
            ner = NamedEntity(
                name=data["name"],
                type=data.get("type", "")
            )
            # Generate deterministic UUID based on term
            ner.uuid = generate_uuid5(term, "NamedEntity")
            
            self.ner_objects[term] = ner
            self.uuid_maps["ner_objects"][term] = ner.uuid
            
            # Store scores for later relationship establishment
            self._temp_ner_scores[term] = data.get("article_scores", {})
        
        logging.info(f"Loaded {len(self.ner_objects)} named entities")
        logging.info(f"Stored {sum(len(scores) for scores in self._temp_ner_scores.values())} scores for relationship establishment")

    def _load_authors(self) -> None:
        """
        Load authors and generate their UUIDs.
        Phase 1: Only loads core properties and assigns UUIDs.
        Relationship data is stored temporarily for phase 2.
        """
        author_file = self.data_dir / "unified_authors.json"
        if not author_file.exists():
            raise FileNotFoundError(f"Author file not found: {author_file}")
            
        with author_file.open() as f:
            raw_authors = json.load(f)
            
        logging.debug(f"\nLoading author data from {len(raw_authors)} entries")
        for author_id, data in raw_authors.items():
            # Create author with only core properties
            author = Author(
                canonical_name=data["canonical_name"],
                email=data.get("email", "")
            )
            # Generate deterministic UUID based on author_id
            author.uuid = generate_uuid5(author_id, "Author")
            
            self.authors[author_id] = author
            self.uuid_maps["authors"][author_id] = author.uuid
            
            # Store relationship data for phase 2
            self._temp_author_variants[author_id] = data.get("name_variants", [])
            self._temp_article_authors[author_id] = data.get("primary_articles", [])
        
        logging.info(f"Loaded {len(self.authors)} authors")
        logging.info(f"Stored {sum(len(variants) for variants in self._temp_author_variants.values())} name variants for relationship establishment")
        logging.info(f"Stored {sum(len(articles) for articles in self._temp_article_authors.values())} article relationships for establishment")

    def _load_references(self) -> None:
        """
        Load references and generate their UUIDs.
        Phase 1: Only loads core properties and assigns UUIDs.
        Relationship data is stored temporarily for phase 2.
        """
        ref_file = self.data_dir / "unified_references.json"
        if not ref_file.exists():
            raise FileNotFoundError(f"References file not found: {ref_file}")
            
        with ref_file.open() as f:
            raw_refs = json.load(f)
            
        logging.debug(f"\nLoading reference data from {len(raw_refs)} entries")
        for ref_id, data in raw_refs.items():
            # Create reference with only core properties
            ref = Reference(
                title=data.get("title", ""),
                journal=data.get("journal", ""),
                volume=data.get("volume", ""),
                pages=data.get("pages", ""),
                publication_date=data.get("publication_date", ""),
                raw_reference=data.get("raw_reference", "")
            )
            # Generate deterministic UUID based on ref_id
            ref.uuid = generate_uuid5(ref_id, "Reference")
            
            self.references[ref_id] = ref
            self.uuid_maps["references"][ref_id] = ref.uuid
            
            # Store relationship data for phase 2
            self._temp_reference_authors[ref_id] = data.get("unified_authors", [])
            self._temp_reference_occurrences[ref_id] = data.get("occurrences", [])
        
        logging.info(f"Loaded {len(self.references)} references")
        logging.info(f"Stored {sum(len(authors) for authors in self._temp_reference_authors.values())} author relationships for establishment")
        logging.info(f"Stored {sum(len(occs) for occs in self._temp_reference_occurrences.values())} citation occurrences for establishment")

    def _verify_reciprocal_relationship(self, entity1_uuid: str, entity2_uuid: str, 
                                      entity1_refs: Set[str], entity2_refs: Set[str],
                                      relationship_type: str) -> bool:
        """
        Verify that a relationship is properly reciprocal between two entities.
        
        Args:
            entity1_uuid: UUID of first entity
            entity2_uuid: UUID of second entity
            entity1_refs: Set of references from first entity
            entity2_refs: Set of references from second entity
            relationship_type: Type of relationship for logging
            
        Returns:
            bool: True if relationship is reciprocal
        """
        if entity2_uuid not in entity1_refs:
            logging.error(f"{relationship_type}: Missing forward reference {entity1_uuid} -> {entity2_uuid}")
            return False
        if entity1_uuid not in entity2_refs:
            logging.error(f"{relationship_type}: Missing reverse reference {entity2_uuid} -> {entity1_uuid}")
            return False
        return True

    def _establish_author_relationships(self) -> None:
        """
        Phase 2: Establish author relationships using stored temporary data.
        Creates name variants and establishes bidirectional relationships.
        """
        total_variants_created = 0
        total_article_relationships = 0
        articles_without_authors = []
        failed_relationships = 0
        
        # Process relationships from unified authors
        logging.info("Processing author relationships from unified authors...")
        
        # First create name variants
        for author_id, variants in self._temp_author_variants.items():
            if author_id not in self.authors:
                logging.error(f"Author {author_id} not found when creating name variants")
                continue
                
            author = self.authors[author_id]
            
            for variant_name in variants:
                try:
                    # Create name variant object
                    variant = NameVariant(
                        name=variant_name,
                        author_uuid=author.uuid
                    )
                    
                    # Generate deterministic UUID based on author and variant
                    variant.uuid = generate_uuid5(f"{author.uuid}_{variant_name}", "NameVariant")
                    
                    # Store variant
                    variant_id = f"{author_id}_{variant_name}"
                    self.name_variants[variant_id] = variant
                    self.uuid_maps["name_variants"][variant_id] = variant.uuid
                    
                    # Add reference to author
                    author.name_variants.add(variant.uuid)
                    total_variants_created += 1
                    
                except Exception as e:
                    logging.error(f"Failed to create name variant {variant_name} for author {author_id}: {str(e)}")
                    continue
        
        # Then establish article relationships
        for author_id, article_ids in self._temp_article_authors.items():
            if author_id not in self.authors:
                logging.error(f"Author {author_id} not found when establishing article relationships")
                continue
                
            author = self.authors[author_id]
            
            for article_id in article_ids:
                if article_id not in self.articles:
                    logging.error(f"Article {article_id} not found for author {author_id}")
                    continue
                    
                article = self.articles[article_id]
                
                # Establish bidirectional relationship
                author.articles.add(article.uuid)
                article.authors.add(author.uuid)
                
                # Verify reciprocal relationship
                if not self._verify_reciprocal_relationship(
                    author.uuid, article.uuid,
                    author.articles, article.authors,
                    "Author-Article"
                ):
                    failed_relationships += 1
                    continue
                
                total_article_relationships += 1
                # Only log significant relationship counts
                if total_article_relationships % 100 == 0:
                    logging.info(f"Established {total_article_relationships} author-article relationships so far...")
        
        # Check for articles without authors
        for article_id, article in self.articles.items():
            if not article.authors:
                articles_without_authors.append(article_id)
        
        logging.info(f"Created {total_variants_created} name variants")
        logging.info(f"Established {total_article_relationships} author-article relationships")
        if failed_relationships > 0:
            logging.error(f"Failed to establish {failed_relationships} reciprocal relationships")
        if articles_without_authors:
            logging.warning(f"Found {len(articles_without_authors)} articles without authors")
            if len(articles_without_authors) <= 5:
                for article_id in articles_without_authors:
                    logging.warning(f"  - {article_id}")
            else:
                for article_id in articles_without_authors[:5]:
                    logging.warning(f"  - {article_id}")
                logging.warning(f"  ... and {len(articles_without_authors)-5} more")

    def _establish_reference_relationships(self) -> None:
        """
        Phase 2: Establish reference relationships using stored temporary data.
        Creates citation contexts and establishes bidirectional relationships with articles and authors.
        """
        total_author_relationships = 0
        failed_relationships = 0
        refs_with_contexts = set()
        
        # First establish Reference ↔ Author relationships
        logging.info("Establishing Reference ↔ Author relationships...")
        total_expected_author_relationships = sum(len(authors) for authors in self._temp_reference_authors.values())
        logging.info(f"Expecting to establish {total_expected_author_relationships} reference-author relationships")
        
        # Track successful relationships by their unique pair
        successful_relationships = set()
        
        for ref_id, author_ids in self._temp_reference_authors.items():
            if ref_id not in self.references:
                logging.error(f"Reference {ref_id} not found when establishing author relationships")
                continue
            
            ref = self.references[ref_id]
            linked_authors = set()
            
            for author_id in author_ids:
                if author_id not in self.authors:
                    logging.warning(f"Author {author_id} not found for reference {ref_id}")
                    continue
                
                author = self.authors[author_id]
                
                # Create unique identifier for this relationship
                relationship_key = f"{ref.uuid}_{author.uuid}"
                
                # Only establish if not already done
                if relationship_key not in successful_relationships:
                    # Establish bidirectional relationship
                    ref.authors.add(author.uuid)
                    author.authored_references.add(ref.uuid)
                    
                    # Verify reciprocal relationship
                    if not self._verify_reciprocal_relationship(
                        ref.uuid, author.uuid,
                        ref.authors, author.authored_references,
                        "Reference-Author"
                    ):
                        failed_relationships += 1
                        continue
                    
                    successful_relationships.add(relationship_key)
                    linked_authors.add(author_id)
                    total_author_relationships += 1
                    
                    # Log progress every 1000 relationships
                    if total_author_relationships % 1000 == 0:
                        logging.info(f"Established {total_author_relationships}/{total_expected_author_relationships} reference-author relationships...")
            
            # Log summary for this reference only if there were issues
            if not linked_authors and author_ids:
                logging.warning(f"Reference {ref_id}: failed to link any of {len(author_ids)} authors")
        
        # Verify reference-author relationships
        total_verified_relationships = len(successful_relationships)
        
        logging.info(f"Established {total_author_relationships} reference-author relationships")
        logging.info(f"Verified {total_verified_relationships} author-reference relationships")
        if failed_relationships > 0:
            logging.error(f"Failed to establish {failed_relationships} reciprocal relationships")
        if total_verified_relationships != total_author_relationships:
            logging.error(f"Mismatch in reference-author relationships: established {total_author_relationships} but verified {total_verified_relationships}")
            
        # Process article references and citation contexts
        self._establish_citation_contexts()

    def _establish_citation_contexts(self) -> None:
        """
        Create citation contexts and establish article-reference relationships.
        This was split out from _establish_reference_relationships for clarity.
        """
        total_article_relationships = 0
        total_citation_contexts = 0
        failed_relationships = 0
        refs_with_contexts = set()
        
        for ref_id, occurrences in self._temp_reference_occurrences.items():
            if ref_id not in self.references:
                logging.error(f"Reference {ref_id} not found when establishing citation contexts")
                continue
                
            ref = self.references[ref_id]
            ref_has_context = False
            
            for occ in occurrences:
                article_id = occ["article_id"]
                if article_id not in self.articles:
                    logging.error(f"Article {article_id} not found for reference {ref_id}")
                    continue
                    
                article = self.articles[article_id]
                
                # Establish article-reference relationship
                article.references.add(ref.uuid)
                ref.citing_articles.add(article.uuid)
                
                # Verify reciprocal relationship
                if not self._verify_reciprocal_relationship(
                    article.uuid, ref.uuid,
                    article.references, ref.citing_articles,
                    "Article-Reference"
                ):
                    failed_relationships += 1
                    continue
                
                total_article_relationships += 1
                
                # Create citation contexts
                for section in occ.get("sections", []):
                    normalized_section = self._normalize_section_name(section)
                    ctx_key = f"{ref.uuid}_{article.uuid}_{normalized_section}"
                    
                    # Only create a new context if one doesn't exist for this combination
                    if ctx_key not in self.citation_contexts:
                        ctx = CitationContext(
                            section=normalized_section,
                            local_ref_id=occ["local_ref_id"],
                            article_uuid=article.uuid,
                            reference_uuid=ref.uuid
                        )
                        self.citation_contexts[ctx_key] = ctx
                        ref.citation_contexts.add(ctx.uuid)
                        article.citation_contexts.add(ctx.uuid)
                        total_citation_contexts += 1
                        ref_has_context = True
            
            if ref_has_context:
                refs_with_contexts.add(ref_id)
        
        logging.info(f"Established {total_article_relationships} reference-article relationships")
        if failed_relationships > 0:
            logging.error(f"Failed to establish {failed_relationships} reciprocal relationships")
        logging.info(f"Created {total_citation_contexts} citation contexts across {len(refs_with_contexts)} references")

    def _normalize_section_name(self, section: str) -> str:
        """
        Normalize section name to standard sections using fuzzy matching.
        
        Maps various section name formats to a standard set of sections:
        - abstract: abstract, summary, synopsis, background summary
        - introduction: introduction, background, intro, overview
        - methods: methods, methodology, materials and methods, experimental, materials, procedure
        - results: results, findings, observations, experimental results
        - discussion: discussion, conclusions, concluding remarks, discussion and conclusions
        - figures: figures, figure, fig, figs, figure legends, figure captions, supplementary figures
        - tables: tables, table, tab, tabs, table legends, table captions, supplementary tables
        
        Args:
            section: Raw section name to normalize
            
        Returns:
            str: Normalized section name
        """
        section = section.lower().strip()
        if not section:
            return "methods"  # Default section
            
        standard_sections = {
            'abstract': ['abstract', 'summary', 'synopsis', 'background summary', 'article summary'],
            'introduction': ['introduction', 'background', 'intro', 'overview', 'background and motivation'],
            'methods': ['methods', 'methodology', 'materials and methods', 'experimental', 'procedure', 
                       'materials', 'experimental methods', 'experimental setup'],
            'results': ['results', 'findings', 'observations', 'result', 'experimental results', 
                       'results and analysis'],
            'discussion': ['discussion', 'conclusions', 'concluding remarks', 'conclusion', 
                          'discussion and conclusions', 'summary and conclusions'],
            'figures': ['figures', 'figure', 'fig', 'figs', 'figure legends', 'figure captions',
                       'supplementary figures', 'supplementary figure', 'figure s'],
            'tables': ['tables', 'table', 'tab', 'tabs', 'table legends', 'table captions',
                      'supplementary tables', 'supplementary table', 'table s']
        }
        
        # First try exact match with standard section name
        if section in standard_sections:
            return section
            
        # Then try exact match with variants
        for standard, variants in standard_sections.items():
            if section in variants:
                return standard
            
            # Then try partial matches
            for variant in variants:
                # Check if variant is contained in section name
                if variant in section:
                    return standard
                # Check if section name is contained in variant
                if section in variant:
                    return standard
        
        # If no match found, try to infer from content
        if any(word in section for word in ['method', 'procedure', 'protocol']):
            return 'methods'
        if any(word in section for word in ['result', 'finding', 'observation']):
            return 'results'
        if any(word in section for word in ['discuss', 'conclu']):
            return 'discussion'
        if any(word in section for word in ['fig', 'figure']):
            return 'figures'
        if any(word in section for word in ['tab', 'table']):
            return 'tables'
        if any(word in section for word in ['intro', 'background']):
            return 'introduction'
        if any(word in section for word in ['abstract', 'summary']):
            return 'abstract'
        
        return 'methods'  # Default to methods if no match found

    def _establish_relationships(self) -> Relationships:
        """
        Second pass: Create relationship objects and update cross-references
        using the UUID mappings from the first pass.
        
        The order of relationship establishment is important:
        1. Author relationships first (including name variants)
        2. Reference relationships (including citation contexts)
        3. NER relationships
        
        Returns:
            Relationships: Container with all relationship objects created
        """
        # Initialize relationship statistics
        relationship_stats = defaultdict(lambda: {
            "outgoing": defaultdict(int),
            "incoming": defaultdict(int)
        })
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            # Create tasks for each phase
            author_task = progress.add_task("Establishing author relationships...", total=len(self.authors))
            ref_task = progress.add_task("Establishing reference relationships...", total=len(self.references), visible=False)
            ner_task = progress.add_task("Establishing NER relationships...", total=len(self.ner_objects), visible=False)
            
            # Phase 1: Author relationships (including name variants)
            self._establish_author_relationships()
            progress.update(author_task, completed=len(self.authors))
            
            # Phase 2: Reference relationships (including citation contexts)
            progress.update(ref_task, visible=True)
            self._establish_reference_relationships()
            progress.update(ref_task, completed=len(self.references))
            
            # Phase 3: NER relationships
            progress.update(ner_task, visible=True)
            self._establish_ner_relationships()
            progress.update(ner_task, completed=len(self.ner_objects))
        
        # Update relationship statistics
        self._update_relationship_stats(relationship_stats)
        
        # Log relationship statistics
        self._log_relationship_stats(relationship_stats)
        
        # Clean up temporary data
        self._cleanup_temp_data()
        
        # Return relationship objects
        return Relationships(
            name_variants=list(self.name_variants.values()),
            citation_contexts=list(self.citation_contexts.values()),
            ner_scores=list(self.ner_scores.values())
        )

    def _update_relationship_stats(self, stats: Dict) -> None:
        """Update relationship statistics after establishment."""
        # Initialize stats structure for each collection
        collections = [
            "articles", "authors", "references", "named_entities",
            "citation_contexts", "name_variants", "ner_scores"
        ]
        for collection in collections:
            if collection not in stats:
                stats[collection] = {
                    "total": 0,
                    "cross_references": defaultdict(int)
                }

        # Count total items in each collection
        stats["articles"]["total"] = len(self.articles)
        stats["authors"]["total"] = len(self.authors)
        stats["references"]["total"] = len(self.references)
        stats["named_entities"]["total"] = len(self.ner_objects)
        stats["citation_contexts"]["total"] = len(self.citation_contexts)
        stats["name_variants"]["total"] = len(self.name_variants)
        stats["ner_scores"]["total"] = len(self.ner_scores)

        # Count cross-references for articles
        for article in self.articles.values():
            stats["articles"]["cross_references"]["authors"] += len(article.authors)
            stats["articles"]["cross_references"]["references"] += len(article.references)
            stats["articles"]["cross_references"]["named_entities"] += len(article.named_entities)
            stats["articles"]["cross_references"]["ner_scores"] += len(article.ner_scores)
            stats["articles"]["cross_references"]["citation_contexts"] += len(article.citation_contexts)

        # Count cross-references for authors
        for author in self.authors.values():
            stats["authors"]["cross_references"]["articles"] += len(author.articles)
            stats["authors"]["cross_references"]["references"] += len(author.authored_references)
            stats["authors"]["cross_references"]["name_variants"] += len(author.name_variants)

        # Count cross-references for references
        for ref in self.references.values():
            stats["references"]["cross_references"]["authors"] += len(ref.authors)
            stats["references"]["cross_references"]["articles"] += len(ref.citing_articles)
            stats["references"]["cross_references"]["citation_contexts"] += len(ref.citation_contexts)

        # Count cross-references for named entities
        for ner in self.ner_objects.values():
            stats["named_entities"]["cross_references"]["articles"] += len(ner.articles)
            stats["named_entities"]["cross_references"]["ner_scores"] += len(ner.ner_scores)

        # Count cross-references for citation contexts
        for ctx in self.citation_contexts.values():
            stats["citation_contexts"]["cross_references"]["articles"] += 1  # Each context has one article
            stats["citation_contexts"]["cross_references"]["references"] += 1  # Each context has one reference

        # Count cross-references for name variants
        for variant in self.name_variants.values():
            stats["name_variants"]["cross_references"]["authors"] += 1  # Each variant has one author

        # Count cross-references for NER scores
        for score in self.ner_scores.values():
            stats["ner_scores"]["cross_references"]["articles"] += 1  # Each score has one article
            stats["ner_scores"]["cross_references"]["named_entities"] += 1  # Each score has one entity

        # Verify citation context counts match
        ref_to_ctx = sum(len(ref.citation_contexts) for ref in self.references.values())
        ctx_to_ref = len(self.citation_contexts)  # Each context has exactly one reference
        if ref_to_ctx != ctx_to_ref:
            logging.error(f"Citation context count mismatch: {ref_to_ctx} from references vs {ctx_to_ref} total contexts")

    def _log_relationship_stats(self, stats: Dict) -> None:
        """Log relationship statistics after establishment."""
        logging.info("\nRelationship Statistics:")
        for collection, data in stats.items():
            logging.info(f"\n{collection.replace('_', ' ').title()} ({data['total']:,} total):")
            if data["cross_references"]:
                logging.info("   Cross References:")
                for target, count in data["cross_references"].items():
                    logging.info(f"      → {target}: {count:,}")

    def _cleanup_temp_data(self) -> None:
        """Clean up temporary data after relationship establishment."""
        self._temp_ner_scores.clear()
        self._temp_author_variants.clear()
        self._temp_reference_occurrences.clear()
        self._temp_article_authors.clear()
        self._temp_reference_authors.clear()

    def _validate_relationships(self) -> bool:
        """
        Validate all relationships between entities for consistency and completeness.
        Checks that reciprocal relationships have matching counts.
        """
        valid = True
        stats = defaultdict(lambda: {"total": 0, "cross_references": defaultdict(int)})
        
        # Gather relationship statistics
        self._update_relationship_stats(stats)
        
        # Check reciprocal relationships
        logging.info("\nValidating reciprocal relationships:")
        
        # Define all reciprocal relationships to check
        reciprocal_pairs = [
            ("articles", "authors"),
            ("articles", "references"),
            ("articles", "named_entities"),
            ("articles", "ner_scores"),
            ("articles", "citation_contexts"),
            ("authors", "references"),
            ("authors", "name_variants"),
            ("references", "citation_contexts"),
            ("named_entities", "ner_scores")
        ]
        
        for collection1, collection2 in reciprocal_pairs:
            outgoing = stats[collection1]["cross_references"].get(collection2, 0)
            incoming = stats[collection2]["cross_references"].get(collection1, 0)
            
            if outgoing != incoming:
                logging.error(
                    f"Asymmetric relationship found between {collection1} and {collection2}:\n"
                    f"   {collection1} → {collection2}: {outgoing:,}\n"
                    f"   {collection2} → {collection1}: {incoming:,}"
                )
                valid = False
            else:
                logging.info(
                    f"Verified {collection1} ↔ {collection2}: {outgoing:,} references"
                )
        
        return valid
    
    def get_article_summary(self, article_id: str) -> Dict[str, Any]:
        """Get summary of article with all its relationships."""
        if article_id not in self.articles:
            raise KeyError(f"Article not found: {article_id}")
            
        article = self.articles[article_id]
        article_uuid = self.uuid_maps["articles"][article_id]
        
        return {
            "filename": article.filename,
            "authors": [
                {
                    "name": self.authors[author_id].canonical_name,
                    "email": self.authors[author_id].email
                }
                for author_id, author_uuid in self.uuid_maps["authors"].items()
                if author_uuid in article.authors
            ],
            "references": [
                {
                    "title": self.references[ref_id].title or self.references[ref_id].raw_reference,
                    "contexts": [
                        {
                            "section": ctx.section,
                            "local_id": ctx.local_ref_id
                        }
                        for ctx in self.citation_contexts.values()
                        if ctx.reference_uuid == ref_uuid and ctx.article_uuid == article_uuid
                    ]
                }
                for ref_id, ref_uuid in self.uuid_maps["references"].items()
                if ref_uuid in self.get_article_references(article_uuid)
            ],
            "named_entities": [
                {
                    "name": self.ner_objects[ner_id].name,
                    "type": self.ner_objects[ner_id].type,
                    "score": article.ner_scores[ner_uuid]
                }
                for ner_id, ner_uuid in self.uuid_maps["ner_objects"].items()
                if ner_uuid in article.ner_scores
            ]
        }
    
    def extract_subset(self, size: int, seed_article: Optional[str] = None) -> 'LiteratureDataManager':
        """
        Extract connected subset of data starting from seed article.
        First collects all connected entities following dependency order,
        then prunes relationships to maintain closure within the subset.
        
        Args:
            size: Target number of articles to include
            seed_article: Starting article ID (random if None)
            
        Returns:
            LiteratureDataManager: New instance with subset of data
        """
        # Create new instance for subset
        subset = LiteratureDataManager.__new__(LiteratureDataManager)
        subset.data_dir = self.data_dir
        
        # Initialize empty collections
        subset.articles = {}
        subset.authors = {}
        subset.references = {}
        subset.ner_objects = {}
        subset.name_variants = {}
        subset.citation_contexts = {}
        subset.ner_scores = {}
        subset.uuid_maps = {k: {} for k in self.uuid_maps.keys()}
        
        # Select seed article if not provided
        if not seed_article:
            seed_article = random.choice(list(self.articles.keys()))
        elif seed_article not in self.articles:
            raise ValueError(f"Seed article not found: {seed_article}")
            
        # Track articles to process and those already included
        articles_to_process = {seed_article}
        included_articles = set()
        
        # Collection phase: Follow relationships to gather connected entities
        while articles_to_process and len(included_articles) < size:
            current_article_id = articles_to_process.pop()
            if current_article_id in included_articles:
                continue
                
            # 1. Add article and record its UUID mapping
            article = self.articles[current_article_id]
            subset.articles[current_article_id] = article
            subset.uuid_maps["articles"][current_article_id] = article.uuid
            included_articles.add(current_article_id)
            
            # 2. Follow author relationships
            for author_uuid in article.authors:
                author_id = next(aid for aid, uuid in self.uuid_maps["authors"].items() 
                               if uuid == author_uuid)
                if author_id not in subset.authors:
                    author = self.authors[author_id]
                    subset.authors[author_id] = author
                    subset.uuid_maps["authors"][author_id] = author.uuid
                    
                    # Queue author's other articles for processing
                    articles_to_process.update(
                        aid for aid, uuid in self.uuid_maps["articles"].items()
                        if uuid in author.articles and aid not in included_articles
                    )
                    
                    # Add references authored by this author
                    for ref_uuid in author.authored_references:
                        ref_id = next(rid for rid, uuid in self.uuid_maps["references"].items() 
                                    if uuid == ref_uuid)
                        if ref_id not in subset.references:
                            ref = self.references[ref_id]
                            subset.references[ref_id] = ref
                            subset.uuid_maps["references"][ref_id] = ref.uuid
                            
                            # Queue articles that cite this reference
                            articles_to_process.update(
                                aid for aid, uuid in self.uuid_maps["articles"].items()
                                if uuid in ref.citing_articles and aid not in included_articles
                            )
            
            # 3. Follow reference relationships
            for ref_uuid in article.references:
                ref_id = next(rid for rid, uuid in self.uuid_maps["references"].items() 
                             if uuid == ref_uuid)
                if ref_id not in subset.references:
                    ref = self.references[ref_id]
                    subset.references[ref_id] = ref
                    subset.uuid_maps["references"][ref_id] = ref.uuid
                    
                    # Queue articles that cite this reference
                    articles_to_process.update(
                        aid for aid, uuid in self.uuid_maps["articles"].items()
                        if uuid in ref.citing_articles and aid not in included_articles
                    )
            
            # 4. Follow NER relationships
            for ner_uuid in article.named_entities:
                ner_id = next(nid for nid, uuid in self.uuid_maps["ner_objects"].items() 
                             if uuid == ner_uuid)
                if ner_id not in subset.ner_objects:
                    ner = self.ner_objects[ner_id]
                    subset.ner_objects[ner_id] = ner
                    subset.uuid_maps["ner_objects"][ner_id] = ner.uuid
                    
                    # Queue articles that mention this entity
                    articles_to_process.update(
                        aid for aid, uuid in self.uuid_maps["articles"].items()
                        if uuid in ner.articles and aid not in included_articles
                    )
                    
            # If we need more articles and our queue is empty, add a random one
            if len(included_articles) < size and not articles_to_process:
                available_articles = set(self.articles.keys()) - included_articles
                if available_articles:
                    next_article = random.choice(list(available_articles))
                    articles_to_process.add(next_article)
        
        # Now collect relationship entities based on primary entities
        subset._collect_relationship_entities(self)
        
        # Prune phase: Remove relationships to entities not in our subset
        subset._prune_relationships()
        
        logging.info(f"\nExtracted connected subset:")
        logging.info(f"  Articles: {len(subset.articles)}/{len(self.articles)}")
        logging.info(f"  Authors: {len(subset.authors)}/{len(self.authors)}")
        logging.info(f"  References: {len(subset.references)}/{len(self.references)}")
        logging.info(f"  NER Objects: {len(subset.ner_objects)}/{len(self.ner_objects)}")
        logging.info(f"  Name Variants: {len(subset.name_variants)}/{len(self.name_variants)}")
        logging.info(f"  Citation Contexts: {len(subset.citation_contexts)}/{len(self.citation_contexts)}")
        logging.info(f"  NER Scores: {len(subset.ner_scores)}/{len(self.ner_scores)}")
        
        return subset
    
    def _collect_relationship_entities(self, source: 'LiteratureDataManager'):
        """
        Collect relationship entities based on primary entities in the subset.
        
        Args:
            source: Original LiteratureDataManager instance to copy from
        """
        # 1. Collect name variants for included authors
        valid_author_uuids = {author.uuid for author in self.authors.values()}
        for variant_key, variant in source.name_variants.items():
            if variant.author_uuid in valid_author_uuids:
                self.name_variants[variant_key] = variant
                self.uuid_maps["name_variants"][variant_key] = variant.uuid
        
        # 2. Collect citation contexts for included articles and references
        valid_article_uuids = {article.uuid for article in self.articles.values()}
        valid_ref_uuids = {ref.uuid for ref in self.references.values()}
        for ctx_key, ctx in source.citation_contexts.items():
            if (ctx.article_uuid in valid_article_uuids and 
                ctx.reference_uuid in valid_ref_uuids):
                self.citation_contexts[ctx_key] = ctx
                self.uuid_maps["citation_contexts"][ctx_key] = ctx.uuid
        
        # 3. Collect NER scores for included articles and entities
        valid_ner_uuids = {ner.uuid for ner in self.ner_objects.values()}
        for score_key, score in source.ner_scores.items():
            if (score.article_uuid in valid_article_uuids and 
                score.entity_uuid in valid_ner_uuids):
                self.ner_scores[score_key] = score
                self.uuid_maps["ner_scores"][score_key] = score.uuid

    def _prune_relationships(self):
        """
        Prune relationships in the subset to maintain closure.
        Removes any relationships that point to entities not in the subset.
        """
        # Get sets of valid UUIDs in our subset
        valid_article_uuids = {article.uuid for article in self.articles.values()}
        valid_author_uuids = {author.uuid for author in self.authors.values()}
        valid_ref_uuids = {ref.uuid for ref in self.references.values()}
        valid_ner_uuids = {ner.uuid for ner in self.ner_objects.values()}
        valid_score_uuids = {score.uuid for score in self.ner_scores.values()}
        valid_ctx_uuids = {ctx.uuid for ctx in self.citation_contexts.values()}
        valid_variant_uuids = {var.uuid for var in self.name_variants.values()}
        
        # Prune article relationships
        for article in self.articles.values():
            article.authors &= valid_author_uuids
            article.references &= valid_ref_uuids
            article.named_entities &= valid_ner_uuids
            article.ner_scores &= valid_score_uuids
            article.citation_contexts &= valid_ctx_uuids
        
        # Prune author relationships
        for author in self.authors.values():
            author.articles &= valid_article_uuids
            author.authored_references &= valid_ref_uuids
            author.name_variants &= valid_variant_uuids
        
        # Prune reference relationships
        for ref in self.references.values():
            ref.authors &= valid_author_uuids
            ref.citing_articles &= valid_article_uuids
            ref.citation_contexts &= valid_ctx_uuids
            
            # Verify reciprocal relationship with authors
            for author_uuid in ref.authors:
                author_id = next(aid for aid, uuid in self.uuid_maps["authors"].items() 
                               if uuid == author_uuid)
                author = self.authors[author_id]
                if ref.uuid not in author.authored_references:
                    logging.warning(f"Found non-reciprocal author-reference relationship: {author_id} -> {ref.uuid}")
                    ref.authors.remove(author_uuid)
        
        # Prune NER object relationships
        for ner in self.ner_objects.values():
            ner.articles &= valid_article_uuids
            ner.ner_scores &= valid_score_uuids
        
        # Remove any orphaned relationship entities
        self._remove_orphaned_relationships()

    def _remove_orphaned_relationships(self):
        """Remove any relationship entities that no longer have valid connections."""
        # Get valid UUIDs
        valid_article_uuids = {article.uuid for article in self.articles.values()}
        valid_author_uuids = {author.uuid for author in self.authors.values()}
        valid_ref_uuids = {ref.uuid for ref in self.references.values()}
        valid_ner_uuids = {ner.uuid for ner in self.ner_objects.values()}
        
        # Remove name variants without valid authors
        self.name_variants = {
            key: var for key, var in self.name_variants.items()
            if var.author_uuid in valid_author_uuids
        }
        
        # Remove citation contexts without valid articles or references
        self.citation_contexts = {
            key: ctx for key, ctx in self.citation_contexts.items()
            if (ctx.article_uuid in valid_article_uuids and 
                ctx.reference_uuid in valid_ref_uuids)
        }
        
        # Remove NER scores without valid articles or entities
        self.ner_scores = {
            key: score for key, score in self.ner_scores.items()
            if (score.article_uuid in valid_article_uuids and 
                score.entity_uuid in valid_ner_uuids)
        }
        
        # Update UUID maps to match pruned collections
        self.uuid_maps["name_variants"] = {
            key: var.uuid for key, var in self.name_variants.items()
        }
        self.uuid_maps["citation_contexts"] = {
            key: ctx.uuid for key, ctx in self.citation_contexts.items()
        }
        self.uuid_maps["ner_scores"] = {
            key: score.uuid for key, score in self.ner_scores.items()
        }

    def _establish_ner_relationships(self) -> None:
        """
        Phase 1: Establish NER relationships using stored temporary data.
        Creates NER score objects and establishes bidirectional relationships
        between articles and named entities.
        """
        total_scores_created = 0
        total_entity_relationships = 0
        failed_relationships = 0
        score_values = []  # Track all score values for debugging
        
        # Process NER scores and relationships
        for term, article_scores in self._temp_ner_scores.items():
            if term not in self.ner_objects:
                logging.error(f"NER object {term} not found when establishing relationships")
                continue
                
            ner = self.ner_objects[term]
            
            for article_id, score in article_scores.items():
                if article_id not in self.articles:
                    logging.error(f"Article {article_id} not found for NER term {term}")
                    continue
                    
                article = self.articles[article_id]
                score_values.append(float(score))  # Track the score value
                
                # Create score object
                score_obj = NERArticleScore(
                    score=float(score),  # Ensure score is float
                    article_uuid=article.uuid,
                    entity_uuid=ner.uuid
                )
                
                # Store score object
                score_key = f"{ner.uuid}_{article.uuid}"
                self.ner_scores[score_key] = score_obj
                self.uuid_maps["ner_scores"][score_key] = score_obj.uuid
                
                # Establish bidirectional relationships via UUIDs only
                article.ner_scores.add(score_obj.uuid)
                article.named_entities.add(ner.uuid)
                ner.ner_scores.add(score_obj.uuid)
                ner.articles.add(article.uuid)
                
                # Verify reciprocal relationships
                if not self._verify_reciprocal_relationship(
                    article.uuid, ner.uuid,
                    article.named_entities, ner.articles,
                    "Article-Entity"
                ):
                    failed_relationships += 1
                    continue
                
                total_scores_created += 1
                total_entity_relationships += 1
        
        # Log score statistics
        if score_values:
            logging.info(f"NER Score statistics:")
            logging.info(f"  Min score: {min(score_values):.2f}")
            logging.info(f"  Max score: {max(score_values):.2f}")
            logging.info(f"  Avg score: {sum(score_values)/len(score_values):.2f}")
        
        logging.info(f"Created {total_scores_created} NER scores")
        logging.info(f"Established {total_entity_relationships} article-entity relationships")
        if failed_relationships > 0:
            logging.error(f"Failed to establish {failed_relationships} reciprocal relationships")

    def generate_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about the loaded data.
        
        Calculates detailed statistics across all aspects of the dataset:
        - Collection sizes (articles, authors, references, etc.)
        - Distribution statistics (items per collection, relationships)
        - Section citation patterns
        - Named entity type distributions
        - Relationship patterns and network analysis
        
        Returns:
            Dict[str, Any]: Hierarchical dictionary of statistics
        """
        # Build author variant counts first to avoid repeated iterations
        author_variant_counts = defaultdict(int)
        for variant in self.name_variants.values():
            author_variant_counts[variant.author_uuid] += 1
        
        # Get author statistics
        authors_per_article = [len(a.authors) for a in self.articles.values()]
        articles_per_author = [len(a.articles) for a in self.authors.values()]
        
        # Count references per article
        ref_counts = []
        for article in self.articles.values():
            count = self.get_article_reference_count(article.uuid)
            ref_counts.append(count)
        
        # Get other distributions
        ner_terms_per_article = [len(a.ner_scores) for a in self.articles.values()]
        variants_per_author = [author_variant_counts[author.uuid] for author in self.authors.values()]
        
        stats = {
            "collections": {
                "articles": len(self.articles),
                "authors": len(self.authors),
                "references": len(self.references),
                "ner_objects": len(self.ner_objects),
                "name_variants": len(self.name_variants),
                "citation_contexts": len(self.citation_contexts),
                "ner_scores": len(self.ner_scores)
            },
            "distributions": {
                "authors_per_article": self._calculate_distribution(authors_per_article),
                "articles_per_author": self._calculate_distribution(articles_per_author),
                "references_per_article": self._calculate_distribution(ref_counts),
                "ner_terms_per_article": self._calculate_distribution(ner_terms_per_article),
                "variants_per_author": self._calculate_distribution(variants_per_author)
            },
            "section_citations": self._analyze_section_citations(),
            "ner_type_counts": self._count_ner_types(),
            "relationship_patterns": self._analyze_relationship_patterns()
        }
        
        return stats

    def print_statistics(self, detail_level: str = "summary", use_rich: bool = True) -> None:
        """
        Generate and print statistics about the loaded data.
        
        Args:
            detail_level: Level of detail to display
                         "summary": Basic collection sizes and distributions
                         "detailed": Adds section and NER type breakdowns
                         "full": Adds relationship pattern analysis
            use_rich: Whether to use rich formatting for output
        """
        stats = self.generate_statistics()
        
        if not use_rich:
            print(self.format_statistics(stats, detail_level))
            return
        
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Collection sizes table
        size_table = Table(title="Collection Sizes")
        size_table.add_column("Collection", style="cyan")
        size_table.add_column("Count", justify="right", style="green")
        
        for name, count in stats["collections"].items():
            size_table.add_row(name.replace("_", " ").title(), f"{count:,}")
        
        # Distribution table
        dist_table = Table(title="Distribution Statistics")
        dist_table.add_column("Metric")
        dist_table.add_column("Min", justify="right")
        dist_table.add_column("Max", justify="right")
        dist_table.add_column("Average", justify="right")
        dist_table.add_column("Median", justify="right")
        
        for dist_name, dist in stats["distributions"].items():
            dist_table.add_row(
                dist_name.replace("_", " ").title(),
                str(dist["min"]),
                str(dist["max"]),
                f"{dist['avg']:.1f}",
                str(dist["median"])
            )
        
        # Print tables
        console.print("\n[bold]Literature Database Statistics[/bold]\n")
        console.print(size_table)
        console.print("\n")
        console.print(dist_table)
        
        if detail_level != "summary":
            # Section citations
            section_table = Table(title="Citations by Section")
            section_table.add_column("Section", style="cyan")
            section_table.add_column("Citation Count", justify="right", style="green")
            
            for section, count in sorted(
                stats["section_citations"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                section_table.add_row(section, f"{count:,}")
            
            # NER types
            ner_table = Table(title="Named Entity Types")
            ner_table.add_column("Type", style="cyan")
            ner_table.add_column("Count", justify="right", style="green")
            
            for ner_type, count in sorted(
                stats["ner_type_counts"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                ner_table.add_row(ner_type, f"{count:,}")
            
            console.print("\n")
            console.print(section_table)
            console.print("\n")
            console.print(ner_table)
        
        if detail_level == "full":
            patterns = stats["relationship_patterns"]
            
            # Author collaboration panel
            collab_text = (
                f"Total Collaborations: {patterns['author_collaboration']['total_collaborations']:,}\n"
                f"Authors with Collaborators: {patterns['author_collaboration']['authors_with_collaborators']:,}\n"
                f"Max Collaborators per Author: {patterns['author_collaboration']['max_collaborators']:,}"
            )
            console.print("\n")
            console.print(Panel(collab_text, title="Author Collaboration"))
            
            # Citation network panel
            citation_text = (
                f"Total Citations: {patterns['citation_network']['total_citations']:,}\n"
                f"Articles with Citations: {patterns['citation_network']['articles_with_citations']:,}\n"
                f"Unique References Cited: {patterns['citation_network']['references_cited']:,}"
            )
            console.print("\n")
            console.print(Panel(citation_text, title="Citation Network"))
            
            # NER coverage panel
            ner_text = (
                f"Total NER Scores: {patterns['ner_coverage']['total_scores']:,}\n"
                f"Articles with NER: {patterns['ner_coverage']['articles_with_ner']:,}\n"
                f"Score Distribution: "
                f"min={patterns['ner_coverage']['score_distribution']['min']:.2f}, "
                f"max={patterns['ner_coverage']['score_distribution']['max']:.2f}, "
                f"avg={patterns['ner_coverage']['score_distribution']['avg']:.2f}"
            )
            console.print("\n")
            console.print(Panel(ner_text, title="NER Coverage"))

    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics for a list of values."""
        if not values:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        sorted_vals = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "median": sorted_vals[len(sorted_vals) // 2]
        }

    def _analyze_section_citations(self) -> Dict[str, int]:
        """
        Analyze distribution of citations across article sections.
        Only counts citations from citation contexts to avoid double counting.
        
        Returns:
            Dict[str, int]: Mapping of section names to citation counts
        """
        section_counts = defaultdict(int)
        
        # Count citations by section from citation contexts only
        for ctx in self.citation_contexts.values():
            section_counts[ctx.section.lower()] += 1
        
        return dict(section_counts)

    def _count_ner_types(self) -> Dict[str, int]:
        """Count occurrences of each NER type."""
        type_counts = defaultdict(int)
        for ner in self.ner_objects.values():
            type_counts[ner.type] += 1
        return dict(type_counts)

    def _analyze_relationship_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in entity relationships."""
        return {
            "author_collaboration": self._analyze_author_collaboration(),
            "citation_network": self._analyze_citation_network(),
            "ner_coverage": self._analyze_ner_coverage()
        }

    def _analyze_author_collaboration(self) -> Dict[str, Any]:
        """Analyze patterns in author collaboration."""
        collaboration_stats = {
            "total_collaborations": 0,
            "authors_with_collaborators": 0,
            "max_collaborators": 0
        }
        
        author_collaborators = defaultdict(set)
        for article in self.articles.values():
            author_list = list(article.authors)
            if len(author_list) > 1:
                collaboration_stats["total_collaborations"] += 1
                for i in range(len(author_list) - 1):
                    author1 = author_list[i]
                    author2 = author_list[i + 1]
                    author_collaborators[author1].add(author2)
                    author_collaborators[author2].add(author1)
        
        collaboration_stats["authors_with_collaborators"] = sum(
            1 for collabs in author_collaborators.values() if collabs
        )
        collaboration_stats["max_collaborators"] = max(
            (len(collabs) for collabs in author_collaborators.values()),
            default=0
        )
        
        return collaboration_stats

    def _analyze_citation_network(self) -> Dict[str, Any]:
        """Analyze patterns in citation network."""
        return {
            "total_citations": len(self.citation_contexts),
            "articles_with_citations": len(set(ctx.article_uuid for ctx in self.citation_contexts.values())),
            "references_cited": len(set(ctx.reference_uuid for ctx in self.citation_contexts.values())),
            "section_distribution": self._analyze_section_citations(),
            "avg_citations_per_article": len(self.citation_contexts) / len(self.articles) if self.articles else 0
        }

    def _analyze_ner_coverage(self) -> Dict[str, Any]:
        """Analyze patterns in NER coverage."""
        all_scores = [float(score.score) for score in self.ner_scores.values()]
        return {
            "total_scores": len(self.ner_scores),
            "articles_with_ner": len(set(score.article_uuid for score in self.ner_scores.values())),
            "ner_types": self._count_ner_types(),
            "score_distribution": self._calculate_distribution(all_scores)
        }

    def format_statistics(self, stats: Dict[str, Any], detail_level: str = "summary") -> str:
        """
        Format statistics for CLI display.
        
        Args:
            stats: Statistics dictionary from generate_statistics()
            detail_level: Level of detail to display ("summary", "detailed", or "full")
            
        Returns:
            Formatted string for display
        """
        lines = ["Literature Database Statistics", "========================="]
        
        # Collection sizes
        lines.extend([
            "\nCollection Sizes:",
            f"  Articles: {stats['collections']['articles']:,}",
            f"  Authors: {stats['collections']['authors']:,}",
            f"  References: {stats['collections']['references']:,}",
            f"  Named Entities: {stats['collections']['ner_objects']:,}"
        ])
        
        if detail_level != "summary":
            lines.extend([
                f"  Name Variants: {stats['collections']['name_variants']:,}",
                f"  Citation Contexts: {stats['collections']['citation_contexts']:,}",
                f"  NER Scores: {stats['collections']['ner_scores']:,}"
            ])
        
        # Distribution statistics
        lines.extend(["\nDistribution Statistics (min/max/avg/median):"])
        for dist_name, dist in stats["distributions"].items():
            lines.append(
                f"  {dist_name.replace('_', ' ').title()}: "
                f"{dist['min']}/{dist['max']}/{dist['avg']:.1f}/{dist['median']}"
            )
        
        if detail_level != "summary":
            # Section citations
            lines.extend(["\nCitations by Section:"])
            for section, count in sorted(
                stats["section_citations"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"  {section}: {count:,} citations")
            
            # NER types
            lines.extend(["\nNamed Entity Types:"])
            for ner_type, count in sorted(
                stats["ner_type_counts"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                lines.append(f"  {ner_type}: {count:,}")
        
        if detail_level == "full":
            patterns = stats["relationship_patterns"]
            
            lines.extend([
                "\nAuthor Collaboration:",
                f"  Total Collaborations: {patterns['author_collaboration']['total_collaborations']:,}",
                f"  Authors with Collaborators: {patterns['author_collaboration']['authors_with_collaborators']:,}",
                f"  Max Collaborators per Author: {patterns['author_collaboration']['max_collaborators']:,}"
            ])
            
            lines.extend([
                "\nCitation Network:",
                f"  Total Citations: {patterns['citation_network']['total_citations']:,}",
                f"  Articles with Citations: {patterns['citation_network']['articles_with_citations']:,}",
                f"  Unique References Cited: {patterns['citation_network']['references_cited']:,}"
            ])
            
            lines.extend([
                "\nNER Coverage:",
                f"  Total NER Scores: {patterns['ner_coverage']['total_scores']:,}",
                f"  Articles with NER: {patterns['ner_coverage']['articles_with_ner']:,}",
                "  Score Distribution: "
                f"min={patterns['ner_coverage']['score_distribution']['min']:.2f}, "
                f"max={patterns['ner_coverage']['score_distribution']['max']:.2f}, "
                f"avg={patterns['ner_coverage']['score_distribution']['avg']:.2f}"
            ])
        
        return "\n".join(lines)

    def get_article_references(self, article_uuid: str) -> Set[str]:
        """
        Get the set of reference UUIDs cited by an article.
        
        Retrieves all unique references that are cited in the article by examining
        citation contexts. Each reference is included only once, regardless of how
        many times it is cited in the article.
        
        Args:
            article_uuid: UUID of the article to get references for
            
        Returns:
            Set[str]: Set of reference UUIDs cited by the article
        """
        return {ctx.reference_uuid for ctx in self.citation_contexts.values() 
                if ctx.article_uuid == article_uuid}

    def get_article_reference_count(self, article_uuid: str) -> int:
        """
        Get the number of unique references cited by an article.
        
        Counts how many different references are cited in the article,
        regardless of how many times each reference is cited.
        
        Args:
            article_uuid: UUID of the article to count references for
            
        Returns:
            int: Number of unique references cited by the article
        """
        return len(self.get_article_references(article_uuid))

    def create_and_analyze_subset(self, size: int, seed_article: Optional[str] = None) -> Tuple['LiteratureDataManager', Dict[str, Any]]:
        """
        Create and analyze a subset of the data.
        
        This is a convenience method that:
        1. Creates a subset using extract_subset
        2. Analyzes it using analyze_subset
        3. Verifies it's ready for import
        
        Args:
            size: Target number of articles
            seed_article: Optional starting article
            
        Returns:
            Tuple containing:
            - LiteratureDataManager: The created subset
            - Dict[str, Any]: Analysis results
        """
        # Phase 1: Create subset
        logging.info("\nPhase 1: Creating and extracting subset...")
        subset = self.extract_subset(size, seed_article)
        
        # Phase 2: Analyze subset
        logging.info("\nPhase 2: Analyzing subset...")
        analysis = self.analyze_subset(subset)
        
        # Phase 3: Verify import readiness
        logging.info("\nPhase 3: Verifying subset for import...")
        is_ready, warnings = subset.verify_for_import()
        analysis["import_verification"] = {
            "is_ready": is_ready,
            "warnings": warnings
        }
        
        return subset, analysis

    def verify_for_import(self) -> Tuple[bool, List[str]]:
        """
        Verify that this dataset is ready for import.
        
        Checks:
        1. Minimum collection requirements
        2. Relationship integrity
        3. UUID consistency
        4. Data completeness
        
        Returns:
            Tuple containing:
            - bool: True if ready for import
            - List[str]: Any warnings or issues found
        """
        warnings = []
        
        # Check for minimum requirements
        if not self.articles:
            warnings.append("No articles in subset")
        if not self.authors:
            warnings.append("No authors in subset")
        
        # Verify relationship integrity
        for article in self.articles.values():
            # Check for articles without authors
            if not article.authors:
                warnings.append(f"Article {article.filename} has no authors")
            
            # Check for invalid references
            for ref_uuid in article.references:
                if not any(r_uuid == ref_uuid for r_uuid in self.uuid_maps["references"].values()):
                    warnings.append(f"Article {article.filename} has invalid reference {ref_uuid}")
        
        # Verify UUID consistency
        for collection_name in ["articles", "authors", "references", "ner_objects"]:
            collection = getattr(self, collection_name)
            uuid_map = self.uuid_maps[collection_name]
            if len(collection) != len(uuid_map):
                warnings.append(f"UUID map mismatch for {collection_name}")
        
        # Check for data completeness
        for article in self.articles.values():
            if not article.abstract and not article.introduction:
                warnings.append(f"Article {article.filename} missing both abstract and introduction")
        
        return len(warnings) == 0, warnings

    def analyze_subset(self, subset: 'LiteratureDataManager') -> Dict[str, Any]:
        """
        Analyze the composition and relationships of a subset of the data.
        
        Uses existing analysis infrastructure to examine:
        1. Collection sizes and coverage
        2. Relationship integrity
        3. Distribution statistics
        4. Network characteristics
        
        Args:
            subset: LiteratureDataManager instance containing the subset to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including collection coverage, 
                           relationship integrity, and consistency checks
        """
        logging.info("\nAnalyzing data subset...")
        
        # 1. Compare collection sizes
        collection_coverage = {}
        logging.info("\nCollection Coverage:")
        for collection_name in ["articles", "authors", "references", "ner_objects", 
                              "name_variants", "citation_contexts", "ner_scores"]:
            total = len(getattr(self, collection_name))
            subset_size = len(getattr(subset, collection_name))
            if total > 0:
                percentage = subset_size / total * 100
                logging.info(f"  {collection_name}: {subset_size}/{total} ({percentage:.1f}%)")
                collection_coverage[collection_name] = {
                    "size": subset_size,
                    "total": total,
                    "percentage": percentage
                }
        
        # 2. Validate relationship integrity
        logging.info("\nValidating subset relationship integrity...")
        subset._validate_relationships()
        
        # 3. Generate and compare statistics
        full_stats = self.generate_statistics()
        subset_stats = subset.generate_statistics()
        
        logging.info("\nDistribution Comparisons:")
        for dist_name, full_dist in full_stats["distributions"].items():
            subset_dist = subset_stats["distributions"][dist_name]
            logging.info(f"\n{dist_name.replace('_', ' ').title()}:")
            logging.info("  Full dataset:  "
                        f"min={full_dist['min']}, max={full_dist['max']}, "
                        f"avg={full_dist['avg']:.1f}, median={full_dist['median']}")
            logging.info("  Subset:        "
                        f"min={subset_dist['min']}, max={subset_dist['max']}, "
                        f"avg={subset_dist['avg']:.1f}, median={subset_dist['median']}")
        
        # 4. Analyze network characteristics
        logging.info("\nNetwork Analysis:")
        
        # Citation network
        full_citation = full_stats["relationship_patterns"]["citation_network"]
        subset_citation = subset_stats["relationship_patterns"]["citation_network"]
        
        logging.info("\nCitation Network Comparison:")
        logging.info(f"  Citations per article (full): {full_citation['avg_citations_per_article']:.1f}")
        logging.info(f"  Citations per article (subset): {subset_citation['avg_citations_per_article']:.1f}")
        
        # Section distribution
        logging.info("\nSection Distribution:")
        section_coverage = {}
        for section in set(full_stats["section_citations"].keys()) | set(subset_stats["section_citations"].keys()):
            full_count = full_stats["section_citations"].get(section, 0)
            subset_count = subset_stats["section_citations"].get(section, 0)
            if full_count > 0:
                percentage = (subset_count / full_count) * 100
                logging.info(f"  {section}: {subset_count}/{full_count} ({percentage:.1f}%)")
                section_coverage[section] = {
                    "subset_count": subset_count,
                    "full_count": full_count,
                    "percentage": percentage
                }
        
        # 5. Verify subset consistency
        logging.info("\nVerifying subset consistency...")
        
        # Check for external references
        external_refs = defaultdict(int)
        for article in subset.articles.values():
            # Check author references
            for author_uuid in article.authors:
                if not any(a_uuid == author_uuid for a_uuid in self.uuid_maps["authors"].values()):
                    external_refs["article -> author"] += 1
            
            # Check reference references
            for ref_uuid in article.references:
                if not any(r_uuid == ref_uuid for r_uuid in self.uuid_maps["references"].values()):
                    external_refs["article -> reference"] += 1
        
        if external_refs:
            logging.warning("\nFound external references in subset:")
            for ref_type, count in external_refs.items():
                logging.warning(f"  {ref_type}: {count}")
        else:
            logging.info("  No external references found - subset is self-contained")
        
        return {
            "collection_coverage": collection_coverage,
            "section_coverage": section_coverage,
            "statistics_comparison": {
                "full": full_stats,
                "subset": subset_stats
            },
            "consistency_check": {
                "external_references": dict(external_refs),
                "is_self_contained": len(external_refs) == 0
            }
        }

    def prepare_for_import(self, use_subset: bool = False, subset_size: Optional[int] = None) -> Tuple[bool, List[str]]:
        """
        Prepare the data for import to Weaviate database.
        Validates data integrity and optionally creates a subset.
        
        Args:
            use_subset: Whether to use a subset of the data
            subset_size: Size of subset if use_subset is True
            
        Returns:
            Tuple containing:
            - bool: True if data is ready for import
            - List[str]: Any warnings or validation messages
        """
        from ..cli import log_progress
        
        # If using subset, create it first
        if use_subset:
            if not subset_size:
                subset_size = min(100, len(self.articles))  # Default to 100 or all articles if less
            log_progress(f"Creating subset of {subset_size} articles...")
            self = self.extract_subset(subset_size)
        
        # Verify data is ready for import
        is_ready, warnings = self.verify_for_import()
        
        if not is_ready:
            logging.error("Data validation failed. See warnings below:")
            for warning in warnings:
                logging.error(f"  - {warning}")
            return False, warnings
        
        return True, warnings

    def import_to_database(self, client: Optional['weaviate.Client'] = None) -> bool:
        """
        Import the data to Weaviate database.
        Handles the transition to the importer with proper error handling.
        
        Args:
            client: Optional Weaviate client. If not provided, will create a new one.
            
        Returns:
            bool: True if import was successful
        """
        from ..database.importer import WeaviateImporter
        from ..database.schema import SchemaGenerator
        from ..cli import log_progress
        
        # Track if we created the client or were given one
        client_ctx = None
        
        try:
            # Use provided client or create a new one
            if client is None:
                from ..database.client import get_client
                client_ctx = get_client()
                client = client_ctx.__enter__()
            
            # Create importer and import data
            with WeaviateImporter(self, suppress_initial_logging=True) as importer:
                success = importer.import_data()
                if success:
                    log_progress("Data import completed successfully")
                else:
                    logging.error("Data import failed")
                return success
            
        except Exception as e:
            logging.error(f"Error during import: {str(e)}")
            import traceback
            logging.debug(traceback.format_exc())
            return False
        
        finally:
            # Clean up client if we created it
            if client_ctx is not None:
                try:
                    client_ctx.__exit__(None, None, None)
                except Exception as e:
                    logging.error(f"Error closing client: {str(e)}")