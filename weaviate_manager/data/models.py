"""
Data models for the scientific literature database.

This module defines the core entities and their relationships for managing
scientific literature data in Weaviate. Each entity is implemented as a
dataclass with:

Core Features:
- Automatic UUID generation for Weaviate integration
- Cross-reference tracking between entities
- Default value handling for optional fields
- Type hints for all attributes

Entity Types:
- Article: Scientific articles with full text and metadata
- Author: Researchers with canonical names and affiliations
- Reference: Citations and bibliographic information
- NamedEntity: Ontological terms and concepts
- CitationContext: Citation locations and contexts
- NERArticleScore: Entity recognition confidence scores
- NameVariant: Alternative forms of author names

The models are designed to maintain referential integrity and support
efficient data loading and validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from weaviate.util import generate_uuid5
import uuid

@dataclass
class Article:
    """
    Scientific article with full text content and metadata.
    
    Core Attributes:
        filename: Unique source file identifier
        affiliations: Author institutional affiliations
        funding_info: Grant and funding information
        abstract: Article abstract
        introduction: Introduction section
        methods: Methods section
        results: Results section
        discussion: Discussion section
        figures: Figure captions and descriptions
        tables: Table contents and captions
        publication_info: Journal and publication details
        acknowledgements: Article acknowledgements
    
    Cross-references (Sets of UUIDs):
        authors: Set[str] ↔ Author.articles
        references: Set[str] ↔ Reference.citing_articles
        named_entities: Set[str] ↔ NamedEntity.articles
        citation_contexts: Set[str] ↔ CitationContext objects
        ner_scores: Set[str] ↔ NERArticleScore objects
    """
    filename: str
    affiliations: str = ""
    funding_info: str = ""
    abstract: str = ""
    introduction: str = ""
    methods: str = ""
    results: str = ""
    discussion: str = ""
    figures: str = ""
    tables: str = ""
    publication_info: str = ""
    acknowledgements: str = ""
    
    # Cross-references (Sets of UUIDs)
    authors: Set[str] = field(default_factory=set)
    references: Set[str] = field(default_factory=set)
    named_entities: Set[str] = field(default_factory=set)
    citation_contexts: Set[str] = field(default_factory=set)
    ner_scores: Set[str] = field(default_factory=set)
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Initialize cross-reference sets if not provided."""
        if not self.authors:
            self.authors = set()
        if not self.references:
            self.references = set()
        if not self.named_entities:
            self.named_entities = set()
        if not self.citation_contexts:
            self.citation_contexts = set()
        if not self.ner_scores:
            self.ner_scores = set()

@dataclass
class Author:
    """
    Author of scientific articles.
    
    Core Attributes:
        canonical_name: Author's standardized name
        email: Author's email address
    
    Cross-references (Sets of UUIDs):
        articles: Set[str] ↔ Article.authors
        authored_references: Set[str] ↔ Reference.authors
        name_variants: Set[str] ↔ NameVariant objects
    """
    canonical_name: str
    email: str = ""
    
    # Cross-references (Sets of UUIDs)
    articles: Set[str] = field(default_factory=set)
    authored_references: Set[str] = field(default_factory=set)
    name_variants: Set[str] = field(default_factory=set)
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Initialize cross-reference sets if not provided."""
        if not self.articles:
            self.articles = set()
        if not self.authored_references:
            self.authored_references = set()
        if not self.name_variants:
            self.name_variants = set()

@dataclass
class Reference:
    """
    Citation reference to another work.
    
    Core Attributes:
        title: Reference title
        journal: Publication journal
        volume: Journal volume
        pages: Page numbers
        publication_date: Date of publication
        raw_reference: Original citation text
    
    Cross-references (Sets of UUIDs):
        authors: Set[str] ↔ Author.references
        citing_articles: Set[str] ↔ Article.references
        citation_contexts: Set[str] ↔ CitationContext objects
    """
    title: str = ""
    journal: str = ""
    volume: str = ""
    pages: str = ""
    publication_date: str = ""
    raw_reference: str = ""
    
    # Cross-references (Sets of UUIDs)
    authors: Set[str] = field(default_factory=set)
    citing_articles: Set[str] = field(default_factory=set)
    citation_contexts: Set[str] = field(default_factory=set)
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Initialize cross-reference sets if not provided."""
        if not self.authors:
            self.authors = set()
        if not self.citing_articles:
            self.citing_articles = set()
        if not self.citation_contexts:
            self.citation_contexts = set()

@dataclass
class NamedEntity:
    """
    Named entity from ontological terms and types.
    
    Core Attributes:
        name: Entity name
        type: Entity type (e.g., gene, protein, organism)
    
    Cross-references (Sets of UUIDs):
        articles: Set[str] ↔ Article.named_entities
        ner_scores: Set[str] ↔ NERArticleScore objects
    """
    name: str
    type: str = ""
    
    # Cross-references (Sets of UUIDs)
    articles: Set[str] = field(default_factory=set)
    ner_scores: Set[str] = field(default_factory=set)
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Initialize cross-reference sets if not provided."""
        if not self.articles:
            self.articles = set()
        if not self.ner_scores:
            self.ner_scores = set()

@dataclass
class NERArticleScore:
    """
    Named entity recognition confidence score.
    Links a NamedEntity to an Article with a confidence score.
    
    Core Attributes:
        score: Confidence score value
    
    Direct References:
        article_uuid: str -> Article that contains the entity
        entity_uuid: str -> NamedEntity that was found
    """
    score: float
    
    # Direct References
    article_uuid: str
    entity_uuid: str
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Generate UUID if not provided."""
        if not self.uuid:
            self.uuid = str(uuid.uuid4())

@dataclass
class CitationContext:
    """
    Context of a citation within an article.
    Links an Article to a Reference with section context.
    
    Core Attributes:
        section: Section where citation appears
        local_ref_id: Local reference identifier
    
    Direct References:
        article_uuid: str -> Citing Article
        reference_uuid: str -> Cited Reference
    """
    section: str
    local_ref_id: str
    
    # Direct References
    article_uuid: str
    reference_uuid: str
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Generate UUID if not provided."""
        if not self.uuid:
            self.uuid = str(uuid.uuid4())

@dataclass
class NameVariant:
    """
    Alternative form of an author's name.
    Links to an Author with the variant name form.
    
    Core Attributes:
        name: Alternative name form
    
    Direct References:
        author_uuid: str -> Author this is a variant of
    """
    name: str
    
    # Direct References
    author_uuid: str
    
    # UUID for Weaviate
    uuid: Optional[str] = None
    
    def __post_init__(self):
        """Generate UUID if not provided."""
        if not self.uuid:
            self.uuid = str(uuid.uuid4()) 