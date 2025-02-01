"""
Schema generator for Weaviate collections.

This module handles the creation and configuration of Weaviate collections and their properties
for scientific literature data. It manages:
- Primary collections (Article, Author, Reference, etc.)
- Cross-references between collections
- Vectorization and generative AI configurations
- Schema validation and inspection
"""

import logging
from typing import Optional, Dict, Any, List, Tuple

import weaviate
from weaviate.classes.config import Property, DataType, Configure, Tokenization, ReferenceProperty
from weaviate.config import AdditionalConfig, Timeout

from ..config.settings import (
    VECTORIZER_MODEL,
    VECTOR_DIMENSIONS,
    MANAGED_COLLECTIONS,
    DEFAULT_MAX_TOKENS,
    OPENAI_BASE_URL,
    GENERATIVE_MODEL,
    GENERATIVE_MAX_TOKENS,
    GENERATIVE_TEMPERATURE,
    GENERATIVE_TOP_P,
    GENERATIVE_FREQUENCY_PENALTY,
    GENERATIVE_PRESENCE_PENALTY
)
from .inspector import DatabaseInspector

class SchemaGenerator:
    """Generator for creating and managing Weaviate schema for scientific literature data.
    
    This class is responsible for:
    - Creating primary collections with their properties (Article, Author, Reference, etc.)
    - Configuring vectorization and generative AI settings
    - Establishing cross-references between collections
    - Validating schema consistency
    
    The schema follows a structured approach where:
    1. Primary collections are created first with their basic properties
    2. Cross-references are added after all collections exist
    3. Schema validation ensures all required relationships are properly established
    """
    
    def __init__(self, client: weaviate.Client):
        """Initialize schema generator.
        
        Args:
            client: Connected Weaviate client instance used for schema operations
        """
        self.client = client
        self.inspector = DatabaseInspector(client)
        
    def create_base_configs(self) -> Tuple[Configure.Vectorizer, Configure.Generative]:
        """Create vectorizer and generative AI configurations for collections.
        
        Creates configurations for:
        1. Text vectorization using OpenAI's embedding model
        2. Generative AI capabilities using OpenAI's language model
        
        The configurations use parameters from settings including model names,
        token limits, and other model-specific parameters.
        
        Returns:
            tuple: (vectorizer_config, generative_config)
                - vectorizer_config: OpenAI text vectorization configuration
                - generative_config: OpenAI generative AI configuration
        """
        vectorizer_config = Configure.Vectorizer.text2vec_openai(
            model=VECTORIZER_MODEL,
            dimensions=VECTOR_DIMENSIONS,
            base_url=OPENAI_BASE_URL
        )
        
        generative_config = Configure.Generative.openai(
            model=GENERATIVE_MODEL,
            max_tokens=GENERATIVE_MAX_TOKENS,
            base_url=OPENAI_BASE_URL,
            temperature=GENERATIVE_TEMPERATURE,
            top_p=GENERATIVE_TOP_P,
            frequency_penalty=GENERATIVE_FREQUENCY_PENALTY,
            presence_penalty=GENERATIVE_PRESENCE_PENALTY
        )
        
        return vectorizer_config, generative_config
        
    def create_schema(self) -> bool:
        """Create complete schema for all scientific literature collections.
        
        This method orchestrates the complete schema creation process:
        1. Creates base configurations for vectorization and generative AI
        2. Creates primary collections with their properties
        3. Adds cross-references between collections
        4. Validates the final schema
        
        The process is designed to be atomic - if any step fails, the entire
        operation is considered failed and an error is logged.
        
        Returns:
            bool: True if schema was created and validated successfully,
                 False if any step failed
        
        Raises:
            Exception: If any critical error occurs during schema creation
        """
        try:
            logging.debug("Starting schema creation...")
            
            vectorizer_config, generative_config = self.create_base_configs()
            
            # Create primary collections first (without references)
            self._create_primary_collections(vectorizer_config, generative_config)
            
            # Now add all cross-references
            self._add_cross_references()
            
            # Validate the created schema
            if not self.validate_schema():
                logging.error("Schema validation failed")
                return False
            
            logging.info("Schema creation completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error creating schema: {str(e)}")
            raise
            
    def validate_schema(self) -> bool:
        """Validate that created schema matches specifications.
        
        Performs comprehensive validation including:
        - Existence of all required collections
        - Presence of required properties in each collection
        - Correct configuration of cross-references
        - Proper vectorizer and generative AI settings
        
        Returns:
            bool: True if schema is valid and matches specifications,
                 False if any validation check fails
        """
        return self.inspector.validate_schema_consistency()
    
    def get_schema_state(self) -> Dict[str, Any]:
        """Get current state of the schema.
        
        Retrieves detailed information about:
        - Existing collections and their properties
        - Configured cross-references
        - Vectorizer and generative AI settings
        
        Returns:
            Dict[str, Any]: Dictionary containing current schema configuration:
                - collections: List of collection configurations
                - properties: Property definitions for each collection
                - references: Cross-reference configurations
                - settings: Vectorizer and generative AI settings
        """
        return self.inspector.get_schema_info()

    def _create_primary_collections(self, vectorizer_config: Configure.Vectorizer, generative_config: Configure.Generative) -> None:
        """Create all primary collections with their properties.
        
        Creates the following collections:
        - Article: Scientific articles with full text and metadata
        - Author: Article authors with canonical names and contact info
        - Reference: Citations and bibliographic information
        - NamedEntity: Named entities extracted from articles
        - CitationContext: Context of citations within articles
        - NameVariant: Alternative forms of author names
        - NERArticleScore: Named entity recognition confidence scores
        
        Each collection is created with appropriate properties and configurations
        for vectorization and generative AI capabilities.
        
        Args:
            vectorizer_config: OpenAI text vectorization configuration
            generative_config: OpenAI generative AI configuration
            
        Raises:
            Exception: If collection creation fails
        """
        # Article collection
        article_properties = [
            Property(name="filename", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Source filename"),
            Property(name="affiliations", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Author affiliations"),
            Property(name="funding_info", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Funding information"),
            Property(name="abstract", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article abstract"),
            Property(name="introduction", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article introduction"),
            Property(name="methods", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article methods"),
            Property(name="results", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article results"),
            Property(name="discussion", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article discussion"),
            Property(name="figures", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article figures"),
            Property(name="tables", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article tables"),
            Property(name="publication_info", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Publication information"),
            Property(name="acknowledgements", data_type=DataType.TEXT,
                    vectorize=True,
                    tokenization=Tokenization.LOWERCASE,
                    description="Article acknowledgements")
        ]
        
        self.client.collections.create(
            name="Article",
            description="Scientific article with metadata and content",
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            properties=article_properties
        )
        
        # Author collection
        author_properties = [
            Property(
                name="canonical_name",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Author's canonical name"
            ),
            Property(
                name="email",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Author's email"
            )
        ]
        
        self.client.collections.create(
            name="Author",
            description="Author of scientific articles",
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            properties=author_properties
        )
        
        # Reference collection
        reference_properties = [
            Property(
                name="title",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Reference title"
            ),
            Property(
                name="journal",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Journal name"
            ),
            Property(
                name="volume",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Journal volume"
            ),
            Property(
                name="pages",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Page numbers"
            ),
            Property(
                name="publication_date",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Publication date"
            ),
            Property(
                name="raw_reference",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Original reference string"
            )
        ]
        
        self.client.collections.create(
            name="Reference",
            description="Reference cited in articles",
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            properties=reference_properties
        )
        
        # NamedEntity collection
        named_entity_properties = [
            Property(
                name="name",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Entity name"
            ),
            Property(
                name="type",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Entity type"
            )
        ]
        
        self.client.collections.create(
            name="NamedEntity",
            description="Named entity extracted from articles",
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            properties=named_entity_properties
        )

        # CitationContext collection
        citation_context_properties = [
            Property(
                name="section",
                data_type=DataType.TEXT,
                description="Article section containing citation",
                tokenization=Tokenization.LOWERCASE
            ),
            Property(
                name="local_ref_id",
                data_type=DataType.TEXT,
                description="Local reference identifier",
                tokenization=Tokenization.FIELD
            )
        ]
        
        self.client.collections.create(
            name="CitationContext",
            description="Context of citation within article",
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            properties=citation_context_properties
        )
        
        # NameVariant collection
        name_variant_properties = [
            Property(
                name="name",
                data_type=DataType.TEXT,
                vectorize=True,
                tokenization=Tokenization.LOWERCASE,
                description="Alternative form of author name"
            )
        ]
        
        self.client.collections.create(
            name="NameVariant",
            description="Alternative forms of author names",
            vectorizer_config=vectorizer_config,
            generative_config=generative_config,
            properties=name_variant_properties
        )
        
        # NERArticleScore collection - no generative config needed as it only has a numeric score
        ner_score_properties = [
            Property(
                name="score",
                data_type=DataType.NUMBER,
                description="NER confidence score"
            )
        ]
        
        self.client.collections.create(
            name="NERArticleScore",
            description="Named entity recognition scores for articles",
            vectorizer_config=vectorizer_config,  # Keep vectorizer for potential future use
            properties=ner_score_properties
        )

    def _add_cross_references(self):
        """Add cross-references between collections after all collections exist.
        
        Establishes the following key relationships:
        - Article <-> Author (authors of articles)
        - Article <-> Reference (citations in articles)
        - Article <-> NamedEntity (entities found in articles)
        - Article <-> CitationContext (citation contexts)
        - Article <-> NERArticleScore (entity recognition scores)
        - Author <-> NameVariant (alternative name forms)
        - Reference <-> CitationContext (citation contexts)
        - NamedEntity <-> NERArticleScore (entity recognition scores)
        
        Cross-references are bidirectional where appropriate to allow
        traversal in both directions.
        
        Raises:
            Exception: If adding any cross-reference fails
        """
        try:
            logging.info("Adding cross-references between collections...")
            
            # Get all collections
            article_collection = self.client.collections.get("Article")
            author_collection = self.client.collections.get("Author")
            reference_collection = self.client.collections.get("Reference")
            named_entity_collection = self.client.collections.get("NamedEntity")
            citation_context_collection = self.client.collections.get("CitationContext")
            ner_score_collection = self.client.collections.get("NERArticleScore")
            name_variant_collection = self.client.collections.get("NameVariant")
            
            # Add Article references
            article_collection.config.add_reference(
                ReferenceProperty(name="authors", target_collection="Author")
            )
            article_collection.config.add_reference(
                ReferenceProperty(name="references", target_collection="Reference")
            )
            article_collection.config.add_reference(
                ReferenceProperty(name="named_entities", target_collection="NamedEntity")
            )
            article_collection.config.add_reference(
                ReferenceProperty(name="citation_contexts", target_collection="CitationContext")
            )
            article_collection.config.add_reference(
                ReferenceProperty(name="ner_scores", target_collection="NERArticleScore")
            )
            
            # Add Author references
            author_collection.config.add_reference(
                ReferenceProperty(name="primary_articles", target_collection="Article")
            )
            author_collection.config.add_reference(
                ReferenceProperty(name="authored_references", target_collection="Reference")
            )
            author_collection.config.add_reference(
                ReferenceProperty(name="name_variants", target_collection="NameVariant")
            )
            
            # Add NameVariant references
            name_variant_collection.config.add_reference(
                ReferenceProperty(name="author", target_collection="Author")
            )
            
            # Add Reference references
            reference_collection.config.add_reference(
                ReferenceProperty(name="authors", target_collection="Author")
            )
            reference_collection.config.add_reference(
                ReferenceProperty(name="cited_in", target_collection="Article")
            )
            reference_collection.config.add_reference(
                ReferenceProperty(name="citation_contexts", target_collection="CitationContext")
            )
            
            # Add NamedEntity references
            named_entity_collection.config.add_reference(
                ReferenceProperty(name="found_in", target_collection="Article")
            )
            named_entity_collection.config.add_reference(
                ReferenceProperty(name="article_scores", target_collection="NERArticleScore")
            )
            
            # Add CitationContext references
            citation_context_collection.config.add_reference(
                ReferenceProperty(name="article", target_collection="Article")
            )
            citation_context_collection.config.add_reference(
                ReferenceProperty(name="reference", target_collection="Reference")
            )
            
            # Add NERArticleScore references
            ner_score_collection.config.add_reference(
                ReferenceProperty(name="article", target_collection="Article")
            )
            ner_score_collection.config.add_reference(
                ReferenceProperty(name="entity", target_collection="NamedEntity")
            )
            
            logging.info("All cross-references added successfully")
            
        except Exception as e:
            logging.error(f"Error adding cross-references: {str(e)}")
            raise 