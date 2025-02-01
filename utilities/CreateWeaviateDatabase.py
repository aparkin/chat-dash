"""
Weaviate Scientific Literature Database Manager
Version: 1.0.0

Part of a pipeline for processing scientific literature into a searchable database:

Complete Processing Pipeline:
--------------------------
1. ProcessPDFsWithGrobid.py
   - Converts PDFs to structured XML using GROBID
   - Extracts metadata, text content, and citations
   - Output: GROBID TEI XML files

2. ParseTEI.py
   - Parses GROBID TEI XML output
   - Extracts structured data (authors, references, sections)
   - Output: Initial JSON files per article

3. CleanAndConsolidateParsedTEI.py
   - Consolidates and normalizes parsed article data
   - Deduplicates authors and references
   - Links citations and named entities
   - Output: Unified JSON files in processed_output/
     * processed_articles.json
     * unified_authors.json
     * unified_references.json
     * unified_ner_objects.json

4. CreateWeaviateDatabase.py (this script)
   - Imports processed data into Weaviate
   - Creates vector embeddings
   - Establishes cross-references
   - Enables semantic search and analysis

Features:
--------
- Batch importing with error handling and progress tracking
- Text chunking for large content sections
- Cross-referencing between articles, authors, and citations
- Named entity tracking and relationships

Command Line Options:
-------------------
--info            Show database information and collection statistics
--cleanup         Remove existing collections (requires confirmation)
--force           Skip confirmation prompts during cleanup
--show-models     Display available OpenAI models for vectorization
--data-dir PATH   Directory containing processed article JSON files
--subset-size N   Number of articles for test subset (maintains relationships)
--show-import     Show summary of data to be imported without importing
--analyze         Analyze articles for size and chunking requirements

Input Requirements:
-----------------
JSON files in data directory (from CleanAndConsolidateParsedTEI.py):
- processed_articles.json: Articles with full text split into sections
- unified_authors.json: Author information with name variants
- unified_references.json: Citations with contexts and relationships
- unified_ner_objects.json: Named entities with article occurrences

Common Usage:
-----------
Full Pipeline:
    1. python ProcessPDFsWithGrobid.py --input papers/ --output grobid_output/
    2. python ParseTEI.py --input grobid_output/ --output parsed_output/
    3. python CleanAndConsolidateParsedTEI.py --input parsed_output/ --output processed_output/
    4. python CreateWeaviateDatabase.py --data-dir processed_output/

Test Import:
    python CreateWeaviateDatabase.py --data-dir processed_output/ --subset-size 20 --cleanup

Analyze Before Import:
    python CreateWeaviateDatabase.py --data-dir processed_output/ --analyze --show-import

Show Database Status:
    python CreateWeaviateDatabase.py --info
"""

# Standard Library Imports
import json
import logging
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Generator

# Third-Party Imports
import requests
import weaviate
from weaviate.classes.config import Property, DataType, Configure, Tokenization, ReferenceProperty
from weaviate.config import AdditionalConfig, Timeout
from tqdm import tqdm
import tiktoken
from dotenv import load_dotenv
from pathlib import Path

# Find the project root directory (where .env is located)
project_root = Path(__file__).parent.parent
dotenv_path = project_root / '.env'

# Try to load from .env file
load_dotenv(dotenv_path=dotenv_path)


#-----------------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------------

# OpenAI Settings
if True:  # Toggle for development environment
    OPENAI_BASE_URL = os.getenv('CBORG_BASE_URL', "https://api.cborg.lbl.gov")
    OPENAI_API_KEY = os.getenv('CBORG_API_KEY', '')  # Must be set in environment
else:  # Production environment
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Must be set in environment

# AI Model Configuration
VECTORIZER_MODEL = "text-embedding-3-large"
VECTOR_DIMENSIONS = 1024 # Can be [256 1024 3072]
GENERATIVE_MODEL = "gpt-4"
MAX_TOKENS = 4096

# Generative Configuration
GENERATIVE_TEMPERATURE = 0.7
GENERATIVE_TOP_P = 0.95
GENERATIVE_FREQUENCY_PENALTY = 0.5
GENERATIVE_PRESENCE_PENALTY = 0.5

# Collection Configuration
MANAGED_COLLECTIONS = [
    "Author",
    "Reference",
    "NamedEntity",
    "Article",
    "EntityOccurrence"
]

# Database Configuration
MAX_STRING_LENGTH = 65000
BATCH_SIZES = {
    "Article": 10,    # Articles have large text content
    "Reference": 50,  
    "Author": 100,
    "NamedEntity": 100,
    "default": 50
}

ARTICLE_SECTIONS = [
    "abstract", "introduction", "methods",
    "results", "discussion", "figures", "tables"
]

# Add at top of file with other constants
DEFAULT_MAX_TOKENS = 8191  # OpenAI's default context length
WORDS_PER_TOKEN = 0.75  # Conservative estimate for words-to-tokens ratio

#-----------------------------------------------------------------------------
# Data Models
#-----------------------------------------------------------------------------

@dataclass
class ReferenceOccurrence:
    """
    Represents a citation's occurrence within an article.
    
    Tracks where and how references appear in articles, including:
    - Location (sections)
    - Local identification
    - Context information
    
    This granular tracking enables:
    - Citation network analysis
    - Reference usage patterns
    - Citation context studies
    
    Attributes:
        article_id: ID of the citing article
        local_ref_id: Reference's local identifier
        sections: Article sections containing the reference
    """
    article_id: str
    local_ref_id: str
    sections: List[str]

#-----------------------------------------------------------------------------
# Database Operations
#-----------------------------------------------------------------------------

def create_client():
    """
    Initialize and configure Weaviate client with proper settings.
    
    Configures:
    - Custom Weaviate endpoint
    - OpenAI API integration
    - Connection timeouts
    - Security headers
    
    Returns:
        weaviate.Client: Configured client instance
        
    Raises:
        SystemExit: If client creation fails
    """
    try:
        client = weaviate.connect_to_custom(
            http_host="weaviate.kbase.us",
            http_port=443,
            http_secure=True,
            grpc_host="weaviate-grpc.kbase.us",
            grpc_port=443,
            grpc_secure=True,
            headers={
                "X-OpenAI-Api-Key": os.getenv('OPENAI_API_KEY', OPENAI_API_KEY),
            },
            additional_config=AdditionalConfig(
                timeout=Timeout(
                    timeout_config=120,
                    timeout_vectorizer=120
                )
            ),
            skip_init_checks=True  # Add this to skip OpenID config check
        )
        return client
    except Exception as e:
        logging.error(f"Failed to create client: {str(e)}")
        sys.exit(1)

def get_existing_collections(client) -> List[str]:
    """
    Get list of existing collections in the database.
    
    Args:
        client: Weaviate client instance
        
    Returns:
        List[str]: Names of existing collections
        
    Note:
        Returns empty list on error but logs the failure
    """
    try:
        collections = client.collections.list_all(simple=True)
        return list(collections.keys())
    except Exception as e:
        logging.error(f"Error getting collections: {str(e)}")
        return []

def get_database_summary(client) -> Dict[str, Any]:
    """
    Generate comprehensive database status report.
    
    Collects:
    - Collection statistics
    - Property definitions
    - Sample objects
    - Total object counts
    
    Args:
        client: Weaviate client instance
        
    Returns:
        Dict containing:
            collections: Dict of collection info
            total_objects: Total object count
            
    Raises:
        Exception: If summary generation fails
    """
    summary = {
        "collections": {},
        "total_objects": 0
    }
    
    try:
        collections = client.collections.list_all(simple=False)
        
        for collection_name, collection_config in collections.items():
            logging.info(f"Analyzing collection: {collection_name}")
            
            collection = client.collections.get(collection_name)
            count_response = collection.aggregate.over_all(total_count=True)
            object_count = count_response.total_count
            
            samples = []
            for i, item in enumerate(collection.iterator()):
                if i >= 3:  # Limit to 3 samples
                    break
                samples.append({
                    "uuid": item.uuid,
                    "properties": item.properties
                })
            
            summary["collections"][collection_name] = {
                "object_count": object_count,
                "properties": [
                    {"name": prop.name, "data_type": prop.data_type.value}
                    for prop in collection_config.properties
                ],
                "sample_objects": samples
            }
            
            summary["total_objects"] += object_count
            
        return summary
        
    except Exception as e:
        logging.error(f"Error generating database summary: {str(e)}")
        raise

def print_database_summary(summary: Dict[str, Any]):
    """Display formatted database summary."""
    print("\nWeaviate Database Summary")
    print("========================")
    print(f"Total Objects: {summary['total_objects']}")
    print("\nCollections:")
    
    for collection_name, stats in summary["collections"].items():
        print(f"\n{collection_name}:")
        print(f"  Objects: {stats['object_count']}")
        print(f"  Properties: {len(stats['properties'])}")
        print("  Properties:")
        for prop in stats["properties"]:
            print(f"    - {prop['name']}: {prop['data_type']}")

def cleanup_collections(client, collections_to_remove: Optional[List[str]] = None, 
                       force: bool = False) -> bool:
    """
    Remove specified or all collections from database.
    
    Args:
        client: Weaviate client instance
        collections_to_remove: Optional list of collections to remove
                             If None, removes all collections
        force: Skip confirmation prompt if True
        
    Returns:
        bool: True if cleanup successful, False otherwise
        
    Note:
        Prompts for confirmation unless force=True
    """
    try:
        collections = client.collections.list_all(simple=True)
        existing_collections = list(collections.keys())
        logging.info(f"Found existing collections: {existing_collections}")
        
        if collections_to_remove is None:
            collections_to_remove = existing_collections
            
        if not collections_to_remove:
            logging.info("No collections to remove")
            return True
            
        # Confirm unless forced
        if not force:
            print("\nCollections to be removed:")
            for col in collections_to_remove:
                print(f"  - {col}")
            
            response = input("\nProceed with removal? [y/N]: ")
            if response.lower() != 'y':
                logging.info("Cleanup cancelled")
                return False
            
        # Delete collections one by one
        for collection_name in collections_to_remove:
            try:
                client.collections.delete(collection_name)
                logging.info(f"Deleted collection: {collection_name}")
            except Exception as e:
                logging.error(f"Failed to delete collection {collection_name}: {str(e)}")
        
        # Verify cleanup
        remaining = client.collections.list_all(simple=True)
        if remaining:
            logging.warning(f"Collections remaining after cleanup: {list(remaining.keys())}")
            
        return True
                    
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        return False

def check_available_models(base_url: str = OPENAI_BASE_URL, api_key: str = OPENAI_API_KEY):
    """Verify available OpenAI models."""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"\nChecking available models at {base_url}...")
        response = requests.get(
            f"{base_url}/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()
            print("\nAvailable models:")
            for model in models.get('data', []):
                print(f"  - {model['id']}")
                if model['id'] == 'lbl/nomic-embed-text':
                    print(f"\nNomic model details:")
                    print(json.dumps(model, indent=2))
        else:
            print(f"\nError: Failed to get models (Status {response.status_code})")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\nError checking models: {str(e)}")

@contextmanager
def weaviate_client():
    """
    Context manager for Weaviate client lifecycle.
    
    Handles:
    - Client creation
    - Resource cleanup
    - Error handling
    
    Usage:
        with weaviate_client() as client:
            # Use client here
    """
    client = None
    try:
        client = create_client()
        yield client
    finally:
        if client:
            try:
                client.close()
            except Exception as e:
                logging.warning(f"Error closing client: {str(e)}")

#-----------------------------------------------------------------------------
# Data Management
#-----------------------------------------------------------------------------

class LiteratureDataManager:
    """
    Manages scientific literature data loading, validation, and subsetting.
    
    This class handles:
    1. Loading and validating JSON data files
    2. Building relationship maps between entities
    3. Extracting connected subsets of data
    4. Visualizing data relationships
    
    Attributes:
        data_dir: Path to directory containing input JSON files
        articles: Dictionary of article records
        authors: Dictionary of author records
        references: Dictionary of reference records
        ner_objects: Dictionary of named entity records
        author_articles: Mapping of authors to their articles
        author_references: Mapping of authors to their references
        ner_article_scores: Mapping of NER terms to article scores
        reference_occurrences: Mapping of references to their occurrences
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        # Raw data storage
        self.articles: Dict[str, Dict] = {}
        self.authors: Dict[str, Dict] = {}
        self.references: Dict[str, Dict] = {}
        self.ner_objects: Dict[str, Dict] = {}
        
        # Relationship tracking
        self.author_articles = defaultdict(set)
        self.author_references = defaultdict(set)
        self.ner_article_scores = defaultdict(dict)
        self.reference_occurrences = defaultdict(list)
        
    def load_data(self) -> None:
        """Load all data files and build relationship maps."""
        data_files = {
            'articles': 'processed_articles.json',
            'authors': 'unified_authors.json',
            'references': 'unified_references.json',
            'ner_objects': 'unified_ner_objects.json'
        }
        
        # Load raw data
        for key, filename in data_files.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing required file: {filename}")
            
            with open(filepath) as f:
                data = json.load(f)
                setattr(self, key, data)
                logging.info(f"Loaded {len(data)} records from {filename}")
        
        # Build relationships
        self._build_relationships()
        
    def _normalize_section_name(self, section_name: str) -> str:
        """
        Normalize section name for consistent matching.
        
        Args:
            section_name: Raw section name from reference
            
        Returns:
            Normalized section name matching article structure
        """
        # Standard section names in articles
        STANDARD_SECTIONS = {
            'abstract', 'introduction', 'methods', 
            'results', 'discussion', 'figures', 'tables'
        }
        
        # Clean and normalize the input
        cleaned = section_name.lower().strip()
        
        # Direct match to standard sections
        if cleaned in STANDARD_SECTIONS:
            return cleaned
        
        # Common variations
        variations = {
            'methodology': 'methods',
            'experimental': 'methods',
            'materials and methods': 'methods',
            'result': 'results',
            'discussion and conclusion': 'discussion',
            'conclusions': 'discussion',
            'figure': 'figures',
            'table': 'tables'
        }
        
        if cleaned in variations:
            return variations[cleaned]
        
        return cleaned  # Return cleaned version for content matching

    def _find_matching_section(self, section_name: str, article: Dict) -> Optional[str]:
        """
        Find matching section in article for a reference section.
        
        Args:
            section_name: Normalized section name from reference
            article: Article data dictionary
            
        Returns:
            Matching standard section name or None
        """
        # Standard article sections to check
        ARTICLE_SECTIONS = [
            'abstract', 'introduction', 'methods', 
            'results', 'discussion', 'figures', 'tables'
        ]
        
        # Try direct match first
        if section_name in ARTICLE_SECTIONS:
            return section_name
        
        # Check section content for match
        normalized = section_name.lower()
        for section in ARTICLE_SECTIONS:
            # Check if section exists in article
            if section not in article:
                continue
            
            # Check section content
            content = article[section]
            if isinstance(content, str):
                if normalized in content.lower():
                    return section
            elif isinstance(content, list):
                # Handle list-type sections (like figures/tables)
                for item in content:
                    if isinstance(item, str) and normalized in item.lower():
                        return section
                    elif isinstance(item, dict) and any(
                        normalized in str(v).lower() for v in item.values()
                    ):
                        return section
        
        return None

    def _build_relationships(self) -> None:
        """Build all relationship maps."""
        # Author relationships
        for author_id, author in self.authors.items():
            for article_id in author.get('primary_articles', []):
                self.author_articles[author_id].add(article_id)
        
        # Reference relationships with improved section matching
        for ref_id, ref in self.references.items():
            # Author cross-references
            for author_id in ref.get('unified_authors', []):
                self.author_references[author_id].add(ref_id)
            
            # Article occurrences with section matching
            for occ in ref.get('occurrences', []):
                article_id = occ['article_id']
                if article_id not in self.articles:
                    continue
                
                article = self.articles[article_id]
                matched_sections = []
                
                # Process each section reference
                for section in occ['sections']:
                    normalized = self._normalize_section_name(section)
                    matched = self._find_matching_section(normalized, article)
                    if matched:
                        matched_sections.append(matched)
                
                if matched_sections:  # Only create occurrence if we found matches
                    self.reference_occurrences[ref_id].append(
                        ReferenceOccurrence(
                            article_id=article_id,
                            local_ref_id=occ['local_ref_id'],
                            sections=matched_sections
                        )
                    )
        
        # NER relationships
        for ner_id, ner in self.ner_objects.items():
            for article_id, score in ner.get('article_scores', {}).items():
                self.ner_article_scores[ner_id][article_id] = float(score)
    
    def extract_connected_subset(self, size: int, seed_article: Optional[str] = None) -> Dict[str, Dict]:
        """
        Extract connected subset of data starting from seed article.
        
        Args:
            size: Target number of articles
            seed_article: Optional starting article ID
            
        Returns:
            Dict containing connected subset of all entity types
        """
        if not seed_article:
            seed_article = next(iter(self.articles))
        
        subset = {
            'articles': {},
            'authors': {},
            'references': {},
            'ner_objects': {}
        }
        
        # Start with seed article
        articles_to_process = {seed_article}
        processed_articles = set()
        
        while articles_to_process and len(subset['articles']) < size:
            article_id = articles_to_process.pop()
            if article_id not in self.articles or article_id in processed_articles:
                continue
                
            # Add article
            subset['articles'][article_id] = self.articles[article_id]
            processed_articles.add(article_id)
            
            # Add connected authors
            for author_id, articles in self.author_articles.items():
                if article_id in articles:
                    subset['authors'][author_id] = self.authors[author_id]
                    # Add other articles by this author
                    articles_to_process.update(articles - processed_articles)
            
            # Add connected references
            for ref_id, occurrences in self.reference_occurrences.items():
                for occ in occurrences:
                    if occ.article_id == article_id:
                        subset['references'][ref_id] = self.references[ref_id]
                        break
            
            # Add connected NER objects
            for ner_id, scores in self.ner_article_scores.items():
                if article_id in scores:
                    subset['ner_objects'][ner_id] = self.ner_objects[ner_id]
        
        return subset
    
    def print_subset_summary(self, subset: Dict[str, Dict]) -> None:
        """Print human-readable summary of a data subset with detailed relationships."""
        print("\nData Subset Summary")
        print("==================")
        
        # Basic counts
        print("\nEntity Counts:")
        for entity_type, data in subset.items():
            print(f"  {entity_type}: {len(data)}")
        
        # Article details with relationships
        print("\nArticle Relationships:")
        print("--------------------")
        for article_id, article in subset['articles'].items():
            print(f"\n{article_id}:")
            print(f"  Title: {article.get('title', 'N/A')}")
            
            # Authors
            print("  Authors:")
            authors = [
                (author_id, author['canonical_name'])
                for author_id, author in subset['authors'].items()
                if article_id in author.get('primary_articles', [])
            ]
            for author_id, name in authors:
                print(f"    - {name} ({author_id})")
            
            # References by section
            print("  References by Section:")
            section_refs = defaultdict(list)
            for ref_id, ref in subset['references'].items():
                for occ in ref.get('occurrences', []):
                    if occ['article_id'] == article_id:
                        for section in occ['sections']:
                            section_refs[section].append(
                                (ref_id, ref.get('title', 'Untitled'))
                            )
            
            for section, refs in section_refs.items():
                print(f"    {section}:")
                for ref_id, title in refs:
                    print(f"      - {title} ({ref_id})")
            
            # NER terms with scores
            print("  Named Entities (with scores):")
            ner_terms = [
                (ner_id, ner['name'], ner['article_scores'][article_id])
                for ner_id, ner in subset['ner_objects'].items()
                if article_id in ner.get('article_scores', {})
            ]
            # Sort by score
            ner_terms.sort(key=lambda x: x[2], reverse=True)
            for ner_id, name, score in ner_terms:
                print(f"    - {name} ({ner_id}): {score:.3f}")
        
        # Author collaboration network
        print("\nAuthor Collaboration Network:")
        print("--------------------------")
        for author_id, author in subset['authors'].items():
            print(f"\n{author['canonical_name']}:")
            # Find co-authors
            coauthors = set()
            for article_id in author.get('primary_articles', []):
                if article_id in subset['articles']:
                    for coauthor_id, coauthor in subset['authors'].items():
                        if (coauthor_id != author_id and 
                            article_id in coauthor.get('primary_articles', [])):
                            coauthors.add((coauthor_id, coauthor['canonical_name']))
            
            if coauthors:
                print("  Co-authors:")
                for coauthor_id, name in coauthors:
                    print(f"    - {name}")
            
            # Show references authored
            authored_refs = [
                ref for ref_id, ref in subset['references'].items()
                if author_id in ref.get('unified_authors', [])
            ]
            if authored_refs:
                print("  Authored References:")
                for ref in authored_refs:
                    print(f"    - {ref.get('title', 'Untitled')}")
        
        # Reference citation network
        print("\nReference Citation Network:")
        print("------------------------")
        for ref_id, ref in subset['references'].items():
            # Get reference details
            ref_title = ref.get('title') or ref.get('raw_reference', 'Unknown Reference')
            ref_authors = ref.get('authors', [])
            ref_year = ref.get('publication_date', 'Unknown Year')
            
            # Format reference header
            print(f"\nReference: {ref_id}")
            print(f"  Title: {ref_title[:100]}")
            print(f"  Authors: {', '.join(ref_authors[:3])}")  # Show first 3 authors
            if len(ref_authors) > 3:
                print("    + {} more authors".format(len(ref_authors) - 3))
            print(f"  Year: {ref_year}")
            
            # Show citations
            citations = defaultdict(list)
            for occ in ref.get('occurrences', []):
                if occ['article_id'] in subset['articles']:
                    article = subset['articles'][occ['article_id']]
                    article_title = article.get('title') or article.get('filename', 'Untitled')
                    for section in occ['sections']:
                        citations[section].append((occ['article_id'], article_title))
            
            if citations:
                print("  Cited in:")
                for section, articles in sorted(citations.items()):
                    print(f"    {section.title()}:")
                    for article_id, title in articles:
                        print(f"      - {title[:80]} ({article_id})")
            else:
                print("  No citations found in subset")

#-----------------------------------------------------------------------------
# Data Import 
#-----------------------------------------------------------------------------

def create_base_configs(base_url: str) -> Tuple[List[Configure.NamedVectors], Configure.Generative]:
    """
    Create vectorizer and generative AI configurations for Weaviate collections.
    
    Configures:
    - Text vectorization using OpenAI embeddings
    - Generative AI capabilities for semantic search
    - Model parameters for optimal performance
    
    Args:
        base_url: OpenAI API base URL for model access
        
    Returns:
        tuple: (vectorizer_config, generative_config)
            - vectorizer_config: List of named vector configurations
            - generative_config: Generative AI configuration
    """
    vectorizer_config = [
        Configure.NamedVectors.text2vec_openai(
            name="text_vector",
            source_properties=["*"],  # vectorize all text properties
            model=VECTORIZER_MODEL,
            dimensions=VECTOR_DIMENSIONS,
            base_url=base_url
        )
    ]
    
    generative_config = Configure.Generative.openai(
        model=GENERATIVE_MODEL,
        max_tokens=MAX_TOKENS,
        base_url=base_url,
        temperature=GENERATIVE_TEMPERATURE,
        top_p=GENERATIVE_TOP_P,
        frequency_penalty=GENERATIVE_FREQUENCY_PENALTY,
        presence_penalty=GENERATIVE_PRESENCE_PENALTY
    )
    
    return vectorizer_config, generative_config

def create_schema(client, base_url: str):
    """Create complete database schema with collections and references."""
    vectorizer_config, generative_config = create_base_configs(base_url)
    
    # Create Article collection with flat text properties
    article_properties = [
        # Core metadata
        Property(name="filename", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE),
        Property(name="title", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE),
        Property(name="publication_info", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE),
        
        # Main content sections (all vectorized)
        Property(name="abstract", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE,
                description="Article abstract, token-limited"),
        Property(name="introduction", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE,
                description="Introduction section, token-limited"),
        Property(name="methods", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE,
                description="Methods section, token-limited"),
        Property(name="results", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE,
                description="Results section, token-limited"),
        Property(name="discussion", data_type=DataType.TEXT,
                vectorize_property_name=True,
                tokenization=Tokenization.LOWERCASE,
                description="Discussion section, token-limited"),
                
        # Optional metadata about reduction
        Property(name="content_metadata", data_type=DataType.OBJECT,
                description="Metadata about content reduction")
    ]
    
    client.collections.create(
        name="Article",
        vectorizer_config=vectorizer_config,
        generative_config=generative_config,
        properties=article_properties
    )
    logging.info("Created Article collection")

    # 2. Create Author collection with references to Article
    client.collections.create(
        name="Author",
        vectorizer_config=vectorizer_config,
        generative_config=generative_config,
        properties=[
            Property(name="canonical_name", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="email", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="name_variants", data_type=DataType.TEXT_ARRAY,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE)
        ],
        references=[
            ReferenceProperty(name="authored_articles", target_collection="Article")  # many-to-many
        ]
    )
    logging.info("Created Author collection")

    # 3. Create Reference collection with references to both Article and Author
    client.collections.create(
        name="Reference",
        vectorizer_config=vectorizer_config,
        generative_config=generative_config,
        properties=[
            Property(name="title", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="journal", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="publication_date", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="raw_reference", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE)
        ],
        references=[
            ReferenceProperty(name="authors", target_collection="Author"),  # many-to-many
            ReferenceProperty(name="cited_in", target_collection="Article")  # many-to-many
        ]
    )
    logging.info("Created Reference collection")

    # 4. Create NamedEntity collection
    client.collections.create(
        name="NamedEntity",
        vectorizer_config=vectorizer_config,
        generative_config=generative_config,
        properties=[
            Property(name="name", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="type", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE)
        ]
    )
    logging.info("Created NamedEntity collection")

    # 5. Update Article with references (only after all collections exist)
    article_collection = client.collections.get("Article")
    
    # Add references in order of dependency
    # First, add references to basic collections
    article_collection.config.add_reference(
        ReferenceProperty(name="authors", target_collection="Author")
    )
    article_collection.config.add_reference(
        ReferenceProperty(name="references", target_collection="Reference")
    )
    
    # Then add the NamedEntity reference
    article_collection.config.add_reference(
        ReferenceProperty(name="named_entities", target_collection="NamedEntity")
    )
    
    # 6. Update NamedEntity with its Article references
    named_entity_collection = client.collections.get("NamedEntity")
    named_entity_collection.config.add_reference(
        ReferenceProperty(name="article_scores", target_collection="Article")
    )
    
    logging.info("Updated all collections with references")

    # Create CitationContext collection
    client.collections.create(
        name="CitationContext",
        vectorizer_config=vectorizer_config,
        generative_config=generative_config,
        properties=[
            Property(name="sections", data_type=DataType.TEXT_ARRAY,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="local_ref_id", data_type=DataType.TEXT),
            Property(name="context_before", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="context_after", data_type=DataType.TEXT,
                    vectorize_property_name=True,
                    tokenization=Tokenization.LOWERCASE),
            Property(name="citation_type", data_type=DataType.TEXT)
        ],
        references=[
            ReferenceProperty(name="reference", target_collection="Reference"),
            ReferenceProperty(name="article", target_collection="Article")
        ]
    )
    logging.info("Created CitationContext collection")

class DataImporter:
    """
    Manages data import with UUID tracking and cross-references.
    
    Responsibilities:
    - Tracks UUIDs for all imported objects
    - Handles batch operations for efficiency
    - Manages cross-references between collections
    - Chunks long text content
    
    Key Methods:
    - import_data: Main import orchestration
    - _chunk_text: Text content management
    
    Attributes:
        client: Weaviate client instance
        uuid_map: Dict mapping original IDs to Weaviate UUIDs
        batch_sizes: Collection-specific batch sizes
    """
    
    def __init__(self, client, batch_sizes=None, max_tokens=DEFAULT_MAX_TOKENS):
        self.client = client
        self.batch_sizes = batch_sizes or BATCH_SIZES
        self.uuid_map = defaultdict(dict)
        self.max_tokens = max_tokens
        
    def _estimate_tokens(self, text: str) -> int:
        """
        Get exact token count using tiktoken with the same encoding as our embedding model.
        """
        if not hasattr(self, '_tokenizer'):
            # Use cl100k_base for text-embedding-3-large
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        
        return len(self._tokenizer.encode(text))
        
    def _reduce_text_to_token_limit(self, text: str, max_tokens: int = None) -> str:
        """
        Intelligently reduce text to fit within token limit while preserving key content.
        Args:
            text: The text to reduce
            max_tokens: Maximum tokens allowed (defaults to self.max_tokens)
        """
        max_tokens = max_tokens or self.max_tokens
        current_tokens = self._estimate_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        # Calculate target token counts for each section
        intro_tokens = int(max_tokens * 0.4)  # 40% for intro
        middle_tokens = int(max_tokens * 0.3)  # 30% for middle
        conclusion_tokens = int(max_tokens * 0.3)  # 30% for conclusion
        
        # Process introduction
        intro_text = ""
        for para in paragraphs[:3]:
            para_tokens = self._estimate_tokens(para)
            if self._estimate_tokens(intro_text + para + "\n\n") <= intro_tokens:
                intro_text += para + "\n\n"
            else:
                # Truncate to approximate token count by words
                words = para.split()
                remaining_tokens = intro_tokens - self._estimate_tokens(intro_text)
                word_limit = int(remaining_tokens * WORDS_PER_TOKEN)
                intro_text += " ".join(words[:word_limit]) + "...\n\n"
                break
                
        # Process conclusion
        conclusion_text = ""
        for para in reversed(paragraphs[-3:]):
            if self._estimate_tokens(conclusion_text + para + "\n\n") <= conclusion_tokens:
                conclusion_text = para + "\n\n" + conclusion_text
            else:
                words = para.split()
                remaining_tokens = conclusion_tokens - self._estimate_tokens(conclusion_text)
                word_limit = int(remaining_tokens * WORDS_PER_TOKEN)
                conclusion_text = "..." + " ".join(words[-word_limit:]) + "\n\n" + conclusion_text
                break
                
        # Process middle with importance scoring
        middle_paras = paragraphs[3:-3]
        if middle_paras:
            middle_text = " ".join(middle_paras)
            sentences = [s.strip() + "." for s in middle_text.split('.') if s.strip()]
            
            # Score sentences
            scored_sentences = []
            for sentence in sentences:
                score = 0
                # Higher score for sentences with key terms
                score += sum(2 for term in [
                    'significant', 'important', 'key', 'novel', 'demonstrate',
                    'show', 'find', 'conclude', 'reveal', 'result', 'observe',
                    'measure', 'analyze', 'determine', 'identify'
                ] if term.lower() in sentence.lower())
                
                # Higher score for sentences with numbers/measurements
                score += sum(1 for char in sentence if char.isdigit())
                
                # Higher score for sentences with citations
                score += sentence.count('(') + sentence.count('[')
                
                # Higher score for sentences with comparison terms
                score += sum(1 for term in [
                    'whereas', 'however', 'although', 'unlike', 'compared',
                    'contrast', 'instead', 'rather', 'while'
                ] if term.lower() in sentence.lower())
                
                # Store score and token estimate
                scored_sentences.append((
                    score,
                    sentence,
                    self._estimate_tokens(sentence)
                ))
            
            # Select highest scoring sentences up to middle_tokens
            selected_middle = ""
            current_tokens = 0
            for score, sentence, tokens in sorted(scored_sentences, reverse=True):
                if current_tokens + tokens <= middle_tokens:
                    selected_middle += sentence + " "
                    current_tokens += tokens
                else:
                    break
        else:
            selected_middle = ""
        
        # Combine sections with clear separation
        reduced_text = f"{intro_text.strip()}\n\n"
        if selected_middle:
            reduced_text += f"Key Points: {selected_middle.strip()}\n\n"
        reduced_text += conclusion_text.strip()
        
        # Verify final token count
        final_tokens = self._estimate_tokens(reduced_text)
        logging.debug(f"Reduced text from {current_tokens} to {final_tokens} tokens")
        
        return reduced_text

    def _handle_batch_failures(self, batch, operation_type: str, collection_name: str):
        """
        Handle and report batch operation failures.
        
        Args:
            batch: Weaviate batch object
            operation_type: Type of operation (e.g., "object creation", "reference addition")
            collection_name: Name of the collection being processed
        """
        if hasattr(batch, 'failed_objects') and batch.failed_objects:
            logging.error(f"\nBatch {operation_type} failures in {collection_name}:")
            for failed in batch.failed_objects:
                logging.error(f"- Error: {failed.get('error', 'Unknown error')}")
                logging.error(f"  Object: {failed.get('object', 'Unknown object')}")
            raise RuntimeError(f"Batch {operation_type} failed for {len(batch.failed_objects)} objects")

    def _batch_objects(self, items: Dict, batch_size: int) -> Generator[Dict, None, None]:
        """Split dictionary items into batches of specified size."""
        items_list = list(items.items())
        for i in range(0, len(items_list), batch_size):
            yield dict(items_list[i:i + batch_size])

    def import_data(self, data: Dict[str, Dict]):
        """Import data with progress tracking for batches."""
        logging.info("Starting data import...")
        
        # First Pass: Create base objects
        with tqdm(total=4, desc="Creating base records") as pbar:
            # Articles with controlled batch size and progress tracking
            article_collection = self.client.collections.get("Article")
            articles_imported = 0
            total_articles = len(data["articles"])
            
            with tqdm(total=total_articles, desc="Articles", leave=False) as article_pbar:
                # Process articles in smaller batches
                for batch_data in self._batch_objects(data["articles"], self.batch_sizes["Article"]):
                    try:
                        with article_collection.batch.dynamic() as batch:
                            for article_id, article in batch_data.items():
                                processed_article = self._process_article(article)
                                if processed_article:
                                    uuid = batch.add_object(properties=processed_article)
                                    self.uuid_map["Article"][article_id] = uuid
                                    articles_imported += 1
                                    article_pbar.update(1)
                        logging.debug(f"Imported batch of {len(batch_data)} articles successfully")
                    except Exception as e:
                        logging.error(f"Failed to import article batch: {str(e)}")
            
            logging.info(f"Imported {articles_imported}/{total_articles} articles")
            pbar.update(1)
            
            # Authors with batching and progress
            author_collection = self.client.collections.get("Author")
            authors_imported = 0
            total_authors = len(data["authors"])
            
            with tqdm(total=total_authors, desc="Authors", leave=False) as author_pbar:
                for batch_data in self._batch_objects(data["authors"], self.batch_sizes["Author"]):
                    try:
                        with author_collection.batch.dynamic() as batch:
                            for author_id, author in batch_data.items():
                                properties = {
                                    "canonical_name": author["canonical_name"],
                                    "email": author.get("email", ""),
                                    "name_variants": author.get("name_variants", [])
                                }
                                uuid = batch.add_object(properties=properties)
                                self.uuid_map["Author"][author_id] = uuid
                                authors_imported += 1
                                author_pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to import author batch: {str(e)}")
            
            logging.info(f"Imported {authors_imported}/{total_authors} authors")
            pbar.update(1)
            
            # References with batching and progress
            reference_collection = self.client.collections.get("Reference")
            refs_imported = 0
            total_refs = len(data["references"])
            
            with tqdm(total=total_refs, desc="References", leave=False) as ref_pbar:
                for batch_data in self._batch_objects(data["references"], self.batch_sizes["Reference"]):
                    try:
                        with reference_collection.batch.dynamic() as batch:
                            for ref_id, ref in batch_data.items():
                                properties = {
                                    "title": ref.get("title", ""),
                                    "journal": ref.get("journal", ""),
                                    "publication_date": ref.get("publication_date", ""),
                                    "raw_reference": ref.get("raw_reference", "")
                                }
                                uuid = batch.add_object(properties=properties)
                                self.uuid_map["Reference"][ref_id] = uuid
                                refs_imported += 1
                                ref_pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to import reference batch: {str(e)}")
            
            logging.info(f"Imported {refs_imported}/{total_refs} references")
            pbar.update(1)
            
            # NamedEntities with batching and progress
            entity_collection = self.client.collections.get("NamedEntity")
            entities_imported = 0
            total_entities = len(data.get("ner_objects", {}))
            
            with tqdm(total=total_entities, desc="Named Entities", leave=False) as entity_pbar:
                for batch_data in self._batch_objects(data.get("ner_objects", {}), self.batch_sizes["NamedEntity"]):
                    try:
                        with entity_collection.batch.dynamic() as batch:
                            for entity_id, entity in batch_data.items():
                                properties = {
                                    "name": entity["name"],
                                    "type": entity["type"]
                                }
                                uuid = batch.add_object(properties=properties)
                                self.uuid_map["NamedEntity"][entity_id] = uuid
                                entities_imported += 1
                                entity_pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to import entity batch: {str(e)}")
            
            logging.info(f"Imported {entities_imported}/{total_entities} named entities")
            pbar.update(1)

        # Second Pass: Add cross-references
        logging.info("Adding cross-references...")
        with tqdm(total=4, desc="Adding cross-references") as pbar:
            # Article references
            article_collection = self.client.collections.get("Article")
            with tqdm(total=len(data["articles"]), desc="Article refs", leave=False) as ref_pbar:
                for batch_data in self._batch_objects(data["articles"], self.batch_sizes["Article"]):
                    try:
                        with article_collection.batch.dynamic() as batch:
                            for article_id, article in batch_data.items():
                                article_uuid = self.uuid_map["Article"][article_id]
                                
                                # Add author references
                                author_refs = [
                                    self.uuid_map["Author"][author_id]
                                    for author_id in article.get("authors", [])
                                    if author_id in self.uuid_map["Author"]
                                ]
                                
                                if author_refs:
                                    batch.add_reference(
                                        from_uuid=article_uuid,
                                        from_property="authors",
                                        to=author_refs
                                    )
                                ref_pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to add article references batch: {str(e)}")
            pbar.update(1)
            
            # Reference cross-references
            reference_collection = self.client.collections.get("Reference")
            with tqdm(total=len(data["references"]), desc="Reference refs", leave=False) as ref_pbar:
                for batch_data in self._batch_objects(data["references"], self.batch_sizes["Reference"]):
                    try:
                        with reference_collection.batch.dynamic() as batch:
                            for ref_id, ref in batch_data.items():
                                ref_uuid = self.uuid_map["Reference"][ref_id]
                                
                                # Add author references
                                author_refs = [
                                    self.uuid_map["Author"][author_id]
                                    for author_id in ref.get("authors", [])
                                    if author_id in self.uuid_map["Author"]
                                ]
                                
                                if author_refs:
                                    batch.add_reference(
                                        from_uuid=ref_uuid,
                                        from_property="authors",
                                        to=author_refs
                                    )
                                
                                # Add citations
                                for occ in ref.get("occurrences", []):
                                    article_id = occ["article_id"]
                                    if article_id in self.uuid_map["Article"]:
                                        batch.add_reference(
                                                                from_uuid=ref_uuid,
                                                                from_property="cited_in",
                                                                to=[self.uuid_map["Article"][article_id]]
                                                            )
                                ref_pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to add reference cross-references batch: {str(e)}")
            pbar.update(1)
            
            # Named Entity references
            entity_collection = self.client.collections.get("NamedEntity")
            with tqdm(total=len(data.get("ner_objects", {})), desc="Entity refs", leave=False) as ref_pbar:
                for batch_data in self._batch_objects(data.get("ner_objects", {}), self.batch_sizes["NamedEntity"]):
                    try:
                        with entity_collection.batch.dynamic() as batch:
                            for entity_id, entity in batch_data.items():
                                if entity_id not in self.uuid_map["NamedEntity"]:
                                    continue
                                    
                                entity_uuid = self.uuid_map["NamedEntity"][entity_id]
                                
                                # Add article references with scores
                                for occurrence in entity.get("occurrences", []):
                                    article_id = occurrence["article_id"]
                                    if article_id in self.uuid_map["Article"]:
                                        batch.add_reference(
                                            from_uuid=entity_uuid,
                                            from_property="article_scores",
                                            to=[self.uuid_map["Article"][article_id]]
                                        )
                                ref_pbar.update(1)
                    except Exception as e:
                        logging.error(f"Failed to add entity references batch: {str(e)}")
            pbar.update(1)
            
            # Citation Contexts
            context_collection = self.client.collections.get("CitationContext")
            contexts_created = 0
            with tqdm(total=len(data["references"]), desc="Contexts", leave=False) as ctx_pbar:
                for batch_data in self._batch_objects(data["references"], self.batch_sizes["Reference"]):
                    try:
                        with context_collection.batch.dynamic() as batch:
                            for ref_id, ref in batch_data.items():
                                for occ in ref.get("occurrences", []):
                                    article_id = occ["article_id"]
                                    if article_id not in self.uuid_map["Article"]:
                                        continue
                                        
                                    # Create citation context
                                    properties = {
                                        "sections": occ["sections"],
                                        "local_ref_id": occ["local_ref_id"],
                                        "context_before": occ.get("context_before", ""),
                                        "context_after": occ.get("context_after", ""),
                                        "citation_type": occ.get("citation_type", "unknown")
                                    }
                                    
                                    context_uuid = batch.add_object(properties=properties)
                                    
                                    # Add references
                                    batch.add_reference(
                                        from_uuid=context_uuid,
                                        from_property="reference",
                                        to=[self.uuid_map["Reference"][ref_id]]
                                    )
                                    batch.add_reference(
                                        from_uuid=context_uuid,
                                        from_property="article",
                                        to=[self.uuid_map["Article"][article_id]]
                                    )
                                    contexts_created += 1
                                ctx_pbar.update(1)
                    except Exception as e:
                                logging.error(f"Failed to create citation contexts batch: {str(e)}")
            logging.info(f"Created {contexts_created} citation contexts")
            pbar.update(1)

    def _process_article(self, article: Dict) -> Optional[Dict]:
        """Process article data for import, using token-aware text reduction."""
        try:
            # Add token count logging
            token_counts = {
                field: self._estimate_tokens(str(article.get(field, "")))
                for field in ["abstract", "introduction", "methods", "results", "discussion"]
            }
            logging.debug(f"Article {article['filename']}: Token counts = {token_counts}")
            
            # Validate required fields
            if not article.get("filename"):
                raise ValueError("Missing required field: filename")
                
            processed_article = {
                "filename": article["filename"],
                "title": article.get("title", ""),
                "publication_info": article.get("publication_info", "")
            }
            
            # Track reduction details
            reduction_info = {}
            for field in ["abstract", "introduction", "methods", "results", "discussion"]:
                if field in article:
                    content = article[field]
                    original_tokens = self._estimate_tokens(content)
                    reduction_info[field] = {"original_tokens": original_tokens}
                    
                    if original_tokens > self.max_tokens:
                        reduced_content = self._reduce_text_to_token_limit(content)
                        processed_article[field] = reduced_content
                        final_tokens = self._estimate_tokens(reduced_content)
                        reduction_info[field].update({
                            "was_reduced": True,
                            "final_tokens": final_tokens,
                            "reduction_ratio": final_tokens / original_tokens
                        })
                    else:
                        processed_article[field] = content
                        reduction_info[field]["was_reduced"] = False
            
            logging.debug(f"Article {article['filename']} reduction details: {reduction_info}")
            return processed_article
        
        except Exception as e:
            logging.error(f"Error processing article {article.get('filename', 'unknown')}: {str(e)}")
            logging.error(f"Article token counts: {reduction_info}")
            return None

    def verify_import(self, data: Dict[str, Dict]):
        """
        Verify that all data was imported correctly.
        
        Args:
            data: Dictionary containing all data to be imported
            
        Returns:
            Dict containing verification results and any discrepancies
        """
        results = {}
        
        # Map collection names to their data dictionary keys
        collection_data_keys = {
            "Article": "articles",
            "Author": "authors",
            "Reference": "references",
            "NamedEntity": "ner_objects"  # Fix: Use correct key for named entities
        }
        
        for collection_name, data_key in collection_data_keys.items():
            collection = self.client.collections.get(collection_name)
            actual_count = collection.aggregate.over_all(total_count=True).total_count
            expected_count = len(data.get(data_key, {}))
            
            results[collection_name] = {
                "expected": expected_count,
                "actual": actual_count,
                "missing": expected_count - actual_count if actual_count < expected_count else 0,
                "extra": actual_count - expected_count if actual_count > expected_count else 0
            }
            
        return results

def analyze_articles(articles: Dict[str, Dict]) -> Dict:
    """
    Analyze articles for size and chunking requirements.
    
    Analyzes:
    - Content size distribution
    - Fields requiring chunking
    - Potential import issues
    
    Args:
        articles: Dictionary of article data
        
    Returns:
        Dict containing:
            size_distribution: Article sizes in 10KB buckets
            needs_chunking: List of fields needing chunking
            potential_issues: List of problematic articles
    """
    analysis = {
        "size_distribution": defaultdict(int),
        "needs_chunking": [],
        "potential_issues": []
    }
    
    for article_id, article in articles.items():
        # Calculate total content size
        total_size = sum(len(str(article.get(field, ""))) 
                        for field in ["abstract", "introduction", "methods", "results", "discussion"])
        
        # Track size distribution
        size_bucket = total_size // 10000 * 10000  # Group in 10KB buckets
        analysis["size_distribution"][size_bucket] += 1
        
        # Check for chunking needs
        for field in ["abstract", "introduction", "methods", "results", "discussion"]:
            if field in article and len(str(article[field])) > MAX_STRING_LENGTH:
                analysis["needs_chunking"].append({
                    "article_id": article_id,
                    "field": field,
                    "size": len(str(article[field]))
                })
        
        # Flag potential issues
        if total_size > 1000000:  # Flag articles > 1MB
            analysis["potential_issues"].append({
                "article_id": article_id,
                "total_size": total_size,
                "reason": "Very large content"
            })
    
    return analysis

#-----------------------------------------------------------------------------
# CLI Interface
#-----------------------------------------------------------------------------

def main():
    """
    Command-line interface for Weaviate database management.
    
    Operations:
    ----------
    Database Management:
        --info: Show collection statistics and content samples
        --cleanup: Remove existing collections (with confirmation)
        --force: Skip confirmation prompts during cleanup
        --show-models: Display available OpenAI embedding models
    
    Data Operations:
        --data-dir PATH: Directory containing processed JSON files
        --subset-size N: Create test subset preserving relationships
        --show-import: Display data summary without importing
        --analyze: Check article sizes and chunking needs
    
    Processing Flow:
    1. Load and validate input files
    2. Analyze content if requested
    3. Create schema if needed
    4. Import data with progress tracking
    5. Verify import results
    """
    import tracemalloc
    tracemalloc.start()
    
    import argparse
    from contextlib import contextmanager
    import atexit  # Add this import
    
    parser = argparse.ArgumentParser(description='Weaviate Database Management')
    parser.add_argument('--info', action='store_true', 
                       help='Show collection statistics and content samples')
    parser.add_argument('--cleanup', action='store_true',
                       help='Remove existing collections (with confirmation)')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts')
    parser.add_argument('--show-models', action='store_true',
                       help='Display available OpenAI embedding models')
    parser.add_argument('--data-dir', type=Path,
                       help='Directory containing processed JSON files')
    parser.add_argument('--subset-size', type=int,
                       help='Create test subset preserving relationships')
    parser.add_argument('--show-import', action='store_true',
                       help='Display data summary without importing')
    parser.add_argument('--analyze', action='store_true',
                       help='Check article sizes and chunking needs')
    
    args = parser.parse_args()
    
    # Configure logging with proper cleanup
    log_file = 'weaviate_import.log'
    console_handler = logging.StreamHandler(sys.stderr)
    file_handler = logging.FileHandler(log_file)
    
    # Add cleanup function
    def cleanup_handlers():
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
    
    # Register cleanup
    atexit.register(cleanup_handlers)
    
    # Configure handlers
    console_handler.addFilter(lambda record: record.name == __name__)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[file_handler, console_handler]
    )
    
    # Suppress HTTP request logging unless there's an error
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
    
    # Reduce logging from libraries more aggressively
    for logger_name in [
        "weaviate", 
        "weaviate.batch", 
        "weaviate.auth",
        "weaviate.client",
        "weaviate.collections",
        "weaviate.connect"
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        # Only log HTTP related messages if they're errors
        for handler in logger.handlers:
            handler.addFilter(
                lambda record: record.levelno >= logging.ERROR if 'HTTP Request' in record.msg else True
            )
    
    # Only show our application logs on console
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.addFilter(lambda record: record.name == __name__)
    
    # File handler gets everything
    file_handler = logging.FileHandler('weaviate_import.log')
    
    # Replace handlers
    root_logger = logging.getLogger()
    root_logger.handlers = [console_handler, file_handler]
    
    if args.show_models:
        check_available_models()
        return
    
    try:
        if args.cleanup or args.info:
            with weaviate_client() as client:
                # Handle cleanup if requested
                if args.cleanup:
                            logging.info("Cleaning up existing collections")
                            cleanup_collections(client, force=args.force)
                        
                # Show database info if requested
                if args.info:
                    collections = get_existing_collections(client)
                    print("\nExisting Collections:", collections)
                    summary = get_database_summary(client)
                    print_database_summary(summary)

        # Load data if directory provided
        if args.data_dir:
            logging.info(f"Loading data from {args.data_dir}")
            data_manager = LiteratureDataManager(args.data_dir)
            data_manager.load_data()
            
            # Run analysis if requested
            if args.analyze:
                articles = data_manager.articles
                if args.subset_size:
                    # Get subset for analysis
                    subset_data = data_manager.extract_connected_subset(args.subset_size)
                    articles = subset_data['articles']
                
                analysis = analyze_articles(articles)
                
                print("\nArticle Analysis:")
                print("\nSize Distribution (in 10KB buckets):")
                for size, count in sorted(analysis["size_distribution"].items()):
                    print(f"  {size//1000}KB - {(size+10000)//1000}KB: {count} articles")
                
                print("\nArticles Requiring Chunking:")
                for chunk_info in analysis["needs_chunking"]:
                    print(f"  {chunk_info['article_id']}:")
                    print(f"    Field: {chunk_info['field']}")
                    print(f"    Size: {chunk_info['size']/1000:.1f}KB")
                
                print("\nPotential Issues:")
                for issue in analysis["potential_issues"]:
                    print(f"  {issue['article_id']}:")
                    print(f"    Reason: {issue['reason']}")
                    print(f"    Total Size: {issue['total_size']/1000:.1f}KB")
                
                if not args.show_import:  # Don't continue to import if only analyzing
                    return
            
            # Extract subset if requested
            data_to_load = None
            if args.subset_size:
                logging.info(f"Extracting subset of {args.subset_size} articles")
                data_to_load = data_manager.extract_connected_subset(args.subset_size)
            else:
                logging.info("Using complete dataset")
                data_to_load = {
                    'articles': data_manager.articles,
                    'authors': data_manager.authors,
                    'references': data_manager.references,
                    'ner_objects': data_manager.ner_objects
                }

            if args.show_import:  # Only show full dataset summary if requested
                data_manager.print_subset_summary(data_to_load)
                sys.exit(0)
            
            # Database operations
            with weaviate_client() as client:
                # Create schema and import data
                if data_to_load:
                    logging.info("Creating schema")
                    create_schema(client, OPENAI_BASE_URL)
                    
                    logging.info("Importing data")
                    importer = DataImporter(client)
                    importer.import_data(data_to_load)
                    
                    logging.info("Data import complete")
                    
                    if args.info:
                        # Show updated database info
                        collections = get_existing_collections(client)
                        print("\nUpdated Collections:", collections)
                        summary = get_database_summary(client)
                        print_database_summary(summary)
        
                    # After import, verify the results
                    if data_to_load:
                        logging.info("Verifying import...")
                        verification = importer.verify_import(data_to_load)
                        
                        print("\nImport Verification:")
                        for collection, stats in verification.items():
                            print(f"\n{collection}:")
                            print(f"  Expected: {stats['expected']}")
                            print(f"  Actual: {stats['actual']}")
                            if stats['missing']:
                                print(f"  Missing: {stats['missing']} records")
                            if stats['extra']:
                                print(f"  Extra: {stats['extra']} records")
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        sys.exit(1)
    finally:
        tracemalloc.stop()
        cleanup_handlers()  # Ensure handlers are cleaned up

if __name__ == "__main__":
    main()