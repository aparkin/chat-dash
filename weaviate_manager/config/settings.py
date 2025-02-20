"""
Configuration settings for Weaviate Database Manager.

This module defines all configuration constants and parameters for the system,
organized into the following sections:

OpenAI API Configuration:
- API endpoints and authentication
- Model selection and parameters
- Token limits and usage controls

Token Management Configuration:
- Per-model token limits with safety margins
- Text compression thresholds and ratios
- Content preservation settings
- Token estimation parameters

Text Processing Settings:
- Token counting and estimation
- Text compression thresholds
- Content preservation ratios
- Importance scoring weights

Database Configuration:
- Collection definitions and schemas
- Batch processing sizes
- Connection parameters and timeouts
- Security settings

Vector Search Parameters:
- Default result limits
- Similarity thresholds
- Search optimization settings

Each section is clearly marked with comments and includes detailed
descriptions of the parameters and their valid ranges where applicable.
Environment-specific settings (development vs. production) are also handled.

Token Management Details:
- MODEL_TOKEN_LIMITS: Maximum tokens per model
- TOKEN_SAFETY_MARGIN: Buffer for token limits (e.g., 0.95 = 95% of max)
- DEFAULT_MAX_TOKENS: Safe token limit for current model
- WORDS_PER_TOKEN: Conservative estimate for text analysis

Compression Configuration:
- TEXT_REDUCTION_RATIOS: Field-specific compression ratios
- IMPORTANCE_WEIGHTS: Scoring weights for content preservation
- IMPORTANCE_KEY_TERMS: Scientific terms to preserve
- IMPORTANCE_COMPARISON_TERMS: Comparison terms to preserve
"""
import os
from dotenv import load_dotenv
from pathlib import Path

# Find the project root directory (where .env is located)
project_root = Path(__file__).parent.parent.parent
dotenv_path = project_root / '.env'

# Try to load from .env file
load_dotenv(dotenv_path=dotenv_path)

# OpenAI Settings
if False:  # Toggle for development environment
    OPENAI_BASE_URL = os.getenv('CBORG_BASE_URL', "https://api.cborg.lbl.gov")
    OPENAI_API_KEY = os.getenv('CBORG_API_KEY', '')  # Must be set in environment
else:  # Production environment
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Must be set in environment

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")


#-----------------------------------------------------------------------------
# AI Model Configuration
#-----------------------------------------------------------------------------

# Vectorization Settings
VECTORIZER_MODEL = "text-embedding-3-large"  # OpenAI embedding model
VECTOR_DIMENSIONS = 1024  # Embedding dimensions [256, 1024, 3072]

# Generative AI Settings
GENERATIVE_MODEL = "gpt-4"  # Model for text generation
GENERATIVE_MAX_TOKENS = 4096  # Maximum tokens for generated responses
GENERATIVE_TEMPERATURE = 0.7  # Controls randomness (0.0-1.0)
GENERATIVE_TOP_P = 0.95  # Nucleus sampling parameter
GENERATIVE_FREQUENCY_PENALTY = 0.5  # Reduces repetition (-2.0 to 2.0)
GENERATIVE_PRESENCE_PENALTY = 0.5  # Encourages new topics (-2.0 to 2.0)

#-----------------------------------------------------------------------------
# Text Processing Configuration
#-----------------------------------------------------------------------------

# Token Limits for Different Models
MODEL_TOKEN_LIMITS = {
    "text-embedding-3-small": 8191,   # OpenAI's small model limit
    "text-embedding-3-large": 8191,   # OpenAI's large model limit
    "text-embedding-ada-002": 8191,   # Legacy model limit
}

# Safety margin for token limits (90% of max)
TOKEN_SAFETY_MARGIN = 0.95

# Current model's limit with safety margin
DEFAULT_MAX_TOKENS = int(MODEL_TOKEN_LIMITS[VECTORIZER_MODEL] * TOKEN_SAFETY_MARGIN)
WORDS_PER_TOKEN = 0.75  # Conservative estimate for words-to-tokens ratio

# Text Compression Settings
MAX_STRING_LENGTH = 65535  # Maximum string length for Weaviate
COMPRESSION_THRESHOLD = MAX_STRING_LENGTH * 0.9  # When to trigger compression

# Text Reduction Ratios
TEXT_REDUCTION_RATIOS = {
    "introduction": 0.4,    # Preserve 40% for introduction (start of text)
    "key_points": 0.3,      # Preserve 30% for key points (middle, scored by importance)
    "conclusion": 0.3       # Preserve 30% for conclusion (end of text)
}

# Importance Scoring Weights
IMPORTANCE_WEIGHTS = {
    "key_terms": 2.0,       # Weight for scientific key terms (significant, demonstrate, etc.)
    "comparisons": 1.0,     # Weight for comparison terms (whereas, however, etc.)
    "numbers": 1.0,         # Weight for numerical content
    "citations": 1.0        # Weight for citations/references
}

# Key terms for importance scoring
IMPORTANCE_KEY_TERMS = [
    'significant', 'important', 'key', 'novel', 'demonstrate',
    'show', 'find', 'conclude', 'reveal', 'result', 'observe',
    'measure', 'analyze', 'determine', 'identify'
]

IMPORTANCE_COMPARISON_TERMS = [
    'whereas', 'however', 'although', 'unlike', 'compared',
    'contrast', 'instead', 'rather', 'while'
]

#-----------------------------------------------------------------------------
# Collection Configuration
#-----------------------------------------------------------------------------

# Managed Collections with Descriptions
MANAGED_COLLECTIONS = [
    "Article",           # Base article data with full text content
    "Author",           # Author information and canonical names
    "NameVariant",      # Alternative author name representations
    "Reference",        # Citation and reference information
    "NamedEntity",      # Named entities and ontological terms
    "NERArticleScore",  # Entity-article relevance scores
    "CitationContext"   # Citation contexts within articles
]

# Article Sections for Processing
ARTICLE_SECTIONS = [
    "abstract",
    "introduction", 
    "methods",
    "results", 
    "discussion", 
    "figures", 
    "tables"
]

# Batch Processing Sizes
BATCH_SIZES = {
    "Article": 1,      # Articles have large text content
    "Reference": 100,    # References have moderate content
    "Author": 100,      # Author records are small
    "NamedEntity": 100, # Entity records are small
    "NameVariant":100, # Name variants are small text records
    "CitationContext": 100, # Citation contexts have moderate content
    "NERArticleScore": 100, # Score records are very small
    "default": 50       # Default for other collections
}

#-----------------------------------------------------------------------------
# Database Connection Settings
#-----------------------------------------------------------------------------

# Weaviate Connection
WEAVIATE_HOST = "weaviate.kbase.us"
WEAVIATE_HTTP_PORT = 443
WEAVIATE_GRPC_HOST = "weaviate-grpc.kbase.us"
WEAVIATE_GRPC_PORT = 443
WEAVIATE_SECURE = True

# Timeouts (in seconds)
REQUEST_TIMEOUT = (1200, 1200)    # General API request timeout
VECTORIZER_TIMEOUT = 2000        # Specific timeout for vectorization operations

#-----------------------------------------------------------------------------
# Vector Search Configuration
#-----------------------------------------------------------------------------

# Search Configuration
DEFAULT_LIMIT = 10  # Default number of results to return
DEFAULT_CERTAINTY = 0.7  # Default minimum similarity score
VECTOR_SEARCH_LIMIT = 100  # Maximum number of results for vector search
DEFAULT_ALPHA = 0.5  # Default hybrid search balance between vector and keyword
MAX_FIELD_LENGTH = 200  # Maximum length for displayed text fields

# Collection names for cross-collection search
SEARCHABLE_COLLECTIONS = [
    "Article",
    "Author", 
    "Reference",
    "NamedEntity",
    #"CitationContext",
    #"NERArticleScore",
    "NameVariant"
] 