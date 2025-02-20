"""
Command Line Interface for Weaviate Database Manager.

This module provides a CLI for managing a scientific literature database in Weaviate.
Key functionality includes:

Database Management:
- Schema creation and validation
- Collection cleanup and data import
- Database statistics and health monitoring

Data Operations:
- Importing processed scientific articles
- Creating and analyzing data subsets
- Verifying data consistency

Search and Query:
- Semantic search using OpenAI embeddings
- Keyword (BM25) search
- Hybrid search combining both approaches
- Result unification across collections

Monitoring:
- Configuration display
- Schema visualization
- Database statistics
- Detailed logging with configurable verbosity

The CLI is designed for both basic operations and advanced database management,
with comprehensive error handling and progress tracking.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
import json

import weaviate
import openai

from .config.settings import (
    VECTORIZER_MODEL,
    VECTOR_DIMENSIONS,
    MANAGED_COLLECTIONS,
    WEAVIATE_HOST,
    WEAVIATE_HTTP_PORT,
    WEAVIATE_SECURE,
    GENERATIVE_MODEL,
    MODEL_TOKEN_LIMITS,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    DEFAULT_LIMIT,
    DEFAULT_CERTAINTY
)
from .database.schema import SchemaGenerator
from .database.client import get_client
from .data.loader import LiteratureDataManager
from .database.inspector import DatabaseInspector
from .database.importer import WeaviateImporter
from .query.manager import QueryManager
from .query.result_formatter import ResultFormatter
from .query.visualizers import ResultVisualizer

def setup_logging(verbose: bool = False):
    """
    Configure logging with file and console handlers.
    
    File Handler:
    - Writes to 'weaviate_import.log'
    - Always logs at DEBUG level
    - Includes timestamps and module context
    
    Console Handler:
    - DEBUG level when verbose=True
    - INFO level when verbose=False
    - Simplified format for readability
    
    Args:
        verbose: If True, shows DEBUG level messages in console
    
    Side Effects:
        - Creates/overwrites 'weaviate_import.log'
        - Configures root logger and module loggers
        - Suppresses external library logs unless verbose
    """
    # Configure file logging (always at DEBUG level for troubleshooting)
    file_handler = logging.FileHandler('weaviate_import.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Configure console logging (only important messages)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)  # Changed to show more logs
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)
    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure weaviate_manager logger and its children
    for logger_name in [
        'weaviate_manager',
        'weaviate_manager.query',
        'weaviate_manager.query.manager',
        'weaviate_manager.query.result_formatter'
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    # Only suppress external logs if not in verbose mode
    if not verbose:
        for logger_name in [
            "weaviate",
            "weaviate.batch",
            "weaviate.auth",
            "weaviate.client",
            "weaviate.collections",
            "weaviate.connect",
            "weaviate.schema",
            "urllib3.connectionpool"
        ]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)

def log_progress(message: str, count: Optional[int] = None, total: Optional[int] = None):
    """
    Log a progress message with optional completion statistics.
    
    Creates a specially formatted progress message that includes:
    - Custom status message
    - Current count and total (if provided)
    - Percentage completion (if count and total provided)
    
    Args:
        message: Progress status message
        count: Current number of processed items
        total: Total number of items to process
    
    Side Effects:
        - Creates log record with progress=True flag
        - Sends message through configured handlers
    """
    if count is not None and total is not None:
        percentage = (count / total) * 100
        message = f"{message} [{count}/{total} - {percentage:.1f}%]"
    
    # Create a log record with progress flag
    logger = logging.getLogger()
    record = logger.makeRecord(
        logger.name, logging.INFO, "", 0, message, (), None
    )
    record.progress = True
    logger.handle(record)

def show_config():
    """
    Display current configuration settings in formatted tables.
    
    Shows:
    1. Main Configuration:
       - Vectorizer model and dimensions
       - Weaviate connection details
       - OpenAI settings (API key masked)
    
    2. Collections Configuration:
       - Managed collections list
       - Token limits per collection
    
    Side Effects:
        - Prints formatted tables to console
        - Logs full configuration at DEBUG level
    """
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Create main configuration table
    config_table = Table(title="Current Configuration")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    # Add main settings
    config_table.add_row("Vectorizer Model", VECTORIZER_MODEL)
    config_table.add_row("Vector Dimensions", str(VECTOR_DIMENSIONS))
    config_table.add_row("Weaviate Host", f"{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}")
    config_table.add_row("Secure Connection", str(WEAVIATE_SECURE))
    config_table.add_row("OpenAI Base URL", OPENAI_BASE_URL)
    config_table.add_row("OpenAI API Key", "***" + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else "Not Set")
    
    console.print(config_table)
    
    # Create collections table
    collections_table = Table(title="\nManaged Collections")
    collections_table.add_column("Collection", style="cyan")
    collections_table.add_column("Token Limit", style="yellow")
    
    for collection in MANAGED_COLLECTIONS:
        collections_table.add_row(collection, str(MODEL_TOKEN_LIMITS.get(collection, "N/A")))
    
    console.print(collections_table)
    
    # Log full config to debug file only
    logging.debug("Full configuration:")
    logging.debug(f"  VECTORIZER_MODEL: {VECTORIZER_MODEL}")
    logging.debug(f"  VECTOR_DIMENSIONS: {VECTOR_DIMENSIONS}")
    logging.debug(f"  WEAVIATE_HOST: {WEAVIATE_HOST}")
    logging.debug(f"  WEAVIATE_HTTP_PORT: {WEAVIATE_HTTP_PORT}")
    logging.debug(f"  WEAVIATE_SECURE: {WEAVIATE_SECURE}")
    logging.debug(f"  MANAGED_COLLECTIONS: {MANAGED_COLLECTIONS}")
    logging.debug(f"  MODEL_TOKEN_LIMITS: {MODEL_TOKEN_LIMITS}")

def get_database_info(client: weaviate.Client):
    """
    Display comprehensive information about the current database state.
    
    Retrieves and displays:
    - List of collections and their sizes
    - Schema structure and properties
    - Cross-reference configurations
    - Overall database statistics
    
    Args:
        client: Connected Weaviate client instance
    
    Side Effects:
        - Prints formatted database information to console
        - Logs detailed information to debug log
    """
    try:
        inspector = DatabaseInspector(client)
        info = inspector.get_database_info()
        
        if not info['collections']:
            print("[yellow]No collections found in database[/yellow]")
            return
            
        inspector.display_schema_summary()
        
        # Log detailed info for debugging
        logging.debug(f"Full database info: {info}")
        
    except Exception as e:
        logging.error(f"Error getting database info: {str(e)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Detailed error:", exc_info=True)

def cleanup_collections(client: weaviate.Client, force: bool = False) -> bool:
    """
    Remove all managed collections from the database.
    
    Features:
    - Interactive confirmation unless force=True
    - Verification of complete removal
    - Detailed progress logging
    
    Args:
        client: Connected Weaviate client
        force: Skip confirmation prompt if True
    
    Returns:
        bool: True if all collections removed successfully
    
    Side Effects:
        - Deletes collections from database
        - Logs cleanup progress and results
    """
    try:
        # Get existing collections using new API
        collections = client.collections.list_all(simple=True)
        existing_collections = list(collections.keys())
        
        if not force:
            response = input("This will delete all managed collections. Continue? [y/N] ")
            if response.lower() != 'y':
                logging.info("Cleanup cancelled")
                return False
        
        success = True
        for collection in MANAGED_COLLECTIONS:
            try:
                if collection in existing_collections:
                    client.collections.delete(collection)
                    logging.info(f"Deleted collection: {collection}")
            except Exception as e:
                logging.warning(f"Error deleting {collection}: {str(e)}")
                success = False
        
        # Verify cleanup
        remaining = client.collections.list_all(simple=True)
        managed_remaining = [c for c in remaining.keys() if c in MANAGED_COLLECTIONS]
        if managed_remaining:
            logging.warning(f"Managed collections remaining after cleanup: {managed_remaining}")
            return False
        else:
            logging.info("All managed collections removed successfully")
            return True
        
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        return False

def create_schema(client: weaviate.Client):
    """
    Create database schema using SchemaGenerator.
    
    Creates a complete schema including:
    - Primary collections (Article, Author, Reference, etc.)
    - Properties for each collection
    - Cross-references between collections
    - Vectorizer and generative AI configurations
    
    Args:
        client: Connected Weaviate client instance
    
    Side Effects:
        - Creates collections in database
        - Logs creation progress and any errors
    """
    try:
        generator = SchemaGenerator(client)
        generator.create_schema()
        logging.info("Schema creation completed")
        
    except Exception as e:
        logging.error(f"Error creating schema: {str(e)}")

def get_schema_info(client: weaviate.Client) -> dict:
    """
    Get detailed information about current schema and collections.
    
    Retrieves comprehensive schema information including:
    - Collection configurations and descriptions
    - Property definitions and types
    - Cross-reference configurations
    - Object counts per collection
    
    Args:
        client: Connected Weaviate client instance
    
    Returns:
        dict: Schema information with structure:
            {
                'collections': {
                    'collection_name': {
                        'name': str,
                        'description': str,
                        'properties': list[dict],
                        'references': list[dict],
                        'vectorizer': str,
                        'object_count': int
                    }
                },
                'total_objects': int
            }
    
    Raises:
        Exception: If schema retrieval fails
    """
    try:
        schema = client.schema.get()
        info = {
            'collections': {},
            'total_objects': 0
        }
        
        if not schema.get('classes'):
            return info
            
        for cls in schema['classes']:
            name = cls['class']
            if name in MANAGED_COLLECTIONS:
                collection_info = {
                    'name': name,
                    'description': cls.get('description', 'No description'),
                    'properties': [],
                    'references': [],
                    'vectorizer': cls.get('vectorizer'),
                    'object_count': client.collections.get(name).aggregate.over_all().objects
                }
                
                info['total_objects'] += collection_info['object_count']
                
                # Get properties
                for prop in cls.get('properties', []):
                    prop_type = prop.get('dataType', [''])[0]
                    if isinstance(prop_type, str) and prop_type in MANAGED_COLLECTIONS:
                        collection_info['references'].append({
                            'name': prop['name'],
                            'target': prop_type,
                            'description': prop.get('description', '')
                        })
                    else:
                        collection_info['properties'].append({
                            'name': prop['name'],
                            'type': prop_type,
                            'description': prop.get('description', '')
                        })
                
                info['collections'][name] = collection_info
        
        return info
        
    except Exception as e:
        logging.error(f"Error getting schema info: {str(e)}")
        return None

def show_schema(client: weaviate.Client):
    """
    Display detailed information about current schema in formatted tables.
    
    Shows multiple tables including:
    1. Schema Overview:
       - Collection names and descriptions
       - Object counts per collection
       - Total object count
    
    2. Collection Details (for each collection):
       - Properties and their types
       - Cross-references and their targets
       - Vectorizer configuration
    
    Args:
        client: Connected Weaviate client instance
    
    Side Effects:
        - Prints formatted schema information to console
        - Logs any errors encountered
    """
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    try:
        inspector = DatabaseInspector(client)
        schema_info = inspector.get_schema_info()
        
        if not schema_info:
            console.print("[yellow]No schema exists in database[/yellow]")
            return
        
        # Create overview table
        overview = Table(title="Schema Overview", show_header=True)
        overview.add_column("Collection", style="cyan")
        overview.add_column("Description", style="green")
        overview.add_column("Properties", style="yellow")
        overview.add_column("References", style="magenta")
        overview.add_column("Vectorized", style="blue")
        
        for collection_name, config in schema_info.items():
            vectorized_props = sum(1 for p in config.get('properties', []) if p.get('vectorize'))
            overview.add_row(
                collection_name,
                config.get('description', 'No description'),
                str(len(config.get('properties', []))),
                str(len(config.get('references', []))),
                f"✓ ({vectorized_props} props)" if vectorized_props else ""
            )
        
        console.print(overview)
        console.print()
        
        # Show detailed information for each collection
        for collection_name, config in schema_info.items():
            # Collection header with configuration info
            header = Text()
            header.append(collection_name, style="bold cyan")
            if config.get('vectorizer'):
                header.append(f"\nVectorizer: {config['vectorizer']}", style="dim")
            if config.get('vectorizer_config'):
                header.append(f"\nVectorizer Config: {config['vectorizer_config']}", style="dim")
            if config.get('generative_config'):
                header.append(f"\nGenerative Config: {config['generative_config']}", style="dim")
            console.print(Panel(header))
            
            # Properties table
            if config.get('properties'):
                prop_table = Table(show_header=True, header_style="bold", title="Properties")
                prop_table.add_column("Name", style="cyan")
                prop_table.add_column("Type", style="green")
                prop_table.add_column("Description")
                prop_table.add_column("Vectorized", justify="center", style="blue")
                prop_table.add_column("Tokenization", style="yellow")
                prop_table.add_column("Indexing", style="magenta")
                
                for prop in config['properties']:
                    indexing = []
                    if prop.get('indexFilterable'):
                        indexing.append("filterable")
                    if prop.get('indexSearchable'):
                        indexing.append("searchable")
                    
                    prop_table.add_row(
                        prop['name'],
                        prop['type'],
                        prop.get('description', ''),
                        "✓" if prop.get('vectorize') else "",
                        str(prop.get('tokenization', '')),
                        ", ".join(indexing) if indexing else ""
                    )
                console.print(prop_table)
            
            # References table
            if config.get('references'):
                ref_table = Table(show_header=True, header_style="bold", title="References")
                ref_table.add_column("Name", style="magenta")
                ref_table.add_column("Target Collection", style="cyan")
                ref_table.add_column("Description")
                
                for ref in config['references']:
                    ref_table.add_row(
                        ref['name'],
                        ref['target'],
                        ref.get('description', '')
                    )
                console.print(ref_table)
            
            console.print()  # Add spacing between collections
        
        # Log raw info for debugging
        logging.debug("Raw schema info:")
        for collection_name, config in schema_info.items():
            logging.debug(f"\nCollection {collection_name}:")
            for key, value in config.items():
                logging.debug(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"[red]Error showing schema: {str(e)}[/red]")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Detailed error:", exc_info=True)

def show_models():
    """Display available AI models and their configurations."""
    from .config.settings import OPENAI_API_KEY, OPENAI_BASE_URL
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    # Create summary table
    table = Table(title="AI Model Configuration")
    table.add_column("Type", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Token Limit", style="yellow")
    
    # Add configured models
    table.add_row("Vectorizer", VECTORIZER_MODEL, str(MODEL_TOKEN_LIMITS.get(VECTORIZER_MODEL, "N/A")))
    table.add_row("Generative", GENERATIVE_MODEL, str(MODEL_TOKEN_LIMITS.get(GENERATIVE_MODEL, "N/A")))
    
    console.print(table)
    console.print("\nChecking available models...", style="bold blue")
    
    try:
        # Configure OpenAI client
        client = openai.OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        
        # Get models using the client
        models = client.models.list()
        
        # Create available models table
        model_table = Table(title="Available OpenAI Models")
        model_table.add_column("Model ID", style="green")
        model_table.add_column("Status", style="cyan")
        
        for model in models:
            model_table.add_row(
                model.id,
                "✓ In Use" if model.id in [VECTORIZER_MODEL, GENERATIVE_MODEL] else ""
            )
        
        console.print(model_table)
        
        # Log detailed response to debug file only
        logging.debug(f"Full models response: {models}")
            
    except Exception as e:
        console.print(f"[red]Error checking models: {str(e)}[/red]")
        logging.exception("Detailed error information:")

def show_statistics(client: weaviate.Client):
    """Display detailed statistics about loaded data."""
    try:
        inspector = DatabaseInspector(client)
        inspector.display_stats_rich()
        
    except Exception as e:
        logging.error(f"Error showing statistics: {str(e)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("Detailed error:", exc_info=True)

def add_data_args(parser):
    data_group = parser.add_argument_group('Data Management')
    data_group.add_argument('--input-dir', type=str, help='Directory containing input JSON files to load')
    data_group.add_argument('--verify', action='store_true', help='Verify input files')
    data_group.add_argument('--import', action='store_true', help='Import loaded data into database (creates schema if needed)')
    data_group.add_argument('--summarize', action='store_true', help='Show summary of loaded data')
    data_group.add_argument('--detail-level', choices=['summary', 'detailed', 'full'], 
                           default='summary', 
                           help='''Level of detail for data summary:
                                 summary: Basic collection sizes and distributions
                                 detailed: Adds section and NER type breakdowns
                                 full: Adds relationship pattern analysis''')

def schema_exists(client: weaviate.Client) -> bool:
    """Check if schema exists in database."""
    try:
        schema = client.schema.get()
        # Check if any of our managed collections exist
        return any(cls['class'] in MANAGED_COLLECTIONS for cls in schema.get('classes', []))
    except Exception as e:
        logging.error(f"Error checking schema: {str(e)}")
        return False

def handle_schema_creation(client: weaviate.Client, cleanup: bool = False, force: bool = False) -> bool:
    """Handle schema creation logic.
    
    Args:
        client: Weaviate client
        cleanup: Whether to cleanup existing schema
        force: Whether to skip cleanup confirmation
        
    Returns:
        bool: True if schema exists or was created successfully
    """
    try:
        if cleanup:
            cleanup_collections(client, force)
            
        if not schema_exists(client):
            logging.info("Creating schema...")
            generator = SchemaGenerator(client)
            if generator.create_schema():
                logging.info("Schema created successfully")
                return True
            else:
                logging.error("Schema creation failed")
                return False
        else:
            if cleanup:
                logging.error("Schema cleanup failed")
                return False
            logging.info("Schema already exists")
            return True
            
    except Exception as e:
        logging.error(f"Error in schema creation: {str(e)}")
        return False

def handle_data_operations(args, client: Optional[weaviate.Client] = None) -> bool:
    """Handle data loading, schema generation, and import operations."""
    try:
        # Ensure input directory is provided for data operations
        if not args.input_dir:
            if args.summarize:
                logging.error("--summarize requires --input-dir to be specified")
                return False
            elif args.subset_size:
                logging.error("--subset-size requires --input-dir to be specified")
                return False
        
        # Create data manager and load data if input directory provided
        if args.input_dir:
            logging.debug(f"Creating data manager with input directory: {args.input_dir}")
            manager = LiteratureDataManager(args.input_dir)
            
            # Verify input files if requested
            if args.verify:
                logging.debug("Verifying input files")
                valid, errors = manager.verify_input_files()
                if not valid:
                    logging.error("Input file verification failed:")
                    for error in errors:
                        logging.error(f"  {error}")
                    return False
                logging.debug("Input files verified successfully")
            
            # Load data
            logging.debug("Loading data")
            if not manager.load_data():
                logging.error("Failed to load data")
                return False

            # Always show full dataset statistics first if summarize is requested
            if args.summarize:
                logging.info("\n" + "="*80)
                logging.info("FULL DATASET STATISTICS")
                logging.info("="*80)
                manager.print_statistics(detail_level=args.detail_level)

            # Handle subset if requested
            subset_manager = None
            if args.subset_size:
                logging.debug(f"Processing subset request for size {args.subset_size}")
                if args.subset_size <= 0:
                    logging.error("Subset size must be positive")
                    return False

                try:
                    subset_manager, analysis = manager.create_and_analyze_subset(
                        size=args.subset_size,
                        seed_article=args.seed_article
                    )
                    if not subset_manager:
                        logging.error("Failed to create subset")
                        return False
                    logging.debug("Subset created and analyzed successfully")
                    
                    # Show subset statistics if requested
                    if args.summarize:
                        logging.info("\n" + "="*80)
                        logging.info("SUBSET STATISTICS")
                        logging.info("="*80)
                        subset_manager.print_statistics(detail_level=args.detail_level)
                        
                        # Display collection coverage from analysis
                        logging.info("\nCollection Coverage:")
                        for collection, stats in analysis["collection_coverage"].items():
                            logging.info(f"  {collection}: {stats['size']}/{stats['total']} ({stats['percentage']:.1f}%)")
                        
                        # Display consistency check results
                        if analysis["consistency_check"]["external_references"]:
                            logging.warning("\nExternal References Found:")
                            for ref_type, count in analysis["consistency_check"]["external_references"].items():
                                logging.warning(f"  {ref_type}: {count}")
                        else:
                            logging.info("\nNo external references found - subset is self-contained")
                    
                except ValueError as e:
                    logging.error(f"Error creating subset: {str(e)}")
                    return False

            # If we have a client and import is requested, handle database operations
            if client and args.import_:
                # Generate schema if needed
                logging.info("Checking schema status...")
                inspector = DatabaseInspector(client)
                schema_info = inspector.get_schema_info()
                
                if not schema_info:
                    logging.info("No schema found, generating new schema...")
                    schema_generator = SchemaGenerator(client)
                    if not schema_generator.create_schema():
                        logging.error("Failed to create schema")
                        return False
                    logging.info("Schema created successfully")
                else:
                    logging.info("Schema already exists")

                # Import data
                data_manager_to_import = subset_manager if subset_manager else manager
                importer = WeaviateImporter(data_manager_to_import, client)
                if not importer.import_data():
                    logging.error("Failed to import data")
                    return False
                
                logging.info("Import completed successfully")

        logging.debug("All data operations completed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Unexpected error in data operations: {str(e)}")
        logging.debug("Stack trace:", exc_info=True)
        return False

def add_query_args(parser):
    """Add query-related arguments to parser."""
    query_group = parser.add_argument_group('Query Operations')
    query_group.add_argument('--query', type=str, help='Search query text')
    query_group.add_argument('--search-type',
                           choices=['semantic', 'keyword', 'hybrid'],
                           default='hybrid',
                           help='Type of search to perform')
    query_group.add_argument('--alpha', type=float,
                           default=None,
                           help='Balance between keyword and vector search for hybrid (0-1)')
    query_group.add_argument('--min-score', type=float,
                           default=0.0,
                           help='''Minimum score threshold for results. 
                           For semantic search: 0-1 (distance, lower is better)
                           For keyword search: typically >0 (BM25, higher is better)
                           For hybrid search: uses original hybrid score''')
    query_group.add_argument('--limit', type=int,
                           default=DEFAULT_LIMIT,
                           help='Maximum number of results per collection')
    query_group.add_argument('--unify',
                           action='store_true',
                           help='Unify results on articles with cross-references')
    query_group.add_argument('--include-vectors',
                           action='store_true',
                           help='Include vector representations in metadata')
    query_group.add_argument('--output-format',
                           choices=['json', 'rich'],
                           default='rich',
                           help='Output format for results')

def handle_query_operations(args, client):
    """Execute search queries based on command line arguments."""
    try:
        # Set up query manager
        query_manager = QueryManager(client)
        
        # Build query parameters (only those accepted by comprehensive_search)
        search_params = {
            'query_text': args.query,
            'search_type': args.search_type,
            'limit': args.limit,
            'min_score': args.min_score,
            'alpha': args.alpha if args.search_type == 'hybrid' else None,
            'unify_results': args.unify,
        }
        
        # Filter out None values
        search_params = {k: v for k, v in search_params.items() if v is not None}
        
        # Execute query
        result = query_manager.comprehensive_search(**search_params)
        
        # Ensure query info is complete and accurate
        result['query_info'] = {
            'text': args.query,
            'type': args.search_type,
            'parameters': {
                'alpha': args.alpha,
                'min_score': args.min_score,
                'limit': args.limit,
                'unify_results': args.unify,
                'include_vectors': args.include_vectors
            }
        }
        
        # Filter out None values from parameters
        result['query_info']['parameters'] = {
            k: v for k, v in result['query_info']['parameters'].items()
            if v is not None
        }
        
        # Format results
        formatter = ResultFormatter()
        formatted_results = formatter.format_results(result)
        
        # Format and display results
        if args.output_format == 'json':
            print(json.dumps(formatted_results.to_dict(), indent=2))
        else:
            from .query.visualizers.rich import RichVisualizer
            visualizer = RichVisualizer()
            visualizer.show_results(formatted_results)
        
        return True
        
    except Exception as e:
        logging.error(f"Error in query operations: {str(e)}")
        logging.debug("Stack trace:", exc_info=True)
        return False

def setup_argument_parser():
    """Set up the argument parser with organized operation groups."""
    parser = argparse.ArgumentParser(
        description="Weaviate Database Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global options
    parser.add_argument('-v', '--verbose', action='store_true',
                       help="Enable verbose logging")

    # Independent operations
    show_group = parser.add_argument_group('Display Operations')
    show_group.add_argument('--show', 
                           choices=['config', 'models', 'schema', 'stats', 'info'],
                           help='Display information about the system or database')
    show_group.add_argument('--as-diagram', action='store_true',
                           help='Display schema as Mermaid ERD diagram (only with --show schema)')

    cleanup_group = parser.add_argument_group('Database Management')
    cleanup_group.add_argument('--cleanup', action='store_true',
                             help='Remove all collections from database')
    cleanup_group.add_argument('--force', action='store_true',
                             help='Skip confirmation prompts for cleanup')

    # Data-dependent operations
    data_group = parser.add_argument_group('Data Operations')
    data_group.add_argument('--input-dir', type=str, metavar='DIR',
                           help='Directory containing input files to process')
    data_group.add_argument('--verify', action='store_true',
                           help='Verify input files before processing')
    data_group.add_argument('--summarize', action='store_true',
                           help='Show summary of processed data')
    data_group.add_argument('--import', action='store_true', dest='import_',
                           help='Import processed data into database')
    data_group.add_argument('--detail-level',
                           choices=['summary', 'detailed', 'full'],
                           default='summary',
                           help='Level of detail for data summary')

    subset_group = parser.add_argument_group('Subset Operations')
    subset_group.add_argument('--subset-size', type=int,
                             help='Extract subset with specified number of articles')
    subset_group.add_argument('--seed-article', type=str,
                             help='Article ID to use as seed for subset extraction')

    # Add query arguments
    add_query_args(parser)

    return parser

def handle_show_operations(args, client=None) -> bool:
    """Handle show operations for displaying Weaviate database information."""
    try:
        if args.show in ['schema', 'stats', 'info']:
            if not client:
                logging.error("No Weaviate client available for show operations")
                return False

            inspector = DatabaseInspector(client)
            
            if args.show == "schema":
                schema_info = inspector.get_schema_info()
                if not schema_info:
                    logging.info("No schema found in database")
                    return True
                    
                if args.as_diagram:
                    print("\nSchema ERD Diagram (Mermaid format):")
                    print(inspector.generate_mermaid_erd())
                else:
                    inspector.display_schema_summary()
                
            elif args.show == "stats":
                inspector.display_stats_rich()
                
            elif args.show == "info":
                inspector.display_database_info()
                
        elif args.show == "config":
            show_config()
            
        elif args.show == "models":
            show_models()
            
        return True
        
    except Exception as e:
        logging.error(f"Error in show operations: {str(e)}")
        logging.debug("Stack trace:", exc_info=True)
        return False

def main():
    """Main entry point for the CLI."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence noisy loggers
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Parse arguments
    args = setup_argument_parser().parse_args()
    
    # Adjust logging level if verbose flag is set
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        success = True

        # Validate operation combinations
        if args.as_diagram and not args.show == 'schema':
            logging.error("--as-diagram only applies to --show schema")
            return 1
        
        if args.force and not args.cleanup:
            logging.error("--force can only be used with --cleanup")
            return 1
            
        if args.seed_article and not args.subset_size:
            logging.error("--seed-article can only be used with --subset-size")
            return 1
            
        # Validate data operations require input directory
        data_ops = [args.verify, args.summarize, args.import_, args.subset_size]
        if any(op for op in data_ops if op) and not args.input_dir:
            logging.error("--input-dir is required for data operations")
            return 1

        # Handle query operations if query is provided
        if args.query:
            with get_client() as client:
                success = handle_query_operations(args, client)
                return 0 if success else 1

        # Handle database operations that don't require data loading
        if args.cleanup or args.show:
            with get_client() as client:
                # Handle cleanup first if requested
                if args.cleanup:
                    success = cleanup_collections(client, args.force)
                    if not success:
                        logging.error("Cleanup failed")
                        return 1
                    logging.info("Cleanup completed successfully")
                    # If only cleanup was requested, exit
                    if not (args.show or args.input_dir):
                        return 0
                
                # Handle show operations if no data loading is needed
                if args.show and not args.input_dir:
                    success = handle_show_operations(args, client)
                    return 0 if success else 1

        # Handle data operations if input directory is provided
        if args.input_dir:
            # Create data manager and load data
            logging.debug(f"Creating data manager with input directory: {args.input_dir}")
            manager = LiteratureDataManager(args.input_dir)
            
            # Verify input files if requested
            if args.verify:
                logging.debug("Verifying input files")
                valid, errors = manager.verify_input_files()
                if not valid:
                    logging.error("Input file verification failed:")
                    for error in errors:
                        logging.error(f"  {error}")
                    return 1
                logging.debug("Input files verified successfully")
            
            # Load data
            logging.debug("Loading data")
            if not manager.load_data():
                logging.error("Failed to load data")
                return 1

            # Show full dataset statistics if requested
            if args.summarize:
                logging.info("\n" + "="*80)
                logging.info("FULL DATASET STATISTICS")
                logging.info("="*80)
                manager.print_statistics(detail_level=args.detail_level)

            # Handle subset if requested
            subset_manager = None
            if args.subset_size:
                logging.debug(f"Processing subset request for size {args.subset_size}")
                if args.subset_size <= 0:
                    logging.error("Subset size must be positive")
                    return 1

                try:
                    subset_manager, analysis = manager.create_and_analyze_subset(
                        size=args.subset_size,
                        seed_article=args.seed_article
                    )
                    if not subset_manager:
                        logging.error("Failed to create subset")
                        return 1
                    logging.debug("Subset created and analyzed successfully")
                    
                    # Show subset statistics if requested
                    if args.summarize:
                        logging.info("\n" + "="*80)
                        logging.info("SUBSET STATISTICS")
                        logging.info("="*80)
                        subset_manager.print_statistics(detail_level=args.detail_level)
                        
                        # Display collection coverage from analysis
                        logging.info("\nCollection Coverage:")
                        for collection, stats in analysis["collection_coverage"].items():
                            logging.info(f"  {collection}: {stats['size']}/{stats['total']} ({stats['percentage']:.1f}%)")
                        
                        # Display consistency check results
                        if analysis["consistency_check"]["external_references"]:
                            logging.warning("\nExternal References Found:")
                            for ref_type, count in analysis["consistency_check"]["external_references"].items():
                                logging.warning(f"  {ref_type}: {count}")
                        else:
                            logging.info("\nNo external references found - subset is self-contained")
                    
                except ValueError as e:
                    logging.error(f"Error creating subset: {str(e)}")
                    return 1

            # Handle database operations if import is requested
            if args.import_:
                with get_client() as client:
                    # Generate schema if needed
                    logging.info("Checking schema status...")
                    inspector = DatabaseInspector(client)
                    schema_info = inspector.get_schema_info()
                    
                    if not schema_info:
                        logging.info("No schema found, generating new schema...")
                        schema_generator = SchemaGenerator(client)
                        if not schema_generator.create_schema():
                            logging.error("Failed to create schema")
                            return 1
                        logging.info("Schema created successfully")
                    else:
                        logging.info("Schema already exists")

                    # Import data
                    data_manager_to_import = subset_manager if subset_manager else manager
                    importer = WeaviateImporter(data_manager_to_import, client)
                    if not importer.import_data():
                        logging.error("Failed to import data")
                        return 1
                    
                    logging.info("Import completed successfully")

                    # Handle show operations after import if requested
                    if args.show:
                        success = handle_show_operations(args, client)
                        if not success:
                            return 1

        return 0

    except KeyboardInterrupt:
        logging.info("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            logging.debug(traceback.format_exc())
        return 1

def add_examples_to_help(parser):
    """Add usage examples to parser help text."""
    parser.epilog = """
        Examples:
        # Display Information (can be used anytime)
        weaviate-manager --show config
        weaviate-manager --show models
        weaviate-manager --show schema
        weaviate-manager --show schema --as-diagram
        weaviate-manager --show stats
        weaviate-manager --show info
        
        # Database Management (can be used anytime)
        weaviate-manager --cleanup
        weaviate-manager --cleanup --force
        
        # Data Operations (require --input-dir)
        weaviate-manager --input-dir data/processed --verify
        weaviate-manager --input-dir data/processed --summarize --detail-level full
        weaviate-manager --input-dir data/processed --import
        
        # Subset Analysis (requires --input-dir)
        weaviate-manager --input-dir data/processed --subset-size 10
        weaviate-manager --input-dir data/processed --subset-size 10 --seed-article "example.pdf"
        
        # Query Operations
        weaviate-manager --query "machine learning" --search-type hybrid
        weaviate-manager --query "CRISPR" --include-authors --include-entities
        weaviate-manager --query "protein folding" --output-format json
        
        # Combined Operations
        weaviate-manager --input-dir data/processed --verify --summarize --import
        weaviate-manager --query "synthetic biology" --include-references --output-format table
        
        # Enable verbose output (can combine with any operation)
        weaviate-manager -v --show info
        weaviate-manager -v --input-dir data/processed --verify
        weaviate-manager -v --query "gene regulation" --search-type semantic
        """ 

if __name__ == '__main__':
    main()

