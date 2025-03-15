"""
UniProt service implementation.

This service provides a comprehensive interface to the UniProt protein database,
supporting both direct JSON queries and natural language interactions.

Key Features:
1. Query Types:
   - Direct JSON queries with filters
   - Natural language queries with LLM interpretation
   - Single protein lookup by accession/ID
   - Dataset conversion with profiling
   
2. Commands:
   - Natural language: uniprot: [question]
   - Execute query: uniprot.search [query_id]
   - Lookup: uniprot.lookup [accession]
   - Convert to dataset: convert [query_id] to dataset
   - Service info: tell me about uniprot
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import re
import json
from datetime import datetime
import pandas as pd
import traceback
from ydata_profiling import ProfileReport
import asyncio
import requests
from bs4 import BeautifulSoup
import re

from services.base import (
    ChatService,
    ServiceResponse,
    ServiceMessage,
    PreviewIdentifier,
    MessageType
)
from services.llm_service import LLMServiceMixin

from .models import UniProtConfig, QueryResult
from .data_manager import UniProtDataManager
from .query_builder import UniProtQueryBuilder
from .prompts import load_prompt

class UniProtService(ChatService, LLMServiceMixin):
    """Service for querying and analyzing UniProt protein database.
    
    This service provides a comprehensive interface to the UniProt protein database,
    supporting both direct JSON queries and natural language interactions.
    
    Key Features:
    1. Query Types:
       - Direct JSON queries with filters
       - Natural language queries with LLM interpretation
       - Single protein lookup by accession/ID
       - Dataset conversion with profiling
       
    2. Commands:
       - Natural language: uniprot: [question]
       - Execute query: uniprot.search [query_id]
       - Lookup: uniprot.lookup [accession]
       - Convert to dataset: convert [query_id] to dataset
       - Service info: tell me about uniprot
    """
    
    # Regex patterns for extracting uniprot code blocks
    query_block_re = re.compile(r'```uniprot\s*(?:\n|)(.*?)(?:\n|)\s*```', re.DOTALL)
    
    def __init__(self):
        """Initialize UniProt service."""
        # Create default config
        config = UniProtConfig()
        
        ChatService.__init__(self, config.name)
        LLMServiceMixin.__init__(self, config.name)
        
        # Track last original query ID
        self.last_orig_query_id = None
        
        try:
            # Register preview ID prefix (ignore if already registered)
            PreviewIdentifier.register_prefix('uniprot_query')
        except ValueError:
            pass  # Prefix already registered
        
        # Initialize data manager and query builder
        print("\nInitializing UniProt service...")
        self.data_manager = UniProtDataManager(config)
        self.query_builder = UniProtQueryBuilder(self.data_manager)
        print("UniProt service initialization complete.")
        
        # Command patterns
        self.execution_patterns = [
            r'^uniprot\.search\s*$',  # Match bare uniprot.search
            r'^uniprot\.search\s+(?:uniprot_query_)?\d{8}_\d{6}(?:_orig|_alt\d+)\b',
            r'^uniprot\.lookup\s+([A-Z0-9]+)\b',
            r'tell\s+me\s+about\s+uniprot\b',
            r'^convert\s+uniprot_query_\d{8}_\d{6}(?:_orig|_alt\d+)\s+to\s+dataset\b'
        ]
        self.execution_res = [re.compile(p, re.IGNORECASE) for p in self.execution_patterns]
    
    def add_ids_to_blocks(self, text: str) -> str:
        """Add query IDs to UniProt query blocks in text.
        
        This method processes LLM responses to ensure each UniProt query block
        has a proper query ID for tracking and execution.
        
        Args:
            text: Text containing UniProt query blocks
            
        Returns:
            str: Text with query IDs added to blocks
        """
        if not text or '```uniprot' not in text:
            return text
            
        # Track if we've marked a block as primary
        has_primary = False
        # Track alternative numbers used
        alt_numbers = set()
        # Store processed blocks to avoid duplicates
        processed_blocks = set()
        
        def replace_block(match) -> str:
            nonlocal has_primary, alt_numbers
            block = match.group(1).strip()
            
            # Skip if block is empty or just whitespace
            if not block or block.isspace():
                return match.group(0)
            
            # Skip if block already has an ID (using proper regex pattern)
            if re.search(r'--\s*Query ID:\s*uniprot_query_\d{8}_\d{6}(?:_orig|_alt\d+)\b', block):
                # Extract existing alt number if present
                if '_alt' in block:
                    try:
                        alt_num = int(re.search(r'_alt(\d+)', block).group(1))
                        alt_numbers.add(alt_num)
                    except (AttributeError, ValueError):
                        pass
                return match.group(0)
            
            # Skip if block doesn't look like a JSON query
            if not (block.startswith('{') and block.endswith('}')):
                return match.group(0)
            
            try:
                # Validate it's a proper JSON query
                query = json.loads(block)
                if not isinstance(query, dict) or 'query' not in query:
                    return match.group(0)
                
                # Skip if we've already processed this exact block
                block_hash = json.dumps(query, sort_keys=True)
                if block_hash in processed_blocks:
                    return ""  # Remove duplicate blocks
                processed_blocks.add(block_hash)
                
                # Generate new query ID
                query_id = PreviewIdentifier.create_id(prefix="uniprot_query")
                
                # Add suffix based on whether this is primary or alternative
                if not has_primary:
                    query_id = query_id.replace('_orig', '_orig')  # Ensure primary query
                    has_primary = True
                    # Update last_orig_query_id for default search
                    self.last_orig_query_id = query_id
                else:
                    # Find the next available alternative number
                    alt_num = 1
                    while alt_num in alt_numbers:
                        alt_num += 1
                    alt_numbers.add(alt_num)
                    query_id = query_id.replace('_orig', f'_alt{alt_num}')
                
                # Format block with ID
                return f"```uniprot\n{json.dumps(query, indent=2)}\n\n-- Query ID: {query_id}\n```"
            except json.JSONDecodeError:
                return match.group(0)
        
        # Process all blocks
        processed_text = self.query_block_re.sub(replace_block, text)
        
        # Remove any empty lines created by removing duplicates
        return '\n'.join(line for line in processed_text.split('\n') if line.strip())

    def get_llm_response(self, prompt: str) -> str:
        """Get response from LLM.
        
        This method is required by LLMServiceMixin and is used for:
        - Natural language query interpretation
        - Query explanations
        - Result analysis
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        try:
            if 'llm' not in self.context:
                return "LLM service not available"

            # Get data context
            data_context = self._get_data_context()
            query_type = self._detect_query_type(prompt)

            # Prepare system prompt based on context and query type
            system_prompt = f"""You are helping explain a UniProt protein query.

Query validation result: ✓ Valid

Data context:
{json.dumps(data_context, indent=2)}

Please:
1. Explain what this query will do
2. Suggest any improvements or additional fields that might be useful
3. Explain the biological significance of the query

Keep the explanation clear and focused on the biological meaning. Your response should be a clear, well-structured explanation without any code blocks."""

            # Call LLM with proper context
            response = self.context['llm'].complete(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            return response

        except Exception as e:
            return f"Error generating LLM response: {str(e)}"

    def _detect_query_type(self, prompt: str) -> str:
        """Detect the type of query from the prompt."""
        prompt_lower = prompt.lower()
        if "explain" in prompt_lower:
            return "query_explanation"
        elif "analyze" in prompt_lower and "protein" in prompt_lower:
            return "protein_analysis"
        elif "analyze" in prompt_lower and "results" in prompt_lower:
            return "results_analysis"
        else:
            return "natural_language_query"
    
    def can_handle(self, message: str) -> bool:
        """Check if message can be handled by this service."""
        message = message.strip()
        
        # Check for uniprot code blocks
        if self.query_block_re.search(message):
            return True
            
        # Check for execution commands
        for pattern in self.execution_res:
            if pattern.search(message):
                return True
                
        # Check for natural language query
        if message.lower().startswith('uniprot:'):
            return True
            
        return False
    
    def find_recent_query(self, chat_history: list, query_id: str = None) -> tuple[str, str]:
        """Find UniProt query in chat history.
        
        Looks for UniProt blocks in both assistant messages and service messages.
        
        Args:
            chat_history: List of chat messages
            query_id: Optional specific query ID to find
            
        Returns:
            Tuple of (query_text, query_id) if found, (None, None) otherwise
        """
        
        for i, msg in enumerate(reversed(chat_history)):
            
            # Check both assistant messages and uniprot service messages
            if ('```uniprot' in msg.get('content', '').lower() and 
                (msg.get('role') == 'assistant' or 
                 (msg.get('service') == self.name and msg.get('role') == 'system'))):
                content = msg.get('content', '')

                # Extract all UniProt blocks with IDs
                query_blocks = []
                for match in re.finditer(r'```uniprot\s*(.*?)```', content, re.DOTALL):
                    block = match.group(1).strip()
                    
                    # First try to find and extract the query ID
                    id_match = re.search(r'--\s*Query ID:\s*(uniprot_query_\d{8}_\d{6}(?:_orig|_alt\d+))\b', block)
                    if id_match:
                        found_id = id_match.group(1)
                        print(f"Found query ID: {found_id}")
                        # Remove ID comment from query by splitting on -- and taking first part
                        query_parts = block.split('--')
                        query = query_parts[0].strip()
                        
                        try:
                            # Validate it's proper JSON
                            query_json = json.loads(query)
                            if isinstance(query_json, dict) and 'query' in query_json:
                                print(f"Valid JSON query found with ID {found_id}")
                                query_blocks.append((query, found_id))
                        except json.JSONDecodeError as e:
                            print(f"Invalid JSON in block with ID {found_id}: {str(e)}")
                            continue
                    else:
                        print("No query ID found in block")
                
                if query_blocks:
                    if query_id:
                        # Find specific query
                        for query, found_id in query_blocks:
                            if found_id == query_id:
                                print(f"\nFound requested query: {query_id}")
                                return query, found_id
                    else:
                        # Find most recent original query
                        for query, found_id in query_blocks:
                            if found_id.endswith('_orig'):
                                print(f"\nFound most recent original query: {found_id}")
                                return query, found_id
                else:
                    print("No valid query blocks found in message")
        
        print("\n=== No matching query found in chat history ===")
        return None, None

    def parse_request(self, message: str) -> Dict[str, Any]:
        """Parse a message into a request."""
        try:
            # Get chat history from context if available
            chat_history = []
            if hasattr(self, 'context') and 'chat_history' in self.context:
                chat_history = self.context['chat_history']
            
            # Log chat history state for debugging
            
            # Check for direct query syntax
            if message.startswith('uniprot.search'):
                # Extract query ID if provided
                parts = message.split()
                query_id = parts[1] if len(parts) > 1 else None
                
                # Find recent query if no ID provided
                if not query_id:
                    query_text, found_id = self.find_recent_query(chat_history)
                    if found_id:
                        query_id = found_id
                    else:
                        return {
                            'type': 'error',
                            'error': 'No query ID provided and no recent query found'
                        }
                
                return {
                    'type': 'execute_query',
                    'query_id': query_id
                }
            
            # Check for protein lookup syntax
            elif message.startswith('uniprot.lookup'):
                parts = message.split()
                if len(parts) < 2:
                    return {
                        'type': 'error',
                        'error': 'No protein ID provided'
                    }
                return {
                    'type': 'protein_lookup',
                    'protein_id': parts[1]
                }
            
            # Check for dataset conversion syntax
            elif message.startswith('convert ') and ' to dataset' in message:
                # Extract query ID
                match = re.search(r'convert\s+(uniprot_query_\d{8}_\d{6}(?:_orig|_alt\d+))\s+to\s+dataset', message)
                if match:
                    query_id = match.group(1)
                    return {
                        'type': 'convert_dataset',
                        'query_id': query_id
                    }
                else:
                    return {
                        'type': 'error',
                        'error': 'Invalid dataset conversion format. Use: convert [query_id] to dataset'
                    }
            
            # Check for info request
            elif re.search(r'tell\s+me\s+about\s+uniprot', message, re.IGNORECASE):
                return {
                    'type': 'info'
                }
            
            # Default to natural query
            else:
                return {
                    'type': 'natural_query',
                    'query': message
                }
            
        except Exception as e:
            print(f"Error parsing request: {str(e)}")
            return {
                'type': 'error',
                'error': f'Error parsing request: {str(e)}'
            }
    
    def execute(self, request: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the parsed request."""
        # Store context for use in other methods
        self.context = context
        
        request_type = request['type']
        
        try:
            if request_type == 'info':
                response = self._handle_info_request(context)
            elif request_type == 'direct_query':
                response = self._handle_direct_query(request['query'], context)
            elif request_type == 'invalid_query':
                response = self._handle_invalid_query(request['error'], request['raw_text'], context)
            elif request_type == 'natural_query':
                response = self._handle_natural_query(request['query'], context)
            elif request_type == 'execute_query':
                response = self._handle_query_execution(request['query_id'], context)
            elif request_type == 'protein_lookup':
                response = self._handle_protein_lookup(request['protein_id'], context)
            elif request_type == 'convert_dataset':
                response = self._handle_dataset_conversion(request['query_id'], context)
            elif request_type == 'test':
                response = self._run_self_test(context)
            elif request_type == 'error':
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=self._format_service_message(request['message'], MessageType.ERROR),
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            else:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=self._format_service_message(f"Unknown request type: {request_type}", MessageType.ERROR),
                            message_type=MessageType.ERROR
                        )
                    ]
                )

            # Format all messages in the response
            for msg in response.messages:
                msg.content = self._format_service_message(msg.content, msg.message_type)
            
            return response
            
        except Exception as e:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=self._format_service_message(f"Error executing request: {str(e)}", MessageType.ERROR),
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_info_request(self, context: Dict[str, Any]) -> ServiceResponse:
        """Handle 'tell me about uniprot' request.
        
        Generates a comprehensive overview of the UniProt database and service
        using the LLM, including statistics and API capabilities.
        """
        try:
            # Get data context for the prompt
            data_context = self._get_data_context()
            
            # Fetch current UniProt statistics
            uniprot_stats = self._fetch_uniprot_statistics()
            
            # Extract API query capabilities
            api_capabilities = self._extract_api_query_capabilities()
            
            # Create a combined context for the prompt
            prompt_context = {
                'data_context': json.dumps(data_context, indent=2),
                'uniprot_stats': json.dumps(uniprot_stats, indent=2),
                'api_capabilities': json.dumps(api_capabilities, indent=2)
            }
            
            # Generate LLM response using the info prompt
            llm_info = self._call_llm(
                messages=[{'role': 'user', 'content': 'Tell me about the UniProt database and service'}],
                system_prompt=load_prompt('info', prompt_context)
            )
            
            # Format the statistics section
            stats_section = "## UniProt Statistics\n\n"
            stats_section += f"- **Total Entries**: {uniprot_stats['total_entries']}\n"
            stats_section += f"- **Reviewed Entries**: {uniprot_stats['reviewed_entries']}\n"
            stats_section += f"- **Unreviewed Entries**: {uniprot_stats['unreviewed_entries']}\n\n"
            
            stats_section += "### Taxonomic Distribution\n"
            for taxon, count in uniprot_stats['taxonomic_distribution'].items():
                stats_section += f"- **{taxon.capitalize()}**: {count}\n"
            
            # Add model organisms section
            stats_section += "\n### Model Organisms\n"
            for organism, data in uniprot_stats['model_organisms'].items():
                stats_section += f"#### {organism.capitalize()}\n"
                for metric, value in data.items():
                    stats_section += f"- **{metric.replace('_', ' ').capitalize()}**: {value}\n"
                stats_section += "\n"
            
            # Add annotation metrics section if available
            if 'annotation_metrics' in uniprot_stats:
                stats_section += "### Annotation Metrics\n"
                for metric, value in uniprot_stats['annotation_metrics'].items():
                    stats_section += f"- **{metric.replace('_', ' ').capitalize()}**: {value}\n"
                stats_section += "\n"
            
            # Add microbial focus section if available
            if 'microbial_focus' in uniprot_stats:
                stats_section += "### Microbial Focus\n"
                for category, count in uniprot_stats['microbial_focus'].items():
                    stats_section += f"- **{category.replace('_', ' ').capitalize()}**: {count}\n"
                stats_section += "\n"
            
            stats_section += f"*Last Updated: {uniprot_stats['last_update']}*\n"
            
            # Format the query capabilities section
            query_section = "## Query Capabilities\n\n"
            query_section += "### Common Search Fields\n"
            
            # Group fields by category
            field_categories = {
                "Identifiers": ["accession", "id", "gene"],
                "Names and Descriptions": ["protein_name", "gene_exact"],
                "Taxonomy": ["organism_name", "organism_id"],
                "Quality and Curation": ["reviewed", "annotation_score"],
                "Function and Location": ["function", "go", "go_id", "pathway", "subcellular_location"],
                "Structure": ["structure_3d", "domain", "length", "mass"],
                "Disease and Phenotype": ["disease", "disease_id", "variant"],
                "Interactions and Expression": ["interactor", "tissue", "organelle"]
            }
            
            # Add fields by category
            for category, fields in field_categories.items():
                query_section += f"\n#### {category}\n"
                for field in fields:
                    if field in api_capabilities['common_search_fields']:
                        query_section += f"- **{field}**: {api_capabilities['common_search_fields'][field]}\n"
            
            query_section += "\n### Query Syntax Examples\n"
            for name, example in api_capabilities['query_syntax_examples'].items():
                query_section += f"- **{name.capitalize()}**: `{example}`\n"
            
            query_section += "\n### Recommended Field Groups\n"
            for group_name, fields in api_capabilities['recommended_output_fields'].items():
                query_section += f"- **{group_name.capitalize()}**: `{', '.join(fields)}`\n"
            
            # Add a header to the response
            info_text = "# UniProt Service\n\n"
            info_text += llm_info + "\n\n"
            info_text += stats_section + "\n"
            info_text += query_section + "\n"
            info_text += "## Commands\n\n"
            info_text += "- `uniprot: [question]` - Ask a natural language question about proteins\n"
            info_text += "- `uniprot.lookup [accession]` - Look up a specific protein by UniProt accession\n"
            info_text += "- `uniprot.search [query_id]` - Execute a previously created query\n"
            info_text += "- `convert [query_id] to dataset` - Convert query results to a dataset\n\n"
            info_text += "## Examples\n\n"
            info_text += "1. Natural language query:\n"
            info_text += "   ```\n"
            info_text += "   uniprot: Find all human proteins associated with Alzheimer's disease\n"
            info_text += "   ```\n\n"
            info_text += "2. Direct JSON query:\n"
            info_text += "   ```uniprot\n"
            info_text += '   {\n'
            info_text += '     "query": "gene:BRCA1 AND organism:\\"Homo sapiens\\"",\n'
            info_text += '     "fields": ["accession", "id", "gene_names", "protein_name", "organism_name"],\n'
            info_text += '     "size": 10\n'
            info_text += '   }\n'
            info_text += "   ```\n\n"
            info_text += "3. Protein lookup:\n"
            info_text += "   ```\n"
            info_text += "   uniprot.lookup P04637\n"
            info_text += "   ```\n"
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=info_text,
                        message_type=MessageType.INFO
                    )
                ]
            )
        except Exception as e:
            # Fallback to static info if LLM fails
            fallback_info = """
# UniProt Service

The UniProt service provides access to the [UniProt Knowledgebase](https://www.uniprot.org/), a comprehensive resource for protein sequence and functional information.

## Features

- **Search proteins** with specific criteria using UniProt query syntax
- **Natural language queries** to find proteins matching biological descriptions
- **Single protein lookup** to retrieve detailed information about a specific protein
- **Result analysis** with biological context and insights
- **Dataset conversion** for further analysis

## Statistics

- UniProtKB contains millions of protein sequences and functional information
- Swiss-Prot: Manually reviewed and annotated entries with high-quality information
- TrEMBL: Computationally analyzed entries awaiting full manual annotation
- The database covers proteins from all domains of life: Bacteria, Archaea, and Eukaryotes

## Query Capabilities

### Common Search Fields
- **gene**: Gene name (e.g., TP53, BRCA1)
- **protein_name**: Protein name (e.g., 'Tumor protein p53')
- **organism_name**: Species name (e.g., 'Homo sapiens')
- **reviewed**: Review status (true for Swiss-Prot, false for TrEMBL)
- **length**: Sequence length (e.g., [100 TO 200])
- **go**: Gene Ontology terms (e.g., 'transcription')
- **disease**: Disease association (e.g., 'cancer')

### Query Syntax Examples
- **Basic**: `gene:TP53 AND organism:"Homo sapiens"`
- **Complex**: `gene:TP53 AND organism:"Homo sapiens" AND reviewed:true`
- **Function**: `go:"DNA repair" AND reviewed:true`
- **Disease**: `disease:cancer AND reviewed:true`

## Commands

- `uniprot: [question]` - Ask a natural language question about proteins
- `uniprot.lookup [accession]` - Look up a specific protein by UniProt accession
- `uniprot.search [query_id]` - Execute a previously created query
- `convert [query_id] to dataset` - Convert query results to a dataset

## Examples

1. Natural language query:
   ```
   uniprot: Find all human proteins associated with Alzheimer's disease
   ```

2. Direct JSON query:
   ```uniprot
   {
     "query": "gene:BRCA1 AND organism:\"Homo sapiens\"",
     "fields": ["accession", "id", "gene_names", "protein_name", "organism_name"],
     "size": 10
   }
   ```

3. Protein lookup:
   ```
   uniprot.lookup P04637
   ```
"""
            print(f"Error generating info response: {str(e)}")
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=fallback_info,
                        message_type=MessageType.INFO
                    )
                ]
            )
    
    def _handle_direct_query(self, query: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Handle direct JSON query."""
        try:
            # Validate query
            valid, message = self._validate_query(query)
            
            # Get data context for LLM
            prompt_context = {
                'query_details': {
                    'query': query,
                    'validation': (valid, message)
                },
                'data_context': self._get_data_context(),
                'chat_history': self._format_chat_history(context.get('chat_history', []))
            }
            
            if valid:
                try:
                    # Ensure required fields with defaults
                    query_with_defaults = query.copy()
                    if 'format' not in query_with_defaults:
                        query_with_defaults['format'] = 'json'
                    if 'size' not in query_with_defaults:
                        query_with_defaults['size'] = 10
                    if 'fields' not in query_with_defaults:
                        query_with_defaults['fields'] = ['accession', 'id', 'protein_name', 'gene_names', 'organism_name']
                    
                    # Generate query ID for future reference
                    query_id = PreviewIdentifier.create_id('uniprot_query')
                    
                    # Prepare query details for store update
                    query_details = {
                        'query': query_with_defaults,
                        'creation_time': datetime.now().isoformat()
                    }
                    
                    # Create query preview message
                    query_preview = f"""
### Query 1

```uniprot
{json.dumps(query_with_defaults, indent=2)}

-- Query ID: {query_id}
```

To execute this query:
```
uniprot.search {query_id}
```"""
                    
                    # Add query block to chat history
                    if 'chat_history' in context:
                        context['chat_history'].append({
                            'role': 'assistant',
                            'content': f"```uniprot\n{json.dumps(query_with_defaults, indent=2)}\n\n-- Query ID: {query_id}\n```"
                        })
                    
                    messages = [
                        ServiceMessage(
                            service=self.name,
                            content=query_preview,
                            message_type=MessageType.SUGGESTION
                        )
                    ]
                    
                    # Get LLM interpretation if available
                    try:
                        if 'llm' in context:
                            system_prompt = f"""You are helping explain a UniProt protein query.

Query validation result: ✓ Valid

Data context:
{prompt_context['data_context']}

Please:
1. Explain what this query will do
2. Suggest any improvements or additional fields that might be useful
3. Explain the biological significance of the query

Keep the explanation clear and focused on the biological meaning. Your response should be a clear, well-structured explanation without any code blocks."""
                            
                            interpretation = self._call_llm(
                                messages=[{'role': 'user', 'content': f'Please explain this query:\n```json\n{json.dumps(query_with_defaults, indent=2)}\n```'}],
                                system_prompt=system_prompt
                            )
                            
                            if interpretation:
                                messages.append(
                                    ServiceMessage(
                                        service=self.name,
                                        content=interpretation,
                                        message_type=MessageType.SUMMARY
                                    )
                                )
                    except Exception as e:
                        # Log error but continue with query processing
                        print(f"Error getting LLM interpretation: {str(e)}")
                    
                    # Prepare store updates
                    store_updates = {
                        'successful_queries_store': {
                            query_id: query_details
                        }
                    }
                    
                    return ServiceResponse(
                        messages=messages,
                        store_updates=store_updates
                    )
                    
                except Exception as e:
                    return ServiceResponse(
                        messages=[
                            ServiceMessage(
                                service=self.name,
                                content=f"Error preparing query: {str(e)}",
                                message_type=MessageType.ERROR
                            )
                        ]
                    )
                
            else:
                # For invalid queries, get LLM help with correction
                system_prompt = """You are helping validate and fix a UniProt protein query.

Query validation result: ✗ Invalid
Validation message: {0}

Original query:
{1}

Data context:
{2}

Please:
1. Explain what is wrong with the query
2. Provide a corrected version that:
   - Uses proper JSON format
   - Includes all relevant fields from the original query
   - Uses available fields only
   - Is wrapped in a ```uniprot``` code block
3. Explain the biological significance

Your response MUST include a complete JSON query with:
- 'query' field using proper syntax
- 'fields' array with relevant fields
- 'size' parameter
- All wrapped in proper JSON format

Example format:
```uniprot
{{
  "query": "keywords:\\"glycolysis\\" AND organism:\\"Homo sapiens\\" AND reviewed:true",
  "fields": ["accession", "gene_names", "protein_name", "cc_function"],
  "size": 20
}}
```""".format(message, json.dumps(query, indent=2), json.dumps(prompt_context['data_context'], indent=2))

                # Get LLM help with correction if available
                try:
                    interpretation = self._call_llm(
                        messages=[{'role': 'user', 'content': 'Please help fix this query'}],
                        system_prompt=system_prompt
                    )
                    
                    # Add query IDs to any uniprot blocks
                    interpretation = self.add_ids_to_blocks(interpretation)
                    
                    return ServiceResponse(
                        messages=[
                            ServiceMessage(
                                service=self.name,
                                content=interpretation,
                                message_type=MessageType.ERROR
                            )
                        ]
                    )
                except Exception as e:
                    # Without LLM, return basic error format
                    error_message = f"""
## Invalid Query Format

There was an error in your UniProt query:

```
{message}
```

**Your query:**
```json
{json.dumps(query, indent=2)}
```

### Example of correct format:

```uniprot
{
  "query": "keywords:\\"glycolysis\\" AND organism:\\"Homo sapiens\\"",
  "fields": ["accession", "id", "gene_names", "protein_name"],
  "size": 10
}

-- Query ID: uniprot_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_orig
```

Please correct the format and try again, or use a natural language query with `uniprot: your question`.
"""
                    return ServiceResponse(
                        messages=[
                            ServiceMessage(
                                service=self.name,
                                content=error_message,
                                message_type=MessageType.ERROR
                            )
                        ]
                    )
                
        except Exception as e:
            error_message = f"Error handling query: {str(e)}\n\nStack trace:\n```\n{traceback.format_exc()}\n```"
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=error_message,
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_invalid_query(self, error: str, raw_text: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle invalid JSON query."""
        try:
            # Get data context for LLM
            prompt_context = {
                'query_details': {
                    'raw_text': raw_text,
                    'error': error
                },
                'data_context': self._get_data_context()
            }
            
            # Generate system prompt for query correction
            system_prompt = """You are helping validate and fix a UniProt protein query.

Query validation error: {0}

Original query:
{1}

Data context:
{2}

Please:
1. Explain what is wrong with the query
2. Provide a corrected version that:
   - Uses proper JSON format
   - Includes all relevant fields from the original query
   - Uses available fields only
   - Is wrapped in a ```uniprot``` code block
3. Explain the biological significance

Your response MUST include a complete JSON query with:
- 'query' field using proper syntax
- 'fields' array with relevant fields
- 'size' parameter
- All wrapped in proper JSON format

Example format:
```uniprot
{{
  "query": "keywords:\\"glycolysis\\" AND organism:\\"Homo sapiens\\" AND reviewed:true",
  "fields": ["accession", "gene_names", "protein_name", "cc_function"],
  "size": 20
}}
```""".format(error, raw_text, json.dumps(prompt_context['data_context'], indent=2))

            # Get LLM help with correction if available
            try:
                interpretation = self._call_llm(
                    messages=[{'role': 'user', 'content': 'Please help fix this query'}],
                    system_prompt=system_prompt
                )
                
                # Add query IDs to any uniprot blocks
                interpretation = self.add_ids_to_blocks(interpretation)
                
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=interpretation,
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            except Exception as e:
                # Without LLM, return basic error format
                error_message = f"""
## Invalid Query Format

There was an error in your UniProt query:

```
{error}
```

**Your query:**
```
{raw_text}
```

### Example of correct format:

```uniprot
{
  "query": "keywords:\\"glycolysis\\" AND organism:\\"Homo sapiens\\"",
  "fields": ["accession", "id", "gene_names", "protein_name"],
  "size": 10
}

-- Query ID: uniprot_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_orig
```

Please correct the format and try again, or use a natural language query with `uniprot: your question`.
"""
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=error_message,
                            message_type=MessageType.ERROR
                        )
                    ]
                )
                
        except Exception as e:
            error_message = f"Error handling invalid query: {str(e)}"
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=error_message,
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_natural_query(self, query: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle natural language query by generating query suggestions."""
        try:
            # Get chat history from context
            chat_history = context.get('chat_history', [])
            
            # Get LLM suggestions
            system_prompt = load_prompt('natural', {
                'data_context': self._get_data_context(),
                'user_request': query,
                'chat_history': self._format_chat_history(chat_history)
            })
            
            suggestions = self._call_llm(
                messages=[{'role': 'user', 'content': query}],
                system_prompt=system_prompt
            )
            
            # Process the suggestions to add query IDs to blocks
            modified_suggestions = self.add_ids_to_blocks(suggestions)
            
            # Extract queries from the modified suggestions to verify we have valid queries
            queries = []
            for match in self.query_block_re.finditer(modified_suggestions):
                try:
                    block = match.group(1).strip()
                    
                    if not block or block.isspace():
                        continue
                    
                    # Extract query ID if present
                    query_id_match = re.search(r'--\s*Query ID:\s*(uniprot_query_\d{8}_\d{6}(?:_orig|_alt\d+))', block)
                    if query_id_match:
                        query_id = query_id_match.group(1)
                        # Update last_orig_query_id if this is an original query
                        if '_orig' in query_id:
                            self.last_orig_query_id = query_id
                    
                    # Extract just the JSON part
                    json_part = re.match(r'(\{.*\})', block, re.DOTALL)
                    if json_part:
                        json_str = json_part.group(1)
                        query_dict = json.loads(json_str)
                        if isinstance(query_dict, dict) and 'query' in query_dict:
                            queries.append(query_dict)
                except Exception as e:
                    print(f"Error processing query block: {str(e)}")
                    continue
            
            if not queries:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content="Could not generate valid UniProt queries from your request. Please try rephrasing.",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Return only the query suggestions
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=modified_suggestions,
                        message_type=MessageType.SUGGESTION
                    )
                ]
            )
            
        except Exception as e:
            traceback.print_exc()
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error processing natural language query: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _generate_query_description(self, query: Dict[str, Any]) -> str:
        """Generate a human-readable description of a query."""
        try:
            query_str = query['query']
            fields = query.get('fields', [])
            size = query.get('size', 10)
            
            # Basic description based on query string
            description = f"Search for proteins matching: {query_str}"
            
            # Add field info if specified
            if fields:
                description += f"\nReturning fields: {', '.join(fields)}"
            
            # Add size limit
            description += f"\nLimit: {size} results"
            
            return description
        except Exception as e:
            return f"Query description unavailable: {str(e)}"
    
    def _handle_query_execution(self, query_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Execute a stored query."""
        try:
            # Ensure LLM is available for analysis generation
            if 'llm' not in context:
                context['llm'] = True
            
            # Get the central stores from context
            successful_queries_store = context.get('successful_queries_store', {})
            
            # First try to find the query in chat history
            structured_query = self._find_query_by_id(query_id, context)
            
            # If not found in chat history, check if it exists in the central store
            if not structured_query and query_id in successful_queries_store:
                query_details = successful_queries_store.get(query_id, {})
                structured_query = query_details.get('query')
            
            if not structured_query:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Query ID `{query_id}` not found in chat history. Please create a query first.",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Get description from store if available
            description = query_details.get('description') if query_id in successful_queries_store else None
            
            # Generate description if not available
            if not description:
                description = self._generate_query_description(structured_query)
            
            # Execute the query
            if 'size' not in structured_query:
                structured_query['size'] = 10
            if 'fields' not in structured_query:
                structured_query['fields'] = ['accession', 'id', 'protein_name', 'gene_names', 'organism_name']
            if 'format' not in structured_query:
                structured_query['format'] = 'json'
            
            # Execute the query
            query_string = structured_query['query']
            fields = structured_query['fields']
            format = structured_query['format']
            size = structured_query['size']
            
            try:
                # Get results
                results_df = self.data_manager.search_proteins(
                    query=query_string,
                    fields=fields,
                    format=format,
                    size=size
                )
                
                # Create result object
                query_result = QueryResult(
                    dataframe=results_df,
                    metadata=structured_query,
                    description=description
                )
                
                # Prepare query details for store update
                query_details = {
                    'query': structured_query,
                    'result': query_result.to_preview(max_rows=min(5, len(results_df))),
                    'execution_time': datetime.now().isoformat()
                }
                
                # Generate result message
                result_count = len(results_df)
                result_message = f"""## Query Results: {result_count} Proteins Found

**Query:** {description}  
**Query ID:** `{query_id}`

{self._format_results_preview(results_df, max_rows=5)}

To convert these results to a dataset for further analysis:
```
convert {query_id} to dataset
```"""
                
                messages = [
                    ServiceMessage(
                        service=self.name,
                        content=result_message,
                        message_type=MessageType.RESULT
                    )
                ]
                
                # Generate analysis if there are results and LLM is available
                if result_count > 0 and 'llm' in context:
                    analysis = self._generate_results_analysis(query_result, context)
                    if analysis:
                        messages.append(
                            ServiceMessage(
                                service=self.name,
                                content=analysis,
                                message_type=MessageType.SUMMARY
                            )
                        )

                # Prepare store updates
                store_updates = {
                    'successful_queries_store': {
                        query_id: query_details
                    }
                }
                
                return ServiceResponse(
                    messages=messages,
                    store_updates=store_updates
                )
                
            except Exception as e:
                error_msg = str(e)
                if "400 Client Error" in error_msg:
                    # Extract the actual error message from UniProt API response
                    import re
                    api_error = re.search(r'{"url":.*?"messages":\[(.*?)\]}', error_msg)
                    if api_error:
                        error_msg = f"UniProt API Error: {api_error.group(1)}"
                    error_msg += "\n\nPlease check your query syntax and try again."
                
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=error_msg,
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
        except Exception as e:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error executing query: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_protein_lookup(self, protein_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Handle protein lookup by ID/accession."""
        try:
            # Get protein data
            protein_data = self.data_manager.get_protein_by_id(protein_id)
            
            # Debug: Print the entry type
            entry_type = protein_data.get("entryType", "Unknown")
            
            # Extract key information for display
            accession = protein_data.get("primaryAccession", protein_id)
            protein_name = "Unknown"
            if "proteinDescription" in protein_data:
                protein_desc = protein_data["proteinDescription"]
                if "recommendedName" in protein_desc:
                    protein_name = protein_desc["recommendedName"].get("fullName", {}).get("value", "Unknown")
                elif "submittedName" in protein_desc and len(protein_desc["submittedName"]) > 0:
                    protein_name = protein_desc["submittedName"][0].get("fullName", {}).get("value", "Unknown")
            
            organism = protein_data.get("organism", {}).get("scientificName", "Unknown")
            gene_names = []
            for gene in protein_data.get("genes", []):
                if "geneName" in gene:
                    gene_names.append(gene["geneName"].get("value", ""))
            gene_names_str = ", ".join(gene_names) if gene_names else "Unknown"
            
            # Format sequence information
            sequence = protein_data.get("sequence", {})
            sequence_length = sequence.get("length", 0)
            
            # Check if the entry is reviewed (SwissProt)
            is_reviewed = "reviewed" in entry_type.lower() or "swiss-prot" in entry_type.lower()
            
            # Generate formatted message
            protein_info = f"""
## Protein: {protein_name}

**Accession:** [{accession}](https://www.uniprot.org/uniprotkb/{accession})  
**Organism:** {organism}  
**Gene Names:** {gene_names_str}  
**Sequence Length:** {sequence_length} amino acids  
**Entry Type:** {"Reviewed (SwissProt)" if is_reviewed else "Unreviewed (TrEMBL)"}  

### Description
{self._extract_protein_function(protein_data)}

### Features
{self._extract_protein_features(protein_data)}

### Database References
{self._extract_protein_references(protein_data)}

[View full entry on UniProt](https://www.uniprot.org/uniprotkb/{accession})
"""
            
            # Generate analysis with LLM
            analysis = self._generate_protein_analysis(protein_data, accession, context)
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=protein_info,
                        message_type=MessageType.RESULT
                    ),
                    ServiceMessage(
                        service=self.name,
                        content=analysis,
                        message_type=MessageType.SUMMARY
                    )
                ]
            )
        except Exception as e:
            error_message = f"Error retrieving protein {protein_id}: {str(e)}"
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=error_message,
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _handle_dataset_conversion(self, query_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Convert query results to a dataset."""
        # Get the central stores from context
        successful_queries_store = context.get('successful_queries_store', {})
        
        # Check if query exists in the central store
        if query_id not in successful_queries_store:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Query ID `{query_id}` not found in the query store. Please create and execute a query first.",
                        message_type=MessageType.ERROR
                    )
                ]
            )
        
        # Check if query has been executed
        query_details = successful_queries_store[query_id]
        if 'result' not in query_details:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Query `{query_id}` has not been executed yet. Please execute it first with `uniprot.search {query_id}`",
                        message_type=MessageType.ERROR
                    )
                ]
            )
        
        try:
            # Re-execute query to get fresh results (in case cache expired)
            structured_query = query_details['query']
            description = query_details.get('description', 'UniProt Query')
            
            # Execute the query
            query_string = structured_query['query']
            fields = structured_query['fields']
            format = structured_query['format']
            size = structured_query['size']
            
            # Get results
            results_df = self.data_manager.search_proteins(
                query=query_string,
                fields=fields,
                format=format,
                size=size
            )
            
            # Create dataset ID
            dataset_id = f"uniprot_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate dataset metadata
            metadata = {
                'source': 'UniProt',
                'query_id': query_id,
                'creation_time': self._get_creation_timestamp(),
                'description': description,
                'rows': len(results_df),
                'columns': results_df.columns.tolist(),
                'query': structured_query,
                'selectable': True,  # Enable dataset selection
                'transformations': []  # Track any transformations
            }
            
            # Validate metadata
            valid, error = self._validate_dataset_metadata(metadata)
            if not valid:
                return ServiceResponse(
                    messages=[
                        ServiceMessage(
                            service=self.name,
                            content=f"Error creating dataset: {error}",
                            message_type=MessageType.ERROR
                        )
                    ]
                )
            
            # Generate profile report if requested
            profile_html = None
            try:
                # Generate minimal profile report
                profile = ProfileReport(
                    results_df,
                    title=f"UniProt Dataset Profile: {description}",
                    minimal=True
                )
                profile_html = profile.to_html()
            except Exception as e:
                print(f"Error generating profile report: {str(e)}")
            
            # Create dataset preview
            preview = f"""## Dataset Created: {len(results_df)} Proteins

**Dataset ID:** `{dataset_id}`  
**Source:** UniProt Query  
**Description:** {description}

### Data Preview:
```
{results_df.head(5).to_string()}
```

### Column Information:
- Total Columns: {len(results_df.columns)}
- Columns: {', '.join(results_df.columns.tolist())}

You can now use this dataset for further analysis.
"""
            
            # Convert DataFrame to serializable format
            df_dict = results_df.to_dict('records')
            
            # Prepare dataset for storage
            dataset_info = {
                'df': df_dict,  # Store as dict instead of raw DataFrame
                'metadata': metadata,
                'profile_report': profile_html
            }
            
            # Prepare store updates
            store_updates = {
                'datasets_store': {
                    dataset_id: dataset_info
                }
            }
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=preview,
                        message_type=MessageType.RESULT
                    )
                ],
                store_updates=store_updates
            )
            
        except Exception as e:
            error_message = f"Error converting to dataset: {str(e)}"
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=error_message,
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _format_results_preview(self, df: pd.DataFrame, max_rows: int = 5) -> str:
        """Format DataFrame preview as markdown table."""
        if len(df) == 0:
            return "No results found."
            
        # Limit to max_rows
        preview_df = df.head(max_rows)
        
        # Convert to markdown table
        markdown_table = preview_df.to_markdown(index=False)
        
        # Add note if there are more rows
        if len(df) > max_rows:
            markdown_table += f"\n\n*Showing {max_rows} of {len(df)} results*"
            
        return markdown_table
    
    def _generate_query_explanation(self, query: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Generate explanation of a query using LLM."""
        try:
            # Generate prompt directly
            prompt = f"""Please explain this UniProt query in biological terms:

Query: {query['query']}

Explain:
1. What types of proteins this query will find
2. Any specific conditions or filters being applied
3. The biological significance of these proteins

Keep the explanation clear and concise, focusing on the biological meaning rather than the technical syntax."""
            
            # Get LLM response
            response = self.get_llm_response(prompt)
            
            return f"### Query Explanation\n\n{response}"
        except Exception as e:
            return f"Error generating query explanation: {str(e)}"
    
    def _create_compact_protein_data(self, protein_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact representation of protein data for LLM analysis.
        
        Extracts only the essential information focused on gene function and organism information
        to prevent context window errors when sending to the LLM.
        
        Args:
            protein_data: Full protein data from UniProt API
            
        Returns:
            Dict containing only the essential information for analysis
        """
        compact_data = {
            'basic_info': {
                'accession': protein_data.get('primaryAccession', ''),
                'entry_name': protein_data.get('uniProtkbId', ''),
                'protein_name': '',
                'gene_names': [],
                'organism': protein_data.get('organism', {}).get('scientificName', ''),
                'taxonomy_id': protein_data.get('organism', {}).get('taxonId', ''),
                'reviewed': "reviewed" in protein_data.get('entryType', '').lower() or "swiss-prot" in protein_data.get('entryType', '').lower()
            },
            'function': {},
            'features': [],
            'go_terms': [],
            'disease_associations': [],
            'sequence_info': {
                'length': protein_data.get('sequence', {}).get('length', 0),
                'mass': protein_data.get('sequence', {}).get('mass', 0)
            }
        }
        
        # Extract protein name
        if 'proteinDescription' in protein_data:
            protein_desc = protein_data['proteinDescription']
            if 'recommendedName' in protein_desc:
                compact_data['basic_info']['protein_name'] = protein_desc['recommendedName'].get('fullName', {}).get('value', '')
            elif 'submittedName' in protein_desc and len(protein_desc['submittedName']) > 0:
                compact_data['basic_info']['protein_name'] = protein_desc['submittedName'][0].get('fullName', {}).get('value', '')
        
        # Extract gene names
        for gene in protein_data.get('genes', []):
            if 'geneName' in gene:
                compact_data['basic_info']['gene_names'].append(gene['geneName'].get('value', ''))
        
        # Extract function information
        if 'comments' in protein_data:
            function_comments = [c for c in protein_data['comments'] if c.get('commentType') == 'FUNCTION']
            if function_comments:
                compact_data['function'] = {
                    'description': function_comments[0].get('texts', [{}])[0].get('value', '')
                }
        
        # Extract key features (limit to 10 most important)
        if 'features' in protein_data:
            important_feature_types = [
                'ACTIVE_SITE', 'BINDING_SITE', 'DOMAIN', 'MOTIF', 'REGION', 
                'SITE', 'DISEASE', 'VARIANT', 'MUTAGENESIS'
            ]
            
            important_features = [
                f for f in protein_data.get('features', [])
                if f.get('type') in important_feature_types
            ][:10]  # Limit to 10 features
            
            for feature in important_features:
                compact_feature = {
                    'type': feature.get('type', ''),
                    'description': feature.get('description', ''),
                    'location': f"{feature.get('location', {}).get('start', {}).get('value', '')}-{feature.get('location', {}).get('end', {}).get('value', '')}"
                }
                compact_data['features'].append(compact_feature)
        
        # Extract GO terms (limit to 10)
        if 'uniProtKBCrossReferences' in protein_data:
            go_refs = [
                ref for ref in protein_data.get('uniProtKBCrossReferences', [])
                if ref.get('database') == 'GO'
            ][:10]  # Limit to 10 GO terms
            
            for go_ref in go_refs:
                go_term = {
                    'id': go_ref.get('id', ''),
                    'term': ''
                }
                
                for property in go_ref.get('properties', []):
                    if property.get('key') == 'GoTerm':
                        go_term['term'] = property.get('value', '')
                        break
                
                compact_data['go_terms'].append(go_term)
        
        # Extract disease associations
        if 'comments' in protein_data:
            disease_comments = [c for c in protein_data['comments'] if c.get('commentType') == 'DISEASE']
            
            for comment in disease_comments:
                for disease in comment.get('diseases', []):
                    disease_info = {
                        'name': disease.get('diseaseId', ''),
                        'description': disease.get('description', '')
                    }
                    compact_data['disease_associations'].append(disease_info)
        
        return compact_data

    def _generate_protein_analysis(self, protein_data: Dict[str, Any], protein_id: str, context: Dict[str, Any]) -> str:
        """Generate LLM-based analysis of a protein."""
        try:
            # Extract key protein information
            accession = protein_data.get('primaryAccession', protein_id)
            protein_name = protein_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown protein')
            organism = protein_data.get('organism', {}).get('scientificName', 'Unknown organism')
            
            # Extract text sections
            function_text = self._extract_protein_function(protein_data)
            features_text = self._extract_protein_features(protein_data)
            references_text = self._extract_protein_references(protein_data)
            
            # Create compact representation of protein data
            compact_data = self._create_compact_protein_data(protein_data)
            
            # Format protein data for the prompt
            formatted_data = json.dumps(compact_data, indent=2)
            
            # Prepare prompt context with all required fields
            prompt_context = {
                'protein_id': accession,
                'protein_data': formatted_data,
                'accession': accession,
                'protein_name': protein_name,
                'organism_name': organism,
                'organism': organism,
                'function_text': function_text,
                'features_text': features_text,
                'references_text': references_text
            }
            
            # Merge with passed context while giving priority to our specific values
            merged_context = {**context, **prompt_context}
            
            return self._call_llm(
                messages=[{'role': 'user', 'content': 'Analyze this protein'}],
                system_prompt=load_prompt('protein_analysis', merged_context)
            )
        except Exception as e:
            return f"Error analyzing protein: {str(e)}"
    
    def _generate_results_analysis(self, query_result: QueryResult, context: Dict[str, Any]) -> Optional[str]:
        """Generate analysis of query results using LLM."""
        try:
            if 'llm' not in context:
                return None
                
            # Get the full dataframe
            df = query_result.dataframe
            
            # Extract key information for summarization
            summary_data = {
                "query_description": query_result.description,
                "total_proteins": len(df),
                "data_sample": df.head(5).to_markdown(index=False)
            }
            
            # Add a note about sample data
            summary_data["sample_note"] = f"Note: The above is a sample of {len(df)} total results. The full dataset statistics are provided below."
            
            # Extract unique organisms with comprehensive statistics
            organisms_summary = ""
            if 'organism_name' in df.columns:
                organisms = df['organism_name'].dropna().unique().tolist()
                if organisms:
                    # Count occurrences of each organism
                    organism_counts = df['organism_name'].value_counts().to_dict()
                    # Sort organisms by frequency
                    top_organisms = sorted(organism_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Format the summary with counts
                    organisms_summary = f"Organisms ({len(organisms)} unique):\n"
                    organisms_summary += "\nTop organisms by frequency:\n"
                    for org, count in top_organisms:
                        percentage = (count / len(df)) * 100
                        organisms_summary += f"- {org}: {count} proteins ({percentage:.1f}%)\n"
                    
                    # Add taxonomic diversity if available
                    if 'taxonomy_id' in df.columns:
                        unique_taxa = df['taxonomy_id'].dropna().nunique()
                        organisms_summary += f"\nTaxonomic diversity: {unique_taxa} unique taxonomy IDs"
                
                summary_data["organisms_summary"] = organisms_summary
            else:
                summary_data["organisms_summary"] = ""
            
            # Extract unique protein names with functional categorization if possible
            protein_names_summary = ""
            if 'protein_name' in df.columns:
                protein_names = df['protein_name'].dropna().unique().tolist()
                if protein_names:
                    # Count occurrences of each protein name
                    protein_counts = df['protein_name'].value_counts().to_dict()
                    # Sort proteins by frequency
                    top_proteins = sorted(protein_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Format the summary with counts
                    protein_names_summary = f"Protein Names ({len(protein_names)} unique):\n"
                    protein_names_summary += "\nTop proteins by frequency:\n"
                    for prot, count in top_proteins:
                        percentage = (count / len(df)) * 100
                        protein_names_summary += f"- {prot}: {count} entries ({percentage:.1f}%)\n"
                    
                    # Try to identify protein families by common keywords
                    common_keywords = []
                    for name in protein_names:
                        if isinstance(name, str):
                            words = re.findall(r'\b[A-Za-z][A-Za-z0-9-]+\b', name.lower())
                            common_keywords.extend(words)
                    
                    # Count keyword frequencies
                    from collections import Counter
                    keyword_counter = Counter(common_keywords)
                    
                    # Filter out common words that aren't likely to be protein family indicators
                    common_words = {'protein', 'the', 'and', 'with', 'for', 'from', 'that', 'this', 'not', 'are', 'has'}
                    keyword_counter = Counter({k: v for k, v in keyword_counter.items() if k not in common_words and v > 1})
                    
                    if keyword_counter:
                        top_keywords = keyword_counter.most_common(10)
                        protein_names_summary += "\nCommon protein name keywords:\n"
                        for keyword, count in top_keywords:
                            protein_names_summary += f"- {keyword}: appears in {count} protein names\n"
                
                summary_data["protein_names_summary"] = protein_names_summary
            else:
                summary_data["protein_names_summary"] = ""
            
            # Extract unique gene names with comprehensive statistics
            gene_names_summary = ""
            if 'gene_names' in df.columns:
                # Gene names might be in various formats, so we need to handle them carefully
                all_genes = []
                gene_to_proteins = {}  # Map genes to their protein counts
                
                for idx, genes in enumerate(df['gene_names'].dropna()):
                    if isinstance(genes, str):
                        # Split by spaces or commas if it's a string
                        gene_list = [g.strip() for g in re.split(r'[,\s]+', genes) if g.strip()]
                        all_genes.extend(gene_list)
                        
                        # Track which proteins have this gene
                        for gene in gene_list:
                            if gene not in gene_to_proteins:
                                gene_to_proteins[gene] = 0
                            gene_to_proteins[gene] += 1
                            
                    elif isinstance(genes, list):
                        all_genes.extend(genes)
                        
                        # Track which proteins have this gene
                        for gene in genes:
                            if gene not in gene_to_proteins:
                                gene_to_proteins[gene] = 0
                            gene_to_proteins[gene] += 1
                
                unique_genes = list(set(all_genes))
                if unique_genes:
                    # Sort genes by frequency
                    top_genes = sorted(gene_to_proteins.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    # Format the summary with counts
                    gene_names_summary = f"Gene Names ({len(unique_genes)} unique):\n"
                    gene_names_summary += "\nTop genes by frequency:\n"
                    for gene, count in top_genes:
                        percentage = (count / len(df)) * 100
                        gene_names_summary += f"- {gene}: {count} proteins ({percentage:.1f}%)\n"
                    
                    # Try to identify gene families by common prefixes
                    gene_prefixes = {}
                    for gene in unique_genes:
                        if isinstance(gene, str) and len(gene) >= 3:
                            prefix = gene[:3]
                            if prefix not in gene_prefixes:
                                gene_prefixes[prefix] = 0
                            gene_prefixes[prefix] += 1
                    
                    # Filter to prefixes that appear multiple times
                    common_prefixes = {k: v for k, v in gene_prefixes.items() if v > 2 and k.isalpha()}
                    if common_prefixes:
                        top_prefixes = sorted(common_prefixes.items(), key=lambda x: x[1], reverse=True)[:5]
                        gene_names_summary += "\nCommon gene name prefixes (potential gene families):\n"
                        for prefix, count in top_prefixes:
                            gene_names_summary += f"- {prefix}*: {count} genes\n"
                
                summary_data["gene_names_summary"] = gene_names_summary
            else:
                summary_data["gene_names_summary"] = ""
            
            # Add additional dataset statistics if available
            additional_stats = []
            
            # Check for protein length statistics
            if 'length' in df.columns:
                length_stats = {
                    'min': df['length'].min(),
                    'max': df['length'].max(),
                    'mean': df['length'].mean(),
                    'median': df['length'].median()
                }
                additional_stats.append(f"Protein Length Statistics:\n- Min: {length_stats['min']}\n- Max: {length_stats['max']}\n- Mean: {length_stats['mean']:.1f}\n- Median: {length_stats['median']}")
            
            # Check for review status distribution
            if 'reviewed' in df.columns:
                reviewed_count = df['reviewed'].sum() if df['reviewed'].dtype == bool else df['reviewed'].str.lower().isin(['true', 'yes', '1']).sum()
                unreviewed_count = len(df) - reviewed_count
                additional_stats.append(f"Review Status:\n- Reviewed (SwissProt): {reviewed_count} proteins ({(reviewed_count/len(df))*100:.1f}%)\n- Unreviewed (TrEMBL): {unreviewed_count} proteins ({(unreviewed_count/len(df))*100:.1f}%)")
            
            if additional_stats:
                summary_data["additional_stats"] = "\n\n".join(additional_stats)
            else:
                summary_data["additional_stats"] = ""
            
            # Load the results summary prompt
            prompt = load_prompt('results_summary', summary_data)
            
            # Use standard LLM framework with a specific system prompt
            system_prompt = """You are a protein biology expert analyzing UniProt query results.
Your task is to provide a comprehensive biological summary of the proteins found in a search.
Focus on identifying patterns in:
- Organism diversity and evolutionary relationships
- Protein families and their functions
- Biological pathways represented
- Functional diversity
- Research implications

Be specific about pathways, protein functions, and biological processes when possible.
Avoid generic statements and focus on the biological significance of the results.
IMPORTANT: Your analysis should be based on the FULL dataset statistics provided, not just the sample data shown."""
            
            response = self._call_llm(
                messages=[{'role': 'user', 'content': prompt}],
                system_prompt=system_prompt
            )
            
            # Extract content from the response object
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
            else:
                # Fallback to string conversion if response doesn't have expected structure
                content = str(response)
            
            return f"### Biological Summary\n\n{content}"
            
        except Exception as e:
            print(f"Error generating results analysis: {str(e)}")
            traceback.print_exc()
            return None  # Return None instead of error message to avoid showing errors to user
    
    def _extract_protein_function(self, protein_data: Dict[str, Any]) -> str:
        """Extract function description from protein data."""
        comments = protein_data.get("comments", [])
        function_comments = [c for c in comments if c.get("commentType") == "FUNCTION"]
        
        if not function_comments:
            return "*Function information not available*"
            
        function_texts = []
        for comment in function_comments:
            if "texts" in comment:
                for text in comment["texts"]:
                    if "value" in text:
                        function_texts.append(text["value"])
                        
        return "\n\n".join(function_texts) if function_texts else "*Function information not available*"
    
    def _extract_protein_features(self, protein_data: Dict[str, Any]) -> str:
        """Extract key features from protein data."""
        features = protein_data.get("features", [])
        
        if not features:
            return "*Feature information not available*"
            
        # Group features by type
        feature_types = {}
        for feature in features:
            feature_type = feature.get("type", "Unknown")
            if feature_type not in feature_types:
                feature_types[feature_type] = []
            feature_types[feature_type].append(feature)
            
        # Format feature summary
        summary_parts = []
        for feature_type, features_list in sorted(feature_types.items()):
            count = len(features_list)
            summary_parts.append(f"- **{feature_type}**: {count} feature{'s' if count != 1 else ''}")
            
            # Add details for important feature types
            if feature_type in ["DOMAIN", "BINDING", "ACTIVE_SITE", "VARIANT"]:
                details = []
                # Limit to 5 examples
                for feature in features_list[:5]:
                    description = feature.get("description", "")
                    if description:
                        details.append(f"  - {description}")
                if details:
                    summary_parts.extend(details)
                if len(features_list) > 5:
                    summary_parts.append(f"  - *and {len(features_list) - 5} more...*")
                    
        return "\n".join(summary_parts)
    
    def _extract_protein_references(self, protein_data: Dict[str, Any]) -> str:
        """Extract database cross-references from protein data."""
        references = protein_data.get("uniProtKBCrossReferences", [])
        
        if not references:
            return "*No database references available*"
            
        # Group references by database
        db_refs = {}
        for ref in references:
            db_type = ref.get("database", "Unknown")
            if db_type not in db_refs:
                db_refs[db_type] = []
            db_refs[db_type].append(ref)
            
        # Format reference summary
        summary_parts = []
        priority_dbs = ["PDB", "GO", "OMIM", "Pfam", "InterPro", "Reactome"]
        
        # First add priority databases
        for db in priority_dbs:
            if db in db_refs:
                count = len(db_refs[db])
                summary_parts.append(f"- **{db}**: {count} reference{'s' if count != 1 else ''}")
                del db_refs[db]
                
        # Then add other databases
        for db, refs in sorted(db_refs.items()):
            count = len(refs)
            summary_parts.append(f"- **{db}**: {count} reference{'s' if count != 1 else ''}")
            
        return "\n".join(summary_parts)

    def get_help_text(self) -> str:
        """Get help text for the service."""
        return """
🧬 **UniProt Protein Database**
- Search proteins: `uniprot: [natural language query]`
- Create a direct query: 
  ````
  ```uniprot
  {
    "query": "gene:BRCA1 AND organism:\\"Homo sapiens\\"",
    "fields": ["accession", "id", "gene_names"],
    "size": 10
  }
  ```
  ````
- Look up protein: `uniprot.lookup P04637`
- Execute query: `uniprot.search uniprot_query_id`
- Convert to dataset: `convert uniprot_query_id to dataset`
"""

    def get_llm_prompt_addition(self) -> str:
        """Get LLM prompt addition for UniProt capabilities."""
        return """
UniProt Protein Database Capabilities:

1. Query Types:
   - Direct JSON queries with field filters
   - Natural language queries with biological interpretation
   - Single protein lookups by accession/ID
   - Dataset conversion with profiling

2. Search Fields:
   a) Core Fields:
      - accession, id, entry_name
      - protein_name, gene_names
      - organism_name, taxonomy_id
      - sequence, length, mass
   
   b) Annotations:
      - Gene Ontology (go)
      - Enzyme Commission (ec)
      - Disease associations
      - Pathways
      - Subcellular locations
   
   c) Quality:
      - reviewed (SwissProt/TrEMBL)
      - annotation_score
      - created/modified dates

3. Query Structure:
   ```uniprot
   {
     "query": "field:value AND field2:value2",
     "fields": ["field1", "field2"],
     "format": "json",
     "size": number
   }
   ```

4. Response Types:
   - Query validation and explanation
   - Result previews with statistics
   - Protein analysis and insights
   - Dataset conversion with profiles
"""

    def _format_chat_history(self, chat_history: Optional[List[Dict]] = None) -> str:
        """Format chat history for LLM context.
        
        Args:
            chat_history: List of chat messages to format, or None
            
        Returns:
            str: Formatted chat history
        """
        if not chat_history:
            return ""
            
        formatted = []
        for msg in chat_history[-5:]:  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '').strip()
            if content:
                formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _get_data_context(self) -> Dict[str, Any]:
        """Get current data context for LLM prompts.
        
        This provides static information about UniProt API capabilities and query formation,
        rather than dynamic data stats which aren't available through the API.
        """
        return {
            'available_fields': [
                'accession', 'id', 'entry_name', 'protein_name', 'gene_names',
                'organism_name', 'taxonomy_id', 'sequence', 'length', 'mass',
                'go', 'ec', 'interpro', 'pfam', 'keywords', 'reviewed',
                'annotation_score', 'created', 'modified', 'subcellular_location',
                'cc_function', 'ft_binding'
            ],
            'query_syntax': {
                'field_operators': [':', 'AND', 'OR', 'NOT'],
                'value_formats': {
                    'exact_match': 'field:value',
                    'phrase_match': 'field:"multiple words"',
                    'range': 'field:[X TO Y]',
                    'existence': 'field:*'
                }
            },
            'special_queries': {
                'reviewed_only': 'reviewed:true',
                'experimental_evidence': 'existence:"ECO:0000269"',
                'with_structure': 'database:PDB',
                'with_go_annotation': 'go:*',
                'taxonomy_search': 'taxonomy_id:286',  # Example for Pseudomonas
                'go_process': 'go:"nitrate metabolic process"',
                'go_function': 'go:"nitrate reductase activity"'
            },
            'example_queries': [
                # Basic queries
                'gene:narG AND reviewed:true',
                'protein_name:"nitrate reductase" AND taxonomy_id:286',
                
                # GO term queries
                'go:GO:0042128 AND reviewed:true',  # nitrate assimilation
                'go:"denitrification process" AND taxonomy_id:286',
                
                # Complex queries
                'taxonomy_id:286 AND (gene:nar OR gene:nir OR gene:nos) AND reviewed:true',
                'go:"nitrogen compound metabolic process" AND taxonomy_id:286 AND reviewed:true'
            ]
        }

    def process_message(self, message: str, chat_history: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Process a message and return a response.
        
        Args:
            message: The message to process
            chat_history: List of previous chat messages for context
            context: Additional context from the app
            
        Returns:
            str: The service's response
        """
        try:
            # Initialize or update context
            if context is None:
                context = {}
            
            # Initialize self.context if not exists
            if not hasattr(self, 'context'):
                self.context = {}
            
            # Merge provided context with existing context
            self.context.update(context)
            
            # Parse the request
            request = self.parse_request(message)
            
            # Process based on request type
            if request['type'] == 'natural_query':
                return self._process_natural_query(request['query'], self.context)
            elif request['type'] == 'direct_query':
                return self._process_direct_query(request['query'], self.context)
            elif request['type'] == 'protein_lookup':
                return self._process_protein_lookup(request['protein_id'], self.context)
            else:
                error_msg = f"Unknown request type: {request['type']}"
                return error_msg
                
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            return error_msg

    def _run_self_test(self, context: Dict[str, Any]) -> ServiceResponse:
        """Run self-tests for UniProt service functionality.
        
        This method tests core functionality within the current application context:
        1. Query parsing and validation
        2. LLM integration
        3. Error handling
        4. Context management
        """
        test_results = []
        passed = 0
        total = 0
        
        def run_test(name: str, test_fn) -> bool:
            nonlocal passed, total
            total += 1
            try:
                test_fn()
                test_results.append(f"✓ {name}")
                passed += 1
                return True
            except Exception as e:
                test_results.append(f"✗ {name}: {str(e)}")
                return False
        
        # 1. Test query parsing
        def test_query_parsing():
            # Test direct query
            message = """```uniprot
{
    "query": "gene:BRCA1 AND organism:\\"Homo sapiens\\"",
    "fields": ["accession", "id", "gene_names"],
    "size": 10
}
```"""
            request = self.parse_request(message)
            assert request['type'] == 'direct_query'
            assert isinstance(request['query'], dict)
            assert 'query' in request['query']
            assert 'fields' in request['query']
        
        run_test("Query parsing", test_query_parsing)
        
        # 2. Test command parsing
        def test_command_parsing():
            # Test info request
            request = self.parse_request("tell me about uniprot")
            assert request['type'] == 'info'
            
            # Test test command
            request = self.parse_request("test uniprot service")
            assert request['type'] == 'test'
            
            # Test query execution
            request = self.parse_request("uniprot.search uniprot_query_20240101_123456_orig")
            assert request['type'] == 'execute_query'
            assert request['query_id'] == 'uniprot_query_20240101_123456_orig'
        
        run_test("Command parsing", test_command_parsing)
        
        # 3. Test query validation
        def test_query_validation():
            # Test valid query
            query = {
                "query": "gene:BRCA1 AND organism:\"Homo sapiens\"",
                "fields": ["accession", "id", "gene_names"],
                "size": 10
            }
            try:
                self.query_builder.build_query_from_json(query)
            except Exception as e:
                raise AssertionError(f"Valid query failed validation: {str(e)}")
            
            # Test invalid query
            invalid_query = {
                "invalid_field": "value"
            }
            try:
                self.query_builder.build_query_from_json(invalid_query)
                raise AssertionError("Invalid query passed validation")
            except Exception:
                pass  # Expected to fail
        
        run_test("Query validation", test_query_validation)
        
        # 4. Test LLM integration
        if 'llm' in context:
            def test_llm_integration():
                # Test query explanation generation
                query = {
                    "query": "gene:BRCA1 AND organism:\"Homo sapiens\"",
                    "fields": ["accession", "id", "gene_names"],
                    "size": 10
                }
                explanation = self._generate_query_explanation(query, context)
                assert explanation is not None
                assert isinstance(explanation, str)
                assert len(explanation) > 0
                
                # Test protein analysis generation
                protein_data = {
                    "primaryAccession": "P04637",
                    "proteinDescription": {
                        "recommendedName": {
                            "fullName": {"value": "Cellular tumor antigen p53"}
                        }
                    },
                    "organism": {"scientificName": "Homo sapiens"}
                }
                analysis = self._generate_protein_analysis(protein_data, "P04637", context)
                assert analysis is not None
                assert isinstance(analysis, str)
                assert len(analysis) > 0
            
            run_test("LLM integration", test_llm_integration)
        else:
            test_results.append("⚠ Skipped LLM tests - no LLM service available")
        
        # 5. Test error handling
        def test_error_handling():
            # Test invalid query handling
            response = self._handle_invalid_query(
                "Invalid JSON",
                "{invalid json",
                context
            )
            assert isinstance(response, ServiceResponse)
            assert len(response.messages) > 0
            assert response.messages[0].message_type == MessageType.ERROR
            
            # Test invalid query ID handling
            response = self._handle_query_execution(
                "invalid_query_id",
                context
            )
            assert isinstance(response, ServiceResponse)
            assert len(response.messages) > 0
            assert response.messages[0].message_type == MessageType.ERROR
        
        run_test("Error handling", test_error_handling)
        
        # Format results
        summary = [
            "### UniProt Service Self-Test Results",
            f"\nPassed: {passed}/{total} tests\n",
            "Detailed Results:"
        ] + test_results
        
        if 'llm' in context:
            summary.append("\nTests run with LLM service available")
        else:
            summary.append("\n⚠ Some tests skipped - no LLM service available")
        
        return ServiceResponse(
            messages=[ServiceMessage(
                service=self.name,
                content="\n".join(summary),
                message_type=MessageType.INFO if passed == total else MessageType.WARNING
            )]
        ) 

    def summarize(self, content: str, chat_history: list) -> str:
        """Generate a summary of the given content.
        
        This method is required by LLMServiceMixin and is used to summarize
        protein data, query results, or other content.
        
        Args:
            content: The content to summarize
            chat_history: List of previous chat messages for context
            
        Returns:
            A summary of the content as a string
        """
        try:
            if 'llm' not in self.context:
                return "Summary not available - no LLM service"
            
            # Format prompt for summarization
            prompt = f"""Please provide a concise summary of the following protein-related content:

{content}

Focus on key biological insights and important features. Keep the summary clear and informative."""

            # Get summary from LLM
            summary = self.context['llm'].complete(prompt)
            return summary
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def _validate_query(self, query: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate and normalize query structure and parameters.
        
        This method both validates the query and attempts to fix common issues.
        
        Args:
            query: Query dictionary to validate and normalize
            
        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Make a copy to avoid modifying the original
            query = query.copy()
            
            # Check required fields
            if 'query' not in query:
                return False, "Missing required field 'query'"
                
            # Validate fields list
            if 'fields' in query:
                available_fields = self._get_data_context()['available_fields']
                invalid_fields = [f for f in query['fields'] if f not in available_fields]
                if invalid_fields:
                    return False, f"Invalid fields: {', '.join(invalid_fields)}"
            
            # Validate format
            if 'format' in query and query['format'] not in ['json', 'tsv']:
                return False, "Format must be 'json' or 'tsv'"
                
            # Validate size
            if 'size' in query:
                try:
                    size = int(query['size'])
                    if size < 1 or size > 500:
                        return False, "Size must be between 1 and 500"
                except (ValueError, TypeError):
                    return False, "Size must be a positive integer"
                    
            # Validate and normalize query syntax
            query_str = query['query']
            if not isinstance(query_str, str):
                return False, "Query must be a string"
                
            # Check for balanced quotes and parentheses
            if query_str.count('"') % 2 != 0:
                return False, "Unmatched quotes in query"
            if query_str.count('(') != query_str.count(')'):
                return False, "Unmatched parentheses in query"
                
            # Split into parts and normalize each
            query_parts = []
            for part in query_str.split(' AND '):
                part = part.strip()
                
                # Handle NOT clauses
                not_prefix = ''
                if part.startswith('NOT '):
                    not_prefix = 'NOT '
                    part = part[4:].strip()
                
                # Extract field and value
                if ':' not in part:
                    return False, f"Invalid query part (missing field:value format): {part}"
                
                field, value = part.split(':', 1)
                field = field.lower().strip()
                value = value.strip()
                
                # Normalize reviewed field
                if field == 'reviewed':
                    if value.lower() in ['yes', 'true', '1']:
                        part = 'reviewed:true'
                    elif value.lower() in ['no', 'false', '0']:
                        part = 'reviewed:false'
                    else:
                        return False, "The 'reviewed' filter value must be 'true' or 'false'"
                
                # Handle organism/taxonomy searches
                elif field in ['organism', 'organism_name']:
                    # Remove wildcards from organism names
                    if '*' in value:
                        return False, "Wildcards are not supported in organism names. Use taxonomy_id for genus-level searches."
                    # Ensure proper quoting
                    if ' ' in value and not (value.startswith('"') and value.endswith('"')):
                        part = f'{field}:"{value}"'
                
                # Handle GO terms
                elif field.startswith('go'):
                    # If it's a GO ID
                    if value.startswith('GO:'):
                        part = f'{field}:{value}'
                    # If it's a text search
                    elif '"' not in value and ' ' in value:
                        part = f'{field}:"{value}"'
                    # If it's a wildcard search
                    elif '*' in value and not value.startswith('"'):
                        part = f'{field}:"{value.replace("*", "")}"'
                
                # Handle gene names
                elif field == 'gene':
                    # Remove wildcards from gene names unless in quotes
                    if '*' in value and not (value.startswith('"') and value.endswith('"')):
                        value = value.replace('*', '')
                    # Add parentheses for multi-word gene names
                    if ' ' in value and not (value.startswith('(') and value.endswith(')')):
                        part = f'{field}:({value})'
                
                # Handle protein names
                elif field == 'protein_name':
                    if ' ' in value and not (value.startswith('"') and value.endswith('"')):
                        part = f'{field}:"{value}"'
                
                query_parts.append(not_prefix + part)
            
            # Reconstruct normalized query
            query['query'] = ' AND '.join(query_parts)
            
            return True, "Query is valid"
            
        except Exception as e:
            return False, f"Error validating query: {str(e)}"

    def _execute_query(self, query: Dict[str, Any]) -> QueryResult:
        """Execute a validated query.
        
        Args:
            query: Validated query dictionary
            
        Returns:
            QueryResult containing results and metadata
        """
        # Set defaults if not specified
        query_with_defaults = query.copy()
        if 'format' not in query_with_defaults:
            query_with_defaults['format'] = 'json'
        if 'size' not in query_with_defaults:
            query_with_defaults['size'] = 10
            
        # Execute query through data manager
        results_df = self.data_manager.search_proteins(
            query=query_with_defaults['query'],
            fields=query_with_defaults.get('fields'),
            format=query_with_defaults['format'],
            size=query_with_defaults['size']
        )
        
        # Generate query description
        description = self.query_builder.generate_query_description(query_with_defaults)
        
        # Create metadata
        metadata = {
            'query': query_with_defaults,
            'execution_time': datetime.now().isoformat(),
            'total_rows': len(results_df),
            'columns': list(results_df.columns)
        }
        
        return QueryResult(
            dataframe=results_df,
            metadata=metadata,
            description=description
        ) 

    def _format_service_message(self, content: str, message_type: MessageType) -> str:
        """Format content with standard service message headers."""
        header = f"Service: {self.name}\nType: {message_type.name}\n\n"
        return header + content 

    def _find_query_by_id(self, query_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find query by ID from chat history.
        
        This method checks the chat history to find a query matching the given ID.
        
        Args:
            query_id: The query ID to find
            context: The context containing chat history
            
        Returns:
            The query object if found, None otherwise
        """
        # Check chat history
        chat_history = context.get('chat_history', [])
        query_text, _ = self.find_recent_query(chat_history, query_id)
        if query_text:
            try:
                return json.loads(query_text)
            except json.JSONDecodeError:
                pass
        
        return None 

    def lookup(self, protein_id: str) -> str:
        """Lookup a protein by accession ID.
        
        This is a convenience method for direct protein lookup without going through
        the full service request/response flow.
        
        Args:
            protein_id: UniProt accession ID
            
        Returns:
            Formatted protein information as a string
        """
        try:
            # Create a minimal context
            context = {'llm': None}
            
            # Call the internal handler
            response = self._handle_protein_lookup(protein_id, context)
            
            # Format the response
            result = ""
            for message in response.messages:
                result += message.content + "\n\n"
                
            return result.strip()
        except Exception as e:
            return f"Error looking up protein {protein_id}: {str(e)}"
            
    def _process_protein_lookup(self, protein_id: str, context: Dict[str, Any]) -> ServiceResponse:
        """Process a protein lookup request.
        
        This is a wrapper around _handle_protein_lookup for use in the process_message flow.
        """
        return self._handle_protein_lookup(protein_id, context) 

    def _fetch_uniprot_statistics(self) -> Dict[str, Any]:
        """Fetch and provide current UniProt statistics.
        
        Returns:
            Dict containing various statistics about UniProt database.
        """
        # Initialize with static data as fallback
        uniprot_stats = {
            'total_entries': "Over 250 million",
            'reviewed_entries': "Over 570,000 (Swiss-Prot)",
            'unreviewed_entries': "Over 250 million (TrEMBL)",
            'taxonomic_distribution': {
                'bacteria': "Over 150 million entries",
                'archaea': "Over 2 million entries",
                'eukaryota': "Over 80 million entries",
                'viruses': "Over 10 million entries",
                'plants': "Over 15 million entries",
                'fungi': "Over 8 million entries"
            },
            'model_organisms': {
                'human': {
                    'total': "Over 20,000 proteins",
                    'reviewed': "Over 18,000 proteins",
                    'with_function': "Approximately 90%",
                    'with_structure': "Over 6,000 proteins"
                },
                'mouse': {
                    'total': "Over 22,000 proteins",
                    'reviewed': "Over 17,000 proteins"
                },
                'e_coli': {
                    'total': "Over 4,500 proteins",
                    'reviewed': "Over 4,300 proteins",
                    'with_function': "Over 85%"
                },
                'arabidopsis': {
                    'total': "Over 27,000 proteins",
                    'reviewed': "Over 15,000 proteins",
                    'with_function': "Approximately 75%"
                },
                'rice': {
                    'total': "Over 48,000 proteins",
                    'reviewed': "Over 3,500 proteins"
                },
                'yeast': {
                    'total': "Over 6,000 proteins",
                    'reviewed': "Over 5,900 proteins",
                    'with_function': "Over 90%"
                }
            },
            'annotation_metrics': {
                'with_go_annotation': "Over 80% of reviewed entries",
                'with_experimental_evidence': "Over 60% of reviewed entries",
                'with_3d_structure': "Over 40% of reviewed entries",
                'with_disease_annotation': "Over 6,000 entries",
                'with_pathway_annotation': "Over 50% of reviewed entries",
                'with_interaction_data': "Over 30% of reviewed entries"
            },
            'microbial_focus': {
                'pathogens': "Over 20 million entries",
                'environmental': "Over 50 million entries",
                'industrial': "Over 10 million entries",
                'extremophiles': "Over 5 million entries"
            },
            'plant_focus': {
                'crops': "Over 8 million entries",
                'model_plants': "Over 2 million entries",
                'medicinal_plants': "Over 1 million entries"
            },
            'last_update': "Recent"
        }
        
        try:
            # First, find the most recent release directory
            import requests
            from bs4 import BeautifulSoup
            import re
            
            # Get the FTP directory listing
            ftp_url = "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
            response = requests.get(ftp_url, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all release directories (they start with "release-")
                release_dirs = []
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href and href.startswith('release-'):
                        release_dirs.append(href.strip('/'))
                
                # Sort to find the most recent one
                if release_dirs:
                    release_dirs.sort(reverse=True)
                    latest_release = release_dirs[0]
                    
                    # Now get the statistics files from the knowledgebase directory
                    swissprot_stats_url = f"{ftp_url}{latest_release}/knowledgebase/UniProtKB_SwissProt-relstat.html"
                    trembl_stats_url = f"{ftp_url}{latest_release}/knowledgebase/UniProtKB_TrEMBL-relstat.html"
                    
                    # Parse SwissProt statistics
                    sp_response = requests.get(swissprot_stats_url, timeout=10)
                    if sp_response.status_code == 200:
                        sp_soup = BeautifulSoup(sp_response.text, 'html.parser')
                        
                        # Extract total entries
                        entry_count_text = sp_soup.find(string=re.compile("Number of entries:"))
                        if entry_count_text:
                            parent = entry_count_text.parent
                            if parent and parent.next_sibling:
                                count_text = parent.next_sibling.strip()
                                uniprot_stats['reviewed_entries'] = f"{count_text} (Swiss-Prot)"
                        
                        # Extract taxonomic distribution
                        taxonomy_table = sp_soup.find('table', {'border': '1'})
                        if taxonomy_table:
                            rows = taxonomy_table.find_all('tr')
                            for row in rows[1:]:  # Skip header row
                                cells = row.find_all('td')
                                if len(cells) >= 2:
                                    taxon = cells[0].text.strip().lower()
                                    count = cells[1].text.strip()
                                    
                                    # Map to our taxonomy categories
                                    if 'bacteria' in taxon:
                                        uniprot_stats['taxonomic_distribution']['bacteria'] = f"{count} entries (reviewed)"
                                    elif 'archaea' in taxon:
                                        uniprot_stats['taxonomic_distribution']['archaea'] = f"{count} entries (reviewed)"
                                    elif 'eukaryota' in taxon:
                                        uniprot_stats['taxonomic_distribution']['eukaryota'] = f"{count} entries (reviewed)"
                                    elif 'virus' in taxon:
                                        uniprot_stats['taxonomic_distribution']['viruses'] = f"{count} entries (reviewed)"
                        
                        # Extract release date
                        release_date_text = sp_soup.find(string=re.compile("Release date:"))
                        if release_date_text:
                            parent = release_date_text.parent
                            if parent and parent.next_sibling:
                                date_text = parent.next_sibling.strip()
                                uniprot_stats['last_update'] = date_text
                    
                    # Parse TrEMBL statistics
                    tr_response = requests.get(trembl_stats_url, timeout=10)
                    if tr_response.status_code == 200:
                        tr_soup = BeautifulSoup(tr_response.text, 'html.parser')
                        
                        # Extract total entries
                        entry_count_text = tr_soup.find(string=re.compile("Number of entries:"))
                        if entry_count_text:
                            parent = entry_count_text.parent
                            if parent and parent.next_sibling:
                                count_text = parent.next_sibling.strip()
                                uniprot_stats['unreviewed_entries'] = f"{count_text} (TrEMBL)"
                        
                        # Calculate total entries
                        try:
                            reviewed = int(uniprot_stats['reviewed_entries'].split()[0].replace(',', ''))
                            unreviewed = int(uniprot_stats['unreviewed_entries'].split()[0].replace(',', ''))
                            total = reviewed + unreviewed
                            uniprot_stats['total_entries'] = f"{total:,}"
                        except (ValueError, IndexError):
                            # Keep the default if parsing fails
                            pass
                        
                        # Extract taxonomic distribution for TrEMBL
                        taxonomy_table = tr_soup.find('table', {'border': '1'})
                        if taxonomy_table:
                            rows = taxonomy_table.find_all('tr')
                            for row in rows[1:]:  # Skip header row
                                cells = row.find_all('td')
                                if len(cells) >= 2:
                                    taxon = cells[0].text.strip().lower()
                                    count = cells[1].text.strip()
                                    
                                    # Update our taxonomy categories with combined counts
                                    if 'bacteria' in taxon:
                                        current = uniprot_stats['taxonomic_distribution']['bacteria']
                                        uniprot_stats['taxonomic_distribution']['bacteria'] = f"{current} + {count} entries (unreviewed)"
                                    elif 'archaea' in taxon:
                                        current = uniprot_stats['taxonomic_distribution']['archaea']
                                        uniprot_stats['taxonomic_distribution']['archaea'] = f"{current} + {count} entries (unreviewed)"
                                    elif 'eukaryota' in taxon:
                                        current = uniprot_stats['taxonomic_distribution']['eukaryota']
                                        uniprot_stats['taxonomic_distribution']['eukaryota'] = f"{current} + {count} entries (unreviewed)"
                                    elif 'virus' in taxon:
                                        current = uniprot_stats['taxonomic_distribution']['viruses']
                                        uniprot_stats['taxonomic_distribution']['viruses'] = f"{current} + {count} entries (unreviewed)"
            
            # Try to get some specific statistics from the REST API
            try:
                # Get human reviewed proteins count
                human_query = "organism_id:9606 AND reviewed:true"
                human_url = f"https://rest.uniprot.org/uniprotkb/search?query={human_query}&format=json&size=1"
                human_response = requests.get(human_url, timeout=5)
                if human_response.status_code == 200:
                    human_data = human_response.json()
                    if 'total' in human_data:
                        human_count = human_data['total']
                        uniprot_stats['model_organisms']['human']['reviewed'] = f"{human_count:,} proteins"
                
                # Get Arabidopsis reviewed proteins count
                arabidopsis_query = "organism_id:3702 AND reviewed:true"
                arabidopsis_url = f"https://rest.uniprot.org/uniprotkb/search?query={arabidopsis_query}&format=json&size=1"
                arabidopsis_response = requests.get(arabidopsis_url, timeout=5)
                if arabidopsis_response.status_code == 200:
                    arabidopsis_data = arabidopsis_response.json()
                    if 'total' in arabidopsis_data:
                        arabidopsis_count = arabidopsis_data['total']
                        uniprot_stats['model_organisms']['arabidopsis']['reviewed'] = f"{arabidopsis_count:,} proteins"
                
                # Get E. coli reviewed proteins count
                ecoli_query = "organism_id:83333 AND reviewed:true"
                ecoli_url = f"https://rest.uniprot.org/uniprotkb/search?query={ecoli_query}&format=json&size=1"
                ecoli_response = requests.get(ecoli_url, timeout=5)
                if ecoli_response.status_code == 200:
                    ecoli_data = ecoli_response.json()
                    if 'total' in ecoli_data:
                        ecoli_count = ecoli_data['total']
                        uniprot_stats['model_organisms']['e_coli']['reviewed'] = f"{ecoli_count:,} proteins"
                
                # Get proteins with 3D structure count
                structure_query = "database:PDB AND reviewed:true"
                structure_url = f"https://rest.uniprot.org/uniprotkb/search?query={structure_query}&format=json&size=1"
                structure_response = requests.get(structure_url, timeout=5)
                if structure_response.status_code == 200:
                    structure_data = structure_response.json()
                    if 'total' in structure_data:
                        structure_count = structure_data['total']
                        uniprot_stats['annotation_metrics']['with_3d_structure'] = f"{structure_count:,} entries"
            
            except Exception as e:
                print(f"Error fetching specific statistics: {str(e)}")
        
        except Exception as e:
            print(f"Error fetching statistics from FTP site: {str(e)}")
        
        return uniprot_stats

    def _extract_api_query_capabilities(self) -> Dict[str, Any]:
        """Extract the most relevant query capabilities from the UniProt API.
        
        Focuses on the most useful search terms and filters for users,
        particularly those related to biological function and annotation.
        
        Returns:
            Dictionary containing key query fields and examples
        """
        return {
            "common_search_fields": {
                # Identifiers
                "accession": "UniProt accession ID (e.g., P04637)",
                "id": "UniProt ID (e.g., P53_HUMAN)",
                "gene": "Gene name (e.g., TP53, BRCA1)",
                
                # Names and descriptions
                "protein_name": "Protein name (e.g., 'Tumor protein p53')",
                "gene_exact": "Exact gene name match (e.g., gene_exact:TP53)",
                
                # Taxonomy
                "organism_name": "Species name (e.g., 'Homo sapiens')",
                "organism_id": "Taxonomy identifier (e.g., 9606 for humans)",
                
                # Quality and curation
                "reviewed": "Review status (true for Swiss-Prot, false for TrEMBL)",
                "annotation_score": "Annotation score 1-5 (higher is better)",
                
                # Sequence properties
                "length": "Sequence length (e.g., [100 TO 200])",
                "mass": "Molecular weight (e.g., [10000 TO 50000])",
                
                # Function and location
                "function": "Protein function (e.g., function:kinase)",
                "go": "Gene Ontology terms (e.g., 'transcription')",
                "go_id": "GO identifier (e.g., GO:0006915 for apoptosis)",
                "pathway": "Pathway annotation (e.g., pathway:Apoptosis)",
                "subcellular_location": "Cellular location (e.g., 'nucleus')",
                
                # Structure
                "structure_3d": "Has 3D structure (true/false)",
                "domain": "Protein domain (e.g., domain:Kinase)",
                
                # Disease and phenotype
                "disease": "Disease association (e.g., 'cancer')",
                "disease_id": "Disease identifier (e.g., DI-03887)",
                "variant": "Protein variant (e.g., variant:rs334)",
                
                # Interactions
                "interactor": "Protein interaction partner (e.g., interactor:P04637)",
                
                # Expression
                "tissue": "Tissue expression (e.g., tissue:brain)",
                "organelle": "Subcellular organelle (e.g., organelle:mitochondrion)"
            },
            "query_syntax_examples": {
                "basic": "gene:TP53 AND organism:\"Homo sapiens\"",
                "complex": "gene:TP53 AND organism:\"Homo sapiens\" AND reviewed:true",
                "range": "length:[100 TO 200] AND organism:\"Homo sapiens\"",
                "exclusion": "gene:TP53 NOT organism:\"Mus musculus\"",
                "function": "go:\"DNA repair\" AND reviewed:true",
                "disease": "disease:cancer AND reviewed:true",
                "structure": "structure_3d:true AND function:kinase",
                "pathway": "pathway:Apoptosis AND reviewed:true",
                "location": "subcellular_location:nucleus AND organism_id:9606",
                "combined": "gene:BRCA* AND disease:cancer AND reviewed:true AND organism_id:9606"
            },
            "biological_field_groups": {
                "function": ["function", "go", "go_id", "pathway", "ec"],
                "structure": ["structure_3d", "domain", "feature", "length", "mass"],
                "disease": ["disease", "disease_id", "variant", "phenotype"],
                "location": ["subcellular_location", "organelle", "tissue"],
                "taxonomy": ["organism_name", "organism_id", "taxonomy_lineage"],
                "interactions": ["interactor", "binary_interaction"]
            },
            "recommended_output_fields": {
                "basic": ["accession", "id", "gene_names", "protein_name", "organism_name", "length", "reviewed"],
                "functional": ["accession", "gene_names", "protein_name", "cc_function", "go", "pathway"],
                "structural": ["accession", "gene_names", "ft_domain", "ft_binding", "structure_3d", "mass"],
                "disease": ["accession", "gene_names", "protein_name", "cc_disease", "variant", "ft_variant"]
            }
        }