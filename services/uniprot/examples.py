"""
Examples of using the UniProt service.

This script provides examples of how to use the UniProt service,
including direct queries, natural language queries, and protein lookups.
"""

import json
import asyncio
import re
import sys
from pathlib import Path
from typing import List, Optional

# Add the root directory to Python path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

from services.uniprot.service import UniProtService
from services.uniprot.models import UniProtConfig
from services.base import ServiceMessage, ServiceResponse
from services.chat_llm_service import ChatLLMService


def print_message(message):
    """Print a service message in a formatted way."""
    print(f"\n--- {message.message_type.name} ---")
    print(message.content)
    print("---\n")


async def run_examples():
    """Run example usages of the UniProt service."""
    print("\nInitializing UniProt service...")
    service = UniProtService()
    print("UniProt service initialized.\n")
    
    # Initialize LLM service for enhanced responses
    llm_service = ChatLLMService()
    context = {
        'llm': llm_service,
        'chat_history': [],
        'successful_queries_store': {},
        'datasets_store': {}
    }
    
    # Example 1: Service Info
    print("\n=== EXAMPLE 1: SERVICE INFO ===\n")
    request = service.parse_request("tell me about uniprot")
    response = service.execute(request, context)
    for message in response.messages:
        print_message(message)
        # Update chat history
        context['chat_history'].append({
            'role': 'assistant',
            'content': message.content,
            'service': service.name
        })
    
    # Example 2: Direct Query
    print("\n=== EXAMPLE 2: DIRECT QUERY ===\n")
    
    query_json = """
    {
      "query": "gene:BRCA1 AND organism_name:\\"Homo sapiens\\"",
      "fields": ["accession", "id", "gene_names", "protein_name", "organism_name"],
      "size": 5
    }
    """
    
    response = await service.process_message(f"```uniprot\n{query_json}\n```", context)
    for message in response.messages:
        print_message(message)
        # Update chat history
        context['chat_history'].append({
            'role': 'assistant',
            'content': message.content,
            'service': service.name
        })
    
    # Save the query ID for later
    query_id = None
    for message in response.messages:
        if "Query ID:" in message.content:
            match = re.search(r'`(uniprot_query_\d{8}_\d{6}(?:_orig|_alt\d+))`', message.content)
            if match:
                query_id = match.group(1)
                break
    
    if query_id:
        # Example 3: Execute Query
        print("\n=== EXAMPLE 3: EXECUTE QUERY ===\n")
        
        response = await service.process_message(f"uniprot.search {query_id}", context)
        for message in response.messages:
            print_message(message)
            # Update chat history
            context['chat_history'].append({
                'role': 'assistant',
                'content': message.content,
                'service': service.name
            })
            
            # Update stores if needed
            if response.store_updates:
                for store_name, updates in response.store_updates.items():
                    if store_name in context:
                        context[store_name].update(updates)
    
    # Example 4: Natural Language Query
    print("\n=== EXAMPLE 4: NATURAL LANGUAGE QUERY ===\n")
    
    nl_query = "uniprot: Find human proteins associated with Alzheimer's disease"
    response = await service.process_message(nl_query, context)
    for message in response.messages:
        print_message(message)
        # Update chat history
        context['chat_history'].append({
            'role': 'assistant',
            'content': message.content,
            'service': service.name
        })
    
    # Example 5: Protein Lookup
    print("\n=== EXAMPLE 5: PROTEIN LOOKUP ===\n")
    
    lookup_query = "uniprot.lookup P04637"
    response = await service.process_message(lookup_query, context)
    for message in response.messages:
        print_message(message)
        # Update chat history
        context['chat_history'].append({
            'role': 'assistant',
            'content': message.content,
            'service': service.name
        })


if __name__ == "__main__":
    asyncio.run(run_examples()) 