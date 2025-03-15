"""
Query builder for UniProt service.

This module provides the UniProtQueryBuilder class which handles:
1. Construction of UniProt API queries from various inputs
2. Query validation and formatting
3. Natural language query interpretation
4. Field mapping and validation
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import logging

from .models import UniProtConfig

class UniProtQueryBuilder:
    """Builder for UniProt API queries.
    
    Handles:
    - Direct JSON query construction
    - Natural language to query conversion
    - Query validation and formatting
    - Field mapping for UniProt API
    """
    
    def __init__(self, data_manager):
        """Initialize with data manager.
        
        Args:
            data_manager: UniProt data manager instance
        """
        self.data_manager = data_manager
        self.logger = logging.getLogger(__name__)
        
        # Common fields to retrieve
        self.default_fields = [
            "accession", 
            "id", 
            "protein_name", 
            "gene_names", 
            "organism_name", 
            "length", 
            "reviewed", 
            "annotation_score"
        ]
        
        # Mapping of common terms to UniProt query syntax
        self.field_mapping = {
            "gene": "gene:",
            "protein": "protein_name:",
            "organism": "organism_name:",
            "taxonomy": "taxonomy_id:",
            "disease": "disease:",
            "pathway": "pathway:",
            "function": "go:",
            "subcellular location": "subcellular_location:",
            "enzyme": "ec:"
        }
        
        # Common operators
        self.operators = {
            "and": "AND",
            "or": "OR",
            "not": "NOT"
        }
    
    def build_query_from_json(self, query_json: Dict[str, Any]) -> Dict[str, Any]:
        """Build structured query from JSON input.
        
        Args:
            query_json: Dictionary with query parameters
            
        Returns:
            Dictionary with validated query parameters
            
        Raises:
            ValueError: For invalid query parameters
        """
        # Validate required fields
        if "query" not in query_json:
            raise ValueError("Query parameter is required")
            
        # Set defaults for optional parameters
        fields = query_json.get("fields", self.default_fields)
        format = query_json.get("format", "json")
        size = query_json.get("size", 25)
        
        # Validate size
        try:
            size = int(size)
            if size < 1 or size > 500:
                self.logger.warning(f"Invalid size value {size}, using default of 25")
                size = 25
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid size value {size}, using default of 25")
            size = 25
            
        # Validate format
        if format not in ["json", "tsv"]:
            self.logger.warning(f"Invalid format {format}, using default of json")
            format = "json"
            
        # Validate fields
        if not fields or not isinstance(fields, list):
            self.logger.warning(f"Invalid fields, using defaults")
            fields = self.default_fields
            
        return {
            "query": query_json["query"],
            "fields": fields,
            "format": format,
            "size": size
        }
    
    def build_query_from_natural_language(self, query_text: str, llm_service) -> Dict[str, Any]:
        """Build structured query from natural language.
        
        Args:
            query_text: Natural language query text
            llm_service: LLM service for query interpretation
            
        Returns:
            Dict containing structured query
        """
        try:
            # Generate prompt directly instead of loading from file
            prompt = f"""You are an expert bioinformatician specializing in protein databases. 
Your task is to convert a natural language query about proteins into a structured UniProt query.

Natural language query: "{query_text}"

Please convert this into a structured UniProt query using the following syntax rules:
1. Field-specific searches use field:value format (e.g., gene:BRCA1)
2. Combined conditions use AND, OR, NOT operators (in uppercase)
3. Phrases should be quoted (e.g., organism:"Homo sapiens")
4. Common fields include: gene, protein_name, organism_name, disease, pathway, go, ec, length, mass, etc.

Example queries:
1. Find human proteins involved in DNA repair:
   {{
     "query": "go:\\"DNA repair\\" AND organism:\\"Homo sapiens\\" AND reviewed:true",
     "fields": ["accession", "id", "gene_names", "protein_name", "organism_name", "go"],
     "format": "json",
     "size": 10
   }}

2. Find bacterial chemotaxis proteins:
   {{
     "query": "go:\\"bacterial chemotaxis\\" AND reviewed:true",
     "fields": ["accession", "id", "gene_names", "protein_name", "organism_name", "go"],
     "format": "json",
     "size": 10
   }}

Respond with a JSON object containing:
- query: The UniProt search syntax string
- fields: Array of fields to retrieve (include at least accession, id, protein_name, gene_names, organism_name)
- format: Response format (should be json)
- size: Maximum number of results (between 1-100)

Your response should be valid JSON only, no other text."""
            
            # Get structured query from LLM
            response = llm_service.get_llm_response(prompt)
            
            # Parse response as JSON
            try:
                structured_query = json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(f"LLM response was not valid JSON: {str(e)}")
            
            # Validate query structure
            if not isinstance(structured_query, dict):
                raise ValueError("Query must be a dictionary")
            
            required_fields = ['query', 'fields', 'format', 'size']
            for field in required_fields:
                if field not in structured_query:
                    raise ValueError(f"Missing required field: {field}")
            
            # Set defaults if not specified
            structured_query.setdefault('format', 'json')
            structured_query.setdefault('size', 10)
            
            return structured_query
            
        except Exception as e:
            raise ValueError(f"Error building query from natural language: {str(e)}")
    
    def generate_query_description(self, query: Dict[str, Any]) -> str:
        """Generate human-readable description of a query.
        
        Args:
            query: Structured query dictionary
            
        Returns:
            Human-readable description string
        """
        # Start with the raw query
        query_str = query.get("query", "")
        
        # Replace field names with more readable versions
        for readable, syntax in self.field_mapping.items():
            query_str = query_str.replace(syntax, f"{readable} ")
            
        # Replace operators
        for readable, syntax in self.operators.items():
            query_str = query_str.replace(f" {syntax} ", f" {readable} ")
            
        # Add information about the expected results
        size = query.get("size", 25)
        description = f"Query for up to {size} proteins where {query_str}"
        
        # Add information about the fields being returned
        fields = query.get("fields", [])
        if fields and len(fields) > 0:
            field_str = ", ".join(field.replace("_", " ") for field in fields)
            description += f", returning {field_str}"
            
        return description
    
    def _get_nlp_prompt(self, natural_query: str) -> str:
        """Generate prompt for natural language query interpretation.
        
        Args:
            natural_query: Natural language query
            
        Returns:
            Prompt string for LLM
        """
        return f"""
You are an expert bioinformatician specializing in protein databases. 
Your task is to convert a natural language query about proteins into a structured UniProt query.

Natural language query: "{natural_query}"

Please convert this into a structured UniProt query using the following syntax rules:
1. Field-specific searches use field:value format (e.g., gene:BRCA1)
2. Combined conditions use AND, OR, NOT operators (in uppercase)
3. Phrases should be quoted (e.g., organism:"Homo sapiens")
4. Common fields include: gene, protein_name, organism_name, disease, pathway, go, ec, length, mass, etc.

Respond with a JSON object containing:
- query: The UniProt search syntax string
- fields: Array of fields to retrieve (include at least accession, id, protein_name, gene_names, organism_name, length, reviewed)
- format: Response format (should be json)
- size: Maximum number of results (between 1-100)

Your response should be valid JSON only, no other text.
"""
    
    def _extract_query_from_llm(self, llm_response: str) -> Dict[str, Any]:
        """Extract structured query from LLM response.
        
        Args:
            llm_response: Response from LLM
            
        Returns:
            Dictionary with structured query parameters
            
        Raises:
            ValueError: If unable to extract valid JSON
        """
        # Try to extract JSON from the response
        try:
            # Extract JSON object using regex in case there's surrounding text
            match = re.search(r'({[\s\S]*})', llm_response)
            if match:
                json_str = match.group(1)
                query = json.loads(json_str)
                return query
            else:
                # If no match, try parsing the whole response
                query = json.loads(llm_response)
                return query
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to extract query from LLM response: {e}")
            self.logger.debug(f"LLM response: {llm_response}")
            
            # Fallback: create a simple query
            return {
                "query": "accession:*",  # Query for all proteins (will be limited by size)
                "fields": self.default_fields,
                "format": "json",
                "size": 10
            } 