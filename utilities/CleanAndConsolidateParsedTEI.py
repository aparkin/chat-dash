#!/usr/bin/env python3
"""
CleanAndConsolidateParsedTEI.py

This script processes parsed TEI (Text Encoding Initiative) documents from scientific articles
and consolidates them into a structured format suitable for database import. It handles multiple
aspects of scientific articles including authors, references, named entities, and section content.

It is run after ParseTEI.py. An example usage is:

ls data/grobid_output/output/*.json  | python3 utilities/CleanAndConsolidateParsedTEI.py --output-dir data/grobid_output/processed_output --workers 5

Key Features:
- Parallel processing of article sections using multiprocessing
- Intelligent section classification using pattern matching and LLM
- Author name parsing and consolidation
- Reference deduplication and linking
- Named Entity Recognition (NER) result processing
- Progress tracking with nested progress bars

Processing Pipeline:
1. Author Processing:
   - Parses and normalizes author names
   - Consolidates author variants
   - Links authors across articles and references

2. Reference Processing:
   - Consolidates references across articles
   - Links references to unified authors
   - Tracks reference occurrences in different sections

3. NER Processing:
   - Consolidates named entities (genes, organisms, etc.)
   - Tracks entity occurrences and confidence scores

4. Article Processing:
   - Classifies sections (intro, methods, results, discussion)
   - Processes figures and tables
   - Extracts metadata (affiliations, funding, etc.)

Usage:
    ls *.json | python CleanAndConsolidateParsedTEI.py --output-dir /path/to/output --workers 4

Input:
    - List of JSON files (from stdin) containing parsed TEI data
    - Each JSON file should contain a single article's parsed content

Output:
    - unified_authors.json: Consolidated author information
    - unified_references.json: Consolidated reference information
    - unified_ner_objects.json: Consolidated named entities
    - processed_articles.json: Processed article content and metadata

Dependencies:
    - Python 3.7+
    - openai
    - tqdm
    - multiprocessing
    - unidecode
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from unidecode import unidecode
import time
from tqdm import tqdm  # For progress bars
import multiprocessing as mp
from functools import partial
import os
from dotenv import load_dotenv
from pathlib import Path

# Find the project root directory (where .env is located)
project_root = Path(__file__).parent
dotenv_path = project_root / '.env'

# Try to load from .env file
load_dotenv(dotenv_path=dotenv_path)

from dataclasses import dataclass
from typing import List, Dict, Set
import openai

# OpenAI Settings
if True:  # Toggle for development environment
    OPENAI_BASE_URL = os.getenv('CBORG_BASE_URL', "https://api.cborg.lbl.gov")
    OPENAI_API_KEY = os.getenv('CBORG_API_KEY', '')  # Must be set in environment
else:  # Production environment
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # Must be set in environment

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

class AuthorNameParser:
    """
    Parses and normalizes author names from various formats into structured components.
    
    Handles common name formats including:
    - "Last, First Middle"
    - "First Middle Last"
    - Names with prefixes (de, van, von, etc.)
    
    Attributes:
        name_prefixes (set): Common prefixes that might appear in last names
    """
    
    def __init__(self):
        # Common prefixes that might appear in last names
        self.name_prefixes = {'de', 'van', 'von', 'del', 'della', 'di', 'da', 'la', 'le', 'den'}
        
    def parse_complex_name(self, name: str) -> dict:
        """
        Parse complex author names into standardized components.
        
        Args:
            name (str): Full author name in any common format
            
        Returns:
            dict: Parsed name components with keys:
                - first_names: List of first names
                - middle_names: List of middle names
                - last_name: Complete last name including prefixes
                - full_name: Original full name
        """
        # Remove extra whitespace and commas
        name = ' '.join(name.split())
        parts = [p.strip() for p in name.split(',')]
        
        if len(parts) >= 2:
            # Handle "Last, First Middle" format
            last_name = parts[0]
            other_names = parts[1].split()
        else:
            # Handle "First Middle Last" format
            parts = name.split()
            # Start from the end to identify last name with prefixes
            last_name_parts = []
            other_names = []
            
            i = len(parts) - 1
            while i >= 0:
                current = parts[i].lower()
                # If we find a prefix, include it and the following word in last name
                if current in self.name_prefixes and i < len(parts) - 1:
                    last_name_parts.insert(0, parts[i])
                    i -= 1
                    continue
                # If we haven't found any last name parts yet, this must be part of the last name
                if not last_name_parts:
                    last_name_parts.insert(0, parts[i])
                else:
                    # We've found all last name parts
                    other_names = parts[:i+1]
                    break
                i -= 1
            
            last_name = ' '.join(last_name_parts)
            
        # Separate first and middle names
        if other_names:
            first_names = [other_names[0]]
            middle_names = other_names[1:] if len(other_names) > 1 else []
        else:
            first_names = []
            middle_names = []
            
        return {
            'first_names': first_names,
            'middle_names': middle_names,
            'last_name': last_name,
            'full_name': name  # preserve original full name
        }
    
    def get_normalized_forms(self, parsed_name: dict) -> List[str]:
        """
        Generate different normalized forms of the name for matching.
        
        Creates multiple standardized versions of the name to help with
        author matching across different formats.
        
        Args:
            parsed_name (dict): Output from parse_complex_name()
            
        Returns:
            List[str]: Different normalized forms of the name, including:
                - Full form with all components
                - Initial form (first initial + middle initials + last name)
        """
        forms = []
        
        # Full form
        full = []
        if parsed_name['first_names']:
            full.extend(parsed_name['first_names'])
        if parsed_name['middle_names']:
            full.extend(parsed_name['middle_names'])
        full.append(parsed_name['last_name'])
        forms.append(' '.join(full))
        
        # First initial, middle initials, last name
        initials = []
        if parsed_name['first_names']:
            initials.append(parsed_name['first_names'][0][0])
        for middle in parsed_name['middle_names']:
            initials.append(middle[0])
        forms.append(f"{' '.join(initials)} {parsed_name['last_name']}")
        
        return forms

@dataclass
class UnifiedAuthor:
    """
    Represents a unified author entity across multiple articles and references.
    
    Tracks all variants of an author's name and their appearances in both
    primary articles and references.
    
    Attributes:
        canonical_name (str): Most complete form of the author's name
        email (Optional[str]): Author's email if available
        name_variants (Set[str]): All original forms of the name found
        primary_articles (Set[str]): Articles where author is primary
        reference_appearances (Set[str]): References where author appears
    """
    canonical_name: str  # Most complete form of the name
    email: Optional[str]
    name_variants: Set[str]  # All original forms of the name found
    primary_articles: Set[str]  # Articles where author is primary
    reference_appearances: Set[str]  # References where author appears
    
    def add_variant(self, variant: str):
        """
        Add a name variant and update canonical name if more complete.
        
        Args:
            variant (str): New variant of the author's name
            
        Updates canonical_name if the new variant contains more information
        (e.g., full middle name instead of initial).
        """
        self.name_variants.add(variant)
        # Update canonical_name if this variant is more complete
        if len(variant.split()) > len(self.canonical_name.split()):
            self.canonical_name = variant

class AuthorProcessor:
    """
    Processes and consolidates author information across multiple articles.
    
    Handles the complete pipeline of author processing:
    1. Name parsing and normalization
    2. Author matching and consolidation
    3. Tracking author appearances
    4. Managing author relationships
    
    Attributes:
        name_parser (AuthorNameParser): Parser for handling author names
        unified_authors (Dict[str, UnifiedAuthor]): Consolidated author data
        original_to_unified (Dict[str, str]): Maps original names to unified keys
    """
    
    def __init__(self):
        self.name_parser = AuthorNameParser()
        self.unified_authors = {}  # key: normalized form -> UnifiedAuthor
        self.original_to_unified = {}  # key: original form -> normalized form
        
    def process_articles(self, parsed_articles: List[Dict]):
        """
        Process author information from multiple articles.
        
        Performs two passes:
        1. Process primary article authors
        2. Process reference authors
        
        Args:
            parsed_articles (List[Dict]): List of parsed article data
        """
        # First pass: process primary article authors
        for article in parsed_articles:
            for author in article.get('authors', []):
                self.process_author(
                    author.get('name', ''),
                    author.get('email'),
                    article['filename'],
                    is_primary=True
                )
        
        # Second pass: process reference authors
        for article in parsed_articles:
            for ref in article.get('references', []):
                for author_str in ref.get('authors', []):
                    self.process_author(
                        author_str,
                        email=None,
                        source_id=ref['id'],
                        is_primary=False
                    )
    
    def process_author(self, name: str, email: Optional[str], source_id: str, is_primary: bool):
        """
        Process a single author name and maintain mappings.
        
        Args:
            name (str): Author's name
            email (Optional[str]): Author's email if available
            source_id (str): ID of the source (article or reference)
            is_primary (bool): True if author is from primary article
        """
        parsed_name = self.name_parser.parse_complex_name(name)
        normalized_forms = self.name_parser.get_normalized_forms(parsed_name)
        
        # Try to find existing unified author
        unified_key = None
        for form in normalized_forms:
            if form in self.unified_authors:
                unified_key = form
                break
        
        if unified_key is None:
            # Create new unified author
            unified_key = normalized_forms[0]  # Use first normalized form as key
            self.unified_authors[unified_key] = UnifiedAuthor(
                canonical_name=name,
                email=email,
                name_variants={name},
                primary_articles=set(),
                reference_appearances=set()
            )
        else:
            # Update existing unified author
            self.unified_authors[unified_key].add_variant(name)
            if email and not self.unified_authors[unified_key].email:
                self.unified_authors[unified_key].email = email
        
        # Update source tracking
        if is_primary:
            self.unified_authors[unified_key].primary_articles.add(source_id)
        else:
            self.unified_authors[unified_key].reference_appearances.add(source_id)
            
        # Maintain mapping from original to unified
        self.original_to_unified[name] = unified_key
    
    def find_author_key(self, author_str: str) -> Optional[str]:
        """
        Find the unified author key for a given author string.
        
        Args:
            author_str (str): Author name to look up
            
        Returns:
            Optional[str]: Key to unified author if found, None otherwise
        """
        # First check direct mapping
        if author_str in self.original_to_unified:
            return self.original_to_unified[author_str]
        
        # If not found, try normalizing and matching
        parsed_name = self.name_parser.parse_complex_name(author_str)
        normalized_forms = self.name_parser.get_normalized_forms(parsed_name)
        
        for form in normalized_forms:
            if form in self.unified_authors:
                # Add this variant to our mappings
                self.original_to_unified[author_str] = form
                self.unified_authors[form].add_variant(author_str)
                return form
                
        return None

@dataclass
class ReferenceOccurrence:
    """
    Tracks where and how a reference appears in an article.
    
    Attributes:
        article_id (str): Identifier of the article containing the reference
        local_ref_id (str): The article-specific ID (e.g., "b6")
        sections (List[str]): Sections where this reference appears
    """
    article_id: str
    local_ref_id: str  # The article-specific ID (e.g., "b6")
    sections: List[str]  # Sections where this reference appears

@dataclass
class UnifiedReference:
    """
    Represents a consolidated reference across multiple articles.
    
    Tracks all occurrences of a reference and maintains links to unified authors.
    
    Attributes:
        title (str): Reference title
        authors (List[str]): Original author strings
        unified_authors (List[str]): Keys to UnifiedAuthor objects
        journal (str): Journal name
        volume (str): Volume information
        pages (str): Page numbers
        publication_date (str): Publication date
        raw_reference (str): Original reference string
        occurrences (List[ReferenceOccurrence]): List of appearances in articles
    """
    title: str
    authors: List[str]  # Original author strings
    unified_authors: List[str]  # Keys to UnifiedAuthor objects
    journal: str
    volume: str
    pages: str
    publication_date: str
    raw_reference: str
    occurrences: List[ReferenceOccurrence]

class ReferenceProcessor:
    """
    Processes and consolidates references across multiple articles.
    
    Links references to unified authors and tracks their occurrences
    throughout the corpus.
    
    Attributes:
        unified_references (Dict[str, UnifiedReference]): Consolidated references
        author_processor (AuthorProcessor): For author linking
    """
    
    def __init__(self, author_processor: AuthorProcessor):
        """
        Initialize the reference processor.
        
        Args:
            author_processor (AuthorProcessor): Processor for author linking
        """
        self.unified_references = {}
        self.author_processor = author_processor
        
    def generate_reference_key(self, ref: dict) -> str:
        """
        Generate a stable key for a reference based on its metadata
        """
        key_parts = [
            ref.get('title', '').lower()[:50],
            '_'.join(ref.get('authors', [])[:2]),
            ref.get('publication_date', ''),
            ref.get('journal', '')
        ]
        return '_'.join(filter(None, key_parts))
    
    def process_articles(self, parsed_articles: List[Dict]):
        for article in parsed_articles:
            for ref in article.get('references', []):
                ref_key = self.generate_reference_key(ref)
                
                # Look up unified authors from the already-processed data
                unified_author_keys = []
                for author_str in ref.get('authors', []):
                    # Try to find this author in the unified authors
                    author_key = self.author_processor.find_author_key(author_str)
                    if author_key:
                        unified_author_keys.append(author_key)
                        # Update the author's reference appearances with this reference
                        self.author_processor.unified_authors[author_key].reference_appearances.add(ref_key)
                
                if ref_key not in self.unified_references:
                    self.unified_references[ref_key] = UnifiedReference(
                        title=ref.get('title', ''),
                        authors=ref.get('authors', []),  # Original strings
                        unified_authors=unified_author_keys,  # Keys to pre-unified authors
                        journal=ref.get('journal', ''),
                        volume=ref.get('volume', ''),
                        pages=ref.get('pages', ''),
                        publication_date=ref.get('publication_date', ''),
                        raw_reference=ref.get('raw_reference', ''),
                        occurrences=[]
                    )
                
                # Track this occurrence
                sections = []
                for section in article.get('body_sections', []):
                    if f"#{ref['id']}" in section.get('citations', []):
                        sections.append(section['title'])
                
                occurrence = ReferenceOccurrence(
                    article_id=article['filename'],
                    local_ref_id=ref['id'],
                    sections=sections
                )
                self.unified_references[ref_key].occurrences.append(occurrence)

    def get_reference_authors(self, ref_key: str) -> List[UnifiedAuthor]:
        """Get all unified authors for a reference"""
        if ref_key not in self.unified_references:
            return []
        return [
            self.author_processor.unified_authors[author_key]
            for author_key in self.unified_references[ref_key].unified_authors
        ]

@dataclass
class UnifiedNERObject:
    """
    Represents a consolidated named entity across multiple articles.
    
    Tracks occurrences and confidence scores for named entities like
    genes, bioprocesses, chemicals, or organisms.
    
    Attributes:
        name (str): Entity name
        type (str): Entity type (Gene, Bioprocess, Chemical, or Organism)
        article_scores (Dict[str, float]): Confidence scores by article
    """
    name: str
    type: str  # Gene, Bioprocess, Chemical, or Organism
    article_scores: Dict[str, float]  # article_id -> best_score

class NERProcessor:
    def __init__(self):
        self.unified_ner_objects = {}
        
    def process_articles(self, parsed_articles: List[Dict]):
        for article in parsed_articles:
            # Look for properties starting with 'ner_results'
            ner_properties = [
                prop for prop in article.keys() 
                if prop.startswith('ner_results')
            ]
            
            for prop in ner_properties:
                # Extract type from property name (after last hyphen)
                ner_type = prop.split('-')[-1]
                
                for ner_obj in article.get(prop, []):
                    term = ner_obj['term']
                    score = ner_obj['best_score']
                    
                    key = f"{ner_type}:{term}"
                    if key not in self.unified_ner_objects:
                        self.unified_ner_objects[key] = UnifiedNERObject(
                            name=term,
                            type=ner_type,
                            article_scores={}
                        )
                    
                    self.unified_ner_objects[key].article_scores[article['filename']] = score

@dataclass
class ProcessedArticle:
    filename: str
    affiliations: str
    abstract: str
    funding_info: str
    publication_info: str
    acknowledgements: str  # Added acknowledgements
    introduction: str
    methods: str
    results: str
    discussion: str
    figures: str
    tables: str

class ArticleProcessor:
    def __init__(self, num_workers: int = 1):
        self.processed_articles: Dict[str, ProcessedArticle] = {}
        self.num_workers = num_workers
        
        # Common section title patterns
        self.section_patterns = {
            'introduction': [
                r'introduction',
                r'background',
                r'overview'
            ],
            'methods': [
                r'methods',
                r'materials and methods',
                r'experimental procedures',
                r'methodology',
                r'experimental setup',
                r'experimental design'
            ],
            'results': [
                r'results',
                r'findings',
                r'experimental results',
                r'observations'
            ],
            'discussion': [
                r'discussion',
                r'conclusions',
                r'concluding remarks',
                r'general discussion'
            ]
        }
        # Compile all patterns
        self.section_patterns = {
            k: [re.compile(p, re.IGNORECASE) for p in patterns]
            for k, patterns in self.section_patterns.items()
        }
        
    @staticmethod
    def process_section(section: Dict) -> Tuple[str, str, str]:
        """Process a single section - designed for multiprocessing"""
        title = section['title']
        text = section['text']
        
        # First try pattern matching
        title_lower = title.lower()
        patterns = {
            'introduction': [r'introduction', r'background', r'overview'],
            'methods': [r'methods', r'materials and methods', r'experimental procedures'],
            'results': [r'results', r'findings', r'experimental results'],
            'discussion': [r'discussion', r'conclusions', r'concluding remarks']
        }
        
        for section_type, pattern_list in patterns.items():
            if any(re.search(p, title_lower) for p in pattern_list):
                return title, text, section_type
        
        # If no match, use LLM
        try:
            # Create client inside worker process
            client = openai.OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                base_url=os.getenv('OPENAI_BASE_URL')
            )
            
            response = client.chat.completions.create(
                model="openai/gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing scientific articles. "
                     "Your task is to determine which standard section a given text belongs to. "
                     "Respond with only one word: introduction, methods, results, or discussion."},
                    {"role": "user", "content": f"Title: {title}\n\nText excerpt: {text[:1500]}\n\n"
                     "Choose one of: introduction, methods, results, discussion"}
                ],
                temperature=0.1,
                max_tokens=10
            )
            section_type = response.choices[0].message.content.strip().lower()
            if section_type in {'introduction', 'methods', 'results', 'discussion'}:
                return title, text, section_type
            
        except Exception as e:
            print(f"Error in LLM section inference: {str(e)}", file=sys.stderr)
        
        return title, text, None

    def process_articles(self, parsed_articles: List[Dict]):
        total_articles = len(parsed_articles)
        total_sections = sum(len(article.get('body_sections', [])) for article in parsed_articles)
        
        # Create a process pool
        pool = mp.Pool(self.num_workers)
        
        with tqdm(total=total_articles, desc="Processing articles", unit="article") as article_pbar:
            for article in parsed_articles:
                # Initialize section texts
                section_texts = {
                    'introduction': [],
                    'methods': [],
                    'results': [],
                    'discussion': []
                }
                
                # Process sections in parallel
                sections = article.get('body_sections', [])
                
                # Process sections with progress bar
                with tqdm(total=len(sections), 
                         desc=f"Sections ({article['filename']})", 
                         leave=False) as section_pbar:
                    for title, text, section_type in pool.imap_unordered(self.process_section, sections):
                        if section_type:
                            section_texts[section_type].append(f"{title}: {text}")
                            section_pbar.set_postfix({'type': section_type})
                        else:
                            section_pbar.set_postfix({'type': 'unknown'})
                        section_pbar.update(1)
                
                # Update article progress with section distribution
                distribution = {k: len(v) for k, v in section_texts.items()}
                article_pbar.set_postfix(distribution)
                
                # Process the rest of the article data
                figures_str = '\n'.join(
                    f"Figure_{i+1}: {fig}"
                    for i, fig in enumerate(article.get('figures', []))
                )
                
                tables_str = '\n'.join(
                    f"Table_{i+1}: {table}"
                    for i, table in enumerate(article.get('tables', []))
                )
                
                self.processed_articles[article['filename']] = ProcessedArticle(
                    filename=article['filename'],
                    affiliations=self.combine_affiliations(article.get('affiliations', [])),
                    abstract=article.get('abstract', ''),
                    funding_info=self.process_funding_info(article.get('funding_info', [])),
                    publication_info=self.process_publication_info(article.get('publication_info', {})),
                    acknowledgements=article.get('acknowledgements', ''),
                    introduction='\n'.join(section_texts['introduction']),
                    methods='\n'.join(section_texts['methods']),
                    results='\n'.join(section_texts['results']),
                    discussion='\n'.join(section_texts['discussion']),
                    figures=figures_str,
                    tables=tables_str
                )
                
                article_pbar.update(1)
        
        pool.close()
        pool.join()

    def combine_affiliations(self, affiliations: List[Dict]) -> str:
        """Combine organization names and addresses"""
        affiliation_strings = []
        for aff in affiliations:
            org_names = aff.get('org_names', [])
            address = aff.get('address', '')
            if org_names:  # Only process if there are org names
                org_str = '; '.join(org_names)
                if address:
                    affiliation_strings.append(f"{org_str}: {address}")
                else:
                    affiliation_strings.append(org_str)
        return ' | '.join(affiliation_strings)
        
    def process_funding_info(self, funding_info: List[Dict]) -> str:
        """Process funding information into a single string"""
        funding_strings = []
        for info in funding_info:
            funder = info.get('funder', '').strip()
            grant = info.get('grant_number', '').strip()
            if funder or grant:  # Include if either field has content
                parts = []
                if funder:
                    parts.append(f"Funder: {funder}")
                if grant:
                    parts.append(f"Grant: {grant}")
                funding_strings.append(' - '.join(parts))
        return '; '.join(funding_strings)
        
    def process_publication_info(self, pub_info: Dict) -> str:
        """Process publication information into a single string"""
        info_parts = []
        
        # Map of fields to their labels
        fields = {
            'publisher': 'Publisher',
            'availability': 'Availability',
            'published_date': 'Published',
            'md5': 'MD5',
            'doi': 'DOI',
            'submission_note': 'Submission Note'
        }
        
        # Add non-empty fields
        for field, label in fields.items():
            value = pub_info.get(field, '').strip()
            if value:  # Only include non-empty strings
                info_parts.append(f"{label}: {value}")
            
        return '; '.join(info_parts)

class PreprocessingPipeline:
    """
    Coordinates the complete preprocessing pipeline for scientific articles.
    
    Manages the sequence of processing steps:
    1. Author processing and consolidation
    2. Reference processing and linking
    3. Named entity consolidation
    4. Article content processing
    
    Attributes:
        output_dir (Path): Directory for output files
        author_processor (AuthorProcessor): Handles author processing
        reference_processor (ReferenceProcessor): Handles reference processing
        ner_processor (NERProcessor): Handles named entity processing
        article_processor (ArticleProcessor): Handles article content processing
    """
    def __init__(self, output_dir: Path, num_workers: int = 1):
        self.output_dir = output_dir
        self.author_processor = AuthorProcessor()
        self.reference_processor = ReferenceProcessor(self.author_processor)
        self.ner_processor = NERProcessor()
        self.article_processor = ArticleProcessor(num_workers)
        
    def process_files(self, input_files: List[str]):
        """
        Process multiple input files through the complete pipeline.
        
        Args:
            input_files (List[str]): List of input file paths
            
        Returns:
            Dict: Processed data including authors, references, NER objects, and articles
        """
        parsed_articles = []
        
        for file_path in input_files:
            try:
                # Get the full filename without the final .json extension
                filename = Path(file_path).name
                if filename.endswith('.json'):
                    filename = filename[:-5]  # Remove .json
                
                # Don't make any assumptions about other parts of the filename
                # Just preserve it as-is minus the .json
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    article_data = json.load(f)
                    # Add filename to article data
                    article_data['filename'] = filename
                    parsed_articles.append(article_data)
            except json.JSONDecodeError:
                print(f"Error: Could not parse JSON from {file_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)
        
        return self.process(parsed_articles)
    
    def process(self, parsed_articles: List[Dict]):
        """
        Run the complete preprocessing pipeline on parsed articles.
        
        Coordinates the sequence of processing steps and saves results.
        
        Args:
            parsed_articles (List[Dict]): List of parsed article data
            
        Returns:
            Dict: Processed data including all components
        """
        print("Processing authors...", file=sys.stderr)
        self.author_processor.process_articles(parsed_articles)
        
        print("Processing references...", file=sys.stderr)
        self.reference_processor.process_articles(parsed_articles)
        
        print("Processing NER objects...", file=sys.stderr)
        self.ner_processor.process_articles(parsed_articles)
        
        print("Processing articles...", file=sys.stderr)
        self.article_processor.process_articles(parsed_articles)
        
        # Save results to output directory
        self.save_results()
        
        return {
            'authors': self.author_processor.unified_authors,
            'references': self.reference_processor.unified_references,
            'ner_objects': self.ner_processor.unified_ner_objects,
            'articles': self.article_processor.processed_articles
        }
    
    def save_results(self):
        """
        Save all processed data to output files.
        
        Creates JSON files for:
        - unified_authors.json
        - unified_references.json
        - unified_ner_objects.json
        - processed_articles.json
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save authors
        with open(self.output_dir / 'unified_authors.json', 'w', encoding='utf-8') as f:
            json.dump({
                name: {
                    'canonical_name': author.canonical_name,
                    'email': author.email,
                    'name_variants': list(author.name_variants),
                    'primary_articles': list(author.primary_articles),
                    'reference_appearances': list(author.reference_appearances)
                }
                for name, author in self.author_processor.unified_authors.items()
            }, f, indent=2)
        
        # Save references
        with open(self.output_dir / 'unified_references.json', 'w', encoding='utf-8') as f:
            json.dump({
                ref_id: {
                    'title': ref.title,
                    'authors': ref.authors,
                    'unified_authors': ref.unified_authors,
                    'journal': ref.journal,
                    'volume': ref.volume,
                    'pages': ref.pages,
                    'publication_date': ref.publication_date,
                    'raw_reference': ref.raw_reference,
                    'occurrences': [
                        {
                            'article_id': occ.article_id,
                            'local_ref_id': occ.local_ref_id,
                            'sections': occ.sections
                        }
                        for occ in ref.occurrences
                    ]
                }
                for ref_id, ref in self.reference_processor.unified_references.items()
            }, f, indent=2)
        
        # Save NER objects
        with open(self.output_dir / 'unified_ner_objects.json', 'w', encoding='utf-8') as f:
            json.dump({
                ner_id: {
                    'name': ner.name,
                    'type': ner.type,
                    'article_scores': ner.article_scores
                }
                for ner_id, ner in self.ner_processor.unified_ner_objects.items()
            }, f, indent=2)
        
        # Save processed articles
        with open(self.output_dir / 'processed_articles.json', 'w', encoding='utf-8') as f:
            json.dump({
                filename: {
                    'filename': article.filename,
                    'affiliations': article.affiliations,
                    'abstract': article.abstract,
                    'funding_info': article.funding_info,
                    'publication_info': article.publication_info,
                    'acknowledgements': article.acknowledgements,
                    'introduction': article.introduction,
                    'methods': article.methods,
                    'results': article.results,
                    'discussion': article.discussion,
                    'figures': article.figures,
                    'tables': article.tables
                }
                for filename, article in self.article_processor.processed_articles.items()
            }, f, indent=2)

def validate_processed_data(data: Dict) -> Tuple[bool, List[str]]:
    """Validate processed data before saving to JSON."""
    errors = []
    
    # Validate articles
    for article_id, article in data['articles'].items():
        if not article.get('filename'):
            errors.append(f"Article {article_id} missing filename")
        if not any([article.get('abstract'), article.get('introduction'), 
                   article.get('methods'), article.get('results'),
                   article.get('discussion')]):
            errors.append(f"Article {article_id} has no content sections")
    
    # Validate references
    for ref_id, ref in data['references'].items():
        if not ref.get('raw_reference'):
            errors.append(f"Reference {ref_id} missing raw text")
        if not ref.get('occurrences'):
            errors.append(f"Reference {ref_id} has no occurrences")
    
    # Validate named entities
    for entity_id, entity in data['ner_objects'].items():
        if not entity.get('name') or not entity.get('type'):
            errors.append(f"Entity {entity_id} missing name or type")
        if not entity.get('article_scores'):
            errors.append(f"Entity {entity_id} has no article occurrences")
    
    return len(errors) == 0, errors

def main():
    parser = argparse.ArgumentParser(description='Preprocess parsed article data for Weaviate import')
    parser.add_argument('--output-dir', type=Path, required=True,
                      help='Directory to store processed output files')
    parser.add_argument('--workers', type=int, default=1,
                      help='Number of worker processes for parallel processing (default: 1)')
    args = parser.parse_args()
    
    # Read list of input files from stdin
    input_files = [line.strip() for line in sys.stdin if line.strip()]
    
    if not input_files:
        print("Error: No input files provided", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nPreprocessing Pipeline Configuration:")
    print(f"Output directory: {args.output_dir}")
    print(f"Worker processes: {args.workers}")
    print(f"Input files: {len(input_files)}")
    
    pipeline = PreprocessingPipeline(args.output_dir, args.workers)
    pipeline.process_files(input_files)
    print(f"\nProcessing complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()