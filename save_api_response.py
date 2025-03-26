#!/usr/bin/env python
"""
Script to capture and save NMDC API responses for debugging.

This script makes direct calls to the NMDC API and saves the responses
to JSON files for offline testing with debug_transform.py.
"""

import json
import logging
import argparse
import requests
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nmdc_api_capture")

# API configuration
API_BASE_URL = "https://api.microbiomedata.org"
DEFAULT_PER_PAGE = 20  # Lower value for quick tests

def get_entity_endpoint(entity_type: str) -> str:
    """Convert entity type to plural endpoint name."""
    # Map entity types to their endpoint names
    entity_map = {
        "study": "studies",
        "biosample": "biosamples",
        "data_object": "data_objects"
    }
    
    # Return mapped endpoint or pluralize by adding 's'
    return entity_map.get(entity_type, f"{entity_type}s")

def make_api_request(entity_type, filter_str=None, per_page=DEFAULT_PER_PAGE):
    """Make a request to the NMDC API."""
    endpoint = get_entity_endpoint(entity_type)
    url = f"{API_BASE_URL}/{endpoint}?per_page={per_page}&cursor=*"
    
    if filter_str:
        url += f"&filter={filter_str}"
    
    logger.info(f"Making API request to: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"API request successful. Retrieved {len(data.get('results', []))} results.")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                logger.error(f"API error details: {error_detail}")
            except:
                logger.error(f"Status code: {e.response.status_code}")
        return None

def save_api_response(data, file_path):
    """Save API response to a JSON file."""
    if data is None:
        logger.error("No data to save")
        return False
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved API response to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving API response: {str(e)}")
        return False

def create_sample_filter(entity_type):
    """Create a sample filter based on entity type."""
    if entity_type == "biosample":
        return "specific_ecosystem.search:soil"
    elif entity_type == "study":
        return "study_category.search:soil"
    elif entity_type == "data_object":
        return "file_type_description.search:fastq"
    else:
        return None

def main():
    """Main function to capture API responses."""
    parser = argparse.ArgumentParser(description="Capture NMDC API responses for debugging")
    parser.add_argument("--entity-type", "-e", default="biosample", 
                       choices=["biosample", "study", "data_object"],
                       help="Entity type to request")
    parser.add_argument("--filter", "-f", help="Filter string for the API request")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--count", "-c", type=int, default=DEFAULT_PER_PAGE,
                       help=f"Number of results to request (default: {DEFAULT_PER_PAGE})")
    args = parser.parse_args()
    
    # Create default output filename if not provided
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"{args.entity_type}_data_{timestamp}.json"
    
    # Use sample filter if none provided
    if not args.filter:
        args.filter = create_sample_filter(args.entity_type)
        if args.filter:
            logger.info(f"Using sample filter: {args.filter}")
    
    # Make the API request
    response_data = make_api_request(args.entity_type, args.filter, args.count)
    
    # Save the response
    if response_data:
        success = save_api_response(response_data, args.output)
        if success:
            # Also save just the results array for easy loading
            results = response_data.get('results', [])
            if results:
                results_path = args.output.replace('.json', '_results.json')
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                logger.info(f"Saved {len(results)} results to {results_path}")
    else:
        logger.error("Failed to retrieve API response")

if __name__ == "__main__":
    main() 