"""
USGS Data Catalog service for retrieving and analyzing USGS datasets.

This service provides functionality to:
1. Search USGS datasets based on spatial bounds and keywords
2. Retrieve detailed dataset metadata
3. Process and analyze USGS data in conjunction with existing datasets
"""

import requests
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import json
from urllib.parse import urljoin
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import logging
import re

from .base import ChatService, ServiceMessage, ServiceResponse, MessageType, PreviewIdentifier
from .llm_service import LLMServiceMixin

logger = logging.getLogger(__name__)

class USGSQueryType:
    """Enumeration of supported USGS query types."""
    RADIUS = "radius"
    BBOX = "bbox"
    KEYWORD = "keyword"
    ID = "id"
    
class USGSDataCategory:
    """Common USGS data categories for filtering."""
    ELEVATION = "elevation"
    HYDROGRAPHY = "hydrography"
    WATER_QUALITY = "water-quality"
    GROUNDWATER = "groundwater"
    GEOLOGY = "geology"
    LAND_USE = "land-use"
    
    @classmethod
    def get_all(cls) -> List[str]:
        """Get all available categories."""
        return [attr for attr in dir(cls) if not attr.startswith('_') and isinstance(getattr(cls, attr), str)]

class USGSService(ChatService, LLMServiceMixin):
    """Service for interacting with USGS Data Catalog API."""
    
    BASE_URL = "https://www.sciencebase.gov/catalog/items"
    EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers
    
    def __init__(self):
        ChatService.__init__(self, "usgs")
        LLMServiceMixin.__init__(self, "usgs")
        PreviewIdentifier.register_prefix("usgs")
        
    def can_handle(self, message: str) -> bool:
        """Determine if this service can handle the message."""
        commands = [
            "usgs", "geological survey", "elevation", "water data",
            "topographic", "geographic", "hydrologic"
        ]
        return any(cmd.lower() in message.lower() for cmd in commands)
    
    def parse_request(self, query: str) -> Dict[str, Any]:
        """Parse a natural language query into structured parameters."""
        query = query.lower().strip()
        params = {
            "raw_query": query,
            "query_type": USGSQueryType.KEYWORD,  # Default type
            "spatial_bounds": None,
            "keywords": [],
            "category": None,
            "dataset_id": None
        }

        # Check for dataset ID first
        if "dataset" in query:
            dataset_match = re.search(r'dataset\s+(\S+)', query)
            if dataset_match:
                params["dataset_id"] = dataset_match.group(1)
                params["query_type"] = USGSQueryType.ID
                return params

        # Check for radius search
        if "radius" in query:
            radius_match = re.search(r'radius\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(\d+\.?\d*)', query)
            if radius_match:
                lat = float(radius_match.group(1))
                lon = float(radius_match.group(2))
                radius_km = float(radius_match.group(3))
                params["query_type"] = USGSQueryType.RADIUS
                params["spatial_bounds"] = {
                    "center_lon": lon,
                    "center_lat": lat,
                    "radius_km": radius_km
                }
                remaining_text = query[radius_match.end():].strip()
                params["category"] = self._find_category(remaining_text)
                return params

        # Check for bounding box
        if "bbox" in query:
            bbox_match = re.search(r'bbox\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)', query)
            if bbox_match:
                params["query_type"] = USGSQueryType.BBOX
                params["spatial_bounds"] = {
                    "west": float(bbox_match.group(1)),
                    "south": float(bbox_match.group(2)),
                    "east": float(bbox_match.group(3)),
                    "north": float(bbox_match.group(4))
                }
                remaining_text = query[bbox_match.end():].strip()
                params["category"] = self._find_category(remaining_text)
                return params

        # Check for search command
        if "search" in query:
            search_idx = query.index("search")
            remaining_text = query[search_idx + len("search"):].strip()
            params["category"] = self._find_category(remaining_text)
            if not params["category"]:
                # If no category found, treat remaining text as keywords
                params["keywords"] = [word for word in remaining_text.split() if word not in ["usgs", "search"]]
            return params

        # Check for category in the full query
        params["category"] = self._find_category(query)
        
        # Extract remaining keywords
        words = query.split()
        keywords = []
        i = 0
        while i < len(words):
            if words[i] not in ["usgs", "search", "category"]:
                # Check if this word is part of a hyphenated term
                if i + 2 < len(words) and words[i + 1] == "-":
                    keywords.append(words[i] + "-" + words[i + 2])
                    i += 3
                else:
                    keywords.append(words[i])
                    i += 1
            else:
                i += 1
        
        params["keywords"] = keywords
        return params
    
    def _find_category(self, text: str) -> Optional[str]:
        """Find a category in the given text."""
        # Check for explicit category
        category_match = re.search(r'category[:\s]+([a-zA-Z0-9-]+)', text)
        if category_match:
            return category_match.group(1)
        
        # Check for hyphenated terms that might be categories
        hyphen_match = re.search(r'([a-zA-Z0-9]+)-([a-zA-Z0-9]+)', text)
        if hyphen_match:
            category = hyphen_match.group(1) + "-" + hyphen_match.group(2)
            # Only return if it's a known category
            if category in [cat.lower() for cat in USGSDataCategory.get_all()]:
                return category
        
        # Check for single-word categories
        words = text.split()
        for word in words:
            if word in [cat.lower() for cat in USGSDataCategory.get_all()]:
                return word
        
        return None
    
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute the USGS data retrieval request."""
        try:
            if "error" in params:
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content=params["error"],
                        message_type=MessageType.ERROR
                    )]
                )
                
            if params["query_type"] == USGSQueryType.RADIUS:
                results = self._search_by_radius(
                    params["spatial_bounds"]["center_lon"],
                    params["spatial_bounds"]["center_lat"],
                    params["spatial_bounds"]["radius_km"]
                )
            elif params["query_type"] == USGSQueryType.BBOX:
                results = self._search_datasets(params)
            elif params["query_type"] == USGSQueryType.ID:
                results = self._get_dataset_by_id(params["dataset_id"])
            else:
                results = self._search_datasets(params)
            
            if not results or not results.get("items"):
                return ServiceResponse(
                    messages=[ServiceMessage(
                        service=self.name,
                        content="No USGS datasets found matching your criteria.",
                        message_type=MessageType.INFO
                    )]
                )
            
            # Convert results to DataFrame and generate preview ID
            df = self._results_to_dataframe(results)
            preview_id = PreviewIdentifier.create_id("usgs")
            
            # Create preview content
            preview_content = self._create_preview_content(df, preview_id, params)
            
            # Format detailed results
            detailed_content = self._format_search_results(results)
            
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=preview_content,
                        message_type=MessageType.RESULT
                    ),
                    ServiceMessage(
                        service=self.name,
                        content=detailed_content,
                        message_type=MessageType.SUMMARY
                    )
                ],
                store_updates={
                    f"usgs_results_{preview_id}": df,
                    "last_usgs_results": df,  # Keep this for backward compatibility
                    "last_usgs_preview_id": preview_id
                }
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[
                    ServiceMessage(
                        service=self.name,
                        content=f"Error accessing USGS Data Catalog: {str(e)}",
                        message_type=MessageType.ERROR
                    )
                ]
            )
    
    def _search_by_radius(self, lon: float, lat: float, radius_km: float) -> Dict[str, Any]:
        """Search for datasets within a radius of a point."""
        # Convert radius to approximate bounding box for initial filter
        # This is a rough approximation that will be refined
        km_per_degree = 111.32  # approximate km per degree at equator
        delta_lat = radius_km / km_per_degree
        delta_lon = radius_km / (km_per_degree * cos(radians(lat)))
        
        bounds = {
            "west": lon - delta_lon,
            "south": lat - delta_lat,
            "east": lon + delta_lon,
            "north": lat + delta_lat
        }
        
        # Get initial results using bounding box
        results = self._search_datasets({"spatial_bounds": bounds})
        
        if not results.get("items"):
            return results
            
        # Filter results by actual distance
        filtered_items = []
        for item in results["items"]:
            # Extract dataset center coordinates (this is simplified - real implementation
            # would need to handle various spatial extent formats)
            if "spatial" in item:
                dataset_lat = item["spatial"].get("latitude", 0)
                dataset_lon = item["spatial"].get("longitude", 0)
                
                if self._haversine_distance(lat, lon, dataset_lat, dataset_lon) <= radius_km:
                    filtered_items.append(item)
        
        results["items"] = filtered_items
        return results
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in kilometers.
        
        Uses the haversine formula to calculate the great-circle distance between two points,
        accounting for the Earth's spherical shape.
        
        Args:
            lat1, lon1: Coordinates of first point in decimal degrees
            lat2, lon2: Coordinates of second point in decimal degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula components
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Calculate distance
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return self.EARTH_RADIUS_KM * c
    
    def _get_dataset_by_id(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve detailed information about a specific dataset."""
        endpoint = urljoin(self.BASE_URL, f"datasets/{dataset_id}")
        response = requests.get(endpoint)
        response.raise_for_status()
        
        return {"items": [response.json()]}
    
    def _results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert API results to a pandas DataFrame."""
        items = results.get("items", [])
        
        # Extract relevant fields
        data = []
        for item in items:
            row = {
                "id": item.get("identifier"),
                "title": item.get("title"),
                "description": item.get("description"),
                "publisher": item.get("publisher", {}).get("name"),
                "date_published": item.get("datePublished"),
                "category": item.get("category", []),
                "format": item.get("distribution", [{}])[0].get("format", ""),
                "download_url": item.get("distribution", [{}])[0].get("downloadURL", ""),
                "spatial_coverage": json.dumps(item.get("spatial", {}))
            }
            data.append(row)
            
        return pd.DataFrame(data)
    
    def _search_datasets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for datasets using the ScienceBase API."""
        query_params = {
            "format": "json",
            "max": 50,
            "fields": "title,summary,spatial,tags,dates,webLinks,browseCategories"
        }

        # Handle spatial queries
        if params["spatial_bounds"]:
            if params["query_type"] == "bbox":
                bounds = params["spatial_bounds"]
                query_params["filter"] = f"boundingBox({bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']})"
            elif params["query_type"] == "radius":
                center = params["spatial_bounds"]
                query_params["filter"] = f"distance({center['center_lon']},{center['center_lat']},{center['radius_km']}km)"

        # Add search terms
        search_terms = []
        if params["keywords"]:
            search_terms.extend(params["keywords"])
        if params["category"]:
            search_terms.append(f'browseCategory:"{params["category"]}"')
        if search_terms:
            query_params["q"] = " ".join(search_terms)

        # Make the API request
        self.logger.info(f"Making API request to {self.BASE_URL}")
        self.logger.info(f"Query parameters: {json.dumps(query_params, indent=2)}")
        
        try:
            response = requests.get(self.BASE_URL, params=query_params)
            
            # Log response details
            self.logger.info(f"Response status: {response.status_code}")
            self.logger.info(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
            
            response.raise_for_status()
            
            data = response.json()
            
            # Convert ScienceBase format to our expected format
            items = []
            for item in data.get("items", []):
                # Extract dates
                dates = item.get("dates", [])
                date_published = next((date["dateString"] for date in dates if date["type"] == "Publication"), "")
                
                # Extract download URL
                web_links = item.get("webLinks", [])
                download_url = next((link["uri"] for link in web_links if link.get("type") == "download"), "")
                
                processed_item = {
                    "identifier": item.get("id"),
                    "title": item.get("title"),
                    "description": item.get("summary", ""),
                    "publisher": {"name": "USGS"},
                    "datePublished": date_published,
                    "category": item.get("browseCategories", []),
                    "distribution": [{
                        "format": "Unknown",
                        "downloadURL": download_url
                    }],
                    "spatial": item.get("spatial", {})
                }
                items.append(processed_item)
            
            result = {"items": items}
            self.logger.info(f"Found {len(items)} items in response")
            
            if not items:
                self.logger.warning("No items found in response")
                if data.get("error"):
                    self.logger.error(f"API error: {data['error']}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                self.logger.error(f"Response text: {e.response.text}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse API response: {str(e)}")
            self.logger.error(f"Raw response: {response.text[:1000]}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in API call: {str(e)}")
            raise
    
    def _format_search_results(self, results: Dict[str, Any]) -> str:
        """Format search results into markdown."""
        items = results.get("items", [])
        
        if not items:
            return "No datasets found."
            
        output = ["### USGS Datasets Found\n"]
        
        for item in items[:5]:  # Limit to top 5 results
            title = item.get("title", "Untitled Dataset")
            description = item.get("description", "No description available.")
            identifier = item.get("identifier", "")
            
            output.extend([
                f"#### {title}",
                f"**ID**: `{identifier}`",
                f"{description[:200]}..." if len(description) > 200 else description,
                "\n"
            ])
            
        if len(items) > 5:
            output.append(f"\n*Showing top 5 of {len(items)} results*")
            
        return "\n".join(output)
    
    def get_help_text(self) -> str:
        """Get help text for the USGS service."""
        return """
### USGS Data Service

Search and retrieve data from the USGS Data Catalog:

Spatial Queries:
- `usgs radius <lat> <lon> <miles>` - Find datasets within radius of point
- `usgs bbox <west> <south> <east> <north>` - Find datasets in bounding box

Category Queries:
- `usgs search <category>` - Search by category (elevation, hydrography, etc.)
- `usgs id <dataset_id>` - Get detailed information about a specific dataset

Available Categories:
{}

Results are returned as both formatted text and a pandas DataFrame stored for further analysis.
""".format("\n".join([f"- {cat}" for cat in USGSDataCategory.get_all()]))

    def _create_preview_content(self, df: pd.DataFrame, preview_id: str, params: Dict[str, Any]) -> str:
        """Create a preview of the results with query context."""
        # Get query type description
        query_desc = self._get_query_description(params)
        
        # Create summary statistics
        total_results = len(df)
        unique_categories = df['category'].explode().unique()
        date_range = None
        if 'date_published' in df.columns and not df['date_published'].isna().all():
            date_range = f"{df['date_published'].min()} to {df['date_published'].max()}"
        
        # Format preview content
        preview_lines = [
            f"### USGS Dataset Search Results `{preview_id}`\n",
            f"**Query**: {query_desc}",
            f"**Total Results**: {total_results}",
        ]
        
        if date_range:
            preview_lines.append(f"**Date Range**: {date_range}")
            
        if len(unique_categories) > 0:
            categories_str = ", ".join([f"`{cat}`" for cat in unique_categories if cat])
            preview_lines.append(f"**Categories**: {categories_str}")
            
        # Add DataFrame preview
        preview_lines.extend([
            "\n**Preview of Results**:",
            "```",
            df.head(3).to_string() if len(df) > 0 else "No results",
            "```"
        ])
        
        return "\n".join(preview_lines)
        
    def _get_query_description(self, params: Dict[str, Any]) -> str:
        """Generate a human-readable description of the query."""
        if params["query_type"] == USGSQueryType.RADIUS:
            return (f"Datasets within {params['radius_km']} kilometers of "
                   f"({params['spatial_bounds']['center_lon']:.4f}, {params['spatial_bounds']['center_lat']:.4f})")
        elif params["query_type"] == USGSQueryType.BBOX:
            bounds = params["spatial_bounds"]
            return (f"Datasets in bounding box: W:{bounds['west']:.4f}, "
                   f"S:{bounds['south']:.4f}, E:{bounds['east']:.4f}, "
                   f"N:{bounds['north']:.4f}")
        elif params["query_type"] == USGSQueryType.ID:
            return f"Dataset with ID: {params['dataset_id']}"
        elif params["query_type"] == USGSQueryType.KEYWORD:
            return f"Datasets in category: {params['category']}"
        else:
            return "General dataset search"

    def process_message(self, message: str, chat_history: List[Dict[str, Any]]) -> str:
        """Process a message using LLM capabilities.
        
        This method is used to:
        1. Interpret natural language queries about USGS data
        2. Extract spatial and categorical constraints
        3. Generate human-friendly explanations of results
        """
        system_prompt = """You are a USGS data expert assistant. Your role is to:
1. Help users find relevant USGS datasets based on their needs
2. Interpret geographic areas and data types they're interested in
3. Explain the significance and relationships between different datasets
4. Suggest relevant additional datasets based on their interests

When processing queries:
- Extract specific geographic bounds or points of interest
- Identify relevant data categories (elevation, water quality, etc.)
- Consider temporal aspects if mentioned
- Look for relationships between different data types"""

        response = self._call_llm(
            messages=[{"role": "user", "content": message}],
            system_prompt=system_prompt
        )
        
        return response.strip()
    
    def summarize(self, content: str, chat_history: List[Dict[str, Any]]) -> str:
        """Summarize USGS dataset information.
        
        This method provides:
        1. Concise summaries of dataset collections
        2. Key relationships between datasets
        3. Geographic and temporal patterns
        4. Suggestions for additional relevant datasets
        """
        system_prompt = """Summarize the USGS dataset information focusing on:
1. Key patterns and relationships between datasets
2. Geographic coverage and any spatial patterns
3. Temporal trends if present
4. Data quality and completeness
5. Potential applications and use cases

Format the summary with:
- Overview of the data collection
- Key findings and patterns
- Suggested next steps or related datasets"""

        response = self._call_llm(
            messages=[{"role": "user", "content": f"Please summarize this USGS dataset information:\n\n{content}"}],
            system_prompt=system_prompt
        )
        
        return response.strip()
    
    def get_llm_prompt_addition(self) -> str:
        """Provide context about USGS service capabilities for LLM prompts."""
        return """
USGS Data Capabilities:
- Search for datasets within a radius of a point (e.g., "find elevation data within 10 miles of San Francisco")
- Search within bounding boxes (e.g., "find water quality data in the Bay Area")
- Search by data categories (elevation, hydrography, water quality, groundwater, geology, land use)
- Retrieve detailed dataset metadata and download information
- Analyze spatial and temporal patterns in dataset availability
- Convert results to analyzable dataframes

Available Categories:
{}

The service can handle both precise coordinate-based queries and natural language geographic descriptions.""".format(
            "\n".join(f"- {cat}" for cat in USGSDataCategory.get_all())
        ) 