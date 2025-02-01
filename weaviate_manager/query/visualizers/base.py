"""
Base visualization components for search results.

This module provides the core visualization functionality for displaying
standardized search results in various formats.
"""

from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from ..result_model import StandardResults, CollectionItem, ItemProperties

class ResultVisualizer:
    """Base class for visualizing search results."""
    
    def __init__(self):
        """Initialize the visualizer."""
        self.console = Console()
        
    def display_results(self, results: StandardResults):
        """
        Display complete search results.
        
        Args:
            results: Standardized search results to display
        """
        self._show_query_info(results.query_info)
        self._show_summary(results.summary)
        self._show_collection_results(results.collection_results)
        if results.unified_results:
            self._show_unified_results(results.unified_results)
            
    def _show_query_info(self, query_info):
        """Display query information."""
        self.console.print("\n=== Search Query ===", style="bold magenta")
        self.console.print(f"Query: {query_info.text}")
        self.console.print(f"Search type: {query_info.type}")
        
        # Show relevant parameters based on search type
        if query_info.type == 'hybrid':
            if 'alpha' in query_info.parameters:
                self.console.print(
                    f"Alpha (keyword/vector balance): {query_info.parameters['alpha']}"
                )
        
        # Show other common parameters
        if 'min_score' in query_info.parameters:
            self.console.print(f"Minimum score: {query_info.parameters['min_score']}")
        if 'limit' in query_info.parameters:
            self.console.print(f"Result limit per collection: {query_info.parameters['limit']}")
            
    def _show_summary(self, summary):
        """Display results summary."""
        self.console.print("\n=== Search Results Summary ===", style="bold magenta")
        self.console.print(f"Total matches: {summary.total_matches}")
        
        # Show collection counts
        for collection, count in summary.collection_counts.items():
            if count > 0:
                self.console.print(f"- {collection}: {count} matches")
                
        # Show unified articles count if present
        if summary.unified_articles is not None:
            self.console.print(
                f"\nUnified into {summary.unified_articles} distinct articles"
            )
            
        self.console.print("=" * 50)
        
    def _create_base_table(self, title: str) -> Table:
        """Create a base table with standard columns."""
        table = Table(
            title=title,
            box=box.SIMPLE,
            show_header=True,
            header_style="bold",
            show_lines=True,
            expand=True
        )
        
        # Add standard columns
        table.add_column("Score", style="magenta", justify="right", width=10)
        table.add_column("Certainty", style="yellow", justify="right", width=10)
        table.add_column("ID", style="cyan", width=36)
        
        return table
        
    def _add_property_columns(self, table: Table, properties: ItemProperties):
        """Add columns for item properties."""
        for name in sorted(properties.values.keys()):
            width = properties.display_widths.get(name, 30)
            table.add_column(
                name.replace('_', ' ').title(),
                justify="left",
                width=width,
                overflow="fold"
            )
            
    def _format_property_value(self, value: Any, max_length: int = 50) -> str:
        """Format a property value for display."""
        if value is None:
            return ""
            
        # Handle lists and dicts
        if isinstance(value, (list, dict)):
            value = str(value)
            
        # Truncate if too long
        value = str(value)
        if len(value) > max_length:
            return value[:max_length-3] + "..."
            
        return value
        
    def _show_collection_results(self, collection_results: Dict[str, List[CollectionItem]]):
        """Display results for each collection."""
        for collection_name, items in collection_results.items():
            if not items:
                continue
                
            # Create table for this collection
            table = self._create_base_table(f"{collection_name} Results")
            
            # Add property columns based on first item
            first_item = items[0]
            self._add_property_columns(table, first_item.properties)
            
            # Track explanations for display after table
            explanations = []
            
            # Add rows
            for item in items:
                # Start with standard columns
                row = [
                    f"{item.score:.3f}",
                    f"{item.certainty:.3f}",
                    item.id
                ]
                
                # Add property values
                for prop_name in sorted(item.properties.values.keys()):
                    value = item.properties.values.get(prop_name)
                    max_length = item.properties.display_widths.get(prop_name, 50)
                    row.append(self._format_property_value(value, max_length))
                
                table.add_row(*row)
                
                if item.score_explanation:
                    explanations.append((item.id, item.score_explanation))
            
            # Display table
            self.console.print(table)
            
            # Show explanations if any
            if explanations:
                self.console.print("\nScore Explanations:", style="bold")
                for item_id, explanation in explanations:
                    self.console.print(f"{item_id}:", style="cyan")
                    self.console.print(f"  {explanation}")
            
            # Show cross-references if any
            self._show_cross_references(items)
            
            self.console.print()  # Add spacing between collections
            
    def _show_unified_results(self, unified_results: List[CollectionItem]):
        """Display unified results."""
        if not unified_results:
            return
            
        self.console.print("\n=== Unified Results ===", style="bold magenta")
        
        # Create table
        table = self._create_base_table("Unified Article Results")
        
        # Add standard article columns
        table.add_column("Title/Filename", style="green", width=50)
        table.add_column("Abstract/Content", style="yellow", width=50)
        table.add_column("Matched Collections", style="red", width=30)
        
        # Track explanations
        explanations = []
        
        for item in unified_results:
            # Get title/filename and abstract/content
            props = item.properties.values
            title = (
                props.get('title') or 
                props.get('filename') or 
                'No title/filename'
            )
            content = (
                props.get('abstract') or 
                props.get('content') or 
                'No abstract/content'
            )
            
            # Format matched collections
            matched = []
            for coll, refs in item.cross_references.items():
                if refs:
                    matched.append(f"{coll} ({len(refs)})")
            
            # Add row
            table.add_row(
                f"{item.score:.3f}",
                f"{item.certainty:.3f}",
                item.id,
                self._format_property_value(title),
                self._format_property_value(content),
                ", ".join(matched) or "Direct hit only"
            )
            
            if item.score_explanation:
                explanations.append((item.id, item.score_explanation))
        
        # Display table
        self.console.print(table)
        
        # Show explanations
        if explanations:
            self.console.print("\nScore Explanations:", style="bold")
            for item_id, explanation in explanations:
                self.console.print(f"{item_id}:", style="cyan")
                self.console.print(f"  {explanation}")
                
        self.console.print()
        
    def _show_cross_references(self, items: List[CollectionItem]):
        """Display cross-references for items."""
        # Group items by their cross-references
        for item in items:
            if not item.cross_references:
                continue
                
            self.console.print(f"\nCross-References for {item.id}")
            
            for ref_type, ref_ids in item.cross_references.items():
                if not ref_ids:
                    continue
                    
                # Create table for this reference type
                table = Table(
                    title=f"{ref_type} References",
                    box=box.DOUBLE_EDGE
                )
                table.add_column("ID", style="cyan")
                table.add_column("Count", style="yellow")
                
                table.add_row(
                    ", ".join(ref_ids[:5]) + 
                    ("..." if len(ref_ids) > 5 else ""),
                    str(len(ref_ids))
                )
                
                self.console.print(table) 