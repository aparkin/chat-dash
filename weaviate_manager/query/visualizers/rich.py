"""
Rich text visualizer for search results.

This module provides a rich text visualization of search results,
with detailed formatting and structure.
"""

from typing import Dict, List, Optional, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from rich.style import Style
from datetime import datetime
from rich.columns import Columns

from .base import ResultVisualizer
from ..result_model import StandardResults, CollectionItem

class RichVisualizer(ResultVisualizer):
    """Rich text visualizer for search results."""
    
    # Default maximum length for property values in display
    DEFAULT_MAX_LENGTH = 50
    
    def __init__(self):
        """Initialize the visualizer."""
        self.console = Console()
        
    def show_results(self, results: StandardResults):
        """Display results using rich formatting."""
        # Show query information
        self._show_query_info(results.query_info)
        
        # Show summary statistics
        self._show_summary(results.summary)
        
        # Show either unified results or collection results, not both
        if results.unified_results is not None:
            self._show_unified_results(results.unified_results)
        else:
            self._show_collection_results(results.collection_results)
    
    def _show_query_info(self, query_info):
        """Display query information."""
        info_text = Text()
        info_text.append("\n=== Query Information ===\n", style="bold cyan")
        info_text.append(f"Query: ", style="bold")
        info_text.append(query_info.text + "\n")
        info_text.append(f"Type: ", style="bold")
        info_text.append(query_info.type + "\n")
        info_text.append(f"Time: ", style="bold")
        info_text.append(query_info.timestamp.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        
        # Show parameters
        info_text.append("Parameters:\n", style="bold")
        for key, value in query_info.parameters.items():
            info_text.append(f"  {key}: ", style="bold")
            info_text.append(f"{value}\n")
            
        self.console.print(Panel(info_text, title="Query Info", border_style="cyan"))
        
    def _show_summary(self, summary):
        """Display summary statistics."""
        summary_text = Text()
        summary_text.append("\n=== Results Summary ===\n", style="bold magenta")
        summary_text.append(f"Total Matches: ", style="bold")
        summary_text.append(f"{summary.total_matches}\n")
        
        summary_text.append("Collection Counts:\n", style="bold")
        for collection, count in summary.collection_counts.items():
            summary_text.append(f"  {collection}: ", style="bold")
            summary_text.append(f"{count}\n")
            
        if summary.unified_articles is not None:
            summary_text.append(f"Unified Articles: ", style="bold")
            summary_text.append(f"{summary.unified_articles}\n")
            
        self.console.print(Panel(summary_text, title="Summary", border_style="magenta"))
        
    def _show_collection_results(self, collection_results: Dict[str, List[CollectionItem]]):
        """Display collection results in tables."""
        for collection, items in collection_results.items():
            if not items:
                continue
                
            # Create table for this collection
            table = Table(title=f"{collection} Results", show_header=True)
            
            # Add standard columns
            table.add_column("ID", style="dim")
            table.add_column("Score", justify="right", style="green")
            
            # Only add certainty column if any items have certainty
            show_certainty = any(item.certainty is not None for item in items)
            if show_certainty:
                table.add_column("Certainty", justify="right", style="blue")
            
            # Get property names from first item
            property_names = []
            if items and items[0].properties:
                property_names = list(items[0].properties.values.keys())
                for prop_name in property_names:
                    table.add_column(prop_name.title(), style="cyan")
            
            # Add rows
            for item in items:
                row = [
                    item.id,
                    f"{item.score:.3f}"
                ]
                
                # Add certainty if column exists
                if show_certainty:
                    row.append(f"{item.certainty:.3f}" if item.certainty is not None else "")
                
                # Add property values with truncation
                if item.properties:
                    for prop_name in property_names:
                        value = item.properties.values.get(prop_name, '')
                        formatted_value = str(value)
                        if len(formatted_value) > self.DEFAULT_MAX_LENGTH:
                            formatted_value = formatted_value[:self.DEFAULT_MAX_LENGTH-3] + "..."
                        row.append(formatted_value)
                
                table.add_row(*row)
            
            self.console.print(table)
            self.console.print()
            
            # Show cross-references and score explanations
            for item in items:
                panels = []
                
                # Add cross-references panel if present
                if item.cross_references:
                    ref_text = Text()
                    ref_text.append(f"Cross-references for {item.id}:\n", style="bold yellow")
                    for ref_type, ref_ids in item.cross_references.items():
                        ref_text.append(f"  {ref_type}: ", style="bold")
                        ref_text.append(f"{', '.join(ref_ids)}\n")
                    panels.append(Panel(ref_text, border_style="yellow"))
                
                # Add score explanation panel if present
                if item.score_explanation:
                    panels.append(Panel(
                        f"Score explanation for {item.id}:\n{item.score_explanation}",
                        border_style="blue"
                    ))
                
                # Print panels in columns if multiple exist
                if len(panels) > 1:
                    self.console.print(Columns(panels))
                elif panels:
                    self.console.print(panels[0])
    
    def _show_unified_results(self, unified_results: List[CollectionItem]):
        """Display unified results with clear source tracking."""
        if not unified_results:
            return
            
        self.console.print("\n[bold green]=== Unified Results ===[/bold green]")
        
        # Create unified results table
        table = Table(
            show_header=True,
            header_style="bold",
            box=ROUNDED,
            title="Unified Results",
            title_style="bold green"
        )
        
        # Add standard columns
        table.add_column("Score", justify="right", style="cyan")
        table.add_column("ID", style="blue")
        table.add_column("Sources", style="magenta")
        
        # Add property columns based on first item
        first_item = unified_results[0]
        property_columns = sorted(first_item.properties.values.keys())
        for prop in property_columns:
            table.add_column(prop.title(), style="white")
        
        # Track seen IDs for duplicate detection
        seen_ids = set()
        
        # Add rows
        for item in unified_results:
            # Check for duplicates
            if item.id in seen_ids:
                self.console.print(f"[red]Warning: Duplicate unified result ID: {item.id}[/red]")
                continue
            seen_ids.add(item.id)
            
            # Determine sources from cross-references
            sources = []
            if item.cross_references:
                for collection, refs in item.cross_references.items():
                    if refs:  # Only include collections with actual references
                        sources.append(collection)
            source_str = ", ".join(sources) if sources else "Direct"
            
            # Format row data
            row = [
                f"{item.score:.3f}",
                item.id,
                source_str
            ]
            
            # Add property values with truncation
            for prop in property_columns:
                value = item.properties.values.get(prop, '')
                formatted_value = str(value)
                if len(formatted_value) > self.DEFAULT_MAX_LENGTH:
                    formatted_value = formatted_value[:self.DEFAULT_MAX_LENGTH-3] + "..."
                row.append(formatted_value)
            
            table.add_row(*row)
            
            # Show detailed cross-references and score explanation
            panels = []
            
            # Add cross-references panel if present
            if item.cross_references:
                ref_text = Text()
                ref_text.append(f"Cross-references for {item.id}:\n", style="bold yellow")
                for ref_type, refs in item.cross_references.items():
                    if refs:  # Only show collections with references
                        ref_text.append(f"  {ref_type}: ", style="bold")
                        ref_text.append(f"{', '.join(refs)}\n")
                panels.append(Panel(ref_text, border_style="yellow"))
            
            # Add score explanation panel if present
            if item.score_explanation:
                panels.append(Panel(
                    f"Score explanation for {item.id}:\n{item.score_explanation}",
                    border_style="blue"
                ))
            
            # Print panels side by side if both exist
            if len(panels) == 2:
                self.console.print(Columns(panels))
            elif panels:
                self.console.print(panels[0])
        
        self.console.print(table) 