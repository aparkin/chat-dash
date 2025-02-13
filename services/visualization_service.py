from typing import Dict, Any, Optional, Tuple
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import colorsys
import numpy as np
from .base import ChatService, ServiceResponse, ServiceMessage, ServiceContext
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import pdist

class VisualizationType:
    """Base class for visualization types with standard interface."""
    
    def __init__(self):
        self.required_params = set()
        self.optional_params = set()
        
    def validate_params(self, params: dict, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate parameters against requirements and dataframe."""
        missing = self.required_params - set(params.keys())
        if missing:
            return False, f"Missing required parameters: {', '.join(missing)}"
        return True, ""
        
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract parameters from message. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement extract_params")
        
    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create the visualization figure. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement create_figure")
        
    def apply_view_settings(self, fig: go.Figure, view_settings: Optional[Dict] = None) -> go.Figure:
        """Apply stored view settings to the figure.
        
        Args:
            fig: The plotly figure to update
            view_settings: Dictionary of view settings (zoom, center, range, shapes, etc.)
            
        Returns:
            Updated figure with view settings applied
        """
        if not view_settings:
            return fig
            
        # Create a copy of the layout to modify
        layout_updates = {}
        
        # Apply relevant view settings
        for key, value in view_settings.items():
            # Handle different types of view settings
            if 'zoom' in key:
                if 'mapbox' in key:
                    layout_updates.setdefault('mapbox', {})['zoom'] = value
                else:
                    axis = key.split('.')[0]
                    layout_updates[axis] = layout_updates.get(axis, {})
                    layout_updates[axis]['zoom'] = value
                    
            elif 'center' in key:
                if 'mapbox' in key:
                    layout_updates.setdefault('mapbox', {})['center'] = value
                    
            elif 'range' in key:
                axis = key.split('.')[0]
                layout_updates[axis] = layout_updates.get(axis, {})
                layout_updates[axis]['range'] = value
                
            elif 'domain' in key:
                axis = key.split('.')[0]
                layout_updates[axis] = layout_updates.get(axis, {})
                layout_updates[axis]['domain'] = value
                
            elif key == 'shapes':
                # Restore any drawn shapes
                layout_updates['shapes'] = value
        
        # Update the figure layout
        if layout_updates:
            fig.update_layout(**layout_updates)
            
        return fig

    @staticmethod
    def generate_colors(n: int) -> list:
        """Generate n visually distinct colors.
        
        Args:
            n: Number of colors needed
            
        Returns:
            List of colors in RGB format suitable for plotly
        """
        colors = []
        for i in range(n):
            h = i / n  # Spread hues evenly
            s = 0.7    # Moderate saturation
            v = 0.9    # High value for visibility
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            
            # Convert to 0-255 range and then to plotly rgb string
            colors.append(f'rgb({int(r*255)},{int(g*255)},{int(b*255)})')
        return colors

    @staticmethod
    def is_valid_color(color_str: str) -> bool:
        """Check if a string is a valid color specification.
        
        Args:
            color_str: Color string to validate
            
        Returns:
            bool: True if valid color specification
        """
        # Common named colors (extend this list as needed)
        valid_colors = {
            'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
            'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
            'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
            'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson',
            'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray',
            'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
            'darkorange', 'darkorchid', 'darkred', 'darksalmon',
            'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
            'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue',
            'firebrick', 'forestgreen', 'fuchsia', 'gainsboro', 'gold',
            'gray', 'green', 'greenyellow', 'hotpink', 'indianred',
            'indigo', 'khaki', 'lavender', 'lime', 'magenta',
            'maroon', 'navy', 'olive', 'orange', 'purple',
            'red', 'silver', 'teal', 'white', 'yellow'
        }
        
        return (
            color_str.lower() in valid_colors or
            bool(re.match(r'^#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?$', color_str)) or  # Hex colors
            bool(re.match(r'^rgb\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)$', color_str))  # RGB format
        )

class BubblePlot(VisualizationType):
    """Bubble plot visualization type."""
    
    def __init__(self):
        super().__init__()
        self.required_params = {'x_column', 'y_column'}
        self.optional_params = {'size', 'color'}
    
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract plot parameters supporting both x/y and vs syntaxes."""
        params = {}
        
        # Try x=, y= syntax first - preserve original case in parameter
        x_match = re.search(r'x=(\w+)', message, re.IGNORECASE)
        y_match = re.search(r'y=(\w+)', message, re.IGNORECASE)
        
        if x_match and y_match:
            x_col = x_match.group(1)  # Keep original case
            y_col = y_match.group(1)  # Keep original case
        else:
            # Try vs/versus/against syntax
            vs_match = re.search(r'plot\s+(\w+)\s+(?:vs|versus|against)\s+(\w+)', message, re.IGNORECASE)
            if not vs_match:
                return {}, "Could not parse plot parameters"
            y_col = vs_match.group(1)  # Keep original case
            x_col = vs_match.group(2)  # Keep original case
        
        # Case-sensitive column check
        if x_col not in df.columns or y_col not in df.columns:
            return {}, f"Column not found: {x_col if x_col not in df.columns else y_col}"
            
        params['x_column'] = x_col
        params['y_column'] = y_col
        
        # Handle optional parameters - preserve case in parameter names
        for param in ['color', 'size']:
            match = re.search(rf'{param}=(\w+)', message, re.IGNORECASE)
            if match:
                col = match.group(1)  # Keep original case
                if col not in df.columns:
                    return {}, f"{param.capitalize()} column not found: {col}"
                params[param] = col
                
        return params, None
    
    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create bubble plot figure."""
        try:
            # Process size parameter
            if params.get('size'):
                if params['size'] in df.columns:
                    # Column-based size
                    size_values = df[params['size']]
                    marker_size = 10 + 40 * (size_values - size_values.min()) / (size_values.max() - size_values.min())
                else:
                    # Static size - try to convert to number
                    try:
                        marker_size = float(params['size'])
                    except (ValueError, TypeError):
                        marker_size = 20  # Default if conversion fails
            else:
                marker_size = 20
            
            # Process color parameter
            color_values = None
            color_discrete = False
            colormap = 'viridis'  # Default colormap for numeric data
            
            if params.get('color'):
                if params['color'] in df.columns:
                    color_values = df[params['color']]
                    if pd.api.types.is_numeric_dtype(color_values):
                        color_discrete = False
                    else:
                        color_discrete = True
                        unique_values = color_values.nunique()
                        color_sequence = self.generate_colors(unique_values)
                else:
                    # Check if it's a valid color specification
                    if self.is_valid_color(params['color']):
                        color_values = params['color']
                    else:
                        color_values = 'blue'  # Default color
            
            # Create the plot
            fig = px.scatter(
                df,
                x=params['x_column'],
                y=params['y_column'],
                size=marker_size if isinstance(marker_size, pd.Series) else None,
                color=color_values if isinstance(color_values, pd.Series) else None,
                color_continuous_scale=None if color_discrete else colormap,
                color_discrete_sequence=color_sequence if color_discrete else None,
                title=f'Bubble Plot: {params["y_column"]} vs {params["x_column"]}'
            )
            
            # Update markers for static values
            if not isinstance(marker_size, pd.Series):
                fig.update_traces(marker=dict(size=marker_size))
            if not isinstance(color_values, pd.Series) and color_values is not None:
                fig.update_traces(marker=dict(color=color_values))
            
            # Improve layout
            fig.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    title=params['x_column']
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    title=params['y_column']
                ),
                dragmode='pan',  # Enable panning
                modebar=dict(
                    orientation='h',
                    bgcolor='rgba(255,255,255,0.7)',
                    color='rgb(128,128,128)',
                    activecolor='rgb(50,50,50)'
                ),
                selectdirection='any',  # Enable selection in any direction
                clickmode='event+select',  # Enable both event and selection modes
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating bubble plot: {str(e)}")

class HeatmapPlot(VisualizationType):
    """Heatmap visualization type."""
    
    def __init__(self):
        super().__init__()
        self.required_params = {'columns'}
        self.optional_params = {'rows', 'standardize', 'cluster', 'colormap', 'transpose', 'fcol'}
    
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract heatmap parameters with regex support for rows/columns."""
        # Initialize params with default values
        params = {
            'transpose': False,
            'standardize': None,
            'cluster': None,
            'colormap': None,
            'fcol': None,
            'columns': None
        }
        
        # Define valid parameter names
        valid_params = {'columns', 'rows', 'standardize', 'cluster', 'colormap', 'transpose', 'fcol'}
        
        # Find all parameter assignments in the message
        param_matches = re.finditer(r'(\w+)=([^\s]+)', message)
        unknown_params = []
        
        for match in param_matches:
            param_name = match.group(1)
            if param_name not in valid_params:
                unknown_params.append(param_name)
        
        if unknown_params:
            return {}, f"Unknown parameter(s): {', '.join(unknown_params)}. Valid parameters are: {', '.join(sorted(valid_params))}"
        
        # Extract columns parameter with better error handling
        if 'columns=' in message:
            cols_match = re.search(r'columns=\[(.*?)\]', message)
            cols_regex_match = re.search(r'columns=(\S+)', message)
            
            if not (cols_match or cols_regex_match):
                return {}, "Malformed columns parameter. Use format: columns=[col1,col2,...] or columns=regex_pattern"
            
            if cols_match:
                cols = [c.strip() for c in cols_match.group(1).split(',')]
                if not cols:
                    return {}, "Empty column list provided"
                invalid_cols = [col for col in cols if col not in df.columns]
                if invalid_cols:
                    return {}, f"Column(s) not found in dataset: {', '.join(invalid_cols)}"
                params['columns'] = cols
            elif cols_regex_match:
                pattern = cols_regex_match.group(1)
                try:
                    regex = re.compile(pattern)
                    matching_cols = [col for col in df.columns if regex.search(col)]
                    if not matching_cols:
                        return {}, f"Column regex '{pattern}' matched no columns in dataset"
                    params['columns'] = matching_cols
                except re.error:
                    return {}, f"Invalid regex pattern for columns: '{pattern}'"
        else:
            params['columns'] = list(df.columns)  # Default to all columns
        
        # Extract rows parameter with regex and fcol support
        if 'rows=' in message:
            rows_match = re.search(r'rows=\[(.*?)\]', message)
            rows_regex_match = re.search(r'rows=(\S+)', message)
            
            if not (rows_match or rows_regex_match):
                return {}, "Malformed rows parameter. Use format: rows=[row1,row2,...] or rows=regex_pattern fcol=column_name"
            
            if rows_match:
                rows = [r.strip() for r in rows_match.group(1).split(',')]
                if not rows:
                    return {}, "Empty row list provided"
                invalid_rows = [row for row in rows if row not in df.columns]
                if invalid_rows:
                    return {}, f"Row(s) not found in dataset: {', '.join(invalid_rows)}"
                params['rows'] = rows
            else:  # Using regex pattern
                fcol_match = re.search(r'fcol=(\w+)', message)
                if not fcol_match:
                    return {}, "When using regex for rows, must specify fcol=column_name to filter on"
                
                fcol = fcol_match.group(1)
                if fcol not in df.columns:
                    return {}, f"Filter column '{fcol}' not found in dataset"
                
                pattern = rows_regex_match.group(1)
                try:
                    regex = re.compile(pattern)
                    # Filter the dataframe based on the regex pattern in fcol
                    filtered_indices = df[df[fcol].astype(str).str.match(regex)].index.tolist()
                    if not filtered_indices:
                        return {}, f"Row regex '{pattern}' matched no values in column '{fcol}'"
                    params['row_indices'] = filtered_indices
                    params['fcol'] = fcol
                except re.error:
                    return {}, f"Invalid regex pattern for rows: '{pattern}'"
        
        # Standardize parameter - strict validation
        std_match = re.search(r'standardize=(\w+)', message)
        if std_match:
            std_value = std_match.group(1).lower()
            if std_value not in ['rows', 'columns']:
                return {}, f"Invalid value for standardize: '{std_value}'. Must be 'rows' or 'columns'"
            params['standardize'] = std_value
        
        # Cluster parameter - strict validation
        cluster_match = re.search(r'cluster=(\w+)', message)
        if cluster_match:
            cluster_value = cluster_match.group(1).lower()
            if cluster_value not in ['rows', 'columns', 'both']:
                return {}, f"Invalid value for cluster: '{cluster_value}'. Must be 'rows', 'columns', or 'both'"
            params['cluster'] = cluster_value
        
        # Colormap parameter - strict validation with sorted options
        colormap_match = re.search(r'colormap=(\w+)', message)
        if colormap_match:
            colormap = colormap_match.group(1)
            valid_colormaps = sorted(px.colors.named_colorscales())
            if colormap not in valid_colormaps:
                return {}, f"Invalid colormap: '{colormap}'. Valid options (alphabetically):\n{', '.join(valid_colormaps)}"
            params['colormap'] = colormap
        
        # Transpose parameter - strict validation
        transpose_match = re.search(r'transpose=(\w+)', message)
        if transpose_match:
            transpose_value = transpose_match.group(1).lower()
            if transpose_value not in ['true', 'false']:
                return {}, f"Invalid value for transpose: '{transpose_value}'. Must be 'true' or 'false'"
            params['transpose'] = transpose_value == 'true'
        elif 'transpose' in message.lower():
            params['transpose'] = True
        
        return params, None

    def preprocess_data(self, df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, Optional[dict]]:
        """Prepare data for heatmap visualization."""
        # First filter the data if row_indices are provided
        if 'row_indices' in params:
            df = df.loc[params['row_indices']].reset_index(drop=True)  # Reset index after filtering
        
        # Make a copy to avoid modifying original data
        if params['columns'] is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data = df[numeric_cols].copy()
        else:
            data = df[params['columns']].copy()
            
        # Store original index values if they exist
        original_index = df.index
        
        # Handle missing values more robustly
        if data.isna().any().any():
            # For each column, fill NaN with column median
            for col in data.columns:
                data[col] = data[col].fillna(data[col].median())
            # Fill any remaining NaNs with 0
            data = data.fillna(0)
            print(f"Warning: Missing values were present and filled with median values")
        
        # Apply standardization before clustering
        if params.get('standardize') == 'rows':
            # Check for zero variance rows
            row_std = data.std(axis=1)
            zero_var_rows = row_std == 0
            if zero_var_rows.any():
                print(f"Warning: {zero_var_rows.sum()} rows had zero variance and were not standardized")
                data.loc[~zero_var_rows] = ((data.loc[~zero_var_rows].T - data.loc[~zero_var_rows].mean(axis=1)) / 
                                          data.loc[~zero_var_rows].std(axis=1)).T
        elif params.get('standardize') == 'columns':
            # Check for zero variance columns
            col_std = data.std()
            zero_var_cols = col_std == 0
            if zero_var_cols.any():
                print(f"Warning: {zero_var_cols.sum()} columns had zero variance and were not standardized")
                data.loc[:, ~zero_var_cols] = ((data.loc[:, ~zero_var_cols] - data.loc[:, ~zero_var_cols].mean()) / 
                                             data.loc[:, ~zero_var_cols].std())
        
        clustering_info = {
            'row_linkage': None,
            'col_linkage': None,
            'row_order': None,
            'col_order': None,
            'row_labels': data.index.tolist(),
            'col_labels': data.columns.tolist()
        }
        
        # Apply clustering if requested
        if params.get('cluster'):
            if params['cluster'] in ['rows', 'both']:
                try:
                    row_dist = pdist(data, metric='euclidean')
                    row_linkage = linkage(row_dist, method='ward')
                    row_linkage = optimal_leaf_ordering(row_linkage, row_dist)
                    row_order = leaves_list(row_linkage)
                    data = data.iloc[row_order]
                    clustering_info['row_linkage'] = row_linkage
                    clustering_info['row_order'] = row_order
                except Exception as e:
                    print(f"Warning: Row clustering failed: {str(e)}")
            
            if params['cluster'] in ['columns', 'both']:
                try:
                    col_dist = pdist(data.T, metric='euclidean')
                    col_linkage = linkage(col_dist, method='ward')
                    col_linkage = optimal_leaf_ordering(col_linkage, col_dist)
                    col_order = leaves_list(col_linkage)
                    data = data.iloc[:, col_order]
                    clustering_info['col_linkage'] = col_linkage
                    clustering_info['col_order'] = col_order
                except Exception as e:
                    print(f"Warning: Column clustering failed: {str(e)}")
        
        return data, clustering_info
    
    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create heatmap figure."""
        try:
            # Preprocess the data
            data, clustering_info = self.preprocess_data(df, params)
            
            # Determine if we should center the colorscale
            center_scale = params.get('standardize') is not None
            
            # Apply transpose if requested before creating the figure
            if params.get('transpose'):
                data = data.T
                
            # Calculate axis positions based on actual data size
            y_positions = np.arange(len(data))
            x_positions = np.arange(len(data.columns))
            
            # Create heatmap with final transformed data
            fig = go.Figure(data=go.Heatmap(
                z=data.values,
                x=data.columns,
                y=np.arange(len(data)),  # Use numeric positions for y
                colorscale=params.get('colormap', 'viridis'),
                zmid=0 if center_scale else None,
                text=np.round(data.values, decimals=2),
                texttemplate='%{text}',
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            # Improve layout
            fig.update_layout(
                title='Correlation Heatmap',
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis=dict(
                    title='',
                    showgrid=False,
                    tickangle=45,
                    tickmode='array',
                    ticktext=data.columns,
                    tickvals=x_positions,
                    automargin=True
                ),
                yaxis=dict(
                    title='',
                    showgrid=False,
                    tickmode='array',
                    ticktext=data.index,
                    tickvals=y_positions,
                    automargin=True,
                    side='left'
                ),
                width=800,
                height=800
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating heatmap: {str(e)}")

class GeoMap(VisualizationType):
    """Geographic map visualization type."""
    
    def __init__(self):
        super().__init__()
        self.required_params = {'latitude', 'longitude'}
        self.optional_params = {'size', 'color'}
    
    def extract_params(self, message: str, df: pd.DataFrame) -> Tuple[dict, str]:
        """Extract map parameters."""
        params = {}
        
        # Required parameters
        lat_match = re.search(r'latitude=(\w+)', message)
        lon_match = re.search(r'longitude=(\w+)', message)
        
        if not (lat_match and lon_match):
            return {}, "Map requires both latitude and longitude parameters"
        
        lat_col = lat_match.group(1)
        lon_col = lon_match.group(1)
        
        if lat_col not in df.columns or lon_col not in df.columns:
            return {}, f"Column not found: {lat_col if lat_col not in df.columns else lon_col}"
        
        params['latitude'] = lat_col
        params['longitude'] = lon_col
        
        # Optional parameters
        for param in ['color', 'size']:
            match = re.search(rf'{param}=(\w+)', message)
            if match:
                col = match.group(1)
                if col not in df.columns:
                    try:
                        # Try to convert to float for size parameter
                        if param == 'size':
                            float(col)
                            params[param] = col
                            continue
                    except ValueError:
                        pass
                    return {}, f"{param.capitalize()} column not found: {col}"
                params[param] = col
        
        return params, None

    def create_figure(self, params: dict, df: pd.DataFrame) -> go.Figure:
        """Create geographic map visualization."""
        try:
            # Validate required columns
            for param in ['latitude', 'longitude']:
                if param not in params:
                    raise Exception(f"Missing required parameter: {param}")
                if params[param] not in df.columns:
                    raise Exception(f"Column not found: {params[param]}")
                    
            # Extract coordinates
            lat = df[params['latitude']]
            lon = df[params['longitude']]
            
            # Handle invalid coordinates
            valid_coords = (
                lat.between(-90, 90) & 
                lon.between(-180, 180) & 
                lat.notna() & 
                lon.notna()
            )
            
            if not valid_coords.any():
                raise Exception("No valid coordinates found in data")
                
            if (~valid_coords).any():
                print(f"Warning: {(~valid_coords).sum()} invalid coordinates removed")
                df = df[valid_coords].copy()
                lat = lat[valid_coords]
                lon = lon[valid_coords]
            
            # Process size parameter
            if params.get('size'):
                if params['size'] in df.columns:
                    # Column-based size
                    size_values = df[params['size']]
                    if not pd.to_numeric(size_values, errors='coerce').notna().all():
                        raise Exception(f"Size column '{params['size']}' must contain numeric values")
                    # Scale sizes for better visualization
                    size_min, size_max = 10, 50  # Reasonable size range
                    marker_size = size_min + (size_max - size_min) * (
                        (size_values - size_values.min()) / 
                        (size_values.max() - size_values.min())
                    )
                else:
                    # Static size - try to convert to number
                    try:
                        marker_size = float(params['size'])
                    except (ValueError, TypeError):
                        marker_size = 15  # Default if conversion fails
            else:
                marker_size = 15  # Default size
            
            # Process color parameter
            if params.get('color'):
                if params['color'] in df.columns:
                    color_values = df[params['color']]
                    if pd.api.types.is_numeric_dtype(color_values):
                        # Numeric color scale
                        marker_color = color_values
                        colorscale = 'viridis'
                    else:
                        # Categorical colors
                        unique_values = color_values.unique()
                        color_sequence = self.generate_colors(len(unique_values))
                        color_map = dict(zip(unique_values, color_sequence))
                        marker_color = [color_map[val] for val in color_values]
                        colorscale = None
                else:
                    # Static color
                    marker_color = params['color'] if self.is_valid_color(params['color']) else 'blue'
                    colorscale = None
            else:
                marker_color = 'blue'
                colorscale = None
            
            # Create the map
            fig = go.Figure()
            
            # Add scatter mapbox trace
            scatter_kwargs = {
                'lat': lat,
                'lon': lon,
                'mode': 'markers',
                'marker': {
                    'size': marker_size,
                    'color': marker_color,
                },
                'text': [
                    f"Latitude: {lat:.4f}<br>"
                    f"Longitude: {lon:.4f}<br>"
                    + (f"{params['size']}: {size}<br>" if params.get('size') in df.columns else "")
                    + (f"{params['color']}: {color}<br>" if params.get('color') in df.columns else "")
                    for lat, lon, size, color in zip(
                        lat, lon,
                        df[params['size']] if params.get('size') in df.columns else [None] * len(lat),
                        df[params['color']] if params.get('color') in df.columns else [None] * len(lat)
                    )
                ],
                'hoverinfo': 'text'
            }
            
            # Add colorscale if using numeric or categorical colors
            if params.get('color') in df.columns:
                if pd.api.types.is_numeric_dtype(df[params['color']]):
                    scatter_kwargs['marker']['colorscale'] = colorscale
                    scatter_kwargs['marker']['colorbar'] = dict(
                        title=params['color'],
                        titleside='right',
                        thickness=20,
                        len=0.9,
                        x=0.02,  # Move colorbar to the left side
                        xpad=0
                    )
                    scatter_kwargs['marker']['showscale'] = True
                else:
                    # For categorical data, create a discrete color scale
                    unique_values = sorted(df[params['color']].unique())
                    color_sequence = self.generate_colors(len(unique_values))
                    color_map = dict(zip(unique_values, color_sequence))
                    scatter_kwargs['marker']['color'] = [color_map[val] for val in df[params['color']]]
                    # Add a legend instead of colorbar for categorical data
                    scatter_kwargs['showlegend'] = True
                    scatter_kwargs['name'] = params['color']  # This will show in the legend
                    # Create separate traces for legend entries
                    for val, color in color_map.items():
                        fig.add_trace(go.Scattermapbox(
                            lat=[None],
                            lon=[None],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=str(val),
                            showlegend=True,
                            hoverinfo='skip'
                        ))
                
            fig.add_trace(go.Scattermapbox(**scatter_kwargs))
            
            # Update layout for mapbox
            center_lat = lat.mean()
            center_lon = lon.mean()
            
            # Calculate zoom based on coordinate spread
            lat_range = lat.max() - lat.min()
            lon_range = lon.max() - lon.min()
            # Adjust zoom calculation for tighter view
            # Use log scale to handle different ranges more gracefully
            zoom = 8.5 - np.log2(max(lat_range, lon_range) + 0.0001)  # Much smaller offset for geographic coordinates
            zoom = max(1, min(20, zoom))  # Ensure zoom is within valid range
            
            fig.update_layout(
                mapbox=dict(
                    style='carto-positron',  # Light, clean map style
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=zoom
                ),
                margin=dict(l=0, r=0, t=30, b=0),
                title=dict(
                    text="Geographic Distribution",
                    x=0.5,
                    xanchor='center'
                ),
                dragmode='pan',  # Enable panning
                modebar=dict(
                    orientation='h',
                    bgcolor='rgba(255,255,255,0.7)',
                    color='rgb(128,128,128)',
                    activecolor='rgb(50,50,50)'
                ),
                selectdirection='any',  # Enable selection in any direction
                clickmode='event+select',  # Enable both event and selection modes
                hovermode='closest'
            )
            
            return fig
            
        except Exception as e:
            raise Exception(f"Error creating map: {str(e)}")

class VisualizationService(ChatService):
    """Service for handling data visualization requests.
    
    This service provides a standardized interface for creating and managing
    data visualizations in the ChatDash application.
    """
    
    def __init__(self):
        super().__init__("visualization")
        self.viz_types = {
            'bubble': BubblePlot(),
            'heatmap': HeatmapPlot(),
            'map': GeoMap()
        }
    
    def can_handle(self, message: str) -> bool:
        """Detect visualization requests.
        
        Currently detects:
        - plot commands (e.g., 'plot x vs y')
        - map commands (with lat/long parameters)
        - heatmap commands
        """
        message = message.lower().strip()
        return any(message.startswith(cmd) for cmd in ['plot', 'map', 'heatmap'])

    def parse_request(self, message: str) -> Dict[str, Any]:
        """Extract visualization parameters from the message.
        
        Returns:
            Dict containing:
            - type: visualization type
            - message: original message for testing
        """
        # Only use lowercase for type detection, preserve original message for parameters
        message_lower = message.lower().strip()
        
        # Determine visualization type
        if message_lower.startswith('plot'):
            viz_type = 'bubble'
        elif message_lower.startswith('heatmap'):
            viz_type = 'heatmap'
        elif message_lower.startswith('map'):
            viz_type = 'map'
        else:
            viz_type = 'unknown'
            
        return {
            'type': viz_type,
            'message': message.strip()  # Preserve original case
        }
    
    def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> ServiceResponse:
        """Execute visualization request.
        
        Args:
            params: Parameters extracted from message
            context: Current application context
            
        Returns:
            ServiceResponse with visualization result
        """
        # Check for selected dataset
        selected_dataset = context.get('selected_dataset')
        datasets = context.get('datasets_store', {})
        
        if not datasets:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No datasets are currently loaded. Please load a dataset first.",
                    message_type="error"
                )],
                context=None
            )
            
        if not selected_dataset:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content="No dataset is selected. Please select a dataset first.",
                    message_type="error"
                )],
                context=None
            )
            
        # Get the visualization type
        viz_type = self.viz_types.get(params['type'])
        if not viz_type:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Visualization type '{params['type']}' not implemented yet.",
                    message_type="error"
                )],
                context=None
            )
            
        # Get the dataset
        df = pd.DataFrame(datasets[selected_dataset]['df'])
        
        # Extract parameters for the visualization
        viz_params, error = viz_type.extract_params(params['message'], df)
        if error:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error parsing visualization parameters: {error}",
                    message_type="error"
                )],
                context=None
            )
            
        # Create the visualization
        try:
            fig = viz_type.create_figure(viz_params, df)
            
            # Apply any stored view settings
            if 'viz_state' in context and 'view_settings' in context['viz_state']:
                fig = viz_type.apply_view_settings(fig, context['viz_state']['view_settings'])
            
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Creating {params['type']} visualization. Switching to visualization tab.",
                    message_type="info"
                )],
                context=None,
                state_updates={
                    'active_tab': 'tab-viz',
                    'viz_state': {
                        'type': params['type'],
                        'params': viz_params,
                        'df': df.to_dict('records'),
                        'view_settings': context.get('viz_state', {}).get('view_settings', {}),
                        'figure': fig.to_dict()  # Add the figure to the state
                    }
                }
            )
            
        except Exception as e:
            return ServiceResponse(
                messages=[ServiceMessage(
                    service=self.name,
                    content=f"Error creating visualization: {str(e)}",
                    message_type="error"
                )],
                context=None
            ) 