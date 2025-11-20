"""
Map Interface Component for Solar PV Layout Generator
Provides interactive map with drawing tools using Folium and Streamlit-Folium
"""

import folium
from folium import plugins
import streamlit as st
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Polygon, Point
from typing import List, Dict, Optional, Tuple
import json


class MapInterface:
    """Interactive map interface for drawing site boundaries and displaying solar layouts"""

    def __init__(self):
        self.map = None
        self.drawn_shapes = []
        self.module_layer = None
        self.shadow_layer = None

    def render_map(
        self,
        center: Tuple[float, float] = (37.7749, -122.4194),
        zoom: int = 13,
        height: int = 600,
        key: str = "map"
    ) -> Dict:
        """
        Render an interactive Folium map with drawing tools

        Parameters:
        -----------
        center : tuple
            (latitude, longitude) for map center
        zoom : int
            Initial zoom level
        height : int
            Map height in pixels
        key : str
            Unique key for streamlit component

        Returns:
        --------
        dict : Map data including drawn features
        """
        # Create base map
        self.map = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )

        # Add satellite imagery layer
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(self.map)

        # Add layer control
        folium.LayerControl().add_to(self.map)

        # Add drawing tools
        self.add_drawing_tools()

        # Add measurement tool
        plugins.MeasureControl(
            position='topleft',
            primary_length_unit='meters',
            secondary_length_unit='kilometers',
            primary_area_unit='sqmeters',
            secondary_area_unit='hectares'
        ).add_to(self.map)

        # Add fullscreen option
        plugins.Fullscreen(
            position='topright',
            title='Expand map',
            title_cancel='Exit fullscreen',
            force_separate_button=True
        ).add_to(self.map)

        # Add geocoder for location search
        plugins.Geocoder(
            collapsed=False,
            position='topleft',
            placeholder='Search for location...'
        ).add_to(self.map)

        # Render map in Streamlit
        map_data = st_folium(
            self.map,
            height=height,
            width=None,
            key=key,
            returned_objects=["all_drawings", "last_active_drawing"]
        )

        return map_data

    def add_drawing_tools(self):
        """Add drawing tools to the map"""
        # Create draw plugin
        draw = plugins.Draw(
            export=True,
            position='topleft',
            draw_options={
                'polyline': False,
                'rectangle': {
                    'shapeOptions': {
                        'color': '#ff7800',
                        'weight': 3,
                        'fillOpacity': 0.3
                    }
                },
                'polygon': {
                    'allowIntersection': False,
                    'drawError': {
                        'color': '#e1e100',
                        'message': 'Polygon edges cannot intersect!'
                    },
                    'shapeOptions': {
                        'color': '#ff7800',
                        'weight': 3,
                        'fillOpacity': 0.3
                    }
                },
                'circle': False,
                'marker': False,
                'circlemarker': False
            },
            edit_options={
                'edit': True,
                'remove': True
            }
        )

        draw.add_to(self.map)

    def get_drawn_shapes(self, map_data: Dict) -> List[Polygon]:
        """
        Extract polygon shapes from map drawing data

        Parameters:
        -----------
        map_data : dict
            Data returned from st_folium

        Returns:
        --------
        list : List of Shapely Polygon objects
        """
        polygons = []

        if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
            for feature in map_data['all_drawings']:
                if feature['geometry']['type'] in ['Polygon', 'Rectangle']:
                    coords = feature['geometry']['coordinates'][0]
                    # Convert from [lon, lat] to [lat, lon] for Shapely
                    coords_latlon = [(coord[1], coord[0]) for coord in coords]
                    polygon = Polygon(coords_latlon)
                    polygons.append(polygon)

        return polygons

    def add_module_layer(
        self,
        layout_gdf: gpd.GeoDataFrame,
        color: str = '#3388ff',
        opacity: float = 0.7
    ):
        """
        Add solar module layout as a layer on the map

        Parameters:
        -----------
        layout_gdf : GeoDataFrame
            GeoDataFrame containing module geometries
        color : str
            Color for module polygons
        opacity : float
            Fill opacity for modules
        """
        if self.map is None:
            raise ValueError("Map must be rendered before adding layers")

        # Create feature group for modules
        module_group = folium.FeatureGroup(name='Solar Modules', show=True)

        for idx, row in layout_gdf.iterrows():
            # Get geometry
            geom = row.geometry

            # Convert to GeoJSON-like structure
            if geom.geom_type == 'Polygon':
                coords = [[coord[1], coord[0]] for coord in geom.exterior.coords]

                # Create tooltip with module info
                tooltip_text = f"""
                Module ID: {idx}<br>
                Power: {row.get('power_watts', 'N/A')} W<br>
                Position: ({coords[0][0]:.6f}, {coords[0][1]:.6f})
                """

                folium.Polygon(
                    locations=coords,
                    color=color,
                    weight=1,
                    fill=True,
                    fillColor=color,
                    fillOpacity=opacity,
                    tooltip=tooltip_text,
                    popup=folium.Popup(tooltip_text, max_width=300)
                ).add_to(module_group)

        module_group.add_to(self.map)

    def add_shadow_overlay(
        self,
        shadow_gdf: gpd.GeoDataFrame,
        colormap: str = 'YlOrRd'
    ):
        """
        Add shadow analysis heatmap overlay

        Parameters:
        -----------
        shadow_gdf : GeoDataFrame
            GeoDataFrame with shadow intensity values
        colormap : str
            Matplotlib colormap name
        """
        if self.map is None:
            raise ValueError("Map must be rendered before adding layers")

        # Create feature group for shadows
        shadow_group = folium.FeatureGroup(name='Shadow Analysis', show=True)

        # Normalize shadow values for colormap
        if 'shadow_intensity' in shadow_gdf.columns:
            min_val = shadow_gdf['shadow_intensity'].min()
            max_val = shadow_gdf['shadow_intensity'].max()

            for idx, row in shadow_gdf.iterrows():
                geom = row.geometry
                intensity = row['shadow_intensity']

                # Normalize to 0-1
                normalized = (intensity - min_val) / (max_val - min_val) if max_val > min_val else 0

                # Convert to color (red for high shadow, yellow for low)
                color = self._get_color_from_value(normalized, colormap)

                if geom.geom_type == 'Polygon':
                    coords = [[coord[1], coord[0]] for coord in geom.exterior.coords]

                    tooltip_text = f"Shadow Intensity: {intensity:.2%}"

                    folium.Polygon(
                        locations=coords,
                        color=color,
                        weight=0.5,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.5,
                        tooltip=tooltip_text
                    ).add_to(shadow_group)

        shadow_group.add_to(self.map)

    def add_boundary_highlight(
        self,
        boundary_polygon: Polygon,
        color: str = '#ff7800',
        weight: int = 3
    ):
        """
        Highlight the site boundary on the map

        Parameters:
        -----------
        boundary_polygon : Polygon
            Shapely polygon of site boundary
        color : str
            Border color
        weight : int
            Border weight
        """
        if self.map is None:
            raise ValueError("Map must be rendered before adding layers")

        coords = [[coord[1], coord[0]] for coord in boundary_polygon.exterior.coords]

        folium.Polygon(
            locations=coords,
            color=color,
            weight=weight,
            fill=True,
            fillColor=color,
            fillOpacity=0.1,
            popup='Site Boundary'
        ).add_to(self.map)

    @staticmethod
    def _get_color_from_value(value: float, colormap: str = 'YlOrRd') -> str:
        """
        Convert normalized value (0-1) to hex color

        Parameters:
        -----------
        value : float
            Normalized value between 0 and 1
        colormap : str
            Matplotlib colormap name

        Returns:
        --------
        str : Hex color string
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        cmap = cm.get_cmap(colormap)
        rgb = cmap(value)[:3]
        return mcolors.rgb2hex(rgb)

    def export_to_geojson(
        self,
        geometry_list: List[Polygon],
        properties_list: Optional[List[Dict]] = None
    ) -> str:
        """
        Export geometries to GeoJSON format

        Parameters:
        -----------
        geometry_list : list
            List of Shapely geometries
        properties_list : list, optional
            List of property dictionaries for each geometry

        Returns:
        --------
        str : GeoJSON string
        """
        features = []

        for i, geom in enumerate(geometry_list):
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': geom.geom_type,
                    'coordinates': [[[coord[1], coord[0]] for coord in geom.exterior.coords]]
                },
                'properties': properties_list[i] if properties_list else {'id': i}
            }
            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        return json.dumps(geojson, indent=2)
