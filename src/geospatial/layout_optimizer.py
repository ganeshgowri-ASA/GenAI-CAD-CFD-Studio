"""
Solar PV Layout Optimizer
Generates optimal solar panel layouts within site boundaries
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate, translate
from typing import List, Tuple, Dict, Optional
import pandas as pd
from datetime import datetime
import math


class LayoutOptimizer:
    """Optimizes solar panel layout within site boundaries"""

    def __init__(
        self,
        module_width: float,
        module_length: float,
        module_power: int,
        row_spacing: float = 1.0,
        column_spacing: float = 0.02,
        tilt_angle: float = 20.0,
        azimuth: float = 180.0
    ):
        """
        Initialize layout optimizer

        Parameters:
        -----------
        module_width : float
            Module width in meters
        module_length : float
            Module length in meters
        module_power : int
            Module power in watts
        row_spacing : float
            Spacing between rows in meters
        column_spacing : float
            Spacing between columns in meters
        tilt_angle : float
            Module tilt angle in degrees
        azimuth : float
            Module azimuth in degrees (0=N, 90=E, 180=S, 270=W)
        """
        self.module_width = module_width
        self.module_length = module_length
        self.module_power = module_power
        self.row_spacing = row_spacing
        self.column_spacing = column_spacing
        self.tilt_angle = tilt_angle
        self.azimuth = azimuth

        # Calculate effective row spacing (accounting for tilt)
        self.effective_row_spacing = self._calculate_effective_spacing()

    def _calculate_effective_spacing(self) -> float:
        """Calculate effective row spacing accounting for tilt angle"""
        # Shadow projection increases with tilt
        tilt_rad = math.radians(self.tilt_angle)
        shadow_projection = self.module_length * math.sin(tilt_rad)
        return self.row_spacing + shadow_projection

    def generate_layout(
        self,
        boundary: Polygon,
        orientation: str = 'landscape'
    ) -> gpd.GeoDataFrame:
        """
        Generate solar module layout within boundary

        Parameters:
        -----------
        boundary : Polygon
            Site boundary polygon
        orientation : str
            Module orientation: 'landscape' or 'portrait'

        Returns:
        --------
        GeoDataFrame : Module layout with geometries and attributes
        """
        # Get boundary bounds
        minx, miny, maxx, maxy = boundary.bounds

        # Adjust module dimensions based on orientation
        if orientation == 'portrait':
            width, length = self.module_length, self.module_width
        else:
            width, length = self.module_width, self.module_length

        # Calculate grid dimensions
        total_width = width + self.column_spacing
        total_length = length + self.effective_row_spacing

        # Generate module positions
        modules = []
        module_id = 0

        # Calculate rotation angle (azimuth offset from south)
        rotation_angle = self.azimuth - 180  # Relative to south (180°)

        y = miny
        row = 0

        while y + length <= maxy:
            x = minx
            col = 0

            while x + width <= maxx:
                # Create module rectangle
                module_rect = box(x, y, x + width, y + length)

                # Rotate if needed
                if rotation_angle != 0:
                    # Rotate around center
                    center_x = x + width / 2
                    center_y = y + length / 2
                    module_rect = rotate(
                        module_rect,
                        rotation_angle,
                        origin=(center_x, center_y)
                    )

                # Check if module is within boundary
                if boundary.contains(module_rect):
                    modules.append({
                        'module_id': module_id,
                        'row': row,
                        'column': col,
                        'geometry': module_rect,
                        'power_watts': self.module_power,
                        'tilt_angle': self.tilt_angle,
                        'azimuth': self.azimuth,
                        'area_m2': width * length
                    })
                    module_id += 1

                x += total_width
                col += 1

            y += total_length
            row += 1

        # Create GeoDataFrame
        if modules:
            gdf = gpd.GeoDataFrame(modules, crs='EPSG:4326')
        else:
            # Empty GeoDataFrame with correct schema
            gdf = gpd.GeoDataFrame(
                columns=['module_id', 'row', 'column', 'geometry', 'power_watts',
                        'tilt_angle', 'azimuth', 'area_m2'],
                crs='EPSG:4326'
            )

        return gdf

    def calculate_statistics(
        self,
        layout_gdf: gpd.GeoDataFrame,
        boundary: Polygon
    ) -> Dict:
        """
        Calculate layout statistics

        Parameters:
        -----------
        layout_gdf : GeoDataFrame
            Module layout
        boundary : Polygon
            Site boundary

        Returns:
        --------
        dict : Statistics dictionary
        """
        if len(layout_gdf) == 0:
            return {
                'total_modules': 0,
                'total_capacity_kw': 0.0,
                'total_module_area_m2': 0.0,
                'site_area_m2': boundary.area,
                'coverage_percentage': 0.0,
                'modules_per_row': 0,
                'number_of_rows': 0,
                'avg_power_density_w_m2': 0.0
            }

        total_modules = len(layout_gdf)
        total_capacity_w = layout_gdf['power_watts'].sum()
        total_capacity_kw = total_capacity_w / 1000
        total_module_area = layout_gdf['area_m2'].sum()
        site_area = boundary.area
        coverage_percentage = (total_module_area / site_area) * 100

        # Calculate rows and columns
        modules_per_row = layout_gdf.groupby('row').size().mean()
        number_of_rows = layout_gdf['row'].max() + 1

        # Power density
        avg_power_density = total_capacity_w / total_module_area if total_module_area > 0 else 0

        return {
            'total_modules': total_modules,
            'total_capacity_kw': total_capacity_kw,
            'total_module_area_m2': total_module_area,
            'site_area_m2': site_area,
            'coverage_percentage': coverage_percentage,
            'modules_per_row': modules_per_row,
            'number_of_rows': number_of_rows,
            'avg_power_density_w_m2': avg_power_density
        }


class ShadowAnalyzer:
    """Analyzes shadow patterns for solar installations"""

    def __init__(self, latitude: float, longitude: float):
        """
        Initialize shadow analyzer

        Parameters:
        -----------
        latitude : float
            Site latitude
        longitude : float
            Site longitude
        """
        self.latitude = latitude
        self.longitude = longitude

    def calculate_sun_position(
        self,
        timestamp: datetime
    ) -> Tuple[float, float]:
        """
        Calculate sun position (altitude and azimuth) at given time

        Parameters:
        -----------
        timestamp : datetime
            Time for sun position calculation

        Returns:
        --------
        tuple : (altitude, azimuth) in degrees
        """
        # Simplified sun position calculation
        # For production, use pvlib or pysolar library

        # Calculate day of year
        day_of_year = timestamp.timetuple().tm_yday

        # Calculate hour angle
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_angle = (hour - 12) * 15  # degrees

        # Calculate declination (simplified)
        declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

        # Calculate solar altitude
        lat_rad = math.radians(self.latitude)
        dec_rad = math.radians(declination)
        ha_rad = math.radians(hour_angle)

        sin_altitude = (
            math.sin(lat_rad) * math.sin(dec_rad) +
            math.cos(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
        )
        altitude = math.degrees(math.asin(max(-1, min(1, sin_altitude))))

        # Calculate solar azimuth
        cos_azimuth = (
            (math.sin(dec_rad) - math.sin(lat_rad) * math.sin(math.radians(altitude))) /
            (math.cos(lat_rad) * math.cos(math.radians(altitude)))
        )
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.degrees(math.acos(cos_azimuth))

        # Adjust azimuth for afternoon
        if hour > 12:
            azimuth = 360 - azimuth

        return altitude, azimuth

    def analyze_shadows(
        self,
        layout_gdf: gpd.GeoDataFrame,
        timestamp: datetime,
        grid_resolution: float = 1.0
    ) -> gpd.GeoDataFrame:
        """
        Analyze shadow patterns for module layout

        Parameters:
        -----------
        layout_gdf : GeoDataFrame
            Module layout
        timestamp : datetime
            Time for shadow analysis
        grid_resolution : float
            Grid cell size in meters

        Returns:
        --------
        GeoDataFrame : Shadow intensity grid
        """
        if len(layout_gdf) == 0:
            return gpd.GeoDataFrame(columns=['geometry', 'shadow_intensity'], crs='EPSG:4326')

        # Get sun position
        altitude, azimuth = self.calculate_sun_position(timestamp)

        # If sun is below horizon, everything is in shadow
        if altitude <= 0:
            shadow_intensity = 1.0  # Full shadow
        else:
            # Calculate shadow intensity based on sun altitude
            # Lower sun = longer shadows = higher intensity
            shadow_intensity = max(0, 1 - (altitude / 90))

        # Get layout bounds
        bounds = layout_gdf.total_bounds
        minx, miny, maxx, maxy = bounds

        # Create shadow grid
        shadow_cells = []

        x = minx
        while x < maxx:
            y = miny
            while y < maxy:
                cell = box(x, y, x + grid_resolution, y + grid_resolution)

                # Check if cell intersects with any module
                intersects = layout_gdf.intersects(cell).any()

                if intersects:
                    shadow_cells.append({
                        'geometry': cell,
                        'shadow_intensity': shadow_intensity,
                        'sun_altitude': altitude,
                        'sun_azimuth': azimuth
                    })

                y += grid_resolution
            x += grid_resolution

        # Create GeoDataFrame
        if shadow_cells:
            shadow_gdf = gpd.GeoDataFrame(shadow_cells, crs='EPSG:4326')
        else:
            shadow_gdf = gpd.GeoDataFrame(
                columns=['geometry', 'shadow_intensity', 'sun_altitude', 'sun_azimuth'],
                crs='EPSG:4326'
            )

        return shadow_gdf

    def generate_shadow_report(
        self,
        layout_gdf: gpd.GeoDataFrame,
        start_time: datetime,
        end_time: datetime,
        time_step_hours: int = 1
    ) -> Dict:
        """
        Generate comprehensive shadow analysis report

        Parameters:
        -----------
        layout_gdf : GeoDataFrame
            Module layout
        start_time : datetime
            Start time for analysis
        end_time : datetime
            End time for analysis
        time_step_hours : int
            Time step in hours

        Returns:
        --------
        dict : Shadow analysis report
        """
        # Time range
        current_time = start_time
        shadow_data = []

        while current_time <= end_time:
            altitude, azimuth = self.calculate_sun_position(current_time)

            shadow_data.append({
                'timestamp': current_time,
                'sun_altitude': altitude,
                'sun_azimuth': azimuth,
                'is_daylight': altitude > 0
            })

            current_time = current_time.replace(hour=current_time.hour + time_step_hours)

        # Create DataFrame
        df = pd.DataFrame(shadow_data)

        # Calculate statistics
        daylight_hours = df[df['is_daylight']]['is_daylight'].count()
        avg_altitude = df[df['is_daylight']]['sun_altitude'].mean()

        report = {
            'analysis_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'time_step_hours': time_step_hours
            },
            'sun_statistics': {
                'total_hours_analyzed': len(df),
                'daylight_hours': int(daylight_hours),
                'avg_sun_altitude': float(avg_altitude) if not pd.isna(avg_altitude) else 0.0,
                'max_sun_altitude': float(df['sun_altitude'].max()),
                'min_sun_altitude': float(df['sun_altitude'].min())
            },
            'shadow_timeline': df.to_dict('records')
        }

        return report
