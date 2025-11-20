"""
Shadow analysis for Solar PV installations.

Provides sun position calculations and shadow projection analysis
for optimizing solar panel placement and energy production estimation.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.collections import PatchCollection


class ShadowAnalyzer:
    """
    Analyzes shadows cast by solar modules and obstacles.

    Provides sun position calculations and shadow projection for
    time-series analysis of shading effects on solar installations.
    """

    def __init__(self):
        """Initialize the ShadowAnalyzer."""
        pass

    def calculate_sun_position(
        self,
        lat: float,
        lon: float,
        date: datetime,
        time: datetime
    ) -> Tuple[float, float]:
        """
        Calculate sun position (azimuth and elevation) for a given location and time.

        Uses a simplified solar position algorithm. For production use,
        consider using pvlib for higher accuracy.

        Args:
            lat: Latitude in degrees (negative for South)
            lon: Longitude in degrees (negative for West)
            date: Date for calculation
            time: Time for calculation (can be same datetime object)

        Returns:
            Tuple of (azimuth, elevation) in degrees
            - azimuth: 0° = North, 90° = East, 180° = South, 270° = West
            - elevation: 0° = horizon, 90° = zenith

        Example:
            >>> analyzer = ShadowAnalyzer()
            >>> dt = datetime(2024, 6, 21, 12, 0)
            >>> azimuth, elevation = analyzer.calculate_sun_position(
            ...     40.7128, -74.0060, dt, dt
            ... )
        """
        # Combine date and time if they're separate
        if date.date() != time.date() or date.time() != time.time():
            dt = datetime.combine(date.date(), time.time())
        else:
            dt = time

        # Calculate Julian day
        jd = self._datetime_to_julian(dt)

        # Calculate number of days since J2000.0
        n = jd - 2451545.0

        # Mean longitude of the Sun
        L = (280.460 + 0.9856474 * n) % 360

        # Mean anomaly
        g = (357.528 + 0.9856003 * n) % 360
        g_rad = np.radians(g)

        # Ecliptic longitude
        lambda_sun = L + 1.915 * np.sin(g_rad) + 0.020 * np.sin(2 * g_rad)
        lambda_rad = np.radians(lambda_sun)

        # Obliquity of the ecliptic
        epsilon = 23.439 - 0.0000004 * n
        epsilon_rad = np.radians(epsilon)

        # Right ascension
        alpha = np.degrees(
            np.arctan2(np.cos(epsilon_rad) * np.sin(lambda_rad), np.cos(lambda_rad))
        )

        # Declination
        delta = np.degrees(
            np.arcsin(np.sin(epsilon_rad) * np.sin(lambda_rad))
        )
        delta_rad = np.radians(delta)

        # Local hour angle
        # Greenwich hour angle
        gha = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360

        # Local hour angle
        lha = (gha + lon - alpha) % 360
        if lha > 180:
            lha -= 360
        lha_rad = np.radians(lha)

        # Latitude in radians
        lat_rad = np.radians(lat)

        # Calculate elevation (altitude)
        sin_elevation = (
            np.sin(lat_rad) * np.sin(delta_rad) +
            np.cos(lat_rad) * np.cos(delta_rad) * np.cos(lha_rad)
        )
        elevation = np.degrees(np.arcsin(sin_elevation))

        # Calculate azimuth
        cos_azimuth = (
            (np.sin(delta_rad) - np.sin(lat_rad) * sin_elevation) /
            (np.cos(lat_rad) * np.cos(np.radians(elevation)))
        )
        # Clamp to valid range
        cos_azimuth = np.clip(cos_azimuth, -1.0, 1.0)

        azimuth = np.degrees(np.arccos(cos_azimuth))

        # Adjust azimuth based on hour angle
        if lha > 0:
            azimuth = 360 - azimuth

        return azimuth, elevation

    def _datetime_to_julian(self, dt: datetime) -> float:
        """
        Convert datetime to Julian day.

        Args:
            dt: Datetime object

        Returns:
            Julian day number
        """
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3

        jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

        # Add fractional day
        jd = jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0

        return jd

    def project_shadows(
        self,
        modules: gpd.GeoDataFrame,
        sun_azimuth: float,
        sun_elevation: float,
        height: float = 2.0,
        tilt_angle: float = 0.0
    ) -> List[Polygon]:
        """
        Project shadows from modules based on sun position.

        Args:
            modules: GeoDataFrame of module layouts
            sun_azimuth: Sun azimuth angle in degrees (0° = North)
            sun_elevation: Sun elevation angle in degrees (0° = horizon)
            height: Height of modules above ground in meters
            tilt_angle: Tilt angle of modules in degrees (0° = flat)

        Returns:
            List of Shapely polygons representing shadow areas

        Note:
            Shadows are projected onto the ground plane (z=0).
            Higher tilt angles create longer shadows.
        """
        if sun_elevation <= 0:
            # Sun below horizon - no meaningful shadows or complete darkness
            return []

        shadows = []

        # Convert angles to radians
        azimuth_rad = np.radians(sun_azimuth)
        elevation_rad = np.radians(sun_elevation)
        tilt_rad = np.radians(tilt_angle)

        # Calculate shadow length
        # Effective height considering tilt
        effective_height = height + height * np.sin(tilt_rad)
        shadow_length = effective_height / np.tan(elevation_rad)

        # Shadow direction (opposite to sun azimuth)
        shadow_direction = (sun_azimuth + 180) % 360
        shadow_dir_rad = np.radians(shadow_direction)

        # Calculate shadow offset
        dx = shadow_length * np.sin(shadow_dir_rad)
        dy = shadow_length * np.cos(shadow_dir_rad)

        for idx, row in modules.iterrows():
            module_geom = row.geometry

            # Get module coordinates
            coords = list(module_geom.exterior.coords)

            # Project each coordinate to create shadow
            shadow_coords = [(x + dx, y + dy) for x, y in coords]

            # Create shadow polygon (union of module and projected area)
            original_coords = list(coords)
            all_coords = original_coords + shadow_coords

            try:
                shadow_polygon = Polygon(all_coords).convex_hull
                shadows.append(shadow_polygon)
            except Exception:
                # Skip invalid shadows
                continue

        return shadows

    def analyze_shading(
        self,
        layout: gpd.GeoDataFrame,
        lat: float,
        lon: float,
        date_range: Tuple[datetime, datetime],
        time_steps: int = 24,
        height: float = 2.0,
        tilt_angle: float = 0.0
    ) -> Dict[int, float]:
        """
        Analyze shading over a time period.

        Args:
            layout: GeoDataFrame of module layout
            lat: Latitude in degrees
            lon: Longitude in degrees
            date_range: Tuple of (start_date, end_date)
            time_steps: Number of time steps per day
            height: Module height above ground in meters
            tilt_angle: Module tilt angle in degrees

        Returns:
            Dictionary mapping module_id to total shadow hours

        Example:
            >>> start = datetime(2024, 6, 1)
            >>> end = datetime(2024, 6, 30)
            >>> shading_map = analyzer.analyze_shading(
            ...     layout, 40.7, -74.0, (start, end), time_steps=12
            ... )
        """
        start_date, end_date = date_range

        # Initialize shadow hours counter for each module
        shadow_hours = {module_id: 0.0 for module_id in layout['module_id']}

        # Calculate time delta
        total_hours = (end_date - start_date).total_seconds() / 3600
        hours_per_step = 24.0 / time_steps
        total_steps = int(total_hours / hours_per_step)

        current_time = start_date

        for step in range(total_steps):
            # Calculate sun position
            azimuth, elevation = self.calculate_sun_position(
                lat, lon, current_time, current_time
            )

            # Only calculate shadows when sun is above horizon
            if elevation > 0:
                # Project shadows
                shadows = self.project_shadows(
                    layout, azimuth, elevation, height, tilt_angle
                )

                # Check each module for shadow coverage
                for idx, row in layout.iterrows():
                    module_id = row.module_id
                    module_geom = row.geometry

                    # Check intersection with any shadow
                    for shadow in shadows:
                        if module_geom.intersects(shadow):
                            # Calculate overlap percentage
                            overlap = module_geom.intersection(shadow)
                            overlap_ratio = overlap.area / module_geom.area

                            # Add shadow time proportional to coverage
                            shadow_hours[module_id] += hours_per_step * overlap_ratio
                            break  # Count only once per time step

            current_time += timedelta(hours=hours_per_step)

        return shadow_hours

    def visualize_shadows(
        self,
        layout: gpd.GeoDataFrame,
        shadows: List[Polygon],
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize module layout and shadows.

        Args:
            layout: GeoDataFrame of module layout
            shadows: List of shadow polygons
            figsize: Figure size (width, height) in inches
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object

        Example:
            >>> fig = analyzer.visualize_shadows(layout, shadows)
            >>> plt.show()
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Plot modules
        layout.plot(ax=ax, color='blue', alpha=0.6, edgecolor='black', label='Modules')

        # Plot shadows
        if shadows:
            shadow_gdf = gpd.GeoDataFrame(geometry=shadows)
            shadow_gdf.plot(ax=ax, color='gray', alpha=0.4, edgecolor='darkgray', label='Shadows')

        # Add labels and legend
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('Solar Module Layout with Shadow Analysis', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def calculate_solar_irradiance_factor(
        self,
        shadow_hours: Dict[int, float],
        total_hours: float
    ) -> Dict[int, float]:
        """
        Calculate solar irradiance factor for each module.

        Args:
            shadow_hours: Dictionary of shadow hours per module
            total_hours: Total analysis period in hours

        Returns:
            Dictionary mapping module_id to irradiance factor (0-1)
            where 1 = no shading, 0 = completely shadowed

        Example:
            >>> factors = analyzer.calculate_solar_irradiance_factor(
            ...     shadow_hours, total_hours=720
            ... )
        """
        irradiance_factors = {}

        for module_id, shaded_hours in shadow_hours.items():
            if total_hours > 0:
                # Factor is percentage of time not in shadow
                factor = 1.0 - (shaded_hours / total_hours)
                irradiance_factors[module_id] = max(0.0, min(1.0, factor))
            else:
                irradiance_factors[module_id] = 1.0

        return irradiance_factors

    def get_sun_path(
        self,
        lat: float,
        lon: float,
        date: datetime,
        time_steps: int = 48
    ) -> List[Tuple[float, float, datetime]]:
        """
        Calculate sun path for a given day.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            date: Date for sun path calculation
            time_steps: Number of points in the sun path

        Returns:
            List of tuples (azimuth, elevation, time) for the day

        Example:
            >>> sun_path = analyzer.get_sun_path(40.7, -74.0, datetime(2024, 6, 21))
        """
        sun_path = []

        start_time = datetime.combine(date.date(), datetime.min.time())
        hours_per_step = 24.0 / time_steps

        for step in range(time_steps):
            current_time = start_time + timedelta(hours=step * hours_per_step)
            azimuth, elevation = self.calculate_sun_position(
                lat, lon, current_time, current_time
            )

            sun_path.append((azimuth, elevation, current_time))

        return sun_path

    def estimate_energy_loss(
        self,
        shadow_hours: Dict[int, float],
        module_power: float,
        total_hours: float
    ) -> Dict[str, float]:
        """
        Estimate energy loss due to shading.

        Args:
            shadow_hours: Dictionary of shadow hours per module
            module_power: Power rating per module in kW
            total_hours: Total analysis period in hours

        Returns:
            Dictionary containing:
                - total_potential_kwh: Total potential energy without shading
                - total_actual_kwh: Actual energy accounting for shading
                - total_loss_kwh: Energy loss due to shading
                - loss_percentage: Percentage of energy lost

        Example:
            >>> loss = analyzer.estimate_energy_loss(shadow_hours, 0.4, 720)
        """
        num_modules = len(shadow_hours)
        total_potential = num_modules * module_power * total_hours
        total_shaded_hours = sum(shadow_hours.values())
        total_loss = total_shaded_hours * module_power

        total_actual = total_potential - total_loss
        loss_percentage = (total_loss / total_potential * 100) if total_potential > 0 else 0

        return {
            'total_potential_kwh': total_potential,
            'total_actual_kwh': total_actual,
            'total_loss_kwh': total_loss,
            'loss_percentage': loss_percentage,
            'num_modules': num_modules,
        }
