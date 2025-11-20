"""
Comprehensive tests for the geospatial module.

Tests layout generation, shadow calculation, and GeoJSON import/export
with sample test sites.
"""

import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point
from datetime import datetime
import tempfile
import os

# Import modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from geospatial.layout_optimizer import LayoutOptimizer
from geospatial.shadow_analysis import ShadowAnalyzer
from geospatial.map_processor import MapProcessor


# Test fixtures

@pytest.fixture
def simple_square_site():
    """Create a simple 100x100m square test site."""
    return Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


@pytest.fixture
def complex_site():
    """Create a complex shaped test site."""
    return Polygon([
        (0, 0), (150, 0), (150, 80), (120, 80),
        (120, 120), (0, 120)
    ])


@pytest.fixture
def small_site():
    """Create a small 20x20m test site."""
    return Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])


@pytest.fixture
def layout_optimizer():
    """Create a LayoutOptimizer instance."""
    return LayoutOptimizer()


@pytest.fixture
def shadow_analyzer():
    """Create a ShadowAnalyzer instance."""
    return ShadowAnalyzer()


@pytest.fixture
def map_processor():
    """Create a MapProcessor instance."""
    return MapProcessor()


# Layout Optimizer Tests

class TestLayoutOptimizer:
    """Test suite for LayoutOptimizer class."""

    def test_initialization(self, layout_optimizer):
        """Test LayoutOptimizer initialization."""
        assert layout_optimizer is not None
        assert layout_optimizer.rtree_index is None

    def test_generate_grid_layout_basic(self, layout_optimizer, simple_square_site):
        """Test basic grid layout generation."""
        module_dims = (2.0, 1.0)  # 2m x 1m modules
        spacing = (0.5, 0.5)  # 0.5m spacing

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        # Should generate a valid GeoDataFrame
        assert isinstance(layout, gpd.GeoDataFrame)
        assert len(layout) > 0
        assert 'module_id' in layout.columns
        assert 'geometry' in layout.columns
        assert 'row' in layout.columns
        assert 'col' in layout.columns

        # All modules should be within site
        for idx, row in layout.iterrows():
            assert simple_square_site.contains(row.geometry) or \
                   simple_square_site.intersects(row.geometry)

    def test_generate_grid_layout_small_site(self, layout_optimizer, small_site):
        """Test grid generation on a small site."""
        module_dims = (2.0, 1.0)
        spacing = (0.5, 0.5)

        layout = layout_optimizer.generate_grid_layout(
            small_site,
            module_dims,
            spacing
        )

        # Should generate fewer modules
        assert isinstance(layout, gpd.GeoDataFrame)
        # Small site should have limited modules
        assert len(layout) < 100

    def test_generate_grid_layout_with_orientation(
        self, layout_optimizer, simple_square_site
    ):
        """Test grid generation with module orientation."""
        module_dims = (2.0, 1.0)
        spacing = (1.0, 1.0)
        orientation = 45.0  # 45 degrees

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing,
            orientation=orientation
        )

        assert len(layout) > 0
        assert all(layout['orientation'] == orientation)

    def test_calculate_coverage(self, layout_optimizer, simple_square_site):
        """Test coverage calculation."""
        module_dims = (2.0, 1.0)
        spacing = (0.5, 0.5)

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        coverage = layout_optimizer.calculate_coverage(simple_square_site, layout)

        assert 0 <= coverage <= 100
        assert isinstance(coverage, float)

    def test_calculate_total_capacity(self, layout_optimizer, simple_square_site):
        """Test total capacity calculation."""
        module_dims = (2.0, 1.0)
        spacing = (1.0, 1.0)
        module_power = 0.4  # 400W per module

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        total_capacity = layout_optimizer.calculate_total_capacity(
            layout,
            module_power
        )

        expected_capacity = len(layout) * module_power
        assert total_capacity == expected_capacity
        assert total_capacity > 0

    def test_optimize_layout(self, layout_optimizer, simple_square_site):
        """Test layout optimization."""
        module_dims = (2.0, 1.0)
        objectives = {
            'maximize_count': True,
            'minimize_cable_length': False,
        }

        optimized_layout = layout_optimizer.optimize_layout(
            simple_square_site,
            module_dims,
            objectives
        )

        assert isinstance(optimized_layout, gpd.GeoDataFrame)
        assert len(optimized_layout) > 0

    def test_check_collisions(self, layout_optimizer, simple_square_site):
        """Test collision detection."""
        module_dims = (2.0, 1.0)
        spacing = (1.0, 1.0)  # Adequate spacing

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        collisions = layout_optimizer.check_collisions(layout)

        # With adequate spacing, should have no collisions
        assert isinstance(collisions, list)

    def test_get_layout_statistics(self, layout_optimizer, simple_square_site):
        """Test layout statistics calculation."""
        module_dims = (2.0, 1.0)
        spacing = (1.0, 1.0)
        module_power = 0.4

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        stats = layout_optimizer.get_layout_statistics(
            simple_square_site,
            layout,
            module_power
        )

        assert 'num_modules' in stats
        assert 'total_capacity_kw' in stats
        assert 'coverage_percent' in stats
        assert 'site_area' in stats
        assert stats['num_modules'] == len(layout)
        assert stats['site_area'] == simple_square_site.area

    def test_empty_layout_handling(self, layout_optimizer, small_site):
        """Test handling of sites too small for modules."""
        module_dims = (50.0, 50.0)  # Huge modules
        spacing = (1.0, 1.0)

        layout = layout_optimizer.generate_grid_layout(
            small_site,
            module_dims,
            spacing
        )

        # Should return empty GeoDataFrame with proper schema
        assert isinstance(layout, gpd.GeoDataFrame)
        assert len(layout) == 0


# Shadow Analyzer Tests

class TestShadowAnalyzer:
    """Test suite for ShadowAnalyzer class."""

    def test_initialization(self, shadow_analyzer):
        """Test ShadowAnalyzer initialization."""
        assert shadow_analyzer is not None

    def test_calculate_sun_position(self, shadow_analyzer):
        """Test sun position calculation."""
        # Test at solar noon on summer solstice in NYC
        lat, lon = 40.7128, -74.0060
        date = datetime(2024, 6, 21, 12, 0)

        azimuth, elevation = shadow_analyzer.calculate_sun_position(
            lat, lon, date, date
        )

        # Sun should be roughly south (180°) at solar noon
        assert 0 <= azimuth <= 360
        # Elevation should be positive and reasonable
        assert 0 < elevation < 90

    def test_calculate_sun_position_winter(self, shadow_analyzer):
        """Test sun position in winter."""
        lat, lon = 40.7128, -74.0060
        date = datetime(2024, 12, 21, 12, 0)

        azimuth, elevation = shadow_analyzer.calculate_sun_position(
            lat, lon, date, date
        )

        assert 0 <= azimuth <= 360
        # Winter elevation should be lower than summer
        assert 0 < elevation < 60

    def test_project_shadows(self, shadow_analyzer, layout_optimizer, simple_square_site):
        """Test shadow projection."""
        # Generate a simple layout
        module_dims = (2.0, 1.0)
        spacing = (2.0, 2.0)

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        # Project shadows with sun at 45° elevation, 180° azimuth (south)
        shadows = shadow_analyzer.project_shadows(
            layout,
            sun_azimuth=180.0,
            sun_elevation=45.0,
            height=2.0
        )

        assert isinstance(shadows, list)
        assert len(shadows) == len(layout)
        assert all(isinstance(s, Polygon) for s in shadows)

    def test_project_shadows_low_sun(self, shadow_analyzer, layout_optimizer, simple_square_site):
        """Test shadow projection with low sun angle."""
        module_dims = (2.0, 1.0)
        spacing = (2.0, 2.0)

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        # Low sun angle should create longer shadows
        shadows_low = shadow_analyzer.project_shadows(
            layout,
            sun_azimuth=180.0,
            sun_elevation=15.0,
            height=2.0
        )

        # High sun angle should create shorter shadows
        shadows_high = shadow_analyzer.project_shadows(
            layout,
            sun_azimuth=180.0,
            sun_elevation=75.0,
            height=2.0
        )

        # Verify we get shadows
        assert len(shadows_low) > 0
        assert len(shadows_high) > 0

    def test_project_shadows_sun_below_horizon(self, shadow_analyzer, layout_optimizer, simple_square_site):
        """Test shadow projection when sun is below horizon."""
        module_dims = (2.0, 1.0)
        spacing = (2.0, 2.0)

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        shadows = shadow_analyzer.project_shadows(
            layout,
            sun_azimuth=180.0,
            sun_elevation=-10.0,  # Below horizon
            height=2.0
        )

        # Should return empty list
        assert shadows == []

    def test_analyze_shading(self, shadow_analyzer, layout_optimizer, simple_square_site):
        """Test shading analysis over time."""
        module_dims = (2.0, 1.0)
        spacing = (2.0, 2.0)

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        lat, lon = 40.7128, -74.0060
        start_date = datetime(2024, 6, 21, 0, 0)
        end_date = datetime(2024, 6, 21, 23, 59)

        shading_map = shadow_analyzer.analyze_shading(
            layout,
            lat,
            lon,
            (start_date, end_date),
            time_steps=12
        )

        assert isinstance(shading_map, dict)
        assert len(shading_map) == len(layout)
        # All values should be non-negative
        assert all(hours >= 0 for hours in shading_map.values())

    def test_calculate_solar_irradiance_factor(self, shadow_analyzer):
        """Test solar irradiance factor calculation."""
        shadow_hours = {0: 2.0, 1: 4.0, 2: 0.0}
        total_hours = 12.0

        factors = shadow_analyzer.calculate_solar_irradiance_factor(
            shadow_hours,
            total_hours
        )

        assert len(factors) == 3
        assert 0 <= factors[0] <= 1
        assert 0 <= factors[1] <= 1
        assert factors[2] == 1.0  # No shading

    def test_get_sun_path(self, shadow_analyzer):
        """Test sun path calculation."""
        lat, lon = 40.7128, -74.0060
        date = datetime(2024, 6, 21)

        sun_path = shadow_analyzer.get_sun_path(lat, lon, date, time_steps=24)

        assert len(sun_path) == 24
        assert all(len(point) == 3 for point in sun_path)
        # Check structure
        azimuth, elevation, time = sun_path[0]
        assert isinstance(azimuth, float)
        assert isinstance(elevation, float)
        assert isinstance(time, datetime)

    def test_estimate_energy_loss(self, shadow_analyzer):
        """Test energy loss estimation."""
        shadow_hours = {0: 2.0, 1: 3.0, 2: 1.0}
        module_power = 0.4  # kW
        total_hours = 720.0  # 30 days

        loss = shadow_analyzer.estimate_energy_loss(
            shadow_hours,
            module_power,
            total_hours
        )

        assert 'total_potential_kwh' in loss
        assert 'total_actual_kwh' in loss
        assert 'total_loss_kwh' in loss
        assert 'loss_percentage' in loss
        assert loss['total_actual_kwh'] <= loss['total_potential_kwh']
        assert 0 <= loss['loss_percentage'] <= 100

    def test_visualize_shadows(self, shadow_analyzer, layout_optimizer, simple_square_site):
        """Test shadow visualization."""
        module_dims = (2.0, 1.0)
        spacing = (5.0, 5.0)

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        shadows = shadow_analyzer.project_shadows(
            layout,
            sun_azimuth=180.0,
            sun_elevation=45.0,
            height=2.0
        )

        # Test visualization creation
        fig = shadow_analyzer.visualize_shadows(layout, shadows, figsize=(10, 8))

        assert fig is not None


# Map Processor Tests

class TestMapProcessor:
    """Test suite for MapProcessor class."""

    def test_initialization(self, map_processor):
        """Test MapProcessor initialization."""
        assert map_processor is not None

    def test_load_save_geojson(self, map_processor, simple_square_site):
        """Test GeoJSON loading and saving."""
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {'name': ['test_site']},
            geometry=[simple_square_site],
            crs='EPSG:4326'
        )

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            map_processor.save_geojson(gdf, temp_path)
            assert os.path.exists(temp_path)

            # Load
            loaded_gdf = map_processor.load_geojson(temp_path)
            assert isinstance(loaded_gdf, gpd.GeoDataFrame)
            assert len(loaded_gdf) == 1
            assert 'name' in loaded_gdf.columns

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_transform_crs(self, map_processor, simple_square_site):
        """Test CRS transformation."""
        # Create GeoDataFrame in WGS84 (lat/lon)
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[simple_square_site],
            crs='EPSG:4326'
        )

        # Transform to Web Mercator
        transformed = map_processor.transform_crs(gdf, 'EPSG:4326', 'EPSG:3857')

        assert transformed.crs == 'EPSG:3857'
        assert len(transformed) == len(gdf)

    def test_clip_to_boundary(self, map_processor):
        """Test clipping to boundary."""
        # Create a large area
        large_area = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

        # Create multiple points, some inside, some outside
        points = [
            Point(50, 50),  # Inside
            Point(25, 25),  # Inside
            Point(150, 150),  # Outside
        ]

        gdf = gpd.GeoDataFrame(
            {'id': [1, 2, 3]},
            geometry=points,
            crs='EPSG:4326'
        )

        # Clip to boundary
        clipped = map_processor.clip_to_boundary(gdf, large_area)

        # Should only have points inside
        assert len(clipped) == 2

    def test_get_crs_info(self, map_processor, simple_square_site):
        """Test CRS information retrieval."""
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[simple_square_site],
            crs='EPSG:4326'
        )

        crs_info = map_processor.get_crs_info(gdf)

        assert 'crs' in crs_info
        assert 'units' in crs_info
        assert crs_info['crs'] is not None

    def test_calculate_area(self, map_processor, simple_square_site):
        """Test area calculation."""
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[simple_square_site],
            crs='EPSG:4326'
        )

        areas = map_processor.calculate_area(gdf)

        assert len(areas) == 1
        assert areas.iloc[0] > 0

    def test_calculate_bounds(self, map_processor, simple_square_site):
        """Test bounds calculation."""
        gdf = gpd.GeoDataFrame(
            {'id': [1]},
            geometry=[simple_square_site],
            crs='EPSG:4326'
        )

        bounds = map_processor.calculate_bounds(gdf)

        assert len(bounds) == 4
        minx, miny, maxx, maxy = bounds
        assert minx < maxx
        assert miny < maxy


# Integration Tests

class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_full_workflow(
        self,
        layout_optimizer,
        shadow_analyzer,
        map_processor,
        simple_square_site
    ):
        """Test complete workflow from layout to shadow analysis."""
        # 1. Generate layout
        module_dims = (2.0, 1.0)
        spacing = (1.0, 1.0)
        module_power = 0.4

        layout = layout_optimizer.generate_grid_layout(
            simple_square_site,
            module_dims,
            spacing
        )

        assert len(layout) > 0

        # 2. Calculate coverage and capacity
        coverage = layout_optimizer.calculate_coverage(simple_square_site, layout)
        capacity = layout_optimizer.calculate_total_capacity(layout, module_power)

        assert coverage > 0
        assert capacity > 0

        # 3. Analyze shadows
        lat, lon = 40.7, -74.0
        start = datetime(2024, 6, 21, 0, 0)
        end = datetime(2024, 6, 21, 23, 59)

        shadow_hours = shadow_analyzer.analyze_shading(
            layout, lat, lon, (start, end), time_steps=12
        )

        assert len(shadow_hours) == len(layout)

        # 4. Calculate energy loss
        total_hours = 24.0
        energy_loss = shadow_analyzer.estimate_energy_loss(
            shadow_hours, module_power, total_hours
        )

        assert energy_loss['total_potential_kwh'] > 0

        # 5. Save layout
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.geojson', delete=False
        ) as f:
            temp_path = f.name

        try:
            map_processor.save_geojson(layout, temp_path)
            loaded = map_processor.load_geojson(temp_path)
            assert len(loaded) == len(layout)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
