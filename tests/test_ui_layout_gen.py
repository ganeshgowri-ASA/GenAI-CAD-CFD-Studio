"""
Unit tests for Solar PV Layout Generator UI Components
Tests for map interface, module configurator, and layout optimizer
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
import pandas as pd
from datetime import datetime
import json

# Import modules to test
from src.ui.components.module_configurator import ModuleConfig, ModuleConfigurator
from src.geospatial.layout_optimizer import LayoutOptimizer, ShadowAnalyzer


class TestModuleConfig(unittest.TestCase):
    """Test ModuleConfig dataclass"""

    def test_module_config_creation(self):
        """Test creating a module configuration"""
        config = ModuleConfig(
            width=1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=180.0
        )

        self.assertEqual(config.width, 1.0)
        self.assertEqual(config.length, 2.0)
        self.assertEqual(config.power_watts, 400)

    def test_module_config_to_dict(self):
        """Test converting config to dictionary"""
        config = ModuleConfig(
            width=1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=180.0
        )

        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['width'], 1.0)
        self.assertEqual(config_dict['power_watts'], 400)

    def test_module_config_validation_valid(self):
        """Test validation with valid configuration"""
        config = ModuleConfig(
            width=1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=180.0
        )

        is_valid, error_msg = config.validate()

        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_module_config_validation_invalid_width(self):
        """Test validation with invalid width"""
        config = ModuleConfig(
            width=-1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=180.0
        )

        is_valid, error_msg = config.validate()

        self.assertFalse(is_valid)
        self.assertIn("width", error_msg.lower())

    def test_module_config_validation_invalid_tilt(self):
        """Test validation with invalid tilt angle"""
        config = ModuleConfig(
            width=1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=100.0,  # Invalid: > 90
            azimuth=180.0
        )

        is_valid, error_msg = config.validate()

        self.assertFalse(is_valid)
        self.assertIn("tilt", error_msg.lower())

    def test_module_config_validation_invalid_azimuth(self):
        """Test validation with invalid azimuth"""
        config = ModuleConfig(
            width=1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=400.0  # Invalid: > 360
        )

        is_valid, error_msg = config.validate()

        self.assertFalse(is_valid)
        self.assertIn("azimuth", error_msg.lower())


class TestModuleConfigurator(unittest.TestCase):
    """Test ModuleConfigurator component"""

    def test_presets_exist(self):
        """Test that module presets are defined"""
        self.assertIsInstance(ModuleConfigurator.PRESETS, dict)
        self.assertGreater(len(ModuleConfigurator.PRESETS), 0)

    def test_preset_structure(self):
        """Test that presets have correct structure"""
        for preset_name, preset_config in ModuleConfigurator.PRESETS.items():
            self.assertIn('width', preset_config)
            self.assertIn('length', preset_config)
            self.assertIn('power_watts', preset_config)
            self.assertIn('row_spacing', preset_config)
            self.assertIn('column_spacing', preset_config)
            self.assertIn('tilt_angle', preset_config)
            self.assertIn('azimuth', preset_config)

    def test_get_direction_name(self):
        """Test direction name conversion"""
        self.assertEqual(ModuleConfigurator._get_direction_name(0), "North")
        self.assertEqual(ModuleConfigurator._get_direction_name(90), "East")
        self.assertEqual(ModuleConfigurator._get_direction_name(180), "South")
        self.assertEqual(ModuleConfigurator._get_direction_name(270), "West")

    def test_export_import_config(self):
        """Test exporting and importing configuration"""
        configurator = ModuleConfigurator()
        configurator.config = ModuleConfig(
            width=1.0,
            length=2.0,
            power_watts=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=180.0
        )

        # Export
        config_json = configurator.export_config()
        self.assertIsInstance(config_json, str)

        # Import
        success = configurator.import_config(config_json)
        self.assertTrue(success)
        self.assertEqual(configurator.config.width, 1.0)


class TestLayoutOptimizer(unittest.TestCase):
    """Test LayoutOptimizer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = LayoutOptimizer(
            module_width=1.0,
            module_length=2.0,
            module_power=400,
            row_spacing=1.0,
            column_spacing=0.02,
            tilt_angle=20.0,
            azimuth=180.0
        )

    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.module_width, 1.0)
        self.assertEqual(self.optimizer.module_length, 2.0)
        self.assertEqual(self.optimizer.module_power, 400)

    def test_calculate_effective_spacing(self):
        """Test effective spacing calculation"""
        spacing = self.optimizer._calculate_effective_spacing()
        self.assertIsInstance(spacing, float)
        self.assertGreater(spacing, self.optimizer.row_spacing)

    def test_generate_layout_small_boundary(self):
        """Test layout generation with small boundary"""
        # Create a small rectangular boundary
        boundary = box(0, 0, 10, 10)

        layout_gdf = self.optimizer.generate_layout(boundary)

        self.assertIsInstance(layout_gdf, gpd.GeoDataFrame)
        self.assertGreater(len(layout_gdf), 0)

    def test_generate_layout_large_boundary(self):
        """Test layout generation with larger boundary"""
        # Create a larger boundary
        boundary = box(0, 0, 50, 50)

        layout_gdf = self.optimizer.generate_layout(boundary)

        self.assertIsInstance(layout_gdf, gpd.GeoDataFrame)
        self.assertGreater(len(layout_gdf), 5)  # Should have multiple modules

    def test_generate_layout_polygon_boundary(self):
        """Test layout generation with polygon boundary"""
        # Create a polygon boundary
        boundary = Polygon([(0, 0), (20, 0), (20, 20), (10, 25), (0, 20)])

        layout_gdf = self.optimizer.generate_layout(boundary)

        self.assertIsInstance(layout_gdf, gpd.GeoDataFrame)

    def test_layout_modules_within_boundary(self):
        """Test that all modules are within boundary"""
        boundary = box(0, 0, 20, 20)

        layout_gdf = self.optimizer.generate_layout(boundary)

        for idx, row in layout_gdf.iterrows():
            self.assertTrue(boundary.contains(row.geometry))

    def test_layout_attributes(self):
        """Test that layout has correct attributes"""
        boundary = box(0, 0, 20, 20)

        layout_gdf = self.optimizer.generate_layout(boundary)

        required_columns = ['module_id', 'row', 'column', 'power_watts',
                           'tilt_angle', 'azimuth', 'area_m2']

        for col in required_columns:
            self.assertIn(col, layout_gdf.columns)

    def test_calculate_statistics_with_modules(self):
        """Test statistics calculation with modules"""
        boundary = box(0, 0, 20, 20)
        layout_gdf = self.optimizer.generate_layout(boundary)

        stats = self.optimizer.calculate_statistics(layout_gdf, boundary)

        self.assertIsInstance(stats, dict)
        self.assertIn('total_modules', stats)
        self.assertIn('total_capacity_kw', stats)
        self.assertIn('coverage_percentage', stats)
        self.assertGreater(stats['total_modules'], 0)
        self.assertGreater(stats['total_capacity_kw'], 0)

    def test_calculate_statistics_empty_layout(self):
        """Test statistics calculation with empty layout"""
        boundary = box(0, 0, 1, 1)  # Too small for any modules
        layout_gdf = self.optimizer.generate_layout(boundary)

        stats = self.optimizer.calculate_statistics(layout_gdf, boundary)

        self.assertEqual(stats['total_modules'], 0)
        self.assertEqual(stats['total_capacity_kw'], 0.0)

    def test_layout_orientation_landscape(self):
        """Test layout with landscape orientation"""
        boundary = box(0, 0, 20, 20)

        layout_gdf = self.optimizer.generate_layout(boundary, orientation='landscape')

        self.assertGreater(len(layout_gdf), 0)

    def test_layout_orientation_portrait(self):
        """Test layout with portrait orientation"""
        boundary = box(0, 0, 20, 20)

        layout_gdf = self.optimizer.generate_layout(boundary, orientation='portrait')

        self.assertGreater(len(layout_gdf), 0)


class TestShadowAnalyzer(unittest.TestCase):
    """Test ShadowAnalyzer class"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ShadowAnalyzer(
            latitude=37.7749,
            longitude=-122.4194
        )

    def test_analyzer_initialization(self):
        """Test shadow analyzer initialization"""
        self.assertEqual(self.analyzer.latitude, 37.7749)
        self.assertEqual(self.analyzer.longitude, -122.4194)

    def test_calculate_sun_position_noon(self):
        """Test sun position calculation at noon"""
        timestamp = datetime(2024, 6, 21, 12, 0)  # Summer solstice, noon

        altitude, azimuth = self.analyzer.calculate_sun_position(timestamp)

        self.assertIsInstance(altitude, float)
        self.assertIsInstance(azimuth, float)
        self.assertGreater(altitude, 0)  # Sun should be above horizon
        self.assertGreaterEqual(azimuth, 0)
        self.assertLessEqual(azimuth, 360)

    def test_calculate_sun_position_night(self):
        """Test sun position calculation at night"""
        timestamp = datetime(2024, 6, 21, 0, 0)  # Midnight

        altitude, azimuth = self.analyzer.calculate_sun_position(timestamp)

        # Sun should be below horizon at midnight
        self.assertLess(altitude, 30)  # Low or below horizon

    def test_analyze_shadows_with_modules(self):
        """Test shadow analysis with module layout"""
        # Create simple module layout
        modules = [
            {
                'module_id': 0,
                'row': 0,
                'column': 0,
                'geometry': box(0, 0, 1, 2),
                'power_watts': 400,
                'tilt_angle': 20,
                'azimuth': 180,
                'area_m2': 2.0
            }
        ]
        layout_gdf = gpd.GeoDataFrame(modules, crs='EPSG:4326')

        timestamp = datetime(2024, 6, 21, 12, 0)

        shadow_gdf = self.analyzer.analyze_shadows(layout_gdf, timestamp)

        self.assertIsInstance(shadow_gdf, gpd.GeoDataFrame)

    def test_analyze_shadows_empty_layout(self):
        """Test shadow analysis with empty layout"""
        layout_gdf = gpd.GeoDataFrame(
            columns=['module_id', 'row', 'column', 'geometry', 'power_watts',
                    'tilt_angle', 'azimuth', 'area_m2'],
            crs='EPSG:4326'
        )

        timestamp = datetime(2024, 6, 21, 12, 0)

        shadow_gdf = self.analyzer.analyze_shadows(layout_gdf, timestamp)

        self.assertEqual(len(shadow_gdf), 0)

    def test_generate_shadow_report(self):
        """Test shadow report generation"""
        # Create simple module layout
        modules = [
            {
                'module_id': 0,
                'row': 0,
                'column': 0,
                'geometry': box(0, 0, 1, 2),
                'power_watts': 400,
                'tilt_angle': 20,
                'azimuth': 180,
                'area_m2': 2.0
            }
        ]
        layout_gdf = gpd.GeoDataFrame(modules, crs='EPSG:4326')

        start_time = datetime(2024, 6, 21, 6, 0)
        end_time = datetime(2024, 6, 21, 18, 0)

        report = self.analyzer.generate_shadow_report(
            layout_gdf,
            start_time,
            end_time,
            time_step_hours=1
        )

        self.assertIsInstance(report, dict)
        self.assertIn('analysis_period', report)
        self.assertIn('sun_statistics', report)
        self.assertIn('shadow_timeline', report)

    def test_shadow_report_structure(self):
        """Test shadow report has correct structure"""
        modules = [
            {
                'module_id': 0,
                'row': 0,
                'column': 0,
                'geometry': box(0, 0, 1, 2),
                'power_watts': 400,
                'tilt_angle': 20,
                'azimuth': 180,
                'area_m2': 2.0
            }
        ]
        layout_gdf = gpd.GeoDataFrame(modules, crs='EPSG:4326')

        start_time = datetime(2024, 6, 21, 6, 0)
        end_time = datetime(2024, 6, 21, 18, 0)

        report = self.analyzer.generate_shadow_report(
            layout_gdf,
            start_time,
            end_time,
            time_step_hours=2
        )

        # Check structure
        self.assertIn('start', report['analysis_period'])
        self.assertIn('end', report['analysis_period'])
        self.assertIn('daylight_hours', report['sun_statistics'])
        self.assertIsInstance(report['shadow_timeline'], list)


class TestMapInterfaceHelpers(unittest.TestCase):
    """Test MapInterface helper methods"""

    def test_get_color_from_value(self):
        """Test color conversion from normalized value"""
        from src.ui.components.map_interface import MapInterface

        # Test extremes
        color_0 = MapInterface._get_color_from_value(0.0)
        color_1 = MapInterface._get_color_from_value(1.0)

        self.assertIsInstance(color_0, str)
        self.assertIsInstance(color_1, str)
        self.assertTrue(color_0.startswith('#'))
        self.assertTrue(color_1.startswith('#'))

    def test_export_to_geojson(self):
        """Test GeoJSON export"""
        from src.ui.components.map_interface import MapInterface

        map_interface = MapInterface()

        # Create test geometries
        polygons = [
            box(0, 0, 1, 1),
            box(2, 2, 3, 3)
        ]

        properties = [
            {'id': 0, 'type': 'module'},
            {'id': 1, 'type': 'module'}
        ]

        geojson_str = map_interface.export_to_geojson(polygons, properties)

        self.assertIsInstance(geojson_str, str)

        # Validate JSON
        geojson = json.loads(geojson_str)
        self.assertEqual(geojson['type'], 'FeatureCollection')
        self.assertEqual(len(geojson['features']), 2)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
