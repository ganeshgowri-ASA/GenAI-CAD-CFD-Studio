"""
Tests for File Import UI functionality.
Tests file upload flow, geometry parsing, and preview rendering.
"""

import unittest
import tempfile
import os
import numpy as np
from pathlib import Path
import struct

# Import modules to test
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.io import universal_importer
from src.visualization import preview_basic
from src.ui.components import file_uploader


class TestUniversalImporter(unittest.TestCase):
    """Test universal importer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_stl_binary(self, filename: str) -> str:
        """Create a simple binary STL file for testing."""
        filepath = os.path.join(self.temp_dir, filename)

        with open(filepath, 'wb') as f:
            # Write header
            header = b'Test STL file' + b' ' * 67
            f.write(header)

            # Write number of triangles
            num_triangles = 2
            f.write(struct.pack('<I', num_triangles))

            # Write 2 triangles (a simple square split into 2 triangles)
            for _ in range(num_triangles):
                # Normal
                f.write(struct.pack('<3f', 0.0, 0.0, 1.0))

                # Vertices
                f.write(struct.pack('<3f', 0.0, 0.0, 0.0))
                f.write(struct.pack('<3f', 1.0, 0.0, 0.0))
                f.write(struct.pack('<3f', 1.0, 1.0, 0.0))

                # Attribute byte count
                f.write(struct.pack('<H', 0))

        return filepath

    def create_test_stl_ascii(self, filename: str) -> str:
        """Create a simple ASCII STL file for testing."""
        filepath = os.path.join(self.temp_dir, filename)

        with open(filepath, 'w') as f:
            f.write("solid test\n")
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write("      vertex 0 0 0\n")
            f.write("      vertex 1 0 0\n")
            f.write("      vertex 1 1 0\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
            f.write("  facet normal 0 0 1\n")
            f.write("    outer loop\n")
            f.write("      vertex 0 0 0\n")
            f.write("      vertex 1 1 0\n")
            f.write("      vertex 0 1 0\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
            f.write("endsolid test\n")

        return filepath

    def create_test_obj(self, filename: str) -> str:
        """Create a simple OBJ file for testing."""
        filepath = os.path.join(self.temp_dir, filename)

        with open(filepath, 'w') as f:
            f.write("# Test OBJ file\n")
            f.write("v 0 0 0\n")
            f.write("v 1 0 0\n")
            f.write("v 1 1 0\n")
            f.write("v 0 1 0\n")
            f.write("f 1 2 3\n")
            f.write("f 1 3 4\n")

        return filepath

    def test_parse_stl_binary(self):
        """Test parsing binary STL file."""
        filepath = self.create_test_stl_binary("test.stl")
        geometry = universal_importer.parse(filepath)

        self.assertIsInstance(geometry, universal_importer.GeometryData)
        self.assertEqual(geometry.num_vertices, 6)  # 2 triangles * 3 vertices
        self.assertEqual(geometry.num_faces, 2)
        self.assertIsNotNone(geometry.normals)

    def test_parse_stl_ascii(self):
        """Test parsing ASCII STL file."""
        filepath = self.create_test_stl_ascii("test_ascii.stl")
        geometry = universal_importer.parse(filepath)

        self.assertIsInstance(geometry, universal_importer.GeometryData)
        self.assertEqual(geometry.num_vertices, 6)
        self.assertEqual(geometry.num_faces, 2)

    def test_parse_obj(self):
        """Test parsing OBJ file."""
        filepath = self.create_test_obj("test.obj")
        geometry = universal_importer.parse(filepath)

        self.assertIsInstance(geometry, universal_importer.GeometryData)
        self.assertEqual(geometry.num_vertices, 4)
        self.assertEqual(geometry.num_faces, 2)

    def test_file_not_found(self):
        """Test error handling for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            universal_importer.parse("nonexistent.stl")

    def test_unsupported_format(self):
        """Test error handling for unsupported format."""
        filepath = os.path.join(self.temp_dir, "test.xyz")
        with open(filepath, 'w') as f:
            f.write("dummy content")

        with self.assertRaises(ValueError):
            universal_importer.parse(filepath)


class TestGeometryData(unittest.TestCase):
    """Test GeometryData class."""

    def setUp(self):
        """Set up test geometry."""
        self.geometry = universal_importer.GeometryData()

        # Create a simple cube
        self.geometry.vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float32)

        self.geometry.faces = np.array([
            [0, 1, 2, 3],  # Bottom
            [4, 5, 6, 7],  # Top
        ], dtype=np.object_)

    def test_num_vertices(self):
        """Test vertex count."""
        self.assertEqual(self.geometry.num_vertices, 8)

    def test_num_faces(self):
        """Test face count."""
        self.assertEqual(self.geometry.num_faces, 2)

    def test_bounding_box(self):
        """Test bounding box calculation."""
        bbox_min, bbox_max = self.geometry.bounding_box

        np.testing.assert_array_equal(bbox_min, [0, 0, 0])
        np.testing.assert_array_equal(bbox_max, [1, 1, 1])

    def test_bounding_box_dimensions(self):
        """Test bounding box dimensions."""
        dims = self.geometry.bounding_box_dimensions

        np.testing.assert_array_equal(dims, [1, 1, 1])

    def test_volume(self):
        """Test volume calculation."""
        volume = self.geometry.volume
        self.assertAlmostEqual(volume, 1.0, places=5)

    def test_empty_geometry(self):
        """Test empty geometry handling."""
        empty_geom = universal_importer.GeometryData()

        self.assertEqual(empty_geom.num_vertices, 0)
        self.assertEqual(empty_geom.num_faces, 0)
        self.assertEqual(empty_geom.volume, 0.0)


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""

    def setUp(self):
        """Set up test geometry."""
        self.vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]
        ], dtype=np.float32)

        self.faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int32)

    def test_plot_mesh_3d(self):
        """Test 3D mesh plotting."""
        fig = preview_basic.plot_mesh_3d(self.vertices, self.faces)

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)  # Should have one mesh trace

    def test_plot_wireframe(self):
        """Test wireframe mode."""
        fig = preview_basic.plot_mesh_3d(
            self.vertices,
            self.faces,
            mode='wireframe'
        )

        self.assertIsNotNone(fig)

    def test_plot_both_modes(self):
        """Test combined solid and wireframe mode."""
        fig = preview_basic.plot_mesh_3d(
            self.vertices,
            self.faces,
            mode='both'
        )

        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 2)  # Should have mesh + wireframe

    def test_empty_mesh(self):
        """Test handling of empty mesh."""
        empty_vertices = np.array([])
        empty_faces = np.array([])

        fig = preview_basic.plot_mesh_3d(empty_vertices, empty_faces)

        self.assertIsNotNone(fig)

    def test_geometry_data_plot(self):
        """Test plotting from GeometryData object."""
        geometry = universal_importer.GeometryData()
        geometry.vertices = self.vertices
        geometry.faces = self.faces
        geometry.metadata = {'file_type': 'test'}

        fig = preview_basic.plot_geometry_data(geometry)

        self.assertIsNotNone(fig)

    def test_camera_controls(self):
        """Test camera preset generation."""
        cameras = preview_basic.create_camera_controls()

        self.assertIn('isometric', cameras)
        self.assertIn('top', cameras)
        self.assertIn('front', cameras)
        self.assertIn('side', cameras)

    def test_mesh_statistics(self):
        """Test mesh statistics calculation."""
        stats = preview_basic.get_mesh_statistics(self.vertices, self.faces)

        self.assertEqual(stats['num_vertices'], 4)
        self.assertEqual(stats['num_faces'], 2)
        self.assertIn('bounding_box_min', stats)
        self.assertIn('bounding_box_max', stats)
        self.assertIn('dimensions', stats)


class TestFileUploaderComponents(unittest.TestCase):
    """Test file uploader components."""

    def test_format_file_size(self):
        """Test file size formatting."""
        self.assertEqual(file_uploader.format_file_size(1024), "1.00 KB")
        self.assertEqual(file_uploader.format_file_size(1024 * 1024), "1.00 MB")
        self.assertEqual(file_uploader.format_file_size(1024 * 1024 * 1024), "1.00 GB")

    def test_supported_formats(self):
        """Test supported formats dictionary."""
        formats = file_uploader.SUPPORTED_FORMATS

        self.assertIn('stl', formats)
        self.assertIn('obj', formats)
        self.assertIn('dxf', formats)
        self.assertIn('step', formats)

    def test_custom_css_generation(self):
        """Test CSS generation."""
        css = file_uploader.get_custom_css()

        self.assertIsInstance(css, str)
        self.assertIn('<style>', css)
        self.assertIn('</style>', css)


class TestTriangulation(unittest.TestCase):
    """Test face triangulation."""

    def test_triangulate_triangles(self):
        """Test triangulation of already triangular faces."""
        faces = np.array([[0, 1, 2], [2, 3, 0]], dtype=np.int32)
        result = preview_basic._triangulate_faces(faces)

        self.assertEqual(len(result), 2)

    def test_triangulate_quads(self):
        """Test triangulation of quad faces."""
        faces = np.array([[0, 1, 2, 3]], dtype=np.object_)
        result = preview_basic._triangulate_faces(faces)

        self.assertEqual(len(result), 2)  # Quad splits into 2 triangles

    def test_triangulate_polygon(self):
        """Test triangulation of polygon (>4 vertices)."""
        faces = np.array([[0, 1, 2, 3, 4]], dtype=np.object_)
        result = preview_basic._triangulate_faces(faces)

        self.assertEqual(len(result), 3)  # Pentagon splits into 3 triangles

    def test_empty_faces(self):
        """Test empty faces array."""
        faces = np.array([])
        result = preview_basic._triangulate_faces(faces)

        self.assertEqual(len(result), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_test_stl(self):
        """Create test STL file."""
        filepath = os.path.join(self.temp_dir, "test.stl")

        with open(filepath, 'wb') as f:
            header = b'Test' + b' ' * 76
            f.write(header)

            # 1 triangle
            f.write(struct.pack('<I', 1))

            # Normal and vertices
            f.write(struct.pack('<3f', 0.0, 0.0, 1.0))
            f.write(struct.pack('<3f', 0.0, 0.0, 0.0))
            f.write(struct.pack('<3f', 1.0, 0.0, 0.0))
            f.write(struct.pack('<3f', 0.5, 1.0, 0.0))
            f.write(struct.pack('<H', 0))

        return filepath

    def test_full_workflow(self):
        """Test complete file import to visualization workflow."""
        # Create test file
        filepath = self.create_test_stl()

        # Parse file
        geometry = universal_importer.parse(filepath)

        # Verify parsing
        self.assertIsInstance(geometry, universal_importer.GeometryData)
        self.assertGreater(geometry.num_vertices, 0)
        self.assertGreater(geometry.num_faces, 0)

        # Calculate metrics
        bbox_min, bbox_max = geometry.bounding_box
        dims = geometry.bounding_box_dimensions

        self.assertEqual(len(bbox_min), 3)
        self.assertEqual(len(bbox_max), 3)
        self.assertEqual(len(dims), 3)

        # Generate visualization
        fig = preview_basic.plot_geometry_data(geometry)

        self.assertIsNotNone(fig)
        self.assertGreater(len(fig.data), 0)

    def test_metadata_preservation(self):
        """Test that metadata is preserved through workflow."""
        filepath = self.create_test_stl()
        geometry = universal_importer.parse(filepath)

        self.assertIn('file_path', geometry.metadata)
        self.assertIn('file_type', geometry.metadata)
        self.assertIn('file_size', geometry.metadata)

        self.assertEqual(geometry.metadata['file_type'], 'stl')


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
