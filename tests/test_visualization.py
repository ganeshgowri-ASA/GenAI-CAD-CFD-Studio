"""
Comprehensive tests for the visualization module.

Tests PyVista rendering, Plotly charts, export functionality,
and utility functions.
"""

import pytest
import numpy as np
import pyvista as pv
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Import visualization components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    PyVistaViewer,
    PlotlyCharts,
    ExportRenderer,
    MeshConverter,
    MeshOptimizer,
    MeshCache,
    convert_mesh,
    optimize_for_display
)


class TestPyVistaViewer:
    """Tests for PyVistaViewer class."""

    @pytest.fixture
    def viewer(self):
        """Create PyVistaViewer instance."""
        return PyVistaViewer(theme='document')

    @pytest.fixture
    def sample_mesh(self):
        """Create sample mesh for testing."""
        return pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)

    @pytest.fixture
    def sample_cfd_mesh(self):
        """Create sample CFD mesh with scalar fields."""
        mesh = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)

        # Add scalar fields
        mesh['pressure'] = np.random.random(mesh.n_points) * 100
        mesh['temperature'] = np.random.random(mesh.n_points) * 300 + 273

        # Add vector field (velocity)
        velocity = np.random.random((mesh.n_points, 3)) * 10
        mesh['velocity'] = velocity

        return mesh

    def test_viewer_initialization(self, viewer):
        """Test viewer initialization."""
        assert viewer.theme == 'document'

    def test_render_cad_model_with_mesh_object(self, viewer, sample_mesh):
        """Test rendering CAD model from mesh object."""
        plotter = viewer.render_cad_model(sample_mesh)

        assert plotter is not None
        assert isinstance(plotter, pv.Plotter)

        plotter.close()

    def test_render_cad_model_with_config(self, viewer, sample_mesh):
        """Test rendering with custom configuration."""
        config = {
            'window_size': (800, 600),
            'background': 'lightgray',
            'color': 'blue',
            'opacity': 0.8,
            'show_edges': True,
        }

        plotter = viewer.render_cad_model(sample_mesh, plotter_config=config)

        assert plotter is not None
        plotter.close()

    def test_render_cfd_results_scalar(self, viewer, sample_cfd_mesh):
        """Test rendering CFD results with scalar field."""
        plotter = viewer.render_cfd_results(
            sample_cfd_mesh,
            field='pressure',
            cmap='jet',
            show_scalar_bar=True
        )

        assert plotter is not None
        plotter.close()

    def test_render_cfd_results_vector(self, viewer, sample_cfd_mesh):
        """Test rendering CFD results with vector field."""
        plotter = viewer.render_cfd_results(
            sample_cfd_mesh,
            field='velocity',
            show_streamlines=True,
            n_streamlines=50
        )

        assert plotter is not None
        plotter.close()

    def test_render_cfd_results_with_slices(self, viewer, sample_cfd_mesh):
        """Test rendering with slice planes."""
        plotter = viewer.render_cfd_results(
            sample_cfd_mesh,
            field='temperature',
            slice_normal=(1, 0, 0),
            n_slices=3
        )

        assert plotter is not None
        plotter.close()

    def test_render_mesh_quality(self, viewer):
        """Test mesh quality visualization."""
        # Create unstructured grid
        mesh = pv.Sphere().cast_to_unstructured_grid()

        plotter = viewer.render_mesh_quality(
            mesh,
            quality_metric='aspect_ratio',
            threshold=0.3
        )

        assert plotter is not None
        plotter.close()

    def test_add_annotations_text(self, viewer, sample_mesh):
        """Test adding text annotations."""
        plotter = viewer.render_cad_model(sample_mesh)

        labels = {
            'center': {
                'type': 'text',
                'position': (0, 0, 0),
                'text': 'Center Point',
                'color': 'red'
            }
        }

        plotter = viewer.add_annotations(plotter, labels)
        assert plotter is not None
        plotter.close()

    def test_add_annotations_arrow(self, viewer, sample_mesh):
        """Test adding arrow annotations."""
        plotter = viewer.render_cad_model(sample_mesh)

        labels = {
            'force_vector': {
                'type': 'arrow',
                'position': (0, 0, 0),
                'direction': (1, 0, 0),
                'scale': 0.5,
                'text': 'Force',
                'color': 'blue'
            }
        }

        plotter = viewer.add_annotations(plotter, labels)
        assert plotter is not None
        plotter.close()

    def test_add_annotations_dimension(self, viewer, sample_mesh):
        """Test adding dimension annotations."""
        plotter = viewer.render_cad_model(sample_mesh)

        labels = {
            'diameter': {
                'type': 'dimension',
                'position': (-1, 0, 0),
                'end_position': (1, 0, 0),
                'text': '2.00m',
                'color': 'black'
            }
        }

        plotter = viewer.add_annotations(plotter, labels)
        assert plotter is not None
        plotter.close()

    def test_export_image(self, viewer, sample_mesh, tmp_path):
        """Test exporting image."""
        plotter = viewer.render_cad_model(sample_mesh)

        output_file = tmp_path / "test_export.png"
        viewer.export_image(plotter, output_file, resolution=(800, 600))

        assert output_file.exists()
        plotter.close()

    def test_create_sample_mesh_sphere(self, viewer):
        """Test creating sample sphere mesh."""
        mesh = viewer.create_sample_mesh('sphere')
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0

    def test_create_sample_mesh_cube(self, viewer):
        """Test creating sample cube mesh."""
        mesh = viewer.create_sample_mesh('cube')
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0

    def test_load_mesh_invalid_file(self, viewer):
        """Test loading non-existent mesh file."""
        with pytest.raises(FileNotFoundError):
            viewer._load_mesh("/nonexistent/file.stl")


class TestPlotlyCharts:
    """Tests for PlotlyCharts class."""

    @pytest.fixture
    def charts(self):
        """Create PlotlyCharts instance."""
        return PlotlyCharts(theme='plotly_white')

    @pytest.fixture
    def residual_data(self):
        """Create sample residual data."""
        return {
            'iteration': list(range(1, 101)),
            'continuity': np.logspace(-3, -6, 100),
            'x-velocity': np.logspace(-3, -6, 100) * 1.2,
            'y-velocity': np.logspace(-3, -6, 100) * 0.8,
            'energy': np.logspace(-3, -7, 100),
        }

    @pytest.fixture
    def convergence_data(self):
        """Create sample convergence data."""
        return {
            'iteration': list(range(1, 51)),
            'drag_coefficient': 0.5 + 0.1 * np.exp(-np.arange(50) / 10),
            'lift_coefficient': 0.3 + 0.05 * np.exp(-np.arange(50) / 15),
        }

    def test_charts_initialization(self, charts):
        """Test charts initialization."""
        assert charts.theme == 'plotly_white'

    def test_plot_residuals(self, charts, residual_data):
        """Test plotting residuals."""
        fig = charts.plot_residuals(residual_data, log_scale=True)

        assert fig is not None
        assert len(fig.data) == 4  # 4 residual fields
        assert fig.layout.yaxis.type == 'log'

    def test_plot_residuals_linear_scale(self, charts, residual_data):
        """Test plotting residuals with linear scale."""
        fig = charts.plot_residuals(residual_data, log_scale=False)

        assert fig is not None
        assert fig.layout.yaxis.type == 'linear'

    def test_plot_convergence(self, charts, convergence_data):
        """Test plotting convergence history."""
        fig = charts.plot_convergence(convergence_data)

        assert fig is not None
        assert len(fig.data) >= 2  # At least 2 metrics

    def test_plot_statistics_bar(self, charts):
        """Test bar chart statistics."""
        data = {
            'Metric A': 10.5,
            'Metric B': 23.7,
            'Metric C': 15.2,
        }

        fig = charts.plot_statistics(data, chart_type='bar')

        assert fig is not None
        assert len(fig.data) == 1

    def test_plot_statistics_pie(self, charts):
        """Test pie chart statistics."""
        data = {
            'Category A': 30,
            'Category B': 45,
            'Category C': 25,
        }

        fig = charts.plot_statistics(data, chart_type='pie')

        assert fig is not None
        assert len(fig.data) == 1

    def test_plot_statistics_horizontal_bar(self, charts):
        """Test horizontal bar chart."""
        data = {
            'Metric 1': 10,
            'Metric 2': 20,
            'Metric 3': 15,
        }

        fig = charts.plot_statistics(data, chart_type='horizontal_bar')

        assert fig is not None

    def test_plot_shadow_heatmap(self, charts):
        """Test shadow hours heatmap."""
        shadow_hours = np.random.random((12, 24)) * 10  # 12 months x 24 hours

        x_labels = [f"Month {i}" for i in range(1, 13)]
        y_labels = [f"{i}:00" for i in range(24)]

        fig = charts.plot_shadow_heatmap(
            shadow_hours,
            x_labels=x_labels,
            y_labels=y_labels
        )

        assert fig is not None

    def test_plot_comparison(self, charts):
        """Test comparison plot."""
        data = {
            'Simulation 1': {
                'time': [0, 1, 2, 3, 4, 5],
                'value': [0, 1, 4, 9, 16, 25]
            },
            'Simulation 2': {
                'time': [0, 1, 2, 3, 4, 5],
                'value': [0, 2, 8, 18, 32, 50]
            }
        }

        fig = charts.plot_comparison(data, x_col='time', y_col='value')

        assert fig is not None
        assert len(fig.data) == 2  # 2 datasets

    def test_plot_3d_scatter(self, charts):
        """Test 3D scatter plot."""
        data = {
            'x': np.random.random(50),
            'y': np.random.random(50),
            'z': np.random.random(50),
            'magnitude': np.random.random(50)
        }

        fig = charts.plot_3d_scatter(
            data,
            x='x',
            y='y',
            z='z',
            color='magnitude'
        )

        assert fig is not None

    def test_plot_contour(self, charts):
        """Test contour plot."""
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(np.sqrt(X**2 + Y**2))

        fig = charts.plot_contour(x, y, Z, n_contours=15)

        assert fig is not None


class TestExportRenderer:
    """Tests for ExportRenderer class."""

    @pytest.fixture
    def renderer(self):
        """Create ExportRenderer instance."""
        return ExportRenderer()

    @pytest.fixture
    def sample_mesh(self):
        """Create sample mesh."""
        return pv.Sphere(radius=1.0)

    def test_render_high_quality(self, renderer, sample_mesh, tmp_path):
        """Test high-quality rendering."""
        output_file = tmp_path / "high_quality.png"

        renderer.render_high_quality(
            sample_mesh,
            output_file,
            resolution=(1920, 1080),
            anti_aliasing=True,
            ambient_occlusion=False,  # Disable for testing
            shadows=True
        )

        assert output_file.exists()

    def test_render_high_quality_with_scalars(self, renderer, tmp_path):
        """Test rendering with scalar field."""
        mesh = pv.Sphere()
        mesh['elevation'] = mesh.points[:, 2]

        output_file = tmp_path / "scalar_render.png"

        renderer.render_high_quality(
            mesh,
            output_file,
            scalars='elevation',
            cmap='viridis'
        )

        assert output_file.exists()

    def test_batch_render(self, renderer, tmp_path):
        """Test batch rendering."""
        meshes = [
            (pv.Sphere(), "sphere"),
            (pv.Cube(), "cube"),
            (pv.Cylinder(), "cylinder")
        ]

        views = ['iso', 'xy', 'xz']

        files = renderer.batch_render(
            meshes,
            views,
            tmp_path,
            resolution=(800, 600)
        )

        assert len(files) == len(meshes) * len(views)
        for f in files:
            assert f.exists()


class TestMeshConverter:
    """Tests for MeshConverter utilities."""

    @pytest.fixture
    def converter(self):
        """Create MeshConverter instance."""
        return MeshConverter()

    @pytest.fixture
    def pyvista_mesh(self):
        """Create PyVista mesh."""
        return pv.Sphere(radius=1.0)

    def test_numpy_to_pyvista(self, converter):
        """Test converting numpy arrays to PyVista."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ], dtype=float)

        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])

        mesh = converter.numpy_to_pyvista(vertices, faces)

        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points == 4
        assert mesh.n_cells == 2

    def test_pyvista_to_numpy(self, converter, pyvista_mesh):
        """Test converting PyVista to numpy."""
        vertices, faces = converter.pyvista_to_numpy(pyvista_mesh)

        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3


class TestMeshOptimizer:
    """Tests for MeshOptimizer utilities."""

    @pytest.fixture
    def optimizer(self):
        """Create MeshOptimizer instance."""
        return MeshOptimizer()

    @pytest.fixture
    def dense_mesh(self):
        """Create dense mesh for testing."""
        return pv.Sphere(radius=1.0, theta_resolution=50, phi_resolution=50)

    def test_decimate_mesh(self, optimizer, dense_mesh):
        """Test mesh decimation."""
        original_cells = dense_mesh.n_cells

        decimated = optimizer.decimate_mesh(dense_mesh, target_reduction=0.5)

        assert decimated.n_cells < original_cells
        assert decimated.n_cells <= original_cells * 0.6  # Allow some tolerance

    def test_smooth_mesh(self, optimizer, dense_mesh):
        """Test mesh smoothing."""
        smoothed = optimizer.smooth_mesh(dense_mesh, n_iterations=10)

        assert isinstance(smoothed, pv.PolyData)
        assert smoothed.n_points == dense_mesh.n_points

    def test_create_lod_hierarchy(self, optimizer, dense_mesh):
        """Test LOD hierarchy creation."""
        lod_meshes = optimizer.create_lod_hierarchy(dense_mesh, n_levels=4)

        assert len(lod_meshes) == 4
        # Each level should have fewer cells than previous
        for i in range(1, len(lod_meshes)):
            assert lod_meshes[i].n_cells <= lod_meshes[i-1].n_cells

    def test_repair_mesh(self, optimizer):
        """Test mesh repair."""
        mesh = pv.Sphere()
        repaired = optimizer.repair_mesh(mesh)

        assert isinstance(repaired, pv.PolyData)

    def test_subdivide_mesh(self, optimizer):
        """Test mesh subdivision."""
        mesh = pv.Sphere(theta_resolution=10, phi_resolution=10)
        original_cells = mesh.n_cells

        subdivided = optimizer.subdivide_mesh(mesh, n_subdivisions=1)

        assert subdivided.n_cells > original_cells


class TestMeshCache:
    """Tests for MeshCache utilities."""

    @pytest.fixture
    def cache(self, tmp_path):
        """Create MeshCache instance with temporary directory."""
        cache_dir = tmp_path / "mesh_cache"
        return MeshCache(cache_dir=cache_dir)

    @pytest.fixture
    def test_mesh_file(self, tmp_path):
        """Create temporary mesh file."""
        mesh = pv.Sphere()
        filepath = tmp_path / "test_mesh.vtk"
        mesh.save(str(filepath))
        return filepath

    def test_cache_save_and_get(self, cache, test_mesh_file):
        """Test saving and retrieving cached mesh."""
        mesh = pv.read(str(test_mesh_file))
        operation = "decimate"
        params = {"reduction": 0.5}

        # Save to cache
        cache.save_cached_mesh(mesh, test_mesh_file, operation, params)

        # Retrieve from cache
        cached_mesh = cache.get_cached_mesh(test_mesh_file, operation, params)

        assert cached_mesh is not None
        assert cached_mesh.n_points == mesh.n_points

    def test_cache_miss(self, cache, test_mesh_file):
        """Test cache miss."""
        cached_mesh = cache.get_cached_mesh(
            test_mesh_file,
            "nonexistent_operation",
            {}
        )

        assert cached_mesh is None

    def test_clear_cache(self, cache, test_mesh_file):
        """Test clearing cache."""
        mesh = pv.read(str(test_mesh_file))
        cache.save_cached_mesh(mesh, test_mesh_file, "test", {})

        cache.clear_cache()

        # Cache should be empty
        cached_mesh = cache.get_cached_mesh(test_mesh_file, "test", {})
        assert cached_mesh is None


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_convert_mesh_pyvista_to_numpy(self):
        """Test mesh conversion to numpy."""
        mesh = pv.Sphere()
        vertices, faces = convert_mesh(mesh, target_format='numpy')

        assert isinstance(vertices, np.ndarray)
        assert isinstance(faces, np.ndarray)

    def test_optimize_for_display(self):
        """Test display optimization."""
        # Create large mesh
        mesh = pv.Sphere(theta_resolution=200, phi_resolution=200)
        original_cells = mesh.n_cells

        # Optimize for display
        optimized = optimize_for_display(mesh, max_cells=10000, smooth=True)

        assert optimized.n_cells <= 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
