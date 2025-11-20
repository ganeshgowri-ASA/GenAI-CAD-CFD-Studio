"""
PyVista-based 3D visualization engine for CAD models and CFD results.

This module provides the core PyVistaViewer class for rendering CAD geometries,
CFD simulation results, and mesh quality visualizations.

NOTE: This module requires PyVista to be installed. It is optional and may not
be available in all deployment environments (e.g., Streamlit Cloud).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING, Any
from pathlib import Path
import warnings

# Use TYPE_CHECKING to avoid runtime import errors while preserving type hints
if TYPE_CHECKING:
    import pyvista as pv

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None  # type: ignore


class PyVistaViewer:
    """
    Professional 3D visualization engine using PyVista.

    Handles rendering of CAD models, CFD results, mesh quality analysis,
    and provides annotation and export capabilities.
    """

    def __init__(self, theme: str = "document"):
        """
        Initialize PyVista viewer with default settings.

        Args:
            theme: PyVista theme ('default', 'dark', 'document', 'paraview')

        Raises:
            ImportError: If PyVista is not available
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is not available. This module requires PyVista to be installed. "
                "Install with: pip install pyvista"
            )
        self.theme = theme
        pv.set_plot_theme(theme)

    def render_cad_model(
        self,
        geometry_data: Union[str, Path, pv.PolyData, pv.UnstructuredGrid],
        plotter_config: Optional[Dict] = None
    ) -> pv.Plotter:
        """
        Render CAD model from various formats (STEP, STL, VTK).

        Args:
            geometry_data: Path to mesh file or PyVista mesh object
            plotter_config: Configuration dictionary for plotter settings
                - window_size: tuple (width, height), default (1024, 768)
                - background: color string, default 'white'
                - show_edges: bool, default True
                - color: mesh color, default 'lightblue'
                - opacity: float 0-1, default 1.0
                - lighting: bool, default True
                - smooth_shading: bool, default True

        Returns:
            pv.Plotter: Configured PyVista plotter object
        """
        # Default configuration
        config = {
            'window_size': (1024, 768),
            'background': 'white',
            'show_edges': True,
            'color': 'lightblue',
            'opacity': 1.0,
            'lighting': True,
            'smooth_shading': True,
        }
        if plotter_config:
            config.update(plotter_config)

        # Load mesh if path is provided
        mesh = self._load_mesh(geometry_data)

        # Create plotter
        plotter = pv.Plotter(
            window_size=config['window_size'],
            off_screen=config.get('off_screen', False)
        )
        plotter.set_background(config['background'])

        # Add mesh with configuration
        plotter.add_mesh(
            mesh,
            color=config['color'],
            show_edges=config['show_edges'],
            opacity=config['opacity'],
            smooth_shading=config['smooth_shading'],
            lighting=config['lighting']
        )

        # Configure lighting
        if config['lighting']:
            plotter.add_light(pv.Light(position=(10, 10, 10), light_type='camera'))
            plotter.enable_shadows()

        # Set camera position (isometric view by default)
        plotter.camera_position = 'iso'
        plotter.reset_camera()

        return plotter

    def render_cfd_results(
        self,
        result_data: Union[str, Path, pv.UnstructuredGrid],
        field: str = 'velocity',
        **kwargs
    ) -> pv.Plotter:
        """
        Render CFD simulation results with scalar/vector field visualization.

        Args:
            result_data: Path to VTK results file or PyVista grid object
            field: Field name to visualize ('velocity', 'pressure', 'temperature', etc.)
            **kwargs: Additional configuration
                - show_scalar_bar: bool, default True
                - scalar_bar_args: dict, colorbar configuration
                - cmap: colormap name, default 'jet'
                - clim: tuple (min, max), color limits
                - show_streamlines: bool, default False for velocity fields
                - slice_normal: tuple (x, y, z), normal vector for slice plane
                - n_slices: int, number of slice planes

        Returns:
            pv.Plotter: Configured plotter with CFD visualization
        """
        # Load results
        mesh = self._load_mesh(result_data)

        # Verify field exists
        if field not in mesh.array_names:
            available = ', '.join(mesh.array_names)
            raise ValueError(f"Field '{field}' not found. Available: {available}")

        # Create plotter
        plotter = pv.Plotter(window_size=kwargs.get('window_size', (1024, 768)))
        plotter.set_background(kwargs.get('background', 'white'))

        # Determine if field is scalar or vector
        field_data = mesh[field]
        is_vector = len(field_data.shape) > 1 and field_data.shape[1] == 3

        # Configure scalar bar
        scalar_bar_args = kwargs.get('scalar_bar_args', {})
        default_sbar = {
            'title': field.capitalize(),
            'title_font_size': 20,
            'label_font_size': 16,
            'n_labels': 5,
            'italic': False,
            'fmt': '%.2e',
            'position_x': 0.05,
            'position_y': 0.05,
        }
        scalar_bar_args = {**default_sbar, **scalar_bar_args}

        # Handle slice planes
        if 'slice_normal' in kwargs or 'n_slices' in kwargs:
            normal = kwargs.get('slice_normal', (1, 0, 0))
            n_slices = kwargs.get('n_slices', 5)
            slices = mesh.slice_along_axis(n=n_slices, axis=normal)

            plotter.add_mesh(
                slices,
                scalars=field,
                cmap=kwargs.get('cmap', 'jet'),
                clim=kwargs.get('clim'),
                show_scalar_bar=kwargs.get('show_scalar_bar', True),
                scalar_bar_args=scalar_bar_args,
                opacity=kwargs.get('opacity', 1.0)
            )
        else:
            # Add full mesh
            plotter.add_mesh(
                mesh,
                scalars=field,
                cmap=kwargs.get('cmap', 'jet'),
                clim=kwargs.get('clim'),
                show_scalar_bar=kwargs.get('show_scalar_bar', True),
                scalar_bar_args=scalar_bar_args,
                opacity=kwargs.get('opacity', 0.8)
            )

        # Add streamlines for velocity fields
        if is_vector and kwargs.get('show_streamlines', False):
            streamlines = mesh.streamlines(
                vectors=field,
                n_points=kwargs.get('n_streamlines', 100),
                max_time=kwargs.get('streamline_length', 100.0),
                integration_direction='both'
            )
            plotter.add_mesh(
                streamlines,
                line_width=2,
                color='black',
                opacity=0.6
            )

        plotter.camera_position = 'iso'
        plotter.reset_camera()

        return plotter

    def render_mesh_quality(
        self,
        mesh: Union[str, Path, pv.UnstructuredGrid],
        quality_metric: str = 'aspect_ratio',
        threshold: float = 0.3
    ) -> pv.Plotter:
        """
        Display mesh with quality coloring and highlight bad elements.

        Args:
            mesh: Mesh object or path to mesh file
            quality_metric: Metric to compute ('aspect_ratio', 'skewness', 'jacobian')
            threshold: Threshold below which elements are considered bad

        Returns:
            pv.Plotter: Plotter with mesh quality visualization
        """
        # Load mesh
        mesh_obj = self._load_mesh(mesh)

        # Compute quality metric
        if quality_metric == 'aspect_ratio':
            quality = mesh_obj.compute_cell_quality(quality_measure='aspect_ratio')
            metric_name = 'CellQuality'
        elif quality_metric == 'skewness':
            quality = mesh_obj.compute_cell_quality(quality_measure='skewness')
            metric_name = 'CellQuality'
        elif quality_metric == 'jacobian':
            quality = mesh_obj.compute_cell_quality(quality_measure='scaled_jacobian')
            metric_name = 'CellQuality'
        else:
            raise ValueError(f"Unknown quality metric: {quality_metric}")

        # Create plotter
        plotter = pv.Plotter(window_size=(1024, 768))
        plotter.set_background('white')

        # Add mesh with quality coloring
        plotter.add_mesh(
            quality,
            scalars=metric_name,
            cmap='RdYlGn',  # Red = bad, Green = good
            show_scalar_bar=True,
            scalar_bar_args={
                'title': f'{quality_metric.replace("_", " ").title()}',
                'title_font_size': 20,
                'label_font_size': 16,
            },
            show_edges=True,
            edge_color='gray',
            opacity=0.9
        )

        # Highlight bad elements
        if metric_name in quality.array_names:
            bad_cells = quality.threshold(
                value=threshold,
                scalars=metric_name,
                invert=True
            )

            if bad_cells.n_cells > 0:
                plotter.add_mesh(
                    bad_cells,
                    color='red',
                    opacity=1.0,
                    show_edges=True,
                    edge_color='darkred',
                    line_width=2,
                    label=f'Poor Quality (<{threshold})'
                )
                plotter.add_legend()

        plotter.camera_position = 'iso'
        plotter.reset_camera()

        return plotter

    def add_annotations(
        self,
        plotter: pv.Plotter,
        labels: Dict[str, Union[Tuple, Dict]]
    ) -> pv.Plotter:
        """
        Add text labels, arrows, and dimension annotations to plotter.

        Args:
            plotter: Existing PyVista plotter
            labels: Dictionary of annotations
                Format: {
                    'label_name': {
                        'type': 'text' | 'arrow' | 'dimension',
                        'position': (x, y, z),
                        'text': 'Label text',
                        'color': 'black',
                        'font_size': 12,
                        # For arrows:
                        'direction': (dx, dy, dz),
                        'scale': 1.0,
                        # For dimensions:
                        'end_position': (x2, y2, z2)
                    }
                }

        Returns:
            pv.Plotter: Modified plotter with annotations
        """
        for name, config in labels.items():
            anno_type = config.get('type', 'text')
            position = config.get('position', (0, 0, 0))

            if anno_type == 'text':
                plotter.add_point_labels(
                    [position],
                    [config.get('text', name)],
                    font_size=config.get('font_size', 12),
                    text_color=config.get('color', 'black'),
                    point_size=config.get('point_size', 10),
                    point_color=config.get('point_color', 'red'),
                    render_points_as_spheres=True,
                    always_visible=config.get('always_visible', True)
                )

            elif anno_type == 'arrow':
                direction = np.array(config.get('direction', (1, 0, 0)))
                scale = config.get('scale', 1.0)
                direction = direction / np.linalg.norm(direction) * scale

                arrow = pv.Arrow(
                    start=position,
                    direction=direction,
                    scale=config.get('arrow_scale', 0.5)
                )
                plotter.add_mesh(
                    arrow,
                    color=config.get('color', 'red'),
                    opacity=config.get('opacity', 1.0)
                )

                # Add label at arrow tip
                if 'text' in config:
                    label_pos = np.array(position) + direction
                    plotter.add_point_labels(
                        [label_pos],
                        [config['text']],
                        font_size=config.get('font_size', 12),
                        text_color=config.get('color', 'red')
                    )

            elif anno_type == 'dimension':
                end_pos = config.get('end_position', (1, 0, 0))

                # Draw dimension line
                line = pv.Line(position, end_pos)
                plotter.add_mesh(
                    line,
                    color=config.get('color', 'black'),
                    line_width=config.get('line_width', 2)
                )

                # Calculate and display distance
                distance = np.linalg.norm(np.array(end_pos) - np.array(position))
                mid_point = (np.array(position) + np.array(end_pos)) / 2

                dim_text = config.get('text', f'{distance:.2f}')
                plotter.add_point_labels(
                    [mid_point],
                    [dim_text],
                    font_size=config.get('font_size', 12),
                    text_color=config.get('color', 'black')
                )

        return plotter

    def export_image(
        self,
        plotter: pv.Plotter,
        filepath: Union[str, Path],
        resolution: Tuple[int, int] = (1920, 1080),
        transparent_background: bool = False
    ) -> None:
        """
        Export plotter view to high-quality image file.

        Args:
            plotter: PyVista plotter to export
            filepath: Output file path (.png, .jpg, .svg)
            resolution: Image resolution (width, height)
            transparent_background: Use transparent background for PNG
        """
        filepath = Path(filepath)

        # Set window size to desired resolution
        plotter.window_size = resolution

        # Export screenshot
        plotter.screenshot(
            filename=str(filepath),
            transparent_background=transparent_background,
            return_img=False
        )

        print(f"Image exported to: {filepath}")

    def _load_mesh(
        self,
        geometry_data: Union[str, Path, pv.PolyData, pv.UnstructuredGrid, pv.DataSet]
    ) -> pv.DataSet:
        """
        Load mesh from file or return existing mesh object.

        Args:
            geometry_data: File path or mesh object

        Returns:
            pv.DataSet: Loaded mesh
        """
        # If already a PyVista object, return it
        if isinstance(geometry_data, (pv.PolyData, pv.UnstructuredGrid, pv.DataSet)):
            return geometry_data

        # Convert to Path
        filepath = Path(geometry_data)

        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        # Load based on extension
        ext = filepath.suffix.lower()

        if ext in ['.vtk', '.vtu', '.vtp', '.vts', '.vti']:
            mesh = pv.read(str(filepath))
        elif ext == '.stl':
            mesh = pv.read(str(filepath))
        elif ext in ['.ply', '.obj']:
            mesh = pv.read(str(filepath))
        elif ext == '.step' or ext == '.stp':
            # For STEP files, we would need additional libraries like pythonOCC
            # For now, raise an informative error
            raise NotImplementedError(
                "STEP file loading requires additional libraries (pythonOCC). "
                "Please convert to STL, OBJ, or VTK format."
            )
        else:
            # Try generic read
            try:
                mesh = pv.read(str(filepath))
            except Exception as e:
                raise ValueError(f"Unsupported file format: {ext}. Error: {e}")

        return mesh

    @staticmethod
    def create_sample_mesh(mesh_type: str = 'sphere'):
        """
        Create sample mesh for testing and demonstrations.

        Args:
            mesh_type: Type of sample mesh ('sphere', 'cube', 'cylinder', 'cone')

        Returns:
            pv.PolyData: Sample mesh

        Raises:
            ImportError: If PyVista is not available
            ValueError: If mesh_type is unknown
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is not available. Cannot create sample mesh. "
                "Install with: pip install pyvista"
            )

        if mesh_type == 'sphere':
            return pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)
        elif mesh_type == 'cube':
            return pv.Cube()
        elif mesh_type == 'cylinder':
            return pv.Cylinder(radius=0.5, height=2.0)
        elif mesh_type == 'cone':
            return pv.Cone(radius=1.0, height=2.0)
        else:
            raise ValueError(f"Unknown mesh type: {mesh_type}")
