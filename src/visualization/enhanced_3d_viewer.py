"""
Enhanced 3D Viewer with Measurement Tools

Interactive 3D viewer with distance, angle, and area measurement capabilities.
Built on PyVista with additional CAD-specific features.
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path
import numpy as np

try:
    import pyvista as pv
    from pyvista import examples
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    logging.warning("PyVista not installed. Enhanced 3D viewer disabled.")

try:
    import vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

logger = logging.getLogger(__name__)


class MeasurementTool:
    """Measurement tools for 3D CAD models."""

    @staticmethod
    def distance(point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point [x, y, z]
            point2: Second point [x, y, z]

        Returns:
            Distance in model units
        """
        return np.linalg.norm(np.array(point2) - np.array(point1))

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray, degrees: bool = True) -> float:
        """
        Calculate angle between two vectors.

        Args:
            v1: First vector
            v2: Second vector
            degrees: Return result in degrees (True) or radians (False)

        Returns:
            Angle between vectors
        """
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return np.degrees(angle) if degrees else angle

    @staticmethod
    def angle_between_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, degrees: bool = True) -> float:
        """
        Calculate angle formed by three points (p1-p2-p3), with vertex at p2.

        Args:
            p1: First point
            p2: Vertex point
            p3: Third point
            degrees: Return result in degrees

        Returns:
            Angle at p2
        """
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)

        return MeasurementTool.angle_between_vectors(v1, v2, degrees)

    @staticmethod
    def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate area of a triangle defined by three points.

        Args:
            p1, p2, p3: Triangle vertices

        Returns:
            Triangle area
        """
        v1 = np.array(p2) - np.array(p1)
        v2 = np.array(p3) - np.array(p1)

        cross = np.cross(v1, v2)
        area = 0.5 * np.linalg.norm(cross)

        return area

    @staticmethod
    def surface_area(mesh) -> float:
        """
        Calculate total surface area of a mesh.

        Args:
            mesh: PyVista mesh object

        Returns:
            Total surface area
        """
        if not HAS_PYVISTA:
            return 0.0

        if hasattr(mesh, 'area'):
            return mesh.area
        else:
            # Manual calculation
            if not hasattr(mesh, 'faces'):
                return 0.0

            total_area = 0.0
            points = mesh.points

            # Iterate through faces
            face_array = mesh.faces
            i = 0
            while i < len(face_array):
                n_points = face_array[i]
                i += 1

                if n_points >= 3:
                    # Get triangle vertices
                    p1 = points[face_array[i]]
                    p2 = points[face_array[i + 1]]
                    p3 = points[face_array[i + 2]]

                    # Add triangle area
                    total_area += MeasurementTool.triangle_area(p1, p2, p3)

                i += n_points

            return total_area

    @staticmethod
    def bounding_box_volume(mesh) -> float:
        """
        Calculate bounding box volume.

        Args:
            mesh: PyVista mesh object

        Returns:
            Bounding box volume
        """
        if not HAS_PYVISTA:
            return 0.0

        bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        dx = bounds[1] - bounds[0]
        dy = bounds[3] - bounds[2]
        dz = bounds[5] - bounds[4]

        return dx * dy * dz


class Enhanced3DViewer:
    """
    Enhanced 3D viewer with measurement and editing tools.

    Features:
    - Interactive 3D visualization
    - Distance measurement
    - Angle measurement
    - Area measurement
    - Point picking
    - Annotation tools
    - Multiple view modes
    - Screenshot export
    """

    def __init__(self):
        """Initialize the enhanced 3D viewer."""
        if not HAS_PYVISTA:
            raise ImportError("PyVista is required. Install with: pip install pyvista")

        self.plotter = None
        self.mesh = None
        self.measurement_points = []
        self.annotations = []

    def load_mesh(self, mesh_path: Path) -> bool:
        """
        Load mesh from file.

        Args:
            mesh_path: Path to mesh file (STL, OBJ, PLY, VTK, etc.)

        Returns:
            True if loaded successfully
        """
        try:
            self.mesh = pv.read(str(mesh_path))
            logger.info(f"Loaded mesh: {mesh_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load mesh: {e}")
            return False

    def create_plotter(
        self,
        off_screen: bool = False,
        window_size: Tuple[int, int] = (1024, 768)
    ) -> pv.Plotter:
        """
        Create PyVista plotter.

        Args:
            off_screen: Render off-screen (for screenshots)
            window_size: Window dimensions

        Returns:
            PyVista plotter instance
        """
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        return self.plotter

    def render_mesh(
        self,
        mesh=None,
        color: str = 'lightblue',
        show_edges: bool = True,
        opacity: float = 1.0,
        lighting: bool = True
    ) -> None:
        """
        Render mesh in the plotter.

        Args:
            mesh: Mesh to render (uses loaded mesh if None)
            color: Mesh color
            show_edges: Show edge lines
            opacity: Mesh opacity (0-1)
            lighting: Enable lighting
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            logger.error("No mesh to render")
            return

        if self.plotter is None:
            self.create_plotter()

        self.plotter.add_mesh(
            mesh,
            color=color,
            show_edges=show_edges,
            opacity=opacity,
            lighting=lighting
        )

    def measure_distance_interactive(self) -> Optional[float]:
        """
        Interactive distance measurement.

        User picks two points, distance is calculated.

        Returns:
            Distance between picked points or None
        """
        if self.plotter is None or self.mesh is None:
            logger.error("Plotter or mesh not initialized")
            return None

        logger.info("Click two points to measure distance")

        # Enable point picking
        picked_points = self.plotter.enable_point_picking(
            callback=None,
            use_mesh=True,
            show_message=True
        )

        # For non-interactive mode, return None
        # In interactive mode, this would wait for user input
        return None

    def measure_distance_points(self, point1: np.ndarray, point2: np.ndarray) -> Dict[str, Any]:
        """
        Measure distance between two points and add visualization.

        Args:
            point1: First point [x, y, z]
            point2: Second point [x, y, z]

        Returns:
            Measurement results dictionary
        """
        distance = MeasurementTool.distance(point1, point2)

        # Add visualization
        if self.plotter is not None:
            # Draw line between points
            line = pv.Line(point1, point2)
            self.plotter.add_mesh(line, color='red', line_width=3, label='Distance')

            # Add point markers
            self.plotter.add_points(
                np.array([point1, point2]),
                color='red',
                point_size=10,
                render_points_as_spheres=True
            )

            # Add text annotation
            midpoint = (np.array(point1) + np.array(point2)) / 2
            label = f"{distance:.2f} mm"
            self.plotter.add_point_labels(
                [midpoint],
                [label],
                font_size=12,
                text_color='red',
                shape_opacity=0.7
            )

        return {
            'type': 'distance',
            'point1': point1.tolist(),
            'point2': point2.tolist(),
            'distance': distance,
            'units': 'mm'
        }

    def measure_angle_points(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> Dict[str, Any]:
        """
        Measure angle between three points (p1-p2-p3).

        Args:
            p1: First point
            p2: Vertex point
            p3: Third point

        Returns:
            Measurement results dictionary
        """
        angle = MeasurementTool.angle_between_points(p1, p2, p3, degrees=True)

        # Add visualization
        if self.plotter is not None:
            # Draw lines
            line1 = pv.Line(p2, p1)
            line2 = pv.Line(p2, p3)
            self.plotter.add_mesh(line1, color='blue', line_width=2)
            self.plotter.add_mesh(line2, color='blue', line_width=2)

            # Add point markers
            self.plotter.add_points(
                np.array([p1, p2, p3]),
                color='blue',
                point_size=10,
                render_points_as_spheres=True
            )

            # Add angle annotation
            label = f"{angle:.2f}°"
            self.plotter.add_point_labels(
                [p2],
                [label],
                font_size=12,
                text_color='blue',
                shape_opacity=0.7
            )

        return {
            'type': 'angle',
            'point1': p1.tolist(),
            'vertex': p2.tolist(),
            'point3': p3.tolist(),
            'angle_degrees': angle,
            'angle_radians': np.radians(angle)
        }

    def measure_area_triangle(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
    ) -> Dict[str, Any]:
        """
        Measure area of triangle.

        Args:
            p1, p2, p3: Triangle vertices

        Returns:
            Measurement results dictionary
        """
        area = MeasurementTool.triangle_area(p1, p2, p3)

        # Add visualization
        if self.plotter is not None:
            # Create triangle mesh
            points = np.array([p1, p2, p3])
            faces = np.array([[3, 0, 1, 2]])
            triangle_mesh = pv.PolyData(points, faces)

            self.plotter.add_mesh(
                triangle_mesh,
                color='green',
                opacity=0.5,
                show_edges=True,
                edge_color='darkgreen',
                line_width=2
            )

            # Add centroid label
            centroid = np.mean(points, axis=0)
            label = f"{area:.2f} mm²"
            self.plotter.add_point_labels(
                [centroid],
                [label],
                font_size=12,
                text_color='green',
                shape_opacity=0.7
            )

        return {
            'type': 'area',
            'vertices': [p1.tolist(), p2.tolist(), p3.tolist()],
            'area': area,
            'units': 'mm²'
        }

    def get_mesh_info(self, mesh=None) -> Dict[str, Any]:
        """
        Get comprehensive mesh information.

        Args:
            mesh: Mesh to analyze (uses loaded mesh if None)

        Returns:
            Dictionary with mesh statistics
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            return {}

        bounds = mesh.bounds
        center = mesh.center

        info = {
            'n_points': mesh.n_points,
            'n_cells': mesh.n_cells,
            'bounds': {
                'x': [bounds[0], bounds[1]],
                'y': [bounds[2], bounds[3]],
                'z': [bounds[4], bounds[5]]
            },
            'dimensions': {
                'x': bounds[1] - bounds[0],
                'y': bounds[3] - bounds[2],
                'z': bounds[5] - bounds[4]
            },
            'center': center.tolist(),
            'surface_area': MeasurementTool.surface_area(mesh),
            'bounding_box_volume': MeasurementTool.bounding_box_volume(mesh)
        }

        return info

    def add_axes(self) -> None:
        """Add coordinate axes to the viewer."""
        if self.plotter is not None:
            self.plotter.add_axes()

    def add_ruler(self) -> None:
        """Add ruler/scale bar."""
        if self.plotter is not None:
            self.plotter.add_ruler()

    def set_camera_position(self, position: str = 'iso') -> None:
        """
        Set camera to predefined position.

        Args:
            position: 'iso', 'top', 'front', 'side', 'bottom'
        """
        if self.plotter is None:
            return

        if position == 'iso':
            self.plotter.view_isometric()
        elif position == 'top':
            self.plotter.view_xy()
        elif position == 'front':
            self.plotter.view_xz()
        elif position == 'side':
            self.plotter.view_yz()

    def export_screenshot(self, output_path: Path, resolution: Tuple[int, int] = (1920, 1080)) -> Path:
        """
        Export screenshot of current view.

        Args:
            output_path: Output image path
            resolution: Image resolution (width, height)

        Returns:
            Path to exported screenshot
        """
        if self.plotter is None:
            logger.error("Plotter not initialized")
            return None

        self.plotter.window_size = resolution
        self.plotter.screenshot(str(output_path))

        logger.info(f"Screenshot exported: {output_path}")
        return output_path

    def show(self, **kwargs) -> None:
        """Display the viewer window."""
        if self.plotter is not None:
            self.plotter.show(**kwargs)

    def close(self) -> None:
        """Close the viewer."""
        if self.plotter is not None:
            self.plotter.close()
