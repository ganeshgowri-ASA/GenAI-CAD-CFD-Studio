"""
Universal Importer Module

Provides a unified interface for importing CAD and mesh files in multiple formats.
Automatically detects file format and routes to appropriate parser.
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import warnings

from .dxf_parser import DXFParser
from .step_handler import STEPHandler
from .stl_handler import STLHandler
from .mesh_converter import MeshConverter


class UniversalImporter:
    """
    Universal file importer with automatic format detection.

    Supports:
    - DXF files (2D/3D drawings)
    - STEP files (3D CAD models)
    - STL files (3D mesh)
    - 20+ mesh formats via meshio (VTK, MSH, ANSYS, etc.)

    Returns unified geometry dictionary with:
    - vertices: NumPy array of vertex coordinates
    - faces: NumPy array of face indices (for meshes)
    - bounds: Bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
    - volume: Volume (if applicable)
    - metadata: Format-specific metadata
    """

    # File extension to format mapping
    FORMAT_MAP = {
        # CAD formats
        'dxf': 'dxf',
        'step': 'step',
        'stp': 'step',
        'iges': 'step',
        'igs': 'step',
        'brep': 'step',
        # Mesh formats
        'stl': 'stl',
        'obj': 'mesh',
        'ply': 'mesh',
        'off': 'mesh',
        # FEA/CFD mesh formats
        'vtk': 'mesh',
        'vtu': 'mesh',
        'vts': 'mesh',
        'vtr': 'mesh',
        'vtp': 'mesh',
        'pvtu': 'mesh',
        'msh': 'mesh',
        'ans': 'mesh',
        'inp': 'mesh',
        'cgns': 'mesh',
        'xml': 'mesh',
        'e': 'mesh',
        'exo': 'mesh',
        'f3grid': 'mesh',
        'h5m': 'mesh',
        'mdpa': 'mesh',
        'med': 'mesh',
        'bdf': 'mesh',
        'nas': 'mesh',
        'dat': 'mesh',
        'ugrid': 'mesh',
        'xdmf': 'mesh',
        'xmf': 'mesh'
    }

    def __init__(self):
        """Initialize universal importer."""
        self.dxf_parser = None
        self.step_handler = None
        self.stl_handler = None
        self.mesh_converter = None
        self._progress_callback = None

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """
        Set a progress callback function for large file operations.

        Args:
            callback: Function that takes (message: str, progress: float) where
                     progress is between 0.0 and 1.0
        """
        self._progress_callback = callback

    def _report_progress(self, message: str, progress: float):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(message, progress)

    def detect_format(self, filepath: str) -> str:
        """
        Detect file format from extension.

        Args:
            filepath: Path to the file

        Returns:
            Format type: 'dxf', 'step', 'stl', or 'mesh'

        Raises:
            ValueError: If format is not supported
        """
        ext = Path(filepath).suffix.lstrip('.').lower()

        if ext not in self.FORMAT_MAP:
            raise ValueError(
                f"Unsupported file format: .{ext}\n"
                f"Supported formats: {', '.join(sorted(set(self.FORMAT_MAP.keys())))}"
            )

        return self.FORMAT_MAP[ext]

    def import_file(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """
        Import a CAD or mesh file and return unified geometry dictionary.

        Args:
            filepath: Path to the file to import
            **kwargs: Additional arguments passed to specific parsers

        Returns:
            Dictionary containing:
                - 'vertices': NumPy array (N x 3)
                - 'faces': NumPy array (M x 3 or variable) [for meshes]
                - 'edges': List of edge definitions [for 2D/wireframe]
                - 'bounds': Tuple (xmin, xmax, ymin, ymax, zmin, zmax)
                - 'volume': float [if applicable]
                - 'surface_area': float [if applicable]
                - 'metadata': Dict with format-specific data
                - 'format': str (detected format)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        self._report_progress(f"Detecting format for {filepath.name}", 0.1)
        format_type = self.detect_format(str(filepath))

        self._report_progress(f"Loading {format_type.upper()} file", 0.2)

        if format_type == 'dxf':
            return self._import_dxf(str(filepath), **kwargs)
        elif format_type == 'step':
            return self._import_step(str(filepath), **kwargs)
        elif format_type == 'stl':
            return self._import_stl(str(filepath), **kwargs)
        elif format_type == 'mesh':
            return self._import_mesh(str(filepath), **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _import_dxf(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Import DXF file."""
        if self.dxf_parser is None:
            self.dxf_parser = DXFParser()

        self._report_progress("Parsing DXF geometry", 0.4)
        dxf_data = self.dxf_parser.parse(filepath)

        self._report_progress("Converting DXF to unified format", 0.7)

        # Extract vertices from various entity types
        vertices = []

        # Add vertices from lines
        for line in dxf_data.get('lines', []):
            vertices.extend([line['start'], line['end']])

        # Add vertices from circles (sample points)
        for circle in dxf_data.get('circles', []):
            center = circle['center']
            radius = circle['radius']
            # Sample 32 points around the circle
            angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
            for angle in angles:
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                vertices.append([x, y, center[2]])

        # Add vertices from polylines
        for polyline in dxf_data.get('polylines', []):
            vertices.extend(polyline['points'])

        # Convert to numpy array
        if vertices:
            vertices_array = np.array(vertices, dtype=float)
        else:
            vertices_array = np.array([], dtype=float).reshape(0, 3)

        self._report_progress("DXF import complete", 1.0)

        return {
            'vertices': vertices_array,
            'faces': None,  # DXF is primarily 2D/wireframe
            'edges': self._extract_dxf_edges(dxf_data),
            'bounds': dxf_data['bounds'],
            'volume': None,  # Not applicable for 2D
            'surface_area': None,
            'metadata': {
                'format': 'dxf',
                'dxf_version': dxf_data['dxf_version'],
                'units': dxf_data['units'],
                'layers': dxf_data['layers'],
                'entity_counts': {
                    'lines': len(dxf_data.get('lines', [])),
                    'arcs': len(dxf_data.get('arcs', [])),
                    'circles': len(dxf_data.get('circles', [])),
                    'polylines': len(dxf_data.get('polylines', []))
                }
            },
            'format': 'dxf',
            'raw_data': dxf_data  # Include raw DXF data for advanced users
        }

    def _extract_dxf_edges(self, dxf_data: Dict[str, Any]) -> list:
        """Extract edge definitions from DXF data."""
        edges = []

        # Lines are direct edges
        for line in dxf_data.get('lines', []):
            edges.append({
                'type': 'line',
                'start': line['start'],
                'end': line['end'],
                'layer': line['layer']
            })

        # Arcs
        for arc in dxf_data.get('arcs', []):
            edges.append({
                'type': 'arc',
                'center': arc['center'],
                'radius': arc['radius'],
                'start_angle': arc['start_angle'],
                'end_angle': arc['end_angle'],
                'layer': arc['layer']
            })

        # Circles
        for circle in dxf_data.get('circles', []):
            edges.append({
                'type': 'circle',
                'center': circle['center'],
                'radius': circle['radius'],
                'layer': circle['layer']
            })

        # Polylines
        for polyline in dxf_data.get('polylines', []):
            edges.append({
                'type': 'polyline',
                'points': polyline['points'],
                'is_closed': polyline['is_closed'],
                'layer': polyline['layer']
            })

        return edges

    def _import_step(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Import STEP file."""
        if self.step_handler is None:
            self.step_handler = STEPHandler()

        self._report_progress("Loading STEP model", 0.4)
        shape = self.step_handler.import_step(filepath)

        self._report_progress("Extracting geometry properties", 0.7)
        properties = self.step_handler.get_properties(shape)

        # Extract vertices
        vertices = self.step_handler.extract_vertices(shape)

        self._report_progress("STEP import complete", 1.0)

        return {
            'vertices': vertices,
            'faces': None,  # STEP is CAD, not mesh-based
            'edges': None,
            'bounds': properties['bounds'],
            'volume': properties['volume'],
            'surface_area': properties['surface_area'],
            'metadata': {
                'format': 'step',
                'center_of_mass': properties['center_of_mass'],
                'topology': properties['topology'],
                'dimensions': properties['dimensions']
            },
            'format': 'step',
            'shape': shape  # Include OCC shape for advanced users
        }

    def _import_stl(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Import STL file."""
        if self.stl_handler is None:
            self.stl_handler = STLHandler()

        self._report_progress("Loading STL mesh", 0.4)
        mesh = self.stl_handler.load_mesh(filepath, **kwargs)

        self._report_progress("Calculating mesh properties", 0.7)
        properties = self.stl_handler.calculate_properties(mesh)
        vertices, faces = self.stl_handler.get_vertices_and_faces(mesh)

        self._report_progress("STL import complete", 1.0)

        return {
            'vertices': vertices,
            'faces': faces,
            'edges': None,
            'bounds': properties['bounds'],
            'volume': properties['volume'],
            'surface_area': properties['surface_area'],
            'metadata': {
                'format': 'stl',
                'is_watertight': properties['is_watertight'],
                'is_convex': properties['is_convex'],
                'vertex_count': properties['vertex_count'],
                'face_count': properties['face_count'],
                'edge_count': properties['edge_count'],
                'center_mass': properties['center_mass'],
                'centroid': properties['centroid'],
                'extents': properties['extents']
            },
            'format': 'stl',
            'mesh': mesh  # Include trimesh object for advanced users
        }

    def _import_mesh(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """Import generic mesh file using meshio."""
        if self.mesh_converter is None:
            self.mesh_converter = MeshConverter()

        self._report_progress("Loading mesh file", 0.4)
        mesh = self.mesh_converter.read(filepath)

        self._report_progress("Extracting mesh information", 0.7)
        info = self.mesh_converter.get_mesh_info(mesh)

        # Get vertices
        vertices = mesh.points

        # Extract faces from cells
        faces = self._extract_faces_from_cells(mesh.cells)

        self._report_progress("Mesh import complete", 1.0)

        # Calculate volume if possible (only for tetrahedral meshes)
        volume = None
        surface_area = None

        return {
            'vertices': vertices,
            'faces': faces,
            'edges': None,
            'bounds': (
                info['bounds']['xmin'], info['bounds']['xmax'],
                info['bounds']['ymin'], info['bounds']['ymax'],
                info['bounds']['zmin'], info['bounds']['zmax']
            ),
            'volume': volume,
            'surface_area': surface_area,
            'metadata': {
                'format': 'mesh',
                'num_points': info['num_points'],
                'num_cells': info['num_cells'],
                'cell_types': info['cell_types'],
                'cell_counts': info['cell_counts'],
                'dimension': info['dimension'],
                'point_data_fields': info['point_data_keys'],
                'cell_data_fields': info['cell_data_keys']
            },
            'format': 'mesh',
            'mesh': mesh  # Include meshio object for advanced users
        }

    def _extract_faces_from_cells(self, cells) -> Optional[np.ndarray]:
        """Extract face connectivity from meshio cells."""
        all_faces = []

        for cell_block in cells:
            cell_type = cell_block.type
            cell_data = cell_block.data

            # Handle different cell types
            if cell_type == 'triangle':
                all_faces.append(cell_data)
            elif cell_type == 'quad':
                # Convert quads to triangles
                for quad in cell_data:
                    all_faces.append([quad[0], quad[1], quad[2]])
                    all_faces.append([quad[0], quad[2], quad[3]])
            elif cell_type == 'tetra':
                # Extract surface triangles from tetrahedra
                for tet in cell_data:
                    # 4 triangular faces per tetrahedron
                    all_faces.extend([
                        [tet[0], tet[1], tet[2]],
                        [tet[0], tet[1], tet[3]],
                        [tet[0], tet[2], tet[3]],
                        [tet[1], tet[2], tet[3]]
                    ])

        if all_faces:
            return np.vstack(all_faces)
        return None

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        """
        Get dictionary of all supported file formats.

        Returns:
            Dictionary mapping file extension to format category
        """
        return UniversalImporter.FORMAT_MAP.copy()

    def is_format_supported(self, filepath: str) -> bool:
        """
        Check if a file format is supported.

        Args:
            filepath: File path to check

        Returns:
            True if format is supported, False otherwise
        """
        try:
            self.detect_format(filepath)
            return True
        except ValueError:
            return False

    def get_format_info(self, filepath: str) -> Dict[str, str]:
        """
        Get information about a file's format.

        Args:
            filepath: File path to check

        Returns:
            Dictionary with format information
        """
        try:
            format_type = self.detect_format(filepath)
            ext = Path(filepath).suffix.lstrip('.').lower()

            format_names = {
                'dxf': 'AutoCAD DXF (Drawing Exchange Format)',
                'step': 'STEP (Standard for Exchange of Product Data)',
                'stl': 'STL (Stereolithography)',
                'mesh': 'Generic Mesh Format'
            }

            return {
                'extension': ext,
                'format_type': format_type,
                'format_name': format_names.get(format_type, 'Unknown'),
                'supported': True
            }
        except ValueError as e:
            return {
                'extension': Path(filepath).suffix.lstrip('.').lower(),
                'format_type': None,
                'format_name': 'Unsupported',
                'supported': False,
                'error': str(e)
            }
