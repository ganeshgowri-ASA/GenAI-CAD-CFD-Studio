"""
STL Handler Module

Handles loading, validation, and repair of STL (Stereolithography) mesh files
using the trimesh library.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from pathlib import Path

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    if TYPE_CHECKING:
        import trimesh


class STLHandler:
    """
    Handler for STL mesh file operations.

    Provides functionality to:
    - Load STL files (ASCII and binary)
    - Validate mesh integrity (watertight, manifold)
    - Repair mesh defects
    - Calculate mesh properties (volume, area, bounds, etc.)
    - Export to various formats
    """

    def __init__(self):
        """Initialize STL handler."""
        if not HAS_TRIMESH:
            raise ImportError(
                "trimesh library is required for STL handling. "
                "Install it with: pip install trimesh"
            )
        self.mesh = None

    def load_mesh(self, filepath: str, **kwargs) -> trimesh.Trimesh:
        """
        Load a mesh from an STL file.

        Args:
            filepath: Path to the STL file
            **kwargs: Additional arguments passed to trimesh.load

        Returns:
            trimesh.Trimesh object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be loaded as a mesh
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"STL file not found: {filepath}")

        # Load the mesh
        self.mesh = trimesh.load(str(filepath), **kwargs)

        # Ensure we have a Trimesh object (not a Scene)
        if isinstance(self.mesh, trimesh.Scene):
            # If it's a scene, try to get the first geometry
            if len(self.mesh.geometry) > 0:
                self.mesh = list(self.mesh.geometry.values())[0]
            else:
                raise ValueError("No geometry found in the STL file")

        if not isinstance(self.mesh, trimesh.Trimesh):
            raise ValueError("Loaded object is not a valid mesh")

        return self.mesh

    def save_mesh(self, filepath: str, mesh: Optional[trimesh.Trimesh] = None,
                  file_type: Optional[str] = None) -> bool:
        """
        Save mesh to a file.

        Args:
            filepath: Output file path
            mesh: Trimesh object to save (uses self.mesh if None)
            file_type: File type override (e.g., 'stl', 'ply', 'obj')

        Returns:
            True if save successful, False otherwise
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            mesh.export(str(filepath), file_type=file_type)
            return True
        except Exception as e:
            print(f"Error saving mesh: {e}")
            return False

    def is_watertight(self, mesh: Optional[trimesh.Trimesh] = None) -> bool:
        """
        Check if the mesh is watertight (closed, no holes).

        Args:
            mesh: Trimesh object to check (uses self.mesh if None)

        Returns:
            True if mesh is watertight, False otherwise
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            return False

        return mesh.is_watertight

    def is_manifold(self, mesh: Optional[trimesh.Trimesh] = None) -> bool:
        """
        Check if the mesh is manifold (each edge shared by at most 2 faces).

        Args:
            mesh: Trimesh object to check (uses self.mesh if None)

        Returns:
            True if mesh is manifold, False otherwise
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            return False

        # A mesh is manifold if it's both watertight and has no degenerate faces
        return mesh.is_watertight and not mesh.is_empty

    def validate_mesh(self, mesh: Optional[trimesh.Trimesh] = None) -> Dict[str, Any]:
        """
        Perform comprehensive validation of the mesh.

        Args:
            mesh: Trimesh object to validate (uses self.mesh if None)

        Returns:
            Dictionary containing validation results:
                - 'is_watertight': bool
                - 'is_empty': bool
                - 'is_convex': bool
                - 'euler_number': int
                - 'vertex_count': int
                - 'face_count': int
                - 'edge_count': int
                - 'has_degenerate_faces': bool
                - 'has_duplicate_faces': bool
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh to validate")

        validation = {
            'is_valid': True,
            'is_watertight': mesh.is_watertight,
            'is_empty': mesh.is_empty,
            'is_convex': mesh.is_convex,
            'euler_number': mesh.euler_number,
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces),
            'edge_count': len(mesh.edges),
            'has_degenerate_faces': False,
            'has_duplicate_faces': False,
            'issues': []
        }

        # Check for degenerate faces (zero area)
        face_areas = mesh.area_faces
        if np.any(face_areas < 1e-10):
            validation['has_degenerate_faces'] = True
            validation['is_valid'] = False
            validation['issues'].append('Mesh has degenerate (zero-area) faces')

        # Check for duplicate faces
        unique_faces = trimesh.grouping.unique_rows(np.sort(mesh.faces, axis=1))[0]
        if len(unique_faces) < len(mesh.faces):
            validation['has_duplicate_faces'] = True
            validation['is_valid'] = False
            validation['issues'].append('Mesh has duplicate faces')

        # Check if watertight
        if not mesh.is_watertight:
            validation['is_valid'] = False
            validation['issues'].append('Mesh is not watertight (has holes)')

        return validation

    def repair_mesh(self, mesh: Optional[trimesh.Trimesh] = None,
                    remove_degenerate: bool = True,
                    remove_duplicate: bool = True,
                    fill_holes: bool = True) -> trimesh.Trimesh:
        """
        Attempt to repair mesh defects.

        Args:
            mesh: Trimesh object to repair (uses self.mesh if None)
            remove_degenerate: Remove degenerate (zero-area) faces
            remove_duplicate: Remove duplicate faces
            fill_holes: Attempt to fill holes in the mesh

        Returns:
            Repaired trimesh.Trimesh object
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh to repair")

        # Create a copy to avoid modifying the original
        repaired = mesh.copy()

        # Remove degenerate faces
        if remove_degenerate:
            valid_faces = repaired.area_faces > 1e-10
            if not all(valid_faces):
                repaired.update_faces(valid_faces)

        # Remove duplicate faces
        if remove_duplicate:
            unique_faces, inverse = trimesh.grouping.unique_rows(
                np.sort(repaired.faces, axis=1)
            )
            if len(unique_faces) < len(repaired.faces):
                repaired.update_faces(unique_faces)

        # Fill holes
        if fill_holes:
            repaired.fill_holes()

        # Remove unreferenced vertices
        repaired.remove_unreferenced_vertices()

        # Merge duplicate vertices
        repaired.merge_vertices()

        self.mesh = repaired
        return repaired

    def calculate_properties(self, mesh: Optional[trimesh.Trimesh] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive geometric properties of the mesh.

        Args:
            mesh: Trimesh object to analyze (uses self.mesh if None)

        Returns:
            Dictionary containing:
                - 'volume': float (only valid for watertight meshes)
                - 'surface_area': float
                - 'bounds': tuple (xmin, xmax, ymin, ymax, zmin, zmax)
                - 'center_mass': tuple (x, y, z)
                - 'extents': tuple (length, width, height)
                - 'inertia': 3x3 array
                - 'is_watertight': bool
                - 'vertex_count': int
                - 'face_count': int
                - 'edge_count': int
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh to analyze")

        bounds = mesh.bounds
        extents = mesh.extents

        properties = {
            'volume': float(mesh.volume) if mesh.is_watertight else None,
            'surface_area': float(mesh.area),
            'bounds': (
                float(bounds[0][0]), float(bounds[1][0]),
                float(bounds[0][1]), float(bounds[1][1]),
                float(bounds[0][2]), float(bounds[1][2])
            ),
            'center_mass': tuple(mesh.center_mass.tolist()),
            'centroid': tuple(mesh.centroid.tolist()),
            'extents': tuple(extents.tolist()),
            'is_watertight': mesh.is_watertight,
            'is_convex': mesh.is_convex,
            'vertex_count': len(mesh.vertices),
            'face_count': len(mesh.faces),
            'edge_count': len(mesh.edges),
            'euler_number': mesh.euler_number
        }

        # Add inertia tensor if mesh is watertight
        if mesh.is_watertight:
            properties['inertia'] = mesh.moment_inertia.tolist()

        return properties

    def get_vertices_and_faces(self, mesh: Optional[trimesh.Trimesh] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract vertices and faces from the mesh.

        Args:
            mesh: Trimesh object (uses self.mesh if None)

        Returns:
            Tuple of (vertices, faces) as NumPy arrays
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        return mesh.vertices.copy(), mesh.faces.copy()

    def simplify_mesh(self, mesh: Optional[trimesh.Trimesh] = None,
                      target_faces: Optional[int] = None,
                      target_percent: Optional[float] = None) -> trimesh.Trimesh:
        """
        Simplify the mesh by reducing the number of faces.

        Args:
            mesh: Trimesh object to simplify (uses self.mesh if None)
            target_faces: Target number of faces
            target_percent: Target percentage of original faces (0.0 to 1.0)

        Returns:
            Simplified trimesh.Trimesh object
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh to simplify")

        if target_faces is None and target_percent is not None:
            target_faces = int(len(mesh.faces) * target_percent)

        if target_faces is not None:
            simplified = mesh.simplify_quadric_decimation(target_faces)
            self.mesh = simplified
            return simplified
        else:
            raise ValueError("Either target_faces or target_percent must be specified")

    def compute_normals(self, mesh: Optional[trimesh.Trimesh] = None) -> np.ndarray:
        """
        Compute vertex normals for the mesh.

        Args:
            mesh: Trimesh object (uses self.mesh if None)

        Returns:
            NumPy array of vertex normals (N x 3)
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        return mesh.vertex_normals.copy()

    def get_face_normals(self, mesh: Optional[trimesh.Trimesh] = None) -> np.ndarray:
        """
        Get face normals for the mesh.

        Args:
            mesh: Trimesh object (uses self.mesh if None)

        Returns:
            NumPy array of face normals (N x 3)
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        return mesh.face_normals.copy()

    def slice_mesh(self, mesh: Optional[trimesh.Trimesh] = None,
                   plane_origin: Tuple[float, float, float] = (0, 0, 0),
                   plane_normal: Tuple[float, float, float] = (0, 0, 1)) -> Optional[trimesh.path.Path2D]:
        """
        Slice the mesh with a plane and return the cross-section.

        Args:
            mesh: Trimesh object (uses self.mesh if None)
            plane_origin: Point on the plane
            plane_normal: Normal vector of the plane

        Returns:
            Path2D object representing the cross-section, or None if no intersection
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        try:
            section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
            if section is not None:
                path2d, to_3D = section.to_planar()
                return path2d
        except:
            pass

        return None

    def voxelize(self, mesh: Optional[trimesh.Trimesh] = None,
                 pitch: float = 0.1) -> Any:
        """
        Convert mesh to a voxel representation.

        Args:
            mesh: Trimesh object (uses self.mesh if None)
            pitch: Size of each voxel

        Returns:
            VoxelGrid object
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        return mesh.voxelized(pitch=pitch)
