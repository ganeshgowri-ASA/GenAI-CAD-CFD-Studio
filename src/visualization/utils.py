"""
Utility functions for mesh conversion, optimization, and caching.

Provides helper functions for converting between mesh formats, handling
large meshes with LOD and decimation, and performance caching.
"""

import pyvista as pv
import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
import hashlib
import pickle
from functools import lru_cache


class MeshConverter:
    """
    Utilities for converting between different mesh formats.

    Handles conversion between trimesh, PyVista, VTK, and other formats.
    """

    @staticmethod
    def trimesh_to_pyvista(trimesh_obj) -> pv.PolyData:
        """
        Convert trimesh object to PyVista PolyData.

        Args:
            trimesh_obj: Trimesh mesh object

        Returns:
            pv.PolyData: Converted PyVista mesh
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required. Install with: pip install trimesh")

        # Extract vertices and faces
        vertices = trimesh_obj.vertices
        faces = trimesh_obj.faces

        # Convert faces to PyVista format (prepend face size)
        faces_pv = np.hstack([[3, *face] for face in faces])

        # Create PyVista mesh
        mesh = pv.PolyData(vertices, faces_pv)

        # Transfer vertex normals if available
        if hasattr(trimesh_obj, 'vertex_normals'):
            mesh['vertex_normals'] = trimesh_obj.vertex_normals

        # Transfer face normals if available
        if hasattr(trimesh_obj, 'face_normals'):
            mesh.cell_data['face_normals'] = trimesh_obj.face_normals

        return mesh

    @staticmethod
    def pyvista_to_trimesh(pyvista_mesh: pv.PolyData):
        """
        Convert PyVista PolyData to trimesh object.

        Args:
            pyvista_mesh: PyVista mesh

        Returns:
            trimesh.Trimesh: Converted trimesh object
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required. Install with: pip install trimesh")

        # Extract vertices
        vertices = pyvista_mesh.points

        # Extract faces (remove the face size prefix)
        faces = pyvista_mesh.faces.reshape(-1, 4)[:, 1:]

        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh

    @staticmethod
    def numpy_to_pyvista(
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None
    ) -> pv.PolyData:
        """
        Convert numpy arrays to PyVista PolyData.

        Args:
            vertices: Vertex array (N x 3)
            faces: Face array (M x 3 for triangular mesh)
            normals: Optional normal vectors (N x 3)

        Returns:
            pv.PolyData: PyVista mesh
        """
        # Convert faces to PyVista format
        faces_pv = np.hstack([[3, *face] for face in faces])

        # Create mesh
        mesh = pv.PolyData(vertices, faces_pv)

        # Add normals if provided
        if normals is not None:
            mesh['normals'] = normals

        return mesh

    @staticmethod
    def pyvista_to_numpy(
        pyvista_mesh: pv.PolyData
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert PyVista PolyData to numpy arrays.

        Args:
            pyvista_mesh: PyVista mesh

        Returns:
            Tuple of (vertices, faces) numpy arrays
        """
        vertices = pyvista_mesh.points
        faces = pyvista_mesh.faces.reshape(-1, 4)[:, 1:]

        return vertices, faces


class MeshOptimizer:
    """
    Utilities for optimizing and processing meshes.

    Handles LOD generation, decimation, smoothing, and large mesh handling.
    """

    @staticmethod
    def decimate_mesh(
        mesh: pv.PolyData,
        target_reduction: float = 0.5,
        preserve_topology: bool = True
    ) -> pv.PolyData:
        """
        Reduce mesh complexity using decimation.

        Args:
            mesh: Input PyVista mesh
            target_reduction: Fraction of triangles to remove (0.0 to 1.0)
            preserve_topology: Preserve mesh topology during decimation

        Returns:
            pv.PolyData: Decimated mesh
        """
        decimated = mesh.decimate(
            target_reduction=target_reduction,
            preserve_topology=preserve_topology
        )

        print(f"Decimation: {mesh.n_cells} → {decimated.n_cells} cells "
              f"({target_reduction * 100:.1f}% reduction)")

        return decimated

    @staticmethod
    def smooth_mesh(
        mesh: pv.PolyData,
        n_iterations: int = 50,
        relaxation_factor: float = 0.1,
        feature_angle: float = 120.0
    ) -> pv.PolyData:
        """
        Smooth mesh using Laplacian smoothing.

        Args:
            mesh: Input PyVista mesh
            n_iterations: Number of smoothing iterations
            relaxation_factor: Relaxation factor (0.0 to 1.0)
            feature_angle: Feature angle for edge preservation

        Returns:
            pv.PolyData: Smoothed mesh
        """
        smoothed = mesh.smooth(
            n_iter=n_iterations,
            relaxation_factor=relaxation_factor,
            feature_angle=feature_angle
        )

        return smoothed

    @staticmethod
    def create_lod_hierarchy(
        mesh: pv.PolyData,
        n_levels: int = 4
    ) -> list:
        """
        Create Level-of-Detail (LOD) hierarchy for large meshes.

        Args:
            mesh: Input high-resolution mesh
            n_levels: Number of LOD levels to generate

        Returns:
            List of meshes from highest to lowest detail
        """
        lod_meshes = [mesh]  # Level 0: original mesh

        for level in range(1, n_levels):
            # Progressively reduce detail
            reduction = 1.0 - (1.0 / (2 ** level))
            reduced = MeshOptimizer.decimate_mesh(
                mesh,
                target_reduction=min(reduction, 0.9),
                preserve_topology=True
            )
            lod_meshes.append(reduced)

        print(f"Created {n_levels} LOD levels:")
        for i, lod_mesh in enumerate(lod_meshes):
            print(f"  Level {i}: {lod_mesh.n_cells:,} cells")

        return lod_meshes

    @staticmethod
    def compute_mesh_quality(
        mesh: pv.UnstructuredGrid,
        metric: str = 'aspect_ratio'
    ) -> np.ndarray:
        """
        Compute mesh quality metrics.

        Args:
            mesh: Input mesh
            metric: Quality metric ('aspect_ratio', 'skewness', 'jacobian', 'volume')

        Returns:
            Array of quality values per cell
        """
        quality_mesh = mesh.compute_cell_quality(quality_measure=metric)
        return quality_mesh['CellQuality']

    @staticmethod
    def repair_mesh(mesh: pv.PolyData) -> pv.PolyData:
        """
        Repair mesh by removing degenerate triangles and filling holes.

        Args:
            mesh: Input mesh with potential issues

        Returns:
            pv.PolyData: Repaired mesh
        """
        # Remove degenerate cells
        repaired = mesh.clean()

        # Fill holes
        try:
            repaired = repaired.fill_holes(hole_size=1000.0)
        except Exception:
            pass  # Not all meshes can have holes filled

        print(f"Mesh repair: {mesh.n_cells} → {repaired.n_cells} cells")

        return repaired

    @staticmethod
    def subdivide_mesh(
        mesh: pv.PolyData,
        n_subdivisions: int = 1,
        method: str = 'linear'
    ) -> pv.PolyData:
        """
        Subdivide mesh to increase resolution.

        Args:
            mesh: Input mesh
            n_subdivisions: Number of subdivision iterations
            method: Subdivision method ('linear', 'butterfly', 'loop')

        Returns:
            pv.PolyData: Subdivided mesh
        """
        subdivided = mesh.subdivide(
            nsub=n_subdivisions,
            subfilter=method
        )

        print(f"Subdivision: {mesh.n_cells} → {subdivided.n_cells} cells")

        return subdivided


class MeshCache:
    """
    Caching utilities for mesh operations to improve performance.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize mesh cache.

        Args:
            cache_dir: Directory for cache files (default: ./.mesh_cache)
        """
        self.cache_dir = cache_dir or Path(".mesh_cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, filepath: Path, operation: str, params: dict) -> str:
        """
        Generate cache key for mesh operation.

        Args:
            filepath: Path to mesh file
            operation: Operation name
            params: Operation parameters

        Returns:
            Cache key string
        """
        # Combine filepath, modification time, operation, and params
        mtime = filepath.stat().st_mtime if filepath.exists() else 0
        key_data = f"{filepath}_{mtime}_{operation}_{params}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_cached_mesh(
        self,
        filepath: Path,
        operation: str,
        params: dict
    ) -> Optional[pv.DataSet]:
        """
        Retrieve cached mesh if available.

        Args:
            filepath: Original mesh file path
            operation: Operation name
            params: Operation parameters

        Returns:
            Cached mesh or None if not found
        """
        cache_key = self._get_cache_key(filepath, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.vtk"

        if cache_file.exists():
            try:
                mesh = pv.read(str(cache_file))
                print(f"Loaded cached mesh: {cache_file.name}")
                return mesh
            except Exception as e:
                print(f"Cache read error: {e}")
                return None

        return None

    def save_cached_mesh(
        self,
        mesh: pv.DataSet,
        filepath: Path,
        operation: str,
        params: dict
    ) -> None:
        """
        Save mesh to cache.

        Args:
            mesh: Mesh to cache
            filepath: Original mesh file path
            operation: Operation name
            params: Operation parameters
        """
        cache_key = self._get_cache_key(filepath, operation, params)
        cache_file = self.cache_dir / f"{cache_key}.vtk"

        try:
            mesh.save(str(cache_file))
            print(f"Saved mesh to cache: {cache_file.name}")
        except Exception as e:
            print(f"Cache write error: {e}")

    def clear_cache(self) -> None:
        """Clear all cached meshes."""
        for cache_file in self.cache_dir.glob("*.vtk"):
            cache_file.unlink()
        print("Cache cleared")


class ColorMapUtils:
    """
    Utilities for working with colormaps and color schemes.
    """

    @staticmethod
    def get_diverging_colormap(
        data: np.ndarray,
        center_value: float = 0.0
    ) -> Tuple[str, Tuple[float, float]]:
        """
        Get diverging colormap with centered color limits.

        Args:
            data: Data array
            center_value: Value to center the colormap on

        Returns:
            Tuple of (colormap_name, (vmin, vmax))
        """
        abs_max = max(abs(data.min() - center_value), abs(data.max() - center_value))
        vmin = center_value - abs_max
        vmax = center_value + abs_max

        return 'coolwarm', (vmin, vmax)

    @staticmethod
    def create_custom_colormap(
        colors: list,
        n_colors: int = 256
    ) -> list:
        """
        Create custom colormap from list of colors.

        Args:
            colors: List of color strings or RGB tuples
            n_colors: Number of colors in resulting colormap

        Returns:
            List of interpolated colors
        """
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.pyplot as plt

        cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_colors)
        return [cmap(i) for i in np.linspace(0, 1, n_colors)]


# Convenience functions
def convert_mesh(source, target_format: str = 'pyvista'):
    """
    Convert mesh between formats.

    Args:
        source: Source mesh object
        target_format: Target format ('pyvista', 'trimesh', 'numpy')

    Returns:
        Converted mesh
    """
    converter = MeshConverter()

    # Detect source format
    source_type = type(source).__name__

    if target_format == 'pyvista':
        if 'trimesh' in source_type.lower():
            return converter.trimesh_to_pyvista(source)
        elif isinstance(source, tuple) and len(source) == 2:
            # Assume numpy arrays
            return converter.numpy_to_pyvista(source[0], source[1])
        else:
            return source  # Already PyVista

    elif target_format == 'trimesh':
        if isinstance(source, pv.PolyData):
            return converter.pyvista_to_trimesh(source)
        else:
            raise ValueError("Source must be PyVista PolyData")

    elif target_format == 'numpy':
        if isinstance(source, pv.PolyData):
            return converter.pyvista_to_numpy(source)
        else:
            raise ValueError("Source must be PyVista PolyData")

    else:
        raise ValueError(f"Unknown target format: {target_format}")


def optimize_for_display(
    mesh: pv.PolyData,
    max_cells: int = 1000000,
    smooth: bool = False
) -> pv.PolyData:
    """
    Optimize mesh for interactive display.

    Args:
        mesh: Input mesh
        max_cells: Maximum number of cells for display
        smooth: Apply smoothing after decimation

    Returns:
        Optimized mesh
    """
    optimizer = MeshOptimizer()

    # Decimate if too large
    if mesh.n_cells > max_cells:
        reduction = 1.0 - (max_cells / mesh.n_cells)
        mesh = optimizer.decimate_mesh(mesh, target_reduction=reduction)

    # Smooth if requested
    if smooth:
        mesh = optimizer.smooth_mesh(mesh)

    return mesh
