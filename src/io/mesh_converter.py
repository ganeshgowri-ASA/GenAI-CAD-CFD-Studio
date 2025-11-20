"""
Mesh Converter Module

Provides universal mesh format conversion using meshio library.
Supports 20+ mesh formats including VTK, MSH, ANSYS, Abaqus, and more.
"""

import meshio
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path


class MeshConverter:
    """
    Universal mesh format converter using meshio.

    Supports formats including:
    - VTK (.vtk, .vtu, .vts, .vtr, .vtp)
    - Gmsh (.msh)
    - ANSYS (.ans)
    - Abaqus (.inp)
    - CGNS (.cgns)
    - DOLFIN XML (.xml)
    - Exodus (.e, .exo)
    - FLAC3D (.f3grid)
    - H5M (.h5m)
    - Kratos (.mdpa)
    - Med (.med)
    - Nastran (.bdf, .nas)
    - Neuroglancer (.neuroglancer)
    - OBJ (.obj)
    - OFF (.off)
    - PERMAS (.post, .post.gz, .dato, .dato.gz)
    - PLY (.ply)
    - STL (.stl)
    - SVG (.svg)
    - Tecplot (.dat)
    - TetGen (.node, .ele)
    - UGRID (.ugrid)
    - WKT (.wkt)
    - XDMF (.xdmf, .xmf)
    """

    # Supported file extensions
    SUPPORTED_FORMATS = {
        # VTK family
        'vtk': 'VTK',
        'vtu': 'VTK Unstructured Grid',
        'vts': 'VTK Structured Grid',
        'vtr': 'VTK Rectilinear Grid',
        'vtp': 'VTK Poly Data',
        'pvtu': 'Parallel VTK Unstructured Grid',
        # Mesh formats
        'msh': 'Gmsh',
        'ans': 'ANSYS',
        'inp': 'Abaqus',
        'cgns': 'CGNS',
        'xml': 'DOLFIN XML',
        'e': 'Exodus',
        'exo': 'Exodus',
        'f3grid': 'FLAC3D',
        'h5m': 'H5M',
        'mdpa': 'Kratos',
        'med': 'MED',
        'bdf': 'Nastran',
        'nas': 'Nastran',
        'obj': 'Wavefront OBJ',
        'off': 'OFF',
        'post': 'PERMAS',
        'dato': 'PERMAS',
        'ply': 'Stanford PLY',
        'stl': 'STL',
        'svg': 'SVG',
        'dat': 'Tecplot',
        'node': 'TetGen',
        'ele': 'TetGen',
        'ugrid': 'UGRID',
        'wkt': 'WKT',
        'xdmf': 'XDMF',
        'xmf': 'XDMF'
    }

    def __init__(self):
        """Initialize mesh converter."""
        self.mesh = None

    def read(self, filepath: str) -> meshio.Mesh:
        """
        Read a mesh file in any supported format.

        Args:
            filepath: Path to the mesh file

        Returns:
            meshio.Mesh object

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the format is not supported
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")

        ext = filepath.suffix.lstrip('.').lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        self.mesh = meshio.read(str(filepath))
        return self.mesh

    def write(self, filepath: str, mesh: Optional[meshio.Mesh] = None,
              file_format: Optional[str] = None, **kwargs) -> bool:
        """
        Write mesh to a file in the specified format.

        Args:
            filepath: Output file path
            mesh: meshio.Mesh object to write (uses self.mesh if None)
            file_format: Output format (auto-detected from extension if None)
            **kwargs: Additional arguments passed to meshio.write

        Returns:
            True if write successful

        Raises:
            ValueError: If no mesh is available or format is unsupported
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh to write")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if file_format is None:
            ext = filepath.suffix.lstrip('.').lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file format: {ext}. "
                    f"Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
                )

        try:
            meshio.write(str(filepath), mesh, file_format=file_format, **kwargs)
            return True
        except Exception as e:
            raise RuntimeError(f"Error writing mesh: {e}")

    def convert(self, input_file: str, output_file: str,
                preserve_metadata: bool = True, **kwargs) -> bool:
        """
        Convert a mesh from one format to another.

        Args:
            input_file: Input mesh file path
            output_file: Output mesh file path
            preserve_metadata: Whether to preserve cell and point data
            **kwargs: Additional arguments passed to meshio.write

        Returns:
            True if conversion successful

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If format is not supported
        """
        # Read input mesh
        mesh = self.read(input_file)

        # Remove metadata if requested
        if not preserve_metadata:
            mesh.point_data = {}
            mesh.cell_data = {}
            mesh.field_data = {}

        # Write to output format
        return self.write(output_file, mesh, **kwargs)

    def get_mesh_info(self, mesh: Optional[meshio.Mesh] = None) -> Dict[str, Any]:
        """
        Get information about the mesh structure.

        Args:
            mesh: meshio.Mesh object (uses self.mesh if None)

        Returns:
            Dictionary containing mesh information
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        info = {
            'num_points': len(mesh.points),
            'num_cells': sum(len(cells.data) for cells in mesh.cells),
            'cell_types': [cell.type for cell in mesh.cells],
            'point_data_keys': list(mesh.point_data.keys()),
            'cell_data_keys': list(mesh.cell_data.keys()),
            'field_data': dict(mesh.field_data) if mesh.field_data else {},
            'bounds': self._calculate_bounds(mesh),
            'dimension': mesh.points.shape[1] if len(mesh.points) > 0 else 0
        }

        # Count cells by type
        cell_counts = {}
        for cell in mesh.cells:
            cell_counts[cell.type] = len(cell.data)
        info['cell_counts'] = cell_counts

        return info

    def _calculate_bounds(self, mesh: meshio.Mesh) -> Dict[str, float]:
        """
        Calculate bounding box of the mesh.

        Args:
            mesh: meshio.Mesh object

        Returns:
            Dictionary with xmin, xmax, ymin, ymax, zmin, zmax
        """
        if len(mesh.points) == 0:
            return {
                'xmin': 0.0, 'xmax': 0.0,
                'ymin': 0.0, 'ymax': 0.0,
                'zmin': 0.0, 'zmax': 0.0
            }

        points = mesh.points
        bounds = {
            'xmin': float(np.min(points[:, 0])),
            'xmax': float(np.max(points[:, 0])),
            'ymin': float(np.min(points[:, 1])),
            'ymax': float(np.max(points[:, 1]))
        }

        # Add z bounds if 3D
        if points.shape[1] >= 3:
            bounds['zmin'] = float(np.min(points[:, 2]))
            bounds['zmax'] = float(np.max(points[:, 2]))
        else:
            bounds['zmin'] = 0.0
            bounds['zmax'] = 0.0

        return bounds

    def extract_surface(self, mesh: Optional[meshio.Mesh] = None) -> meshio.Mesh:
        """
        Extract the surface mesh from a volume mesh.

        Args:
            mesh: meshio.Mesh object (uses self.mesh if None)

        Returns:
            meshio.Mesh object containing only surface elements
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        # Filter for surface cell types (triangles and quads)
        surface_cells = []
        surface_cell_data = {}

        for i, cell in enumerate(mesh.cells):
            if cell.type in ['triangle', 'quad', 'polygon']:
                surface_cells.append(cell)

                # Preserve cell data for surface cells
                for key in mesh.cell_data.keys():
                    if key not in surface_cell_data:
                        surface_cell_data[key] = []
                    surface_cell_data[key].append(mesh.cell_data[key][i])

        surface_mesh = meshio.Mesh(
            points=mesh.points,
            cells=surface_cells,
            point_data=mesh.point_data,
            cell_data=surface_cell_data,
            field_data=mesh.field_data
        )

        return surface_mesh

    def merge_meshes(self, meshes: List[meshio.Mesh]) -> meshio.Mesh:
        """
        Merge multiple meshes into a single mesh.

        Args:
            meshes: List of meshio.Mesh objects to merge

        Returns:
            Combined meshio.Mesh object
        """
        if not meshes:
            raise ValueError("No meshes to merge")

        if len(meshes) == 1:
            return meshes[0]

        # Combine points
        all_points = []
        all_cells = []
        point_offset = 0

        for mesh in meshes:
            all_points.append(mesh.points)

            # Adjust cell indices by point offset
            for cell in mesh.cells:
                adjusted_data = cell.data + point_offset
                all_cells.append(meshio.CellBlock(cell.type, adjusted_data))

            point_offset += len(mesh.points)

        combined_points = np.vstack(all_points)

        merged_mesh = meshio.Mesh(
            points=combined_points,
            cells=all_cells
        )

        self.mesh = merged_mesh
        return merged_mesh

    def scale_mesh(self, scale_factor: Union[float, List[float]],
                   mesh: Optional[meshio.Mesh] = None) -> meshio.Mesh:
        """
        Scale the mesh by a factor.

        Args:
            scale_factor: Uniform scale factor or list of [sx, sy, sz]
            mesh: meshio.Mesh object (uses self.mesh if None)

        Returns:
            Scaled meshio.Mesh object
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        scaled_points = mesh.points.copy()

        if isinstance(scale_factor, (int, float)):
            scaled_points *= scale_factor
        elif isinstance(scale_factor, (list, tuple, np.ndarray)):
            scaled_points *= np.array(scale_factor)
        else:
            raise ValueError("scale_factor must be a number or list/array")

        scaled_mesh = meshio.Mesh(
            points=scaled_points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data
        )

        self.mesh = scaled_mesh
        return scaled_mesh

    def translate_mesh(self, translation: List[float],
                       mesh: Optional[meshio.Mesh] = None) -> meshio.Mesh:
        """
        Translate the mesh by a vector.

        Args:
            translation: Translation vector [tx, ty, tz]
            mesh: meshio.Mesh object (uses self.mesh if None)

        Returns:
            Translated meshio.Mesh object
        """
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            raise ValueError("No mesh available")

        translated_points = mesh.points.copy()
        translated_points += np.array(translation)

        translated_mesh = meshio.Mesh(
            points=translated_points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data
        )

        self.mesh = translated_mesh
        return translated_mesh

    @staticmethod
    def get_supported_formats() -> Dict[str, str]:
        """
        Get dictionary of supported file formats.

        Returns:
            Dictionary mapping file extension to format name
        """
        return MeshConverter.SUPPORTED_FORMATS.copy()

    def validate_format(self, filepath: str) -> bool:
        """
        Check if a file format is supported.

        Args:
            filepath: File path to check

        Returns:
            True if format is supported, False otherwise
        """
        ext = Path(filepath).suffix.lstrip('.').lower()
        return ext in self.SUPPORTED_FORMATS
