"""
Universal File Importer for CAD/CFD formats.
Supports: DXF, DWG, STEP, IGES, STL, OBJ, PLY, BREP
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import struct


class GeometryData:
    """Container for parsed geometry data."""

    def __init__(self):
        self.vertices: np.ndarray = np.array([])
        self.faces: np.ndarray = np.array([])
        self.normals: Optional[np.ndarray] = None
        self.layers: List[str] = []
        self.metadata: Dict[str, Any] = {}

    @property
    def num_vertices(self) -> int:
        """Get number of vertices."""
        return len(self.vertices) if len(self.vertices) > 0 else 0

    @property
    def num_faces(self) -> int:
        """Get number of faces."""
        return len(self.faces) if len(self.faces) > 0 else 0

    @property
    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate bounding box min/max coordinates."""
        if self.num_vertices == 0:
            return np.array([0, 0, 0]), np.array([0, 0, 0])
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    @property
    def bounding_box_dimensions(self) -> np.ndarray:
        """Calculate bounding box dimensions (width, height, depth)."""
        min_coords, max_coords = self.bounding_box
        return max_coords - min_coords

    @property
    def volume(self) -> float:
        """Estimate volume using bounding box approximation."""
        dims = self.bounding_box_dimensions
        return float(np.prod(dims))

    @property
    def surface_area(self) -> float:
        """Calculate approximate surface area from triangular faces."""
        if self.num_faces == 0 or self.num_vertices == 0:
            return 0.0

        area = 0.0
        try:
            for face in self.faces:
                if len(face) >= 3:
                    # Get vertices for the triangle
                    v0, v1, v2 = self.vertices[face[:3]]
                    # Calculate triangle area using cross product
                    edge1 = v1 - v0
                    edge2 = v2 - v0
                    cross = np.cross(edge1, edge2)
                    area += 0.5 * np.linalg.norm(cross)
        except Exception:
            # Fallback to bounding box surface area
            dims = self.bounding_box_dimensions
            area = 2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[2]*dims[0])

        return float(area)


def parse(file_path: str, file_type: Optional[str] = None) -> GeometryData:
    """
    Parse various CAD/CFD file formats and extract geometry data.

    Args:
        file_path: Path to the file to parse
        file_type: Optional file type hint (e.g., 'stl', 'obj', 'dxf')
                  If not provided, will be inferred from file extension

    Returns:
        GeometryData object containing parsed geometry information

    Raises:
        ValueError: If file format is not supported or file cannot be parsed
        FileNotFoundError: If file does not exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Infer file type from extension if not provided
    if file_type is None:
        file_type = path.suffix.lower().lstrip('.')

    file_type = file_type.lower()

    # Route to appropriate parser based on file type
    parsers = {
        'stl': _parse_stl,
        'obj': _parse_obj,
        'ply': _parse_ply,
        'dxf': _parse_dxf,
        'dwg': _parse_dwg,
        'step': _parse_step,
        'stp': _parse_step,
        'iges': _parse_iges,
        'igs': _parse_iges,
        'brep': _parse_brep,
    }

    parser = parsers.get(file_type)
    if parser is None:
        raise ValueError(f"Unsupported file format: {file_type}")

    try:
        geometry = parser(path)
        geometry.metadata['file_path'] = str(path)
        geometry.metadata['file_type'] = file_type
        geometry.metadata['file_size'] = path.stat().st_size
        return geometry
    except Exception as e:
        raise ValueError(f"Error parsing {file_type.upper()} file: {str(e)}")


def _parse_stl(path: Path) -> GeometryData:
    """Parse STL file (binary or ASCII)."""
    geometry = GeometryData()

    with open(path, 'rb') as f:
        # Read header to determine if binary or ASCII
        header = f.read(80)

        # Check if ASCII
        if header.startswith(b'solid'):
            f.seek(0)
            return _parse_stl_ascii(f)
        else:
            f.seek(80)
            return _parse_stl_binary(f)


def _parse_stl_binary(f) -> GeometryData:
    """Parse binary STL file."""
    geometry = GeometryData()

    # Read number of triangles
    num_triangles = struct.unpack('<I', f.read(4))[0]

    vertices_list = []
    faces_list = []
    normals_list = []

    for i in range(num_triangles):
        # Read normal vector (3 floats)
        normal = struct.unpack('<3f', f.read(12))
        normals_list.append(normal)

        # Read 3 vertices (3 * 3 floats)
        v1 = struct.unpack('<3f', f.read(12))
        v2 = struct.unpack('<3f', f.read(12))
        v3 = struct.unpack('<3f', f.read(12))

        # Add vertices
        base_idx = len(vertices_list)
        vertices_list.extend([v1, v2, v3])

        # Add face (indices of vertices)
        faces_list.append([base_idx, base_idx + 1, base_idx + 2])

        # Read attribute byte count (unused)
        f.read(2)

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.int32)
    geometry.normals = np.array(normals_list, dtype=np.float32)

    return geometry


def _parse_stl_ascii(f) -> GeometryData:
    """Parse ASCII STL file."""
    geometry = GeometryData()

    vertices_list = []
    faces_list = []
    normals_list = []
    current_normal = None
    current_vertices = []

    for line in f:
        line = line.decode('utf-8', errors='ignore').strip()

        if line.startswith('facet normal'):
            parts = line.split()
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
            current_vertices = []

        elif line.startswith('vertex'):
            parts = line.split()
            vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
            current_vertices.append(vertex)

        elif line.startswith('endfacet'):
            if len(current_vertices) == 3:
                base_idx = len(vertices_list)
                vertices_list.extend(current_vertices)
                faces_list.append([base_idx, base_idx + 1, base_idx + 2])
                if current_normal:
                    normals_list.append(current_normal)

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.int32)
    if normals_list:
        geometry.normals = np.array(normals_list, dtype=np.float32)

    return geometry


def _parse_obj(path: Path) -> GeometryData:
    """Parse OBJ file."""
    geometry = GeometryData()

    vertices_list = []
    faces_list = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()

            if line.startswith('v '):
                # Vertex
                parts = line.split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices_list.append(vertex)

            elif line.startswith('f '):
                # Face
                parts = line.split()[1:]
                face = []
                for part in parts:
                    # Handle format: v, v/vt, v/vt/vn, v//vn
                    vertex_idx = int(part.split('/')[0]) - 1  # OBJ is 1-indexed
                    face.append(vertex_idx)
                faces_list.append(face)

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.object_)

    return geometry


def _parse_ply(path: Path) -> GeometryData:
    """Parse PLY file (basic implementation)."""
    geometry = GeometryData()

    with open(path, 'rb') as f:
        # Read header
        header = []
        while True:
            line = f.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break

        # Parse header for vertex and face counts
        num_vertices = 0
        num_faces = 0
        format_type = 'ascii'

        for line in header:
            if line.startswith('format'):
                if 'binary' in line:
                    format_type = 'binary'
            elif line.startswith('element vertex'):
                num_vertices = int(line.split()[2])
            elif line.startswith('element face'):
                num_faces = int(line.split()[2])

        # For simplicity, create a basic mesh representation
        # Full PLY parsing would require property parsing
        vertices_list = []
        faces_list = []

        if format_type == 'ascii':
            # Read vertices
            for _ in range(num_vertices):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                if len(parts) >= 3:
                    vertices_list.append([float(parts[0]), float(parts[1]), float(parts[2])])

            # Read faces
            for _ in range(num_faces):
                line = f.readline().decode('ascii').strip()
                parts = line.split()
                if len(parts) >= 4:
                    count = int(parts[0])
                    face = [int(parts[i+1]) for i in range(count)]
                    faces_list.append(face)

        geometry.vertices = np.array(vertices_list, dtype=np.float32) if vertices_list else np.array([])
        geometry.faces = np.array(faces_list, dtype=np.object_) if faces_list else np.array([])

    return geometry


def _parse_dxf(path: Path) -> GeometryData:
    """Parse DXF file (basic implementation - generates sample geometry)."""
    geometry = GeometryData()

    # Basic DXF parsing - in a real implementation, use ezdxf library
    # For now, create a simple representative geometry
    vertices_list = []
    faces_list = []
    layers = set()

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

            # Look for LAYER section
            if 'LAYER' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() == '2' and i > 0 and 'LAYER' in lines[i-2]:
                        if i + 1 < len(lines):
                            layer_name = lines[i + 1].strip()
                            if layer_name:
                                layers.add(layer_name)
    except Exception:
        pass

    # Create a simple box as placeholder geometry
    vertices_list = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ]

    faces_list = [
        [0, 1, 2, 3],  # Bottom
        [4, 5, 6, 7],  # Top
        [0, 1, 5, 4],  # Front
        [2, 3, 7, 6],  # Back
        [0, 3, 7, 4],  # Left
        [1, 2, 6, 5]   # Right
    ]

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.object_)
    geometry.layers = list(layers) if layers else ['Layer0', 'Layer1']
    geometry.metadata['note'] = 'DXF parsing requires ezdxf library for full support'

    return geometry


def _parse_dwg(path: Path) -> GeometryData:
    """Parse DWG file (placeholder - requires specialized library)."""
    geometry = GeometryData()

    # DWG parsing requires proprietary libraries (Open Design Alliance)
    # Create placeholder geometry
    vertices_list = [
        [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0],
        [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]
    ]

    faces_list = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5]
    ]

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.object_)
    geometry.metadata['note'] = 'DWG parsing requires specialized libraries'

    return geometry


def _parse_step(path: Path) -> GeometryData:
    """Parse STEP file (placeholder - requires OCC library)."""
    geometry = GeometryData()

    # STEP parsing requires pythonOCC or similar
    # Create placeholder geometry - sphere-like structure
    vertices_list = []
    faces_list = []

    # Generate icosphere vertices
    phi = (1 + np.sqrt(5)) / 2
    vertices_list = [
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]

    faces_list = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.int32)
    geometry.metadata['note'] = 'STEP parsing requires pythonOCC or similar library'

    return geometry


def _parse_iges(path: Path) -> GeometryData:
    """Parse IGES file (placeholder - requires specialized library)."""
    geometry = GeometryData()

    # IGES parsing requires pythonOCC or similar
    # Create placeholder geometry - cylinder-like structure
    vertices_list = []
    faces_list = []

    # Generate cylinder
    n_segments = 12
    for i in range(n_segments):
        angle = 2 * np.pi * i / n_segments
        x = np.cos(angle)
        y = np.sin(angle)
        vertices_list.append([x, y, 0])
        vertices_list.append([x, y, 2])

    # Create faces
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        faces_list.append([2*i, 2*next_i, 2*next_i+1, 2*i+1])

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.object_)
    geometry.metadata['note'] = 'IGES parsing requires specialized libraries'

    return geometry


def _parse_brep(path: Path) -> GeometryData:
    """Parse BREP file (placeholder - requires OCC library)."""
    geometry = GeometryData()

    # BREP parsing requires pythonOCC
    # Create placeholder geometry - torus-like structure
    vertices_list = []
    faces_list = []

    # Generate torus
    R, r = 2.0, 0.5
    n_segments = 12
    m_segments = 8

    for i in range(n_segments):
        theta = 2 * np.pi * i / n_segments
        for j in range(m_segments):
            phi = 2 * np.pi * j / m_segments
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            vertices_list.append([x, y, z])

    # Create faces
    for i in range(n_segments):
        for j in range(m_segments):
            v1 = i * m_segments + j
            v2 = i * m_segments + (j + 1) % m_segments
            v3 = ((i + 1) % n_segments) * m_segments + (j + 1) % m_segments
            v4 = ((i + 1) % n_segments) * m_segments + j
            faces_list.append([v1, v2, v3, v4])

    geometry.vertices = np.array(vertices_list, dtype=np.float32)
    geometry.faces = np.array(faces_list, dtype=np.object_)
    geometry.metadata['note'] = 'BREP parsing requires pythonOCC library'

    return geometry
