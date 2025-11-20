"""Geometry utility functions for GenAI CAD/CFD Studio.

This module provides various geometry calculation functions for CAD/CFD
applications including bounding boxes, centroids, areas, volumes, and
unit conversions.
"""

from typing import List, Tuple, Dict, Union
import math


# Type aliases
Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]
BoundingBox = Dict[str, Point3D]


# Unit conversion factors (to meters)
UNIT_TO_METERS = {
    # Length
    'mm': 0.001,
    'cm': 0.01,
    'm': 1.0,
    'km': 1000.0,
    'in': 0.0254,
    'ft': 0.3048,
    'yd': 0.9144,
    'mi': 1609.344,
    # Area (to square meters)
    'mm2': 1e-6,
    'cm2': 1e-4,
    'm2': 1.0,
    'km2': 1e6,
    'in2': 0.00064516,
    'ft2': 0.09290304,
    'yd2': 0.83612736,
    'mi2': 2589988.110336,
    # Volume (to cubic meters)
    'mm3': 1e-9,
    'cm3': 1e-6,
    'm3': 1.0,
    'km3': 1e9,
    'in3': 1.6387064e-5,
    'ft3': 0.028316846592,
    'yd3': 0.764554857984,
    'mi3': 4168181825.440579584,
}


def calculate_bounding_box(points: List[Point3D]) -> BoundingBox:
    """Calculate the axis-aligned bounding box for a set of 3D points.

    Args:
        points: List of 3D points as (x, y, z) tuples.

    Returns:
        Dictionary with 'min' and 'max' keys containing the minimum and
        maximum corners of the bounding box, and 'center' and 'dimensions'.

    Raises:
        ValueError: If points list is empty.

    Example:
        >>> points = [(0, 0, 0), (1, 2, 3), (4, 1, 2)]
        >>> bbox = calculate_bounding_box(points)
        >>> print(bbox['min'])
        (0, 0, 0)
        >>> print(bbox['max'])
        (4, 2, 3)
        >>> print(bbox['dimensions'])
        (4, 2, 3)
    """
    if not points:
        raise ValueError("Points list cannot be empty")

    # Initialize with first point
    min_x = max_x = points[0][0]
    min_y = max_y = points[0][1]
    min_z = max_z = points[0][2]

    # Find min and max for each dimension
    for x, y, z in points:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    min_point = (min_x, min_y, min_z)
    max_point = (max_x, max_y, max_z)

    # Calculate center and dimensions
    center = (
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
        (min_z + max_z) / 2
    )

    dimensions = (
        max_x - min_x,
        max_y - min_y,
        max_z - min_z
    )

    return {
        'min': min_point,
        'max': max_point,
        'center': center,
        'dimensions': dimensions
    }


def calculate_centroid(polygon: List[Point2D]) -> Point2D:
    """Calculate the centroid of a 2D polygon.

    Uses the formula for the centroid of a polygon defined by its vertices.

    Args:
        polygon: List of 2D points as (x, y) tuples defining the polygon
            vertices in order (clockwise or counter-clockwise).

    Returns:
        Centroid point as (x, y) tuple.

    Raises:
        ValueError: If polygon has fewer than 3 points.

    Example:
        >>> polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]  # Rectangle
        >>> centroid = calculate_centroid(polygon)
        >>> print(centroid)
        (2.0, 1.5)
    """
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 points")

    # Calculate signed area and centroid using the shoelace formula
    area = 0.0
    cx = 0.0
    cy = 0.0

    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]

        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area *= 0.5

    # Avoid division by zero
    if abs(area) < 1e-10:
        # Degenerate polygon, return average of points
        n = len(polygon)
        return (
            sum(p[0] for p in polygon) / n,
            sum(p[1] for p in polygon) / n
        )

    cx /= (6.0 * area)
    cy /= (6.0 * area)

    return (cx, cy)


def calculate_area(polygon: List[Point2D]) -> float:
    """Calculate the area of a 2D polygon.

    Uses the shoelace formula (also known as Gauss's area formula).

    Args:
        polygon: List of 2D points as (x, y) tuples defining the polygon
            vertices in order.

    Returns:
        Area of the polygon (always positive).

    Raises:
        ValueError: If polygon has fewer than 3 points.

    Example:
        >>> polygon = [(0, 0), (4, 0), (4, 3), (0, 3)]  # Rectangle 4x3
        >>> area = calculate_area(polygon)
        >>> print(area)
        12.0
    """
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 points")

    # Shoelace formula
    area = 0.0
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        area += x0 * y1 - x1 * y0

    return abs(area) / 2.0


def calculate_volume(mesh: Dict[str, List]) -> float:
    """Calculate the volume of a 3D mesh using the divergence theorem.

    The mesh should be a closed surface represented by triangular faces.

    Args:
        mesh: Dictionary with 'vertices' (list of Point3D) and 'faces'
            (list of tuples of 3 vertex indices).

    Returns:
        Volume enclosed by the mesh (always positive).

    Raises:
        ValueError: If mesh data is invalid.

    Example:
        >>> # Unit cube
        >>> vertices = [
        ...     (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        ...     (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
        ... ]
        >>> faces = [
        ...     (0, 1, 2), (0, 2, 3),  # Bottom
        ...     (4, 6, 5), (4, 7, 6),  # Top
        ...     (0, 4, 5), (0, 5, 1),  # Front
        ...     (2, 6, 7), (2, 7, 3),  # Back
        ...     (0, 3, 7), (0, 7, 4),  # Left
        ...     (1, 5, 6), (1, 6, 2),  # Right
        ... ]
        >>> mesh = {'vertices': vertices, 'faces': faces}
        >>> volume = calculate_volume(mesh)
        >>> print(f"{volume:.1f}")
        1.0
    """
    if 'vertices' not in mesh or 'faces' not in mesh:
        raise ValueError("Mesh must contain 'vertices' and 'faces' keys")

    vertices = mesh['vertices']
    faces = mesh['faces']

    if not vertices or not faces:
        raise ValueError("Mesh must have at least one vertex and one face")

    # Calculate signed volume using divergence theorem
    volume = 0.0

    for face in faces:
        if len(face) != 3:
            raise ValueError("All faces must be triangles (3 vertices)")

        # Get vertices of the triangle
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]

        # Calculate contribution of this triangle to the volume
        # Using the formula: V = (1/6) * sum(dot(v, cross(v1-v0, v2-v0)))
        # where v is any vertex of the triangle
        cross_product = (
            (v1[1] - v0[1]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1]),
            (v1[2] - v0[2]) * (v2[0] - v0[0]) - (v1[0] - v0[0]) * (v2[2] - v0[2]),
            (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
        )

        dot_product = v0[0] * cross_product[0] + v0[1] * cross_product[1] + v0[2] * cross_product[2]
        volume += dot_product

    return abs(volume) / 6.0


def convert_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> float:
    """Convert a value from one unit to another.

    Supports length, area, and volume units.

    Args:
        value: Numerical value to convert.
        from_unit: Source unit (e.g., 'mm', 'cm', 'm', 'km', 'in', 'ft').
        to_unit: Target unit.

    Returns:
        Converted value.

    Raises:
        ValueError: If units are not supported or incompatible.

    Example:
        >>> convert_units(1000, 'mm', 'm')
        1.0
        >>> convert_units(1, 'ft', 'cm')
        30.48
        >>> convert_units(1, 'm2', 'cm2')
        10000.0
    """
    # Normalize unit names
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Check if units are supported
    if from_unit not in UNIT_TO_METERS:
        raise ValueError(f"Unsupported source unit: '{from_unit}'")
    if to_unit not in UNIT_TO_METERS:
        raise ValueError(f"Unsupported target unit: '{to_unit}'")

    # Check if units are compatible (same dimension)
    def get_dimension(unit: str) -> str:
        if unit.endswith('3'):
            return 'volume'
        elif unit.endswith('2'):
            return 'area'
        else:
            return 'length'

    from_dim = get_dimension(from_unit)
    to_dim = get_dimension(to_unit)

    if from_dim != to_dim:
        raise ValueError(
            f"Cannot convert between incompatible dimensions: "
            f"'{from_unit}' ({from_dim}) to '{to_unit}' ({to_dim})"
        )

    # Convert to base unit (meters) then to target unit
    base_value = value * UNIT_TO_METERS[from_unit]
    result = base_value / UNIT_TO_METERS[to_unit]

    return result


def calculate_distance(p1: Point3D, p2: Point3D) -> float:
    """Calculate Euclidean distance between two 3D points.

    Args:
        p1: First point as (x, y, z) tuple.
        p2: Second point as (x, y, z) tuple.

    Returns:
        Distance between the two points.

    Example:
        >>> p1 = (0, 0, 0)
        >>> p2 = (3, 4, 0)
        >>> distance = calculate_distance(p1, p2)
        >>> print(distance)
        5.0
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    return math.sqrt(dx * dx + dy * dy + dz * dz)


def calculate_normal(p1: Point3D, p2: Point3D, p3: Point3D) -> Point3D:
    """Calculate the normal vector of a triangle defined by three points.

    The normal is calculated using the cross product of two edge vectors.

    Args:
        p1: First point of the triangle.
        p2: Second point of the triangle.
        p3: Third point of the triangle.

    Returns:
        Normalized normal vector as (x, y, z) tuple.

    Example:
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 0, 0)
        >>> p3 = (0, 1, 0)
        >>> normal = calculate_normal(p1, p2, p3)
        >>> print(f"({normal[0]:.1f}, {normal[1]:.1f}, {normal[2]:.1f})")
        (0.0, 0.0, 1.0)
    """
    # Calculate edge vectors
    v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])

    # Calculate cross product
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]

    # Normalize
    length = math.sqrt(nx * nx + ny * ny + nz * nz)

    if length < 1e-10:
        # Degenerate triangle
        return (0.0, 0.0, 0.0)

    return (nx / length, ny / length, nz / length)


def calculate_sphere_volume(radius: float) -> float:
    """Calculate the volume of a sphere.

    Args:
        radius: Radius of the sphere.

    Returns:
        Volume of the sphere.

    Example:
        >>> volume = calculate_sphere_volume(1.0)
        >>> print(f"{volume:.2f}")
        4.19
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")

    return (4.0 / 3.0) * math.pi * radius ** 3


def calculate_cylinder_volume(radius: float, height: float) -> float:
    """Calculate the volume of a cylinder.

    Args:
        radius: Radius of the cylinder base.
        height: Height of the cylinder.

    Returns:
        Volume of the cylinder.

    Example:
        >>> volume = calculate_cylinder_volume(2.0, 5.0)
        >>> print(f"{volume:.2f}")
        62.83
    """
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative")

    return math.pi * radius ** 2 * height


def calculate_cone_volume(radius: float, height: float) -> float:
    """Calculate the volume of a cone.

    Args:
        radius: Radius of the cone base.
        height: Height of the cone.

    Returns:
        Volume of the cone.

    Example:
        >>> volume = calculate_cone_volume(3.0, 4.0)
        >>> print(f"{volume:.2f}")
        37.70
    """
    if radius < 0 or height < 0:
        raise ValueError("Radius and height must be non-negative")

    return (1.0 / 3.0) * math.pi * radius ** 2 * height
