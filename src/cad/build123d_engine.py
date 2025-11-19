"""
Build123D CAD Engine - Direct parametric CAD modeling using build123d.

This module provides a comprehensive interface for creating CAD models using
the build123d library, supporting primitives, operations, and boolean operations.
"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging

try:
    from build123d import (
        Part, Box, Cylinder, Sphere, Cone,
        extrude, revolve, loft, sweep,
        Axis, Plane, Location, Vector,
        Mode, export_step, export_stl
    )
    BUILD123D_AVAILABLE = True
except ImportError:
    BUILD123D_AVAILABLE = False
    logging.warning("build123d not installed. Install with: pip install build123d>=0.10.0")

logger = logging.getLogger(__name__)


class Build123DEngine:
    """
    Build123D-based CAD generation engine.

    Provides methods for creating primitive shapes, performing operations,
    and exporting to various CAD formats.
    """

    def __init__(self):
        """Initialize the Build123D engine."""
        if not BUILD123D_AVAILABLE:
            raise ImportError(
                "build123d is not installed. Install with: pip install build123d>=0.10.0"
            )
        logger.info("Build123D engine initialized")

    def generate_from_params(self, dimensions: Dict[str, Any]) -> Part:
        """
        Generate a CAD part from dimension parameters.

        Args:
            dimensions: Dictionary containing shape type and dimensions
                       e.g., {'type': 'box', 'length': 10, 'width': 5, 'height': 3}

        Returns:
            Part: The generated build123d Part object

        Raises:
            ValueError: If shape type is unknown or parameters are invalid
        """
        shape_type = dimensions.get('type', '').lower()

        if shape_type == 'box':
            return self._create_box(dimensions)
        elif shape_type == 'cylinder':
            return self._create_cylinder(dimensions)
        elif shape_type == 'sphere':
            return self._create_sphere(dimensions)
        elif shape_type == 'cone':
            return self._create_cone(dimensions)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

    def _create_box(self, params: Dict[str, Any]) -> Part:
        """Create a box primitive."""
        length = params.get('length', params.get('l', 10))
        width = params.get('width', params.get('w', 10))
        height = params.get('height', params.get('h', 10))

        logger.info(f"Creating box: {length}x{width}x{height}")
        box = Box(length, width, height)
        return Part() + box

    def _create_cylinder(self, params: Dict[str, Any]) -> Part:
        """Create a cylinder primitive."""
        radius = params.get('radius', params.get('r', 5))
        height = params.get('height', params.get('h', 10))

        logger.info(f"Creating cylinder: radius={radius}, height={height}")
        cylinder = Cylinder(radius, height)
        return Part() + cylinder

    def _create_sphere(self, params: Dict[str, Any]) -> Part:
        """Create a sphere primitive."""
        radius = params.get('radius', params.get('r', 5))

        logger.info(f"Creating sphere: radius={radius}")
        sphere = Sphere(radius)
        return Part() + sphere

    def _create_cone(self, params: Dict[str, Any]) -> Part:
        """Create a cone primitive."""
        bottom_radius = params.get('bottom_radius', params.get('r1', 5))
        top_radius = params.get('top_radius', params.get('r2', 0))
        height = params.get('height', params.get('h', 10))

        logger.info(f"Creating cone: bottom_r={bottom_radius}, top_r={top_radius}, h={height}")
        cone = Cone(bottom_radius, top_radius, height)
        return Part() + cone

    def union(self, *parts: Part) -> Part:
        """
        Perform boolean union on parts.

        Args:
            *parts: Variable number of Part objects to union

        Returns:
            Part: The unified part
        """
        if not parts:
            raise ValueError("At least one part required for union")

        result = parts[0]
        for part in parts[1:]:
            result = result + part

        logger.info(f"Union of {len(parts)} parts completed")
        return result

    def subtract(self, base: Part, *tools: Part) -> Part:
        """
        Perform boolean subtraction.

        Args:
            base: The base part to subtract from
            *tools: Parts to subtract from the base

        Returns:
            Part: The result of subtraction
        """
        if not tools:
            raise ValueError("At least one tool part required for subtraction")

        result = base
        for tool in tools:
            result = result - tool

        logger.info(f"Subtraction of {len(tools)} parts from base completed")
        return result

    def intersect(self, *parts: Part) -> Part:
        """
        Perform boolean intersection on parts.

        Args:
            *parts: Variable number of Part objects to intersect

        Returns:
            Part: The intersection of all parts
        """
        if len(parts) < 2:
            raise ValueError("At least two parts required for intersection")

        result = parts[0]
        for part in parts[1:]:
            result = result & part

        logger.info(f"Intersection of {len(parts)} parts completed")
        return result

    def export_step(self, part: Part, filepath: Union[str, Path]) -> None:
        """
        Export part to STEP format.

        Args:
            part: The Part object to export
            filepath: Output file path (will add .step extension if missing)
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() not in ['.step', '.stp']:
            filepath = filepath.with_suffix('.step')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        export_step(part, str(filepath))
        logger.info(f"Exported STEP file to {filepath}")

    def export_stl(
        self,
        part: Part,
        filepath: Union[str, Path],
        resolution: str = 'high'
    ) -> None:
        """
        Export part to STL format.

        Args:
            part: The Part object to export
            filepath: Output file path (will add .stl extension if missing)
            resolution: Quality of mesh ('low', 'medium', 'high')
        """
        filepath = Path(filepath)
        if filepath.suffix.lower() != '.stl':
            filepath = filepath.with_suffix('.stl')

        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Map resolution to tolerance and angular tolerance
        resolution_map = {
            'low': (0.5, 30),
            'medium': (0.1, 20),
            'high': (0.01, 10),
        }

        tolerance, angular_tolerance = resolution_map.get(
            resolution.lower(),
            resolution_map['high']
        )

        export_stl(
            part,
            str(filepath),
            tolerance=tolerance,
            angular_tolerance=angular_tolerance
        )
        logger.info(f"Exported STL file to {filepath} with {resolution} resolution")

    def create_composite(self, operations: List[Dict[str, Any]]) -> Part:
        """
        Create a composite part from a list of operations.

        Args:
            operations: List of operation dictionaries, each containing:
                       - type: 'primitive', 'union', 'subtract', 'intersect'
                       - params: parameters for the operation

        Returns:
            Part: The final composite part

        Example:
            operations = [
                {'type': 'primitive', 'params': {'type': 'box', 'length': 10, 'width': 10, 'height': 10}},
                {'type': 'subtract', 'params': {'type': 'cylinder', 'radius': 3, 'height': 12}}
            ]
        """
        if not operations:
            raise ValueError("At least one operation required")

        # First operation should be a primitive
        first_op = operations[0]
        if first_op.get('type') != 'primitive':
            raise ValueError("First operation must be a primitive")

        result = self.generate_from_params(first_op['params'])

        # Process subsequent operations
        for op in operations[1:]:
            op_type = op.get('type', '').lower()
            params = op.get('params', {})

            if op_type == 'primitive':
                # Add another primitive
                new_part = self.generate_from_params(params)
                result = self.union(result, new_part)
            elif op_type == 'union':
                new_part = self.generate_from_params(params)
                result = self.union(result, new_part)
            elif op_type == 'subtract':
                tool_part = self.generate_from_params(params)
                result = self.subtract(result, tool_part)
            elif op_type == 'intersect':
                intersect_part = self.generate_from_params(params)
                result = self.intersect(result, intersect_part)
            else:
                logger.warning(f"Unknown operation type: {op_type}, skipping")

        logger.info(f"Composite part created from {len(operations)} operations")
        return result
