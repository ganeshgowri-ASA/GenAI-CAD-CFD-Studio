"""
DXF Parser Module

Handles parsing of DXF (Drawing Exchange Format) files using ezdxf library.
Supports DXF versions R12 through R2018.
"""

import ezdxf
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from pathlib import Path


class DXFParser:
    """
    Parser for DXF files with geometry extraction capabilities.

    Supports extraction of:
    - Lines
    - Arcs
    - Circles
    - Polylines
    - Dimensions
    - Layers
    - Blocks
    """

    def __init__(self):
        """Initialize DXF parser."""
        self.doc = None
        self.modelspace = None

    def parse(self, filepath: str) -> Dict[str, Any]:
        """
        Parse a DXF file and extract geometry information.

        Args:
            filepath: Path to the DXF file

        Returns:
            Dictionary containing:
                - 'lines': List of line entities
                - 'arcs': List of arc entities
                - 'circles': List of circle entities
                - 'polylines': List of polyline entities
                - 'dimensions': List of dimension entities
                - 'layers': List of layer information
                - 'blocks': List of block definitions
                - 'bounds': Bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
                - 'dxf_version': DXF file version
                - 'units': Drawing units

        Raises:
            FileNotFoundError: If the file doesn't exist
            ezdxf.DXFError: If the file is not a valid DXF file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"DXF file not found: {filepath}")

        # Load DXF document
        self.doc = ezdxf.readfile(str(filepath))
        self.modelspace = self.doc.modelspace()

        # Extract geometry
        result = {
            'lines': self._extract_lines(),
            'arcs': self._extract_arcs(),
            'circles': self._extract_circles(),
            'polylines': self._extract_polylines(),
            'dimensions': self._extract_dimensions(),
            'layers': self._extract_layers(),
            'blocks': self._extract_blocks(),
            'bounds': self._calculate_bounds(),
            'dxf_version': self.doc.dxfversion,
            'units': self._get_units(),
            'metadata': self._extract_metadata()
        }

        return result

    def _extract_lines(self) -> List[Dict[str, Any]]:
        """Extract all LINE entities from the DXF file."""
        lines = []
        for line in self.modelspace.query('LINE'):
            lines.append({
                'start': (line.dxf.start.x, line.dxf.start.y, line.dxf.start.z),
                'end': (line.dxf.end.x, line.dxf.end.y, line.dxf.end.z),
                'layer': line.dxf.layer,
                'color': line.dxf.color,
                'linetype': line.dxf.linetype
            })
        return lines

    def _extract_arcs(self) -> List[Dict[str, Any]]:
        """Extract all ARC entities from the DXF file."""
        arcs = []
        for arc in self.modelspace.query('ARC'):
            arcs.append({
                'center': (arc.dxf.center.x, arc.dxf.center.y, arc.dxf.center.z),
                'radius': arc.dxf.radius,
                'start_angle': arc.dxf.start_angle,
                'end_angle': arc.dxf.end_angle,
                'layer': arc.dxf.layer,
                'color': arc.dxf.color,
                'linetype': arc.dxf.linetype
            })
        return arcs

    def _extract_circles(self) -> List[Dict[str, Any]]:
        """Extract all CIRCLE entities from the DXF file."""
        circles = []
        for circle in self.modelspace.query('CIRCLE'):
            circles.append({
                'center': (circle.dxf.center.x, circle.dxf.center.y, circle.dxf.center.z),
                'radius': circle.dxf.radius,
                'layer': circle.dxf.layer,
                'color': circle.dxf.color,
                'linetype': circle.dxf.linetype
            })
        return circles

    def _extract_polylines(self) -> List[Dict[str, Any]]:
        """Extract all POLYLINE and LWPOLYLINE entities from the DXF file."""
        polylines = []

        # Extract POLYLINE entities
        for polyline in self.modelspace.query('POLYLINE'):
            points = []
            for vertex in polyline.vertices:
                points.append((vertex.dxf.location.x, vertex.dxf.location.y, vertex.dxf.location.z))

            polylines.append({
                'points': points,
                'is_closed': polyline.is_closed,
                'layer': polyline.dxf.layer,
                'color': polyline.dxf.color,
                'linetype': polyline.dxf.linetype,
                'type': 'POLYLINE'
            })

        # Extract LWPOLYLINE (lightweight polyline) entities
        for lwpolyline in self.modelspace.query('LWPOLYLINE'):
            points = []
            for point in lwpolyline.get_points('xy'):
                points.append((point[0], point[1], 0.0))

            polylines.append({
                'points': points,
                'is_closed': lwpolyline.closed,
                'layer': lwpolyline.dxf.layer,
                'color': lwpolyline.dxf.color,
                'linetype': lwpolyline.dxf.linetype,
                'type': 'LWPOLYLINE'
            })

        return polylines

    def _extract_dimensions(self) -> List[Dict[str, Any]]:
        """Extract all DIMENSION entities from the DXF file."""
        dimensions = []
        for dim in self.modelspace.query('DIMENSION'):
            dim_data = {
                'layer': dim.dxf.layer,
                'dimtype': dim.dimtype,
                'text': getattr(dim.dxf, 'text', ''),
            }

            # Try to get dimension points if available
            if hasattr(dim.dxf, 'defpoint'):
                dim_data['defpoint'] = (dim.dxf.defpoint.x, dim.dxf.defpoint.y, dim.dxf.defpoint.z)
            if hasattr(dim.dxf, 'defpoint2'):
                dim_data['defpoint2'] = (dim.dxf.defpoint2.x, dim.dxf.defpoint2.y, dim.dxf.defpoint2.z)
            if hasattr(dim.dxf, 'defpoint3'):
                dim_data['defpoint3'] = (dim.dxf.defpoint3.x, dim.dxf.defpoint3.y, dim.dxf.defpoint3.z)

            dimensions.append(dim_data)

        return dimensions

    def _extract_layers(self) -> List[Dict[str, Any]]:
        """Extract all layer information from the DXF file."""
        layers = []
        for layer in self.doc.layers:
            layers.append({
                'name': layer.dxf.name,
                'color': layer.dxf.color,
                'linetype': layer.dxf.linetype,
                'is_on': layer.is_on(),
                'is_locked': layer.is_locked(),
                'is_frozen': layer.is_frozen()
            })
        return layers

    def _extract_blocks(self) -> List[Dict[str, Any]]:
        """Extract all block definitions from the DXF file."""
        blocks = []
        for block in self.doc.blocks:
            if not block.name.startswith('*'):  # Skip model/paper space blocks
                blocks.append({
                    'name': block.name,
                    'base_point': (block.block.dxf.base_point.x,
                                  block.block.dxf.base_point.y,
                                  block.block.dxf.base_point.z),
                    'entity_count': len(list(block))
                })
        return blocks

    def _calculate_bounds(self) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate the bounding box of all geometry.

        Returns:
            Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        all_points = []

        # Collect points from lines
        for line in self.modelspace.query('LINE'):
            all_points.append([line.dxf.start.x, line.dxf.start.y, line.dxf.start.z])
            all_points.append([line.dxf.end.x, line.dxf.end.y, line.dxf.end.z])

        # Collect points from circles (approximate bounds)
        for circle in self.modelspace.query('CIRCLE'):
            cx, cy, cz = circle.dxf.center.x, circle.dxf.center.y, circle.dxf.center.z
            r = circle.dxf.radius
            all_points.extend([
                [cx - r, cy - r, cz],
                [cx + r, cy + r, cz]
            ])

        # Collect points from arcs (approximate bounds)
        for arc in self.modelspace.query('ARC'):
            cx, cy, cz = arc.dxf.center.x, arc.dxf.center.y, arc.dxf.center.z
            r = arc.dxf.radius
            all_points.extend([
                [cx - r, cy - r, cz],
                [cx + r, cy + r, cz]
            ])

        # Collect points from polylines
        for polyline in self.modelspace.query('POLYLINE'):
            for vertex in polyline.vertices:
                all_points.append([vertex.dxf.location.x, vertex.dxf.location.y, vertex.dxf.location.z])

        for lwpolyline in self.modelspace.query('LWPOLYLINE'):
            for point in lwpolyline.get_points('xy'):
                all_points.append([point[0], point[1], 0.0])

        if not all_points:
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        points_array = np.array(all_points)
        mins = points_array.min(axis=0)
        maxs = points_array.max(axis=0)

        return (float(mins[0]), float(maxs[0]),
                float(mins[1]), float(maxs[1]),
                float(mins[2]), float(maxs[2]))

    def _get_units(self) -> str:
        """Get the drawing units from the DXF file."""
        if hasattr(self.doc.header, '$INSUNITS'):
            units_code = self.doc.header.get('$INSUNITS', 0)
            units_map = {
                0: 'Unitless',
                1: 'Inches',
                2: 'Feet',
                3: 'Miles',
                4: 'Millimeters',
                5: 'Centimeters',
                6: 'Meters',
                7: 'Kilometers',
                8: 'Microinches',
                9: 'Mils',
                10: 'Yards',
                11: 'Angstroms',
                12: 'Nanometers',
                13: 'Microns',
                14: 'Decimeters',
                15: 'Decameters',
                16: 'Hectometers',
                17: 'Gigameters',
                18: 'Astronomical units',
                19: 'Light years',
                20: 'Parsecs'
            }
            return units_map.get(units_code, 'Unknown')
        return 'Unknown'

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the DXF file."""
        metadata = {
            'entity_count': len(list(self.modelspace)),
            'layer_count': len(list(self.doc.layers)),
            'block_count': len([b for b in self.doc.blocks if not b.name.startswith('*')])
        }

        # Try to get additional header information
        try:
            if hasattr(self.doc.header, '$TDCREATE'):
                metadata['creation_date'] = str(self.doc.header.get('$TDCREATE', ''))
            if hasattr(self.doc.header, '$TDUPDATE'):
                metadata['modification_date'] = str(self.doc.header.get('$TDUPDATE', ''))
        except:
            pass

        return metadata

    def get_entity_count(self) -> Dict[str, int]:
        """
        Get count of each entity type in the DXF file.

        Returns:
            Dictionary with entity types as keys and counts as values
        """
        if not self.modelspace:
            return {}

        entity_types = {}
        for entity in self.modelspace:
            entity_type = entity.dxftype()
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        return entity_types
