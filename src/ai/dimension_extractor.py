"""
Dimension extraction and validation for CAD models.

This module provides utilities to parse dimensional information from text,
validate dimensions, and convert between different units.
"""

import re
from typing import Dict, List, Optional, Tuple, Union


class DimensionExtractor:
    """
    Extract and validate dimensional information from text.

    Supports various unit formats (cm, mm, m, inches, feet) and converts
    all measurements to meters internally for consistency.
    """

    # Conversion factors to meters
    UNIT_CONVERSIONS = {
        'm': 1.0,
        'meter': 1.0,
        'meters': 1.0,
        'cm': 0.01,
        'centimeter': 0.01,
        'centimeters': 0.01,
        'mm': 0.001,
        'millimeter': 0.001,
        'millimeters': 0.001,
        'in': 0.0254,
        'inch': 0.0254,
        'inches': 0.0254,
        '"': 0.0254,
        'ft': 0.3048,
        'foot': 0.3048,
        'feet': 0.3048,
        "'": 0.3048,
    }

    # Regex patterns for dimension extraction
    DIMENSION_PATTERNS = [
        # Pattern: "10cm x 5cm x 3cm"
        r'(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
        # Pattern: "10 x 5 x 3 cm" (common unit at end)
        r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
        # Pattern: "length: 10cm, width: 5cm, height: 3cm"
        r'length[:\s]+(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
        # Pattern: "width: 5cm" or "w: 5cm"
        r'(?:width|w)[:\s]+(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
        # Pattern: "height: 3cm" or "h: 3cm"
        r'(?:height|h)[:\s]+(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
        # Pattern: "diameter: 10cm" or "d: 10cm"
        r'(?:diameter|d)[:\s]+(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
        # Pattern: "radius: 5cm" or "r: 5cm"
        r'(?:radius|r)[:\s]+(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')',
    ]

    def __init__(self):
        """Initialize the DimensionExtractor."""
        pass

    def parse_dimensions(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Parse dimensional information from text.

        Args:
            text: Input text containing dimensional information

        Returns:
            Dictionary containing extracted dimensions and metadata:
            {
                'length': float (in meters),
                'width': float (in meters),
                'height': float (in meters),
                'diameter': float (in meters),
                'radius': float (in meters),
                'original_unit': str,
                'format': str (description of detected format)
            }

        Example:
            >>> extractor = DimensionExtractor()
            >>> dims = extractor.parse_dimensions("10cm x 5cm x 3cm")
            >>> dims['length']  # Returns 0.1 (meters)
        """
        text = text.lower().strip()
        dimensions = {}

        # Try pattern: "10cm x 5cm x 3cm" (individual units)
        pattern = self.DIMENSION_PATTERNS[0]
        match = re.search(pattern, text)
        if match:
            length, length_unit, width, width_unit, height, height_unit = match.groups()
            dimensions['length'] = self._convert_to_meters(float(length), length_unit)
            dimensions['width'] = self._convert_to_meters(float(width), width_unit)
            dimensions['height'] = self._convert_to_meters(float(height), height_unit)
            dimensions['original_unit'] = length_unit
            dimensions['format'] = 'l x w x h (individual units)'
            return dimensions

        # Try pattern: "10 x 5 x 3 cm" (common unit)
        pattern = self.DIMENSION_PATTERNS[1]
        match = re.search(pattern, text)
        if match:
            length, width, height, unit = match.groups()
            dimensions['length'] = self._convert_to_meters(float(length), unit)
            dimensions['width'] = self._convert_to_meters(float(width), unit)
            dimensions['height'] = self._convert_to_meters(float(height), unit)
            dimensions['original_unit'] = unit
            dimensions['format'] = 'l x w x h (common unit)'
            return dimensions

        # Try individual dimension patterns
        for i, param_name in enumerate(['length', 'width', 'height', 'diameter', 'radius'], start=2):
            pattern = self.DIMENSION_PATTERNS[i]
            match = re.search(pattern, text)
            if match:
                value, unit = match.groups()
                dimensions[param_name] = self._convert_to_meters(float(value), unit)
                if 'original_unit' not in dimensions:
                    dimensions['original_unit'] = unit
                if 'format' not in dimensions:
                    dimensions['format'] = 'labeled dimensions'

        return dimensions

    def _convert_to_meters(self, value: float, unit: str) -> float:
        """
        Convert a value from the given unit to meters.

        Args:
            value: Numerical value
            unit: Unit string (cm, mm, m, in, ft, etc.)

        Returns:
            Value converted to meters

        Raises:
            ValueError: If unit is not recognized
        """
        unit = unit.lower().strip()
        if unit not in self.UNIT_CONVERSIONS:
            raise ValueError(f"Unknown unit: {unit}")

        return value * self.UNIT_CONVERSIONS[unit]

    def convert_from_meters(self, value: float, target_unit: str) -> float:
        """
        Convert a value from meters to the target unit.

        Args:
            value: Value in meters
            target_unit: Target unit (cm, mm, m, in, ft, etc.)

        Returns:
            Value converted to target unit

        Raises:
            ValueError: If unit is not recognized
        """
        target_unit = target_unit.lower().strip()
        if target_unit not in self.UNIT_CONVERSIONS:
            raise ValueError(f"Unknown unit: {target_unit}")

        return value / self.UNIT_CONVERSIONS[target_unit]

    def validate_dimensions(self, dims: Dict[str, Union[float, str]]) -> bool:
        """
        Validate dimensional data.

        Checks that:
        - All dimension values are positive
        - Required dimensions are present for common shapes
        - Values are within reasonable ranges

        Args:
            dims: Dictionary of dimensional data

        Returns:
            True if valid, False otherwise

        Example:
            >>> dims = {'length': 0.1, 'width': 0.05, 'height': 0.03}
            >>> extractor.validate_dimensions(dims)
            True
        """
        # Check for positive values
        numeric_keys = ['length', 'width', 'height', 'diameter', 'radius']
        for key in numeric_keys:
            if key in dims:
                if not isinstance(dims[key], (int, float)):
                    return False
                if dims[key] <= 0:
                    return False
                # Check reasonable range (0.001mm to 100m)
                if dims[key] < 1e-6 or dims[key] > 100:
                    return False

        # Check for minimum required dimensions
        has_box_dims = all(k in dims for k in ['length', 'width', 'height'])
        has_cylinder_dims = ('height' in dims) and (('diameter' in dims) or ('radius' in dims))
        has_sphere_dims = ('diameter' in dims) or ('radius' in dims)

        if not (has_box_dims or has_cylinder_dims or has_sphere_dims):
            # Not enough dimensions for a basic shape
            return len(dims) > 0  # At least some dimension data

        # Validate radius/diameter relationship if both present
        if 'radius' in dims and 'diameter' in dims:
            # Check if they're consistent (within 1% tolerance)
            expected_diameter = dims['radius'] * 2
            if abs(dims['diameter'] - expected_diameter) / expected_diameter > 0.01:
                return False

        return True

    def suggest_corrections(self, dims: Dict[str, Union[float, str]]) -> List[str]:
        """
        Suggest corrections for dimensional data.

        Args:
            dims: Dictionary of dimensional data

        Returns:
            List of suggestion strings for improving the dimensional data

        Example:
            >>> dims = {'length': 1000}  # Very large value
            >>> suggestions = extractor.suggest_corrections(dims)
        """
        suggestions = []

        numeric_keys = ['length', 'width', 'height', 'diameter', 'radius']

        # Check for missing dimensions
        if not any(k in dims for k in numeric_keys):
            suggestions.append("No dimensional data found. Please specify dimensions.")
            return suggestions

        # Check for very large/small values
        for key in numeric_keys:
            if key in dims:
                value = dims[key]
                if value > 10:
                    suggestions.append(
                        f"{key.capitalize()} is {value}m ({value*100}cm). "
                        f"Did you mean {value}cm instead?"
                    )
                elif value < 0.001:
                    suggestions.append(
                        f"{key.capitalize()} is {value}m ({value*1000}mm). "
                        f"This is very small. Please verify."
                    )

        # Check for incomplete box dimensions
        box_keys = ['length', 'width', 'height']
        present_box_dims = [k for k in box_keys if k in dims]
        if 0 < len(present_box_dims) < 3:
            missing = [k for k in box_keys if k not in dims]
            suggestions.append(
                f"Box dimensions incomplete. Missing: {', '.join(missing)}"
            )

        # Check for radius/diameter consistency
        if 'radius' in dims and 'diameter' not in dims:
            suggestions.append(
                f"Consider adding diameter ({dims['radius']*2}m) for clarity"
            )
        elif 'diameter' in dims and 'radius' not in dims:
            suggestions.append(
                f"Consider adding radius ({dims['diameter']/2}m) for clarity"
            )
        elif 'radius' in dims and 'diameter' in dims:
            expected_diameter = dims['radius'] * 2
            if abs(dims['diameter'] - expected_diameter) / expected_diameter > 0.01:
                suggestions.append(
                    f"Radius and diameter are inconsistent. "
                    f"Radius: {dims['radius']}m, Diameter: {dims['diameter']}m. "
                    f"Expected diameter: {expected_diameter}m"
                )

        if not suggestions:
            suggestions.append("Dimensions look good!")

        return suggestions

    def extract_all_numbers(self, text: str) -> List[Tuple[float, Optional[str]]]:
        """
        Extract all numbers with optional units from text.

        Args:
            text: Input text

        Returns:
            List of tuples (value, unit) where unit may be None

        Example:
            >>> extractor.extract_all_numbers("box 10cm by 5mm tall")
            [(10.0, 'cm'), (5.0, 'mm')]
        """
        pattern = r'(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch|inches|ft|feet|"|\')?'
        matches = re.findall(pattern, text.lower())

        results = []
        for value, unit in matches:
            unit = unit if unit else None
            results.append((float(value), unit))

        return results
