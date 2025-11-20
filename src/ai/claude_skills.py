"""
Claude AI Skills for Design Studio
Handles AI-powered dimension extraction and design interpretation
"""
import re
import json
from typing import Dict, List, Optional, Any


class ClaudeSkills:
    """
    AI skills for extracting CAD parameters from natural language
    """

    # Common object types and their expected parameters
    OBJECT_TEMPLATES = {
        "box": ["length", "width", "height"],
        "cube": ["size"],
        "cylinder": ["radius", "height"],
        "sphere": ["radius"],
        "cone": ["radius", "height"],
        "torus": ["major_radius", "minor_radius"],
        "pipe": ["inner_radius", "outer_radius", "length"],
        "plate": ["length", "width", "thickness"]
    }

    # Unit conversion patterns
    UNIT_PATTERNS = {
        "mm": ["mm", "millimeter", "millimeters"],
        "cm": ["cm", "centimeter", "centimeters"],
        "m": ["m", "meter", "meters"],
        "inches": ["in", "inch", "inches", '"'],
        "feet": ["ft", "foot", "feet", "'"]
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude skills

        Args:
            api_key: Optional Anthropic API key for Claude integration
        """
        self.api_key = api_key

    def extract_dimensions(self, prompt: str) -> Dict[str, Any]:
        """
        Extract CAD dimensions and parameters from natural language prompt

        Args:
            prompt: User's natural language description

        Returns:
            Dictionary containing extracted parameters
        """
        # Convert to lowercase for easier parsing
        prompt_lower = prompt.lower()

        # Extract object type
        object_type = self._extract_object_type(prompt_lower)

        # Extract dimensions
        dimensions = self._extract_numeric_dimensions(prompt)

        # Extract unit
        unit = self._extract_unit(prompt_lower)

        # Build parameter dictionary
        params = {
            "object_type": object_type,
            "unit": unit,
            "description": prompt
        }

        # Add extracted dimensions
        params.update(dimensions)

        # If no dimensions found, provide defaults based on object type
        if not dimensions:
            params.update(self._get_default_dimensions(object_type))

        return params

    def _extract_object_type(self, prompt: str) -> str:
        """
        Extract the object type from the prompt

        Args:
            prompt: Lowercase prompt text

        Returns:
            Detected object type
        """
        # Check for known object types
        for obj_type in self.OBJECT_TEMPLATES.keys():
            if obj_type in prompt:
                return obj_type

        # Check for common synonyms
        if "rectangular" in prompt or "block" in prompt:
            return "box"
        if "round" in prompt or "circular" in prompt:
            if "flat" in prompt or "disc" in prompt:
                return "cylinder"
            return "sphere"
        if "square" in prompt and ("plate" in prompt or "flat" in prompt):
            return "plate"

        return "box"  # Default

    def _extract_numeric_dimensions(self, prompt: str) -> Dict[str, float]:
        """
        Extract numeric dimensions from the prompt

        Args:
            prompt: User prompt

        Returns:
            Dictionary of dimension name to value
        """
        dimensions = {}

        # Patterns for dimension extraction
        patterns = [
            # "length 100mm" or "length: 100 mm"
            (r'length[:\s]+(\d+\.?\d*)', 'length'),
            (r'width[:\s]+(\d+\.?\d*)', 'width'),
            (r'height[:\s]+(\d+\.?\d*)', 'height'),
            (r'depth[:\s]+(\d+\.?\d*)', 'depth'),
            (r'radius[:\s]+(\d+\.?\d*)', 'radius'),
            (r'diameter[:\s]+(\d+\.?\d*)', 'diameter'),
            (r'thickness[:\s]+(\d+\.?\d*)', 'thickness'),
            (r'size[:\s]+(\d+\.?\d*)', 'size'),
            # "100mm long"
            (r'(\d+\.?\d*)\s*(?:mm|cm|m|in|inches|ft)?\s*long', 'length'),
            (r'(\d+\.?\d*)\s*(?:mm|cm|m|in|inches|ft)?\s*wide', 'width'),
            (r'(\d+\.?\d*)\s*(?:mm|cm|m|in|inches|ft)?\s*tall', 'height'),
            (r'(\d+\.?\d*)\s*(?:mm|cm|m|in|inches|ft)?\s*high', 'height'),
            # "100 x 50 x 30" format
        ]

        for pattern, dim_name in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                try:
                    dimensions[dim_name] = float(match.group(1))
                except (ValueError, IndexError):
                    pass

        # Try to extract from "X x Y x Z" format
        dimension_match = re.search(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*x\s*(\d+\.?\d*)', prompt)
        if dimension_match and not dimensions:
            dimensions['length'] = float(dimension_match.group(1))
            dimensions['width'] = float(dimension_match.group(2))
            dimensions['height'] = float(dimension_match.group(3))

        # Convert diameter to radius if needed
        if 'diameter' in dimensions and 'radius' not in dimensions:
            dimensions['radius'] = dimensions['diameter'] / 2

        return dimensions

    def _extract_unit(self, prompt: str) -> str:
        """
        Extract the unit of measurement from the prompt

        Args:
            prompt: Lowercase prompt text

        Returns:
            Detected unit (defaults to 'mm')
        """
        for unit, patterns in self.UNIT_PATTERNS.items():
            for pattern in patterns:
                if pattern in prompt:
                    return unit

        return "mm"  # Default unit

    def _get_default_dimensions(self, object_type: str) -> Dict[str, float]:
        """
        Get default dimensions for an object type

        Args:
            object_type: Type of object

        Returns:
            Dictionary of default dimensions
        """
        defaults = {
            "box": {"length": 100, "width": 50, "height": 30},
            "cube": {"size": 50},
            "cylinder": {"radius": 25, "height": 100},
            "sphere": {"radius": 50},
            "cone": {"radius": 50, "height": 100},
            "torus": {"major_radius": 50, "minor_radius": 10},
            "pipe": {"inner_radius": 20, "outer_radius": 25, "length": 100},
            "plate": {"length": 100, "width": 100, "thickness": 5}
        }

        return defaults.get(object_type, {"length": 100, "width": 50, "height": 30})

    def generate_ai_response(self, prompt: str, extracted_params: Dict[str, Any]) -> str:
        """
        Generate an AI response confirming the extracted parameters

        Args:
            prompt: User's original prompt
            extracted_params: Extracted parameters

        Returns:
            AI response text
        """
        object_type = extracted_params.get("object_type", "object")
        unit = extracted_params.get("unit", "mm")

        # Build response
        response_parts = [
            f"I understand you want to create a **{object_type}**.",
            "",
            "Here's what I've extracted:"
        ]

        # Add dimensions
        for key, value in extracted_params.items():
            if key not in ["object_type", "unit", "description"] and isinstance(value, (int, float)):
                response_parts.append(f"- **{key.replace('_', ' ').title()}**: {value} {unit}")

        response_parts.extend([
            "",
            "Please review and adjust the parameters in the form below, then click **Generate** to create your 3D model."
        ])

        return "\n".join(response_parts)

    def suggest_improvements(self, params: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements or considerations for the design

        Args:
            params: Current design parameters

        Returns:
            List of suggestions
        """
        suggestions = []

        object_type = params.get("object_type", "")

        # Aspect ratio checks
        if "length" in params and "width" in params:
            ratio = params["length"] / params["width"]
            if ratio > 10:
                suggestions.append("💡 High length-to-width ratio. Consider if this is intentional.")

        # Thickness checks for plates
        if object_type == "plate" and "thickness" in params:
            if params["thickness"] > params.get("length", 100) / 2:
                suggestions.append("💡 Thickness is quite large relative to length. Consider reducing.")

        # Volume checks
        if "radius" in params and "height" in params:
            volume = 3.14159 * (params["radius"] ** 2) * params["height"]
            if volume > 1000000:
                suggestions.append(f"💡 Large volume (~{volume/1000000:.1f}L). Ensure this is correct.")

        return suggestions

    def get_suggested_materials(self, object_type: str) -> List[str]:
        """
        Get suggested materials based on object type

        Args:
            object_type: Type of object

        Returns:
            List of suggested materials
        """
        materials = {
            "box": ["Aluminum", "Steel", "Plastic (ABS)", "Wood"],
            "cylinder": ["Aluminum", "Steel", "Brass", "Plastic (PLA)"],
            "plate": ["Aluminum", "Steel", "Acrylic", "Carbon Fiber"],
            "pipe": ["Steel", "Copper", "PVC", "Stainless Steel"],
            "sphere": ["Steel", "Plastic (ABS)", "Rubber"],
        }

        return materials.get(object_type, ["Aluminum", "Steel", "Plastic"])
