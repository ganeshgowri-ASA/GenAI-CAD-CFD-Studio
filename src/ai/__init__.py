"""
AI Core Module for GenAI-CAD-CFD-Studio

Natural language processing and computer vision for CAD generation.

This package provides:
- Claude AI integration for intent extraction
- Dimension parsing and validation
- Sketch interpretation using computer vision
- Prompt templates for various CAD agents

Example Usage:
    >>> from ai import ClaudeSkills, DimensionExtractor, SketchInterpreter
    >>>
    >>> # Initialize Claude with API key
    >>> claude = ClaudeSkills(api_key="your-api-key")
    >>>
    >>> # Extract dimensions from natural language
    >>> result = claude.extract_intent_and_dimensions("Create a 10cm x 5cm x 3cm box")
    >>> print(result['type'])  # 'box'
    >>> print(result['dimensions'])  # {'length': 0.1, 'width': 0.05, 'height': 0.03}
    >>>
    >>> # Parse dimensions from text
    >>> extractor = DimensionExtractor()
    >>> dims = extractor.parse_dimensions("10cm x 5cm x 3cm")
    >>>
    >>> # Interpret sketches
    >>> interpreter = SketchInterpreter()
    >>> interpreter.load_image("sketch.png")
    >>> edges = interpreter.detect_edges()
    >>> contours = interpreter.extract_contours(edges)
    >>> geometries = interpreter.contour_to_geometry(contours)
"""

from .claude_skills import ClaudeSkills
from .dimension_extractor import DimensionExtractor
from .sketch_interpreter import SketchInterpreter
from .prompt_templates import (
    ZOO_KCL_TEMPLATE,
    ADAM_NL_TEMPLATE,
    BUILD123D_PYTHON_TEMPLATE,
    DIMENSION_EXTRACTION_TEMPLATE,
    CLARIFICATION_TEMPLATE,
    format_prompt,
    optimize_for_agent,
    get_template_for_agent
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "ClaudeSkills",
    "DimensionExtractor",
    "SketchInterpreter",

    # Template constants
    "ZOO_KCL_TEMPLATE",
    "ADAM_NL_TEMPLATE",
    "BUILD123D_PYTHON_TEMPLATE",
    "DIMENSION_EXTRACTION_TEMPLATE",
    "CLARIFICATION_TEMPLATE",

    # Template functions
    "format_prompt",
    "optimize_for_agent",
    "get_template_for_agent",
]
