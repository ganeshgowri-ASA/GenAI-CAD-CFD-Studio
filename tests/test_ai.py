"""
Comprehensive tests for AI Core module.

Tests cover:
- Claude dimension extraction (with mocked API calls)
- Dimension parser with various formats
- Sketch interpreter with sample images
- Prompt templates and formatting
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.dimension_extractor import DimensionExtractor
from ai.prompt_templates import (
    format_prompt,
    optimize_for_agent,
    get_template_for_agent,
    ZOO_KCL_TEMPLATE,
    ADAM_NL_TEMPLATE,
    BUILD123D_PYTHON_TEMPLATE
)
from ai.sketch_interpreter import SketchInterpreter
from ai.claude_skills import ClaudeSkills


# ============================================================================
# DimensionExtractor Tests
# ============================================================================

class TestDimensionExtractor:
    """Test dimension extraction and validation."""

    def test_init(self):
        """Test DimensionExtractor initialization."""
        extractor = DimensionExtractor()
        assert extractor is not None

    def test_parse_dimensions_with_individual_units(self):
        """Test parsing '10cm x 5cm x 3cm' format."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions("10cm x 5cm x 3cm")

        assert 'length' in dims
        assert 'width' in dims
        assert 'height' in dims
        assert dims['length'] == pytest.approx(0.1, rel=1e-6)  # 10cm = 0.1m
        assert dims['width'] == pytest.approx(0.05, rel=1e-6)   # 5cm = 0.05m
        assert dims['height'] == pytest.approx(0.03, rel=1e-6)  # 3cm = 0.03m
        assert dims['original_unit'] == 'cm'

    def test_parse_dimensions_with_common_unit(self):
        """Test parsing '10 x 5 x 3 cm' format."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions("10 x 5 x 3 cm")

        assert dims['length'] == pytest.approx(0.1, rel=1e-6)
        assert dims['width'] == pytest.approx(0.05, rel=1e-6)
        assert dims['height'] == pytest.approx(0.03, rel=1e-6)
        assert dims['original_unit'] == 'cm'
        assert dims['format'] == 'l x w x h (common unit)'

    def test_parse_dimensions_millimeters(self):
        """Test parsing dimensions in millimeters."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions("100mm x 50mm x 30mm")

        assert dims['length'] == pytest.approx(0.1, rel=1e-6)
        assert dims['width'] == pytest.approx(0.05, rel=1e-6)
        assert dims['height'] == pytest.approx(0.03, rel=1e-6)
        assert dims['original_unit'] == 'mm'

    def test_parse_dimensions_inches(self):
        """Test parsing dimensions in inches."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions('4" x 2" x 1"')

        assert dims['length'] == pytest.approx(0.1016, rel=1e-3)  # 4 inches
        assert dims['width'] == pytest.approx(0.0508, rel=1e-3)   # 2 inches
        assert dims['height'] == pytest.approx(0.0254, rel=1e-3)  # 1 inch
        assert dims['original_unit'] == '"'

    def test_parse_labeled_dimensions(self):
        """Test parsing labeled dimensions."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions("length: 10cm, width: 5cm, height: 3cm")

        assert 'length' in dims
        assert 'width' in dims
        assert 'height' in dims
        assert dims['length'] == pytest.approx(0.1, rel=1e-6)
        assert dims['format'] == 'labeled dimensions'

    def test_parse_diameter(self):
        """Test parsing diameter."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions("diameter: 10cm")

        assert 'diameter' in dims
        assert dims['diameter'] == pytest.approx(0.1, rel=1e-6)

    def test_parse_radius(self):
        """Test parsing radius."""
        extractor = DimensionExtractor()
        dims = extractor.parse_dimensions("radius: 5cm")

        assert 'radius' in dims
        assert dims['radius'] == pytest.approx(0.05, rel=1e-6)

    def test_convert_to_meters(self):
        """Test unit conversion to meters."""
        extractor = DimensionExtractor()

        assert extractor._convert_to_meters(1, 'm') == 1.0
        assert extractor._convert_to_meters(100, 'cm') == 1.0
        assert extractor._convert_to_meters(1000, 'mm') == 1.0
        assert extractor._convert_to_meters(39.3701, 'in') == pytest.approx(1.0, rel=1e-3)

    def test_convert_to_meters_invalid_unit(self):
        """Test conversion with invalid unit."""
        extractor = DimensionExtractor()

        with pytest.raises(ValueError):
            extractor._convert_to_meters(10, 'invalid')

    def test_convert_from_meters(self):
        """Test unit conversion from meters."""
        extractor = DimensionExtractor()

        assert extractor.convert_from_meters(1.0, 'm') == 1.0
        assert extractor.convert_from_meters(1.0, 'cm') == 100.0
        assert extractor.convert_from_meters(1.0, 'mm') == 1000.0

    def test_validate_dimensions_valid(self):
        """Test validation of valid dimensions."""
        extractor = DimensionExtractor()
        dims = {'length': 0.1, 'width': 0.05, 'height': 0.03}

        assert extractor.validate_dimensions(dims) is True

    def test_validate_dimensions_negative(self):
        """Test validation rejects negative dimensions."""
        extractor = DimensionExtractor()
        dims = {'length': -0.1, 'width': 0.05, 'height': 0.03}

        assert extractor.validate_dimensions(dims) is False

    def test_validate_dimensions_too_large(self):
        """Test validation rejects unreasonably large dimensions."""
        extractor = DimensionExtractor()
        dims = {'length': 1000, 'width': 0.05, 'height': 0.03}  # 1000m is too large

        assert extractor.validate_dimensions(dims) is False

    def test_validate_dimensions_too_small(self):
        """Test validation rejects unreasonably small dimensions."""
        extractor = DimensionExtractor()
        dims = {'length': 1e-7, 'width': 0.05, 'height': 0.03}  # Too small

        assert extractor.validate_dimensions(dims) is False

    def test_validate_dimensions_inconsistent_radius_diameter(self):
        """Test validation detects inconsistent radius/diameter."""
        extractor = DimensionExtractor()
        dims = {'radius': 0.05, 'diameter': 0.15}  # Should be 0.1

        assert extractor.validate_dimensions(dims) is False

    def test_suggest_corrections_very_large(self):
        """Test suggestions for very large values."""
        extractor = DimensionExtractor()
        dims = {'length': 100}  # 100 meters is probably meant to be cm

        suggestions = extractor.suggest_corrections(dims)
        assert len(suggestions) > 0
        assert any('100m' in s for s in suggestions)

    def test_suggest_corrections_incomplete_box(self):
        """Test suggestions for incomplete box dimensions."""
        extractor = DimensionExtractor()
        dims = {'length': 0.1, 'width': 0.05}  # Missing height

        suggestions = extractor.suggest_corrections(dims)
        assert any('incomplete' in s.lower() for s in suggestions)
        assert any('height' in s.lower() for s in suggestions)

    def test_suggest_corrections_good_dimensions(self):
        """Test suggestions for valid dimensions."""
        extractor = DimensionExtractor()
        dims = {'length': 0.1, 'width': 0.05, 'height': 0.03}

        suggestions = extractor.suggest_corrections(dims)
        assert any('good' in s.lower() for s in suggestions)

    def test_extract_all_numbers(self):
        """Test extracting all numbers from text."""
        extractor = DimensionExtractor()
        results = extractor.extract_all_numbers("box 10cm by 5mm tall")

        assert len(results) >= 2
        assert (10.0, 'cm') in results
        assert (5.0, 'mm') in results


# ============================================================================
# PromptTemplates Tests
# ============================================================================

class TestPromptTemplates:
    """Test prompt template formatting and optimization."""

    def test_format_prompt_basic(self):
        """Test basic prompt formatting."""
        template = "Create a {object_type} with {dimensions}"
        params = {'object_type': 'box', 'dimensions': '10x5x3'}

        result = format_prompt(template, params)
        assert 'box' in result
        assert '10x5x3' in result

    def test_format_prompt_with_defaults(self):
        """Test formatting with default parameters."""
        template = "Object: {object_type}, Material: {materials}"
        params = {'object_type': 'cylinder'}

        result = format_prompt(template, params)
        assert 'cylinder' in result
        assert 'default material' in result

    def test_format_prompt_zoo_kcl(self):
        """Test ZOO_KCL_TEMPLATE formatting."""
        params = {
            'description': 'A simple box',
            'object_type': 'box',
            'dimensions': '10x5x3',
            'unit': 'cm',
            'materials': 'aluminum',
            'constraints': 'none'
        }

        result = format_prompt(ZOO_KCL_TEMPLATE, params)
        assert 'box' in result
        assert 'aluminum' in result
        assert 'KCL' in result

    def test_format_prompt_build123d(self):
        """Test BUILD123D_PYTHON_TEMPLATE formatting."""
        params = {
            'description': 'A cylinder',
            'object_type': 'cylinder',
            'dimensions': 'r=5, h=10',
            'unit': 'cm',
            'materials': 'steel',
            'constraints': 'none'
        }

        result = format_prompt(BUILD123D_PYTHON_TEMPLATE, params)
        assert 'cylinder' in result
        assert 'Build123D' in result
        assert 'Python' in result

    def test_optimize_for_agent_zoo_kcl(self):
        """Test optimization for Zoo KCL agent."""
        prompt = "Create a box"
        result = optimize_for_agent(prompt, 'zoo_kcl')

        assert 'sketch' in result.lower()
        assert 'Create a box' in result

    def test_optimize_for_agent_adam_nl(self):
        """Test optimization for ADAM NL agent."""
        prompt = "Create a cylinder"
        result = optimize_for_agent(prompt, 'adam_nl')

        assert 'natural language' in result.lower()
        assert 'Create a cylinder' in result

    def test_optimize_for_agent_build123d(self):
        """Test optimization for Build123D agent."""
        prompt = "Create a sphere"
        result = optimize_for_agent(prompt, 'build123d')

        assert 'python' in result.lower()
        assert 'Build123D' in result

    def test_optimize_for_agent_invalid(self):
        """Test optimization with invalid agent."""
        with pytest.raises(ValueError):
            optimize_for_agent("Create a box", "invalid_agent")

    def test_get_template_for_agent(self):
        """Test getting template for specific agent."""
        template = get_template_for_agent('zoo_kcl')
        assert template == ZOO_KCL_TEMPLATE

        template = get_template_for_agent('adam_nl')
        assert template == ADAM_NL_TEMPLATE

        template = get_template_for_agent('build123d')
        assert template == BUILD123D_PYTHON_TEMPLATE

    def test_get_template_for_agent_invalid(self):
        """Test getting template with invalid agent."""
        with pytest.raises(ValueError):
            get_template_for_agent('invalid_agent')


# ============================================================================
# SketchInterpreter Tests
# ============================================================================

class TestSketchInterpreter:
    """Test sketch interpretation and computer vision."""

    def test_init(self):
        """Test SketchInterpreter initialization."""
        interpreter = SketchInterpreter()
        assert interpreter is not None
        assert interpreter.image is None
        assert interpreter.edges is None

    def test_load_image_from_bytes(self):
        """Test loading image from bytes."""
        interpreter = SketchInterpreter()

        # Create a simple test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]  # White square

        # Encode to PNG bytes
        _, buffer = cv2.imencode('.png', test_image)
        image_bytes = buffer.tobytes()

        # Load from bytes
        loaded = interpreter.load_image_from_bytes(image_bytes)

        assert loaded is not None
        assert loaded.shape == (100, 100, 3)
        assert interpreter.image is not None

    def test_detect_edges(self):
        """Test edge detection."""
        interpreter = SketchInterpreter()

        # Create test image with a white square
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]

        edges = interpreter.detect_edges(test_image)

        assert edges is not None
        assert edges.shape == (100, 100)
        assert interpreter.edges is not None

    def test_detect_edges_no_image_loaded(self):
        """Test edge detection without loaded image."""
        interpreter = SketchInterpreter()

        with pytest.raises(ValueError):
            interpreter.detect_edges()

    def test_extract_contours(self):
        """Test contour extraction."""
        interpreter = SketchInterpreter()

        # Create test image with a white square
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[50:150, 50:150] = [255, 255, 255]

        edges = interpreter.detect_edges(test_image)
        contours = interpreter.extract_contours(edges)

        assert contours is not None
        assert len(contours) > 0
        assert interpreter.contours is not None

    def test_extract_contours_with_min_area(self):
        """Test contour extraction with minimum area filter."""
        interpreter = SketchInterpreter()

        # Create test image with two squares of different sizes
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        test_image[10:30, 10:30] = [255, 255, 255]    # Small square
        test_image[50:150, 50:150] = [255, 255, 255]  # Large square

        edges = interpreter.detect_edges(test_image)
        contours = interpreter.extract_contours(edges, min_area=500)

        # Should only get the large square
        assert len(contours) >= 1

    def test_contour_to_geometry_rectangle(self):
        """Test converting rectangle contour to geometry."""
        interpreter = SketchInterpreter()

        # Create test image with a rectangle
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 120), (255, 255, 255), -1)

        edges = interpreter.detect_edges(test_image)
        contours = interpreter.extract_contours(edges)
        geometries = interpreter.contour_to_geometry(contours)

        assert len(geometries) > 0
        assert geometries[0]['type'] == 'rectangle'
        assert 'width' in geometries[0]['properties']
        assert 'height' in geometries[0]['properties']

    def test_contour_to_geometry_circle(self):
        """Test converting circle contour to geometry."""
        interpreter = SketchInterpreter()

        # Create test image with a circle
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(test_image, (100, 100), 40, (255, 255, 255), -1)

        edges = interpreter.detect_edges(test_image)
        contours = interpreter.extract_contours(edges)
        geometries = interpreter.contour_to_geometry(contours)

        assert len(geometries) > 0
        # Should be detected as circle or polygon
        assert geometries[0]['type'] in ['circle', 'ellipse', 'polygon']

    def test_contour_to_geometry_triangle(self):
        """Test converting triangle contour to geometry."""
        interpreter = SketchInterpreter()

        # Create test image with a triangle
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        pts = np.array([[100, 30], [50, 150], [150, 150]], np.int32)
        cv2.fillPoly(test_image, [pts], (255, 255, 255))

        edges = interpreter.detect_edges(test_image)
        contours = interpreter.extract_contours(edges)
        geometries = interpreter.contour_to_geometry(contours)

        assert len(geometries) > 0
        assert geometries[0]['type'] == 'triangle'
        assert geometries[0]['vertices'] == 3

    def test_visualize_detection(self):
        """Test visualization of detected shapes."""
        interpreter = SketchInterpreter()

        # Create test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)

        interpreter.load_image_from_bytes(cv2.imencode('.png', test_image)[1].tobytes())
        edges = interpreter.detect_edges()
        contours = interpreter.extract_contours(edges)
        geometries = interpreter.contour_to_geometry(contours)

        annotated = interpreter.visualize_detection(geometries=geometries)

        assert annotated is not None
        assert annotated.shape == test_image.shape

    def test_get_cad_specifications(self):
        """Test getting CAD specifications from detected geometry."""
        interpreter = SketchInterpreter()

        # Create test image
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)

        interpreter.load_image_from_bytes(cv2.imencode('.png', test_image)[1].tobytes())
        interpreter.detect_edges()
        interpreter.extract_contours()

        specs = interpreter.get_cad_specifications()

        assert 'num_shapes' in specs
        assert 'shapes' in specs
        assert 'image_dimensions' in specs
        assert specs['num_shapes'] > 0


# ============================================================================
# ClaudeSkills Tests (with mocked API)
# ============================================================================

class TestClaudeSkills:
    """Test Claude integration with mocked API calls."""

    def test_init_valid_api_key(self):
        """Test initialization with valid API key."""
        skills = ClaudeSkills(api_key="test-api-key")
        assert skills is not None
        assert skills.dimension_extractor is not None

    def test_init_empty_api_key(self):
        """Test initialization with empty API key."""
        with pytest.raises(ValueError):
            ClaudeSkills(api_key="")

    def test_init_none_api_key(self):
        """Test initialization with None API key."""
        with pytest.raises(ValueError):
            ClaudeSkills(api_key=None)

    @patch('anthropic.Anthropic')
    def test_extract_intent_and_dimensions_success(self, mock_anthropic):
        """Test successful intent and dimension extraction."""
        # Mock the API response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''```json
{
    "object_type": "box",
    "dimensions": {
        "length": 0.1,
        "width": 0.05,
        "height": 0.03
    },
    "unit": "m",
    "materials": "aluminum",
    "constraints": ["rigid"],
    "confidence": 0.9,
    "ambiguities": []
}
```'''

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        skills = ClaudeSkills(api_key="test-key")
        result = skills.extract_intent_and_dimensions("Create a 10cm x 5cm x 3cm aluminum box")

        assert result['type'] == 'box'
        assert result['unit'] == 'm'
        assert result['materials'] == 'aluminum'
        assert result['confidence'] == 0.9

    @patch('anthropic.Anthropic')
    def test_extract_intent_and_dimensions_with_fallback(self, mock_anthropic):
        """Test dimension extraction with regex fallback on API error."""
        # Mock API to raise an error
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        skills = ClaudeSkills(api_key="test-key")
        result = skills.extract_intent_and_dimensions("10cm x 5cm x 3cm box")

        # Should fall back to regex extraction
        assert 'type' in result
        assert 'dimensions' in result
        assert result['confidence'] <= 0.5  # Lower confidence due to fallback

    @patch('anthropic.Anthropic')
    def test_generate_cad_description(self, mock_anthropic):
        """Test CAD description generation."""
        # Mock extraction response
        mock_extract_response = Mock()
        mock_extract_response.content = [Mock()]
        mock_extract_response.content[0].text = '''```json
{
    "object_type": "cylinder",
    "dimensions": {"radius": 0.05, "height": 0.1},
    "unit": "m",
    "materials": "steel",
    "constraints": [],
    "confidence": 0.9,
    "ambiguities": []
}
```'''

        # Mock generation response
        mock_gen_response = Mock()
        mock_gen_response.content = [Mock()]
        mock_gen_response.content[0].text = "from build123d import *\n# Create cylinder\ncyl = Cylinder(radius=0.05, height=0.1)"

        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_extract_response, mock_gen_response]
        mock_anthropic.return_value = mock_client

        skills = ClaudeSkills(api_key="test-key")
        result = skills.generate_cad_description("Create a steel cylinder 5cm radius, 10cm tall")

        assert 'Cylinder' in result or 'cylinder' in result

    @patch('anthropic.Anthropic')
    def test_clarify_ambiguity(self, mock_anthropic):
        """Test clarification question generation."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "1. What are the dimensions?\n2. What unit?\n3. What material?"

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        skills = ClaudeSkills(api_key="test-key")
        result = skills.clarify_ambiguity("Create a box", ['length', 'width', 'height'])

        assert 'dimensions' in result.lower() or 'length' in result.lower()

    @patch('anthropic.Anthropic')
    def test_validate_and_extract(self, mock_anthropic):
        """Test combined validation and extraction."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''```json
{
    "object_type": "box",
    "dimensions": {"length": 0.1, "width": 0.05, "height": 0.03},
    "unit": "m",
    "materials": "default",
    "constraints": [],
    "confidence": 0.9,
    "ambiguities": []
}
```'''

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        skills = ClaudeSkills(api_key="test-key")
        result = skills.validate_and_extract("10cm x 5cm x 3cm box")

        assert 'valid' in result
        assert 'suggestions' in result
        assert result['valid'] is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration between modules."""

    def test_dimension_extractor_with_claude_format(self):
        """Test that DimensionExtractor works with Claude output format."""
        extractor = DimensionExtractor()

        # Simulate Claude-extracted dimensions
        dims = {
            'length': 0.1,
            'width': 0.05,
            'height': 0.03,
            'original_unit': 'm'
        }

        assert extractor.validate_dimensions(dims) is True
        suggestions = extractor.suggest_corrections(dims)
        assert len(suggestions) > 0

    def test_sketch_to_dimensions_workflow(self):
        """Test workflow from sketch to dimensions."""
        interpreter = SketchInterpreter()

        # Create a test box sketch
        test_image = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (150, 150), (255, 255, 255), -1)

        # Process image
        interpreter.load_image_from_bytes(cv2.imencode('.png', test_image)[1].tobytes())
        interpreter.detect_edges()
        interpreter.extract_contours()
        specs = interpreter.get_cad_specifications()

        # Verify we got useful data
        assert specs['num_shapes'] > 0
        assert len(specs['shapes']) > 0
        assert specs['shapes'][0]['type'] in ['rectangle', 'polygon']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src/ai', '--cov-report=term-missing'])
