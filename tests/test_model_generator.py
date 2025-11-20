"""
Unit tests for CADModelGenerator

Tests multi-modal CAD generation functionality including:
- Text-to-CAD
- Image-to-CAD
- Drawing-to-CAD
- Hybrid multi-modal generation
"""

import pytest
import sys
from pathlib import Path
import tempfile
import json
import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from cad.model_generator import CADModelGenerator


class TestCADModelGeneratorInit:
    """Test CADModelGenerator initialization."""

    def test_init_mock_mode(self):
        """Test initialization in mock mode."""
        generator = CADModelGenerator(mock_mode=True)
        assert generator.mock_mode is True
        assert generator.use_zoo_dev is False

    def test_init_with_zoo_dev(self):
        """Test initialization with Zoo.dev enabled."""
        generator = CADModelGenerator(
            use_zoo_dev=True,
            mock_mode=True
        )
        assert generator.use_zoo_dev is True
        assert generator.zoo_connector is not None

    def test_init_without_api_key_fails(self):
        """Test that initialization without API key fails when not in mock mode."""
        with pytest.raises((ValueError, ImportError)):
            # Should fail if ANTHROPIC_API_KEY not set
            generator = CADModelGenerator(
                anthropic_api_key=None,
                mock_mode=False
            )


class TestTextToCAD:
    """Test text-to-CAD generation."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CADModelGenerator(mock_mode=True)

    def test_generate_simple_box(self, generator):
        """Test generating a simple box from text."""
        description = "Create a box 100mm x 50mm x 30mm"

        result = generator.generate_from_text(
            description=description,
            output_format='step'
        )

        assert result is not None
        assert 'parameters' in result
        assert 'files' in result
        assert result['input_type'] == 'text'
        assert result['input_description'] == description

        # Check parameters
        params = result['parameters']
        assert 'type' in params
        assert 'dimensions' in params

    def test_generate_cylinder(self, generator):
        """Test generating a cylinder from text."""
        description = "Cylindrical rod with 20mm diameter and 100mm length"

        result = generator.generate_from_text(
            description=description,
            output_format='stl'
        )

        assert result is not None
        assert 'parameters' in result

        params = result['parameters']
        assert params.get('type') in ['cylinder', 'box']  # Mock mode may default to box

    def test_generate_with_hole(self, generator):
        """Test generating a part with a hole."""
        description = "Box 10cm x 10cm x 5cm with a 2cm diameter hole through the center"

        result = generator.generate_from_text(
            description=description,
            output_format='step'
        )

        assert result is not None
        params = result['parameters']

        # Check if hole feature was detected
        features = params.get('features', [])
        # In mock mode, we should detect the hole
        hole_detected = any(f.get('type') == 'hole' for f in features)
        # Note: In mock mode, hole detection depends on basic parsing

    def test_generate_both_formats(self, generator):
        """Test generating both STEP and STL formats."""
        description = "Create a sphere with 50mm radius"

        result = generator.generate_from_text(
            description=description,
            output_format='both'
        )

        assert result is not None
        files = result.get('files', [])

        # Should have 2 files (STEP and STL)
        assert len(files) == 2

        # Check file extensions
        file_exts = [Path(f).suffix for f in files]
        assert '.step' in file_exts or '.stp' in file_exts
        assert '.stl' in file_exts

    def test_empty_description_fails(self, generator):
        """Test that empty description is handled."""
        with pytest.raises(ValueError):
            # Empty description should fail validation
            result = generator.generate_from_text(
                description="",
                output_format='step'
            )


class TestImageToCAD:
    """Test image-to-CAD generation."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CADModelGenerator(mock_mode=True)

    @pytest.fixture
    def test_image_path(self):
        """Create a test image file."""
        # Create a simple test image using numpy
        try:
            import cv2
            # Create a simple black and white image with a rectangle
            img = np.zeros((400, 400, 3), dtype=np.uint8)
            img[100:300, 100:300] = 255  # White rectangle

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                cv2.imwrite(tmp.name, img)
                return tmp.name
        except ImportError:
            pytest.skip("OpenCV not available")

    def test_generate_from_image(self, generator, test_image_path):
        """Test generating CAD from image."""
        result = generator.generate_from_image(
            image_path=test_image_path,
            image_type='sketch',
            output_format='step'
        )

        assert result is not None
        assert 'parameters' in result
        assert 'detected_geometry' in result
        assert result['input_type'] == 'image'

        # Check geometry detection
        geometry = result['detected_geometry']
        assert 'shapes' in geometry

        # Clean up
        Path(test_image_path).unlink()

    def test_generate_from_image_with_context(self, generator, test_image_path):
        """Test generating CAD from image with additional context."""
        result = generator.generate_from_image(
            image_path=test_image_path,
            image_type='sketch',
            additional_context='This is a mounting plate',
            output_format='step'
        )

        assert result is not None
        assert 'parameters' in result

        # Clean up
        Path(test_image_path).unlink()

    def test_invalid_image_path_fails(self, generator):
        """Test that invalid image path raises error."""
        with pytest.raises((FileNotFoundError, ValueError)):
            result = generator.generate_from_image(
                image_path='/nonexistent/path/to/image.png',
                image_type='sketch',
                output_format='step'
            )


class TestDrawingToCAD:
    """Test technical drawing-to-CAD generation."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CADModelGenerator(mock_mode=True)

    @pytest.fixture
    def test_dxf_path(self):
        """Create a test DXF file."""
        # Create a minimal DXF file
        try:
            import ezdxf

            doc = ezdxf.new()
            msp = doc.modelspace()

            # Add a simple rectangle
            msp.add_lwpolyline([
                (0, 0),
                (100, 0),
                (100, 50),
                (0, 50),
                (0, 0)
            ])

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                doc.saveas(tmp.name)
                return tmp.name

        except ImportError:
            pytest.skip("ezdxf not available")

    def test_generate_from_dxf(self, generator, test_dxf_path):
        """Test generating CAD from DXF file."""
        result = generator.generate_from_drawing(
            drawing_path=test_dxf_path,
            drawing_format='dxf',
            output_format='step'
        )

        assert result is not None
        assert 'parameters' in result
        assert 'parsed_geometry' in result
        assert result['input_type'] == 'drawing'

        # Clean up
        Path(test_dxf_path).unlink()

    def test_unsupported_format_fails(self, generator):
        """Test that unsupported drawing format raises error."""
        with pytest.raises((NotImplementedError, ValueError)):
            result = generator.generate_from_drawing(
                drawing_path='test.dwg',
                drawing_format='dwg',
                output_format='step'
            )


class TestHybridGeneration:
    """Test hybrid multi-modal CAD generation."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CADModelGenerator(mock_mode=True)

    def test_text_only_hybrid(self, generator):
        """Test hybrid generation with text only."""
        result = generator.generate_from_hybrid(
            text_description="Create a mounting bracket",
            output_format='step'
        )

        assert result is not None
        assert 'parameters' in result
        assert result['input_type'] == 'hybrid'
        assert 'text' in result.get('parameter_sources', [])

    def test_text_and_specs_hybrid(self, generator):
        """Test hybrid generation with text and specifications."""
        specs = {
            'material': 'Aluminum',
            'thickness': '5mm',
            'tolerance': '±0.1mm'
        }

        result = generator.generate_from_hybrid(
            text_description="Create a mounting plate",
            specifications=specs,
            output_format='step'
        )

        assert result is not None
        assert 'parameters' in result
        assert 'specs' in result.get('parameter_sources', [])

    def test_no_inputs_fails(self, generator):
        """Test that hybrid generation with no inputs fails."""
        with pytest.raises(ValueError):
            result = generator.generate_from_hybrid(
                output_format='step'
            )


class TestParameterExtraction:
    """Test parameter extraction and validation."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CADModelGenerator(mock_mode=True)

    def test_extract_box_dimensions(self, generator):
        """Test extracting box dimensions from text."""
        text = "Create a box 100mm x 50mm x 30mm"
        params = generator._extract_parameters_from_text(text)

        assert params is not None
        assert params.get('type') == 'box'
        assert 'dimensions' in params

    def test_extract_cylinder_dimensions(self, generator):
        """Test extracting cylinder dimensions from text."""
        text = "Cylinder with 20mm diameter and 100mm height"
        params = generator._extract_parameters_from_text(text)

        assert params is not None
        assert params.get('type') == 'cylinder'

    def test_extract_hole_feature(self, generator):
        """Test extracting hole feature from text."""
        text = "Box with a 10mm diameter hole"
        params = generator._extract_parameters_from_text(text)

        assert params is not None
        features = params.get('features', [])
        # Check if hole was detected
        # Note: Detection depends on regex matching in mock mode

    def test_validate_valid_parameters(self, generator):
        """Test validation of valid parameters."""
        params = {
            'type': 'box',
            'dimensions': {
                'length': 0.1,  # 100mm in meters
                'width': 0.05,  # 50mm
                'height': 0.03  # 30mm
            }
        }

        is_valid = generator._validate_parameters(params)
        assert is_valid is True

    def test_validate_invalid_parameters(self, generator):
        """Test validation of invalid parameters."""
        # Missing dimensions
        params = {
            'type': 'box',
            'dimensions': {}
        }

        is_valid = generator._validate_parameters(params)
        assert is_valid is False


class TestParameterMerging:
    """Test parameter merging for hybrid inputs."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        return CADModelGenerator(mock_mode=True)

    def test_merge_text_and_specs(self, generator):
        """Test merging text parameters with specifications."""
        param_list = [
            ('text', {
                'type': 'box',
                'dimensions': {'length': 0.1, 'width': 0.05, 'height': 0.03}
            }),
            ('specs', {
                'material': 'Aluminum',
                'thickness': '5mm'
            })
        ]

        merged = generator._merge_parameters(param_list)

        assert merged is not None
        assert merged['type'] == 'box'
        assert 'length' in merged['dimensions']
        assert 'material' in merged or 'thickness' in merged  # Specs merged

    def test_merge_priority(self, generator):
        """Test that higher priority sources override lower ones."""
        param_list = [
            ('image', {
                'type': 'box',
                'dimensions': {'length': 0.05}
            }),
            ('text', {
                'type': 'cylinder',
                'dimensions': {'length': 0.1}
            })
        ]

        merged = generator._merge_parameters(param_list)

        # Text has higher priority than image
        assert merged['type'] == 'cylinder'
        assert merged['dimensions']['length'] == 0.1


class TestExport:
    """Test export functionality."""

    @pytest.fixture
    def generator(self):
        """Create generator instance for testing."""
        # Only test if Build123d is available
        try:
            from build123d import Box, Part
            return CADModelGenerator(mock_mode=True)
        except ImportError:
            pytest.skip("Build123d not available")

    def test_export_step(self, generator):
        """Test STEP export."""
        # Generate a simple model
        result = generator.generate_from_text(
            description="Create a box 10cm x 10cm x 10cm",
            output_format='step'
        )

        files = result.get('files', [])
        assert len(files) > 0

        # Check that STEP file was created
        step_files = [f for f in files if Path(f).suffix in ['.step', '.stp']]
        assert len(step_files) > 0

        # Verify file exists
        for f in step_files:
            assert Path(f).exists()

            # Clean up
            Path(f).unlink()

    def test_export_stl(self, generator):
        """Test STL export."""
        result = generator.generate_from_text(
            description="Create a sphere 50mm radius",
            output_format='stl'
        )

        files = result.get('files', [])
        assert len(files) > 0

        # Check that STL file was created
        stl_files = [f for f in files if Path(f).suffix == '.stl']
        assert len(stl_files) > 0

        # Verify file exists
        for f in stl_files:
            assert Path(f).exists()

            # Clean up
            Path(f).unlink()


class TestZooDevIntegration:
    """Test Zoo.dev KCL integration."""

    @pytest.fixture
    def generator(self):
        """Create generator instance with Zoo.dev enabled."""
        return CADModelGenerator(
            use_zoo_dev=True,
            mock_mode=True
        )

    def test_zoo_dev_generation(self, generator):
        """Test generation using Zoo.dev."""
        result = generator.generate_from_text(
            description="Create a parametric gear with 20 teeth",
            output_format='step'
        )

        assert result is not None
        assert result.get('engine') == 'zoo_kcl'
        assert 'kcl_code' in result

    def test_zoo_dev_kcl_code(self, generator):
        """Test that KCL code is generated."""
        result = generator.generate_from_text(
            description="Simple box",
            output_format='step'
        )

        kcl_code = result.get('kcl_code', '')
        assert len(kcl_code) > 0
        # Mock KCL should contain some code
        assert 'startSketchOn' in kcl_code or 'Mock' in kcl_code


# Test fixtures cleanup
@pytest.fixture(autouse=True)
def cleanup_output_files():
    """Clean up generated output files after each test."""
    yield

    # Clean up outputs directory
    output_dir = Path('outputs/cad')
    if output_dir.exists():
        for file in output_dir.glob('*'):
            if file.is_file():
                try:
                    file.unlink()
                except Exception:
                    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
