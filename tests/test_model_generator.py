"""
Unit tests for CAD Model Generator
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import json

try:
    from src.cad.model_generator import CADModelGenerator, CADGenerationResult
    from src.ai.dimension_extractor import DimensionExtractor
    from src.ai.claude_skills import ClaudeSkills
    HAS_IMPORTS = True
except ImportError:
    HAS_IMPORTS = False


@unittest.skipIf(not HAS_IMPORTS, "Required modules not available")
class TestCADGenerationResult(unittest.TestCase):
    """Test CADGenerationResult class."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = CADGenerationResult(
            success=True,
            message="Generation successful",
            parameters={'type': 'box', 'length': 100}
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Generation successful")
        self.assertEqual(result.parameters['type'], 'box')
        self.assertIsNotNone(result.timestamp)

    def test_failed_result(self):
        """Test creating a failed result."""
        result = CADGenerationResult(
            success=False,
            message="Generation failed: Invalid parameters"
        )

        self.assertFalse(result.success)
        self.assertEqual(result.message, "Generation failed: Invalid parameters")

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = CADGenerationResult(
            success=True,
            message="Test",
            parameters={'test': 123},
            export_paths={'step': '/path/to/file.step'}
        )

        result_dict = result.to_dict()

        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict['success'])
        self.assertEqual(result_dict['parameters']['test'], 123)
        self.assertEqual(result_dict['export_paths']['step'], '/path/to/file.step')

    def test_repr(self):
        """Test string representation."""
        result = CADGenerationResult(success=True, message="Success!")
        self.assertIn("SUCCESS", repr(result))

        result = CADGenerationResult(success=False, message="Failed!")
        self.assertIn("FAILED", repr(result))


@unittest.skipIf(not HAS_IMPORTS, "Required modules not available")
class TestCADModelGenerator(unittest.TestCase):
    """Test CADModelGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"

        # Initialize generator without API keys (will use fallback methods)
        self.generator = CADModelGenerator(
            claude_api_key=None,
            zoo_api_key=None,
            default_engine='build123d',
            default_unit='mm',
            output_dir=str(self.output_dir)
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.default_engine, 'build123d')
        self.assertEqual(self.generator.default_unit, 'mm')
        self.assertTrue(self.output_dir.exists())

    def test_initialization_with_api_keys(self):
        """Test initialization with API keys."""
        # Use mock keys - won't actually connect
        generator = CADModelGenerator(
            claude_api_key="test_key",
            zoo_api_key="test_zoo_key",
            output_dir=str(self.output_dir)
        )

        self.assertEqual(generator.claude_api_key, "test_key")
        self.assertEqual(generator.zoo_api_key, "test_zoo_key")

    def test_parameter_extraction_from_text(self):
        """Test parameter extraction from text."""
        description = "Create a box 100mm x 50mm x 30mm"

        params = self.generator._extract_parameters_from_text(description)

        self.assertIsInstance(params, dict)
        self.assertIn('object_type', params)
        # Should extract dimensions
        self.assertTrue(
            'length' in params or 'width' in params or 'height' in params,
            "Should extract at least one dimension"
        )

    def test_parameter_extraction_cylinder(self):
        """Test parameter extraction for cylinder."""
        description = "Make a cylinder with radius 25mm and height 100mm"

        params = self.generator._extract_parameters_from_text(description)

        self.assertEqual(params.get('object_type'), 'cylinder')
        self.assertIn('radius', params)
        self.assertIn('height', params)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        valid_params = {
            'type': 'box',
            'length': 100,
            'width': 50,
            'height': 30
        }

        validated = self.generator._validate_parameters(valid_params)
        self.assertIsInstance(validated, dict)
        self.assertEqual(validated['type'], 'box')

        # Missing type
        no_type = {'length': 100}
        validated = self.generator._validate_parameters(no_type)
        self.assertIn('type', validated)  # Should add default type

    def test_engine_selection(self):
        """Test engine selection logic."""
        # Standard primitive should select build123d
        params = {'type': 'box', 'length': 100}
        engine = self.generator._select_engine(params)
        self.assertEqual(engine, 'build123d')

        # Custom type might select zoo if available
        params = {'type': 'custom', 'description': 'complex part'}
        engine = self.generator._select_engine(params)
        self.assertIn(engine, ['zoo', 'build123d'])

    def test_create_prompt_from_params(self):
        """Test prompt creation from parameters."""
        params = {
            'type': 'box',
            'length': 100,
            'width': 50,
            'height': 30
        }

        prompt = self.generator._create_prompt_from_params(params, None)

        self.assertIsInstance(prompt, str)
        self.assertIn('box', prompt.lower())
        self.assertIn('100', prompt)

        # Test with existing description
        prompt = self.generator._create_prompt_from_params(params, "Create a box")
        self.assertEqual(prompt, "Create a box")

    def test_create_prompt_cylinder(self):
        """Test prompt creation for cylinder."""
        params = {
            'type': 'cylinder',
            'radius': 25,
            'height': 100
        }

        prompt = self.generator._create_prompt_from_params(params, None)

        self.assertIn('cylinder', prompt.lower())
        self.assertIn('25', prompt)
        self.assertIn('100', prompt)

    def test_generate_from_text_no_build123d(self):
        """Test text generation when build123d is not available."""
        # Create generator without build123d
        generator = CADModelGenerator(
            claude_api_key=None,
            zoo_api_key=None,
            default_engine='build123d',
            output_dir=str(self.output_dir)
        )

        # If build123d is not available, should fail gracefully
        if not generator.build123d_engine:
            result = generator.generate_from_text("Create a box 100x100x100")
            self.assertFalse(result.success)
            self.assertIn("not available", result.message.lower())

    def test_validate_parameters_with_extractor(self):
        """Test parameter validation using dimension extractor."""
        params = {
            'length': 0.1,  # 100mm in meters
            'width': 0.05,
            'height': 0.03
        }

        validated = self.generator._validate_parameters(params)
        self.assertIsInstance(validated, dict)

    def test_output_directory_creation(self):
        """Test that output directory is created."""
        test_output = Path(self.temp_dir) / "test_output"

        generator = CADModelGenerator(output_dir=str(test_output))

        self.assertTrue(test_output.exists())
        self.assertTrue(test_output.is_dir())


@unittest.skipIf(not HAS_IMPORTS, "Required modules not available")
class TestParameterExtraction(unittest.TestCase):
    """Test parameter extraction functionality."""

    def test_dimension_extractor(self):
        """Test DimensionExtractor."""
        extractor = DimensionExtractor()

        # Test simple format
        text = "10cm x 5cm x 3cm"
        dims = extractor.parse_dimensions(text)

        self.assertIn('length', dims)
        self.assertIn('width', dims)
        self.assertIn('height', dims)

        # Should convert to meters
        self.assertAlmostEqual(dims['length'], 0.1, places=5)
        self.assertAlmostEqual(dims['width'], 0.05, places=5)
        self.assertAlmostEqual(dims['height'], 0.03, places=5)

    def test_dimension_extractor_labeled(self):
        """Test labeled dimension extraction."""
        extractor = DimensionExtractor()

        text = "length: 100mm, width: 50mm, height: 30mm"
        dims = extractor.parse_dimensions(text)

        self.assertIn('length', dims)
        self.assertIn('width', dims)
        self.assertIn('height', dims)

    def test_dimension_validation(self):
        """Test dimension validation."""
        extractor = DimensionExtractor()

        # Valid dimensions
        valid_dims = {'length': 0.1, 'width': 0.05, 'height': 0.03}
        self.assertTrue(extractor.validate_dimensions(valid_dims))

        # Invalid: negative
        invalid_dims = {'length': -0.1}
        self.assertFalse(extractor.validate_dimensions(invalid_dims))

        # Invalid: too large
        invalid_dims = {'length': 1000}
        self.assertFalse(extractor.validate_dimensions(invalid_dims))

    def test_dimension_suggestions(self):
        """Test dimension correction suggestions."""
        extractor = DimensionExtractor()

        # Very large value (likely wrong unit)
        dims = {'length': 1000}  # 1000 meters
        suggestions = extractor.suggest_corrections(dims)

        self.assertIsInstance(suggestions, list)
        self.assertTrue(len(suggestions) > 0)

    def test_claude_skills(self):
        """Test ClaudeSkills extraction."""
        skills = ClaudeSkills()

        text = "Create a box 100mm x 50mm x 30mm"
        params = skills.extract_dimensions(text)

        self.assertIn('object_type', params)
        self.assertIn('unit', params)
        self.assertEqual(params['object_type'], 'box')

    def test_claude_skills_cylinder(self):
        """Test ClaudeSkills for cylinder."""
        skills = ClaudeSkills()

        text = "Make a cylinder 50mm diameter, 100mm tall"
        params = skills.extract_dimensions(text)

        self.assertEqual(params['object_type'], 'cylinder')
        self.assertIn('radius', params)
        self.assertIn('height', params)

    def test_claude_skills_default_dimensions(self):
        """Test default dimensions when none specified."""
        skills = ClaudeSkills()

        text = "Create a box"
        params = skills.extract_dimensions(text)

        # Should have default dimensions
        self.assertIn('length', params)
        self.assertIn('width', params)
        self.assertIn('height', params)


@unittest.skipIf(not HAS_IMPORTS, "Required modules not available")
class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"

    def tearDown(self):
        """Clean up test fixtures."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_text_to_parameters_pipeline(self):
        """Test full pipeline from text to parameters."""
        generator = CADModelGenerator(output_dir=str(self.output_dir))

        description = "Create a cylindrical container 100mm diameter and 150mm tall"

        # Extract parameters
        params = generator._extract_parameters_from_text(description)

        # Validate
        validated = generator._validate_parameters(params)

        # Check results
        self.assertIsInstance(validated, dict)
        self.assertIn('object_type', validated)

    def test_parameter_to_prompt_pipeline(self):
        """Test pipeline from parameters to prompt."""
        generator = CADModelGenerator(output_dir=str(self.output_dir))

        params = {
            'type': 'box',
            'length': 100,
            'width': 50,
            'height': 30,
            'unit': 'mm'
        }

        # Create prompt
        prompt = generator._create_prompt_from_params(params, None)

        # Verify prompt
        self.assertIsInstance(prompt, str)
        self.assertTrue(len(prompt) > 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCADGenerationResult))
    suite.addTests(loader.loadTestsFromTestCase(TestCADModelGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
