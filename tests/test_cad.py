"""
Comprehensive tests for CAD generation engines.

Tests cover:
- Build123D engine operations
- Zoo.dev connector (mock mode)
- Adam.new connector (mock mode)
- Unified interface and auto-selection
- Geometry validation
- Integration tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cad import (
    Build123DEngine,
    ZooDevConnector,
    AdamNewConnector,
    UnifiedCADInterface,
    CADResult,
    validate_geometry,
    quick_validate,
    validate_with_report,
    suggest_fixes,
    ValidationResult,
    ValidationIssue
)


class TestBuild123DEngine:
    """Tests for Build123D CAD engine."""

    @pytest.fixture
    def engine(self):
        """Create Build123D engine instance."""
        try:
            return Build123DEngine()
        except ImportError:
            pytest.skip("build123d not installed")

    def test_create_box(self, engine):
        """Test creating a box primitive."""
        params = {
            'type': 'box',
            'length': 10,
            'width': 5,
            'height': 3
        }
        part = engine.generate_from_params(params)
        assert part is not None
        # Volume should be approximately length * width * height
        assert abs(part.volume - 150) < 0.1

    def test_create_box_short_params(self, engine):
        """Test creating a box with short parameter names."""
        params = {
            'type': 'box',
            'l': 10,
            'w': 5,
            'h': 3
        }
        part = engine.generate_from_params(params)
        assert part is not None
        assert abs(part.volume - 150) < 0.1

    def test_create_cylinder(self, engine):
        """Test creating a cylinder primitive."""
        params = {
            'type': 'cylinder',
            'radius': 5,
            'height': 10
        }
        part = engine.generate_from_params(params)
        assert part is not None
        # Volume should be approximately pi * r^2 * h
        import math
        expected_volume = math.pi * 5**2 * 10
        assert abs(part.volume - expected_volume) < 1.0

    def test_create_sphere(self, engine):
        """Test creating a sphere primitive."""
        params = {
            'type': 'sphere',
            'radius': 5
        }
        part = engine.generate_from_params(params)
        assert part is not None
        # Volume should be approximately 4/3 * pi * r^3
        import math
        expected_volume = 4/3 * math.pi * 5**3
        assert abs(part.volume - expected_volume) < 10.0

    def test_create_cone(self, engine):
        """Test creating a cone primitive."""
        params = {
            'type': 'cone',
            'bottom_radius': 5,
            'top_radius': 0,
            'height': 10
        }
        part = engine.generate_from_params(params)
        assert part is not None
        assert part.volume > 0

    def test_union_operation(self, engine):
        """Test boolean union operation."""
        box1 = engine.generate_from_params({
            'type': 'box',
            'length': 10,
            'width': 10,
            'height': 10
        })
        box2 = engine.generate_from_params({
            'type': 'box',
            'length': 5,
            'width': 5,
            'height': 5
        })

        result = engine.union(box1, box2)
        assert result is not None
        # Volume should be at least as large as the larger box
        assert result.volume >= 1000

    def test_subtract_operation(self, engine):
        """Test boolean subtraction operation."""
        box = engine.generate_from_params({
            'type': 'box',
            'length': 10,
            'width': 10,
            'height': 10
        })
        cylinder = engine.generate_from_params({
            'type': 'cylinder',
            'radius': 3,
            'height': 12
        })

        result = engine.subtract(box, cylinder)
        assert result is not None
        # Volume should be less than original box
        assert result.volume < 1000

    def test_intersect_operation(self, engine):
        """Test boolean intersection operation."""
        box = engine.generate_from_params({
            'type': 'box',
            'length': 10,
            'width': 10,
            'height': 10
        })
        sphere = engine.generate_from_params({
            'type': 'sphere',
            'radius': 7
        })

        result = engine.intersect(box, sphere)
        assert result is not None
        assert result.volume > 0
        # Volume should be less than both original shapes
        assert result.volume < 1000

    def test_export_step(self, engine):
        """Test STEP export."""
        part = engine.generate_from_params({
            'type': 'box',
            'length': 10,
            'width': 10,
            'height': 10
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.step"
            engine.export_step(part, str(filepath))
            assert filepath.exists()
            assert filepath.stat().st_size > 0

    def test_export_stl(self, engine):
        """Test STL export with different resolutions."""
        part = engine.generate_from_params({
            'type': 'sphere',
            'radius': 5
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            for resolution in ['low', 'medium', 'high']:
                filepath = Path(tmpdir) / f"test_{resolution}.stl"
                engine.export_stl(part, str(filepath), resolution=resolution)
                assert filepath.exists()
                assert filepath.stat().st_size > 0

    def test_create_composite(self, engine):
        """Test creating composite parts."""
        operations = [
            {
                'type': 'primitive',
                'params': {'type': 'box', 'length': 10, 'width': 10, 'height': 10}
            },
            {
                'type': 'subtract',
                'params': {'type': 'cylinder', 'radius': 3, 'height': 12}
            }
        ]

        result = engine.create_composite(operations)
        assert result is not None
        assert result.volume > 0
        assert result.volume < 1000  # Less than original box

    def test_invalid_shape_type(self, engine):
        """Test error handling for invalid shape type."""
        with pytest.raises(ValueError):
            engine.generate_from_params({'type': 'invalid_shape'})

    def test_union_no_parts(self, engine):
        """Test union with no parts raises error."""
        with pytest.raises(ValueError):
            engine.union()

    def test_subtract_no_tools(self, engine):
        """Test subtract with no tools raises error."""
        box = engine.generate_from_params({'type': 'box'})
        with pytest.raises(ValueError):
            engine.subtract(box)

    def test_intersect_one_part(self, engine):
        """Test intersect with only one part raises error."""
        box = engine.generate_from_params({'type': 'box'})
        with pytest.raises(ValueError):
            engine.intersect(box)


class TestZooDevConnector:
    """Tests for Zoo.dev connector in mock mode."""

    @pytest.fixture
    def connector(self):
        """Create Zoo connector in mock mode."""
        return ZooDevConnector(mock_mode=True)

    def test_initialization(self):
        """Test connector initialization."""
        connector = ZooDevConnector(mock_mode=True)
        assert connector.mock_mode is True
        assert connector.api_key is None

    def test_initialization_with_api_key(self):
        """Test connector initialization with API key."""
        connector = ZooDevConnector(api_key="test_key", mock_mode=False)
        assert connector.api_key == "test_key"
        assert connector.mock_mode is False

    def test_initialization_without_api_key_non_mock(self):
        """Test that initialization fails without API key in non-mock mode."""
        with pytest.raises(ValueError):
            ZooDevConnector(mock_mode=False)

    def test_generate_kcl(self, connector):
        """Test KCL code generation."""
        prompt = "Create a box with length 10, width 5, height 3"
        kcl_code = connector.generate_kcl(prompt)

        assert kcl_code is not None
        assert len(kcl_code) > 0
        assert 'Mock KCL' in kcl_code

    def test_generate_kcl_empty_prompt(self, connector):
        """Test KCL generation with empty prompt raises error."""
        with pytest.raises(ValueError):
            connector.generate_kcl("")

    def test_execute_kcl(self, connector):
        """Test KCL code execution."""
        kcl_code = "const box = startSketchOn('XY')"
        model_url = connector.execute_kcl(kcl_code)

        assert model_url is not None
        assert len(model_url) > 0
        assert model_url.startswith('https://')

    def test_execute_kcl_empty_code(self, connector):
        """Test KCL execution with empty code raises error."""
        with pytest.raises(ValueError):
            connector.execute_kcl("")

    def test_download_model(self, connector):
        """Test model download."""
        url = "https://example.com/model.glb"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.glb"
            connector.download_model(url, str(output_path))

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_generate_model_complete_workflow(self, connector):
        """Test complete workflow: generate, execute, download."""
        prompt = "Create a cylinder"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.glb"
            result = connector.generate_model(prompt, str(output_path))

            assert 'kcl_code' in result
            assert 'model_url' in result
            assert 'local_path' in result
            assert output_path.exists()

    def test_generate_model_without_download(self, connector):
        """Test generate model without download."""
        prompt = "Create a sphere"
        result = connector.generate_model(prompt)

        assert 'kcl_code' in result
        assert 'model_url' in result
        assert 'local_path' not in result

    def test_context_manager(self):
        """Test connector as context manager."""
        with ZooDevConnector(mock_mode=True) as connector:
            kcl = connector.generate_kcl("test")
            assert kcl is not None


class TestAdamNewConnector:
    """Tests for Adam.new connector in mock mode."""

    @pytest.fixture
    def connector(self):
        """Create Adam connector in mock mode."""
        return AdamNewConnector(mock_mode=True)

    def test_initialization(self):
        """Test connector initialization."""
        connector = AdamNewConnector(mock_mode=True)
        assert connector.mock_mode is True
        assert connector.api_key is None
        assert len(connector.conversation.messages) == 0

    def test_initialization_with_api_key(self):
        """Test connector initialization with API key."""
        connector = AdamNewConnector(api_key="test_key", mock_mode=False)
        assert connector.api_key == "test_key"
        assert connector.mock_mode is False

    def test_initialization_without_api_key_non_mock(self):
        """Test that initialization fails without API key in non-mock mode."""
        with pytest.raises(ValueError):
            AdamNewConnector(mock_mode=False)

    def test_generate_from_nl(self, connector):
        """Test natural language generation."""
        prompt = "Create a box 10x10x10"
        result = connector.generate_from_nl(prompt)

        assert 'model_id' in result
        assert 'status' in result
        assert result['status'] == 'completed'
        assert len(connector.conversation.messages) == 2  # User + assistant

    def test_generate_from_nl_empty_prompt(self, connector):
        """Test generation with empty prompt raises error."""
        with pytest.raises(ValueError):
            connector.generate_from_nl("")

    def test_refine_model(self, connector):
        """Test model refinement."""
        # First generate a model
        result1 = connector.generate_from_nl("Create a box")
        model_id = result1['model_id']

        # Then refine it
        result2 = connector.refine_model(model_id, "Make it taller")

        assert 'model_id' in result2
        assert result2['model_id'] != model_id  # Should be a new model
        assert len(connector.conversation.messages) == 4  # 2 exchanges

    def test_refine_model_empty_feedback(self, connector):
        """Test refinement with empty feedback raises error."""
        with pytest.raises(ValueError):
            connector.refine_model("model_id", "")

    def test_refine_model_empty_id(self, connector):
        """Test refinement with empty model_id raises error."""
        with pytest.raises(ValueError):
            connector.refine_model("", "feedback")

    def test_download_formats(self, connector):
        """Test downloading multiple formats."""
        result = connector.generate_from_nl("Create a sphere")
        model_id = result['model_id']

        formats = ['step', 'stl', 'obj']

        with tempfile.TemporaryDirectory() as tmpdir:
            downloaded = connector.download_formats(model_id, formats, tmpdir)

            assert len(downloaded) == len(formats)
            for fmt in formats:
                assert fmt in downloaded
                assert Path(downloaded[fmt]).exists()

    def test_download_formats_empty_list(self, connector):
        """Test download with empty formats list raises error."""
        with pytest.raises(ValueError):
            connector.download_formats("model_id", [])

    def test_get_model_info(self, connector):
        """Test getting model information."""
        result = connector.generate_from_nl("Create a box")
        model_id = result['model_id']

        info = connector.get_model_info(model_id)

        assert 'model_id' in info
        assert 'status' in info
        assert 'formats_available' in info

    def test_clear_conversation(self, connector):
        """Test clearing conversation history."""
        connector.generate_from_nl("Create a box")
        assert len(connector.conversation.messages) > 0

        connector.clear_conversation()
        assert len(connector.conversation.messages) == 0

    def test_context_manager(self):
        """Test connector as context manager."""
        with AdamNewConnector(mock_mode=True) as connector:
            result = connector.generate_from_nl("test")
            assert result is not None


class TestUnifiedCADInterface:
    """Tests for unified CAD interface."""

    @pytest.fixture
    def interface(self):
        """Create unified interface in mock mode."""
        return UnifiedCADInterface(mock_mode=True)

    def test_initialization(self, interface):
        """Test interface initialization."""
        assert interface.mock_mode is True

    def test_auto_select_engine_parametric(self, interface):
        """Test auto-selection for parametric prompts."""
        prompts = [
            "box length 10 width 5 height 3",
            "cylinder radius 5 height 10",
            "sphere radius 7"
        ]

        for prompt in prompts:
            engine = interface.auto_select_engine(prompt)
            assert engine == 'build123d'

    def test_auto_select_engine_kcl(self, interface):
        """Test auto-selection for KCL-related prompts."""
        prompts = [
            "generate KCL code for a box",
            "create a parametric sketch",
        ]

        for prompt in prompts:
            engine = interface.auto_select_engine(prompt)
            assert engine == 'zoo'

    def test_auto_select_engine_conversational(self, interface):
        """Test auto-selection for conversational prompts."""
        prompts = [
            "I need a design for a solar panel mount",
            "Create a custom bracket for mounting equipment",
        ]

        for prompt in prompts:
            engine = interface.auto_select_engine(prompt)
            assert engine == 'adam'

    def test_auto_select_engine_structured_data(self, interface):
        """Test auto-selection for structured JSON data."""
        prompt = '{"type": "box", "length": 10, "width": 5, "height": 3}'
        engine = interface.auto_select_engine(prompt)
        assert engine == 'build123d'

    def test_generate_auto_selection(self, interface):
        """Test generation with auto engine selection."""
        result = interface.generate(
            "Create a design for a mounting bracket",
            engine='auto'
        )

        assert isinstance(result, CADResult)
        assert result.engine in ['build123d', 'zoo', 'adam']
        assert result.prompt is not None

    def test_generate_zoo_explicit(self, interface):
        """Test generation with explicit Zoo engine."""
        result = interface.generate(
            "Create a box",
            engine='zoo'
        )

        assert isinstance(result, CADResult)
        assert result.engine == 'zoo'
        assert 'kcl_code' in result.metadata

    def test_generate_adam_explicit(self, interface):
        """Test generation with explicit Adam engine."""
        result = interface.generate(
            "Create a sphere",
            engine='adam'
        )

        assert isinstance(result, CADResult)
        assert result.engine == 'adam'
        assert 'model_id' in result.metadata

    def test_generate_invalid_engine(self, interface):
        """Test generation with invalid engine raises error."""
        with pytest.raises(ValueError):
            interface.generate("test", engine='invalid')

    def test_refine(self, interface):
        """Test model refinement."""
        # Generate initial model
        result1 = interface.generate("Create a box", engine='adam')
        model_id = result1.metadata['model_id']

        # Refine it
        result2 = interface.refine(model_id, "Make it bigger")

        assert isinstance(result2, CADResult)
        assert result2.engine == 'adam'
        assert result2.metadata['parent_model'] == model_id

    def test_refine_non_adam_engine(self, interface):
        """Test refinement with non-Adam engine raises error."""
        with pytest.raises(ValueError):
            interface.refine("model_id", "feedback", engine='zoo')


class TestCADResult:
    """Tests for CADResult class."""

    def test_cad_result_creation(self):
        """Test CAD result creation."""
        result = CADResult(
            engine='adam',
            model='model_123',
            metadata={'status': 'completed'},
            prompt='Create a box'
        )

        assert result.engine == 'adam'
        assert result.model == 'model_123'
        assert result.prompt == 'Create a box'

    def test_get_metadata_all(self):
        """Test getting all metadata."""
        metadata = {'key1': 'value1', 'key2': 'value2'}
        result = CADResult(engine='test', model='test', metadata=metadata)

        all_meta = result.get_metadata()
        assert all_meta == metadata
        assert all_meta is not metadata  # Should be a copy

    def test_get_metadata_specific_key(self):
        """Test getting specific metadata key."""
        metadata = {'key1': 'value1', 'key2': 'value2'}
        result = CADResult(engine='test', model='test', metadata=metadata)

        assert result.get_metadata('key1') == 'value1'
        assert result.get_metadata('key2') == 'value2'
        assert result.get_metadata('nonexistent') is None


class TestValidation:
    """Tests for CAD validation."""

    @pytest.fixture
    def engine(self):
        """Create Build123D engine instance."""
        try:
            return Build123DEngine()
        except ImportError:
            pytest.skip("build123d not installed")

    def test_validate_valid_box(self, engine):
        """Test validation of valid box."""
        part = engine.generate_from_params({
            'type': 'box',
            'length': 10,
            'width': 10,
            'height': 10
        })

        result = validate_geometry(part)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert 'volume' in result.metrics
        assert result.metrics['volume'] > 0

    def test_validate_valid_cylinder(self, engine):
        """Test validation of valid cylinder."""
        part = engine.generate_from_params({
            'type': 'cylinder',
            'radius': 5,
            'height': 10
        })

        result = validate_geometry(part)
        assert result.is_valid is True
        assert not result.has_errors()

    def test_validation_result_methods(self):
        """Test ValidationResult methods."""
        result = ValidationResult(is_valid=True)

        # Add some issues
        result.issues.append(ValidationIssue(
            severity='error',
            issue_type='test_error',
            message='Test error message'
        ))
        result.issues.append(ValidationIssue(
            severity='warning',
            issue_type='test_warning',
            message='Test warning message'
        ))

        assert result.has_errors()
        assert result.has_warnings()
        assert len(result.get_errors()) == 1
        assert len(result.get_warnings()) == 1

    def test_validation_summary(self, engine):
        """Test validation summary generation."""
        part = engine.generate_from_params({'type': 'box'})
        result = validate_geometry(part)

        summary = result.summary()
        assert isinstance(summary, str)
        assert 'Validation Status' in summary
        assert 'Errors:' in summary
        assert 'Warnings:' in summary

    def test_quick_validate(self, engine):
        """Test quick validation."""
        part = engine.generate_from_params({'type': 'sphere', 'radius': 5})
        is_valid = quick_validate(part)

        assert isinstance(is_valid, bool)
        assert is_valid is True

    def test_validate_with_report(self, engine):
        """Test validation with report."""
        part = engine.generate_from_params({'type': 'cone'})
        report = validate_with_report(part)

        assert isinstance(report, str)
        assert len(report) > 0

    def test_suggest_fixes(self):
        """Test fix suggestions."""
        result = ValidationResult(is_valid=False)
        result.issues.append(ValidationIssue(
            severity='error',
            issue_type='invalid_volume',
            message='Volume is negative',
            suggestion='Check face orientations'
        ))

        fixes = suggest_fixes(result)
        assert isinstance(fixes, list)
        assert len(fixes) > 0


class TestIntegration:
    """Integration tests - complete workflows."""

    def test_prompt_to_step_file_zoo(self):
        """Test complete workflow: prompt -> STEP file (Zoo)."""
        interface = UnifiedCADInterface(mock_mode=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.step"

            # Generate using Zoo
            result = interface.generate(
                "Create a mounting bracket",
                engine='zoo',
                output_path=str(output_path)
            )

            assert result.engine == 'zoo'
            assert output_path.exists()

    def test_prompt_to_step_file_adam(self):
        """Test complete workflow: prompt -> STEP file (Adam)."""
        interface = UnifiedCADInterface(mock_mode=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.step"

            # Generate using Adam
            result = interface.generate(
                "Design a custom bracket",
                engine='adam',
                output_path=str(output_path),
                formats=['step']
            )

            assert result.engine == 'adam'
            # Adam downloads to directory, check metadata
            assert 'local_files' in result.metadata

    def test_iterative_refinement(self):
        """Test iterative model refinement."""
        interface = UnifiedCADInterface(mock_mode=True)

        # Initial generation
        result1 = interface.generate("Create a box", engine='adam')
        model_id = result1.metadata['model_id']

        # First refinement
        result2 = interface.refine(model_id, "Make it taller")
        model_id2 = result2.metadata['model_id']

        # Second refinement
        result3 = interface.refine(model_id2, "Add rounded corners")

        assert result3.metadata['parent_model'] == model_id2

    def test_multi_format_export(self):
        """Test exporting to multiple formats."""
        interface = UnifiedCADInterface(mock_mode=True)

        result = interface.generate(
            "Create a sphere",
            engine='adam',
            formats=['step', 'stl', 'obj']
        )

        # In mock mode, formats should be in metadata
        assert 'formats_available' in result.metadata
        assert 'step' in result.metadata['formats_available']


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=cad', '--cov-report=term-missing'])
