"""
Tests for Design Studio UI Components
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.components.chat_interface import ChatInterface
from src.ui.components.agent_selector import AgentSelector
from src.ui.components.dimension_form import DimensionForm
from src.ui.components.preview_3d import Preview3D
from src.ai.claude_skills import ClaudeSkills


class TestChatInterface:
    """Tests for ChatInterface component"""

    def test_initialization(self):
        """Test ChatInterface initialization"""
        chat = ChatInterface(session_key="test_chat")
        assert chat.session_key == "test_chat"

    def test_handle_user_input(self):
        """Test handling user input"""
        chat = ChatInterface(session_key="test_chat")

        # Mock session state
        with patch('streamlit.session_state', {"test_chat": []}):
            message = chat.handle_user_input("Create a box")
            assert message["role"] == "user"
            assert message["content"] == "Create a box"
            assert "timestamp" in message

    def test_add_assistant_message(self):
        """Test adding assistant message"""
        chat = ChatInterface(session_key="test_chat")

        with patch('streamlit.session_state', {"test_chat": []}):
            chat.add_assistant_message("I understand", show_typing=True)
            # Would check session state in real test

    def test_get_messages(self):
        """Test retrieving messages"""
        chat = ChatInterface(session_key="test_chat")

        test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        with patch('streamlit.session_state', {"test_chat": test_messages}):
            messages = chat.get_messages()
            assert len(messages) == 2
            assert messages[0]["role"] == "user"


class TestAgentSelector:
    """Tests for AgentSelector component"""

    def test_initialization(self):
        """Test AgentSelector initialization"""
        selector = AgentSelector(session_key="test_agent")
        assert selector.session_key == "test_agent"

    def test_get_available_agents(self):
        """Test getting available agents"""
        agents = AgentSelector.get_available_agents()
        assert "build123d" in agents
        assert "zoo_dev" in agents
        assert "adam_new" in agents

    def test_get_agent_info(self):
        """Test getting agent information"""
        selector = AgentSelector()

        # Mock session state
        with patch('streamlit.session_state', {"selected_agent": "build123d"}):
            info = selector.get_agent_info("build123d")
            assert info["name"] == "Build123d"
            assert info["language"] == "Python"
            assert "capabilities" in info

    def test_agent_capabilities(self):
        """Test that all agents have required fields"""
        agents = AgentSelector.get_available_agents()

        required_fields = ["name", "icon", "description", "capabilities", "best_for", "language", "speed"]

        for agent_key, agent_info in agents.items():
            for field in required_fields:
                assert field in agent_info, f"Agent {agent_key} missing field {field}"


class TestDimensionForm:
    """Tests for DimensionForm component"""

    def test_initialization(self):
        """Test DimensionForm initialization"""
        form = DimensionForm(session_key="test_params")
        assert form.session_key == "test_params"

    def test_validate_parameters(self):
        """Test parameter validation"""
        form = DimensionForm()

        # Valid parameters
        valid_params = {"length": 100, "width": 50, "height": 30}
        warnings = form._validate_parameters(valid_params)
        assert len(warnings) == 0

        # Invalid parameters (zero/negative)
        invalid_params = {"length": -10, "width": 0}
        warnings = form._validate_parameters(invalid_params)
        assert len(warnings) > 0

        # Very large parameters
        large_params = {"length": 15000}
        warnings = form._validate_parameters(large_params)
        assert len(warnings) > 0

    def test_export_parameters(self):
        """Test exporting parameters"""
        form = DimensionForm()

        with patch('streamlit.session_state', {"dimension_params": {"length": 100, "width": 50}}):
            exported = form.export_parameters()
            assert "length: 100" in exported
            assert "width: 50" in exported


class TestPreview3D:
    """Tests for Preview3D component"""

    def test_initialization(self):
        """Test Preview3D initialization"""
        preview = Preview3D(session_key="test_preview")
        assert preview.session_key == "test_preview"

    def test_camera_presets(self):
        """Test camera presets"""
        assert "isometric" in Preview3D.CAMERA_PRESETS
        assert "front" in Preview3D.CAMERA_PRESETS
        assert "top" in Preview3D.CAMERA_PRESETS
        assert "right" in Preview3D.CAMERA_PRESETS

        # Check camera preset structure
        iso = Preview3D.CAMERA_PRESETS["isometric"]
        assert "eye" in iso
        assert "center" in iso
        assert "up" in iso

    def test_view_modes(self):
        """Test view modes"""
        assert "solid" in Preview3D.VIEW_MODES
        assert "wireframe" in Preview3D.VIEW_MODES
        assert "shaded" in Preview3D.VIEW_MODES

    def test_generate_box_mesh(self):
        """Test box mesh generation"""
        vertices, faces = Preview3D._generate_box_mesh(100, 50, 30)

        assert len(vertices) == 8  # A box has 8 vertices
        assert len(faces) == 12  # A box has 12 triangular faces
        assert vertices.shape[1] == 3  # 3D coordinates
        assert faces.shape[1] == 3  # Triangular faces

    def test_format_dimensions(self):
        """Test dimension formatting"""
        preview = Preview3D()

        geometry = {"length": 100, "width": 50, "height": 30}
        formatted = preview._format_dimensions(geometry)
        assert "L: 100" in formatted
        assert "W: 50" in formatted
        assert "H: 30" in formatted


class TestClaudeSkills:
    """Tests for Claude AI Skills"""

    def test_initialization(self):
        """Test ClaudeSkills initialization"""
        skills = ClaudeSkills()
        assert skills is not None

    def test_extract_object_type(self):
        """Test object type extraction"""
        skills = ClaudeSkills()

        assert skills._extract_object_type("create a box") == "box"
        assert skills._extract_object_type("make a cylinder") == "cylinder"
        assert skills._extract_object_type("i need a sphere") == "sphere"
        assert skills._extract_object_type("rectangular block") == "box"

    def test_extract_unit(self):
        """Test unit extraction"""
        skills = ClaudeSkills()

        assert skills._extract_unit("100mm long") == "mm"
        assert skills._extract_unit("5 inches wide") == "inches"
        assert skills._extract_unit("2 meters tall") == "m"
        assert skills._extract_unit("10cm") == "cm"

    def test_extract_numeric_dimensions(self):
        """Test numeric dimension extraction"""
        skills = ClaudeSkills()

        # Test simple format
        dims = skills._extract_numeric_dimensions("length 100mm width 50mm height 30mm")
        assert dims["length"] == 100
        assert dims["width"] == 50
        assert dims["height"] == 30

        # Test "X x Y x Z" format
        dims = skills._extract_numeric_dimensions("create a box 100 x 50 x 30")
        assert dims["length"] == 100
        assert dims["width"] == 50
        assert dims["height"] == 30

        # Test natural language format
        dims = skills._extract_numeric_dimensions("make it 100mm long and 50mm wide")
        assert dims["length"] == 100
        assert dims["width"] == 50

    def test_extract_dimensions_complete(self):
        """Test complete dimension extraction"""
        skills = ClaudeSkills()

        prompt = "Create a box 100mm x 50mm x 30mm"
        params = skills.extract_dimensions(prompt)

        assert params["object_type"] == "box"
        assert params["unit"] == "mm"
        assert params["length"] == 100
        assert params["width"] == 50
        assert params["height"] == 30

    def test_default_dimensions(self):
        """Test default dimensions for objects"""
        skills = ClaudeSkills()

        # When no dimensions provided, should give defaults
        params = skills.extract_dimensions("create a box")
        assert "length" in params
        assert "width" in params
        assert "height" in params

    def test_diameter_to_radius_conversion(self):
        """Test diameter to radius conversion"""
        skills = ClaudeSkills()

        dims = skills._extract_numeric_dimensions("diameter 100mm")
        assert dims["diameter"] == 100
        assert dims["radius"] == 50

    def test_generate_ai_response(self):
        """Test AI response generation"""
        skills = ClaudeSkills()

        params = {
            "object_type": "box",
            "length": 100,
            "width": 50,
            "height": 30,
            "unit": "mm"
        }

        response = skills.generate_ai_response("create a box", params)
        assert "box" in response.lower()
        assert "100" in response
        assert "50" in response
        assert "30" in response

    def test_suggest_improvements(self):
        """Test design improvement suggestions"""
        skills = ClaudeSkills()

        # High aspect ratio
        params = {"length": 1000, "width": 50}
        suggestions = skills.suggest_improvements(params)
        assert len(suggestions) > 0

    def test_get_suggested_materials(self):
        """Test material suggestions"""
        skills = ClaudeSkills()

        materials = skills.get_suggested_materials("box")
        assert len(materials) > 0
        assert isinstance(materials, list)


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
