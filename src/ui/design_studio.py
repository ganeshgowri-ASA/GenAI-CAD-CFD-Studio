"""
AI Design Studio - Main UI Module
Complete AI-powered CAD interface with conversational design
"""
import streamlit as st
from typing import Optional, Dict, Any

from src.ui.components.chat_interface import ChatInterface
from src.ui.components.agent_selector import AgentSelector
from src.ui.components.dimension_form import DimensionForm
from src.ui.components.preview_3d import Preview3D
from src.ai.claude_skills import ClaudeSkills


class DesignStudio:
    """
    Main Design Studio application class
    Orchestrates AI-powered CAD design workflow
    """

    def __init__(self):
        """Initialize the Design Studio components"""
        self.chat = ChatInterface(session_key="design_chat")
        self.agent_selector = AgentSelector(session_key="cad_agent")
        self.dimension_form = DimensionForm(session_key="design_params")
        self.preview_3d = Preview3D(session_key="preview_state")
        self.claude_skills = ClaudeSkills()

        # Initialize session state
        if "generated_geometry" not in st.session_state:
            st.session_state.generated_geometry = None
        if "current_params" not in st.session_state:
            st.session_state.current_params = {}

    def render(self):
        """Render the complete Design Studio interface"""
        st.header('🎨 AI Design Studio')

        st.markdown(
            "Create 3D CAD models using natural language. "
            "Describe what you want, and AI will extract dimensions and generate the design."
        )

        st.divider()

        # Main layout: 2 columns (input | preview)
        left_col, right_col = st.columns([1, 1])

        with left_col:
            self._render_input_column()

        with right_col:
            self._render_preview_column()

    def _render_input_column(self):
        """Render the left column with chat and controls"""
        st.subheader("💬 Design Conversation")

        # Render existing chat messages
        self.chat.render_messages()

        # Chat input
        user_prompt = self.chat.render_chat_input(
            placeholder="Describe your design (e.g., 'Create a box 100mm x 50mm x 30mm')"
        )

        # Process user input
        if user_prompt:
            self._process_user_prompt(user_prompt)

        st.divider()

        # Agent selector
        self.agent_selector.render(show_details=True)

        st.divider()

        # Dimension form
        current_params = self.dimension_form.render_form(
            st.session_state.current_params
        )

        # Update current params
        if current_params:
            st.session_state.current_params = current_params

        st.divider()

        # Generation controls
        self._render_generation_controls()

    def _render_preview_column(self):
        """Render the right column with 3D preview and export"""
        # 3D Preview
        self.preview_3d.render_model(st.session_state.generated_geometry)

        st.divider()

        # Export section
        self.preview_3d.render_export_section(st.session_state.generated_geometry)

    def _process_user_prompt(self, prompt: str):
        """
        Process user's natural language prompt

        Args:
            prompt: User's design description
        """
        # Add user message to chat
        self.chat.handle_user_input(prompt)

        # Extract dimensions using Claude skills
        with st.spinner("🤔 Analyzing your design..."):
            extracted_params = self.claude_skills.extract_dimensions(prompt)

        # Generate AI response
        ai_response = self.claude_skills.generate_ai_response(prompt, extracted_params)

        # Add assistant message
        self.chat.add_assistant_message(ai_response, show_typing=True)

        # Update parameters in form
        st.session_state.current_params = extracted_params

        # Rerun to update the UI
        st.rerun()

    def _render_generation_controls(self):
        """Render the generate button and related controls"""
        col1, col2 = st.columns([3, 1])

        with col1:
            generate_button = st.button(
                "🚀 Generate 3D Model",
                use_container_width=True,
                type="primary",
                disabled=not st.session_state.current_params
            )

        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                self._clear_all()
                st.rerun()

        if generate_button:
            self._generate_model()

    def _generate_model(self):
        """Generate the 3D model based on current parameters"""
        params = st.session_state.current_params
        selected_agent = self.agent_selector.get_selected_agent()

        if not params:
            st.error("❌ No parameters to generate. Please describe your design first.")
            return

        with st.spinner(f"⚙️ Generating with {selected_agent}..."):
            # For now, create mock geometry data
            # In production, this would call the actual CAD engines
            geometry_data = self._create_mock_geometry(params)

        # Store the generated geometry
        st.session_state.generated_geometry = geometry_data

        # Add success message to chat
        agent_info = self.agent_selector.get_agent_info(selected_agent)
        success_message = (
            f"✅ **Model generated successfully!**\n\n"
            f"Engine: {agent_info['name']} {agent_info['icon']}\n"
            f"Object: {params.get('object_type', 'custom').title()}\n"
            f"Check the preview on the right →"
        )
        self.chat.add_assistant_message(success_message)

        st.rerun()

    def _create_mock_geometry(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create mock geometry data for preview
        In production, this would call actual CAD engines

        Args:
            params: Design parameters

        Returns:
            Geometry data dictionary
        """
        # Extract dimensions
        geometry = {
            "object_type": params.get("object_type", "box"),
            "unit": params.get("unit", "mm"),
        }

        # Add relevant dimensions
        for key in ["length", "width", "height", "radius", "diameter", "thickness"]:
            if key in params:
                geometry[key] = params[key]

        # Calculate volume (simplified)
        if "length" in geometry and "width" in geometry and "height" in geometry:
            geometry["volume"] = geometry["length"] * geometry["width"] * geometry["height"]
        elif "radius" in geometry and "height" in geometry:
            geometry["volume"] = 3.14159 * (geometry["radius"] ** 2) * geometry["height"]

        return geometry

    def _clear_all(self):
        """Clear all data and reset the studio"""
        self.chat.clear_messages()
        self.dimension_form.clear_parameters()
        st.session_state.generated_geometry = None
        st.session_state.current_params = {}


def main():
    """Main entry point for the Design Studio"""
    # Configure Streamlit page
    st.set_page_config(
        page_title="AI Design Studio",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Apply custom CSS for better appearance
    st.markdown("""
        <style>
        .stButton>button {
            border-radius: 8px;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize and render the studio
    studio = DesignStudio()
    studio.render()


if __name__ == "__main__":
    main()
