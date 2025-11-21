"""
Claude Model Selector Component

Allows users to select Claude models (Opus/Sonnet/Haiku) with cost information
and save preferences.
"""

import streamlit as st
from typing import Dict, Any, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# Claude model information
CLAUDE_MODELS = {
    'claude-3-5-sonnet-20241022': {
        'name': 'Claude 3.5 Sonnet',
        'short_name': 'Sonnet 3.5',
        'description': 'Best balance of intelligence and speed',
        'cost_input': 3.00,  # USD per 1M tokens
        'cost_output': 15.00,
        'tier': 'recommended',
        'use_cases': ['General CAD generation', 'Image analysis', 'Complex reasoning'],
        'emoji': '⚡'
    },
    'claude-3-opus-20240229': {
        'name': 'Claude 3 Opus',
        'short_name': 'Opus',
        'description': 'Most capable model for complex tasks',
        'cost_input': 15.00,
        'cost_output': 75.00,
        'tier': 'premium',
        'use_cases': ['Complex assemblies', 'High-precision analysis', 'Advanced reasoning'],
        'emoji': '👑'
    },
    'claude-3-haiku-20240307': {
        'name': 'Claude 3 Haiku',
        'short_name': 'Haiku',
        'description': 'Fastest and most cost-effective',
        'cost_input': 0.25,
        'cost_output': 1.25,
        'tier': 'economy',
        'use_cases': ['Simple parts', 'Quick iterations', 'Batch processing'],
        'emoji': '🚀'
    },
    'claude-3-sonnet-20240229': {
        'name': 'Claude 3 Sonnet',
        'short_name': 'Sonnet 3',
        'description': 'Balanced performance (legacy)',
        'cost_input': 3.00,
        'cost_output': 15.00,
        'tier': 'standard',
        'use_cases': ['General purpose', 'Standard CAD tasks'],
        'emoji': '⭐'
    }
}

# Default model
DEFAULT_MODEL = 'claude-3-5-sonnet-20241022'


class UserPreferences:
    """Manage user preferences for the CAD application."""

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize user preferences.

        Args:
            config_path: Path to preferences file (defaults to ~/.genai_cad_cfd/preferences.json)
        """
        if config_path is None:
            config_path = Path.home() / '.genai_cad_cfd' / 'preferences.json'

        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.preferences = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load preferences from file."""
        if not self.config_path.exists():
            return self._get_defaults()

        try:
            with open(self.config_path, 'r') as f:
                prefs = json.load(f)
            logger.info("User preferences loaded")
            return prefs
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")
            return self._get_defaults()

    def _get_defaults(self) -> Dict[str, Any]:
        """Get default preferences."""
        return {
            'claude_model': DEFAULT_MODEL,
            'export_formats': ['step', 'stl'],
            'default_engine': 'auto',
            'cad_mode': '3d',
            'assembly_mode': 'single_part',
            'theme': 'light',
            'show_advanced_options': False,
            'auto_export_pdf': False,
            'max_tokens': 4096,
            'temperature': 0.1
        }

    def save(self) -> None:
        """Save preferences to file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.preferences, f, indent=2)
            logger.info("User preferences saved")
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get preference value."""
        return self.preferences.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set preference value."""
        self.preferences[key] = value
        self.save()

    def reset(self) -> None:
        """Reset to defaults."""
        self.preferences = self._get_defaults()
        self.save()


def render_model_selector(
    location: str = "sidebar",
    show_costs: bool = True,
    show_descriptions: bool = True
) -> str:
    """
    Render Claude model selector.

    Args:
        location: Where to render ('sidebar' or 'main')
        show_costs: Show cost information
        show_descriptions: Show model descriptions

    Returns:
        Selected model ID
    """
    # Get preferences
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = UserPreferences()

    prefs = st.session_state.user_preferences
    current_model = prefs.get('claude_model', DEFAULT_MODEL)

    if location == "sidebar":
        container = st.sidebar
    else:
        container = st

    container.markdown("### 🤖 Claude Model Selection")

    # Model selector
    model_options = []
    model_mapping = {}

    for model_id, info in CLAUDE_MODELS.items():
        # Create display label
        if show_costs:
            label = (
                f"{info['emoji']} {info['short_name']} - "
                f"${info['cost_input']:.2f}/${info['cost_output']:.2f} per 1M tokens"
            )
        else:
            label = f"{info['emoji']} {info['short_name']}"

        model_options.append(label)
        model_mapping[label] = model_id

    # Find current selection
    current_label = None
    for label, model_id in model_mapping.items():
        if model_id == current_model:
            current_label = label
            break

    if current_label is None:
        current_label = model_options[0]

    selected_label = container.selectbox(
        "Select Model",
        model_options,
        index=model_options.index(current_label) if current_label in model_options else 0,
        help="Choose the Claude model for CAD generation",
        key="model_selector"
    )

    selected_model_id = model_mapping[selected_label]
    model_info = CLAUDE_MODELS[selected_model_id]

    # Show model details
    if show_descriptions:
        with container.expander("ℹ️ Model Details", expanded=False):
            st.markdown(f"**{model_info['name']}**")
            st.markdown(model_info['description'])

            st.markdown("**Use Cases:**")
            for use_case in model_info['use_cases']:
                st.markdown(f"- {use_case}")

            if show_costs:
                st.markdown("**Pricing:**")
                st.markdown(f"- Input: ${model_info['cost_input']:.2f} per 1M tokens")
                st.markdown(f"- Output: ${model_info['cost_output']:.2f} per 1M tokens")

                # Cost calculator
                st.markdown("**Estimate Cost:**")
                num_prompts = st.number_input(
                    "Number of prompts",
                    min_value=1,
                    value=10,
                    step=1,
                    key="cost_calc_prompts"
                )

                avg_tokens_per_prompt = 1500  # Reasonable estimate
                total_input_tokens = num_prompts * avg_tokens_per_prompt
                total_output_tokens = num_prompts * 500  # Output estimate

                estimated_cost = (
                    (total_input_tokens / 1_000_000) * model_info['cost_input'] +
                    (total_output_tokens / 1_000_000) * model_info['cost_output']
                )

                st.info(f"💰 Estimated: ${estimated_cost:.4f} USD for {num_prompts} prompts")

    # Save preference if changed
    if selected_model_id != current_model:
        prefs.set('claude_model', selected_model_id)
        st.success(f"✅ Model preference saved: {model_info['short_name']}")

    return selected_model_id


def render_cad_generation_options(location: str = "sidebar") -> Dict[str, Any]:
    """
    Render CAD generation options.

    Args:
        location: Where to render ('sidebar' or 'main')

    Returns:
        Dictionary of selected options
    """
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = UserPreferences()

    prefs = st.session_state.user_preferences

    if location == "sidebar":
        container = st.sidebar
    else:
        container = st

    container.markdown("### ⚙️ Generation Options")

    # 2D vs 3D
    cad_mode = container.radio(
        "CAD Mode",
        ["3D", "2D"],
        index=0 if prefs.get('cad_mode', '3d') == '3d' else 1,
        help="Generate 2D sketches or 3D models",
        horizontal=True,
        key="cad_mode_selector"
    )

    if cad_mode != prefs.get('cad_mode', '3d'):
        prefs.set('cad_mode', cad_mode.lower())

    # Single part vs Assembly
    assembly_mode = container.radio(
        "Model Type",
        ["Single Part", "Assembly"],
        index=0 if prefs.get('assembly_mode', 'single_part') == 'single_part' else 1,
        help="Generate a single part or an assembly",
        horizontal=True,
        key="assembly_mode_selector"
    )

    if assembly_mode != prefs.get('assembly_mode', 'single_part'):
        mode_value = 'single_part' if assembly_mode == "Single Part" else 'assembly'
        prefs.set('assembly_mode', mode_value)

    # Export formats
    available_formats = ['STEP', 'STL', 'OBJ', 'GLTF', 'DXF']
    current_formats = prefs.get('export_formats', ['step', 'stl'])
    current_formats_upper = [f.upper() for f in current_formats]

    export_formats = container.multiselect(
        "Export Formats",
        available_formats,
        default=[f for f in available_formats if f.lower() in current_formats],
        help="Select formats to export",
        key="export_formats_selector"
    )

    if export_formats:
        export_formats_lower = [f.lower() for f in export_formats]
        if export_formats_lower != current_formats:
            prefs.set('export_formats', export_formats_lower)

    # Advanced options
    with container.expander("🔧 Advanced Options", expanded=False):
        # Constraints
        st.markdown("**Constraints:**")
        constraints = st.text_area(
            "Geometric constraints",
            placeholder="e.g., 'parallel faces', 'symmetric about X axis'",
            height=80,
            key="constraints_input"
        )

        # Auto PDF export
        auto_pdf = st.checkbox(
            "Auto-export to PDF",
            value=prefs.get('auto_export_pdf', False),
            help="Automatically generate PDF documentation",
            key="auto_pdf_checkbox"
        )

        if auto_pdf != prefs.get('auto_export_pdf', False):
            prefs.set('auto_export_pdf', auto_pdf)

        # Max tokens
        max_tokens = st.slider(
            "Max Tokens",
            min_value=1024,
            max_value=8192,
            value=prefs.get('max_tokens', 4096),
            step=256,
            help="Maximum tokens for model response",
            key="max_tokens_slider"
        )

        if max_tokens != prefs.get('max_tokens', 4096):
            prefs.set('max_tokens', max_tokens)

    return {
        'cad_mode': cad_mode.lower(),
        'assembly_mode': 'single_part' if assembly_mode == "Single Part" else 'assembly',
        'export_formats': [f.lower() for f in export_formats] if export_formats else ['step'],
        'constraints': constraints if constraints else None,
        'auto_export_pdf': auto_pdf,
        'max_tokens': max_tokens
    }


def render_preferences_panel() -> None:
    """Render full preferences management panel."""
    st.header("⚙️ User Preferences")

    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = UserPreferences()

    prefs = st.session_state.user_preferences

    tab1, tab2, tab3 = st.tabs(["Model Selection", "Generation Options", "System"])

    with tab1:
        st.markdown("### Claude Model Settings")
        render_model_selector(location="main", show_costs=True, show_descriptions=True)

    with tab2:
        st.markdown("### CAD Generation Settings")
        options = render_cad_generation_options(location="main")
        st.success("✅ Preferences saved automatically")

    with tab3:
        st.markdown("### System Settings")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                prefs.reset()
                st.success("✅ Preferences reset to defaults")
                st.rerun()

        with col2:
            if st.button("📁 Show Config Location", use_container_width=True):
                st.code(str(prefs.config_path))

        st.markdown("---")
        st.markdown("**Current Preferences:**")
        st.json(prefs.preferences)
