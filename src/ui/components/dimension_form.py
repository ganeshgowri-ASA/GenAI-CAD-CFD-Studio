"""
Dimension Form Component for Design Studio
Displays and allows editing of extracted CAD parameters
"""
import streamlit as st
from typing import Dict, List, Optional, Any


class DimensionForm:
    """
    Dynamic form component for editing CAD dimensions and parameters
    """

    # Supported units for different measurement types
    UNITS = {
        "length": ["mm", "cm", "m", "inches", "feet"],
        "angle": ["degrees", "radians"],
        "volume": ["mm³", "cm³", "m³", "liters"],
        "mass": ["g", "kg", "lbs"]
    }

    def __init__(self, session_key: str = "dimension_params"):
        """
        Initialize the dimension form

        Args:
            session_key: Session state key for storing parameters
        """
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {}

    def render_form(self, extracted_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render a dynamic form based on extracted parameters

        Args:
            extracted_params: Dictionary of parameters extracted from AI

        Returns:
            Dictionary of user-confirmed/edited parameters
        """
        st.subheader("📏 Design Parameters")

        # Update session state with new params if provided
        if extracted_params and extracted_params != st.session_state.get(self.session_key, {}):
            st.session_state[self.session_key] = extracted_params.copy()

        params = st.session_state.get(self.session_key, {})

        if not params:
            st.info("💡 Describe your design in the chat to extract parameters automatically.")
            return {}

        # Display object type if available
        object_type = params.get("object_type", "Custom Object")
        st.markdown(f"**Object Type:** `{object_type}`")

        st.divider()

        # Unit selector
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Dimensions**")
        with col2:
            default_unit = params.get("unit", "mm")
            unit = st.selectbox(
                "Unit",
                self.UNITS["length"],
                index=self.UNITS["length"].index(default_unit) if default_unit in self.UNITS["length"] else 0,
                key="dimension_unit"
            )
            params["unit"] = unit

        # Render parameter fields dynamically
        edited_params = self._render_parameter_fields(params)

        # Add validation summary
        self._render_validation_summary(edited_params)

        return edited_params

    def _render_parameter_fields(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render input fields for each parameter

        Args:
            params: Dictionary of parameters

        Returns:
            Dictionary of edited parameters
        """
        edited_params = params.copy()

        # Skip non-dimension fields
        skip_fields = ["object_type", "unit", "description", "material", "color"]

        # Group parameters by type
        dimension_fields = []
        other_fields = []

        for key, value in params.items():
            if key not in skip_fields:
                if isinstance(value, (int, float)):
                    dimension_fields.append((key, value))
                else:
                    other_fields.append((key, value))

        # Render dimension fields
        if dimension_fields:
            for key, value in dimension_fields:
                edited_params[key] = self._render_number_input(key, value)

        # Render other fields
        if other_fields:
            st.markdown("**Additional Parameters**")
            for key, value in other_fields:
                edited_params[key] = self._render_field_by_type(key, value)

        return edited_params

    def _render_number_input(self, key: str, value: float) -> float:
        """
        Render a number input field with validation

        Args:
            key: Parameter name
            value: Current value

        Returns:
            Edited value
        """
        # Format the label nicely
        label = key.replace("_", " ").title()

        # Determine step size based on current value
        step = 0.1 if value < 10 else (1.0 if value < 100 else 10.0)

        return st.number_input(
            label,
            min_value=0.0,
            value=float(value),
            step=step,
            format="%.2f",
            key=f"param_{key}",
            help=f"Adjust the {label.lower()} of your design"
        )

    def _render_field_by_type(self, key: str, value: Any) -> Any:
        """
        Render an appropriate input field based on value type

        Args:
            key: Parameter name
            value: Current value

        Returns:
            Edited value
        """
        label = key.replace("_", " ").title()

        if isinstance(value, bool):
            return st.checkbox(label, value=value, key=f"param_{key}")
        elif isinstance(value, str):
            return st.text_input(label, value=value, key=f"param_{key}")
        elif isinstance(value, list):
            return st.multiselect(label, options=value, default=value, key=f"param_{key}")
        else:
            return st.text_input(label, value=str(value), key=f"param_{key}")

    def _render_validation_summary(self, params: Dict[str, Any]):
        """
        Display validation status and any warnings

        Args:
            params: Current parameters
        """
        # Validate parameters
        warnings = self._validate_parameters(params)

        if warnings:
            with st.expander("⚠️ Validation Warnings", expanded=False):
                for warning in warnings:
                    st.warning(warning)
        else:
            st.success("✓ All parameters validated successfully")

    def _validate_parameters(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters and return any warnings

        Args:
            params: Parameters to validate

        Returns:
            List of warning messages
        """
        warnings = []

        # Check for zero or negative dimensions
        for key, value in params.items():
            if isinstance(value, (int, float)) and key not in ["angle", "rotation"]:
                if value <= 0:
                    warnings.append(f"❌ {key.replace('_', ' ').title()} should be greater than 0")
                elif value > 10000:
                    warnings.append(f"⚠️ {key.replace('_', ' ').title()} seems very large ({value})")

        return warnings

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current parameters

        Returns:
            Dictionary of current parameters
        """
        return st.session_state.get(self.session_key, {})

    def update_parameters(self, params: Dict[str, Any]):
        """
        Update the parameters

        Args:
            params: New parameters to set
        """
        st.session_state[self.session_key] = params

    def clear_parameters(self):
        """Clear all parameters"""
        st.session_state[self.session_key] = {}

    def export_parameters(self) -> str:
        """
        Export parameters as formatted string

        Returns:
            Formatted parameter string
        """
        params = self.get_parameters()
        if not params:
            return ""

        lines = ["# Design Parameters", ""]
        for key, value in params.items():
            lines.append(f"{key}: {value}")

        return "\n".join(lines)
