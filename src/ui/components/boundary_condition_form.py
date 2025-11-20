"""
Boundary Condition Form Component
Dynamic form for configuring boundary conditions in CFD simulations.
"""

import streamlit as st
from typing import Dict, List, Optional, Any
from enum import Enum


class BCType(Enum):
    """Boundary condition types."""
    INLET = "Inlet"
    OUTLET = "Outlet"
    WALL = "Wall"
    SYMMETRY = "Symmetry"
    EMPTY = "Empty"
    PRESSURE_INLET = "Pressure Inlet"
    PRESSURE_OUTLET = "Pressure Outlet"


class BCForm:
    """
    Component for configuring boundary conditions.

    Features:
    - Dynamic form based on selected patches
    - Multiple BC types support
    - Validation
    - Pre-configured templates
    """

    # BC templates for common scenarios
    BC_TEMPLATES = {
        "External Flow (Wind Tunnel)": {
            "inlet": {"type": "INLET", "velocity": 10.0, "turbulence_intensity": 0.05},
            "outlet": {"type": "OUTLET", "pressure": 0.0},
            "walls": {"type": "WALL", "velocity_type": "no-slip"},
            "symmetry": {"type": "SYMMETRY"}
        },
        "Internal Flow (Pipe)": {
            "inlet": {"type": "INLET", "velocity": 1.0, "turbulence_intensity": 0.05},
            "outlet": {"type": "PRESSURE_OUTLET", "pressure": 0.0},
            "walls": {"type": "WALL", "velocity_type": "no-slip"}
        },
        "Natural Convection": {
            "hot_wall": {"type": "WALL", "temperature": 350.0, "velocity_type": "no-slip"},
            "cold_wall": {"type": "WALL", "temperature": 300.0, "velocity_type": "no-slip"},
            "outlet": {"type": "OUTLET", "pressure": 0.0}
        }
    }

    def __init__(self, session_key: str = "bc_form"):
        """
        Initialize the boundary condition form.

        Args:
            session_key: Unique key for session state storage
        """
        self.session_key = session_key

        # Initialize session state
        if f"{session_key}_conditions" not in st.session_state:
            st.session_state[f"{session_key}_conditions"] = {}
        if f"{session_key}_patches" not in st.session_state:
            st.session_state[f"{session_key}_patches"] = []

    @property
    def conditions(self) -> Dict[str, Dict[str, Any]]:
        """Get boundary conditions."""
        return st.session_state[f"{self.session_key}_conditions"]

    @property
    def patches(self) -> List[str]:
        """Get list of patches/boundaries."""
        return st.session_state[f"{self.session_key}_patches"]

    def set_patches(self, patches: List[str]):
        """Set the list of available patches."""
        st.session_state[f"{self.session_key}_patches"] = patches

    def update_condition(self, patch: str, condition: Dict[str, Any]):
        """Update boundary condition for a patch."""
        st.session_state[f"{self.session_key}_conditions"][patch] = condition

    def render_template_selector(self):
        """Render BC template selector."""
        st.subheader("Quick Templates")

        template = st.selectbox(
            "Load Template",
            ["Custom"] + list(self.BC_TEMPLATES.keys()),
            help="Load pre-configured boundary conditions for common scenarios"
        )

        if template != "Custom":
            if st.button("Apply Template"):
                # Clear existing conditions
                st.session_state[f"{self.session_key}_conditions"] = {}

                # Apply template
                template_bcs = self.BC_TEMPLATES[template]
                for patch_name, bc_data in template_bcs.items():
                    st.session_state[f"{self.session_key}_conditions"][patch_name] = bc_data

                st.success(f"Applied template: {template}")
                st.rerun()

        st.divider()

    def render_patch_form(self, patch_name: str):
        """
        Render form for a specific patch.

        Args:
            patch_name: Name of the patch/boundary
        """
        with st.expander(f"📋 {patch_name}", expanded=True):
            # Get existing condition or create new
            existing_bc = self.conditions.get(patch_name, {})

            # BC Type selector
            bc_type = st.selectbox(
                "Boundary Condition Type",
                [bc.value for bc in BCType],
                index=[bc.value for bc in BCType].index(existing_bc.get("type", "Wall")),
                key=f"bc_type_{patch_name}"
            )

            bc_data = {"type": bc_type}

            # Type-specific parameters
            if bc_type in ["Inlet", "Pressure Inlet"]:
                self._render_inlet_form(patch_name, bc_data, existing_bc)

            elif bc_type in ["Outlet", "Pressure Outlet"]:
                self._render_outlet_form(patch_name, bc_data, existing_bc)

            elif bc_type == "Wall":
                self._render_wall_form(patch_name, bc_data, existing_bc)

            elif bc_type == "Symmetry":
                st.info("Symmetry boundary - no additional parameters needed")

            elif bc_type == "Empty":
                st.info("Empty boundary (for 2D simulations)")

            # Update the condition
            self.update_condition(patch_name, bc_data)

    def _render_inlet_form(self, patch_name: str, bc_data: Dict, existing_bc: Dict):
        """Render inlet boundary condition form."""
        col1, col2 = st.columns(2)

        with col1:
            velocity_type = st.radio(
                "Velocity Specification",
                ["Uniform", "Normal", "Profile"],
                index=["Uniform", "Normal", "Profile"].index(
                    existing_bc.get("velocity_type", "Uniform")
                ),
                key=f"velocity_type_{patch_name}"
            )
            bc_data["velocity_type"] = velocity_type

            if velocity_type == "Uniform":
                velocity = st.number_input(
                    "Velocity Magnitude (m/s)",
                    min_value=0.0,
                    value=existing_bc.get("velocity", 10.0),
                    step=0.1,
                    key=f"velocity_{patch_name}"
                )
                bc_data["velocity"] = velocity

                direction = st.multiselect(
                    "Direction (unit vector)",
                    ["X", "Y", "Z"],
                    default=existing_bc.get("direction", ["X"]),
                    key=f"direction_{patch_name}"
                )
                bc_data["direction"] = direction

            elif velocity_type == "Normal":
                velocity = st.number_input(
                    "Normal Velocity (m/s)",
                    value=existing_bc.get("velocity", 10.0),
                    step=0.1,
                    key=f"normal_velocity_{patch_name}"
                )
                bc_data["velocity"] = velocity

        with col2:
            # Turbulence parameters
            turbulence_intensity = st.slider(
                "Turbulence Intensity (%)",
                min_value=0.1,
                max_value=20.0,
                value=existing_bc.get("turbulence_intensity", 5.0) * 100
                      if "turbulence_intensity" in existing_bc else 5.0,
                step=0.1,
                key=f"turb_intensity_{patch_name}"
            ) / 100
            bc_data["turbulence_intensity"] = turbulence_intensity

            turbulent_viscosity_ratio = st.number_input(
                "Turbulent Viscosity Ratio",
                min_value=1.0,
                max_value=100.0,
                value=existing_bc.get("turbulent_viscosity_ratio", 10.0),
                step=1.0,
                key=f"turb_visc_ratio_{patch_name}"
            )
            bc_data["turbulent_viscosity_ratio"] = turbulent_viscosity_ratio

        # Temperature (if applicable)
        with st.expander("Thermal Settings (Optional)"):
            enable_temp = st.checkbox(
                "Specify Temperature",
                value=existing_bc.get("temperature") is not None,
                key=f"enable_temp_{patch_name}"
            )
            if enable_temp:
                temperature = st.number_input(
                    "Temperature (K)",
                    min_value=200.0,
                    max_value=500.0,
                    value=existing_bc.get("temperature", 300.0),
                    step=1.0,
                    key=f"temperature_{patch_name}"
                )
                bc_data["temperature"] = temperature

    def _render_outlet_form(self, patch_name: str, bc_data: Dict, existing_bc: Dict):
        """Render outlet boundary condition form."""
        col1, col2 = st.columns(2)

        with col1:
            pressure = st.number_input(
                "Static Pressure (Pa)",
                value=existing_bc.get("pressure", 0.0),
                step=100.0,
                key=f"pressure_{patch_name}",
                help="Gauge pressure (relative to reference)"
            )
            bc_data["pressure"] = pressure

        with col2:
            backflow = st.checkbox(
                "Allow Backflow",
                value=existing_bc.get("backflow", False),
                key=f"backflow_{patch_name}",
                help="Allow flow to reverse direction"
            )
            bc_data["backflow"] = backflow

            if backflow:
                backflow_turbulence = st.number_input(
                    "Backflow Turbulence Intensity (%)",
                    min_value=0.1,
                    max_value=20.0,
                    value=existing_bc.get("backflow_turbulence", 5.0),
                    step=0.1,
                    key=f"backflow_turb_{patch_name}"
                )
                bc_data["backflow_turbulence"] = backflow_turbulence

    def _render_wall_form(self, patch_name: str, bc_data: Dict, existing_bc: Dict):
        """Render wall boundary condition form."""
        col1, col2 = st.columns(2)

        with col1:
            velocity_type = st.radio(
                "Wall Motion",
                ["No-slip (Stationary)", "Slip", "Moving Wall", "Rotating Wall"],
                index=["No-slip (Stationary)", "Slip", "Moving Wall", "Rotating Wall"].index(
                    existing_bc.get("velocity_type", "No-slip (Stationary)")
                ),
                key=f"wall_velocity_type_{patch_name}"
            )
            bc_data["velocity_type"] = velocity_type

            if velocity_type == "Moving Wall":
                wall_velocity = st.number_input(
                    "Wall Velocity (m/s)",
                    value=existing_bc.get("wall_velocity", 1.0),
                    step=0.1,
                    key=f"wall_velocity_{patch_name}"
                )
                bc_data["wall_velocity"] = wall_velocity

            elif velocity_type == "Rotating Wall":
                angular_velocity = st.number_input(
                    "Angular Velocity (rad/s)",
                    value=existing_bc.get("angular_velocity", 10.0),
                    step=1.0,
                    key=f"angular_velocity_{patch_name}"
                )
                bc_data["angular_velocity"] = angular_velocity

        with col2:
            # Wall roughness
            roughness_type = st.selectbox(
                "Surface Roughness",
                ["Smooth", "Rough"],
                index=["Smooth", "Rough"].index(existing_bc.get("roughness_type", "Smooth")),
                key=f"roughness_type_{patch_name}"
            )
            bc_data["roughness_type"] = roughness_type

            if roughness_type == "Rough":
                roughness_height = st.number_input(
                    "Roughness Height (m)",
                    min_value=0.0,
                    value=existing_bc.get("roughness_height", 0.001),
                    step=0.0001,
                    format="%.4f",
                    key=f"roughness_height_{patch_name}"
                )
                bc_data["roughness_height"] = roughness_height

        # Thermal boundary condition
        with st.expander("Thermal Settings"):
            thermal_type = st.selectbox(
                "Thermal Boundary",
                ["Adiabatic", "Fixed Temperature", "Fixed Heat Flux"],
                index=["Adiabatic", "Fixed Temperature", "Fixed Heat Flux"].index(
                    existing_bc.get("thermal_type", "Adiabatic")
                ),
                key=f"thermal_type_{patch_name}"
            )
            bc_data["thermal_type"] = thermal_type

            if thermal_type == "Fixed Temperature":
                temperature = st.number_input(
                    "Wall Temperature (K)",
                    min_value=200.0,
                    max_value=500.0,
                    value=existing_bc.get("temperature", 300.0),
                    step=1.0,
                    key=f"wall_temperature_{patch_name}"
                )
                bc_data["temperature"] = temperature

            elif thermal_type == "Fixed Heat Flux":
                heat_flux = st.number_input(
                    "Heat Flux (W/m²)",
                    value=existing_bc.get("heat_flux", 0.0),
                    step=10.0,
                    key=f"heat_flux_{patch_name}"
                )
                bc_data["heat_flux"] = heat_flux

    def render_forms(self):
        """Render forms for all patches."""
        if not self.patches:
            st.warning("⚠️ No patches/boundaries detected. Please load a mesh first.")
            return

        st.subheader(f"Boundary Conditions ({len(self.patches)} patches)")

        # Show template selector
        self.render_template_selector()

        # Render form for each patch
        for patch in self.patches:
            self.render_patch_form(patch)

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate boundary conditions.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check if all patches have conditions
        for patch in self.patches:
            if patch not in self.conditions:
                errors.append(f"Missing boundary condition for patch: {patch}")

        # Check for at least one inlet and one outlet
        has_inlet = any(
            bc.get("type") in ["Inlet", "Pressure Inlet"]
            for bc in self.conditions.values()
        )
        has_outlet = any(
            bc.get("type") in ["Outlet", "Pressure Outlet"]
            for bc in self.conditions.values()
        )

        if not has_inlet:
            errors.append("At least one inlet boundary is required")
        if not has_outlet:
            errors.append("At least one outlet boundary is required")

        return len(errors) == 0, errors

    def get_conditions(self) -> Dict[str, Dict[str, Any]]:
        """Get all boundary conditions."""
        return self.conditions.copy()

    def reset(self):
        """Reset all boundary conditions."""
        st.session_state[f"{self.session_key}_conditions"] = {}
