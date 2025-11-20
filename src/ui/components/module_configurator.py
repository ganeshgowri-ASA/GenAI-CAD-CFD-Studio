"""
Module Configurator Component for Solar PV Layout Generator
Provides input fields and presets for solar module configuration
"""

import streamlit as st
from typing import Dict, Optional
from dataclasses import dataclass
import json


@dataclass
class ModuleConfig:
    """Configuration for solar PV modules"""
    width: float  # meters
    length: float  # meters
    power_watts: int  # watts
    row_spacing: float  # meters
    column_spacing: float  # meters
    tilt_angle: float  # degrees
    azimuth: float  # degrees (0=North, 90=East, 180=South, 270=West)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'width': self.width,
            'length': self.length,
            'power_watts': self.power_watts,
            'row_spacing': self.row_spacing,
            'column_spacing': self.column_spacing,
            'tilt_angle': self.tilt_angle,
            'azimuth': self.azimuth
        }

    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validate module configuration

        Returns:
        --------
        tuple : (is_valid, error_message)
        """
        if self.width <= 0:
            return False, "Module width must be positive"
        if self.length <= 0:
            return False, "Module length must be positive"
        if self.power_watts <= 0:
            return False, "Module power must be positive"
        if self.row_spacing < 0:
            return False, "Row spacing cannot be negative"
        if self.column_spacing < 0:
            return False, "Column spacing cannot be negative"
        if not 0 <= self.tilt_angle <= 90:
            return False, "Tilt angle must be between 0 and 90 degrees"
        if not 0 <= self.azimuth <= 360:
            return False, "Azimuth must be between 0 and 360 degrees"

        return True, None


class ModuleConfigurator:
    """Interactive module configuration component"""

    # Standard module presets (typical solar panel sizes)
    PRESETS = {
        "Standard 60-cell (300W)": {
            'width': 0.992,  # ~39 inches
            'length': 1.650,  # ~65 inches
            'power_watts': 300,
            'row_spacing': 1.0,
            'column_spacing': 0.02,
            'tilt_angle': 20.0,
            'azimuth': 180.0
        },
        "Standard 72-cell (350W)": {
            'width': 0.992,
            'length': 1.960,  # ~77 inches
            'power_watts': 350,
            'row_spacing': 1.2,
            'column_spacing': 0.02,
            'tilt_angle': 20.0,
            'azimuth': 180.0
        },
        "High-efficiency 60-cell (400W)": {
            'width': 1.046,
            'length': 1.690,
            'power_watts': 400,
            'row_spacing': 1.0,
            'column_spacing': 0.02,
            'tilt_angle': 20.0,
            'azimuth': 180.0
        },
        "High-efficiency 72-cell (450W)": {
            'width': 1.046,
            'length': 2.008,
            'power_watts': 450,
            'row_spacing': 1.2,
            'column_spacing': 0.02,
            'tilt_angle': 20.0,
            'azimuth': 180.0
        },
        "Bifacial 144-cell (500W)": {
            'width': 1.134,
            'length': 2.279,
            'power_watts': 500,
            'row_spacing': 1.5,
            'column_spacing': 0.02,
            'tilt_angle': 25.0,
            'azimuth': 180.0
        },
        "Custom": {
            'width': 1.0,
            'length': 2.0,
            'power_watts': 400,
            'row_spacing': 1.0,
            'column_spacing': 0.02,
            'tilt_angle': 20.0,
            'azimuth': 180.0
        }
    }

    def __init__(self):
        self.config = None
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'module_preset' not in st.session_state:
            st.session_state.module_preset = "Standard 60-cell (300W)"
        if 'module_config' not in st.session_state:
            st.session_state.module_config = self.PRESETS["Standard 60-cell (300W)"].copy()

    def render(self) -> ModuleConfig:
        """
        Render the module configurator UI

        Returns:
        --------
        ModuleConfig : Current module configuration
        """
        st.subheader("⚡ Module Configuration")

        # Preset selector
        col1, col2 = st.columns([3, 1])

        with col1:
            preset = st.selectbox(
                "Select Preset",
                options=list(self.PRESETS.keys()),
                index=list(self.PRESETS.keys()).index(st.session_state.module_preset),
                help="Choose a standard module size or custom configuration"
            )

        with col2:
            if st.button("Load Preset", use_container_width=True):
                st.session_state.module_config = self.PRESETS[preset].copy()
                st.session_state.module_preset = preset
                st.rerun()

        # Module dimensions section
        st.markdown("#### 📐 Module Dimensions")
        col1, col2 = st.columns(2)

        with col1:
            width = st.number_input(
                "Width (meters)",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state.module_config['width']),
                step=0.01,
                format="%.3f",
                help="Module width in meters"
            )

        with col2:
            length = st.number_input(
                "Length (meters)",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state.module_config['length']),
                step=0.01,
                format="%.3f",
                help="Module length in meters"
            )

        # Power rating
        power_watts = st.number_input(
            "Power Rating (watts)",
            min_value=50,
            max_value=1000,
            value=int(st.session_state.module_config['power_watts']),
            step=10,
            help="Nominal power output of the module in watts"
        )

        # Spacing section
        st.markdown("#### 📏 Spacing")
        col1, col2 = st.columns(2)

        with col1:
            row_spacing = st.number_input(
                "Row Spacing (meters)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.module_config['row_spacing']),
                step=0.1,
                format="%.2f",
                help="Spacing between module rows (for shadow clearance)"
            )

        with col2:
            column_spacing = st.number_input(
                "Column Spacing (meters)",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.module_config['column_spacing']),
                step=0.01,
                format="%.3f",
                help="Spacing between modules in a row"
            )

        # Orientation section
        st.markdown("#### 🧭 Orientation")
        col1, col2 = st.columns(2)

        with col1:
            tilt_angle = st.slider(
                "Tilt Angle (degrees)",
                min_value=0.0,
                max_value=90.0,
                value=float(st.session_state.module_config['tilt_angle']),
                step=1.0,
                help="Angle of module tilt (0=flat, 90=vertical)"
            )

        with col2:
            azimuth = st.slider(
                "Azimuth (degrees)",
                min_value=0.0,
                max_value=360.0,
                value=float(st.session_state.module_config['azimuth']),
                step=1.0,
                help="Direction the modules face (0=North, 90=East, 180=South, 270=West)"
            )

        # Azimuth direction indicator
        direction = self._get_direction_name(azimuth)
        st.caption(f"📍 Direction: {direction}")

        # Display module area
        module_area = width * length
        st.info(f"📊 Module Area: {module_area:.3f} m² | Efficiency: {power_watts / module_area / 10:.1f}%")

        # Create and validate configuration
        self.config = ModuleConfig(
            width=width,
            length=length,
            power_watts=power_watts,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            tilt_angle=tilt_angle,
            azimuth=azimuth
        )

        # Update session state
        st.session_state.module_config = self.config.to_dict()

        # Validate
        is_valid, error_msg = self.config.validate()
        if not is_valid:
            st.error(f"❌ Validation Error: {error_msg}")

        return self.config

    def render_compact(self) -> ModuleConfig:
        """
        Render a compact version of the configurator (for sidebar)

        Returns:
        --------
        ModuleConfig : Current module configuration
        """
        st.markdown("### ⚡ Module Config")

        # Preset selector
        preset = st.selectbox(
            "Preset",
            options=list(self.PRESETS.keys()),
            index=list(self.PRESETS.keys()).index(st.session_state.module_preset),
            key="preset_compact"
        )

        if st.button("Load", key="load_compact"):
            st.session_state.module_config = self.PRESETS[preset].copy()
            st.session_state.module_preset = preset
            st.rerun()

        # Compact inputs
        with st.expander("📐 Dimensions", expanded=True):
            width = st.number_input(
                "Width (m)",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state.module_config['width']),
                step=0.01,
                format="%.3f",
                key="width_compact"
            )

            length = st.number_input(
                "Length (m)",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state.module_config['length']),
                step=0.01,
                format="%.3f",
                key="length_compact"
            )

            power_watts = st.number_input(
                "Power (W)",
                min_value=50,
                max_value=1000,
                value=int(st.session_state.module_config['power_watts']),
                step=10,
                key="power_compact"
            )

        with st.expander("📏 Spacing"):
            row_spacing = st.number_input(
                "Row (m)",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.module_config['row_spacing']),
                step=0.1,
                format="%.2f",
                key="row_spacing_compact"
            )

            column_spacing = st.number_input(
                "Column (m)",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.module_config['column_spacing']),
                step=0.01,
                format="%.3f",
                key="column_spacing_compact"
            )

        with st.expander("🧭 Orientation"):
            tilt_angle = st.slider(
                "Tilt (°)",
                min_value=0.0,
                max_value=90.0,
                value=float(st.session_state.module_config['tilt_angle']),
                step=1.0,
                key="tilt_compact"
            )

            azimuth = st.slider(
                "Azimuth (°)",
                min_value=0.0,
                max_value=360.0,
                value=float(st.session_state.module_config['azimuth']),
                step=1.0,
                key="azimuth_compact"
            )

            st.caption(f"📍 {self._get_direction_name(azimuth)}")

        # Create configuration
        self.config = ModuleConfig(
            width=width,
            length=length,
            power_watts=power_watts,
            row_spacing=row_spacing,
            column_spacing=column_spacing,
            tilt_angle=tilt_angle,
            azimuth=azimuth
        )

        # Update session state
        st.session_state.module_config = self.config.to_dict()

        return self.config

    @staticmethod
    def _get_direction_name(azimuth: float) -> str:
        """Convert azimuth angle to direction name"""
        directions = [
            ("North", 0, 22.5),
            ("North-East", 22.5, 67.5),
            ("East", 67.5, 112.5),
            ("South-East", 112.5, 157.5),
            ("South", 157.5, 202.5),
            ("South-West", 202.5, 247.5),
            ("West", 247.5, 292.5),
            ("North-West", 292.5, 337.5),
            ("North", 337.5, 360)
        ]

        for name, min_angle, max_angle in directions:
            if min_angle <= azimuth < max_angle:
                return name

        return "North"

    def export_config(self) -> str:
        """Export configuration as JSON string"""
        if self.config is None:
            return "{}"

        return json.dumps(self.config.to_dict(), indent=2)

    def import_config(self, config_json: str) -> bool:
        """
        Import configuration from JSON string

        Returns:
        --------
        bool : True if successful, False otherwise
        """
        try:
            config_dict = json.loads(config_json)
            self.config = ModuleConfig(**config_dict)
            st.session_state.module_config = config_dict
            is_valid, _ = self.config.validate()
            return is_valid
        except Exception as e:
            st.error(f"Failed to import configuration: {str(e)}")
            return False
