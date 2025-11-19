"""
Mesh Configurator Component
Interactive component for configuring mesh parameters and refinement zones.
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class MeshConfigurator:
    """
    Component for configuring mesh parameters for CFD simulations.

    Features:
    - Global mesh size control
    - Refinement zone definition
    - Mesh statistics preview
    - Visual feedback for mesh settings
    """

    # Predefined mesh quality levels
    MESH_PRESETS = {
        "Coarse": {"size": 0.1, "min_size": 0.05, "max_size": 0.2},
        "Medium": {"size": 0.05, "min_size": 0.01, "max_size": 0.1},
        "Fine": {"size": 0.02, "min_size": 0.005, "max_size": 0.05},
        "Very Fine": {"size": 0.01, "min_size": 0.001, "max_size": 0.02},
    }

    def __init__(self, session_key: str = "mesh_config"):
        """
        Initialize the mesh configurator.

        Args:
            session_key: Unique key for session state storage
        """
        self.session_key = session_key

        # Initialize session state
        if f"{session_key}_settings" not in st.session_state:
            st.session_state[f"{session_key}_settings"] = {
                "quality": "Medium",
                "global_size": 0.05,
                "min_size": 0.01,
                "max_size": 0.1,
                "refinement_zones": [],
                "growth_rate": 1.3,
            }

    @property
    def settings(self) -> Dict[str, Any]:
        """Get mesh settings."""
        return st.session_state[f"{self.session_key}_settings"]

    def update_setting(self, key: str, value: Any):
        """Update a mesh setting."""
        st.session_state[f"{self.session_key}_settings"][key] = value

    def render_global_settings(self):
        """Render global mesh settings controls."""
        st.subheader("Global Mesh Settings")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Mesh quality preset
            quality = st.select_slider(
                "Mesh Quality",
                options=list(self.MESH_PRESETS.keys()),
                value=self.settings["quality"],
                help="Select overall mesh quality level"
            )

            # Update settings when quality changes
            if quality != self.settings["quality"]:
                self.update_setting("quality", quality)
                preset = self.MESH_PRESETS[quality]
                self.update_setting("global_size", preset["size"])
                self.update_setting("min_size", preset["min_size"])
                self.update_setting("max_size", preset["max_size"])

        with col2:
            st.metric("Global Size", f"{self.settings['global_size']:.3f} m")

        # Advanced settings
        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)

            with col1:
                global_size = st.number_input(
                    "Global Element Size (m)",
                    min_value=0.001,
                    max_value=1.0,
                    value=self.settings["global_size"],
                    step=0.001,
                    format="%.3f",
                    help="Target element size for the mesh"
                )
                self.update_setting("global_size", global_size)

                min_size = st.number_input(
                    "Minimum Element Size (m)",
                    min_value=0.0001,
                    max_value=global_size,
                    value=self.settings["min_size"],
                    step=0.001,
                    format="%.4f",
                    help="Minimum allowed element size"
                )
                self.update_setting("min_size", min_size)

            with col2:
                max_size = st.number_input(
                    "Maximum Element Size (m)",
                    min_value=global_size,
                    max_value=2.0,
                    value=self.settings["max_size"],
                    step=0.01,
                    format="%.3f",
                    help="Maximum allowed element size"
                )
                self.update_setting("max_size", max_size)

                growth_rate = st.number_input(
                    "Growth Rate",
                    min_value=1.01,
                    max_value=2.0,
                    value=self.settings["growth_rate"],
                    step=0.1,
                    format="%.2f",
                    help="Rate at which elements grow in size"
                )
                self.update_setting("growth_rate", growth_rate)

    def render_refinement_zones(self):
        """Render refinement zone controls."""
        st.subheader("Refinement Zones")

        # List existing refinement zones
        zones = self.settings["refinement_zones"]

        if zones:
            st.write(f"**Active Refinement Zones:** {len(zones)}")

            for idx, zone in enumerate(zones):
                with st.expander(f"Zone {idx + 1}: {zone['name']}"):
                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        st.write(f"**Type:** {zone['type']}")
                        st.write(f"**Size:** {zone['size']:.4f} m")

                    with col2:
                        if zone['type'] == 'Box':
                            st.write(f"**Center:** ({zone['center'][0]:.2f}, {zone['center'][1]:.2f}, {zone['center'][2]:.2f})")
                            st.write(f"**Dimensions:** ({zone['dims'][0]:.2f}, {zone['dims'][1]:.2f}, {zone['dims'][2]:.2f})")
                        elif zone['type'] == 'Sphere':
                            st.write(f"**Center:** ({zone['center'][0]:.2f}, {zone['center'][1]:.2f}, {zone['center'][2]:.2f})")
                            st.write(f"**Radius:** {zone['radius']:.2f} m")

                    with col3:
                        if st.button("🗑️ Remove", key=f"remove_zone_{idx}"):
                            zones.pop(idx)
                            self.update_setting("refinement_zones", zones)
                            st.rerun()
        else:
            st.info("No refinement zones defined. Add zones to refine mesh in specific regions.")

        # Add new refinement zone
        st.divider()
        st.write("**Add New Refinement Zone**")

        col1, col2 = st.columns(2)

        with col1:
            zone_type = st.selectbox("Zone Type", ["Box", "Sphere", "Cylinder"])
            zone_name = st.text_input("Zone Name", value=f"Zone_{len(zones) + 1}")

        with col2:
            zone_size = st.number_input(
                "Element Size in Zone (m)",
                min_value=0.0001,
                max_value=self.settings["global_size"],
                value=self.settings["global_size"] / 2,
                step=0.001,
                format="%.4f"
            )

        # Zone-specific parameters
        if zone_type == "Box":
            st.write("**Box Parameters**")
            col1, col2 = st.columns(2)

            with col1:
                center_x = st.number_input("Center X (m)", value=0.0, step=0.1)
                center_y = st.number_input("Center Y (m)", value=0.0, step=0.1)
                center_z = st.number_input("Center Z (m)", value=0.0, step=0.1)

            with col2:
                dim_x = st.number_input("Width (m)", min_value=0.01, value=1.0, step=0.1)
                dim_y = st.number_input("Height (m)", min_value=0.01, value=1.0, step=0.1)
                dim_z = st.number_input("Depth (m)", min_value=0.01, value=1.0, step=0.1)

            zone_params = {
                "name": zone_name,
                "type": zone_type,
                "size": zone_size,
                "center": [center_x, center_y, center_z],
                "dims": [dim_x, dim_y, dim_z]
            }

        elif zone_type == "Sphere":
            col1, col2 = st.columns(2)

            with col1:
                center_x = st.number_input("Center X (m)", value=0.0, step=0.1)
                center_y = st.number_input("Center Y (m)", value=0.0, step=0.1)
                center_z = st.number_input("Center Z (m)", value=0.0, step=0.1)

            with col2:
                radius = st.number_input("Radius (m)", min_value=0.01, value=0.5, step=0.1)

            zone_params = {
                "name": zone_name,
                "type": zone_type,
                "size": zone_size,
                "center": [center_x, center_y, center_z],
                "radius": radius
            }

        else:  # Cylinder
            col1, col2 = st.columns(2)

            with col1:
                center_x = st.number_input("Center X (m)", value=0.0, step=0.1)
                center_y = st.number_input("Center Y (m)", value=0.0, step=0.1)
                center_z = st.number_input("Center Z (m)", value=0.0, step=0.1)

            with col2:
                radius = st.number_input("Radius (m)", min_value=0.01, value=0.5, step=0.1)
                height = st.number_input("Height (m)", min_value=0.01, value=1.0, step=0.1)

            zone_params = {
                "name": zone_name,
                "type": zone_type,
                "size": zone_size,
                "center": [center_x, center_y, center_z],
                "radius": radius,
                "height": height
            }

        if st.button("➕ Add Refinement Zone", type="primary"):
            zones.append(zone_params)
            self.update_setting("refinement_zones", zones)
            st.success(f"Added refinement zone: {zone_name}")
            st.rerun()

    def estimate_mesh_statistics(self, geometry_volume: float = 1.0,
                                geometry_surface_area: float = 6.0) -> Dict[str, int]:
        """
        Estimate mesh statistics based on current settings.

        Args:
            geometry_volume: Volume of the geometry in m³
            geometry_surface_area: Surface area of the geometry in m²

        Returns:
            Dictionary with estimated node and element counts
        """
        # Simple estimation based on element size
        element_size = self.settings["global_size"]

        # Estimate volume elements (tetrahedral)
        volume_per_element = (element_size ** 3) * 0.5  # Approximate for tetrahedra
        estimated_volume_elements = int(geometry_volume / volume_per_element)

        # Estimate surface elements (triangular)
        area_per_element = (element_size ** 2) * 0.866  # Approximate for triangles
        estimated_surface_elements = int(geometry_surface_area / area_per_element)

        # Nodes (roughly 1.2x elements for tetrahedral mesh)
        estimated_nodes = int(estimated_volume_elements * 1.2)

        # Account for refinement zones
        for zone in self.settings["refinement_zones"]:
            zone_volume = self._estimate_zone_volume(zone)
            zone_element_size = zone["size"]
            zone_volume_per_element = (zone_element_size ** 3) * 0.5
            additional_elements = int(zone_volume / zone_volume_per_element)
            estimated_volume_elements += additional_elements
            estimated_nodes += int(additional_elements * 1.2)

        return {
            "nodes": estimated_nodes,
            "volume_elements": estimated_volume_elements,
            "surface_elements": estimated_surface_elements,
            "total_elements": estimated_volume_elements + estimated_surface_elements
        }

    def _estimate_zone_volume(self, zone: Dict[str, Any]) -> float:
        """Estimate the volume of a refinement zone."""
        if zone["type"] == "Box":
            return zone["dims"][0] * zone["dims"][1] * zone["dims"][2]
        elif zone["type"] == "Sphere":
            return (4/3) * np.pi * (zone["radius"] ** 3)
        elif zone["type"] == "Cylinder":
            return np.pi * (zone["radius"] ** 2) * zone["height"]
        return 0.0

    def render_statistics(self, geometry_volume: float = 1.0,
                         geometry_surface_area: float = 6.0):
        """Render mesh statistics preview."""
        st.subheader("Mesh Statistics (Estimated)")

        stats = self.estimate_mesh_statistics(geometry_volume, geometry_surface_area)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Nodes", f"{stats['nodes']:,}")

        with col2:
            st.metric("Volume Elements", f"{stats['volume_elements']:,}")

        with col3:
            st.metric("Surface Elements", f"{stats['surface_elements']:,}")

        with col4:
            st.metric("Total Elements", f"{stats['total_elements']:,}")

        # Memory estimate (rough)
        memory_mb = stats['total_elements'] * 0.001  # Very rough estimate
        if memory_mb > 1000:
            st.info(f"📊 Estimated memory requirement: ~{memory_mb/1000:.1f} GB")
        else:
            st.info(f"📊 Estimated memory requirement: ~{memory_mb:.0f} MB")

    def get_mesh_config(self) -> Dict[str, Any]:
        """Get the complete mesh configuration."""
        return self.settings.copy()

    def reset(self):
        """Reset mesh configuration to defaults."""
        st.session_state[f"{self.session_key}_settings"] = {
            "quality": "Medium",
            "global_size": 0.05,
            "min_size": 0.01,
            "max_size": 0.1,
            "refinement_zones": [],
            "growth_rate": 1.3,
        }
