"""
3D Preview Component for Design Studio
Interactive 3D visualization using Plotly
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Optional, List, Tuple, Any


class Preview3D:
    """
    Interactive 3D preview component for CAD models
    """

    # Camera preset configurations
    CAMERA_PRESETS = {
        "isometric": {
            "eye": dict(x=1.5, y=1.5, z=1.5),
            "center": dict(x=0, y=0, z=0),
            "up": dict(x=0, y=0, z=1)
        },
        "front": {
            "eye": dict(x=0, y=-2, z=0),
            "center": dict(x=0, y=0, z=0),
            "up": dict(x=0, y=0, z=1)
        },
        "top": {
            "eye": dict(x=0, y=0, z=2),
            "center": dict(x=0, y=0, z=0),
            "up": dict(x=0, y=1, z=0)
        },
        "right": {
            "eye": dict(x=2, y=0, z=0),
            "center": dict(x=0, y=0, z=0),
            "up": dict(x=0, y=0, z=1)
        }
    }

    # View mode configurations
    VIEW_MODES = {
        "solid": {"mode": "solid", "opacity": 1.0},
        "wireframe": {"mode": "wireframe", "opacity": 1.0},
        "shaded": {"mode": "solid", "opacity": 0.7}
    }

    def __init__(self, session_key: str = "preview_3d_state"):
        """
        Initialize the 3D preview component

        Args:
            session_key: Session state key for storing preview state
        """
        self.session_key = session_key
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                "view_mode": "solid",
                "camera_preset": "isometric"
            }

    def render_model(self, geometry_data: Optional[Dict[str, Any]] = None):
        """
        Render a 3D model with interactive controls

        Args:
            geometry_data: Dictionary containing mesh data (vertices, faces, etc.)
        """
        st.subheader("🎨 3D Preview")

        # Render controls
        self._render_controls()

        st.divider()

        # Render the 3D visualization
        if geometry_data:
            self._render_geometry(geometry_data)
        else:
            self._render_placeholder()

    def _render_controls(self):
        """Render view controls (camera presets, view modes)"""
        col1, col2, col3 = st.columns(3)

        with col1:
            view_mode = st.selectbox(
                "View Mode",
                list(self.VIEW_MODES.keys()),
                index=list(self.VIEW_MODES.keys()).index(
                    st.session_state[self.session_key]["view_mode"]
                ),
                key="view_mode_selector"
            )
            st.session_state[self.session_key]["view_mode"] = view_mode

        with col2:
            camera_preset = st.selectbox(
                "Camera View",
                list(self.CAMERA_PRESETS.keys()),
                index=list(self.CAMERA_PRESETS.keys()).index(
                    st.session_state[self.session_key]["camera_preset"]
                ),
                key="camera_preset_selector"
            )
            st.session_state[self.session_key]["camera_preset"] = camera_preset

        with col3:
            if st.button("🔄 Reset View", use_container_width=True):
                st.session_state[self.session_key]["camera_preset"] = "isometric"
                st.rerun()

    def _render_geometry(self, geometry_data: Dict[str, Any]):
        """
        Render the 3D geometry

        Args:
            geometry_data: Dictionary containing geometry information
        """
        # Get current view settings
        view_mode = st.session_state[self.session_key]["view_mode"]
        camera_preset = st.session_state[self.session_key]["camera_preset"]

        # Create the figure
        fig = self._create_figure(geometry_data, view_mode, camera_preset)

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Display model information
        self._render_model_info(geometry_data)

    def _create_figure(
        self,
        geometry_data: Dict[str, Any],
        view_mode: str,
        camera_preset: str
    ) -> go.Figure:
        """
        Create a Plotly figure from geometry data

        Args:
            geometry_data: Geometry data dictionary
            view_mode: Rendering mode (solid, wireframe, shaded)
            camera_preset: Camera view preset

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Check if we have mesh data or need to generate from parameters
        if "vertices" in geometry_data and "faces" in geometry_data:
            vertices = np.array(geometry_data["vertices"])
            faces = np.array(geometry_data["faces"])

            # Add mesh trace
            fig.add_trace(self._create_mesh_trace(vertices, faces, view_mode))
        else:
            # Generate simple geometry from parameters (e.g., box, cylinder)
            fig = self._create_parametric_geometry(geometry_data, view_mode)

        # Apply camera settings
        camera = self.CAMERA_PRESETS[camera_preset]

        # Update layout
        fig.update_layout(
            scene=dict(
                camera=camera,
                xaxis=dict(title="X", showgrid=True, zeroline=True),
                yaxis=dict(title="Y", showgrid=True, zeroline=True),
                zaxis=dict(title="Z", showgrid=True, zeroline=True),
                aspectmode="data"
            ),
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )

        return fig

    def _create_mesh_trace(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        view_mode: str
    ) -> go.Mesh3d:
        """
        Create a mesh trace for Plotly

        Args:
            vertices: Array of vertex coordinates
            faces: Array of face indices
            view_mode: Rendering mode

        Returns:
            Plotly Mesh3d trace
        """
        mode_config = self.VIEW_MODES[view_mode]

        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=mode_config["opacity"],
            flatshading=True if view_mode == "solid" else False
        )

    def _create_parametric_geometry(
        self,
        params: Dict[str, Any],
        view_mode: str
    ) -> go.Figure:
        """
        Create simple parametric geometry from parameters

        Args:
            params: Parameter dictionary
            view_mode: Rendering mode

        Returns:
            Plotly Figure with parametric geometry
        """
        # Default to a box if we don't have specific geometry
        length = params.get("length", params.get("width", 100))
        width = params.get("width", params.get("length", 100))
        height = params.get("height", params.get("depth", 100))

        # Create a simple box
        vertices, faces = self._generate_box_mesh(length, width, height)

        fig = go.Figure()
        fig.add_trace(self._create_mesh_trace(vertices, faces, view_mode))

        return fig

    @staticmethod
    def _generate_box_mesh(length: float, width: float, height: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mesh data for a box

        Args:
            length: Box length
            width: Box width
            height: Box height

        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Center the box at origin
        l, w, h = length / 2, width / 2, height / 2

        # Define vertices
        vertices = np.array([
            [-l, -w, -h], [l, -w, -h], [l, w, -h], [-l, w, -h],  # Bottom face
            [-l, -w, h], [l, -w, h], [l, w, h], [-l, w, h]  # Top face
        ])

        # Define faces (triangles)
        faces = np.array([
            # Bottom
            [0, 1, 2], [0, 2, 3],
            # Top
            [4, 6, 5], [4, 7, 6],
            # Front
            [0, 5, 1], [0, 4, 5],
            # Back
            [2, 7, 3], [2, 6, 7],
            # Left
            [0, 3, 7], [0, 7, 4],
            # Right
            [1, 6, 2], [1, 5, 6]
        ])

        return vertices, faces

    def _render_model_info(self, geometry_data: Dict[str, Any]):
        """
        Display model information and metadata

        Args:
            geometry_data: Geometry data dictionary
        """
        with st.expander("📊 Model Information", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Dimensions", self._format_dimensions(geometry_data))
                if "volume" in geometry_data:
                    st.metric("Volume", f"{geometry_data['volume']:.2f} mm³")

            with col2:
                if "vertices" in geometry_data:
                    st.metric("Vertices", len(geometry_data["vertices"]))
                if "faces" in geometry_data:
                    st.metric("Faces", len(geometry_data["faces"]))

    def _format_dimensions(self, geometry_data: Dict[str, Any]) -> str:
        """
        Format dimensions for display

        Args:
            geometry_data: Geometry data dictionary

        Returns:
            Formatted dimension string
        """
        dims = []
        for key in ["length", "width", "height", "radius", "diameter"]:
            if key in geometry_data:
                dims.append(f"{key[:1].upper()}: {geometry_data[key]}")

        return " × ".join(dims) if dims else "N/A"

    def _render_placeholder(self):
        """Render a placeholder when no geometry is available"""
        st.info("👆 Generate a design to see the 3D preview here")

        # Show a simple example cube
        fig = go.Figure()
        vertices, faces = self._generate_box_mesh(100, 100, 100)
        fig.add_trace(self._create_mesh_trace(vertices, faces, "wireframe"))

        camera = self.CAMERA_PRESETS["isometric"]
        fig.update_layout(
            scene=dict(
                camera=camera,
                xaxis=dict(title="X", showgrid=True),
                yaxis=dict(title="Y", showgrid=True),
                zaxis=dict(title="Z", showgrid=True),
                aspectmode="data"
            ),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True, key="placeholder_preview")

    def render_export_section(self, geometry_data: Optional[Dict[str, Any]] = None):
        """
        Render export options for the model

        Args:
            geometry_data: Current geometry data
        """
        st.subheader("💾 Export")

        if not geometry_data:
            st.info("Generate a model to enable export options")
            return

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📦 STEP", use_container_width=True, disabled=True):
                st.info("STEP export coming soon!")

        with col2:
            if st.button("🗿 STL", use_container_width=True, disabled=True):
                st.info("STL export coming soon!")

        with col3:
            if st.button("📐 OBJ", use_container_width=True, disabled=True):
                st.info("OBJ export coming soon!")

        st.caption("💡 Export functionality will be enabled after CAD engine integration")
