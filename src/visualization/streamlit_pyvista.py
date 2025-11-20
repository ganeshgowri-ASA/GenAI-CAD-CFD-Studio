"""
Streamlit integration wrapper for PyVista visualization.

Provides seamless integration between PyVista 3D rendering and Streamlit
web applications with interactive controls.
"""

import streamlit as st
import pyvista as pv
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

try:
    from stpyvista import stpyvista
    STPYVISTA_AVAILABLE = True
except ImportError:
    STPYVISTA_AVAILABLE = False
    import warnings
    warnings.warn(
        "stpyvista not installed. Install with: pip install stpyvista",
        ImportWarning
    )


class StreamlitPyVista:
    """
    Wrapper class for integrating PyVista with Streamlit applications.

    Handles interactive controls, view management, and proper handling
    of Streamlit's rerun behavior.
    """

    def __init__(self):
        """Initialize Streamlit-PyVista wrapper."""
        if not STPYVISTA_AVAILABLE:
            raise ImportError(
                "stpyvista is required for Streamlit integration. "
                "Install with: pip install stpyvista"
            )

    @staticmethod
    def stpyvista_display(
        plotter: pv.Plotter,
        key: str = "pyvista_viewer",
        panel_kwargs: Optional[Dict] = None,
        **kwargs
    ) -> None:
        """
        Display PyVista plotter in Streamlit with interactive controls.

        Args:
            plotter: PyVista plotter object to display
            key: Unique key for the Streamlit component
            panel_kwargs: Configuration for the panel.js viewer
            **kwargs: Additional arguments
                - use_container_width: bool, default True
                - height: int, default 600
        """
        if not STPYVISTA_AVAILABLE:
            st.error("stpyvista is not installed. Cannot display 3D viewer.")
            return

        # Default panel configuration
        default_panel = {
            'orientation_widget': True,
            'interactive': True,
        }
        if panel_kwargs:
            default_panel.update(panel_kwargs)

        # Display using stpyvista
        stpyvista(
            plotter,
            key=key,
            panel_kwargs=default_panel,
            use_container_width=kwargs.get('use_container_width', True),
            height=kwargs.get('height', 600)
        )

    @staticmethod
    def create_interactive_viewer(
        mesh_data: pv.DataSet,
        key: str = "interactive_viewer",
        sidebar_controls: bool = True,
        scalars: Optional[str] = None
    ) -> None:
        """
        Create interactive 3D viewer with Streamlit controls.

        Args:
            mesh_data: PyVista mesh to visualize
            key: Unique key for the viewer
            sidebar_controls: Show controls in sidebar
            scalars: Name of scalar field to display
        """
        # Initialize session state for viewer settings
        if f'{key}_view' not in st.session_state:
            st.session_state[f'{key}_view'] = 'iso'
        if f'{key}_display_mode' not in st.session_state:
            st.session_state[f'{key}_display_mode'] = 'surface'
        if f'{key}_opacity' not in st.session_state:
            st.session_state[f'{key}_opacity'] = 1.0
        if f'{key}_show_edges' not in st.session_state:
            st.session_state[f'{key}_show_edges'] = False
        if f'{key}_cmap' not in st.session_state:
            st.session_state[f'{key}_cmap'] = 'jet'

        # Create controls container
        controls_container = st.sidebar if sidebar_controls else st

        with controls_container:
            st.subheader("🎨 Viewer Controls")

            # View presets
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📐 Front", key=f"{key}_front", use_container_width=True):
                    st.session_state[f'{key}_view'] = 'xy'
                if st.button("📊 Top", key=f"{key}_top", use_container_width=True):
                    st.session_state[f'{key}_view'] = 'xz'
            with col2:
                if st.button("🔷 Isometric", key=f"{key}_iso", use_container_width=True):
                    st.session_state[f'{key}_view'] = 'iso'
                if st.button("📏 Side", key=f"{key}_side", use_container_width=True):
                    st.session_state[f'{key}_view'] = 'yz'

            st.divider()

            # Display mode
            display_mode = st.selectbox(
                "Display Mode",
                options=['surface', 'wireframe', 'points', 'surface_with_edges'],
                index=['surface', 'wireframe', 'points', 'surface_with_edges'].index(
                    st.session_state[f'{key}_display_mode']
                ),
                key=f"{key}_display_select"
            )
            st.session_state[f'{key}_display_mode'] = display_mode

            # Opacity slider
            opacity = st.slider(
                "Opacity",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state[f'{key}_opacity'],
                step=0.05,
                key=f"{key}_opacity_slider"
            )
            st.session_state[f'{key}_opacity'] = opacity

            # Edge visibility
            show_edges = st.checkbox(
                "Show Edges",
                value=st.session_state[f'{key}_show_edges'],
                key=f"{key}_edges_check"
            )
            st.session_state[f'{key}_show_edges'] = show_edges

            # Colormap selection (if scalars provided)
            if scalars:
                colormaps = ['jet', 'viridis', 'plasma', 'coolwarm', 'rainbow', 'turbo']
                cmap = st.selectbox(
                    "Colormap",
                    options=colormaps,
                    index=colormaps.index(st.session_state[f'{key}_cmap'])
                    if st.session_state[f'{key}_cmap'] in colormaps else 0,
                    key=f"{key}_cmap_select"
                )
                st.session_state[f'{key}_cmap'] = cmap

        # Create plotter based on settings
        plotter = pv.Plotter(window_size=(1024, 768))
        plotter.set_background('white')

        # Configure mesh display based on mode
        mesh_kwargs = {
            'opacity': opacity,
            'show_edges': show_edges or display_mode == 'surface_with_edges',
        }

        if scalars:
            mesh_kwargs['scalars'] = scalars
            mesh_kwargs['cmap'] = st.session_state[f'{key}_cmap']
            mesh_kwargs['show_scalar_bar'] = True

        if display_mode == 'wireframe':
            mesh_kwargs['style'] = 'wireframe'
            mesh_kwargs['color'] = 'black'
            mesh_kwargs['line_width'] = 1
        elif display_mode == 'points':
            mesh_kwargs['style'] = 'points'
            mesh_kwargs['point_size'] = 5
            mesh_kwargs['color'] = 'blue'
        elif display_mode in ['surface', 'surface_with_edges']:
            mesh_kwargs['style'] = 'surface'
            if not scalars:
                mesh_kwargs['color'] = 'lightblue'

        plotter.add_mesh(mesh_data, **mesh_kwargs)

        # Set camera view
        plotter.camera_position = st.session_state[f'{key}_view']
        plotter.reset_camera()

        # Display in Streamlit
        StreamlitPyVista.stpyvista_display(plotter, key=key)

    @staticmethod
    def create_comparison_viewer(
        mesh_list: List[Tuple[pv.DataSet, str]],
        key: str = "comparison_viewer"
    ) -> None:
        """
        Create side-by-side comparison viewer for multiple meshes.

        Args:
            mesh_list: List of (mesh, title) tuples
            key: Unique key for the viewer
        """
        n_meshes = len(mesh_list)

        if n_meshes == 0:
            st.warning("No meshes provided for comparison.")
            return

        # Create subplot layout
        n_cols = min(n_meshes, 3)
        n_rows = (n_meshes + n_cols - 1) // n_cols

        plotter = pv.Plotter(
            shape=(n_rows, n_cols),
            window_size=(1600, 600 * n_rows)
        )
        plotter.set_background('white')

        # Add each mesh to subplot
        for idx, (mesh, title) in enumerate(mesh_list):
            row = idx // n_cols
            col = idx % n_cols

            plotter.subplot(row, col)
            plotter.add_text(title, position='upper_edge', font_size=12)
            plotter.add_mesh(
                mesh,
                color='lightblue',
                show_edges=True,
                opacity=0.9
            )
            plotter.camera_position = 'iso'
            plotter.reset_camera()

        # Link all cameras for synchronized viewing
        if n_meshes > 1:
            plotter.link_views()

        # Display in Streamlit
        StreamlitPyVista.stpyvista_display(plotter, key=key, height=600 * n_rows)

    @staticmethod
    def create_animation_viewer(
        mesh_sequence: List[pv.DataSet],
        key: str = "animation_viewer",
        fps: int = 10,
        loop: bool = True
    ) -> None:
        """
        Create animated viewer for mesh sequences.

        Args:
            mesh_sequence: List of meshes representing animation frames
            key: Unique key for the viewer
            fps: Frames per second
            loop: Loop animation
        """
        if not mesh_sequence:
            st.warning("No mesh sequence provided.")
            return

        # Initialize session state
        if f'{key}_frame' not in st.session_state:
            st.session_state[f'{key}_frame'] = 0
        if f'{key}_playing' not in st.session_state:
            st.session_state[f'{key}_playing'] = False

        # Animation controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])

        with col1:
            if st.button("⏮️", key=f"{key}_first"):
                st.session_state[f'{key}_frame'] = 0

        with col2:
            play_label = "⏸️" if st.session_state[f'{key}_playing'] else "▶️"
            if st.button(play_label, key=f"{key}_play"):
                st.session_state[f'{key}_playing'] = not st.session_state[f'{key}_playing']

        with col3:
            if st.button("⏭️", key=f"{key}_last"):
                st.session_state[f'{key}_frame'] = len(mesh_sequence) - 1

        with col4:
            frame = st.slider(
                "Frame",
                min_value=0,
                max_value=len(mesh_sequence) - 1,
                value=st.session_state[f'{key}_frame'],
                key=f"{key}_frame_slider"
            )
            st.session_state[f'{key}_frame'] = frame

        # Auto-advance if playing
        if st.session_state[f'{key}_playing']:
            next_frame = (frame + 1) % len(mesh_sequence) if loop else min(frame + 1, len(mesh_sequence) - 1)
            st.session_state[f'{key}_frame'] = next_frame
            if not loop and next_frame == len(mesh_sequence) - 1:
                st.session_state[f'{key}_playing'] = False
            st.rerun()

        # Display current frame
        current_mesh = mesh_sequence[frame]
        plotter = pv.Plotter(window_size=(1024, 768))
        plotter.set_background('white')
        plotter.add_mesh(
            current_mesh,
            color='lightblue',
            show_edges=True,
            opacity=0.9
        )
        plotter.camera_position = 'iso'
        plotter.reset_camera()

        StreamlitPyVista.stpyvista_display(plotter, key=f"{key}_frame_{frame}")

        # Display frame info
        st.caption(f"Frame {frame + 1} of {len(mesh_sequence)}")

    @staticmethod
    def create_mesh_info_panel(mesh: pv.DataSet) -> None:
        """
        Display mesh information panel in Streamlit.

        Args:
            mesh: PyVista mesh to analyze
        """
        st.subheader("📊 Mesh Information")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Points", f"{mesh.n_points:,}")
            st.metric("Cells", f"{mesh.n_cells:,}")

        with col2:
            st.metric("Memory", f"{mesh.memory_usage / 1024 / 1024:.2f} MB")
            if hasattr(mesh, 'volume'):
                try:
                    st.metric("Volume", f"{mesh.volume:.4f}")
                except:
                    st.metric("Volume", "N/A")

        # Bounds information
        bounds = mesh.bounds
        st.write("**Bounding Box:**")
        st.write(f"- X: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
        st.write(f"- Y: [{bounds[2]:.3f}, {bounds[3]:.3f}]")
        st.write(f"- Z: [{bounds[4]:.3f}, {bounds[5]:.3f}]")

        # Available arrays
        if mesh.n_arrays > 0:
            st.write("**Available Data Arrays:**")
            for name in mesh.array_names:
                array = mesh[name]
                if hasattr(array, 'shape'):
                    shape_str = f" {array.shape}"
                    dtype_str = f" ({array.dtype})"
                else:
                    shape_str = ""
                    dtype_str = ""
                st.write(f"- `{name}`{shape_str}{dtype_str}")
