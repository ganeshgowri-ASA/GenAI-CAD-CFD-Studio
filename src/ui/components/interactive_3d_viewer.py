"""
Interactive 3D Viewer Component for Streamlit

Provides 3D visualization with measurement tools in Streamlit UI.
"""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import tempfile

try:
    import pyvista as pv
    from stpyvista import stpyvista
    HAS_STPYVISTA = True
except ImportError:
    HAS_STPYVISTA = False
    logging.warning("stpyvista not installed. Interactive 3D viewer limited.")

from ...visualization.enhanced_3d_viewer import Enhanced3DViewer, MeasurementTool

logger = logging.getLogger(__name__)


def render_interactive_3d_viewer(
    mesh_path: Optional[Path] = None,
    mesh_data: Optional[Any] = None,
    show_measurements: bool = True,
    show_info: bool = True,
    export_options: bool = True
) -> None:
    """
    Render interactive 3D viewer in Streamlit.

    Args:
        mesh_path: Path to mesh file
        mesh_data: PyVista mesh object
        show_measurements: Show measurement tools
        show_info: Show mesh information
        export_options: Show export options
    """
    st.subheader("🎨 Interactive 3D Viewer")

    if mesh_path is None and mesh_data is None:
        st.warning("No mesh data provided")
        return

    try:
        # Create viewer
        viewer = Enhanced3DViewer()

        # Load mesh
        if mesh_path:
            if not viewer.load_mesh(mesh_path):
                st.error(f"Failed to load mesh: {mesh_path}")
                return
        elif mesh_data:
            viewer.mesh = mesh_data

        # Create plotter
        plotter = viewer.create_plotter(off_screen=True, window_size=(800, 600))

        # Render options
        with st.expander("⚙️ Rendering Options", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                color = st.color_picker("Mesh Color", "#87CEEB")
                show_edges = st.checkbox("Show Edges", value=True)

            with col2:
                opacity = st.slider("Opacity", 0.0, 1.0, 1.0, 0.1)
                lighting = st.checkbox("Lighting", value=True)

            with col3:
                view_position = st.selectbox(
                    "View",
                    ["Isometric", "Top", "Front", "Side"],
                    index=0
                )

        # Render mesh
        viewer.render_mesh(
            color=color,
            show_edges=show_edges,
            opacity=opacity,
            lighting=lighting
        )

        # Set camera
        view_map = {
            "Isometric": "iso",
            "Top": "top",
            "Front": "front",
            "Side": "side"
        }
        viewer.set_camera_position(view_map[view_position])

        # Add axes
        viewer.add_axes()

        # Display in Streamlit
        if HAS_STPYVISTA:
            stpyvista(plotter)
        else:
            # Fallback: Export screenshot and display
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = Path(tmp.name)
                viewer.export_screenshot(tmp_path, resolution=(800, 600))
                st.image(str(tmp_path), use_container_width=True)
                tmp_path.unlink(missing_ok=True)

        # Mesh information
        if show_info:
            st.markdown("---")
            st.markdown("### 📊 Mesh Information")

            mesh_info = viewer.get_mesh_info()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Points", f"{mesh_info.get('n_points', 0):,}")

            with col2:
                st.metric("Cells", f"{mesh_info.get('n_cells', 0):,}")

            with col3:
                st.metric("Surface Area", f"{mesh_info.get('surface_area', 0):.2f} mm²")

            with col4:
                st.metric("BB Volume", f"{mesh_info.get('bounding_box_volume', 0):.2f} mm³")

            # Dimensions
            with st.expander("📐 Dimensions", expanded=False):
                dims = mesh_info.get('dimensions', {})
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("X", f"{dims.get('x', 0):.2f} mm")
                with col2:
                    st.metric("Y", f"{dims.get('y', 0):.2f} mm")
                with col3:
                    st.metric("Z", f"{dims.get('z', 0):.2f} mm")

                center = mesh_info.get('center', [0, 0, 0])
                st.info(f"**Center:** [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

        # Measurement tools
        if show_measurements:
            st.markdown("---")
            st.markdown("### 📏 Measurement Tools")

            measurement_type = st.selectbox(
                "Measurement Type",
                ["Distance", "Angle", "Area"],
                help="Select measurement type"
            )

            if measurement_type == "Distance":
                _render_distance_measurement(viewer)
            elif measurement_type == "Angle":
                _render_angle_measurement(viewer)
            elif measurement_type == "Area":
                _render_area_measurement(viewer)

        # Export options
        if export_options:
            st.markdown("---")
            st.markdown("### 💾 Export Options")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📸 Export Screenshot", use_container_width=True):
                    try:
                        output_path = Path("model_screenshot.png")
                        viewer.export_screenshot(output_path, resolution=(1920, 1080))
                        st.success(f"✅ Screenshot exported: {output_path}")

                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="Download Screenshot",
                                data=f,
                                file_name="model_screenshot.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"Export failed: {e}")

            with col2:
                if st.button("📊 Export Mesh Info", use_container_width=True):
                    import json
                    mesh_info = viewer.get_mesh_info()
                    info_json = json.dumps(mesh_info, indent=2)

                    st.download_button(
                        label="Download Info (JSON)",
                        data=info_json,
                        file_name="mesh_info.json",
                        mime="application/json",
                        use_container_width=True
                    )

        viewer.close()

    except Exception as e:
        st.error(f"Failed to render 3D viewer: {e}")
        logger.error(f"3D viewer error: {e}", exc_info=True)


def _render_distance_measurement(viewer: Enhanced3DViewer) -> None:
    """Render distance measurement UI."""
    st.markdown("**Measure Distance Between Two Points**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Point 1 (mm)**")
        p1_x = st.number_input("X1", value=0.0, key="dist_p1_x")
        p1_y = st.number_input("Y1", value=0.0, key="dist_p1_y")
        p1_z = st.number_input("Z1", value=0.0, key="dist_p1_z")

    with col2:
        st.markdown("**Point 2 (mm)**")
        p2_x = st.number_input("X2", value=10.0, key="dist_p2_x")
        p2_y = st.number_input("Y2", value=0.0, key="dist_p2_y")
        p2_z = st.number_input("Z2", value=0.0, key="dist_p2_z")

    if st.button("📏 Calculate Distance", use_container_width=True):
        p1 = np.array([p1_x, p1_y, p1_z])
        p2 = np.array([p2_x, p2_y, p2_z])

        result = viewer.measure_distance_points(p1, p2)

        st.success(f"**Distance:** {result['distance']:.3f} mm")

        # Store in session state
        if 'measurements' not in st.session_state:
            st.session_state.measurements = []
        st.session_state.measurements.append(result)


def _render_angle_measurement(viewer: Enhanced3DViewer) -> None:
    """Render angle measurement UI."""
    st.markdown("**Measure Angle Between Three Points (P1-P2-P3)**")
    st.caption("P2 is the vertex of the angle")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Point 1**")
        p1_x = st.number_input("X1", value=10.0, key="ang_p1_x")
        p1_y = st.number_input("Y1", value=0.0, key="ang_p1_y")
        p1_z = st.number_input("Z1", value=0.0, key="ang_p1_z")

    with col2:
        st.markdown("**Point 2 (Vertex)**")
        p2_x = st.number_input("X2", value=0.0, key="ang_p2_x")
        p2_y = st.number_input("Y2", value=0.0, key="ang_p2_y")
        p2_z = st.number_input("Z2", value=0.0, key="ang_p2_z")

    with col3:
        st.markdown("**Point 3**")
        p3_x = st.number_input("X3", value=0.0, key="ang_p3_x")
        p3_y = st.number_input("Y3", value=10.0, key="ang_p3_y")
        p3_z = st.number_input("Z3", value=0.0, key="ang_p3_z")

    if st.button("📐 Calculate Angle", use_container_width=True):
        p1 = np.array([p1_x, p1_y, p1_z])
        p2 = np.array([p2_x, p2_y, p2_z])
        p3 = np.array([p3_x, p3_y, p3_z])

        result = viewer.measure_angle_points(p1, p2, p3)

        st.success(f"**Angle:** {result['angle_degrees']:.2f}° ({result['angle_radians']:.4f} rad)")

        if 'measurements' not in st.session_state:
            st.session_state.measurements = []
        st.session_state.measurements.append(result)


def _render_area_measurement(viewer: Enhanced3DViewer) -> None:
    """Render area measurement UI."""
    st.markdown("**Measure Triangle Area**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Point 1**")
        p1_x = st.number_input("X1", value=0.0, key="area_p1_x")
        p1_y = st.number_input("Y1", value=0.0, key="area_p1_y")
        p1_z = st.number_input("Z1", value=0.0, key="area_p1_z")

    with col2:
        st.markdown("**Point 2**")
        p2_x = st.number_input("X2", value=10.0, key="area_p2_x")
        p2_y = st.number_input("Y2", value=0.0, key="area_p2_y")
        p2_z = st.number_input("Z2", value=0.0, key="area_p2_z")

    with col3:
        st.markdown("**Point 3**")
        p3_x = st.number_input("X3", value=0.0, key="area_p3_x")
        p3_y = st.number_input("Y3", value=10.0, key="area_p3_y")
        p3_z = st.number_input("Z3", value=0.0, key="area_p3_z")

    if st.button("📊 Calculate Area", use_container_width=True):
        p1 = np.array([p1_x, p1_y, p1_z])
        p2 = np.array([p2_x, p2_y, p2_z])
        p3 = np.array([p3_x, p3_y, p3_z])

        result = viewer.measure_area_triangle(p1, p2, p3)

        st.success(f"**Area:** {result['area']:.3f} mm²")

        if 'measurements' not in st.session_state:
            st.session_state.measurements = []
        st.session_state.measurements.append(result)


def render_measurement_history() -> None:
    """Render measurement history."""
    if 'measurements' not in st.session_state or not st.session_state.measurements:
        st.info("No measurements recorded yet")
        return

    st.markdown("### 📋 Measurement History")

    for i, measurement in enumerate(reversed(st.session_state.measurements)):
        with st.expander(f"Measurement #{len(st.session_state.measurements) - i}: {measurement['type'].title()}", expanded=False):
            if measurement['type'] == 'distance':
                st.markdown(f"**Distance:** {measurement['distance']:.3f} {measurement['units']}")
                st.caption(f"From: {measurement['point1']}")
                st.caption(f"To: {measurement['point2']}")

            elif measurement['type'] == 'angle':
                st.markdown(f"**Angle:** {measurement['angle_degrees']:.2f}° ({measurement['angle_radians']:.4f} rad)")
                st.caption(f"P1: {measurement['point1']}")
                st.caption(f"Vertex: {measurement['vertex']}")
                st.caption(f"P3: {measurement['point3']}")

            elif measurement['type'] == 'area':
                st.markdown(f"**Area:** {measurement['area']:.3f} {measurement['units']}")
                st.caption(f"Vertices: {measurement['vertices']}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.measurements = []
            st.rerun()

    with col2:
        if st.button("💾 Export History", use_container_width=True):
            import json
            history_json = json.dumps(st.session_state.measurements, indent=2)

            st.download_button(
                label="Download JSON",
                data=history_json,
                file_name="measurement_history.json",
                mime="application/json",
                use_container_width=True
            )
