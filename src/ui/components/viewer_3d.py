"""
3D Viewer Component using Plotly

Provides interactive 3D visualization for CAD models with:
- Rotation and zoom controls
- Multiple view angles
- Mesh visualization
- Wireframe mode
- Screenshot export

Author: GenAI CAD CFD Studio
Version: 1.0.0
"""

import streamlit as st
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from stl import mesh as stl_mesh
    HAS_NUMPY_STL = True
except ImportError:
    HAS_NUMPY_STL = False


def render_3d_viewer(
    model_data: Optional[Any] = None,
    file_path: Optional[str] = None,
    title: str = "3D Model Viewer"
) -> None:
    """
    Render interactive 3D viewer for CAD models.

    Args:
        model_data: Model data (vertices, faces, etc.)
        file_path: Path to STL file to load
        title: Viewer title
    """
    if not HAS_PLOTLY:
        st.error("Plotly is required for 3D visualization. Install with: pip install plotly")
        return

    st.subheader(title)

    # Load model from file if provided
    if file_path and Path(file_path).exists():
        if file_path.endswith('.stl'):
            model_data = load_stl_file(file_path)
        else:
            st.error(f"Unsupported file format: {file_path}")
            return

    # Check if we have data to display
    if model_data is None:
        st.info("No 3D model loaded. Upload or generate a model to visualize.")
        return

    # Control panel
    col1, col2, col3 = st.columns(3)

    with col1:
        view_mode = st.selectbox(
            "View Mode",
            ["Solid", "Wireframe", "Points"],
            key="3d_view_mode"
        )

    with col2:
        show_axes = st.checkbox("Show Axes", value=True, key="3d_show_axes")

    with col3:
        auto_rotate = st.checkbox("Auto Rotate", value=False, key="3d_auto_rotate")

    # Create 3D figure
    fig = create_3d_figure(
        model_data,
        view_mode=view_mode,
        show_axes=show_axes
    )

    # Display figure
    st.plotly_chart(fig, use_container_width=True)

    # Additional controls
    with st.expander("View Controls"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Front View"):
                st.session_state['camera_view'] = 'front'
                st.rerun()

            if st.button("Top View"):
                st.session_state['camera_view'] = 'top'
                st.rerun()

        with col2:
            if st.button("Side View"):
                st.session_state['camera_view'] = 'side'
                st.rerun()

            if st.button("Isometric"):
                st.session_state['camera_view'] = 'iso'
                st.rerun()

        # Export options
        st.markdown("---")
        if st.button("📸 Export Screenshot", use_container_width=True):
            st.info("Screenshot functionality requires browser support")


def create_3d_figure(
    model_data: Dict[str, Any],
    view_mode: str = "Solid",
    show_axes: bool = True
) -> go.Figure:
    """
    Create Plotly 3D figure from model data.

    Args:
        model_data: Dictionary with 'vertices' and 'faces'
        view_mode: Display mode
        show_axes: Whether to show axes

    Returns:
        Plotly Figure object
    """
    vertices = model_data.get('vertices', [])
    faces = model_data.get('faces', [])

    if not vertices or not faces:
        # Create sample cube if no data
        vertices, faces = create_sample_cube()

    # Extract coordinates
    x, y, z = zip(*vertices)
    i, j, k = zip(*faces)

    # Create mesh
    fig = go.Figure()

    if view_mode == "Solid":
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            color='lightblue',
            opacity=0.8,
            flatshading=True,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.2,
                roughness=0.5
            ),
            lightposition=dict(x=100, y=100, z=100)
        ))
    elif view_mode == "Wireframe":
        # Create wireframe edges
        edges_x, edges_y, edges_z = [], [], []
        for face in faces:
            for i in range(3):
                v1 = vertices[face[i]]
                v2 = vertices[face[(i + 1) % 3]]
                edges_x.extend([v1[0], v2[0], None])
                edges_y.extend([v1[1], v2[1], None])
                edges_z.extend([v1[2], v2[2], None])

        fig.add_trace(go.Scatter3d(
            x=edges_x, y=edges_y, z=edges_z,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Wireframe'
        ))
    else:  # Points
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=3, color='blue'),
            name='Vertices'
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=show_axes, title='X'),
            yaxis=dict(visible=show_axes, title='Y'),
            zaxis=dict(visible=show_axes, title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )

    return fig


def load_stl_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load STL file and convert to viewer format.

    Args:
        file_path: Path to STL file

    Returns:
        Dictionary with vertices and faces
    """
    if not HAS_NUMPY_STL:
        st.error("numpy-stl is required. Install with: pip install numpy-stl")
        return None

    try:
        # Load STL mesh
        mesh = stl_mesh.Mesh.from_file(file_path)

        # Extract vertices and faces
        vertices = []
        faces = []
        vertex_map = {}

        for i, triangle in enumerate(mesh.vectors):
            triangle_indices = []
            for vertex in triangle:
                vertex_tuple = tuple(vertex)
                if vertex_tuple not in vertex_map:
                    vertex_map[vertex_tuple] = len(vertices)
                    vertices.append(vertex)
                triangle_indices.append(vertex_map[vertex_tuple])
            faces.append(triangle_indices)

        return {
            'vertices': vertices,
            'faces': faces
        }

    except Exception as e:
        st.error(f"Failed to load STL file: {e}")
        return None


def create_sample_cube() -> Tuple[List, List]:
    """
    Create sample cube for demonstration.

    Returns:
        Tuple of (vertices, faces)
    """
    # Define cube vertices
    vertices = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ]

    # Define faces (triangles)
    faces = [
        # Bottom
        [0, 1, 2], [0, 2, 3],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Front
        [0, 1, 5], [0, 5, 4],
        # Back
        [2, 3, 7], [2, 7, 6],
        # Left
        [0, 3, 7], [0, 7, 4],
        # Right
        [1, 2, 6], [1, 6, 5]
    ]

    return vertices, faces


def render_model_comparison(
    model1_data: Dict[str, Any],
    model2_data: Dict[str, Any],
    titles: Tuple[str, str] = ("Model 1", "Model 2")
) -> None:
    """
    Render side-by-side comparison of two models.

    Args:
        model1_data: First model data
        model2_data: Second model data
        titles: Tuple of titles for each model
    """
    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{titles[0]}**")
        fig1 = create_3d_figure(model1_data)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(f"**{titles[1]}**")
        fig2 = create_3d_figure(model2_data)
        st.plotly_chart(fig2, use_container_width=True)


def render_model_measurements(model_data: Dict[str, Any]) -> None:
    """
    Display measurements and properties of the model.

    Args:
        model_data: Model data dictionary
    """
    st.subheader("Model Measurements")

    vertices = model_data.get('vertices', [])
    faces = model_data.get('faces', [])

    if not vertices:
        st.info("No model data available")
        return

    # Calculate bounding box
    vertices_array = np.array(vertices)
    min_coords = vertices_array.min(axis=0)
    max_coords = vertices_array.max(axis=0)
    dimensions = max_coords - min_coords

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Length (X)", f"{dimensions[0]:.2f} units")

    with col2:
        st.metric("Width (Y)", f"{dimensions[1]:.2f} units")

    with col3:
        st.metric("Height (Z)", f"{dimensions[2]:.2f} units")

    # Additional info
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Vertices:** {len(vertices)}")
        st.write(f"**Faces:** {len(faces)}")

    with col2:
        st.write(f"**Min:** ({min_coords[0]:.2f}, {min_coords[1]:.2f}, {min_coords[2]:.2f})")
        st.write(f"**Max:** ({max_coords[0]:.2f}, {max_coords[1]:.2f}, {max_coords[2]:.2f})")
