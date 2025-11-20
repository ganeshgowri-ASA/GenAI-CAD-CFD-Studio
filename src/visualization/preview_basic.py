"""
3D Mesh Preview using Plotly for interactive visualization.
Supports rotation, zoom, pan, and wireframe/solid toggle.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Literal, Dict, Any
from ..io.universal_importer import GeometryData


def plot_mesh_3d(
    vertices: np.ndarray,
    faces: np.ndarray,
    mode: Literal['solid', 'wireframe', 'both'] = 'solid',
    color: str = '#1f77b4',
    title: str = '3D Mesh Preview',
    show_axes: bool = True,
    camera_position: Optional[Dict[str, Any]] = None,
    **kwargs
) -> go.Figure:
    """
    Create an interactive 3D mesh plot using Plotly.

    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 or Mx4 array of face indices (triangles or quads)
        mode: Display mode - 'solid', 'wireframe', or 'both'
        color: Color for the mesh (hex or named color)
        title: Plot title
        show_axes: Whether to show axis labels and grid
        camera_position: Optional camera position dictionary
        **kwargs: Additional plotly parameters

    Returns:
        Plotly Figure object ready to display in Streamlit
    """
    if len(vertices) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No geometry data to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig

    # Convert faces to triangles if needed
    triangulated_faces = _triangulate_faces(faces)

    # Extract triangle indices
    i_coords = triangulated_faces[:, 0]
    j_coords = triangulated_faces[:, 1]
    k_coords = triangulated_faces[:, 2]

    # Create the mesh
    traces = []

    if mode in ['solid', 'both']:
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i_coords,
            j=j_coords,
            k=k_coords,
            color=color,
            opacity=0.8 if mode == 'both' else 1.0,
            name='Mesh',
            hovertemplate='<b>Vertex</b><br>' +
                         'X: %{x:.3f}<br>' +
                         'Y: %{y:.3f}<br>' +
                         'Z: %{z:.3f}<br>' +
                         '<extra></extra>',
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                fresnel=0.2,
                specular=0.5,
                roughness=0.5
            ),
            lightposition=dict(
                x=100,
                y=200,
                z=0
            )
        )
        traces.append(mesh_trace)

    if mode in ['wireframe', 'both']:
        # Create wireframe edges
        edge_x = []
        edge_y = []
        edge_z = []

        for face in faces:
            # Handle both triangles and quads
            face_vertices = vertices[face]
            n_verts = len(face_vertices)

            for i in range(n_verts):
                v1 = face_vertices[i]
                v2 = face_vertices[(i + 1) % n_verts]

                edge_x.extend([v1[0], v2[0], None])
                edge_y.extend([v1[1], v2[1], None])
                edge_z.extend([v1[2], v2[2], None])

        wireframe_trace = go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='black' if mode == 'both' else color, width=1),
            name='Wireframe',
            hoverinfo='skip'
        )
        traces.append(wireframe_trace)

    # Create figure
    fig = go.Figure(data=traces)

    # Set camera position if provided
    camera = camera_position if camera_position else dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
    )

    # Configure layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(
                title='X' if show_axes else '',
                showgrid=show_axes,
                showbackground=show_axes,
                gridcolor='lightgray',
                backgroundcolor='white'
            ),
            yaxis=dict(
                title='Y' if show_axes else '',
                showgrid=show_axes,
                showbackground=show_axes,
                gridcolor='lightgray',
                backgroundcolor='white'
            ),
            zaxis=dict(
                title='Z' if show_axes else '',
                showgrid=show_axes,
                showbackground=show_axes,
                gridcolor='lightgray',
                backgroundcolor='white'
            ),
            camera=camera,
            aspectmode='data'  # Maintain aspect ratio
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode='closest',
        **kwargs
    )

    return fig


def plot_geometry_data(
    geometry: GeometryData,
    mode: Literal['solid', 'wireframe', 'both'] = 'solid',
    **kwargs
) -> go.Figure:
    """
    Create 3D plot from GeometryData object.

    Args:
        geometry: GeometryData object from universal_importer
        mode: Display mode - 'solid', 'wireframe', or 'both'
        **kwargs: Additional parameters passed to plot_mesh_3d

    Returns:
        Plotly Figure object
    """
    title = kwargs.pop('title', f'3D Preview - {geometry.metadata.get("file_type", "").upper()}')

    return plot_mesh_3d(
        vertices=geometry.vertices,
        faces=geometry.faces,
        mode=mode,
        title=title,
        **kwargs
    )


def _triangulate_faces(faces: np.ndarray) -> np.ndarray:
    """
    Convert faces (triangles or quads) to triangulated format.

    Args:
        faces: Nx3 or Nx4 array of face indices

    Returns:
        Mx3 array of triangulated faces
    """
    if len(faces) == 0:
        return np.array([])

    triangulated = []

    for face in faces:
        if len(face) == 3:
            # Already a triangle
            triangulated.append(face)
        elif len(face) == 4:
            # Quad - split into 2 triangles
            triangulated.append([face[0], face[1], face[2]])
            triangulated.append([face[0], face[2], face[3]])
        elif len(face) > 4:
            # Polygon - fan triangulation
            for i in range(1, len(face) - 1):
                triangulated.append([face[0], face[i], face[i + 1]])

    return np.array(triangulated, dtype=np.int32)


def create_camera_controls() -> Dict[str, Dict[str, Any]]:
    """
    Create preset camera positions for different views.

    Returns:
        Dictionary of camera presets
    """
    return {
        'isometric': dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        'top': dict(
            eye=dict(x=0, y=0, z=3),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        'front': dict(
            eye=dict(x=0, y=-3, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        'side': dict(
            eye=dict(x=3, y=0, z=0),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
    }


def get_mesh_statistics(vertices: np.ndarray, faces: np.ndarray) -> Dict[str, Any]:
    """
    Calculate mesh statistics.

    Args:
        vertices: Nx3 array of vertex coordinates
        faces: Mx3 or Mx4 array of face indices

    Returns:
        Dictionary of statistics
    """
    if len(vertices) == 0:
        return {
            'num_vertices': 0,
            'num_faces': 0,
            'num_edges': 0,
            'bounding_box_min': [0, 0, 0],
            'bounding_box_max': [0, 0, 0],
            'dimensions': [0, 0, 0],
            'center': [0, 0, 0]
        }

    # Calculate edges (approximate)
    num_edges = sum(len(face) for face in faces)

    # Bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    dimensions = bbox_max - bbox_min
    center = (bbox_min + bbox_max) / 2

    return {
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'num_edges': num_edges,
        'bounding_box_min': bbox_min.tolist(),
        'bounding_box_max': bbox_max.tolist(),
        'dimensions': dimensions.tolist(),
        'center': center.tolist()
    }
