"""
Custom styled file uploader component for Streamlit with drag-drop interface.
Supports progress callbacks and multiple CAD/CFD file formats.
"""

import streamlit as st
from typing import Optional, Callable, List
import io


# Supported file formats
SUPPORTED_FORMATS = {
    'dxf': 'AutoCAD DXF',
    'dwg': 'AutoCAD DWG',
    'step': 'STEP (ISO 10303)',
    'stp': 'STEP (ISO 10303)',
    'iges': 'IGES',
    'igs': 'IGES',
    'stl': 'STL (Stereolithography)',
    'obj': 'Wavefront OBJ',
    'ply': 'Polygon File Format',
    'brep': 'Boundary Representation'
}


def get_custom_css() -> str:
    """
    Get custom CSS for styled file uploader.

    Returns:
        CSS string for styling the uploader
    """
    return """
    <style>
        /* File uploader container */
        .stFileUploader {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            border: 2px dashed #fff;
            transition: all 0.3s ease;
        }

        .stFileUploader:hover {
            border-color: #ffd700;
            box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }

        /* Upload button styling */
        .stFileUploader label {
            color: white !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
        }

        /* Drag and drop text */
        .stFileUploader section {
            border: none !important;
            background: rgba(255,255,255,0.1) !important;
            backdrop-filter: blur(10px) !important;
        }

        .stFileUploader section > div {
            color: white !important;
        }

        /* Progress bar customization */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #00d2ff 0%, #3a47d5 100%);
        }

        /* Info boxes */
        .upload-info-box {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 8px;
            color: white;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .upload-info-box h3 {
            margin-top: 0;
            color: white !important;
        }

        /* Metrics styling */
        .metric-card {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: white;
        }

        .metric-label {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.9);
            margin-top: 0.5rem;
        }
    </style>
    """


def apply_custom_styles():
    """Apply custom CSS styles to Streamlit app."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def custom_file_uploader(
    label: str = "Upload CAD/CFD File",
    accept_multiple: bool = False,
    max_size_mb: int = 200,
    key: Optional[str] = None,
    help_text: Optional[str] = None
) -> Optional[List]:
    """
    Create a custom styled file uploader with support for CAD/CFD formats.

    Args:
        label: Label for the uploader
        accept_multiple: Whether to accept multiple files
        max_size_mb: Maximum file size in MB
        key: Unique key for the widget
        help_text: Help text to display

    Returns:
        List of uploaded files or None
    """
    # Get list of supported extensions
    extensions = list(SUPPORTED_FORMATS.keys())

    # Create help text if not provided
    if help_text is None:
        format_list = ', '.join([f.upper() for f in set(SUPPORTED_FORMATS.values())])
        help_text = f"Supported formats: {format_list}. Max size: {max_size_mb}MB"

    # Display info box with supported formats
    st.markdown(
        f"""
        <div class="upload-info-box">
            <h3>📁 {label}</h3>
            <p>{help_text}</p>
            <p><strong>Drag and drop your file here, or click to browse</strong></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # File uploader
    uploaded_files = st.file_uploader(
        label="",
        type=extensions,
        accept_multiple_files=accept_multiple,
        key=key,
        label_visibility="collapsed"
    )

    return uploaded_files


def show_upload_progress(progress: float, status: str = "Processing..."):
    """
    Display upload/processing progress.

    Args:
        progress: Progress value between 0 and 1
        status: Status message to display
    """
    progress_bar = st.progress(progress)
    status_text = st.empty()
    status_text.text(status)

    return progress_bar, status_text


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def display_file_info(uploaded_file):
    """
    Display information about the uploaded file.

    Args:
        uploaded_file: Streamlit UploadedFile object
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_type = SUPPORTED_FORMATS.get(file_extension, 'Unknown')

    st.markdown("### 📄 File Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">File Name</div>
                <div class="metric-value" style="font-size: 1rem;">{uploaded_file.name}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">File Size</div>
                <div class="metric-value" style="font-size: 1.2rem;">{format_file_size(uploaded_file.size)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Format</div>
                <div class="metric-value" style="font-size: 1rem;">{file_type}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_geometry_metrics(
    num_vertices: int,
    num_faces: int,
    dimensions: List[float],
    volume: float,
    surface_area: float
):
    """
    Display geometry metrics in a nice layout.

    Args:
        num_vertices: Number of vertices
        num_faces: Number of faces
        dimensions: Bounding box dimensions [width, height, depth]
        volume: Volume
        surface_area: Surface area
    """
    st.markdown("### 📊 Geometry Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{num_vertices:,}</div>
                <div class="metric-label">Vertices</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{num_faces:,}</div>
                <div class="metric-label">Faces</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{surface_area:.2f}</div>
                <div class="metric-label">Surface Area</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        st.metric("Width (X)", f"{dimensions[0]:.3f}")

    with col5:
        st.metric("Height (Y)", f"{dimensions[1]:.3f}")

    with col6:
        st.metric("Depth (Z)", f"{dimensions[2]:.3f}")

    with col7:
        st.metric("Volume", f"{volume:.3f}")


def create_download_button(
    data: bytes,
    filename: str,
    label: str = "Download",
    mime_type: str = "application/octet-stream"
):
    """
    Create a styled download button.

    Args:
        data: File data as bytes
        filename: Name for the downloaded file
        label: Button label
        mime_type: MIME type for the file
    """
    st.download_button(
        label=f"⬇️ {label}",
        data=data,
        file_name=filename,
        mime=mime_type,
        use_container_width=True
    )
