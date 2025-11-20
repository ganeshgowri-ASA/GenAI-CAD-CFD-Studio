"""
File Import & Conversion UI Tab for GenAI-CAD-CFD-Studio.

Complete implementation with:
- Multi-format file uploader (DXF, DWG, STEP, IGES, STL, OBJ, PLY, BREP)
- 3D preview with interactive controls
- Geometry information display
- Export and conversion functionality
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Optional
import time
import io

# Import custom components
from .components.file_uploader import (
    apply_custom_styles,
    custom_file_uploader,
    display_file_info,
    display_geometry_metrics,
    create_download_button,
    format_file_size
)

# Import importer and visualization
from ..io import GeometryData, parse as parse_file
from ..visualization.preview_basic import (
    plot_geometry_data,
    create_camera_controls
)


def render_file_import_tab():
    """
    Main function to render the File Import & Conversion tab.
    This is the entry point for the file import UI.
    """
    # Apply custom CSS styles
    apply_custom_styles()

    # Header
    st.header('📁 File Import & Conversion')

    st.markdown(
        """
        Import and visualize CAD/CFD files in various formats.
        Support for industry-standard formats with real-time 3D preview.
        """
    )

    st.markdown("---")

    # File uploader section
    uploaded_file = custom_file_uploader(
        label="Upload Your CAD/CFD File",
        accept_multiple=False,
        max_size_mb=200,
        key="cad_file_uploader"
    )

    # Process uploaded file
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)
    else:
        # Show instructions when no file is uploaded
        show_instructions()


def show_instructions():
    """Display instructions for using the file import feature."""
    st.info(
        """
        **How to use:**
        1. Click or drag-and-drop a CAD/CFD file into the upload area above
        2. Supported formats: DXF, DWG, STEP, IGES, STL, OBJ, PLY, BREP
        3. View your 3D model with interactive controls
        4. Export to different formats as needed
        """
    )

    # Display supported formats in a nice grid
    st.markdown("### 🔧 Supported Formats")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **CAD Formats:**
        - DXF (AutoCAD)
        - DWG (AutoCAD)
        - STEP (ISO 10303)
        - IGES
        """)

    with col2:
        st.markdown("""
        **Mesh Formats:**
        - STL (3D Printing)
        - OBJ (Wavefront)
        - PLY (Polygon)
        """)

    with col3:
        st.markdown("""
        **Advanced:**
        - BREP
        - Coming soon: More formats!
        """)


def process_uploaded_file(uploaded_file):
    """
    Process the uploaded file: parse, display info, show preview.

    Args:
        uploaded_file: Streamlit UploadedFile object
    """
    # Display file information
    display_file_info(uploaded_file)

    st.markdown("---")

    # Create progress bar
    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Update progress
            status_text.text("📂 Reading file...")
            progress_bar.progress(20)
            time.sleep(0.2)

            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            # Update progress
            status_text.text("🔍 Parsing geometry...")
            progress_bar.progress(40)
            time.sleep(0.2)

            # Parse the file
            geometry = parse_file(tmp_path)

            # Update progress
            status_text.text("📊 Extracting information...")
            progress_bar.progress(60)
            time.sleep(0.2)

            # Clean up temp file
            os.unlink(tmp_path)

            # Update progress
            status_text.text("✅ Processing complete!")
            progress_bar.progress(100)
            time.sleep(0.5)

            # Clear progress indicators
            progress_container.empty()

            # Display success message
            st.success("File loaded successfully!")

            # Store geometry in session state for later use
            st.session_state['current_geometry'] = geometry
            st.session_state['current_filename'] = uploaded_file.name

            # Display geometry information
            display_geometry_info(geometry)

            st.markdown("---")

            # Display 3D preview
            display_3d_preview(geometry)

            st.markdown("---")

            # Export section
            display_export_section(geometry, uploaded_file.name)

        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Error processing file: {str(e)}")

            # Show detailed error in expander
            with st.expander("Error Details"):
                st.code(str(e))

            # Show helpful message
            st.info(
                """
                **Troubleshooting Tips:**
                - Ensure the file is not corrupted
                - Check if the file format is supported
                - Try re-exporting from your CAD software
                - Some formats (DXF, DWG, STEP, IGES, BREP) may require additional libraries for full support
                """
            )


def display_geometry_info(geometry: GeometryData):
    """
    Display detailed geometry information.

    Args:
        geometry: Parsed GeometryData object
    """
    # Display metrics
    display_geometry_metrics(
        num_vertices=geometry.num_vertices,
        num_faces=geometry.num_faces,
        dimensions=geometry.bounding_box_dimensions.tolist(),
        volume=geometry.volume,
        surface_area=geometry.surface_area
    )

    # Display layer information for DXF files
    if geometry.layers:
        with st.expander("📋 Layer Information"):
            st.write("**Detected Layers:**")
            for i, layer in enumerate(geometry.layers, 1):
                st.write(f"{i}. {layer}")

    # Display bounding box info
    with st.expander("📐 Bounding Box Details"):
        bbox_min, bbox_max = geometry.bounding_box

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Minimum Coordinates:**")
            st.write(f"- X: {bbox_min[0]:.3f}")
            st.write(f"- Y: {bbox_min[1]:.3f}")
            st.write(f"- Z: {bbox_min[2]:.3f}")

        with col2:
            st.write("**Maximum Coordinates:**")
            st.write(f"- X: {bbox_max[0]:.3f}")
            st.write(f"- Y: {bbox_max[1]:.3f}")
            st.write(f"- Z: {bbox_max[2]:.3f}")

    # Display metadata notes if any
    if 'note' in geometry.metadata:
        st.info(f"ℹ️ {geometry.metadata['note']}")


def display_3d_preview(geometry: GeometryData):
    """
    Display interactive 3D preview of the geometry.

    Args:
        geometry: Parsed GeometryData object
    """
    st.markdown("### 🎨 3D Preview")

    # Preview controls
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        view_mode = st.selectbox(
            "Display Mode",
            options=['solid', 'wireframe', 'both'],
            index=0,
            key='view_mode'
        )

    with col2:
        camera_preset = st.selectbox(
            "Camera View",
            options=['isometric', 'top', 'front', 'side'],
            index=0,
            key='camera_preset'
        )

    with col3:
        show_axes = st.checkbox("Show Axes", value=True, key='show_axes')

    # Get camera position
    camera_controls = create_camera_controls()
    camera_position = camera_controls[camera_preset]

    # Create and display the plot
    try:
        fig = plot_geometry_data(
            geometry=geometry,
            mode=view_mode,
            show_axes=show_axes,
            camera_position=camera_position
        )

        # Display with Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Display interaction hints
        with st.expander("💡 Interaction Tips"):
            st.markdown("""
            **Mouse Controls:**
            - **Left Click + Drag:** Rotate the model
            - **Right Click + Drag:** Pan the view
            - **Scroll Wheel:** Zoom in/out
            - **Double Click:** Reset view

            **Toolbar:**
            - Use the toolbar buttons for zoom, pan, and reset
            - Download plot as PNG using the camera icon
            """)

    except Exception as e:
        st.error(f"Error rendering 3D preview: {str(e)}")


def display_export_section(geometry: GeometryData, original_filename: str):
    """
    Display export and conversion options.

    Args:
        geometry: Parsed GeometryData object
        original_filename: Original filename
    """
    st.markdown("### 📤 Export & Convert")

    st.write("Convert your model to other formats:")

    col1, col2, col3 = st.columns(3)

    # Get base filename without extension
    base_name = Path(original_filename).stem

    with col1:
        st.markdown("**Mesh Formats**")
        if st.button("Export as STL", use_container_width=True):
            export_as_stl(geometry, f"{base_name}.stl")

        if st.button("Export as OBJ", use_container_width=True):
            export_as_obj(geometry, f"{base_name}.obj")

    with col2:
        st.markdown("**Point Cloud**")
        if st.button("Export Vertices (CSV)", use_container_width=True):
            export_vertices_csv(geometry, f"{base_name}_vertices.csv")

        if st.button("Export as PLY", use_container_width=True):
            export_as_ply(geometry, f"{base_name}.ply")

    with col3:
        st.markdown("**Data Export**")
        if st.button("Export Metrics (JSON)", use_container_width=True):
            export_metrics_json(geometry, f"{base_name}_metrics.json")

        if st.button("Export Report (TXT)", use_container_width=True):
            export_report_txt(geometry, f"{base_name}_report.txt")


def export_as_stl(geometry: GeometryData, filename: str):
    """Export geometry as STL file."""
    try:
        import struct

        # Create binary STL
        stl_data = io.BytesIO()

        # Write header (80 bytes)
        header = b'Binary STL file generated by GenAI-CAD-CFD-Studio' + b' ' * 30
        stl_data.write(header[:80])

        # Count triangles
        num_triangles = geometry.num_faces
        stl_data.write(struct.pack('<I', num_triangles))

        # Write triangles
        for face in geometry.faces:
            # Calculate normal (placeholder - should be computed properly)
            normal = [0.0, 0.0, 1.0]
            stl_data.write(struct.pack('<3f', *normal))

            # Write vertices
            for vertex_idx in face[:3]:  # Take first 3 vertices
                vertex = geometry.vertices[vertex_idx]
                stl_data.write(struct.pack('<3f', *vertex))

            # Write attribute byte count
            stl_data.write(struct.pack('<H', 0))

        stl_bytes = stl_data.getvalue()
        create_download_button(stl_bytes, filename, "Download STL", "application/sla")
        st.success(f"✅ STL file ready for download!")

    except Exception as e:
        st.error(f"Error exporting STL: {str(e)}")


def export_as_obj(geometry: GeometryData, filename: str):
    """Export geometry as OBJ file."""
    try:
        obj_data = io.StringIO()

        # Write header
        obj_data.write("# OBJ file generated by GenAI-CAD-CFD-Studio\n")
        obj_data.write(f"# Vertices: {geometry.num_vertices}\n")
        obj_data.write(f"# Faces: {geometry.num_faces}\n\n")

        # Write vertices
        for vertex in geometry.vertices:
            obj_data.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        obj_data.write("\n")

        # Write faces (OBJ is 1-indexed)
        for face in geometry.faces:
            face_str = " ".join([str(idx + 1) for idx in face])
            obj_data.write(f"f {face_str}\n")

        obj_bytes = obj_data.getvalue().encode('utf-8')
        create_download_button(obj_bytes, filename, "Download OBJ", "text/plain")
        st.success(f"✅ OBJ file ready for download!")

    except Exception as e:
        st.error(f"Error exporting OBJ: {str(e)}")


def export_as_ply(geometry: GeometryData, filename: str):
    """Export geometry as PLY file."""
    try:
        ply_data = io.StringIO()

        # Write header
        ply_data.write("ply\n")
        ply_data.write("format ascii 1.0\n")
        ply_data.write(f"element vertex {geometry.num_vertices}\n")
        ply_data.write("property float x\n")
        ply_data.write("property float y\n")
        ply_data.write("property float z\n")
        ply_data.write(f"element face {geometry.num_faces}\n")
        ply_data.write("property list uchar int vertex_indices\n")
        ply_data.write("end_header\n")

        # Write vertices
        for vertex in geometry.vertices:
            ply_data.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

        # Write faces
        for face in geometry.faces:
            face_str = " ".join([str(idx) for idx in face])
            ply_data.write(f"{len(face)} {face_str}\n")

        ply_bytes = ply_data.getvalue().encode('utf-8')
        create_download_button(ply_bytes, filename, "Download PLY", "text/plain")
        st.success(f"✅ PLY file ready for download!")

    except Exception as e:
        st.error(f"Error exporting PLY: {str(e)}")


def export_vertices_csv(geometry: GeometryData, filename: str):
    """Export vertices as CSV file."""
    try:
        csv_data = io.StringIO()

        # Write header
        csv_data.write("index,x,y,z\n")

        # Write vertices
        for i, vertex in enumerate(geometry.vertices):
            csv_data.write(f"{i},{vertex[0]},{vertex[1]},{vertex[2]}\n")

        csv_bytes = csv_data.getvalue().encode('utf-8')
        create_download_button(csv_bytes, filename, "Download CSV", "text/csv")
        st.success(f"✅ Vertex data ready for download!")

    except Exception as e:
        st.error(f"Error exporting CSV: {str(e)}")


def export_metrics_json(geometry: GeometryData, filename: str):
    """Export geometry metrics as JSON."""
    try:
        import json

        bbox_min, bbox_max = geometry.bounding_box

        metrics = {
            "file_info": {
                "file_type": geometry.metadata.get('file_type', 'unknown'),
                "file_size": geometry.metadata.get('file_size', 0)
            },
            "geometry": {
                "num_vertices": geometry.num_vertices,
                "num_faces": geometry.num_faces,
                "bounding_box": {
                    "min": bbox_min.tolist(),
                    "max": bbox_max.tolist(),
                    "dimensions": geometry.bounding_box_dimensions.tolist()
                },
                "volume": geometry.volume,
                "surface_area": geometry.surface_area
            },
            "layers": geometry.layers
        }

        json_str = json.dumps(metrics, indent=2)
        json_bytes = json_str.encode('utf-8')

        create_download_button(json_bytes, filename, "Download JSON", "application/json")
        st.success(f"✅ Metrics data ready for download!")

    except Exception as e:
        st.error(f"Error exporting JSON: {str(e)}")


def export_report_txt(geometry: GeometryData, filename: str):
    """Export detailed report as text file."""
    try:
        report = io.StringIO()

        bbox_min, bbox_max = geometry.bounding_box
        dims = geometry.bounding_box_dimensions

        report.write("=" * 60 + "\n")
        report.write("GenAI-CAD-CFD-Studio - Geometry Analysis Report\n")
        report.write("=" * 60 + "\n\n")

        report.write("FILE INFORMATION\n")
        report.write("-" * 60 + "\n")
        report.write(f"File Type: {geometry.metadata.get('file_type', 'unknown').upper()}\n")
        report.write(f"File Size: {format_file_size(geometry.metadata.get('file_size', 0))}\n\n")

        report.write("GEOMETRY STATISTICS\n")
        report.write("-" * 60 + "\n")
        report.write(f"Number of Vertices: {geometry.num_vertices:,}\n")
        report.write(f"Number of Faces: {geometry.num_faces:,}\n")
        report.write(f"Surface Area: {geometry.surface_area:.6f}\n")
        report.write(f"Volume: {geometry.volume:.6f}\n\n")

        report.write("BOUNDING BOX\n")
        report.write("-" * 60 + "\n")
        report.write(f"Minimum Point: ({bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f})\n")
        report.write(f"Maximum Point: ({bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f})\n")
        report.write(f"Dimensions (W×H×D): {dims[0]:.6f} × {dims[1]:.6f} × {dims[2]:.6f}\n\n")

        if geometry.layers:
            report.write("LAYERS\n")
            report.write("-" * 60 + "\n")
            for i, layer in enumerate(geometry.layers, 1):
                report.write(f"{i}. {layer}\n")
            report.write("\n")

        report.write("=" * 60 + "\n")
        report.write("End of Report\n")
        report.write("=" * 60 + "\n")

        report_bytes = report.getvalue().encode('utf-8')
        create_download_button(report_bytes, filename, "Download Report", "text/plain")
        st.success(f"✅ Report ready for download!")

    except Exception as e:
        st.error(f"Error exporting report: {str(e)}")


# Wrapper function for tab integration
def render():
    """Render function for tab integration"""
    render_file_import_tab()


# Main entry point when run as standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="File Import & Conversion",
        page_icon="📁",
        layout="wide"
    )
    render_file_import_tab()
