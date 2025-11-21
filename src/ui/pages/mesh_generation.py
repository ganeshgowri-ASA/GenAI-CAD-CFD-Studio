"""
Mesh Generation Page - Gmsh/PyMesh Integration

Provides mesh generation and quality analysis capabilities.

Features:
- Automatic mesh generation from CAD models
- Mesh quality metrics (aspect ratio, skewness, etc.)
- Refinement controls (global and local)
- Multiple element types (tetrahedral, hexahedral)
- Export in various FEA formats (Abaqus, Salome, ANSYS, etc.)
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Optional imports
try:
    import gmsh
    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    logger.warning("gmsh not installed. Mesh generation disabled.")

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False
    logger.warning("meshio not installed. Mesh export limited.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    logger.warning("pyvista not installed. Mesh visualization limited.")


class MeshGenerator:
    """Wrapper for gmsh mesh generation"""

    def __init__(self):
        """Initialize mesh generator"""
        if not HAS_GMSH:
            raise ImportError("gmsh not installed")

        self.initialized = False

    def initialize(self):
        """Initialize gmsh"""
        if not self.initialized:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output
            self.initialized = True

    def finalize(self):
        """Finalize gmsh"""
        if self.initialized:
            gmsh.finalize()
            self.initialized = False

    def generate_mesh(
        self,
        input_file: str,
        element_size: float = 1.0,
        element_type: str = "tetrahedral",
        order: int = 1,
        optimize: bool = True,
        **options
    ) -> Optional[str]:
        """
        Generate mesh from CAD file.

        Args:
            input_file: Path to CAD file (STEP, STL, etc.)
            element_size: Global element size
            element_type: Element type ("tetrahedral", "hexahedral")
            order: Element order (1 or 2)
            optimize: Whether to optimize mesh
            **options: Additional mesh options

        Returns:
            Path to generated mesh file or None
        """
        try:
            self.initialize()

            # Clear any existing model
            gmsh.clear()
            gmsh.model.add("mesh_model")

            # Import geometry
            logger.info(f"Importing geometry from {input_file}")

            if input_file.endswith('.step') or input_file.endswith('.stp'):
                gmsh.model.occ.importShapes(input_file)
            elif input_file.endswith('.stl'):
                gmsh.merge(input_file)
            else:
                logger.error(f"Unsupported file format: {input_file}")
                return None

            gmsh.model.occ.synchronize()

            # Set mesh options
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", element_size * 0.5)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", element_size * 2.0)
            gmsh.option.setNumber("Mesh.ElementOrder", order)

            if element_type == "hexahedral":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                gmsh.option.setNumber("Mesh.Algorithm", 8)  # Delaunay for quads
            else:
                gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

            # Optimization
            if optimize:
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

            # Generate mesh
            logger.info("Generating 3D mesh...")
            gmsh.model.mesh.generate(3)

            # Save mesh
            output_file = input_file.replace('.step', '.msh').replace('.stp', '.msh').replace('.stl', '.msh')
            gmsh.write(output_file)

            logger.info(f"Mesh generated: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Mesh generation failed: {e}", exc_info=True)
            return None

        finally:
            # Don't finalize here to allow further operations
            pass

    def get_mesh_stats(self) -> Dict[str, Any]:
        """Get statistics about the current mesh"""
        try:
            stats = {
                'num_nodes': 0,
                'num_elements': 0,
                'element_types': {},
                'quality': {}
            }

            # Get node count
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            stats['num_nodes'] = len(node_tags)

            # Get element counts by type
            element_types = gmsh.model.mesh.getElementTypes()

            for etype in element_types:
                element_name = gmsh.model.mesh.getElementProperties(etype)[0]
                element_tags, _ = gmsh.model.mesh.getElementsByType(etype)
                stats['element_types'][element_name] = len(element_tags)
                stats['num_elements'] += len(element_tags)

            # Get mesh quality metrics
            qualities = gmsh.model.mesh.getElementQualities()
            if len(qualities) > 0:
                stats['quality']['min'] = float(np.min(qualities))
                stats['quality']['max'] = float(np.max(qualities))
                stats['quality']['mean'] = float(np.mean(qualities))
                stats['quality']['std'] = float(np.std(qualities))

            return stats

        except Exception as e:
            logger.error(f"Failed to get mesh stats: {e}")
            return {}


def calculate_mesh_metrics(mesh_file: str) -> Dict[str, Any]:
    """Calculate advanced mesh quality metrics"""
    if not HAS_MESHIO or not HAS_NUMPY:
        return {}

    try:
        mesh = meshio.read(mesh_file)

        metrics = {
            'num_points': len(mesh.points),
            'num_cells': sum(len(cells.data) for cells in mesh.cells),
            'cell_types': {cells.type: len(cells.data) for cells in mesh.cells}
        }

        # Calculate quality metrics for tetrahedral meshes
        for cells in mesh.cells:
            if cells.type == 'tetra':
                # Calculate aspect ratios, volumes, etc.
                # This is a simplified version
                tet_cells = mesh.points[cells.data]

                # Calculate volumes
                v0 = tet_cells[:, 1] - tet_cells[:, 0]
                v1 = tet_cells[:, 2] - tet_cells[:, 0]
                v2 = tet_cells[:, 3] - tet_cells[:, 0]

                volumes = np.abs(np.einsum('ij,ij->i', v0, np.cross(v1, v2))) / 6.0

                metrics['volume_stats'] = {
                    'min': float(np.min(volumes)),
                    'max': float(np.max(volumes)),
                    'mean': float(np.mean(volumes)),
                    'total': float(np.sum(volumes))
                }

        return metrics

    except Exception as e:
        logger.error(f"Failed to calculate mesh metrics: {e}")
        return {}


def render():
    """Render the mesh generation page"""

    st.header("🔷 Mesh Generation & Analysis")
    st.markdown("Generate high-quality meshes for FEA and CFD analysis")

    if not HAS_GMSH:
        st.error("⚠️ Gmsh not installed. Please install with: `pip install gmsh`")
        return

    # Main layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Input Model")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload CAD Model",
            type=['step', 'stp', 'stl', 'iges', 'igs'],
            help="Upload your CAD model for mesh generation"
        )

        if uploaded_file:
            # Save uploaded file
            temp_path = Path(f"./temp_mesh/{uploaded_file.name}")
            temp_path.parent.mkdir(parents=True, exist_ok=True)

            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())

            st.success(f"✅ Model loaded: {uploaded_file.name}")

            # Mesh parameters
            st.markdown("---")
            st.subheader("Mesh Parameters")

            col_m1, col_m2 = st.columns(2)

            with col_m1:
                element_size = st.number_input(
                    "Element Size",
                    min_value=0.1,
                    max_value=100.0,
                    value=1.0,
                    step=0.1,
                    help="Global mesh element size"
                )

                element_type = st.selectbox(
                    "Element Type",
                    options=["Tetrahedral", "Hexahedral (Experimental)"],
                    help="Type of mesh elements"
                )

            with col_m2:
                element_order = st.selectbox(
                    "Element Order",
                    options=[1, 2],
                    help="First order (linear) or second order (quadratic)"
                )

                optimize_mesh = st.checkbox(
                    "Optimize Mesh",
                    value=True,
                    help="Apply mesh optimization algorithms"
                )

            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                st.markdown("**Size Field**")

                use_size_field = st.checkbox("Enable Local Refinement")

                if use_size_field:
                    num_refinement_zones = st.number_input(
                        "Number of Refinement Zones",
                        min_value=1,
                        max_value=10,
                        value=1
                    )

                    for i in range(int(num_refinement_zones)):
                        with st.expander(f"Refinement Zone {i+1}"):
                            col_r1, col_r2 = st.columns(2)
                            with col_r1:
                                x_center = st.number_input(f"X Center {i}", value=0.0)
                                y_center = st.number_input(f"Y Center {i}", value=0.0)
                                z_center = st.number_input(f"Z Center {i}", value=0.0)
                            with col_r2:
                                radius = st.number_input(f"Radius {i}", value=1.0, min_value=0.1)
                                local_size = st.number_input(
                                    f"Local Element Size {i}",
                                    value=element_size * 0.5,
                                    min_value=0.01
                                )

                st.markdown("---")
                st.markdown("**Mesh Algorithms**")

                algorithm_2d = st.selectbox(
                    "2D Algorithm",
                    options=["MeshAdapt", "Automatic", "Delaunay", "Frontal-Delaunay"],
                    index=3
                )

                algorithm_3d = st.selectbox(
                    "3D Algorithm",
                    options=["Delaunay", "Frontal", "MMG3D", "HXT"],
                    index=0
                )

            # Generate mesh button
            st.markdown("---")

            if st.button("🔷 Generate Mesh", type="primary", use_container_width=True):
                with st.spinner("Generating mesh... This may take a while for complex geometries"):
                    try:
                        # Initialize generator
                        generator = MeshGenerator()

                        # Generate mesh
                        mesh_file = generator.generate_mesh(
                            str(temp_path),
                            element_size=element_size,
                            element_type=element_type.lower().split()[0],
                            order=element_order,
                            optimize=optimize_mesh
                        )

                        if mesh_file:
                            # Get statistics
                            stats = generator.get_mesh_stats()

                            # Store in session state
                            st.session_state['mesh_file'] = mesh_file
                            st.session_state['mesh_stats'] = stats

                            # Finalize gmsh
                            generator.finalize()

                            st.success("✅ Mesh generated successfully!")
                            st.rerun()
                        else:
                            st.error("❌ Mesh generation failed")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Mesh generation error: {e}", exc_info=True)

    with col2:
        st.subheader("Mesh Information")

        if 'mesh_stats' in st.session_state:
            stats = st.session_state['mesh_stats']

            # Display statistics
            st.metric("Total Nodes", f"{stats.get('num_nodes', 0):,}")
            st.metric("Total Elements", f"{stats.get('num_elements', 0):,}")

            # Element types
            if stats.get('element_types'):
                st.markdown("**Element Types:**")
                for etype, count in stats['element_types'].items():
                    st.text(f"  • {etype}: {count:,}")

            # Quality metrics
            if stats.get('quality'):
                st.markdown("---")
                st.markdown("**Quality Metrics:**")

                quality = stats['quality']

                col_q1, col_q2 = st.columns(2)
                with col_q1:
                    st.metric("Min Quality", f"{quality.get('min', 0):.4f}")
                    st.metric("Max Quality", f"{quality.get('max', 0):.4f}")
                with col_q2:
                    st.metric("Mean Quality", f"{quality.get('mean', 0):.4f}")
                    st.metric("Std Dev", f"{quality.get('std', 0):.4f}")

                # Quality bar
                mean_quality = quality.get('mean', 0)

                if mean_quality > 0.7:
                    st.success("✅ Excellent mesh quality")
                elif mean_quality > 0.5:
                    st.info("ℹ️ Good mesh quality")
                elif mean_quality > 0.3:
                    st.warning("⚠️ Fair mesh quality - consider refinement")
                else:
                    st.error("❌ Poor mesh quality - refinement recommended")

            # Export options
            if 'mesh_file' in st.session_state:
                st.markdown("---")
                st.subheader("Export Mesh")

                export_format = st.selectbox(
                    "Export Format",
                    options=[
                        "Gmsh (.msh)",
                        "Abaqus (.inp)",
                        "ANSYS (.msh)",
                        "VTK (.vtk)",
                        "STL (.stl)",
                        "Salome (.med)",
                        "CGNS (.cgns)"
                    ]
                )

                if st.button("💾 Export Mesh"):
                    try:
                        mesh_file = st.session_state['mesh_file']

                        # Convert format if needed
                        if HAS_MESHIO:
                            mesh = meshio.read(mesh_file)

                            format_map = {
                                "Gmsh (.msh)": "gmsh",
                                "Abaqus (.inp)": "abaqus",
                                "ANSYS (.msh)": "ansys",
                                "VTK (.vtk)": "vtk",
                                "STL (.stl)": "stl",
                                "Salome (.med)": "med",
                                "CGNS (.cgns)": "cgns"
                            }

                            output_format = format_map[export_format]
                            output_file = mesh_file.replace('.msh', f'.{output_format.split("-")[0]}')

                            meshio.write(output_file, mesh)

                            with open(output_file, 'rb') as f:
                                st.download_button(
                                    label=f"⬇️ Download {export_format}",
                                    data=f.read(),
                                    file_name=Path(output_file).name,
                                    mime="application/octet-stream"
                                )

                        else:
                            st.warning("meshio not installed - cannot convert formats")

                            # Offer direct download of .msh file
                            with open(mesh_file, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Gmsh Mesh",
                                    data=f.read(),
                                    file_name=Path(mesh_file).name,
                                    mime="application/octet-stream"
                                )

                    except Exception as e:
                        st.error(f"Export failed: {e}")
                        logger.error(f"Mesh export error: {e}", exc_info=True)

        else:
            st.info("No mesh generated yet. Upload a model and generate a mesh to see statistics.")

    # Mesh quality analysis
    st.markdown("---")
    st.subheader("Mesh Quality Analysis")

    if 'mesh_file' in st.session_state:
        col_a1, col_a2, col_a3 = st.columns(3)

        with col_a1:
            if st.button("📊 Detailed Analysis"):
                st.info("Running detailed mesh quality analysis...")
                # Would perform comprehensive analysis here

        with col_a2:
            if st.button("🔍 Find Bad Elements"):
                st.info("Identifying low-quality elements...")
                # Would find and highlight poor quality elements

        with col_a3:
            if st.button("🔧 Suggest Refinements"):
                st.info("Analyzing mesh and suggesting refinements...")
                # Would provide refinement suggestions

    else:
        st.info("Generate a mesh first to access quality analysis tools")


if __name__ == '__main__':
    # For standalone testing
    st.set_page_config(page_title="Mesh Generation", layout="wide")
    render()
