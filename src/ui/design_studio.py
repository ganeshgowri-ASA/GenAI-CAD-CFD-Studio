"""
AI Design Studio - Multi-modal CAD generation interface.

Provides comprehensive UI for generating CAD models from:
- Natural language descriptions
- Images/sketches
- Technical drawings
- Hybrid multi-modal inputs
"""

import streamlit as st
import sys
import os
from pathlib import Path
import traceback
import tempfile
import json

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def render():
    """Render the AI Design Studio interface."""

    st.header('🎨 AI Design Studio - Multi-Modal CAD Generation')

    # Check if API keys are configured
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    zoo_key = os.getenv('ZOO_API_KEY')

    if not anthropic_key:
        st.warning("⚠️ ANTHROPIC_API_KEY not set. Some features may be limited.")
        st.info("Set your API key with: `export ANTHROPIC_API_KEY=your_key_here`")

    # Main tabs for different input modes
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Text Description",
        "🖼️ Image/Sketch",
        "📐 Technical Drawing",
        "🔀 Hybrid (Multi-Modal)",
        "⚙️ Settings"
    ])

    # ========================================================================
    # Tab 1: Text-based CAD generation
    # ========================================================================
    with tab1:
        st.subheader("Generate CAD from Natural Language")

        st.markdown("""
        Describe your design in natural language. The AI will extract dimensions,
        shapes, and features to generate a 3D CAD model.

        **Example descriptions:**
        - "Create a box 100mm x 50mm x 30mm"
        - "Cylindrical rod with 20mm diameter and 150mm length"
        - "A mounting bracket 10cm long with 5mm thick walls and 8mm mounting holes"
        """)

        text_description = st.text_area(
            "Design Description:",
            placeholder="Describe your CAD model in detail...",
            height=150,
            key="text_desc"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            output_format = st.selectbox(
                "Output Format:",
                options=['step', 'stl', 'both'],
                index=0,
                key="text_format"
            )

        with col2:
            use_zoo = st.checkbox(
                "Use Zoo.dev KCL",
                value=False,
                help="Use Zoo.dev for KCL-based generation (requires API key)",
                key="text_zoo"
            )

        with col3:
            mock_mode = st.checkbox(
                "Mock Mode (Testing)",
                value=not anthropic_key,
                help="Use mock responses for testing without API",
                key="text_mock"
            )

        if st.button("🚀 Generate CAD Model", type="primary", key="gen_text"):
            if not text_description:
                st.error("Please provide a design description.")
            else:
                generate_from_text(
                    text_description,
                    output_format,
                    use_zoo,
                    mock_mode
                )

    # ========================================================================
    # Tab 2: Image/Sketch-based CAD generation
    # ========================================================================
    with tab2:
        st.subheader("Generate CAD from Image or Sketch")

        st.markdown("""
        Upload an image of a sketch, photo, or hand-drawn design. The AI will
        analyze the image to extract geometry and generate a CAD model.

        **Supported formats:** PNG, JPG, JPEG, GIF, WebP
        """)

        uploaded_image = st.file_uploader(
            "Upload Image:",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            key="image_upload"
        )

        if uploaded_image:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            with col2:
                image_type = st.selectbox(
                    "Image Type:",
                    options=['sketch', 'photo', 'drawing'],
                    index=0,
                    help="Type of image for optimized processing",
                    key="image_type"
                )

                additional_context = st.text_area(
                    "Additional Context (Optional):",
                    placeholder="Provide additional details about the image...",
                    height=100,
                    key="image_context"
                )

                output_format_img = st.selectbox(
                    "Output Format:",
                    options=['step', 'stl', 'both'],
                    index=0,
                    key="image_format"
                )

                mock_mode_img = st.checkbox(
                    "Mock Mode",
                    value=not anthropic_key,
                    key="image_mock"
                )

        if uploaded_image and st.button("🚀 Generate from Image", type="primary", key="gen_image"):
            generate_from_image(
                uploaded_image,
                image_type,
                additional_context if additional_context else None,
                output_format_img,
                mock_mode_img
            )

    # ========================================================================
    # Tab 3: Technical drawing-based CAD generation
    # ========================================================================
    with tab3:
        st.subheader("Generate CAD from Technical Drawing")

        st.markdown("""
        Upload a technical drawing file (DXF, DWG, or PDF). The system will
        parse the drawing and generate a 3D CAD model.

        **Supported formats:** DXF (DWG and PDF support coming soon)
        """)

        uploaded_drawing = st.file_uploader(
            "Upload Drawing:",
            type=['dxf'],  # 'dwg', 'pdf' coming soon
            key="drawing_upload"
        )

        if uploaded_drawing:
            drawing_format = st.selectbox(
                "Drawing Format:",
                options=['dxf'],  # 'dwg', 'pdf'
                index=0,
                key="drawing_format"
            )

            output_format_dwg = st.selectbox(
                "Output Format:",
                options=['step', 'stl', 'both'],
                index=0,
                key="drawing_output"
            )

        if uploaded_drawing and st.button("🚀 Generate from Drawing", type="primary", key="gen_drawing"):
            generate_from_drawing(
                uploaded_drawing,
                drawing_format,
                output_format_dwg
            )

    # ========================================================================
    # Tab 4: Hybrid multi-modal CAD generation
    # ========================================================================
    with tab4:
        st.subheader("Hybrid Multi-Modal CAD Generation")

        st.markdown("""
        Combine multiple input sources for comprehensive CAD generation:
        - Text description + reference image + technical drawing
        - Any combination of inputs

        The system will intelligently merge information from all sources.
        """)

        hybrid_text = st.text_area(
            "Text Description (Optional):",
            placeholder="Describe your design...",
            height=100,
            key="hybrid_text"
        )

        hybrid_image = st.file_uploader(
            "Upload Image (Optional):",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            key="hybrid_image"
        )

        hybrid_drawing = st.file_uploader(
            "Upload Drawing (Optional):",
            type=['dxf'],
            key="hybrid_drawing"
        )

        # Additional specifications
        with st.expander("➕ Additional Specifications"):
            spec_material = st.text_input("Material:", placeholder="e.g., Aluminum, Steel", key="spec_mat")
            spec_thickness = st.text_input("Wall Thickness:", placeholder="e.g., 5mm", key="spec_thick")
            spec_tolerance = st.text_input("Tolerance:", placeholder="e.g., ±0.1mm", key="spec_tol")
            spec_custom = st.text_area("Custom Properties (JSON):", placeholder='{"key": "value"}', key="spec_custom")

        col1, col2 = st.columns(2)
        with col1:
            hybrid_format = st.selectbox(
                "Output Format:",
                options=['step', 'stl', 'both'],
                index=0,
                key="hybrid_format"
            )

        with col2:
            hybrid_mock = st.checkbox(
                "Mock Mode",
                value=not anthropic_key,
                key="hybrid_mock"
            )

        if st.button("🚀 Generate Hybrid CAD Model", type="primary", key="gen_hybrid"):
            if not any([hybrid_text, hybrid_image, hybrid_drawing]):
                st.error("Please provide at least one input source.")
            else:
                # Build specifications dict
                specs = {}
                if spec_material:
                    specs['material'] = spec_material
                if spec_thickness:
                    specs['thickness'] = spec_thickness
                if spec_tolerance:
                    specs['tolerance'] = spec_tolerance
                if spec_custom:
                    try:
                        custom = json.loads(spec_custom)
                        specs.update(custom)
                    except json.JSONDecodeError:
                        st.warning("Custom properties JSON is invalid, skipping.")

                generate_from_hybrid(
                    text_description=hybrid_text if hybrid_text else None,
                    image_file=hybrid_image,
                    drawing_file=hybrid_drawing,
                    specifications=specs if specs else None,
                    output_format=hybrid_format,
                    mock_mode=hybrid_mock
                )

    # ========================================================================
    # Tab 5: Settings
    # ========================================================================
    with tab5:
        st.subheader("⚙️ Settings & Configuration")

        st.markdown("### API Configuration")

        st.code(f"""
# Environment Variables
ANTHROPIC_API_KEY = {"✓ Set" if anthropic_key else "✗ Not Set"}
ZOO_API_KEY = {"✓ Set" if zoo_key else "✗ Not Set"}
        """)

        st.markdown("### Model Selection")
        st.info("""
        **Build123d Engine:**
        - Direct Python-based CAD modeling
        - Full control over geometry
        - Export to STEP, STL

        **Zoo.dev KCL Engine:**
        - Text-to-CAD using KCL language
        - AI-generated parametric models
        - Requires Zoo.dev API key
        """)

        st.markdown("### Output Directories")
        output_dir = Path("outputs/cad")
        st.code(f"Output Directory: {output_dir.absolute()}")

        if st.button("Create Output Directory"):
            output_dir.mkdir(parents=True, exist_ok=True)
            st.success(f"Created: {output_dir.absolute()}")

        st.markdown("### System Information")
        st.code(f"""
Python Version: {sys.version.split()[0]}
Streamlit Version: {st.__version__}
        """)


# ============================================================================
# Generation Functions
# ============================================================================

def generate_from_text(description: str, output_format: str, use_zoo: bool, mock_mode: bool):
    """Generate CAD model from text description."""

    with st.spinner("🔄 Generating CAD model from text..."):
        try:
            from cad.model_generator import CADModelGenerator

            # Initialize generator
            generator = CADModelGenerator(
                use_zoo_dev=use_zoo,
                mock_mode=mock_mode
            )

            # Generate
            result = generator.generate_from_text(
                description=description,
                output_format=output_format
            )

            # Display results
            display_generation_result(result)

        except Exception as e:
            st.error(f"❌ Error generating CAD model: {str(e)}")
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())


def generate_from_image(image_file, image_type: str, additional_context: str, output_format: str, mock_mode: bool):
    """Generate CAD model from image."""

    with st.spinner("🔄 Analyzing image and generating CAD model..."):
        try:
            from cad.model_generator import CADModelGenerator

            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp:
                tmp.write(image_file.read())
                tmp_path = tmp.name

            # Initialize generator
            generator = CADModelGenerator(mock_mode=mock_mode)

            # Generate
            result = generator.generate_from_image(
                image_path=tmp_path,
                image_type=image_type,
                additional_context=additional_context,
                output_format=output_format
            )

            # Clean up temp file
            os.unlink(tmp_path)

            # Display results
            display_generation_result(result)

        except Exception as e:
            st.error(f"❌ Error generating CAD model: {str(e)}")
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())


def generate_from_drawing(drawing_file, drawing_format: str, output_format: str):
    """Generate CAD model from technical drawing."""

    with st.spinner("🔄 Parsing drawing and generating CAD model..."):
        try:
            from cad.model_generator import CADModelGenerator

            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{drawing_format}') as tmp:
                tmp.write(drawing_file.read())
                tmp_path = tmp.name

            # Initialize generator
            generator = CADModelGenerator()

            # Generate
            result = generator.generate_from_drawing(
                drawing_path=tmp_path,
                drawing_format=drawing_format,
                output_format=output_format
            )

            # Clean up temp file
            os.unlink(tmp_path)

            # Display results
            display_generation_result(result)

        except Exception as e:
            st.error(f"❌ Error generating CAD model: {str(e)}")
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())


def generate_from_hybrid(
    text_description: str,
    image_file,
    drawing_file,
    specifications: dict,
    output_format: str,
    mock_mode: bool
):
    """Generate CAD model from hybrid multi-modal inputs."""

    with st.spinner("🔄 Processing multi-modal inputs and generating CAD model..."):
        try:
            from cad.model_generator import CADModelGenerator

            # Save uploaded files to temp locations
            image_path = None
            drawing_path = None

            if image_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image_file.name).suffix) as tmp:
                    tmp.write(image_file.read())
                    image_path = tmp.name

            if drawing_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.dxf') as tmp:
                    tmp.write(drawing_file.read())
                    drawing_path = tmp.name

            # Initialize generator
            generator = CADModelGenerator(mock_mode=mock_mode)

            # Generate
            result = generator.generate_from_hybrid(
                text_description=text_description,
                image_path=image_path,
                drawing_path=drawing_path,
                specifications=specifications,
                output_format=output_format
            )

            # Clean up temp files
            if image_path:
                os.unlink(image_path)
            if drawing_path:
                os.unlink(drawing_path)

            # Display results
            display_generation_result(result)

        except Exception as e:
            st.error(f"❌ Error generating CAD model: {str(e)}")
            with st.expander("Show Error Details"):
                st.code(traceback.format_exc())


def display_generation_result(result: dict):
    """Display CAD generation results."""

    st.success("✅ CAD model generated successfully!")

    # Show parameters
    with st.expander("📊 Extracted Parameters", expanded=True):
        params = result.get('parameters', {})
        st.json(params)

    # Show generated files
    files = result.get('files', [])
    if files:
        st.markdown("### 📁 Generated Files")
        for file_path in files:
            file_path = Path(file_path)
            if file_path.exists():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file_path.name}")
                    st.caption(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
                with col2:
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download",
                            data=f.read(),
                            file_name=file_path.name,
                            mime='application/octet-stream'
                        )

    # Show metadata
    with st.expander("ℹ️ Generation Metadata"):
        metadata = result.get('metadata', {})
        st.json(metadata)

    # Show additional info based on input type
    input_type = result.get('input_type')

    if input_type == 'image':
        with st.expander("🖼️ Image Analysis Results"):
            geometry = result.get('detected_geometry', {})
            st.json(geometry)

    elif input_type == 'drawing':
        with st.expander("📐 Parsed Drawing Data"):
            parsed = result.get('parsed_geometry', {})
            st.json(parsed)

    elif input_type == 'hybrid':
        with st.expander("🔀 Multi-Modal Processing"):
            st.write("**Input Sources:**", result.get('parameter_sources', []))
            st.write("**Merged Parameters:**")
            st.json(result.get('parameters', {}))

    # Preview (if available)
    if result.get('model'):
        st.markdown("### 👁️ 3D Preview")
        st.info("3D preview will be available in the next update!")


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Design Studio",
        page_icon="🎨",
        layout="wide"
    )
    render()
