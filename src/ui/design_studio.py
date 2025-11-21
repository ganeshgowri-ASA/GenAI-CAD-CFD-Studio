"""
AI Design Studio - Comprehensive CAD Model Generation Interface

Multi-modal CAD generation from:
- Natural language descriptions
- Reference photos and images
- Technical drawings (DXF, PDF)
- Hand-drawn sketches
- Hybrid inputs (text + image + specs)
"""

import streamlit as st
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile
import json

# Import CAD generation
try:
    from ..cad.model_generator import CADModelGenerator, CADGenerationResult
    from ..ai.claude_skills import ClaudeSkills
    from ..ai.dimension_extractor import DimensionExtractor
    HAS_CAD_GENERATOR = True
except ImportError as e:
    HAS_CAD_GENERATOR = False
    CAD_IMPORT_ERROR = str(e)

# Import UI components
try:
    from .components.viewer_3d import render_3d_viewer, render_model_measurements
    from .components.api_dashboard import (
        render_compact_api_metrics,
        render_model_selector_with_costs,
        render_export_options,
        render_cad_options,
        render_measurement_tools
    )
    HAS_UI_COMPONENTS = True
except ImportError as e:
    HAS_UI_COMPONENTS = False
    logging.warning(f"UI components not available: {e}")

# Import PDF exporter
try:
    from ..io.pdf_exporter import quick_export_pdf
    HAS_PDF_EXPORT = True
except ImportError:
    HAS_PDF_EXPORT = False
    logging.warning("PDF export not available")

# Import visualization
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def render():
    """Main render function for the AI Design Studio."""
    st.header("🎨 AI Design Studio")
    st.markdown("**Generate CAD models from text, images, sketches, and technical drawings**")

    # Check dependencies
    if not HAS_CAD_GENERATOR:
        st.error(f"""
        ❌ **CAD Generation modules not available**

        Error: {CAD_IMPORT_ERROR}

        Please ensure all dependencies are installed.
        """)
        return

    # Initialize session state
    if 'cad_generator' not in st.session_state:
        st.session_state.cad_generator = None
    if 'generation_result' not in st.session_state:
        st.session_state.generation_result = None
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

    # Sidebar configuration
    render_sidebar()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Text Input",
        "🖼️ Image/Sketch",
        "📐 Technical Drawing",
        "🔀 Hybrid Input",
        "📊 History"
    ])

    with tab1:
        render_text_input_tab()

    with tab2:
        render_image_input_tab()

    with tab3:
        render_drawing_input_tab()

    with tab4:
        render_hybrid_input_tab()

    with tab5:
        render_history_tab()


def render_sidebar():
    """Render sidebar configuration."""
    with st.sidebar:
        st.header("⚙️ Agent Config")

        # API Keys
        st.subheader("API Keys")
        st.caption("Configure API keys to enable CAD generation")

        claude_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Required for advanced NLP and vision analysis",
            key="claude_api_key_input"
        )

        zoo_api_key = st.text_input(
            "Zoo.dev API Key",
            type="password",
            help="Optional - for KCL-based CAD generation",
            key="zoo_api_key_input"
        )

        # Model selector with costs
        st.divider()
        if HAS_UI_COMPONENTS:
            selected_model = render_model_selector_with_costs()
            st.session_state['selected_claude_model'] = selected_model

        # Engine selection
        st.divider()
        st.subheader("CAD Engine")
        engine = st.selectbox(
            "Default Engine",
            ["build123d", "zoo", "auto"],
            index=2,
            help="CAD generation engine (auto = intelligent selection)"
        )

        # Units
        unit = st.selectbox(
            "Default Unit",
            ["mm", "cm", "m", "in", "ft"],
            index=0,
            help="Default unit for dimensions"
        )

        # CAD Options (2D/3D, Part/Assembly selectors)
        st.divider()
        if HAS_UI_COMPONENTS:
            cad_options = render_cad_options()
            st.session_state['cad_options'] = cad_options

        # Export options
        st.divider()
        st.subheader("Export Options")
        export_formats = st.multiselect(
            "Auto-export formats",
            ["step", "stl", "kcl", "dxf"],
            default=["step"],
            help="Automatically export generated models to these formats"
        )

        # Output directory
        output_dir = st.text_input(
            "Output Directory",
            value="./cad_output",
            help="Directory for generated files"
        )

        # Initialize generator button
        if st.button("🔧 Initialize Generator", type="primary", use_container_width=True):
            try:
                # Get API keys from environment if not provided
                final_claude_key = claude_api_key or os.getenv('ANTHROPIC_API_KEY')
                final_zoo_key = zoo_api_key or os.getenv('ZOO_API_KEY')

                # Initialize generator
                st.session_state.cad_generator = CADModelGenerator(
                    claude_api_key=final_claude_key,
                    zoo_api_key=final_zoo_key,
                    default_engine=engine if engine != 'auto' else 'build123d',
                    default_unit=unit,
                    output_dir=output_dir
                )

                st.session_state.engine = engine
                st.session_state.export_formats = export_formats

                st.success("✅ CAD Generator initialized successfully!")

            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")

        # Status
        st.divider()
        if st.session_state.cad_generator:
            st.success("✅ Generator Ready")
        else:
            st.warning("⚠️ Generator not initialized")

        # API Metrics (compact view)
        st.divider()
        if HAS_UI_COMPONENTS:
            render_compact_api_metrics()


def render_text_input_tab():
    """Render text input tab."""
    st.subheader("📝 Generate from Text Description")

    # Show API key info if generator not initialized
    if not st.session_state.cad_generator:
        st.info("💡 **Getting Started:** Configure your API keys in the sidebar (**⚙️ Agent Config**) to enable CAD generation")

    st.markdown("""
    Describe your CAD model in natural language. The AI will extract dimensions and generate the model.

    **Examples:**
    - "Create a box 100mm x 50mm x 30mm"
    - "Make a cylindrical pipe with 50mm diameter and 200mm length"
    - "Design a mounting bracket 150mm long, 80mm wide, 10mm thick"
    """)

    # Text input
    description = st.text_area(
        "Model Description",
        placeholder="Describe the CAD model you want to create...",
        height=150,
        key="text_description"
    )

    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            override_engine = st.selectbox(
                "Engine Override",
                ["auto", "build123d", "zoo"],
                help="Override default engine for this generation"
            )

        with col2:
            custom_unit = st.selectbox(
                "Unit Override",
                ["auto", "mm", "cm", "m", "in", "ft"],
                help="Override default unit"
            )

        material = st.text_input("Material (optional)", placeholder="e.g., Aluminum, Steel")
        notes = st.text_area("Additional Notes (optional)", height=80)

    # Generate button
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("🚀 Generate CAD Model", type="primary", use_container_width=True, key="gen_text"):
            generate_from_text(description, override_engine, material, notes)

    with col2:
        if st.button("🔍 Extract Parameters", use_container_width=True, key="extract_text"):
            extract_and_display_parameters(description)

    with col3:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_text"):
            st.session_state.generation_result = None
            st.rerun()

    # Display results
    display_generation_results()


def render_image_input_tab():
    """Render image/sketch input tab."""
    st.subheader("🖼️ Generate from Image or Sketch")

    # Show API key info if generator not initialized
    if not st.session_state.cad_generator:
        st.info("💡 **Getting Started:** Configure your API keys in the sidebar (**⚙️ Agent Config**) to enable CAD generation")

    st.markdown("""
    Upload a hand-drawn sketch, reference photo, or technical image.
    The AI will analyze edges, shapes, and dimensions.
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image/Sketch",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a sketch, photo, or technical drawing image",
        key="image_upload"
    )

    if uploaded_file:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        with col2:
            # Image type
            image_type = st.selectbox(
                "Image Type",
                ["sketch", "photo", "technical"],
                help="Type of image for better analysis"
            )

            # Supplementary description
            supplement_desc = st.text_area(
                "Supplementary Description (optional)",
                placeholder="e.g., 'Make it 100mm tall' or 'Add 5mm thickness'",
                height=100
            )

            # Engine selection
            override_engine = st.selectbox(
                "Engine",
                ["auto", "build123d", "zoo"],
                key="image_engine"
            )

        # Generate button
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("🚀 Generate from Image", type="primary", use_container_width=True, key="gen_image"):
                generate_from_image(uploaded_file, image_type, supplement_desc, override_engine)

        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_image"):
                st.session_state.generation_result = None
                st.rerun()

    # Display results
    display_generation_results()


def render_drawing_input_tab():
    """Render technical drawing input tab."""
    st.subheader("📐 Generate from Technical Drawing")

    # Show API key info if generator not initialized
    if not st.session_state.cad_generator:
        st.info("💡 **Getting Started:** Configure your API keys in the sidebar (**⚙️ Agent Config**) to enable CAD generation")

    st.markdown("""
    Upload technical drawings in DXF, DWG, or PDF format.
    The system will extract geometry and dimensions.
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Technical Drawing",
        type=['dxf', 'dwg', 'pdf'],
        help="Upload a technical drawing file",
        key="drawing_upload"
    )

    if uploaded_file:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        # Drawing format
        file_ext = uploaded_file.name.split('.')[-1].lower()
        st.info(f"Detected format: {file_ext.upper()}")

        # Supplementary description
        supplement_desc = st.text_area(
            "Supplementary Description (optional)",
            placeholder="Additional context or modifications...",
            height=100,
            key="drawing_desc"
        )

        # Engine selection
        override_engine = st.selectbox(
            "Engine",
            ["auto", "build123d"],
            key="drawing_engine"
        )

        # Generate button
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("🚀 Generate from Drawing", type="primary", use_container_width=True, key="gen_drawing"):
                generate_from_drawing(uploaded_file, file_ext, supplement_desc, override_engine)

        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_drawing"):
                st.session_state.generation_result = None
                st.rerun()

    # Display results
    display_generation_results()


def render_hybrid_input_tab():
    """Render hybrid input tab."""
    st.subheader("🔀 Generate from Multiple Inputs")

    # Show API key info if generator not initialized
    if not st.session_state.cad_generator:
        st.info("💡 **Getting Started:** Configure your API keys in the sidebar (**⚙️ Agent Config**) to enable CAD generation")

    st.markdown("""
    Combine multiple input sources for comprehensive CAD generation:
    - Text description + sketch
    - Technical drawing + specifications
    - Image + dimensions + material specs
    """)

    # Text input
    text_input = st.text_area(
        "Text Description",
        placeholder="Describe your model...",
        height=100,
        key="hybrid_text"
    )

    # Image input
    image_file = st.file_uploader(
        "Image/Sketch (optional)",
        type=['png', 'jpg', 'jpeg'],
        key="hybrid_image"
    )

    # Drawing input
    drawing_file = st.file_uploader(
        "Technical Drawing (optional)",
        type=['dxf', 'dwg', 'pdf'],
        key="hybrid_drawing"
    )

    # Specifications
    with st.expander("📋 Additional Specifications"):
        col1, col2 = st.columns(2)

        with col1:
            material_spec = st.text_input("Material", key="hybrid_material")
            thickness_spec = st.number_input("Thickness (mm)", min_value=0.0, key="hybrid_thickness")

        with col2:
            surface_finish = st.text_input("Surface Finish", key="hybrid_finish")
            tolerance = st.text_input("Tolerance", key="hybrid_tolerance")

        custom_specs = st.text_area(
            "Custom Specifications (JSON)",
            placeholder='{"feature": "value", "other": 123}',
            height=100,
            key="hybrid_specs_json"
        )

    # Engine selection
    override_engine = st.selectbox(
        "Engine",
        ["auto", "build123d", "zoo"],
        key="hybrid_engine"
    )

    # Generate button
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("🚀 Generate from Hybrid Inputs", type="primary", use_container_width=True, key="gen_hybrid"):
            generate_from_hybrid(
                text_input,
                image_file,
                drawing_file,
                material_spec,
                thickness_spec,
                custom_specs,
                override_engine
            )

    with col2:
        if st.button("🗑️ Clear", use_container_width=True, key="clear_hybrid"):
            st.session_state.generation_result = None
            st.rerun()

    # Display results
    display_generation_results()


def render_history_tab():
    """Render generation history tab."""
    st.subheader("📊 Generation History")

    if not st.session_state.generation_history:
        st.info("No generation history yet. Create some models to see them here!")
        return

    # Display history
    for idx, result in enumerate(reversed(st.session_state.generation_history)):
        with st.expander(f"🔹 Generation {len(st.session_state.generation_history) - idx} - {result.timestamp}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Status:** {'✅ Success' if result.success else '❌ Failed'}")
                st.write(f"**Message:** {result.message}")

                if result.parameters:
                    st.write("**Parameters:**")
                    st.json(result.parameters)

            with col2:
                if result.export_paths:
                    st.write("**Exported Files:**")
                    for fmt, path in result.export_paths.items():
                        st.write(f"- {fmt.upper()}: `{path}`")

                if result.metadata:
                    st.write("**Metadata:**")
                    st.json(result.metadata)


# ==================== Generation Functions ====================

def generate_from_text(description: str, engine: str, material: str, notes: str):
    """Generate CAD model from text description."""
    if not st.session_state.cad_generator:
        st.error("❌ **CAD Generator Not Initialized**")
        st.info("📝 **To enable CAD generation:**\n\n1. Open the sidebar (**⚙️ Agent Config**)\n2. Enter your **Anthropic API Key** (required)\n3. Optionally add **Zoo.dev API Key**\n4. Click **🔧 Initialize Generator**")
        return

    if not description:
        st.warning("⚠️ Please provide a description!")
        return

    with st.spinner("🔄 Generating CAD model from text..."):
        try:
            # Prepare parameters
            kwargs = {}
            if material:
                kwargs['material'] = material
            if notes:
                kwargs['notes'] = notes

            # Determine engine
            final_engine = None if engine == 'auto' else engine
            export_formats = st.session_state.get('export_formats', ['step'])

            # Generate
            result = st.session_state.cad_generator.generate_from_text(
                description=description,
                engine=final_engine,
                export_formats=export_formats,
                **kwargs
            )

            # Store result
            st.session_state.generation_result = result
            st.session_state.generation_history.append(result)

            if result.success:
                st.success(f"✅ {result.message}")
            else:
                st.error(f"❌ {result.message}")

        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")


def generate_from_image(uploaded_file, image_type: str, description: str, engine: str):
    """Generate CAD model from image."""
    if not st.session_state.cad_generator:
        st.error("❌ **CAD Generator Not Initialized**")
        st.info("📝 **To enable CAD generation:**\n\n1. Open the sidebar (**⚙️ Agent Config**)\n2. Enter your **Anthropic API Key** (required)\n3. Optionally add **Zoo.dev API Key**\n4. Click **🔧 Initialize Generator**")
        return

    with st.spinner("🔄 Analyzing image and generating CAD model..."):
        try:
            # Save uploaded file to temp location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Determine engine
            final_engine = None if engine == 'auto' else engine
            export_formats = st.session_state.get('export_formats', ['step'])

            # Generate
            result = st.session_state.cad_generator.generate_from_image(
                image_path=tmp_path,
                image_type=image_type,
                description=description if description else None,
                engine=final_engine,
                export_formats=export_formats
            )

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            # Store result
            st.session_state.generation_result = result
            st.session_state.generation_history.append(result)

            if result.success:
                st.success(f"✅ {result.message}")
            else:
                st.error(f"❌ {result.message}")

        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")


def generate_from_drawing(uploaded_file, file_format: str, description: str, engine: str):
    """Generate CAD model from technical drawing."""
    if not st.session_state.cad_generator:
        st.error("❌ **CAD Generator Not Initialized**")
        st.info("📝 **To enable CAD generation:**\n\n1. Open the sidebar (**⚙️ Agent Config**)\n2. Enter your **Anthropic API Key** (required)\n3. Optionally add **Zoo.dev API Key**\n4. Click **🔧 Initialize Generator**")
        return

    with st.spinner(f"🔄 Parsing {file_format.upper()} and generating CAD model..."):
        try:
            # Save uploaded file to temp location
            suffix = f'.{file_format}'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            # Determine engine
            final_engine = None if engine == 'auto' else engine
            export_formats = st.session_state.get('export_formats', ['step'])

            # Generate
            result = st.session_state.cad_generator.generate_from_drawing(
                drawing_path=tmp_path,
                drawing_format=file_format,
                description=description if description else None,
                engine=final_engine,
                export_formats=export_formats
            )

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            # Store result
            st.session_state.generation_result = result
            st.session_state.generation_history.append(result)

            if result.success:
                st.success(f"✅ {result.message}")
            else:
                st.error(f"❌ {result.message}")

        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")


def generate_from_hybrid(
    text: str,
    image_file,
    drawing_file,
    material: str,
    thickness: float,
    custom_specs: str,
    engine: str
):
    """Generate CAD model from hybrid inputs."""
    if not st.session_state.cad_generator:
        st.error("❌ **CAD Generator Not Initialized**")
        st.info("📝 **To enable CAD generation:**\n\n1. Open the sidebar (**⚙️ Agent Config**)\n2. Enter your **Anthropic API Key** (required)\n3. Optionally add **Zoo.dev API Key**\n4. Click **🔧 Initialize Generator**")
        return

    with st.spinner("🔄 Processing hybrid inputs and generating CAD model..."):
        try:
            # Prepare inputs dictionary
            inputs = {}

            if text:
                inputs['text'] = text

            # Handle image file
            if image_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    tmp_file.write(image_file.getvalue())
                    inputs['image'] = tmp_file.name

            # Handle drawing file
            if drawing_file:
                file_ext = drawing_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
                    tmp_file.write(drawing_file.getvalue())
                    inputs['drawing'] = tmp_file.name

            # Add specifications
            specs = {}
            if material:
                specs['material'] = material
            if thickness and thickness > 0:
                specs['thickness'] = thickness

            # Parse custom specs JSON
            if custom_specs:
                try:
                    custom = json.loads(custom_specs)
                    specs.update(custom)
                except json.JSONDecodeError:
                    st.warning("⚠️ Invalid JSON in custom specifications, ignoring...")

            if specs:
                inputs['specs'] = specs

            # Determine engine
            final_engine = None if engine == 'auto' else engine
            export_formats = st.session_state.get('export_formats', ['step'])

            # Generate
            result = st.session_state.cad_generator.generate_from_hybrid(
                inputs=inputs,
                engine=final_engine,
                export_formats=export_formats
            )

            # Clean up temp files
            if 'image' in inputs:
                Path(inputs['image']).unlink(missing_ok=True)
            if 'drawing' in inputs:
                Path(inputs['drawing']).unlink(missing_ok=True)

            # Store result
            st.session_state.generation_result = result
            st.session_state.generation_history.append(result)

            if result.success:
                st.success(f"✅ {result.message}")
            else:
                st.error(f"❌ {result.message}")

        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")


def extract_and_display_parameters(description: str):
    """Extract and display parameters without generating."""
    if not description:
        st.warning("⚠️ Please provide a description!")
        return

    try:
        extractor = DimensionExtractor()
        skills = ClaudeSkills()

        # Extract parameters
        dims = extractor.parse_dimensions(description)
        params = skills.extract_dimensions(description)

        # Display results
        st.subheader("📊 Extracted Parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Dimension Extractor:**")
            st.json(dims)

        with col2:
            st.write("**Claude Skills:**")
            st.json(params)

        # Validation
        if dims:
            is_valid = extractor.validate_dimensions(dims)
            if is_valid:
                st.success("✅ Dimensions are valid")
            else:
                st.warning("⚠️ Dimension validation issues detected")
                suggestions = extractor.suggest_corrections(dims)
                st.write("**Suggestions:**")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")

    except Exception as e:
        st.error(f"❌ Parameter extraction failed: {str(e)}")


def display_generation_results():
    """Display generation results if available."""
    if not st.session_state.generation_result:
        return

    result = st.session_state.generation_result

    st.divider()
    st.subheader("📊 Generation Results")

    # Status
    if result.success:
        st.success(f"✅ {result.message}")

        # Show fallback indicator if used
        if result.metadata.get('fallback'):
            st.info(f"ℹ️ **Fallback Used:** {result.metadata['fallback']}")
            if result.metadata.get('fallback_reason'):
                st.caption(f"Reason: {result.metadata['fallback_reason']}")
    else:
        st.error(f"❌ {result.message}")
        return

    # Create tabs for organized results display
    results_tabs = st.tabs([
        "📊 Overview",
        "👁️ 3D Viewer",
        "📏 Measurements",
        "📤 Export",
        "🔧 Details"
    ])

    # ========== OVERVIEW TAB ==========
    with results_tabs[0]:
        # Parameters
        if result.parameters:
            with st.expander("🔧 Extracted Parameters", expanded=True):
                # Display in a nicer format
                col1, col2, col3 = st.columns(3)
                params = result.parameters

                with col1:
                    if 'length' in params:
                        st.metric("Length", f"{params['length']} mm")
                    if 'object_type' in params or 'type' in params:
                        obj_type = params.get('object_type', params.get('type', 'N/A'))
                        st.write(f"**Type:** {obj_type}")

                with col2:
                    if 'width' in params:
                        st.metric("Width", f"{params['width']} mm")
                    if 'material' in params:
                        st.write(f"**Material:** {params['material']}")

                with col3:
                    if 'height' in params:
                        st.metric("Height", f"{params['height']} mm")
                    if 'radius' in params:
                        st.metric("Radius", f"{params['radius']} mm")

                # Show all parameters
                st.json(params)

        # KCL Code
        if result.kcl_code:
            with st.expander("📝 Generated KCL Code"):
                st.code(result.kcl_code, language='javascript')

                # Copy button
                if st.button("📋 Copy KCL Code", key="copy_kcl"):
                    st.info("Code copied to clipboard! (Feature requires browser support)")

        # Model URL
        if result.model_url:
            st.write(f"**🌐 Model URL:** [{result.model_url}]({result.model_url})")

    # ========== 3D VIEWER TAB ==========
    with results_tabs[1]:
        st.markdown("### 👁️ Interactive 3D Model Viewer")

        # Check if we have model data to visualize
        if result.part and HAS_UI_COMPONENTS:
            # Try to load STL file if available
            stl_path = result.export_paths.get('stl')
            if stl_path and Path(stl_path).exists():
                try:
                    render_3d_viewer(file_path=stl_path, title="Generated CAD Model")
                except Exception as e:
                    st.error(f"Failed to load 3D viewer: {e}")
                    st.info("The model was generated successfully but 3D visualization failed.")
            else:
                st.info("💡 **Tip:** Export to STL format to enable 3D visualization")
                st.write("Current exports:", list(result.export_paths.keys()))
        elif HAS_PLOTLY:
            st.info("3D viewer available - export your model to STL format for visualization")
        else:
            st.warning("3D viewer requires Plotly. Install with: `pip install plotly`")

    # ========== MEASUREMENTS TAB ==========
    with results_tabs[2]:
        st.markdown("### 📏 Model Measurements & Analysis")

        if HAS_UI_COMPONENTS:
            # Show measurement tools
            render_measurement_tools()

            # Show model measurements if we have STL data
            stl_path = result.export_paths.get('stl')
            if stl_path and Path(stl_path).exists():
                try:
                    # Load STL and show measurements
                    from .components.viewer_3d import load_stl_file
                    model_data = load_stl_file(stl_path)
                    if model_data:
                        render_model_measurements(model_data)
                except Exception as e:
                    st.warning(f"Could not load measurements: {e}")
            else:
                st.info("Export to STL format to see detailed measurements")
        else:
            st.info("Measurement tools not available - install required dependencies")

        # Show parameter-based dimensions
        if result.parameters:
            st.markdown("---")
            st.subheader("Parameter-Based Dimensions")
            params = result.parameters

            dimensions = {}
            for key in ['length', 'width', 'height', 'radius', 'diameter', 'thickness']:
                if key in params:
                    dimensions[key.capitalize()] = f"{params[key]} mm"

            if dimensions:
                cols = st.columns(len(dimensions))
                for col, (key, value) in zip(cols, dimensions.items()):
                    col.metric(key, value)

    # ========== EXPORT TAB ==========
    with results_tabs[3]:
        st.markdown("### 📤 Export Options")

        # Existing exports
        if result.export_paths:
            st.subheader("📁 Generated Files")

            for fmt, path in result.export_paths.items():
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**{fmt.upper()}**")
                    st.caption(f"`{Path(path).name}`")

                with col2:
                    if Path(path).exists():
                        file_size = Path(path).stat().st_size
                        st.caption(f"Size: {file_size / 1024:.1f} KB")
                    else:
                        st.caption("File not found")

                with col3:
                    # Download button
                    if Path(path).exists():
                        with open(path, 'rb') as f:
                            st.download_button(
                                label="⬇️",
                                data=f.read(),
                                file_name=Path(path).name,
                                mime='application/octet-stream',
                                key=f"download_{fmt}_{hash(path)}",
                                use_container_width=True
                            )

        # PDF Export
        st.divider()
        st.subheader("📄 PDF Technical Drawing")

        if HAS_PDF_EXPORT:
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write("Export a technical drawing with dimensions and title block")
                pdf_title = st.text_input(
                    "Drawing Title",
                    value="CAD Model",
                    key="pdf_title"
                )

            with col2:
                st.write("")  # Spacing
                st.write("")  # Spacing
                if st.button("📄 Export PDF", type="primary", use_container_width=True, key="export_pdf_btn"):
                    try:
                        # Create PDF export
                        output_path = Path(st.session_state.cad_generator.output_dir) / f"{pdf_title.replace(' ', '_')}.pdf"

                        # Get first image if available (for image-based generations)
                        image_path = None
                        if result.metadata.get('image_metadata'):
                            # Check if we have the original image
                            pass

                        success = quick_export_pdf(
                            output_path=str(output_path),
                            title=pdf_title,
                            dimensions=result.parameters,
                            description=result.message
                        )

                        if success:
                            st.success(f"✅ PDF exported: {output_path.name}")

                            # Offer download
                            with open(output_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download PDF",
                                    data=f.read(),
                                    file_name=output_path.name,
                                    mime='application/pdf',
                                    key="download_pdf"
                                )
                        else:
                            st.error("Failed to export PDF")

                    except Exception as e:
                        st.error(f"PDF export failed: {e}")
        else:
            st.warning("PDF export requires ReportLab. Install with: `pip install reportlab`")

    # ========== DETAILS TAB ==========
    with results_tabs[4]:
        st.markdown("### 🔧 Technical Details")

        # Metadata
        if result.metadata:
            st.subheader("ℹ️ Generation Metadata")

            # Show key metadata in metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                engine = result.metadata.get('engine', 'Unknown')
                st.metric("Engine", engine)

            with col2:
                gen_time = result.metadata.get('generation_time', 0)
                if gen_time:
                    st.metric("Generation Time", f"{gen_time:.2f}s")

            with col3:
                if result.metadata.get('fallback'):
                    st.metric("Fallback", "Yes")

            # Show full metadata
            with st.expander("View Full Metadata"):
                st.json(result.metadata)

        # Generation timestamp
        st.write(f"**Generated:** {result.timestamp}")

        # Image validation metadata if available
        if result.metadata.get('validation_metadata'):
            st.subheader("🖼️ Image Validation")
            validation_meta = result.metadata['validation_metadata']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Format:** {validation_meta.get('format', 'N/A')}")
                st.write(f"**Mode:** {validation_meta.get('mode', 'N/A')}")

            with col2:
                size = validation_meta.get('size', (0, 0))
                st.write(f"**Dimensions:** {size[0]}×{size[1]}")
                file_size = validation_meta.get('file_size', 0)
                st.write(f"**File Size:** {file_size / 1024:.1f} KB")

            with col3:
                quality = validation_meta.get('quality', {})
                if quality.get('quality_score'):
                    st.write(f"**Quality Score:** {quality['quality_score']:.2f}")

            with st.expander("View Full Validation Data"):
                st.json(validation_meta)


if __name__ == "__main__":
    st.set_page_config(
        page_title="AI Design Studio",
        page_icon="🎨",
        layout="wide"
    )
    render()
