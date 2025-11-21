"""
GenAI CAD CFD Studio - Main Application
Defensive version with graceful error handling for Streamlit Cloud deployment
"""

import streamlit as st

# Import custom components with defensive error handling
custom_css_available = False
sidebar_available = False

try:
    from src.ui.components.custom_css import apply_custom_css
    custom_css_available = True
except ImportError as e:
    st.warning(f"Custom CSS component not available: {e}")

try:
    from src.ui.components.sidebar import render_sidebar
    sidebar_available = True
except ImportError as e:
    st.warning(f"Sidebar component not available: {e}")

# Import tab modules with defensive error handling
design_studio_available = False
file_import_available = False
layout_generator_available = False
cfd_analysis_available = False
agent_config_available = False
project_history_available = False
rendering_available = False
mesh_generation_available = False
cfd_analysis_new_available = False

try:
    from src.ui import design_studio
    design_studio_available = True
except ImportError as e:
    st.warning(f"Design Studio module not available: {e}")

try:
    from src.ui import file_import
    file_import_available = True
except ImportError as e:
    st.warning(f"File Import module not available: {e}")

try:
    from src.ui import layout_generator
    layout_generator_available = True
except ImportError as e:
    st.warning(f"Layout Generator module not available: {e}")

try:
    from src.ui import cfd_analysis
    cfd_analysis_available = True
except ImportError as e:
    st.warning(f"CFD Analysis module not available: {e}")

try:
    from src.ui import agent_config
    agent_config_available = True
except ImportError as e:
    st.warning(f"Agent Config module not available: {e}")

try:
    from src.ui import project_history
    project_history_available = True
except ImportError as e:
    st.warning(f"Project History module not available: {e}")

# Import new page modules
try:
    from src.ui.pages import rendering
    rendering_available = True
except ImportError as e:
    st.warning(f"Rendering module not available: {e}")

try:
    from src.ui.pages import mesh_generation
    mesh_generation_available = True
except ImportError as e:
    st.warning(f"Mesh Generation module not available: {e}")

try:
    from src.ui.pages import cfd_analysis as cfd_analysis_new
    cfd_analysis_new_available = True
except ImportError as e:
    st.warning(f"CFD Analysis (new) module not available: {e}")


def render_fallback_tab(tab_name: str):
    """Render a fallback message when a tab module is unavailable"""
    st.header(f"{tab_name}")
    st.error(f"""
    ⚠️ **{tab_name} module is currently unavailable**

    This may be due to missing dependencies or import errors.
    Please check the application logs for more details.
    """)
    st.info("""
    **Troubleshooting Tips:**
    - Ensure all required dependencies are installed
    - Check that all module files are present
    - Review error messages above for specific issues
    """)


def main():
    """Main application entry point with defensive error handling"""

    # Page configuration
    try:
        st.set_page_config(
            page_title="GenAI CAD CFD Studio",
            page_icon="🏗️",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio',
                'Report a bug': 'https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio/issues',
                'About': """
                # GenAI CAD CFD Studio

                AI-powered platform for CAD design and CFD analysis.

                **Features:**
                - Natural language CAD generation
                - File import and conversion
                - Geospatial layout design
                - CFD simulation
                - Agent configuration
                - Project history and version control

                Version 1.0.0
                """
            }
        )
    except Exception as e:
        st.error(f"Error setting page config: {e}")

    # Apply custom CSS styling if available
    if custom_css_available:
        try:
            apply_custom_css()
        except Exception as e:
            st.warning(f"Error applying custom CSS: {e}")

    # Render enhanced sidebar with API monitoring and model selection
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Model selector
        with st.expander("🤖 AI Model Selection", expanded=False):
            model_choice = st.selectbox(
                "Claude Model",
                options=[
                    "claude-3-opus-20240229",
                    "claude-3-5-sonnet-20241022",
                    "claude-3-haiku-20240307"
                ],
                index=1,
                help="Select Claude model for CAD generation"
            )
            st.session_state['selected_model'] = model_choice

        # API Configuration
        with st.expander("🔑 API Keys", expanded=False):
            claude_key = st.text_input("Claude API Key", type="password", key="claude_api_key_input")
            zoo_key = st.text_input("Zoo.dev API Key", type="password", key="zoo_api_key_input")
            adam_key = st.text_input("Adam.new API Key", type="password", key="adam_api_key_input")

            if st.button("Save API Keys"):
                if claude_key:
                    st.session_state['claude_api_key'] = claude_key
                if zoo_key:
                    st.session_state['zoo_api_key'] = zoo_key
                if adam_key:
                    st.session_state['adam_api_key'] = adam_key
                st.success("API keys saved!")

        # API Consumption Dashboard
        with st.expander("📊 API Usage Dashboard", expanded=False):
            try:
                from src.utils.api_monitor import get_monitor, APIService

                monitor = get_monitor(create_if_missing=True)

                if monitor:
                    st.markdown("**Overall Statistics**")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Calls", monitor.get_total_calls())
                        st.metric("Success Rate", f"{monitor.get_overall_success_rate():.1f}%")
                    with col2:
                        st.metric("Total Cost", f"${monitor.get_total_cost():.2f}")

                    # Service breakdown
                    st.markdown("---")
                    st.markdown("**By Service:**")

                    for service in APIService:
                        metrics = monitor.get_service_metrics(service)
                        if metrics.total_calls > 0:
                            with st.container():
                                st.markdown(f"**{service.value}**")
                                col_s1, col_s2, col_s3 = st.columns(3)
                                with col_s1:
                                    st.text(f"Calls: {metrics.total_calls}")
                                with col_s2:
                                    st.text(f"Success: {metrics.success_rate():.1f}%")
                                with col_s3:
                                    st.text(f"Cost: ${metrics.total_cost_usd:.2f}")

                    if st.button("Clear Statistics"):
                        monitor.clear_session_data()
                        st.rerun()

                else:
                    st.info("No API usage data yet")

            except Exception as e:
                st.error(f"API monitor error: {e}")

        # Export Options
        with st.expander("💾 Export Options", expanded=False):
            st.markdown("**Default Export Formats:**")

            export_step = st.checkbox("STEP (.step)", value=True)
            export_stl = st.checkbox("STL (.stl)", value=True)
            export_pdf = st.checkbox("PDF Drawing", value=False)
            export_dxf = st.checkbox("DXF (.dxf)", value=False)

            # Store in session state
            st.session_state['export_formats'] = {
                'step': export_step,
                'stl': export_stl,
                'pdf': export_pdf,
                'dxf': export_dxf
            }

        # Render original sidebar if available
        if sidebar_available:
            try:
                render_sidebar()
            except Exception as e:
                st.warning(f"Error rendering sidebar: {e}")

    # Main header
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #0066cc; font-size: 3rem; margin: 0;'>
            🏗️ GenAI CAD CFD Studio
        </h1>
        <p style='color: #6c757d; font-size: 1.2rem; margin-top: 10px;'>
            AI-Powered CAD Design & Computational Fluid Dynamics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Show deployment status
    modules_status = {
        "Design Studio": design_studio_available,
        "File Import": file_import_available,
        "Layout Generator": layout_generator_available,
        "CFD Analysis": cfd_analysis_available or cfd_analysis_new_available,
        "Agent Config": agent_config_available,
        "Project History": project_history_available,
        "Rendering": rendering_available,
        "Mesh Generation": mesh_generation_available
    }

    available_count = sum(modules_status.values())
    total_count = len(modules_status)

    if available_count < total_count:
        with st.expander(f"⚠️ Module Status: {available_count}/{total_count} available", expanded=False):
            for module_name, available in modules_status.items():
                if available:
                    st.success(f"✅ {module_name} - Available")
                else:
                    st.error(f"❌ {module_name} - Unavailable")

    # Tab container with 9 tabs
    tabs = st.tabs([
        "🎨 Design Studio",
        "📁 File Import",
        "🗺️ Layout Generator",
        "🔷 Mesh Generation",
        "🌊 CFD Analysis",
        "🖼️ Rendering",
        "⚙️ Agent Config",
        "📚 Project History"
    ])

    # Route to respective tab modules with error handling
    with tabs[0]:
        if design_studio_available:
            try:
                design_studio.render()
            except Exception as e:
                st.error(f"Error rendering Design Studio: {e}")
                render_fallback_tab("Design Studio")
        else:
            render_fallback_tab("Design Studio")

    with tabs[1]:
        if file_import_available:
            try:
                file_import.render()
            except Exception as e:
                st.error(f"Error rendering File Import: {e}")
                render_fallback_tab("File Import")
        else:
            render_fallback_tab("File Import")

    with tabs[2]:
        if layout_generator_available:
            try:
                layout_generator.render()
            except Exception as e:
                st.error(f"Error rendering Layout Generator: {e}")
                render_fallback_tab("Layout Generator")
        else:
            render_fallback_tab("Layout Generator")

    with tabs[3]:
        if mesh_generation_available:
            try:
                mesh_generation.render()
            except Exception as e:
                st.error(f"Error rendering Mesh Generation: {e}")
                render_fallback_tab("Mesh Generation")
        else:
            render_fallback_tab("Mesh Generation")

    with tabs[4]:
        # Use new CFD analysis if available, otherwise fall back to old one
        if cfd_analysis_new_available:
            try:
                cfd_analysis_new.render()
            except Exception as e:
                st.error(f"Error rendering CFD Analysis: {e}")
                render_fallback_tab("CFD Analysis")
        elif cfd_analysis_available:
            try:
                cfd_analysis.render()
            except Exception as e:
                st.error(f"Error rendering CFD Analysis: {e}")
                render_fallback_tab("CFD Analysis")
        else:
            render_fallback_tab("CFD Analysis")

    with tabs[5]:
        if rendering_available:
            try:
                rendering.render()
            except Exception as e:
                st.error(f"Error rendering Rendering: {e}")
                render_fallback_tab("Rendering")
        else:
            render_fallback_tab("Rendering")

    with tabs[6]:
        if agent_config_available:
            try:
                agent_config.render()
            except Exception as e:
                st.error(f"Error rendering Agent Config: {e}")
                render_fallback_tab("Agent Config")
        else:
            render_fallback_tab("Agent Config")

    with tabs[7]:
        if project_history_available:
            try:
                project_history.render()
            except Exception as e:
                st.error(f"Error rendering Project History: {e}")
                render_fallback_tab("Project History")
        else:
            render_fallback_tab("Project History")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 12px; padding: 20px 0;'>
        <p>
            Built with ❤️ using Streamlit | Powered by AI |
            <a href='https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio' style='color: #0066cc;'>
                GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
