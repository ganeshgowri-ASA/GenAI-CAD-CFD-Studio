"""
GenAI CAD CFD Studio - Main Application
Main orchestrator for the Streamlit application with 6-tab architecture
"""

import streamlit as st
from src.ui.components.custom_css import apply_custom_css
from src.ui.components.sidebar import render_sidebar
from src.ui import design_studio, file_import, layout_generator, cfd_analysis, agent_config, project_history


def main():
    """Main application entry point"""

    # Page configuration
    st.set_page_config(
        page_title="GenAI CAD CFD Studio",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/GenAI-CAD-CFD-Studio',
            'Report a bug': 'https://github.com/yourusername/GenAI-CAD-CFD-Studio/issues',
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

    # Apply custom CSS styling
    apply_custom_css()

    # Render sidebar
    render_sidebar()

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

    # Tab container with 6 tabs
    tabs = st.tabs([
        "🎨 Design Studio",
        "📁 File Import",
        "🗺️ Layout Generator",
        "🌊 CFD Analysis",
        "⚙️ Agent Config",
        "📚 Project History"
    ])

    # Route to respective tab modules
    with tabs[0]:
        design_studio.render()

    with tabs[1]:
        file_import.render()

    with tabs[2]:
        layout_generator.render()

    with tabs[3]:
        cfd_analysis.render()

    with tabs[4]:
        agent_config.render()

    with tabs[5]:
        project_history.render()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6c757d; font-size: 12px; padding: 20px 0;'>
        <p>
            Built with ❤️ using Streamlit | Powered by OpenAI & Anthropic |
            <a href='https://github.com/yourusername/GenAI-CAD-CFD-Studio' style='color: #0066cc;'>
                GitHub
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
