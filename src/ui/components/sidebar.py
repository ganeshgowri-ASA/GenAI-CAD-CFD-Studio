"""
Sidebar Component for GenAI CAD CFD Studio
Displays logo, navigation info, session status, credits, and enhanced features
including API monitoring, model selection, and CAD options
"""

import streamlit as st
from datetime import datetime
from typing import Optional

# Import API monitoring components
try:
    from .api_dashboard import (
        render_compact_api_metrics,
        render_model_selector_with_costs,
        render_export_options,
        render_cad_options,
        render_measurement_tools
    )
    from ...utils.api_monitor import APIMonitor, get_global_monitor
    HAS_API_MONITOR = True
except ImportError:
    HAS_API_MONITOR = False


def render_sidebar(
    show_api_metrics: bool = True,
    show_model_selector: bool = True,
    show_cad_options: bool = True,
    show_export_options: bool = True,
    show_measurement_tools: bool = True,
    api_monitor: Optional[APIMonitor] = None
):
    """
    Render the enhanced application sidebar with branding, info, and features.

    Args:
        show_api_metrics: Show API monitoring metrics
        show_model_selector: Show model selection with costs
        show_cad_options: Show CAD generation options
        show_export_options: Show export format options
        show_measurement_tools: Show measurement tools
        api_monitor: Custom APIMonitor instance (uses global if None)
    """
    with st.sidebar:
        # Logo and Title
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: white; font-size: 28px; margin: 0;'>
                🏗️ GenAI CAD
            </h1>
            <h2 style='color: rgba(255,255,255,0.9); font-size: 20px; margin: 5px 0;'>
                CFD Studio
            </h2>
            <p style='color: rgba(255,255,255,0.7); font-size: 12px; margin-top: 5px;'>
                AI-Powered Design & Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Navigation Info
        st.markdown("""
        <div style='color: white;'>
            <h3 style='color: white; font-size: 18px;'>📋 Navigation</h3>
            <ul style='color: rgba(255,255,255,0.9); font-size: 14px; line-height: 1.8;'>
                <li>🎨 <b>Design Studio</b><br/>
                    <span style='font-size: 12px; color: rgba(255,255,255,0.7);'>
                    Generate CAD from text
                    </span>
                </li>
                <li>📁 <b>File Import</b><br/>
                    <span style='font-size: 12px; color: rgba(255,255,255,0.7);'>
                    Import & convert files
                    </span>
                </li>
                <li>🗺️ <b>Layout Generator</b><br/>
                    <span style='font-size: 12px; color: rgba(255,255,255,0.7);'>
                    Geospatial design tools
                    </span>
                </li>
                <li>🌊 <b>CFD Analysis</b><br/>
                    <span style='font-size: 12px; color: rgba(255,255,255,0.7);'>
                    Fluid dynamics simulation
                    </span>
                </li>
                <li>⚙️ <b>Agent Config</b><br/>
                    <span style='font-size: 12px; color: rgba(255,255,255,0.7);'>
                    API keys & settings
                    </span>
                </li>
                <li>📚 <b>Project History</b><br/>
                    <span style='font-size: 12px; color: rgba(255,255,255,0.7);'>
                    Version control & Git
                    </span>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Session Status
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()

        session_duration = datetime.now() - st.session_state.session_start
        hours, remainder = divmod(session_duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        st.markdown(f"""
        <div style='color: white;'>
            <h3 style='color: white; font-size: 18px;'>📊 Session Status</h3>
            <p style='color: rgba(255,255,255,0.9); font-size: 14px;'>
                <b>Status:</b> <span style='color: #00cc66;'>● Active</span><br/>
                <b>Duration:</b> {hours:02d}:{minutes:02d}:{seconds:02d}<br/>
                <b>Started:</b> {st.session_state.session_start.strftime('%H:%M:%S')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Credits and Info
        st.markdown("""
        <div style='color: rgba(255,255,255,0.8); font-size: 12px; text-align: center; padding: 20px 0;'>
            <p style='margin: 5px 0;'>
                <b>Version:</b> 1.0.0<br/>
                <b>Framework:</b> Streamlit<br/>
                <b>AI Engine:</b> OpenAI + Anthropic
            </p>
            <p style='margin-top: 15px; color: rgba(255,255,255,0.6);'>
                © 2024 GenAI CAD CFD Studio<br/>
                All Rights Reserved
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Quick Stats (if available in session state)
        if hasattr(st.session_state, 'project_count'):
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Projects", st.session_state.get('project_count', 0))
            with col2:
                st.metric("Designs", st.session_state.get('design_count', 0))
            with col3:
                st.metric("Analyses", st.session_state.get('analysis_count', 0))

        # Enhanced Features Section
        st.markdown("---")

        # API Metrics (compact view)
        if show_api_metrics and HAS_API_MONITOR:
            try:
                monitor = api_monitor or get_global_monitor()
                render_compact_api_metrics(monitor)
                st.markdown("---")
            except Exception as e:
                st.caption(f"API metrics unavailable: {e}")

        # Model Selector
        if show_model_selector:
            try:
                with st.expander("🤖 Model Selection", expanded=False):
                    selected_model = render_model_selector_with_costs()
                    # Store in session state for use in generation
                    st.session_state['selected_model'] = selected_model
            except Exception as e:
                st.caption(f"Model selector unavailable: {e}")

        # CAD Options
        if show_cad_options:
            try:
                with st.expander("⚙️ CAD Options", expanded=False):
                    cad_settings = render_cad_options()
                    # Store in session state
                    st.session_state['cad_settings'] = cad_settings
            except Exception as e:
                st.caption(f"CAD options unavailable: {e}")

        # Export Options
        if show_export_options:
            try:
                with st.expander("📤 Export Options", expanded=False):
                    export_settings = render_export_options()
                    # Store in session state
                    st.session_state['export_settings'] = export_settings
            except Exception as e:
                st.caption(f"Export options unavailable: {e}")

        # Measurement Tools
        if show_measurement_tools:
            try:
                with st.expander("📏 Measurement Tools", expanded=False):
                    render_measurement_tools()
            except Exception as e:
                st.caption(f"Measurement tools unavailable: {e}")
