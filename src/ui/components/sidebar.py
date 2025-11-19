"""
Sidebar Component for GenAI CAD CFD Studio
Displays logo, navigation info, session status, and credits
"""

import streamlit as st
from datetime import datetime


def render_sidebar():
    """Render the application sidebar with branding and info"""

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
