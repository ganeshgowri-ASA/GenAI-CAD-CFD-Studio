"""
API Metrics Dashboard Component

Provides real-time visualization of API usage, costs, and performance metrics.

Features:
- Real-time metrics display
- Cost tracking and breakdown
- Provider usage statistics
- Performance charts
- Export functionality

Author: GenAI CAD CFD Studio
Version: 1.0.0
"""

import streamlit as st
from typing import Optional
from datetime import timedelta
import pandas as pd
from pathlib import Path

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ...utils.api_monitor import APIMonitor, APIProvider, APICallStatus, get_global_monitor


def render_api_dashboard(monitor: Optional[APIMonitor] = None) -> None:
    """
    Render comprehensive API metrics dashboard.

    Args:
        monitor: APIMonitor instance (uses global if not provided)
    """
    if monitor is None:
        monitor = get_global_monitor()

    st.subheader("📊 API Usage Dashboard")

    # Get summary stats
    stats = monitor.get_summary_stats()

    if stats['total_calls'] == 0:
        st.info("No API calls recorded yet. Start generating CAD models to see metrics!")
        return

    # Top-level metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total API Calls",
            stats['total_calls'],
            delta=None
        )

    with col2:
        st.metric(
            "Success Rate",
            f"{stats['success_rate']*100:.1f}%",
            delta=None
        )

    with col3:
        st.metric(
            "Total Tokens",
            f"{stats['total_tokens']:,}",
            delta=None
        )

    with col4:
        st.metric(
            "Total Cost",
            f"${stats['total_cost_usd']:.4f}",
            delta=None
        )

    st.markdown("---")

    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Provider usage chart
        st.subheader("🔌 Provider Usage")
        provider_data = stats['by_provider']

        if provider_data and HAS_PLOTLY:
            provider_names = list(provider_data.keys())
            provider_calls = [provider_data[p]['calls'] for p in provider_names]

            fig = px.pie(
                values=provider_calls,
                names=provider_names,
                title="API Calls by Provider",
                hole=0.3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        elif provider_data:
            # Fallback to simple display
            for provider, data in provider_data.items():
                st.write(f"**{provider}**: {data['calls']} calls, ${data['cost_usd']:.4f}")
        else:
            st.info("No provider data available")

    with chart_col2:
        # Status breakdown chart
        st.subheader("✅ Call Status")
        status_data = stats['by_status']

        if status_data and HAS_PLOTLY:
            status_names = list(status_data.keys())
            status_counts = list(status_data.values())

            # Color mapping for statuses
            colors = {
                'success': '#28a745',
                'failed': '#dc3545',
                'timeout': '#ffc107',
                'payment_required': '#fd7e14',
                'rate_limited': '#6c757d'
            }
            status_colors = [colors.get(s, '#6c757d') for s in status_names]

            fig = go.Figure(data=[go.Bar(
                x=status_names,
                y=status_counts,
                marker_color=status_colors
            )])
            fig.update_layout(
                title="API Calls by Status",
                xaxis_title="Status",
                yaxis_title="Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif status_data:
            # Fallback to simple display
            for status, count in status_data.items():
                st.write(f"**{status}**: {count} calls")
        else:
            st.info("No status data available")

    st.markdown("---")

    # Detailed provider statistics
    st.subheader("📈 Detailed Provider Statistics")

    if provider_data:
        # Create DataFrame for detailed view
        provider_df_data = []
        for provider, data in provider_data.items():
            provider_df_data.append({
                'Provider': provider,
                'Calls': data['calls'],
                'Tokens': data['tokens'],
                'Avg Duration (ms)': f"{data['avg_duration_ms']:.0f}",
                'Cost (USD)': f"${data['cost_usd']:.4f}"
            })

        provider_df = pd.DataFrame(provider_df_data)
        st.dataframe(provider_df, use_container_width=True, hide_index=True)
    else:
        st.info("No detailed provider statistics available")

    st.markdown("---")

    # Cost breakdown
    st.subheader("💰 Cost Breakdown")

    cost_breakdown = monitor.get_cost_breakdown()
    if cost_breakdown:
        cost_df_data = []
        for provider, models in cost_breakdown.items():
            for model, cost in models.items():
                cost_df_data.append({
                    'Provider': provider,
                    'Model': model,
                    'Cost (USD)': f"${cost:.4f}"
                })

        cost_df = pd.DataFrame(cost_df_data)
        st.dataframe(cost_df, use_container_width=True, hide_index=True)

        # Total cost summary
        total_cost = sum(
            cost for models in cost_breakdown.values()
            for cost in models.values()
        )
        st.info(f"**Total Estimated Cost: ${total_cost:.4f}**")
    else:
        st.info("No cost data available")

    st.markdown("---")

    # Recent calls
    st.subheader("🕒 Recent API Calls")

    recent_calls = monitor.get_recent_calls(count=10)
    if recent_calls:
        recent_df_data = []
        for call in recent_calls:
            recent_df_data.append({
                'Time': call['timestamp'].split('T')[1].split('.')[0],  # Extract time
                'Provider': call['provider'],
                'Endpoint': call['endpoint'],
                'Status': call['status'],
                'Duration (ms)': f"{call['duration_ms']:.0f}",
                'Tokens': call['tokens_used'],
                'Cost': f"${call['estimated_cost_usd']:.4f}"
            })

        recent_df = pd.DataFrame(recent_df_data)
        st.dataframe(recent_df, use_container_width=True, hide_index=True)
    else:
        st.info("No recent calls to display")

    st.markdown("---")

    # Export and management
    st.subheader("⚙️ Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()

    with col2:
        export_format = st.selectbox("Export Format", ["JSON", "CSV"], key="export_format")
        if st.button("📥 Export Metrics", use_container_width=True):
            try:
                format_str = export_format.lower()
                exported_data = monitor.export_metrics(format=format_str)

                # Provide download button
                st.download_button(
                    label=f"Download {export_format}",
                    data=exported_data,
                    file_name=f"api_metrics.{format_str}",
                    mime=f"application/{format_str}"
                )
                st.success("Metrics exported successfully!")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with col3:
        if st.button("🗑️ Clear Metrics", use_container_width=True):
            monitor.clear_metrics()
            st.success("Metrics cleared!")
            st.rerun()


def render_compact_api_metrics(monitor: Optional[APIMonitor] = None) -> None:
    """
    Render compact API metrics for sidebar or small spaces.

    Args:
        monitor: APIMonitor instance (uses global if not provided)
    """
    if monitor is None:
        monitor = get_global_monitor()

    st.markdown("#### 📊 API Metrics")

    stats = monitor.get_summary_stats()

    if stats['total_calls'] == 0:
        st.caption("No API calls yet")
        return

    # Compact metrics display
    st.metric("Total Calls", stats['total_calls'])
    st.metric("Success Rate", f"{stats['success_rate']*100:.0f}%")
    st.metric("Total Cost", f"${stats['total_cost_usd']:.4f}")

    # Provider breakdown
    st.caption("**By Provider:**")
    provider_data = stats['by_provider']
    for provider, data in provider_data.items():
        st.caption(f"• {provider}: {data['calls']} calls (${data['cost_usd']:.4f})")


def render_model_selector_with_costs() -> str:
    """
    Render model selector with cost information.

    Returns:
        Selected model name
    """
    st.markdown("#### 🤖 Model Selection")

    models = {
        "Claude 3.5 Sonnet": {
            "id": "claude-3-5-sonnet-20241022",
            "input_cost": "$0.003 / 1K tokens",
            "output_cost": "$0.015 / 1K tokens",
            "recommended": True,
            "description": "Best balance of intelligence and speed"
        },
        "Claude 3 Opus": {
            "id": "claude-3-opus-20240229",
            "input_cost": "$0.015 / 1K tokens",
            "output_cost": "$0.075 / 1K tokens",
            "recommended": False,
            "description": "Highest intelligence, slower"
        },
        "Claude 3 Haiku": {
            "id": "claude-3-haiku-20240307",
            "input_cost": "$0.00025 / 1K tokens",
            "output_cost": "$0.00125 / 1K tokens",
            "recommended": False,
            "description": "Fastest, most cost-effective"
        }
    }

    # Display model options
    selected_model = None
    for model_name, info in models.items():
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                is_selected = st.radio(
                    "Select Model",
                    [model_name],
                    key=f"radio_{model_name}",
                    label_visibility="collapsed"
                )
                if is_selected:
                    selected_model = info['id']

                st.caption(f"**{model_name}**")
                if info['recommended']:
                    st.caption("⭐ **Recommended**")
                st.caption(info['description'])
                st.caption(f"Input: {info['input_cost']} | Output: {info['output_cost']}")

            st.markdown("---")

    return selected_model or models["Claude 3.5 Sonnet"]["id"]


def render_export_options() -> dict:
    """
    Render export format options.

    Returns:
        Dictionary with export settings
    """
    st.markdown("#### 📤 Export Options")

    export_settings = {}

    # Format selection
    export_settings['formats'] = st.multiselect(
        "Export Formats",
        options=["STEP", "STL", "DXF", "PDF", "GLTF"],
        default=["STEP"],
        help="Select one or more formats to export"
    )

    # Additional options
    export_settings['include_metadata'] = st.checkbox(
        "Include Metadata",
        value=True,
        help="Include generation metadata in export"
    )

    export_settings['compress'] = st.checkbox(
        "Compress Files",
        value=False,
        help="Create ZIP archive of exported files"
    )

    # Quality settings for STL
    if "STL" in export_settings['formats']:
        export_settings['stl_quality'] = st.select_slider(
            "STL Quality",
            options=["Low", "Medium", "High", "Ultra"],
            value="High",
            help="Higher quality = larger file size"
        )

    return export_settings


def render_cad_options() -> dict:
    """
    Render CAD generation options.

    Returns:
        Dictionary with CAD settings
    """
    st.markdown("#### ⚙️ CAD Options")

    cad_settings = {}

    # Model type
    cad_settings['model_type'] = st.radio(
        "Model Type",
        options=["3D Model", "2D Drawing", "Assembly"],
        index=0,
        help="Type of CAD model to generate"
    )

    # Part vs Assembly
    if cad_settings['model_type'] in ["3D Model", "Assembly"]:
        cad_settings['is_assembly'] = st.checkbox(
            "Multi-part Assembly",
            value=False,
            help="Generate assembly with multiple parts"
        )

    # Constraints
    with st.expander("Constraints & Features"):
        cad_settings['apply_constraints'] = st.checkbox(
            "Apply Geometric Constraints",
            value=True,
            help="Apply constraints like parallel, perpendicular, etc."
        )

        cad_settings['add_fillets'] = st.checkbox(
            "Add Fillets/Rounds",
            value=False,
            help="Automatically add edge fillets"
        )

        if cad_settings['add_fillets']:
            cad_settings['fillet_radius'] = st.number_input(
                "Fillet Radius (mm)",
                min_value=0.1,
                max_value=50.0,
                value=2.0,
                step=0.5
            )

        cad_settings['add_chamfers'] = st.checkbox(
            "Add Chamfers",
            value=False,
            help="Automatically add edge chamfers"
        )

    # Units
    cad_settings['units'] = st.selectbox(
        "Units",
        options=["mm", "cm", "m", "in", "ft"],
        index=0,
        help="Measurement units for the model"
    )

    # Material (for reference only)
    cad_settings['material'] = st.selectbox(
        "Material (Reference)",
        options=["Not specified", "Aluminum", "Steel", "Plastic (ABS)", "Plastic (PLA)", "Wood", "Other"],
        index=0,
        help="Material information (metadata only)"
    )

    return cad_settings


def render_measurement_tools() -> None:
    """
    Render measurement and analysis tools.
    """
    st.markdown("#### 📏 Measurement Tools")

    with st.expander("Measurements"):
        st.write("**Available Tools:**")

        tool = st.radio(
            "Select Tool",
            options=[
                "Distance Measurement",
                "Angle Measurement",
                "Surface Area",
                "Volume",
                "Center of Mass",
                "Bounding Box"
            ],
            index=0
        )

        if tool == "Distance Measurement":
            st.info("Click two points to measure distance")
        elif tool == "Angle Measurement":
            st.info("Click three points to measure angle")
        elif tool == "Surface Area":
            st.info("Select surfaces to calculate total area")
        elif tool == "Volume":
            st.info("Calculate total volume of solid parts")
        elif tool == "Center of Mass":
            st.info("Calculate center of mass for the model")
        elif tool == "Bounding Box":
            st.info("Show bounding box dimensions")

        st.button("Calculate", use_container_width=True)

    with st.expander("Analysis"):
        st.write("**Available Analysis:**")

        analysis = st.radio(
            "Select Analysis",
            options=[
                "Mass Properties",
                "Interference Check",
                "Dimension Analysis"
            ],
            index=0
        )

        st.button("Run Analysis", use_container_width=True)
