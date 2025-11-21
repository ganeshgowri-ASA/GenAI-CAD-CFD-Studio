"""
API Usage Dashboard Component

Displays API usage statistics and credit breakdown in the Streamlit sidebar.
"""

import streamlit as st
from typing import Dict, Any
import logging
from pathlib import Path

from ...utils.api_usage_tracker import get_tracker

logger = logging.getLogger(__name__)


def render_api_usage_dashboard(location: str = "sidebar") -> None:
    """
    Render API usage dashboard.

    Args:
        location: Where to render ('sidebar' or 'main')
    """
    tracker = get_tracker()

    # Get statistics
    today_stats = tracker.get_today_stats()
    week_stats = tracker.get_week_stats()

    if location == "sidebar":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 API Usage Today")

        # Today's summary
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric(
                "Total Calls",
                today_stats['total_calls'],
                delta=None
            )
        with col2:
            success_rate = today_stats['success_rate']
            st.metric(
                "Success Rate",
                f"{success_rate}%",
                delta=None
            )

        # By service breakdown
        if today_stats['by_service']:
            st.sidebar.markdown("**By Service:**")
            for service, stats in today_stats['by_service'].items():
                service_emoji = {
                    'zoo': '🦓',
                    'claude': '🤖',
                    'adam': '🎨',
                    'build123d': '🔧'
                }.get(service, '📡')

                with st.sidebar.expander(f"{service_emoji} {service.title()}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", stats['total'])
                    with col2:
                        st.metric("✓", stats['success'], delta_color="off")
                    with col3:
                        st.metric("✗", stats['failed'], delta_color="off")

        # Cost estimation
        if today_stats['total_cost_usd'] > 0:
            st.sidebar.markdown("**Estimated Cost:**")
            st.sidebar.info(f"💰 ${today_stats['total_cost_usd']:.4f} USD")

        # Token usage
        if today_stats['total_tokens'] > 0:
            st.sidebar.markdown("**Token Usage:**")
            st.sidebar.info(f"🎫 {today_stats['total_tokens']:,} tokens")

        # Week summary
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Past 7 Days:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Calls", week_stats['total_calls'])
        with col2:
            st.metric("Cost", f"${week_stats['total_cost_usd']:.2f}")

    else:  # main area
        st.header("📊 API Usage Dashboard")

        # Tabs for different time periods
        tab1, tab2, tab3 = st.tabs(["Today", "This Week", "This Month"])

        with tab1:
            _render_detailed_stats(today_stats)

        with tab2:
            _render_detailed_stats(week_stats)

        with tab3:
            month_stats = tracker.get_month_stats()
            _render_detailed_stats(month_stats)

        # Recent errors
        st.subheader("🚨 Recent Errors")
        errors = tracker.get_recent_errors(limit=10)

        if errors:
            for error in errors:
                with st.expander(
                    f"{error['timestamp'][:19]} - {error['service']}/{error['operation']}",
                    expanded=False
                ):
                    st.error(error['error_message'])
                    if error['model']:
                        st.caption(f"Model: {error['model']}")
        else:
            st.success("No recent errors! 🎉")

        # Export options
        st.subheader("📥 Export")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export to CSV", use_container_width=True):
                try:
                    export_path = Path("api_usage_export.csv")
                    tracker.export_to_csv(export_path)
                    st.success(f"✅ Exported to {export_path}")

                    # Provide download button
                    with open(export_path, 'rb') as f:
                        st.download_button(
                            label="Download CSV",
                            data=f,
                            file_name="api_usage.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"Export failed: {e}")

        with col2:
            if st.button("Clear Old Records (90+ days)", use_container_width=True):
                try:
                    deleted = tracker.clear_old_records(days=90)
                    st.success(f"✅ Cleared {deleted} old records")
                except Exception as e:
                    st.error(f"Clear failed: {e}")


def _render_detailed_stats(stats: Dict[str, Any]) -> None:
    """Render detailed statistics."""
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Calls",
            stats['total_calls']
        )

    with col2:
        st.metric(
            "Successful",
            stats['successful_calls'],
            delta=None,
            delta_color="normal"
        )

    with col3:
        st.metric(
            "Failed",
            stats['failed_calls'],
            delta=None,
            delta_color="inverse"
        )

    with col4:
        st.metric(
            "Success Rate",
            f"{stats['success_rate']}%"
        )

    # Service breakdown
    if stats['by_service']:
        st.subheader("By Service")

        # Create DataFrame for better display
        service_data = []
        for service, service_stats in stats['by_service'].items():
            service_emoji = {
                'zoo': '🦓',
                'claude': '🤖',
                'adam': '🎨',
                'build123d': '🔧'
            }.get(service, '📡')

            service_data.append({
                'Service': f"{service_emoji} {service.title()}",
                'Total': service_stats['total'],
                'Success': service_stats['success'],
                'Failed': service_stats['failed'],
                'Success Rate': f"{(service_stats['success'] / service_stats['total'] * 100):.1f}%" if service_stats['total'] > 0 else "0%"
            })

        st.dataframe(
            service_data,
            use_container_width=True,
            hide_index=True
        )

    # Model breakdown
    if stats['by_model']:
        st.subheader("By Model")

        model_data = []
        for model, model_stats in stats['by_model'].items():
            model_data.append({
                'Model': model,
                'Calls': model_stats['total'],
                'Success': model_stats['success'],
                'Tokens': f"{model_stats['tokens']:,}" if model_stats['tokens'] > 0 else "-",
                'Cost (USD)': f"${model_stats['cost_usd']:.4f}" if model_stats['cost_usd'] > 0 else "-"
            })

        st.dataframe(
            model_data,
            use_container_width=True,
            hide_index=True
        )

    # Cost and token summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Tokens",
            f"{stats['total_tokens']:,}" if stats['total_tokens'] > 0 else "0"
        )

    with col2:
        st.metric(
            "Total Cost",
            f"${stats['total_cost_usd']:.4f}" if stats['total_cost_usd'] > 0 else "$0.00"
        )

    with col3:
        st.metric(
            "Avg Duration",
            f"{stats['avg_duration_seconds']:.2f}s" if stats['avg_duration_seconds'] > 0 else "0s"
        )


def render_compact_usage_badge() -> None:
    """Render a compact usage badge for the sidebar."""
    tracker = get_tracker()
    today_stats = tracker.get_today_stats()

    total_calls = today_stats['total_calls']
    success_rate = today_stats['success_rate']

    # Color based on success rate
    if success_rate >= 90:
        color = "green"
    elif success_rate >= 70:
        color = "orange"
    else:
        color = "red"

    badge_html = f"""
    <div style="
        background: linear-gradient(135deg, {color}, {color}dd);
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        text-align: center;
    ">
        <div style="font-size: 24px; font-weight: bold; color: white;">
            {total_calls}
        </div>
        <div style="font-size: 12px; color: white; opacity: 0.9;">
            API Calls Today ({success_rate}% success)
        </div>
    </div>
    """

    st.sidebar.markdown(badge_html, unsafe_allow_html=True)


def render_service_details(service: str) -> None:
    """
    Render detailed statistics for a specific service.

    Args:
        service: Service name ('zoo', 'claude', 'adam', 'build123d')
    """
    tracker = get_tracker()

    st.subheader(f"📊 {service.title()} API Statistics")

    # Time period selector
    period = st.selectbox(
        "Time Period",
        ["Last 7 Days", "Last 30 Days", "Last 90 Days"],
        key=f"{service}_period"
    )

    days_map = {
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90
    }

    breakdown = tracker.get_service_breakdown(service, days_map[period])

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Calls", breakdown['total_calls'])

    with col2:
        st.metric("Successful", breakdown['successful_calls'])

    with col3:
        st.metric("Failed", breakdown['failed_calls'])

    # Operations breakdown
    if breakdown['operations']:
        st.subheader("By Operation")

        operation_data = []
        for operation, op_stats in breakdown['operations'].items():
            operation_data.append({
                'Operation': operation.title(),
                'Total': op_stats['total'],
                'Success': op_stats['success'],
                'Failed': op_stats['failed'],
                'Success Rate': f"{(op_stats['success'] / op_stats['total'] * 100):.1f}%" if op_stats['total'] > 0 else "0%"
            })

        st.dataframe(
            operation_data,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(f"No {service} API calls in the selected period.")
