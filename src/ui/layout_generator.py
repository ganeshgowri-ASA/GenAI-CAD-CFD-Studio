"""
Solar PV Layout Generator - Main UI
Interactive map-based solar panel layout design and optimization
"""

import streamlit as st
import geopandas as gpd
from shapely.geometry import Polygon
from datetime import datetime, timedelta
import json
import pandas as pd
from io import BytesIO
import base64

# Import custom components
from src.ui.components.map_interface import MapInterface
from src.ui.components.module_configurator import ModuleConfigurator
from src.geospatial.layout_optimizer import LayoutOptimizer, ShadowAnalyzer


def initialize_session_state():
    """Initialize session state variables"""
    if 'layout_generated' not in st.session_state:
        st.session_state.layout_generated = False
    if 'layout_gdf' not in st.session_state:
        st.session_state.layout_gdf = None
    if 'shadow_gdf' not in st.session_state:
        st.session_state.shadow_gdf = None
    if 'statistics' not in st.session_state:
        st.session_state.statistics = {}
    if 'boundary_polygon' not in st.session_state:
        st.session_state.boundary_polygon = None
    if 'shadow_report' not in st.session_state:
        st.session_state.shadow_report = None


def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title="Solar PV Layout Generator",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.header("🗺️ Solar PV Layout Generator")
    st.markdown("Design and optimize solar panel layouts with interactive mapping tools")

    # Sidebar configuration
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/4CAF50/FFFFFF/?text=Solar+PV+Designer",
                 use_container_width=True)

        st.markdown("---")

        # Module Configurator
        configurator = ModuleConfigurator()
        module_config = configurator.render_compact()

        st.markdown("---")

        # Site Configuration
        st.markdown("### 🌍 Site Configuration")

        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=37.7749,
                step=0.0001,
                format="%.4f",
                help="Site latitude for shadow analysis"
            )

        with col2:
            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-122.4194,
                step=0.0001,
                format="%.4f",
                help="Site longitude"
            )

        # Date/Time for shadow analysis
        st.markdown("#### 🕐 Shadow Analysis Time")

        analysis_date = st.date_input(
            "Date",
            value=datetime.now(),
            help="Date for shadow analysis"
        )

        analysis_time = st.time_input(
            "Time",
            value=datetime.now().time(),
            help="Time for shadow analysis"
        )

        # Combine date and time
        analysis_datetime = datetime.combine(analysis_date, analysis_time)

        # Module orientation
        st.markdown("#### 📱 Module Orientation")
        module_orientation = st.radio(
            "Orientation",
            options=['landscape', 'portrait'],
            index=0,
            help="Physical orientation of modules"
        )

        st.markdown("---")

        # Generate Layout Button
        generate_button = st.button(
            "🚀 Generate Layout",
            type="primary",
            use_container_width=True,
            help="Generate solar panel layout from drawn boundary"
        )

        # Clear button
        if st.button("🗑️ Clear Layout", use_container_width=True):
            st.session_state.layout_generated = False
            st.session_state.layout_gdf = None
            st.session_state.shadow_gdf = None
            st.session_state.statistics = {}
            st.session_state.boundary_polygon = None
            st.session_state.shadow_report = None
            st.rerun()

    # Main content area
    col_map, col_results = st.columns([2, 1])

    with col_map:
        st.subheader("Interactive Map")

        # Instructions
        with st.expander("📖 Instructions", expanded=False):
            st.markdown("""
            1. **Draw Site Boundary**: Use the drawing tools on the map to outline your site
               - 🔲 Rectangle tool for regular shapes
               - 🔺 Polygon tool for custom shapes
            2. **Configure Modules**: Set module specifications in the sidebar
            3. **Set Location**: Enter latitude/longitude for shadow analysis
            4. **Generate Layout**: Click 'Generate Layout' to create the solar array
            5. **Review Results**: Check statistics and shadow analysis
            6. **Export**: Download GeoJSON, CSV, or shadow report
            """)

        # Map interface
        map_interface = MapInterface()

        # Render map
        map_data = map_interface.render_map(
            center=(latitude, longitude),
            zoom=15,
            height=600,
            key="solar_layout_map"
        )

        # Display map coordinates if clicked
        if map_data and map_data.get('last_clicked'):
            clicked = map_data['last_clicked']
            st.caption(f"📍 Last clicked: {clicked['lat']:.6f}, {clicked['lng']:.6f}")

    with col_results:
        st.subheader("Results & Analytics")

        # Check if boundary is drawn
        if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
            boundary_polygons = map_interface.get_drawn_shapes(map_data)

            if boundary_polygons:
                st.success(f"✅ {len(boundary_polygons)} boundary(s) drawn")

                # Use first polygon as boundary
                boundary = boundary_polygons[0]
                st.session_state.boundary_polygon = boundary

                # Display boundary info
                area_m2 = boundary.area * 111320 * 111320  # Rough conversion to m²
                st.metric("Site Area", f"{area_m2:,.0f} m²")
            else:
                st.warning("⚠️ No boundary drawn yet")
                st.info("👆 Use the drawing tools on the map to draw your site boundary")
        else:
            st.info("👆 Draw a boundary on the map to get started")

        # Generate layout when button is clicked
        if generate_button:
            if st.session_state.boundary_polygon is None:
                st.error("❌ Please draw a site boundary first!")
            else:
                with st.spinner("Generating layout..."):
                    # Validate module config
                    is_valid, error_msg = module_config.validate()
                    if not is_valid:
                        st.error(f"❌ Invalid configuration: {error_msg}")
                    else:
                        # Create optimizer
                        optimizer = LayoutOptimizer(
                            module_width=module_config.width,
                            module_length=module_config.length,
                            module_power=module_config.power_watts,
                            row_spacing=module_config.row_spacing,
                            column_spacing=module_config.column_spacing,
                            tilt_angle=module_config.tilt_angle,
                            azimuth=module_config.azimuth
                        )

                        # Generate layout
                        layout_gdf = optimizer.generate_layout(
                            st.session_state.boundary_polygon,
                            orientation=module_orientation
                        )

                        # Calculate statistics
                        stats = optimizer.calculate_statistics(
                            layout_gdf,
                            st.session_state.boundary_polygon
                        )

                        # Perform shadow analysis
                        shadow_analyzer = ShadowAnalyzer(latitude, longitude)
                        shadow_gdf = shadow_analyzer.analyze_shadows(
                            layout_gdf,
                            analysis_datetime,
                            grid_resolution=1.0
                        )

                        # Generate shadow report
                        start_time = analysis_datetime.replace(hour=6, minute=0)
                        end_time = analysis_datetime.replace(hour=18, minute=0)
                        shadow_report = shadow_analyzer.generate_shadow_report(
                            layout_gdf,
                            start_time,
                            end_time,
                            time_step_hours=1
                        )

                        # Store in session state
                        st.session_state.layout_gdf = layout_gdf
                        st.session_state.shadow_gdf = shadow_gdf
                        st.session_state.statistics = stats
                        st.session_state.shadow_report = shadow_report
                        st.session_state.layout_generated = True

                        st.success("✅ Layout generated successfully!")
                        st.rerun()

        # Display results if layout is generated
        if st.session_state.layout_generated and st.session_state.layout_gdf is not None:
            st.markdown("### 📊 Layout Statistics")

            stats = st.session_state.statistics

            # Key metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Total Modules",
                    f"{stats['total_modules']:,}",
                    help="Number of solar modules in layout"
                )
                st.metric(
                    "Coverage",
                    f"{stats['coverage_percentage']:.1f}%",
                    help="Percentage of site area covered by modules"
                )

            with col2:
                st.metric(
                    "Total Capacity",
                    f"{stats['total_capacity_kw']:.1f} kW",
                    help="Total power generation capacity"
                )
                st.metric(
                    "Power Density",
                    f"{stats['avg_power_density_w_m2']:.1f} W/m²",
                    help="Average power per module area"
                )

            # Detailed statistics
            with st.expander("📈 Detailed Statistics", expanded=True):
                st.write(f"**Module Area:** {stats['total_module_area_m2']:.1f} m²")
                st.write(f"**Site Area:** {stats['site_area_m2']:.1f} m²")
                st.write(f"**Number of Rows:** {stats['number_of_rows']}")
                st.write(f"**Avg Modules per Row:** {stats['modules_per_row']:.1f}")

            # Shadow analysis results
            if st.session_state.shadow_report:
                st.markdown("### 🌤️ Shadow Analysis")

                report = st.session_state.shadow_report
                sun_stats = report['sun_statistics']

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Daylight Hours",
                        f"{sun_stats['daylight_hours']}h",
                        help="Hours with sun above horizon"
                    )

                with col2:
                    st.metric(
                        "Avg Sun Altitude",
                        f"{sun_stats['avg_sun_altitude']:.1f}°",
                        help="Average sun altitude during daylight"
                    )

                # Shadow timeline chart
                with st.expander("☀️ Sun Position Timeline", expanded=False):
                    timeline_df = pd.DataFrame(report['shadow_timeline'])
                    if len(timeline_df) > 0:
                        st.line_chart(
                            timeline_df.set_index('timestamp')[['sun_altitude', 'sun_azimuth']],
                            use_container_width=True
                        )

    # Export section (full width below)
    if st.session_state.layout_generated:
        st.markdown("---")
        st.subheader("💾 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Export GeoJSON
            if st.button("📥 Download GeoJSON", use_container_width=True):
                layout_gdf = st.session_state.layout_gdf
                geojson_str = layout_gdf.to_json()

                st.download_button(
                    label="💾 Save GeoJSON",
                    data=geojson_str,
                    file_name=f"solar_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                    mime="application/json",
                    use_container_width=True
                )

        with col2:
            # Export CSV
            if st.button("📥 Download Layout CSV", use_container_width=True):
                layout_gdf = st.session_state.layout_gdf

                # Create CSV with module positions
                df = pd.DataFrame({
                    'module_id': layout_gdf['module_id'],
                    'row': layout_gdf['row'],
                    'column': layout_gdf['column'],
                    'power_watts': layout_gdf['power_watts'],
                    'tilt_angle': layout_gdf['tilt_angle'],
                    'azimuth': layout_gdf['azimuth'],
                    'centroid_lat': layout_gdf.geometry.centroid.y,
                    'centroid_lon': layout_gdf.geometry.centroid.x
                })

                csv = df.to_csv(index=False)

                st.download_button(
                    label="💾 Save CSV",
                    data=csv,
                    file_name=f"module_positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        with col3:
            # Export Shadow Report
            if st.button("📥 Download Shadow Report", use_container_width=True):
                report = st.session_state.shadow_report
                report_json = json.dumps(report, indent=2)

                st.download_button(
                    label="💾 Save Report",
                    data=report_json,
                    file_name=f"shadow_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

    # Footer
    st.markdown("---")
    st.caption("🌞 Solar PV Layout Generator | GenAI CAD-CFD Studio | Powered by Streamlit")


def render():
    """Render function for tab integration"""
    # Import here to avoid circular dependencies at module level
    initialize_session_state()

    # Header
    st.header('🗺️ Solar PV Layout Generator')

    st.markdown(
        """
        Design solar panel layouts on interactive maps with AI-powered optimization.
        """
    )

    # The main content would go here - for now, show a placeholder
    st.info("🚧 Layout Generator UI is being integrated. Core functionality available through main() function.")


if __name__ == "__main__":
    main()
