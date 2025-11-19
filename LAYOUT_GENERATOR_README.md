# 🗺️ Solar PV Layout Generator

An interactive web-based tool for designing and optimizing solar photovoltaic (PV) panel layouts using map-based boundary drawing and real-time shadow analysis.

## Features

### 🎨 Interactive Map Interface
- **Folium-based mapping** with OpenStreetMap and satellite imagery
- **Drawing tools** for site boundary definition:
  - Rectangle tool for regular plots
  - Polygon tool for custom shapes
  - Freehand drawing capabilities
- **Geocoding** for location search
- **Measurement tools** for distance and area
- **Fullscreen mode** for detailed work

### ⚡ Module Configuration
- **Standard presets** for common solar panels:
  - 60-cell modules (300W)
  - 72-cell modules (350W)
  - High-efficiency modules (400-450W)
  - Bifacial modules (500W)
- **Custom configuration** options:
  - Module dimensions (width × length)
  - Power rating
  - Row and column spacing
  - Tilt angle (0-90°)
  - Azimuth orientation (0-360°)

### 📊 Layout Optimization
- **Automatic module placement** within boundaries
- **Spacing optimization** accounting for:
  - Shadow clearance
  - Maintenance access
  - Tilt angle projection
- **Orientation options**:
  - Landscape mode
  - Portrait mode
- **Statistics calculation**:
  - Total module count
  - Total capacity (kW)
  - Coverage percentage
  - Power density (W/m²)

### 🌤️ Shadow Analysis
- **Sun position calculation** based on:
  - Site latitude/longitude
  - Date and time
- **Shadow intensity mapping**
- **Daylight hour analysis**
- **Timeline visualization** of sun path
- **Comprehensive reports**:
  - Hourly sun position
  - Shadow patterns throughout the day

### 💾 Export Capabilities
- **GeoJSON export** for GIS integration
- **CSV export** with module positions:
  - Module ID, row, column
  - Power rating
  - Geographic coordinates
  - Orientation parameters
- **Shadow analysis reports** (JSON format)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ganeshgowri-ASA/GenAI-CAD-CFD-Studio.git
   cd GenAI-CAD-CFD-Studio
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Access the application**:
   - Open your browser to `http://localhost:8501`

## Usage Guide

### Step 1: Draw Site Boundary

1. Navigate the map to your site location using:
   - Search bar (geocoding)
   - Manual pan/zoom
2. Use drawing tools to outline the site:
   - Click the rectangle/polygon tool
   - Draw the boundary on the map
   - Edit or delete as needed

### Step 2: Configure Modules

**In the sidebar:**

1. **Select a preset** or choose "Custom"
2. **Adjust dimensions**:
   - Width and length (meters)
   - Power rating (watts)
3. **Set spacing**:
   - Row spacing (for shadow clearance)
   - Column spacing (between modules)
4. **Configure orientation**:
   - Tilt angle (optimal for latitude)
   - Azimuth (180° = south-facing)

### Step 3: Set Site Location

1. **Enter coordinates**:
   - Latitude (decimal degrees)
   - Longitude (decimal degrees)
2. **Choose analysis time**:
   - Date (for seasonal analysis)
   - Time (for shadow calculation)

### Step 4: Generate Layout

1. Click **"Generate Layout"** button
2. Review the generated design on the map
3. Check statistics in the results panel

### Step 5: Analyze & Export

1. **Review statistics**:
   - Module count and capacity
   - Coverage percentage
   - Power density
2. **Check shadow analysis**:
   - Sun position timeline
   - Shadow intensity map
3. **Export data**:
   - Download GeoJSON for GIS
   - Download CSV for spreadsheet analysis
   - Download shadow report for detailed analysis

## Project Structure

```
GenAI-CAD-CFD-Studio/
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── src/
│   ├── ui/
│   │   ├── layout_generator.py    # Main UI application
│   │   └── components/
│   │       ├── map_interface.py   # Folium map component
│   │       └── module_configurator.py  # Module config UI
│   └── geospatial/
│       └── layout_optimizer.py    # Layout & shadow analysis engine
└── tests/
    └── test_ui_layout_gen.py      # Unit tests
```

## Components

### MapInterface (`src/ui/components/map_interface.py`)

Provides interactive mapping capabilities:
- `render_map()` - Renders Folium map with drawing tools
- `add_drawing_tools()` - Adds polygon/rectangle drawing
- `get_drawn_shapes()` - Extracts drawn boundaries
- `add_module_layer()` - Displays module layout
- `add_shadow_overlay()` - Visualizes shadow analysis
- `export_to_geojson()` - Exports geometries

### ModuleConfigurator (`src/ui/components/module_configurator.py`)

Handles module configuration:
- Standard presets for common panel types
- Input validation
- Configuration export/import
- Direction name conversion (azimuth → compass)

### LayoutOptimizer (`src/geospatial/layout_optimizer.py`)

Generates optimized layouts:
- `generate_layout()` - Places modules within boundary
- `calculate_statistics()` - Computes layout metrics
- Accounts for tilt angle in spacing
- Supports rotation based on azimuth

### ShadowAnalyzer (`src/geospatial/layout_optimizer.py`)

Performs shadow analysis:
- `calculate_sun_position()` - Computes sun altitude/azimuth
- `analyze_shadows()` - Creates shadow intensity grid
- `generate_shadow_report()` - Full-day analysis report

## Technical Details

### Coordinate System
- Uses WGS84 (EPSG:4326) geographic coordinates
- Latitude/longitude in decimal degrees
- Approximate meter-based calculations for layout

### Sun Position Calculation
- Simplified solar position algorithm
- Based on latitude, longitude, date, and time
- For production use, consider integrating:
  - `pvlib` for accurate solar calculations
  - `pysolar` for detailed sun tracking

### Module Spacing
- **Row spacing** accounts for:
  - Tilt angle shadow projection
  - Maintenance access requirements
- **Column spacing** for:
  - Module mounting gaps
  - Thermal expansion

### Shadow Intensity
- Calculated based on sun altitude
- Grid-based visualization
- Hourly timeline analysis

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_ui_layout_gen.py -v
```

## Configuration Examples

### Residential Installation
```python
Module: Standard 60-cell (300W)
Dimensions: 0.992m × 1.650m
Tilt: 30° (latitude-dependent)
Azimuth: 180° (south-facing)
Row Spacing: 1.0m
```

### Commercial Installation
```python
Module: High-efficiency 72-cell (450W)
Dimensions: 1.046m × 2.008m
Tilt: 25°
Azimuth: 180°
Row Spacing: 1.5m
```

### Ground-Mount Bifacial
```python
Module: Bifacial 144-cell (500W)
Dimensions: 1.134m × 2.279m
Tilt: 25°
Azimuth: 180°
Row Spacing: 2.0m (higher for bifacial gain)
```

## Optimization Tips

1. **Tilt Angle**: Generally optimal at site latitude ± 15°
2. **Azimuth**: 180° (south) in Northern Hemisphere, 0° (north) in Southern
3. **Row Spacing**: Increase to reduce inter-row shading
4. **Coverage**: 50-70% is typical for optimal spacing

## Limitations

- Simplified sun position calculation (use pvlib for production)
- Assumes flat terrain (no topography)
- No obstacle shading (trees, buildings)
- Basic shadow analysis (no detailed ray tracing)
- Coordinate-based area calculations (approximate)

## Future Enhancements

- [ ] Integration with pvlib for accurate solar calculations
- [ ] Terrain elevation support (DEM data)
- [ ] 3D visualization of module layout
- [ ] Obstacle detection and avoidance
- [ ] Multi-objective optimization (cost, energy, etc.)
- [ ] Energy yield simulation
- [ ] Electrical string design
- [ ] BOS (Balance of System) component placement
- [ ] Wind load analysis
- [ ] Snow load considerations

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information]

## Acknowledgments

- Built with Streamlit for the web interface
- Folium for interactive mapping
- GeoPandas for geospatial operations
- Shapely for geometric calculations

---

**GenAI CAD-CFD Studio** - Empowering renewable energy design with AI-assisted tools
