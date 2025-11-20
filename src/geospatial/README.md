# Geospatial Module - Solar PV Layout Optimization

This module provides comprehensive tools for optimizing solar photovoltaic (PV) plant layouts using geospatial analysis.

## Features

### 1. Layout Optimization (`layout_optimizer.py`)
- **Grid Layout Generation**: Create regular grids of solar modules within site boundaries
- **Spatial Indexing**: Fast collision detection using Rtree
- **Layout Optimization**: Multi-objective optimization using genetic algorithms
- **Coverage Analysis**: Calculate site coverage percentages
- **Capacity Calculations**: Compute total power capacity

### 2. Shadow Analysis (`shadow_analysis.py`)
- **Sun Position Calculation**: Accurate solar azimuth and elevation computation
- **Shadow Projection**: 3D shadow projection to ground plane
- **Time-Series Analysis**: Analyze shading patterns over time
- **Energy Loss Estimation**: Calculate energy losses due to shading
- **Visualization**: Generate shadow diagrams and sun path charts

### 3. Map Processing (`map_processor.py`)
- **GeoJSON Support**: Load and save geospatial data
- **CRS Transformation**: Convert between coordinate reference systems
- **Spatial Clipping**: Clip data to boundary polygons
- **Area Calculations**: Compute geometric areas

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from geospatial import LayoutOptimizer, ShadowAnalyzer, MapProcessor
from shapely.geometry import Polygon
from datetime import datetime

# Define site boundary
site = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

# Initialize optimizer
optimizer = LayoutOptimizer()

# Generate layout
layout = optimizer.generate_grid_layout(
    site_polygon=site,
    module_dims=(2.0, 1.0),  # 2m x 1m modules
    spacing=(0.5, 0.5),       # 0.5m spacing
    orientation=0.0           # North-facing
)

# Calculate statistics
coverage = optimizer.calculate_coverage(site, layout)
capacity = optimizer.calculate_total_capacity(layout, module_power=0.4)

print(f"Modules: {len(layout)}")
print(f"Coverage: {coverage:.2f}%")
print(f"Capacity: {capacity:.2f} kW")
```

### Shadow Analysis

```python
# Initialize shadow analyzer
analyzer = ShadowAnalyzer()

# Analyze shading for a day
start = datetime(2024, 6, 21, 0, 0)
end = datetime(2024, 6, 21, 23, 59)

shadow_hours = analyzer.analyze_shading(
    layout=layout,
    lat=40.7128,
    lon=-74.0060,
    date_range=(start, end),
    time_steps=24
)

# Estimate energy loss
energy_loss = analyzer.estimate_energy_loss(
    shadow_hours=shadow_hours,
    module_power=0.4,
    total_hours=24.0
)

print(f"Energy loss: {energy_loss['loss_percentage']:.2f}%")
```

### Layout Optimization

```python
# Optimize layout for maximum module count
objectives = {
    'maximize_count': True,
    'minimize_cable_length': False,
}

optimized_layout = optimizer.optimize_layout(
    site=site,
    module_dims=(2.0, 1.0),
    objectives=objectives
)

print(f"Optimized modules: {len(optimized_layout)}")
```

## Module Details

### LayoutOptimizer

#### Methods

- `generate_grid_layout(site_polygon, module_dims, spacing, orientation=0.0)`: Generate regular grid layout
- `optimize_layout(site, module_dims, objectives, constraints=None)`: Optimize layout using genetic algorithm
- `calculate_coverage(site, layout)`: Calculate coverage percentage
- `calculate_total_capacity(layout, module_power)`: Calculate total power capacity
- `check_collisions(layout)`: Detect module collisions
- `get_layout_statistics(site, layout, module_power)`: Get comprehensive statistics

### ShadowAnalyzer

#### Methods

- `calculate_sun_position(lat, lon, date, time)`: Calculate solar azimuth and elevation
- `project_shadows(modules, sun_azimuth, sun_elevation, height, tilt_angle)`: Project module shadows
- `analyze_shading(layout, lat, lon, date_range, time_steps)`: Time-series shadow analysis
- `visualize_shadows(layout, shadows)`: Create visualization
- `get_sun_path(lat, lon, date, time_steps)`: Calculate sun path for a day
- `estimate_energy_loss(shadow_hours, module_power, total_hours)`: Estimate energy loss

### MapProcessor

#### Methods

- `load_geojson(filepath)`: Load GeoJSON file
- `save_geojson(gdf, filepath)`: Save to GeoJSON
- `transform_crs(gdf, from_crs, to_crs)`: Transform coordinate systems
- `clip_to_boundary(gdf, boundary)`: Clip to polygon
- `calculate_area(gdf)`: Calculate areas
- `calculate_bounds(gdf)`: Get bounding box

## Advanced Examples

### Complete Workflow

```python
from geospatial import LayoutOptimizer, ShadowAnalyzer, MapProcessor
from shapely.geometry import Polygon
from datetime import datetime

# 1. Load or create site
site = Polygon([(0, 0), (150, 0), (150, 100), (0, 100)])

# 2. Generate optimal layout
optimizer = LayoutOptimizer()
layout = optimizer.optimize_layout(
    site=site,
    module_dims=(2.0, 1.0),
    objectives={'maximize_count': True}
)

# 3. Analyze shadows
analyzer = ShadowAnalyzer()
shadow_hours = analyzer.analyze_shading(
    layout=layout,
    lat=35.0,
    lon=-120.0,
    date_range=(
        datetime(2024, 1, 1),
        datetime(2024, 12, 31)
    ),
    time_steps=24
)

# 4. Calculate energy metrics
energy_loss = analyzer.estimate_energy_loss(
    shadow_hours, module_power=0.4, total_hours=8760
)

# 5. Save results
processor = MapProcessor()
processor.save_geojson(layout, 'solar_layout.geojson')

# 6. Visualize
shadows = analyzer.project_shadows(
    layout, sun_azimuth=180, sun_elevation=45, height=2.0
)
fig = analyzer.visualize_shadows(layout, shadows)
```

### Custom Site from GeoJSON

```python
# Load existing site boundary
processor = MapProcessor()
site_gdf = processor.load_geojson('site_boundary.geojson')
site = site_gdf.geometry.iloc[0]

# Transform to appropriate CRS for area calculations
site_gdf = processor.transform_crs(site_gdf, 'EPSG:4326', 'EPSG:32633')

# Generate layout
layout = optimizer.generate_grid_layout(
    site_polygon=site_gdf.geometry.iloc[0],
    module_dims=(2.0, 1.0),
    spacing=(1.0, 1.0)
)
```

## Testing

Run the test suite:

```bash
pytest tests/test_geospatial.py -v
```

Run with coverage:

```bash
pytest tests/test_geospatial.py --cov=src/geospatial --cov-report=html
```

## Dependencies

- **shapely** >= 2.0.0: Geometric operations
- **geopandas** >= 0.14.0: Geospatial data handling
- **rtree** >= 1.0.0: Spatial indexing
- **pyproj** >= 3.6.0: Coordinate transformations
- **scipy** >= 1.11.0: Optimization algorithms
- **numpy** >= 1.24.0: Numerical operations
- **matplotlib** >= 3.7.0: Visualization

## Performance Considerations

- **Rtree Indexing**: Used for O(log n) collision detection
- **Vectorized Operations**: NumPy arrays for shadow calculations
- **Sparse Time Sampling**: Adjustable time_steps for shadow analysis
- **Differential Evolution**: Efficient multi-objective optimization

## Coordinate Reference Systems

For accurate area and distance calculations:
- Use projected CRS (e.g., UTM zones, EPSG:32633) for metric calculations
- WGS84 (EPSG:4326) is suitable for global coordinates but uses degrees
- Transform between CRS using `MapProcessor.transform_crs()`

## Known Limitations

- Shadow analysis uses simplified solar position algorithm (adequate for most cases)
- For production solar applications, consider pvlib for higher accuracy
- Optimization runtime increases with site size and module count
- Large time-series analyses may require significant computation time

## Future Enhancements

- Integration with pvlib for enhanced solar calculations
- Support for terrain elevation data
- Wind load analysis
- Cost optimization including cabling and inverters
- Real-time weather data integration
- 3D visualization with terrain

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- Code follows PEP 8 style guidelines
- Documentation is updated
- New features include tests

## License

MIT License - See LICENSE file for details
