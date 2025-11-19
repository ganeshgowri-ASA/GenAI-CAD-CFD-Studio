# GenAI-CAD-CFD-Studio

🚀 Universal AI-Powered CAD & CFD Platform | Democratizing 3D Design & Simulation | Natural Language → Parametric Models | Build123d + Zoo.dev + Adam.new + OpenFOAM | Solar PV, Test Chambers, Digital Twins & More

## 📊 Advanced 3D Visualization Module (PyVista)

Professional-grade 3D visualization system for CAD models, CFD simulation results, and mesh quality analysis.

### Features

- **🎨 CAD Model Rendering**: Support for STEP, STL, VTK, and other mesh formats
- **🌊 CFD Results Visualization**: Scalar/vector fields, streamlines, slice planes
- **📐 Mesh Quality Analysis**: Quality metrics with highlighting of problematic elements
- **📝 Annotations**: Text labels, arrows, dimension lines
- **📸 High-Quality Export**: 4K images, animations (MP4/GIF), interactive HTML
- **🔄 Streamlit Integration**: Interactive web-based visualization with controls
- **📈 2D Plotting**: Convergence plots, residuals, statistics with Plotly
- **⚡ Performance Optimization**: LOD hierarchy, decimation, caching

### Quick Start

```python
from src.visualization import PyVistaViewer, PlotlyCharts

# Create viewer
viewer = PyVistaViewer(theme='document')

# Render CAD model
plotter = viewer.render_cad_model('model.stl', plotter_config={
    'color': 'lightblue',
    'show_edges': True,
    'lighting': True
})
plotter.show()

# Render CFD results
cfd_plotter = viewer.render_cfd_results(
    'results.vtu',
    field='velocity',
    show_streamlines=True,
    cmap='jet'
)
cfd_plotter.show()

# Create convergence plots
charts = PlotlyCharts()
residual_data = {
    'iteration': [1, 2, 3, ...],
    'continuity': [1e-3, 5e-4, 1e-4, ...],
    'velocity': [...]
}
fig = charts.plot_residuals(residual_data, log_scale=True)
fig.show()
```

### Streamlit Integration

```python
import streamlit as st
from src.visualization import StreamlitPyVista, PyVistaViewer

st.title("CAD/CFD Visualization")

# Create interactive viewer
viewer = PyVistaViewer()
mesh = viewer.create_sample_mesh('sphere')

# Display with controls
StreamlitPyVista.create_interactive_viewer(
    mesh,
    key="viewer",
    sidebar_controls=True
)
```

### High-Quality Export

```python
from src.visualization import ExportRenderer

renderer = ExportRenderer()

# Export 4K image
renderer.render_high_quality(
    mesh,
    'output.png',
    resolution=(3840, 2160),
    anti_aliasing=True,
    ambient_occlusion=True,
    shadows=True
)

# Create turntable animation
renderer.create_turntable_animation(
    mesh,
    'turntable.mp4',
    n_frames=120,
    fps=30
)

# Batch render multiple views
renderer.batch_render(
    [(mesh1, 'model1'), (mesh2, 'model2')],
    views=['iso', 'xy', 'xz', 'yz'],
    output_dir='renders/'
)
```

### Mesh Utilities

```python
from src.visualization import MeshOptimizer, MeshConverter, convert_mesh

# Optimize large mesh for display
optimizer = MeshOptimizer()
optimized = optimizer.decimate_mesh(large_mesh, target_reduction=0.7)

# Create LOD hierarchy
lod_levels = optimizer.create_lod_hierarchy(mesh, n_levels=4)

# Convert between formats
trimesh_obj = convert_mesh(pyvista_mesh, target_format='trimesh')
vertices, faces = convert_mesh(pyvista_mesh, target_format='numpy')
```

### Installation

```bash
pip install -r requirements.txt
```

### Testing

```bash
pytest tests/test_visualization.py -v
```

### Module Structure

```
src/visualization/
├── __init__.py              # Module exports
├── pyvista_viewer.py        # Core 3D rendering engine
├── streamlit_pyvista.py     # Streamlit integration
├── plotly_charts.py         # 2D plotting utilities
├── export_renderer.py       # High-quality export
└── utils.py                 # Mesh conversion & optimization

tests/
└── test_visualization.py    # Comprehensive tests
```

### Dependencies

- **pyvista** >= 0.43.0 - 3D visualization
- **vtk** >= 9.3.0 - VTK rendering backend
- **plotly** >= 5.18.0 - Interactive 2D plots
- **stpyvista** >= 0.1.0 - Streamlit integration
- **streamlit** >= 1.30.0 - Web applications
- **numpy**, **pandas** - Data processing
- **trimesh** - Mesh format conversion (optional)

### API Documentation

#### PyVistaViewer

Main class for 3D visualization of CAD and CFD data.

**Methods:**
- `render_cad_model(geometry_data, plotter_config)` - Render CAD model
- `render_cfd_results(result_data, field, **kwargs)` - Render CFD results
- `render_mesh_quality(mesh, quality_metric, threshold)` - Mesh quality visualization
- `add_annotations(plotter, labels)` - Add text/arrow/dimension annotations
- `export_image(plotter, filepath, resolution)` - Export screenshot

#### PlotlyCharts

2D plotting utilities for analysis data.

**Methods:**
- `plot_residuals(iteration_data, log_scale)` - CFD residuals plot
- `plot_convergence(history, metrics)` - Convergence history
- `plot_statistics(data_dict, chart_type)` - Bar/pie charts
- `plot_shadow_heatmap(shadow_hours)` - Heatmap visualization
- `plot_comparison(data, x_col, y_col)` - Multi-dataset comparison
- `plot_3d_scatter(data, x, y, z)` - 3D scatter plot

#### ExportRenderer

High-quality rendering and export functionality.

**Methods:**
- `render_high_quality(mesh, output_file, resolution, **kwargs)` - 4K+ rendering
- `create_animation(meshes_sequence, output_video, fps)` - MP4/GIF animation
- `create_turntable_animation(mesh, output_video)` - 360° rotation
- `batch_render(mesh_list, views, output_dir)` - Batch rendering
- `export_interactive_html(mesh, output_file)` - Interactive HTML export

#### StreamlitPyVista

Streamlit integration wrapper.

**Methods:**
- `stpyvista_display(plotter, key)` - Display in Streamlit
- `create_interactive_viewer(mesh_data, key)` - Interactive viewer with controls
- `create_comparison_viewer(mesh_list)` - Side-by-side comparison
- `create_animation_viewer(mesh_sequence)` - Animation player
- `create_mesh_info_panel(mesh)` - Mesh statistics panel

### Examples

See `tests/test_visualization.py` for comprehensive usage examples.

## License

See LICENSE file for details.
