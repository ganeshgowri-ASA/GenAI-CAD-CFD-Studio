"""
Demonstration of the PyVista Visualization Module.

This script showcases the key features of the visualization module including
CAD rendering, CFD visualization, mesh quality analysis, and export capabilities.
"""

import numpy as np
import pyvista as pv
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import (
    PyVistaViewer,
    PlotlyCharts,
    ExportRenderer,
    MeshOptimizer,
    optimize_for_display
)


def demo_cad_rendering():
    """Demonstrate CAD model rendering."""
    print("\n" + "="*60)
    print("DEMO 1: CAD Model Rendering")
    print("="*60)

    viewer = PyVistaViewer(theme='document')

    # Create sample geometries
    sphere = viewer.create_sample_mesh('sphere')
    cube = viewer.create_sample_mesh('cube')
    cylinder = viewer.create_sample_mesh('cylinder')

    # Render with custom configuration
    plotter = viewer.render_cad_model(sphere, plotter_config={
        'window_size': (1024, 768),
        'background': 'white',
        'color': 'lightblue',
        'show_edges': True,
        'opacity': 0.9,
        'lighting': True
    })

    # Add annotations
    labels = {
        'center': {
            'type': 'text',
            'position': (0, 0, 0),
            'text': 'Center',
            'color': 'red'
        },
        'radius': {
            'type': 'dimension',
            'position': (0, 0, 0),
            'end_position': (1, 0, 0),
            'text': 'R = 1.0',
            'color': 'black'
        }
    }

    viewer.add_annotations(plotter, labels)

    print("Displaying CAD model with annotations...")
    # plotter.show()  # Uncomment to display interactively
    plotter.close()


def demo_cfd_visualization():
    """Demonstrate CFD results visualization."""
    print("\n" + "="*60)
    print("DEMO 2: CFD Results Visualization")
    print("="*60)

    viewer = PyVistaViewer()

    # Create sample mesh with CFD data
    mesh = pv.Sphere(radius=2.0, theta_resolution=50, phi_resolution=50)

    # Simulate pressure field (distance from center)
    points = mesh.points
    distances = np.linalg.norm(points, axis=1)
    mesh['pressure'] = 100 * (1 - distances / distances.max())

    # Simulate temperature field
    mesh['temperature'] = 273 + 50 * np.sin(points[:, 0]) * np.cos(points[:, 1])

    # Simulate velocity field
    velocity = np.zeros((mesh.n_points, 3))
    velocity[:, 0] = -points[:, 1]  # Rotation around z-axis
    velocity[:, 1] = points[:, 0]
    velocity[:, 2] = 0.1 * points[:, 2]
    mesh['velocity'] = velocity

    # Render pressure field
    plotter = viewer.render_cfd_results(
        mesh,
        field='pressure',
        cmap='coolwarm',
        show_scalar_bar=True
    )

    print("Displaying pressure field...")
    # plotter.show()  # Uncomment to display interactively
    plotter.close()

    # Render velocity field with streamlines
    plotter2 = viewer.render_cfd_results(
        mesh,
        field='velocity',
        show_streamlines=False,  # Streamlines can be slow
        cmap='viridis'
    )

    print("Displaying velocity field...")
    # plotter2.show()  # Uncomment to display interactively
    plotter2.close()


def demo_mesh_quality():
    """Demonstrate mesh quality analysis."""
    print("\n" + "="*60)
    print("DEMO 3: Mesh Quality Analysis")
    print("="*60)

    viewer = PyVistaViewer()

    # Create mesh and convert to unstructured grid
    mesh = pv.Sphere(theta_resolution=20, phi_resolution=20)
    mesh = mesh.cast_to_unstructured_grid()

    # Render mesh quality
    plotter = viewer.render_mesh_quality(
        mesh,
        quality_metric='aspect_ratio',
        threshold=0.5
    )

    print("Displaying mesh quality visualization...")
    # plotter.show()  # Uncomment to display interactively
    plotter.close()


def demo_plotly_charts():
    """Demonstrate 2D plotting with Plotly."""
    print("\n" + "="*60)
    print("DEMO 4: 2D Charts with Plotly")
    print("="*60)

    charts = PlotlyCharts(theme='plotly_white')

    # Create sample residual data
    iterations = 100
    residual_data = {
        'iteration': list(range(1, iterations + 1)),
        'continuity': np.logspace(-2, -6, iterations),
        'x-velocity': np.logspace(-2, -6, iterations) * 1.2,
        'y-velocity': np.logspace(-2, -6, iterations) * 0.9,
        'energy': np.logspace(-2, -7, iterations),
    }

    # Plot residuals
    fig = charts.plot_residuals(residual_data, log_scale=True)
    print("Created residuals plot")
    # fig.show()  # Uncomment to display

    # Create convergence data
    convergence_data = {
        'iteration': list(range(1, 51)),
        'drag_coefficient': 0.5 + 0.1 * np.exp(-np.arange(50) / 10) + np.random.randn(50) * 0.01,
        'lift_coefficient': 0.3 + 0.05 * np.exp(-np.arange(50) / 15) + np.random.randn(50) * 0.005,
    }

    fig2 = charts.plot_convergence(convergence_data)
    print("Created convergence plot")
    # fig2.show()  # Uncomment to display

    # Statistics plot
    stats = {
        'Good Elements': 850,
        'Fair Elements': 120,
        'Poor Elements': 30,
    }

    fig3 = charts.plot_statistics(stats, chart_type='pie')
    print("Created statistics pie chart")
    # fig3.show()  # Uncomment to display


def demo_export_renderer():
    """Demonstrate high-quality export."""
    print("\n" + "="*60)
    print("DEMO 5: High-Quality Export")
    print("="*60)

    renderer = ExportRenderer()
    viewer = PyVistaViewer()

    # Create sample mesh
    mesh = viewer.create_sample_mesh('sphere')
    mesh['elevation'] = mesh.points[:, 2]

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # High-quality image export
    print("Rendering high-quality image...")
    renderer.render_high_quality(
        mesh,
        output_dir / "demo_highquality.png",
        resolution=(1920, 1080),
        anti_aliasing=True,
        ambient_occlusion=False,  # Disable for speed
        shadows=True,
        scalars='elevation',
        cmap='viridis'
    )

    # Batch render multiple views
    print("Batch rendering multiple views...")
    meshes = [
        (viewer.create_sample_mesh('sphere'), 'sphere'),
        (viewer.create_sample_mesh('cube'), 'cube'),
    ]

    files = renderer.batch_render(
        meshes,
        views=['iso', 'xy'],
        output_dir=output_dir,
        resolution=(800, 600)
    )

    print(f"Generated {len(files)} images in {output_dir}")


def demo_mesh_optimization():
    """Demonstrate mesh optimization utilities."""
    print("\n" + "="*60)
    print("DEMO 6: Mesh Optimization")
    print("="*60)

    optimizer = MeshOptimizer()

    # Create high-resolution mesh
    mesh = pv.Sphere(theta_resolution=100, phi_resolution=100)
    print(f"Original mesh: {mesh.n_cells:,} cells, {mesh.n_points:,} points")

    # Decimate mesh
    decimated = optimizer.decimate_mesh(mesh, target_reduction=0.8)
    print(f"Decimated mesh: {decimated.n_cells:,} cells")

    # Create LOD hierarchy
    lod_levels = optimizer.create_lod_hierarchy(mesh, n_levels=4)
    print(f"\nLOD Hierarchy:")
    for i, lod in enumerate(lod_levels):
        print(f"  Level {i}: {lod.n_cells:,} cells ({lod.memory_usage / 1024:.1f} KB)")

    # Optimize for display
    optimized = optimize_for_display(mesh, max_cells=10000, smooth=True)
    print(f"\nOptimized for display: {optimized.n_cells:,} cells")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("PyVista Visualization Module - Demonstrations")
    print("="*60)

    try:
        demo_cad_rendering()
        demo_cfd_visualization()
        demo_mesh_quality()
        demo_plotly_charts()
        demo_export_renderer()
        demo_mesh_optimization()

        print("\n" + "="*60)
        print("All demonstrations completed successfully!")
        print("="*60)
        print("\nNote: Uncomment plotter.show() and fig.show() calls")
        print("      to display visualizations interactively.")

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
