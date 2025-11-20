# CFD Module

Comprehensive Computational Fluid Dynamics pipeline for GenAI CAD/CFD Studio.

## Overview

This module provides a complete CFD workflow including:

- **Mesh Generation**: Using Gmsh with support for refinement zones
- **OpenFOAM Integration**: Case setup, execution, and monitoring
- **Result Parsing**: Extract and analyze simulation results
- **Cloud CFD**: SimScale API integration for cloud-based simulations

## Installation

### Prerequisites

1. **Python 3.9+**
2. **Gmsh** (for mesh generation):
   ```bash
   pip install gmsh
   ```

3. **OpenFOAM** (for local simulations):
   - Linux: Follow [OpenFOAM installation guide](https://openfoam.org/download/)
   - Windows: Use WSL2 with OpenFOAM
   - macOS: Use Docker with OpenFOAM

4. **Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Mesh Generation

```python
from cfd.gmsh_mesher import GmshMesher

# Initialize mesher
mesher = GmshMesher(verbose=True)

# Generate mesh from STEP file
mesh_file = mesher.generate_mesh(
    step_file="geometry.step",
    mesh_size=0.1,
    refinement_zones=[
        {
            "type": "box",
            "x_min": 0, "x_max": 1,
            "y_min": 0, "y_max": 1,
            "z_min": 0, "z_max": 1,
            "size": 0.05
        }
    ],
    algorithm="delaunay",
    optimize=True
)

# Get mesh statistics
stats = mesher.get_mesh_stats(mesh_file)
print(f"Nodes: {stats['n_nodes']}, Elements: {stats['n_elements']}")

# Visualize mesh
plotter = mesher.visualize_mesh(mesh_file, show_quality=True)
plotter.show()
```

### 2. OpenFOAM Case Setup

```python
from cfd.pyfoam_wrapper import PyFoamWrapper

# Initialize wrapper
foam = PyFoamWrapper()

# Create case
case_dir = foam.create_case(
    case_dir="./my_simulation",
    solver_type="simpleFoam"
)

# Convert mesh to OpenFOAM format
foam.convert_mesh(
    mesh_file=mesh_file,
    case_dir=case_dir,
    mesh_format="gmsh"
)

# Set boundary conditions
bc_dict = {
    "inlet": {
        "U": {"type": "fixedValue", "value": [10, 0, 0]},
        "p": {"type": "zeroGradient"},
        "k": {"type": "fixedValue", "value": 0.1},
        "epsilon": {"type": "fixedValue", "value": 0.01}
    },
    "outlet": {
        "U": {"type": "zeroGradient"},
        "p": {"type": "fixedValue", "value": 0},
        "k": {"type": "zeroGradient"},
        "epsilon": {"type": "zeroGradient"}
    },
    "walls": {
        "U": {"type": "noSlip"},
        "p": {"type": "zeroGradient"},
        "k": {"type": "kqRWallFunction", "value": 0.1},
        "epsilon": {"type": "epsilonWallFunction", "value": 0.01}
    }
}

foam.set_boundary_conditions(case_dir, bc_dict)

# Set fluid properties
foam.set_fluid_properties(
    case_dir=case_dir,
    fluid="air",
    turbulence_model="k-epsilon"
)
```

### 3. Run Simulation

```python
# Run simulation (serial)
result = foam.run_simulation(
    case_dir=case_dir,
    solver="simpleFoam",
    parallel=False
)

if result["success"]:
    print(f"Simulation completed in {result['runtime']:.2f}s")
else:
    print(f"Simulation failed: {result['return_code']}")

# Check convergence
convergence = foam.check_convergence(case_dir)
if convergence["converged"]:
    print("Simulation converged!")
    print(f"Final residuals: {convergence['final_residuals']}")
else:
    print("Simulation did not converge")
```

### 4. Parse Results

```python
from cfd.result_parser import ResultParser

# Initialize parser
parser = ResultParser(case_dir)

# Load latest results
results = parser.load_results(time_step="latest")

# Extract velocity field
U = parser.extract_field(results, field="U")
print(f"Velocity field shape: {U.shape}")

# Compute statistics
stats = parser.compute_statistics(results, field="U")
print(f"Velocity magnitude: min={stats['magnitude_min']:.2f}, "
      f"max={stats['magnitude_max']:.2f}, mean={stats['magnitude_mean']:.2f}")

# Calculate forces
forces = parser.calculate_forces(
    results=results,
    patch_names=["walls"],
    rho=1.225,  # Air density
    U_inf=10.0,  # Reference velocity
    A_ref=1.0    # Reference area
)
print(f"Drag coefficient: {forces['Cd']:.4f}")
print(f"Lift coefficient: {forces['Cl']:.4f}")

# Export to VTK for visualization
vtk_file = parser.create_vtk_export(
    results=results,
    output_file="results.vtk"
)
```

### 5. Cloud CFD with SimScale (Optional)

```python
from cfd.simscale_api import SimScaleConnector

# Initialize connector (requires API key)
simscale = SimScaleConnector(api_key="your_api_key")
# Or set SIMSCALE_API_KEY environment variable

# Create project
project_id = simscale.create_project(
    name="My CFD Analysis",
    description="Flow simulation"
)

# Upload geometry
geometry_id = simscale.upload_geometry(
    project_id=project_id,
    geometry_file="geometry.step",
    geometry_name="FlowDomain"
)

# Create simulation configuration
config = simscale.create_standard_cfd_config(
    name="Flow Analysis",
    inlet_velocity=10.0,
    fluid="air",
    turbulence_model="K_EPSILON"
)

# Create simulation
sim_id = simscale.create_simulation(
    project_id=project_id,
    geometry_id=geometry_id,
    config=config
)

# Run simulation
run_info = simscale.run_simulation(
    project_id=project_id,
    simulation_id=sim_id,
    wait_for_completion=True,
    poll_interval=30
)

# Download results
result_files = simscale.download_results(
    project_id=project_id,
    simulation_id=sim_id,
    run_id=run_info["run_id"],
    output_dir="./simscale_results"
)
```

## Module Structure

```
src/cfd/
├── __init__.py           # Module initialization
├── gmsh_mesher.py        # Mesh generation with Gmsh
├── pyfoam_wrapper.py     # OpenFOAM case management
├── result_parser.py      # Result parsing and post-processing
└── simscale_api.py       # SimScale cloud CFD integration
```

## Features

### GmshMesher

- Load STEP/IGES geometry files
- Generate 3D tetrahedral meshes
- Define refinement zones (box, distance-based)
- Multiple meshing algorithms (Delaunay, Frontal, MMG3D)
- Mesh optimization
- Quality metrics and statistics
- Visualization with PyVista
- Export to multiple formats (VTK, STL, etc.)

### PyFoamWrapper

- Create OpenFOAM case structure
- Convert meshes (Gmsh, Fluent, STAR-CCM+)
- Set boundary conditions (inlet, outlet, walls)
- Configure fluid properties (air, water, oil, custom)
- Turbulence models (k-epsilon, k-omega SST, Spalart-Allmaras, laminar)
- Run simulations (serial or parallel)
- Monitor convergence
- Support for multiple solvers (simpleFoam, pimpleFoam, etc.)

### ResultParser

- Load OpenFOAM results from time directories
- Extract field data (velocity, pressure, turbulence)
- Calculate forces and coefficients (drag, lift)
- Compute field statistics
- Export to VTK/ParaView format
- Time series analysis
- Line and surface data extraction

### SimScaleConnector

- Authenticate with SimScale API
- Create and manage projects
- Upload geometry files
- Create and configure simulations
- Run cloud simulations
- Monitor simulation status
- Download results

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/test_cfd.py -v

# Run specific test class
python -m pytest tests/test_cfd.py::TestGmshMesher -v

# Run with coverage
python -m pytest tests/test_cfd.py --cov=src/cfd --cov-report=html
```

## Examples

See the `examples/` directory for complete workflow examples:

- `basic_flow_simulation.py`: Simple flow over cylinder
- `turbulent_pipe_flow.py`: Turbulent flow in pipe
- `external_aerodynamics.py`: External aerodynamics analysis
- `cloud_cfd_workflow.py`: Cloud CFD with SimScale

## Requirements

- Python >= 3.9
- gmsh >= 4.13.0
- meshio >= 5.3.0
- pyvista >= 0.43.0
- numpy >= 1.24.0
- requests >= 2.31.0 (for SimScale)
- OpenFOAM (for local simulations)

## Troubleshooting

### OpenFOAM not found

Make sure OpenFOAM is installed and sourced:

```bash
# For OpenFOAM v11
source /opt/openfoam11/etc/bashrc

# For OpenFOAM ESI version
source /usr/lib/openfoam/openfoam2312/etc/bashrc
```

### Gmsh import error

Install gmsh Python API:

```bash
pip install gmsh
```

### PyVista display issues

For headless systems or remote servers:

```python
import pyvista as pv
pv.start_xvfb()  # Use virtual framebuffer
```

### SimScale API authentication

Get your API key from [SimScale API documentation](https://www.simscale.com/docs/simscale/api/) and set it as an environment variable:

```bash
export SIMSCALE_API_KEY="your_api_key_here"
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

See LICENSE file in the root directory.

## References

- [OpenFOAM Documentation](https://www.openfoam.com/documentation/)
- [Gmsh Documentation](https://gmsh.info/doc/texinfo/gmsh.html)
- [SimScale API Documentation](https://www.simscale.com/docs/simscale/api/)
- [PyVista Documentation](https://docs.pyvista.org/)
