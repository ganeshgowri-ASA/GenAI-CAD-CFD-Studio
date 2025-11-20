# CFD Analysis Studio - User Guide

## Overview

The CFD Analysis Studio provides a complete, wizard-based interface for running Computational Fluid Dynamics (CFD) simulations. This guide walks you through the entire workflow from model selection to results visualization.

## Features

- 🎯 **5-Step Wizard Workflow**: Intuitive step-by-step process
- 📐 **Flexible Mesh Generation**: Global and local refinement control
- 🌊 **Multiple Solver Support**: simpleFoam, pimpleFoam, and more
- 🎨 **Interactive 3D Visualization**: PyVista-based results viewer
- 📊 **Real-time Monitoring**: Live residuals and convergence tracking
- ☁️ **Cloud or Local**: Run locally (OpenFOAM) or in cloud (SimScale)

## Workflow Steps

### Step 1: Model Selection

**Purpose**: Load your CAD geometry for CFD analysis

**Options**:
- Upload a new STEP file
- Select from previously uploaded models

**Actions**:
1. Click "Upload STEP File" to browse for your CAD model
2. Or select from the dropdown of existing models
3. Preview the model in the 3D viewer

**Tips**:
- Ensure your CAD model is watertight (closed volume)
- Remove unnecessary internal details for faster meshing
- Check that all surfaces are properly oriented

### Step 2: Mesh Configuration

**Purpose**: Generate computational mesh for simulation

**Global Mesh Settings**:
- **Mesh Quality**: Choose from Coarse, Medium, Fine, or Very Fine
  - Coarse: Fast, less accurate (element size ~0.1m)
  - Medium: Balanced (element size ~0.05m)
  - Fine: Slower, more accurate (element size ~0.02m)
  - Very Fine: Highest accuracy (element size ~0.01m)

- **Advanced Settings**:
  - Global Element Size: Target size for mesh elements
  - Minimum Element Size: Smallest allowed element
  - Maximum Element Size: Largest allowed element
  - Growth Rate: How quickly elements grow in size (1.1-2.0)

**Refinement Zones**:
Add local refinement in regions of interest:

1. **Box Refinement**:
   - Define rectangular region with center and dimensions
   - Specify smaller element size for this region
   - Good for: wakes, shear layers, specific areas of interest

2. **Sphere Refinement**:
   - Define spherical region with center and radius
   - Good for: around objects, point sources

3. **Cylinder Refinement**:
   - Define cylindrical region
   - Good for: pipe flows, rotating regions

**Mesh Statistics**:
- Monitor estimated node and element counts
- Check memory requirements
- Verify element counts are reasonable for your hardware

**Actions**:
1. Select mesh quality preset or customize settings
2. Add refinement zones if needed
3. Click "Generate Mesh"
4. Wait for mesh generation (may take several minutes)
5. Review mesh statistics

### Step 3: Simulation Setup

**Purpose**: Configure solver settings and boundary conditions

#### Solver Configuration

**Solver Type**:
- `simpleFoam`: Steady-state incompressible turbulent flows
- `pimpleFoam`: Transient incompressible flows
- `icoFoam`: Laminar incompressible flows
- `buoyantSimpleFoam`: Steady-state with thermal effects

**Turbulence Model**:
- **Laminar**: No turbulence modeling (Re < 2300)
- **k-epsilon**: Standard model, robust, wall functions
- **k-omega SST**: Better near-wall behavior, separation
- **Spalart-Allmaras**: Single equation, good for aerospace

**Max Iterations**: Number of solver iterations (typically 1000-5000)

#### Fluid Properties

**Presets**:
- Air (20°C): ρ=1.225 kg/m³, ν=1.5×10⁻⁵ m²/s
- Water (20°C): ρ=998 kg/m³, ν=1.0×10⁻⁶ m²/s
- Oil (SAE 30): ρ=875 kg/m³, ν=1.0×10⁻⁴ m²/s
- Custom: Specify your own properties

#### Boundary Conditions

**Templates**:
Quick setup for common scenarios:
- External Flow (Wind Tunnel)
- Internal Flow (Pipe)
- Natural Convection

**Boundary Types**:

1. **Inlet**:
   - Velocity magnitude and direction
   - Turbulence intensity (typically 1-10%)
   - Turbulent viscosity ratio (typically 10)
   - Optional: Temperature

2. **Outlet**:
   - Static pressure (gauge, typically 0 Pa)
   - Backflow handling
   - Backflow turbulence intensity

3. **Wall**:
   - Motion: No-slip, Slip, Moving, Rotating
   - Surface roughness: Smooth or Rough
   - Thermal: Adiabatic, Fixed Temperature, Fixed Heat Flux
   - Optional: Wall velocity for moving walls

4. **Symmetry**:
   - No parameters needed
   - Use to reduce domain size

**Actions**:
1. Select solver type and turbulence model
2. Choose fluid properties or use preset
3. Configure boundary conditions for each patch
4. Use templates for quick setup
5. Validate that all boundaries are configured

### Step 4: Run Simulation

**Purpose**: Execute CFD simulation and monitor progress

#### Execution Settings

**Mode**:
- **Local (OpenFOAM)**: Run on your machine
  - Requires OpenFOAM installation
  - Full control over process
  - No data upload needed

- **Cloud (SimScale)**: Run in cloud
  - No local installation required
  - Access to more computing power
  - Requires internet connection

**Parallel Execution**:
- Enable for faster solving
- Select number of cores (2-16)
- Mesh is automatically decomposed

**Actions**:
1. Select execution mode
2. Configure parallel settings if desired
3. Click "Start Simulation"
4. Monitor progress in real-time

#### Monitoring

**Progress Indicators**:
- Overall progress bar
- Current iteration count
- Estimated time remaining

**Residuals Plot**:
- Live plotting of field residuals
- Logarithmic scale
- Monitor convergence trends
- Fields: U (velocity), p (pressure), k, omega, epsilon

**Convergence**:
- Simulation converges when residuals < 10⁻⁴
- Watch for steady decrease in residuals
- Divergence indicated by increasing residuals

**Controls**:
- Stop Simulation: Terminate solver
- Pause: (if supported)

### Step 5: Results Visualization

**Purpose**: Analyze and visualize simulation results

#### Field Selection

**Available Fields**:
- Velocity: Flow speed and direction
- Pressure: Static pressure distribution
- Temperature: Thermal field (if applicable)
- Turbulent Kinetic Energy: Turbulence intensity
- Vorticity: Rotational flow structures

**Visualization Types**:
- **Contours**: Colored surfaces showing field values
- **Vectors**: Arrows showing direction and magnitude
- **Streamlines**: Flow paths through domain
- **Iso-surfaces**: Surfaces of constant value

#### Visualization Controls

**Slice Plane**:
- Select axis (X, Y, or Z)
- Position slider to move plane
- Show field on plane

**Display Settings**:
- Color Map: viridis, jet, coolwarm, rainbow, etc.
- Show Mesh: Display mesh edges
- Show Edges: Highlight cell boundaries
- Value Range: Auto or manual min/max

**Vector Settings** (for vector plots):
- Arrow scale: Adjust arrow size
- Density: Number of vectors to display

**Streamline Settings**:
- Number of streamlines: 10-200
- Starting location: Automatic or manual

#### Field Statistics

**Summary**:
- Minimum value
- Maximum value
- Mean value
- Standard deviation

**Distribution**:
- Histogram of field values
- Identify outliers

#### Animation (Transient Results)

For time-dependent simulations:
- Time step slider
- Play/Pause controls
- Frame rate control
- Export as video

#### Export Options

**Formats**:
- **VTK (.vtk)**: ParaView compatible
- **VTU (.vtu)**: Unstructured grid format
- **EnSight (.case)**: EnSight Gold format
- **Tecplot (.plt)**: Tecplot format
- **CSV (.csv)**: Spreadsheet data

**Export Settings**:
- Include mesh geometry
- All fields or selected field
- Single time step or range
- Current view or full domain

**Actions**:
1. Select field to visualize
2. Choose visualization type
3. Configure display settings
4. Adjust slice plane or other controls
5. Review statistics
6. Export results as needed
7. Download report

## Best Practices

### Meshing
1. Start with coarse mesh to test setup
2. Refine mesh in high-gradient regions
3. Check mesh quality (aspect ratio, skewness)
4. Balance accuracy and computational cost

### Boundary Conditions
1. Use templates for quick setup
2. Verify inlet/outlet locations are correct
3. Check velocity magnitudes are realistic
4. Ensure pressure reference is set

### Solving
1. Monitor residuals from start
2. Look for steady convergence
3. If diverging, reduce under-relaxation
4. Save intermediate results

### Post-Processing
1. Check mass conservation
2. Verify physical reasonableness
3. Compare with experimental data if available
4. Create multiple views for reporting

## Troubleshooting

### Mesh Generation Fails
- Check CAD geometry is watertight
- Reduce refinement zone complexity
- Increase minimum element size
- Check for geometry errors

### Simulation Diverges
- Reduce under-relaxation factors
- Check boundary conditions
- Refine mesh in high-gradient regions
- Ensure physical properties are reasonable
- Start with lower velocity/simpler case

### Slow Convergence
- Increase number of iterations
- Adjust under-relaxation factors
- Improve mesh quality
- Use better initial conditions
- Try different turbulence model

### Results Look Wrong
- Verify boundary conditions
- Check mesh quality
- Ensure proper convergence
- Review solver settings
- Compare with simpler known cases

## Technical Details

### File Structure
```
project/
├── uploads/          # Uploaded STEP files
├── meshes/          # Generated mesh files (.msh)
├── cases/           # OpenFOAM case directories
├── results/         # Simulation results
└── exports/         # Exported visualizations
```

### Dependencies
- Python 3.8+
- Streamlit 1.28+
- PyVista 0.43+
- Gmsh 4.11+
- OpenFOAM v2112+ (for local execution)
- NumPy, Pandas, Plotly

### Performance Tips
1. Use parallel execution for large cases
2. Write results less frequently
3. Limit refinement zones
4. Balance mesh size vs accuracy
5. Monitor memory usage

## Examples

### External Flow Around Cylinder
1. Upload cylinder STEP file
2. Medium mesh with refinement box behind cylinder
3. simpleFoam with k-omega SST
4. Air at 10 m/s inlet, 0 Pa outlet
5. Run 2000 iterations
6. Visualize velocity and pressure contours

### Internal Pipe Flow
1. Upload pipe geometry
2. Fine mesh at bends
3. simpleFoam with k-epsilon
4. Water at 1 m/s inlet, 0 Pa outlet
5. Wall roughness 0.001 m
6. Streamlines to show flow patterns

### Heat Transfer
1. Upload heat sink geometry
2. Medium mesh with refinement at fins
3. buoyantSimpleFoam
4. Fixed temperature hot wall, ambient air inlet
5. Visualize temperature and velocity

## Support

For issues or questions:
- Check documentation at `/docs`
- Review test cases in `/tests`
- Submit issues on GitHub
- Contact support team

## License

See LICENSE file in repository root.
