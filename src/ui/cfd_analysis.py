"""
CFD Analysis Studio UI
Main interface for CFD simulation workflow using Streamlit.
"""

import streamlit as st
from pathlib import Path
import time
import plotly.graph_objects as go
from typing import Optional, Dict, Any

# Import custom components
from src.ui.components.simulation_wizard import SimulationWizard
from src.ui.components.mesh_configurator import MeshConfigurator
from src.ui.components.boundary_condition_form import BCForm, SolverType, TurbulenceModel
from src.ui.components.results_viewer import ResultsViewer

# Import backend modules
from src.cfd.gmsh_mesher import GmshMesher, create_mesh_from_config
from src.cfd.pyfoam_wrapper import PyFoamWrapper, SolverType as PyFoamSolverType, TurbulenceModel as PyFoamTurbModel
from src.cfd.result_parser import ResultParser
from src.visualization.pyvista_viewer import PyVistaViewer


# Fluid properties database
FLUID_PROPERTIES = {
    "Air (20°C)": {
        "density": 1.225,
        "kinematic_viscosity": 1.5e-5,
        "thermal_conductivity": 0.025,
        "specific_heat": 1005.0
    },
    "Water (20°C)": {
        "density": 998.0,
        "kinematic_viscosity": 1.0e-6,
        "thermal_conductivity": 0.6,
        "specific_heat": 4182.0
    },
    "Oil (SAE 30)": {
        "density": 875.0,
        "kinematic_viscosity": 1.0e-4,
        "thermal_conductivity": 0.14,
        "specific_heat": 1900.0
    },
}


def main():
    """Main CFD Analysis UI application."""

    st.set_page_config(
        page_title="CFD Analysis Studio",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.header("🌊 CFD Analysis Studio")
    st.markdown("*Complete CFD Simulation Workflow - From Geometry to Results*")

    # Initialize wizard
    wizard = SimulationWizard(
        steps=["Model Selection", "Mesh Configuration", "Simulation Setup",
               "Run Simulation", "Results Visualization"],
        session_key="cfd_wizard"
    )

    # Display progress
    wizard.render_progress()

    # STEP 1: Model Selection
    wizard.render_step(0, lambda: render_model_selection(wizard))

    # STEP 2: Mesh Configuration
    wizard.render_step(1, lambda: render_mesh_configuration(wizard))

    # STEP 3: Simulation Setup
    wizard.render_step(2, lambda: render_simulation_setup(wizard))

    # STEP 4: Run Simulation
    wizard.render_step(3, lambda: render_simulation_run(wizard))

    # STEP 5: Results Visualization
    wizard.render_step(4, lambda: render_results_visualization(wizard))


def render_model_selection(wizard: SimulationWizard):
    """Render model selection step."""

    st.markdown("""
    **Select or upload your CAD model for CFD analysis.**

    You can either:
    - Upload a new STEP file
    - Select from previously uploaded models
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Upload STEP file
        uploaded_file = st.file_uploader(
            "Upload STEP File",
            type=["step", "stp"],
            help="Upload a STEP CAD file for analysis"
        )

        if uploaded_file:
            # Save uploaded file
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)

            file_path = upload_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            wizard.set_data("model_file", str(file_path))
            wizard.set_data("model_name", uploaded_file.name)

            st.success(f"✅ Uploaded: {uploaded_file.name}")

        # Or select from existing
        st.markdown("**Or select from existing models:**")

        upload_dir = Path("uploads")
        if upload_dir.exists():
            existing_files = list(upload_dir.glob("*.step")) + list(upload_dir.glob("*.stp"))

            if existing_files:
                selected_file = st.selectbox(
                    "Existing Models",
                    [""] + [f.name for f in existing_files]
                )

                if selected_file:
                    file_path = upload_dir / selected_file
                    wizard.set_data("model_file", str(file_path))
                    wizard.set_data("model_name", selected_file)
                    st.info(f"📁 Selected: {selected_file}")

    with col2:
        # Model preview placeholder
        st.subheader("Model Preview")

        model_file = wizard.get_data("model_file")

        if model_file:
            st.info(f"""
            **Model Information**

            📄 File: {wizard.get_data('model_name')}
            📍 Path: {model_file}

            *3D preview would appear here*
            """)
        else:
            st.warning("No model selected")

    # Navigation
    can_proceed = wizard.get_data("model_file") is not None
    wizard.render_navigation(can_proceed=can_proceed)


def render_mesh_configuration(wizard: SimulationWizard):
    """Render mesh configuration step."""

    model_file = wizard.get_data("model_file")

    if not model_file:
        st.error("⚠️ No model selected. Please go back and select a model.")
        wizard.render_navigation(can_proceed=False)
        return

    st.markdown("**Configure mesh parameters and refinement zones.**")

    # Initialize mesh configurator
    mesh_config = MeshConfigurator(session_key="cfd_mesh")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Global mesh settings
        mesh_config.render_global_settings()

        st.divider()

        # Refinement zones
        mesh_config.render_refinement_zones()

    with col2:
        # Mesh statistics
        mesh_config.render_statistics(
            geometry_volume=wizard.get_data("geometry_volume", 1.0),
            geometry_surface_area=wizard.get_data("geometry_surface_area", 6.0)
        )

        st.divider()

        # Generate mesh button
        st.subheader("Generate Mesh")

        if st.button("🔨 Generate Mesh", type="primary", use_container_width=True):
            with st.spinner("Generating mesh... This may take a few minutes."):
                # Get mesh configuration
                config = mesh_config.get_mesh_config()

                # Generate mesh
                output_dir = Path("meshes")
                output_dir.mkdir(exist_ok=True)
                mesh_file = output_dir / f"{Path(model_file).stem}.msh"

                success, stats = create_mesh_from_config(
                    Path(model_file),
                    config,
                    mesh_file
                )

                if success:
                    st.success("✅ Mesh generated successfully!")

                    # Display statistics
                    st.metric("Total Nodes", f"{stats.get('nodes', 0):,}")
                    st.metric("Total Elements", f"{stats.get('elements', 0):,}")

                    # Save to wizard data
                    wizard.set_data("mesh_file", str(mesh_file))
                    wizard.set_data("mesh_stats", stats)
                else:
                    st.error("❌ Mesh generation failed!")

        # Show mesh status
        if wizard.get_data("mesh_file"):
            st.success(f"✅ Mesh ready: {Path(wizard.get_data('mesh_file')).name}")

    # Navigation
    can_proceed = wizard.get_data("mesh_file") is not None
    wizard.render_navigation(can_proceed=can_proceed)


def render_simulation_setup(wizard: SimulationWizard):
    """Render simulation setup step."""

    mesh_file = wizard.get_data("mesh_file")

    if not mesh_file:
        st.error("⚠️ No mesh generated. Please go back and generate a mesh.")
        wizard.render_navigation(can_proceed=False)
        return

    st.markdown("**Configure simulation parameters and boundary conditions.**")

    # Solver settings
    st.subheader("⚙️ Solver Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        solver_type = st.selectbox(
            "Solver Type",
            ["simpleFoam (Steady)", "pimpleFoam (Transient)", "icoFoam (Laminar)",
             "buoyantSimpleFoam (Thermal)"],
            help="Select the OpenFOAM solver"
        )
        wizard.set_data("solver_type", solver_type.split()[0])

    with col2:
        turbulence_model = st.selectbox(
            "Turbulence Model",
            ["Laminar", "k-epsilon", "k-omega SST", "Spalart-Allmaras"],
            help="Select turbulence model"
        )
        wizard.set_data("turbulence_model", turbulence_model)

    with col3:
        max_iterations = st.number_input(
            "Max Iterations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="Maximum number of iterations"
        )
        wizard.set_data("max_iterations", max_iterations)

    st.divider()

    # Fluid properties
    st.subheader("💧 Fluid Properties")

    col1, col2 = st.columns(2)

    with col1:
        fluid_preset = st.selectbox(
            "Fluid Preset",
            ["Custom"] + list(FLUID_PROPERTIES.keys())
        )

        if fluid_preset != "Custom":
            fluid_props = FLUID_PROPERTIES[fluid_preset]
            wizard.set_data("fluid_properties", fluid_props)

            st.info(f"""
            **{fluid_preset}**
            - Density: {fluid_props['density']} kg/m³
            - Kinematic Viscosity: {fluid_props['kinematic_viscosity']:.2e} m²/s
            """)

    with col2:
        if fluid_preset == "Custom":
            density = st.number_input("Density (kg/m³)", value=1.225, step=0.1, format="%.3f")
            viscosity = st.number_input("Kinematic Viscosity (m²/s)", value=1.5e-5,
                                       step=1e-6, format="%.2e")

            wizard.set_data("fluid_properties", {
                "density": density,
                "kinematic_viscosity": viscosity
            })

    st.divider()

    # Boundary conditions
    st.subheader("🎯 Boundary Conditions")

    # Initialize BC form
    bc_form = BCForm(session_key="cfd_bc")

    # Set example patches (in real implementation, these would be extracted from mesh)
    example_patches = ["inlet", "outlet", "walls", "symmetry"]
    bc_form.set_patches(example_patches)

    # Render BC forms
    bc_form.render_forms()

    # Validate BCs
    is_valid, errors = bc_form.validate()

    if not is_valid:
        st.warning("⚠️ Boundary conditions incomplete:")
        for error in errors:
            st.write(f"  - {error}")

    # Save BC data
    wizard.set_data("boundary_conditions", bc_form.get_conditions())

    # Navigation
    wizard.render_navigation(can_proceed=is_valid)


def render_simulation_run(wizard: SimulationWizard):
    """Render simulation execution step."""

    st.markdown("**Run the CFD simulation and monitor progress.**")

    # Simulation settings
    col1, col2, col3 = st.columns(3)

    with col1:
        run_mode = st.radio(
            "Execution Mode",
            ["Local (OpenFOAM)", "Cloud (SimScale)"],
            help="Select where to run the simulation"
        )

    with col2:
        if run_mode == "Local (OpenFOAM)":
            parallel = st.checkbox("Parallel Execution")

            if parallel:
                num_cores = st.slider("Number of Cores", 2, 16, 4)
            else:
                num_cores = 1

            wizard.set_data("parallel", parallel)
            wizard.set_data("num_cores", num_cores)

    with col3:
        st.metric("Estimated Time", "~15 minutes")
        st.metric("Memory Required", "~2 GB")

    st.divider()

    # Start simulation button
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        start_button = st.button("▶️ Start Simulation", type="primary",
                                use_container_width=True)

    with col2:
        stop_button = st.button("⏹️ Stop Simulation", use_container_width=True)

    # Simulation status
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False

    if start_button:
        st.session_state.simulation_running = True
        wizard.set_data("simulation_status", "running")

    if stop_button:
        st.session_state.simulation_running = False
        wizard.set_data("simulation_status", "stopped")

    st.divider()

    # Progress display
    if st.session_state.simulation_running:
        st.subheader("📊 Simulation Progress")

        # Progress bar
        progress_placeholder = st.empty()
        status_placeholder = st.empty()

        # Simulated progress (in real implementation, this would come from solver)
        if "sim_progress" not in st.session_state:
            st.session_state.sim_progress = 0

        progress_value = min(st.session_state.sim_progress / 100, 1.0)
        progress_placeholder.progress(progress_value)
        status_placeholder.info(f"Iteration {st.session_state.sim_progress} / 1000")

        # Residuals plot
        st.subheader("📉 Residuals")

        # Generate sample residual data
        iterations = list(range(0, st.session_state.sim_progress + 1, 10))
        residuals_data = {
            "U": [1e-1 * (0.95 ** i) for i in range(len(iterations))],
            "p": [1e-1 * (0.93 ** i) for i in range(len(iterations))],
            "k": [1e-2 * (0.94 ** i) for i in range(len(iterations))],
        }

        fig = go.Figure()

        for field, values in residuals_data.items():
            fig.add_trace(go.Scatter(
                x=iterations,
                y=values,
                mode='lines',
                name=field,
                line=dict(width=2)
            ))

        fig.update_layout(
            yaxis_type="log",
            xaxis_title="Iteration",
            yaxis_title="Residual",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Convergence indicator
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("U Residual", f"{residuals_data['U'][-1]:.2e}",
                     delta="Converging" if residuals_data['U'][-1] < 1e-4 else "")

        with col2:
            st.metric("p Residual", f"{residuals_data['p'][-1]:.2e}",
                     delta="Converging" if residuals_data['p'][-1] < 1e-4 else "")

        with col3:
            convergence_status = all(v[-1] < 1e-4 for v in residuals_data.values())
            if convergence_status:
                st.success("✅ Converged!")
                wizard.set_data("simulation_complete", True)
                st.session_state.simulation_running = False
            else:
                st.info("⏳ Running...")

        # Auto-increment progress (simulate)
        if st.session_state.sim_progress < 1000:
            st.session_state.sim_progress += 10
            time.sleep(0.1)
            st.rerun()

    elif wizard.get_data("simulation_complete"):
        st.success("✅ Simulation completed successfully!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Status", "Converged")
        with col2:
            st.metric("Iterations", "1000")
        with col3:
            st.metric("Runtime", "12:34")
        with col4:
            st.metric("Final Residual", "8.2e-5")

    else:
        st.info("👆 Click 'Start Simulation' to begin CFD analysis")

    # Navigation
    can_proceed = wizard.get_data("simulation_complete", False)
    wizard.render_navigation(can_proceed=can_proceed)


def render_results_visualization(wizard: SimulationWizard):
    """Render results visualization step."""

    if not wizard.get_data("simulation_complete"):
        st.error("⚠️ Simulation not complete. Please run the simulation first.")
        wizard.render_navigation(can_proceed=False)
        return

    st.markdown("**Visualize and analyze simulation results.**")

    # Initialize results viewer
    results_viewer = ResultsViewer(session_key="cfd_results")

    # Available fields (would come from simulation results)
    available_fields = ["Velocity", "Pressure", "Temperature", "Turbulent Kinetic Energy"]

    col1, col2 = st.columns([2, 1])

    with col1:
        # 3D Visualization area
        results_viewer.render_visualization_placeholder(mesh_loaded=True)

    with col2:
        # Field selector
        results_viewer.render_field_selector(available_fields)

        st.divider()

        # Slice controls
        results_viewer.render_slice_controls(bounds=(-1, 1, -1, 1, -1, 1))

        st.divider()

        # Display settings
        results_viewer.render_display_settings()

    st.divider()

    # Field statistics
    import numpy as np
    dummy_field_data = np.random.rand(1000)  # Placeholder
    results_viewer.render_statistics(dummy_field_data)

    st.divider()

    # Animation controls (for transient simulations)
    # results_viewer.render_animation_controls(num_timesteps=10)

    # Export controls
    results_viewer.render_export_controls(results_available=True)

    # Final navigation
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 New Analysis", use_container_width=True):
            wizard.reset()
            st.rerun()

    with col2:
        if st.button("📥 Download Report", use_container_width=True):
            st.success("Report generation initiated!")


def render():
    """Render function for tab integration"""
    # For now, call the main function directly
    # In future, this can be refactored to work better within tabs
    st.header('🌊 CFD Analysis Studio')

    st.markdown(
        """
        Run computational fluid dynamics simulations with AI-assisted setup.
        """
    )

    # Placeholder for now
    st.info("🚧 CFD Analysis UI is being integrated. Core functionality available through main() function.")


if __name__ == "__main__":
    main()
