"""
PyFoam Wrapper Module

Provides OpenFOAM case setup, execution, and monitoring capabilities.
"""

import os
import shutil
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyFoamWrapper:
    """
    OpenFOAM case manager and simulation runner.

    This class provides methods to:
    - Create OpenFOAM case structures
    - Convert meshes to OpenFOAM format
    - Set boundary conditions and fluid properties
    - Run simulations (serial or parallel)
    - Monitor convergence
    """

    # Standard fluid properties
    FLUID_PROPERTIES = {
        "air": {
            "density": 1.225,  # kg/m^3
            "kinematic_viscosity": 1.5e-5,  # m^2/s
            "dynamic_viscosity": 1.8375e-5,  # Pa·s
        },
        "water": {
            "density": 1000.0,  # kg/m^3
            "kinematic_viscosity": 1.0e-6,  # m^2/s
            "dynamic_viscosity": 0.001,  # Pa·s
        },
        "oil": {
            "density": 900.0,  # kg/m^3
            "kinematic_viscosity": 1.0e-4,  # m^2/s
            "dynamic_viscosity": 0.09,  # Pa·s
        },
    }

    # Turbulence models
    TURBULENCE_MODELS = {
        "k-epsilon": "kEpsilon",
        "k-omega-sst": "kOmegaSST",
        "spalart-allmaras": "SpalartAllmaras",
        "laminar": "laminar",
    }

    def __init__(self, openfoam_env: Optional[Dict[str, str]] = None):
        """
        Initialize PyFoamWrapper.

        Args:
            openfoam_env: OpenFOAM environment variables (optional)
        """
        self.openfoam_env = openfoam_env or {}
        self._check_openfoam_installation()

    def _check_openfoam_installation(self) -> bool:
        """
        Check if OpenFOAM is properly installed.

        Returns:
            True if OpenFOAM is found
        """
        try:
            result = subprocess.run(
                ["which", "simpleFoam"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("OpenFOAM installation found")
                return True
            else:
                logger.warning(
                    "OpenFOAM not found in PATH. "
                    "Make sure to source OpenFOAM environment before using."
                )
                return False
        except Exception as e:
            logger.warning(f"Could not verify OpenFOAM installation: {str(e)}")
            return False

    def create_case(
        self,
        case_dir: str,
        solver_type: str = "simpleFoam",
        template_case: Optional[str] = None
    ) -> str:
        """
        Create OpenFOAM case structure.

        Args:
            case_dir: Directory for the case
            solver_type: OpenFOAM solver ('simpleFoam', 'pimpleFoam', etc.)
            template_case: Path to template case (if any)

        Returns:
            Path to created case directory

        Raises:
            ValueError: If solver type is invalid
        """
        case_path = Path(case_dir)

        # Create case directory
        case_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creating OpenFOAM case in: {case_path}")

        # If template provided, copy it
        if template_case and os.path.exists(template_case):
            logger.info(f"Copying template case from: {template_case}")
            for item in os.listdir(template_case):
                src = os.path.join(template_case, item)
                dst = os.path.join(case_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        else:
            # Create standard case structure
            self._create_case_structure(case_path, solver_type)

        return str(case_path)

    def _create_case_structure(self, case_path: Path, solver_type: str):
        """
        Create standard OpenFOAM case directory structure.

        Args:
            case_path: Path to case directory
            solver_type: Solver type
        """
        # Create directories
        dirs = ["0", "constant", "system"]
        for d in dirs:
            (case_path / d).mkdir(exist_ok=True)

        # Create subdirectories
        (case_path / "constant" / "polyMesh").mkdir(exist_ok=True)
        (case_path / "constant" / "triSurface").mkdir(exist_ok=True)

        # Create basic control dict
        self._write_control_dict(case_path, solver_type)

        # Create basic fvSchemes
        self._write_fv_schemes(case_path)

        # Create basic fvSolution
        self._write_fv_solution(case_path, solver_type)

        logger.info("Case structure created")

    def _write_control_dict(self, case_path: Path, solver_type: str):
        """Write controlDict file."""
        control_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     {solver_type};

startFrom       startTime;

startTime       0;

stopAt          endTime;

endTime         1000;

deltaT          1;

writeControl    timeStep;

writeInterval   100;

purgeWrite      2;

writeFormat     ascii;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""
        with open(case_path / "system" / "controlDict", "w") as f:
            f.write(control_dict)

    def _write_fv_schemes(self, case_path: Path):
        """Write fvSchemes file."""
        fv_schemes = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method meshWave;
}

// ************************************************************************* //
"""
        with open(case_path / "system" / "fvSchemes", "w") as f:
            f.write(fv_schemes)

    def _write_fv_solution(self, case_path: Path, solver_type: str):
        """Write fvSolution file."""
        fv_solution = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }

    "(k|epsilon|omega|nuTilda)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
        p               1e-4;
        U               1e-4;
        "(k|epsilon|omega|nuTilda)" 1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        "(k|epsilon|omega|nuTilda)" 0.7;
    }
}

// ************************************************************************* //
"""
        with open(case_path / "system" / "fvSolution", "w") as f:
            f.write(fv_solution)

    def convert_mesh(
        self,
        mesh_file: str,
        case_dir: str,
        mesh_format: str = "gmsh"
    ) -> bool:
        """
        Convert mesh to OpenFOAM format.

        Args:
            mesh_file: Path to mesh file
            case_dir: OpenFOAM case directory
            mesh_format: Mesh format ('gmsh', 'fluent', 'star', 'ansys')

        Returns:
            True if conversion successful

        Raises:
            FileNotFoundError: If mesh file doesn't exist
            RuntimeError: If conversion fails
        """
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

        case_path = Path(case_dir)
        if not case_path.exists():
            raise ValueError(f"Case directory doesn't exist: {case_dir}")

        logger.info(f"Converting mesh from {mesh_format} format...")

        try:
            if mesh_format == "gmsh":
                # Use gmshToFoam
                cmd = ["gmshToFoam", mesh_file, "-case", case_dir]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=case_dir,
                    env={**os.environ, **self.openfoam_env}
                )

                if result.returncode != 0:
                    logger.error(f"gmshToFoam failed: {result.stderr}")
                    raise RuntimeError(f"Mesh conversion failed: {result.stderr}")

                logger.info("Mesh converted successfully")
                return True

            else:
                raise ValueError(f"Unsupported mesh format: {mesh_format}")

        except Exception as e:
            logger.error(f"Mesh conversion failed: {str(e)}")
            raise RuntimeError(f"Failed to convert mesh: {str(e)}")

    def set_boundary_conditions(
        self,
        case_dir: str,
        bc_dict: Dict[str, Dict[str, Any]]
    ):
        """
        Set boundary conditions for the case.

        Args:
            case_dir: OpenFOAM case directory
            bc_dict: Dictionary of boundary conditions
                Example:
                {
                    "inlet": {
                        "U": {"type": "fixedValue", "value": [10, 0, 0]},
                        "p": {"type": "zeroGradient"},
                        "k": {"type": "fixedValue", "value": 0.1},
                        "epsilon": {"type": "fixedValue", "value": 0.01}
                    },
                    "outlet": {
                        "U": {"type": "zeroGradient"},
                        "p": {"type": "fixedValue", "value": 0}
                    },
                    "walls": {
                        "U": {"type": "noSlip"},
                        "p": {"type": "zeroGradient"}
                    }
                }
        """
        case_path = Path(case_dir)
        zero_dir = case_path / "0"
        zero_dir.mkdir(exist_ok=True)

        # Determine which fields are needed
        all_fields = set()
        for patch_bcs in bc_dict.values():
            all_fields.update(patch_bcs.keys())

        # Write boundary condition files for each field
        for field in all_fields:
            self._write_field_file(case_dir, field, bc_dict)

        logger.info("Boundary conditions set")

    def _write_field_file(
        self,
        case_dir: str,
        field: str,
        bc_dict: Dict[str, Dict[str, Any]]
    ):
        """
        Write boundary condition file for a field.

        Args:
            case_dir: Case directory
            field: Field name (U, p, k, epsilon, etc.)
            bc_dict: Boundary conditions dictionary
        """
        case_path = Path(case_dir)

        # Determine field class and dimensions
        field_info = {
            "U": ("volVectorField", "[0 1 -1 0 0 0 0]"),
            "p": ("volScalarField", "[0 2 -2 0 0 0 0]"),
            "k": ("volScalarField", "[0 2 -2 0 0 0 0]"),
            "epsilon": ("volScalarField", "[0 2 -3 0 0 0 0]"),
            "omega": ("volScalarField", "[0 0 -1 0 0 0 0]"),
            "nut": ("volScalarField", "[0 2 -1 0 0 0 0]"),
            "nuTilda": ("volScalarField", "[0 2 -1 0 0 0 0]"),
        }

        field_class, dimensions = field_info.get(field, ("volScalarField", "[0 0 0 0 0 0 0]"))

        # Build boundary field entries
        boundary_entries = []
        for patch_name, patch_bcs in bc_dict.items():
            if field in patch_bcs:
                bc = patch_bcs[field]
                bc_type = bc.get("type", "zeroGradient")

                entry = f"    {patch_name}\n    {{\n"
                entry += f"        type            {bc_type};\n"

                # Add value if present
                if "value" in bc:
                    value = bc["value"]
                    if isinstance(value, list):
                        entry += f"        value           uniform ({' '.join(map(str, value))});\n"
                    else:
                        entry += f"        value           uniform {value};\n"

                entry += "    }\n"
                boundary_entries.append(entry)

        # Write field file
        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {field_class};
    object      {field};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      {dimensions};

internalField   uniform {"(0 0 0)" if field == "U" else "0"};

boundaryField
{{
{''.join(boundary_entries)}
}}

// ************************************************************************* //
"""

        with open(case_path / "0" / field, "w") as f:
            f.write(content)

    def set_fluid_properties(
        self,
        case_dir: str,
        fluid: str = "air",
        turbulence_model: str = "k-epsilon",
        custom_properties: Optional[Dict[str, float]] = None
    ):
        """
        Set fluid properties and turbulence model.

        Args:
            case_dir: OpenFOAM case directory
            fluid: Fluid type ('air', 'water', 'oil') or 'custom'
            turbulence_model: Turbulence model ('k-epsilon', 'k-omega-sst', 'laminar')
            custom_properties: Custom fluid properties (if fluid='custom')
        """
        case_path = Path(case_dir)
        constant_dir = case_path / "constant"
        constant_dir.mkdir(exist_ok=True)

        # Get fluid properties
        if custom_properties:
            props = custom_properties
        elif fluid in self.FLUID_PROPERTIES:
            props = self.FLUID_PROPERTIES[fluid]
        else:
            raise ValueError(f"Unknown fluid: {fluid}")

        # Write transport properties
        self._write_transport_properties(case_dir, props)

        # Write turbulence properties
        self._write_turbulence_properties(case_dir, turbulence_model)

        logger.info(f"Fluid properties set for {fluid} with {turbulence_model} turbulence model")

    def _write_transport_properties(self, case_dir: str, props: Dict[str, float]):
        """Write transportProperties file."""
        nu = props.get("kinematic_viscosity", 1.5e-5)

        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

nu              {nu};

// ************************************************************************* //
"""

        with open(Path(case_dir) / "constant" / "transportProperties", "w") as f:
            f.write(content)

    def _write_turbulence_properties(self, case_dir: str, model: str):
        """Write turbulenceProperties file."""
        turbulence_model = self.TURBULENCE_MODELS.get(model, "kEpsilon")
        simulation_type = "RAS" if model != "laminar" else "laminar"

        content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  {simulation_type};

RAS
{{
    RASModel        {turbulence_model};
    turbulence      on;
    printCoeffs     on;
}}

// ************************************************************************* //
"""

        with open(Path(case_dir) / "constant" / "turbulenceProperties", "w") as f:
            f.write(content)

    def run_simulation(
        self,
        case_dir: str,
        solver: str = "simpleFoam",
        parallel: bool = False,
        cores: int = 4,
        background: bool = False
    ) -> Dict[str, Any]:
        """
        Run OpenFOAM simulation.

        Args:
            case_dir: OpenFOAM case directory
            solver: Solver name ('simpleFoam', 'pimpleFoam', etc.)
            parallel: Run in parallel mode
            cores: Number of cores for parallel run
            background: Run in background

        Returns:
            Dictionary with run information:
                - success: bool
                - return_code: int
                - log_file: str
                - runtime: float (seconds)

        Raises:
            RuntimeError: If simulation fails
        """
        case_path = Path(case_dir)
        if not case_path.exists():
            raise ValueError(f"Case directory doesn't exist: {case_dir}")

        logger.info(f"Running {solver} simulation...")
        start_time = time.time()

        log_file = case_path / f"{solver}.log"

        try:
            if parallel:
                # Decompose domain
                self._decompose_domain(case_dir, cores)

                # Run parallel
                cmd = ["mpirun", "-np", str(cores), solver, "-parallel", "-case", case_dir]
            else:
                cmd = [solver, "-case", case_dir]

            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=case_dir,
                    env={**os.environ, **self.openfoam_env}
                )

            runtime = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"Simulation completed successfully in {runtime:.2f}s")
                success = True
            else:
                logger.error(f"Simulation failed with return code {result.returncode}")
                success = False

            return {
                "success": success,
                "return_code": result.returncode,
                "log_file": str(log_file),
                "runtime": runtime
            }

        except Exception as e:
            logger.error(f"Simulation execution failed: {str(e)}")
            raise RuntimeError(f"Failed to run simulation: {str(e)}")

    def _decompose_domain(self, case_dir: str, cores: int):
        """
        Decompose domain for parallel execution.

        Args:
            case_dir: Case directory
            cores: Number of cores
        """
        # Write decomposeParDict
        decompose_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {cores};

method          scotch;

// ************************************************************************* //
"""

        with open(Path(case_dir) / "system" / "decomposeParDict", "w") as f:
            f.write(decompose_dict)

        # Run decomposePar
        logger.info(f"Decomposing domain into {cores} parts...")
        subprocess.run(
            ["decomposePar", "-case", case_dir],
            capture_output=True,
            cwd=case_dir,
            env={**os.environ, **self.openfoam_env}
        )

    def check_convergence(
        self,
        case_dir: str,
        residual_threshold: float = 1e-4
    ) -> Dict[str, Any]:
        """
        Check simulation convergence.

        Args:
            case_dir: OpenFOAM case directory
            residual_threshold: Residual threshold for convergence

        Returns:
            Dictionary with convergence information:
                - converged: bool
                - final_residuals: dict
                - iterations: int
        """
        case_path = Path(case_dir)

        # Try to find log file
        log_files = list(case_path.glob("*.log"))
        if not log_files:
            logger.warning("No log file found")
            return {"converged": False, "error": "No log file found"}

        log_file = log_files[0]

        # Parse residuals from log
        residuals = self._parse_residuals(log_file)

        if not residuals:
            return {"converged": False, "error": "Could not parse residuals"}

        # Check if final residuals are below threshold
        final_residuals = {k: v[-1] if v else float('inf') for k, v in residuals.items()}
        converged = all(r < residual_threshold for r in final_residuals.values())

        return {
            "converged": converged,
            "final_residuals": final_residuals,
            "iterations": len(next(iter(residuals.values()))),
            "residuals_history": residuals
        }

    def _parse_residuals(self, log_file: Path) -> Dict[str, List[float]]:
        """
        Parse residuals from log file.

        Args:
            log_file: Path to log file

        Returns:
            Dictionary mapping field names to residual values
        """
        residuals = {}

        try:
            with open(log_file, "r") as f:
                for line in f:
                    # Look for residual lines
                    # Example: Solving for Ux, Initial residual = 0.123
                    if "Solving for" in line and "Initial residual" in line:
                        parts = line.split()
                        field = parts[2].rstrip(',')
                        residual = float(parts[-1])

                        if field not in residuals:
                            residuals[field] = []
                        residuals[field].append(residual)

        except Exception as e:
            logger.warning(f"Failed to parse residuals: {str(e)}")

        return residuals
