"""
Universal CAD File Converter

Supports conversion between multiple CAD and mesh formats:
- DXF/DWG (AutoCAD)
- STEP/IGES (ISO standards)
- STL (stereolithography)
- FCSTD (FreeCAD native)
- OBJ/PLY (3D mesh)
- GLTF/GLB (web 3D)
- Abaqus/Salome mesh formats

Uses:
- ezdxf for DXF/DWG
- trimesh for mesh operations
- meshio for mesh format conversion
- build123d for STEP/STL
- FreeCAD API (optional) for FCSTD
"""

import logging
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Optional imports
try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False
    logger.warning("ezdxf not installed. DXF/DWG support disabled.")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    logger.warning("trimesh not installed. Mesh conversion limited.")

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False
    logger.warning("meshio not installed. Advanced mesh formats unavailable.")

try:
    from build123d import Part, export_step, export_stl
    HAS_BUILD123D = True
except ImportError:
    HAS_BUILD123D = False
    logger.warning("build123d not installed. STEP/STL export limited.")

try:
    import pygltflib
    HAS_GLTF = True
except ImportError:
    HAS_GLTF = False
    logger.debug("pygltflib not installed. Using trimesh for GLTF.")


class FileFormat(Enum):
    """Supported file formats"""
    # CAD Formats
    DXF = "dxf"
    DWG = "dwg"
    STEP = "step"
    STP = "stp"
    IGES = "iges"
    IGS = "igs"

    # Mesh Formats
    STL = "stl"
    OBJ = "obj"
    PLY = "ply"
    OFF = "off"

    # Web 3D
    GLTF = "gltf"
    GLB = "glb"

    # FreeCAD
    FCSTD = "fcstd"
    FCBAK = "fcbak"

    # Simulation/FEA
    ABAQUS = "inp"
    GMSH = "msh"
    SALOME = "med"
    VTK = "vtk"
    VTU = "vtu"

    # Point Cloud
    XYZ = "xyz"
    PCD = "pcd"

    @classmethod
    def from_extension(cls, ext: str) -> Optional['FileFormat']:
        """Get format from file extension"""
        ext = ext.lower().lstrip('.')
        try:
            return cls(ext)
        except ValueError:
            # Handle aliases
            aliases = {
                'step': cls.STEP,
                'stp': cls.STEP,
                'iges': cls.IGES,
                'igs': cls.IGES,
            }
            return aliases.get(ext)


class ConversionError(Exception):
    """Exception raised when conversion fails"""
    pass


class FileConverter:
    """
    Universal file converter for CAD and mesh formats.

    Supports reading and writing various formats with automatic
    format detection and conversion.
    """

    # Format capabilities
    MESH_FORMATS = {
        FileFormat.STL, FileFormat.OBJ, FileFormat.PLY, FileFormat.OFF,
        FileFormat.GLTF, FileFormat.GLB
    }

    CAD_FORMATS = {
        FileFormat.DXF, FileFormat.DWG, FileFormat.STEP, FileFormat.STP,
        FileFormat.IGES, FileFormat.IGS
    }

    FEA_FORMATS = {
        FileFormat.ABAQUS, FileFormat.GMSH, FileFormat.SALOME,
        FileFormat.VTK, FileFormat.VTU
    }

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize file converter.

        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = Path(temp_dir or './temp_conversions')
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Check capabilities
        self.capabilities = self._check_capabilities()
        logger.info(f"FileConverter initialized with {len(self.capabilities)} format handlers")

    def _check_capabilities(self) -> Dict[FileFormat, bool]:
        """Check which formats are supported"""
        caps = {}

        # DXF/DWG
        caps[FileFormat.DXF] = HAS_EZDXF
        caps[FileFormat.DWG] = HAS_EZDXF

        # Mesh formats (trimesh)
        if HAS_TRIMESH:
            for fmt in [FileFormat.STL, FileFormat.OBJ, FileFormat.PLY, FileFormat.OFF]:
                caps[fmt] = True

        # STEP/STL (build123d)
        if HAS_BUILD123D:
            caps[FileFormat.STEP] = True
            caps[FileFormat.STP] = True
            caps[FileFormat.STL] = True

        # FEA formats (meshio)
        if HAS_MESHIO:
            for fmt in self.FEA_FORMATS:
                caps[fmt] = True

        # GLTF
        caps[FileFormat.GLTF] = HAS_TRIMESH or HAS_GLTF
        caps[FileFormat.GLB] = HAS_TRIMESH or HAS_GLTF

        return caps

    def can_convert(
        self,
        from_format: Union[FileFormat, str],
        to_format: Union[FileFormat, str]
    ) -> bool:
        """
        Check if conversion between formats is supported.

        Args:
            from_format: Source format
            to_format: Target format

        Returns:
            True if conversion is supported
        """
        if isinstance(from_format, str):
            from_format = FileFormat.from_extension(from_format)
        if isinstance(to_format, str):
            to_format = FileFormat.from_extension(to_format)

        if not from_format or not to_format:
            return False

        return (
            self.capabilities.get(from_format, False) and
            self.capabilities.get(to_format, False)
        )

    def convert_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        input_format: Optional[Union[FileFormat, str]] = None,
        output_format: Optional[Union[FileFormat, str]] = None,
        **options
    ) -> bool:
        """
        Convert file from one format to another.

        Args:
            input_path: Input file path
            output_path: Output file path
            input_format: Input format (auto-detected if None)
            output_format: Output format (auto-detected if None)
            **options: Conversion options

        Returns:
            True if successful

        Raises:
            ConversionError: If conversion fails
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Auto-detect formats
        if input_format is None:
            input_format = FileFormat.from_extension(input_path.suffix)
        elif isinstance(input_format, str):
            input_format = FileFormat.from_extension(input_format)

        if output_format is None:
            output_format = FileFormat.from_extension(output_path.suffix)
        elif isinstance(output_format, str):
            output_format = FileFormat.from_extension(output_format)

        if not input_format:
            raise ConversionError(f"Unknown input format: {input_path.suffix}")
        if not output_format:
            raise ConversionError(f"Unknown output format: {output_path.suffix}")

        # Check file exists
        if not input_path.exists():
            raise ConversionError(f"Input file not found: {input_path}")

        # Check conversion support
        if not self.can_convert(input_format, output_format):
            raise ConversionError(
                f"Conversion from {input_format.value} to {output_format.value} not supported"
            )

        logger.info(f"Converting {input_path.name} from {input_format.value} to {output_format.value}")

        try:
            # Route to appropriate converter
            if input_format in self.MESH_FORMATS or output_format in self.MESH_FORMATS:
                return self._convert_mesh(input_path, output_path, input_format, output_format, **options)

            elif input_format == FileFormat.DXF or output_format == FileFormat.DXF:
                return self._convert_dxf(input_path, output_path, input_format, output_format, **options)

            elif input_format in self.CAD_FORMATS or output_format in self.CAD_FORMATS:
                return self._convert_cad(input_path, output_path, input_format, output_format, **options)

            elif input_format in self.FEA_FORMATS or output_format in self.FEA_FORMATS:
                return self._convert_fea(input_path, output_path, input_format, output_format, **options)

            else:
                raise ConversionError(f"No converter available for {input_format.value} -> {output_format.value}")

        except Exception as e:
            logger.error(f"Conversion failed: {e}", exc_info=True)
            raise ConversionError(f"Conversion failed: {str(e)}")

    def _convert_mesh(
        self,
        input_path: Path,
        output_path: Path,
        input_format: FileFormat,
        output_format: FileFormat,
        **options
    ) -> bool:
        """Convert between mesh formats using trimesh"""
        if not HAS_TRIMESH:
            raise ConversionError("trimesh not available for mesh conversion")

        try:
            # Load mesh
            logger.info(f"Loading mesh from {input_path}")
            mesh = trimesh.load(str(input_path))

            # Apply transformations if specified
            if 'scale' in options:
                mesh.apply_scale(options['scale'])

            if 'rotation' in options:
                mesh.apply_transform(trimesh.transformations.rotation_matrix(
                    options['rotation']['angle'],
                    options['rotation']['axis']
                ))

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export
            logger.info(f"Exporting to {output_path}")

            if output_format in [FileFormat.GLTF, FileFormat.GLB]:
                # Special handling for GLTF/GLB
                mesh.export(str(output_path), file_type=output_format.value)
            else:
                mesh.export(str(output_path))

            logger.info(f"Mesh conversion successful: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Mesh conversion failed: {e}", exc_info=True)
            raise ConversionError(f"Mesh conversion failed: {str(e)}")

    def _convert_dxf(
        self,
        input_path: Path,
        output_path: Path,
        input_format: FileFormat,
        output_format: FileFormat,
        **options
    ) -> bool:
        """Convert DXF files"""
        if not HAS_EZDXF:
            raise ConversionError("ezdxf not available for DXF conversion")

        try:
            # Load DXF
            logger.info(f"Loading DXF from {input_path}")
            doc = ezdxf.readfile(str(input_path))

            # Apply modifications if needed
            if 'layer_filter' in options:
                # Filter layers
                layers_to_keep = set(options['layer_filter'])
                msp = doc.modelspace()
                for entity in list(msp):
                    if entity.dxf.layer not in layers_to_keep:
                        msp.delete_entity(entity)

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save
            logger.info(f"Saving to {output_path}")
            doc.saveas(str(output_path))

            logger.info(f"DXF conversion successful: {output_path}")
            return True

        except Exception as e:
            logger.error(f"DXF conversion failed: {e}", exc_info=True)
            raise ConversionError(f"DXF conversion failed: {str(e)}")

    def _convert_cad(
        self,
        input_path: Path,
        output_path: Path,
        input_format: FileFormat,
        output_format: FileFormat,
        **options
    ) -> bool:
        """Convert between CAD formats"""
        # This would require FreeCAD or similar
        # For now, provide basic support via build123d

        if not HAS_BUILD123D:
            raise ConversionError("CAD conversion requires build123d or FreeCAD")

        # Limited CAD conversion - would need full implementation
        raise ConversionError("CAD format conversion not fully implemented. Use FreeCAD for STEP/IGES conversion.")

    def _convert_fea(
        self,
        input_path: Path,
        output_path: Path,
        input_format: FileFormat,
        output_format: FileFormat,
        **options
    ) -> bool:
        """Convert between FEA/mesh formats using meshio"""
        if not HAS_MESHIO:
            raise ConversionError("meshio not available for FEA format conversion")

        try:
            # Read mesh
            logger.info(f"Reading FEA mesh from {input_path}")
            mesh = meshio.read(str(input_path))

            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write mesh
            logger.info(f"Writing FEA mesh to {output_path}")
            meshio.write(str(output_path), mesh)

            logger.info(f"FEA conversion successful: {output_path}")
            return True

        except Exception as e:
            logger.error(f"FEA conversion failed: {e}", exc_info=True)
            raise ConversionError(f"FEA conversion failed: {str(e)}")

    def batch_convert(
        self,
        input_files: List[Union[str, Path]],
        output_dir: Union[str, Path],
        output_format: Union[FileFormat, str],
        **options
    ) -> Dict[str, bool]:
        """
        Batch convert multiple files.

        Args:
            input_files: List of input file paths
            output_dir: Output directory
            output_format: Target format for all files
            **options: Conversion options

        Returns:
            Dictionary mapping input filename to success status
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(output_format, str):
            output_format = FileFormat.from_extension(output_format)

        results = {}

        for input_file in input_files:
            input_path = Path(input_file)
            output_path = output_dir / f"{input_path.stem}.{output_format.value}"

            try:
                logger.info(f"Converting {input_path.name}...")
                success = self.convert_file(
                    input_path,
                    output_path,
                    output_format=output_format,
                    **options
                )
                results[input_path.name] = success
                logger.info(f"✓ {input_path.name} -> {output_path.name}")

            except Exception as e:
                logger.error(f"✗ {input_path.name}: {e}")
                results[input_path.name] = False

        # Summary
        successful = sum(1 for v in results.values() if v)
        logger.info(f"Batch conversion complete: {successful}/{len(input_files)} successful")

        return results

    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get information about a CAD/mesh file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return {'error': 'File not found'}

        file_format = FileFormat.from_extension(file_path.suffix)

        info = {
            'path': str(file_path),
            'name': file_path.name,
            'format': file_format.value if file_format else 'unknown',
            'size_bytes': file_path.stat().st_size,
            'size_mb': file_path.stat().st_size / (1024 * 1024)
        }

        try:
            # Try to get mesh info
            if HAS_TRIMESH and file_format in self.MESH_FORMATS:
                mesh = trimesh.load(str(file_path))

                if isinstance(mesh, trimesh.Scene):
                    # Multi-part mesh
                    info['type'] = 'scene'
                    info['num_geometries'] = len(mesh.geometry)
                    info['total_vertices'] = sum(g.vertices.shape[0] for g in mesh.geometry.values())
                    info['total_faces'] = sum(g.faces.shape[0] for g in mesh.geometry.values())
                else:
                    # Single mesh
                    info['type'] = 'mesh'
                    info['vertices'] = mesh.vertices.shape[0]
                    info['faces'] = mesh.faces.shape[0]
                    info['is_watertight'] = mesh.is_watertight
                    info['is_valid'] = mesh.is_valid
                    info['bounds'] = mesh.bounds.tolist()
                    info['volume'] = float(mesh.volume) if mesh.is_watertight else None

            # Try to get DXF info
            elif HAS_EZDXF and file_format == FileFormat.DXF:
                doc = ezdxf.readfile(str(file_path))
                msp = doc.modelspace()

                info['type'] = 'dxf'
                info['dxf_version'] = doc.dxfversion
                info['num_entities'] = len(msp)
                info['layers'] = [layer.dxf.name for layer in doc.layers]

        except Exception as e:
            info['read_error'] = str(e)
            logger.warning(f"Could not read file details: {e}")

        return info

    def list_supported_formats(self) -> Dict[str, List[str]]:
        """
        List all supported formats by category.

        Returns:
            Dictionary mapping category to list of formats
        """
        supported = {
            'mesh': [],
            'cad': [],
            'fea': [],
            'all': []
        }

        for fmt, available in self.capabilities.items():
            if not available:
                continue

            fmt_str = fmt.value
            supported['all'].append(fmt_str)

            if fmt in self.MESH_FORMATS:
                supported['mesh'].append(fmt_str)
            elif fmt in self.CAD_FORMATS:
                supported['cad'].append(fmt_str)
            elif fmt in self.FEA_FORMATS:
                supported['fea'].append(fmt_str)

        return supported


# Convenience functions
def convert_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    **options
) -> bool:
    """
    Quick file conversion function.

    Args:
        input_path: Input file path
        output_path: Output file path
        **options: Conversion options

    Returns:
        True if successful
    """
    converter = FileConverter()
    return converter.convert_file(input_path, output_path, **options)


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Quick file info function.

    Args:
        file_path: Path to file

    Returns:
        File information dictionary
    """
    converter = FileConverter()
    return converter.get_file_info(file_path)
