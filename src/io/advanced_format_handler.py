"""
Advanced File Format Handler

Support for additional CAD formats: DWG, FreeCAD (.FCStd, .FCbak), SketchUp (.skp)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import json
import tempfile

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class DWGHandler:
    """
    Handler for AutoCAD DWG files.

    DWG is a proprietary format. This handler provides multiple strategies:
    1. Convert to DXF using external tools (LibreCAD, ODA File Converter)
    2. Use cloud-based conversion services
    3. Parse with available Python libraries
    """

    @staticmethod
    def check_converter_available() -> Dict[str, bool]:
        """Check which DWG converters are available."""
        converters = {
            'oda_converter': False,
            'libre_cad': False,
            'dwg2dxf': False
        }

        # Check ODA File Converter
        try:
            result = subprocess.run(
                ['ODAFileConverter', '--version'],
                capture_output=True,
                timeout=5
            )
            converters['oda_converter'] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check LibreCAD
        try:
            result = subprocess.run(
                ['librecad', '--version'],
                capture_output=True,
                timeout=5
            )
            converters['libre_cad'] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return converters

    @staticmethod
    def convert_to_dxf(dwg_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Convert DWG to DXF format.

        Args:
            dwg_path: Input DWG file path
            output_path: Output DXF path (auto-generated if None)

        Returns:
            Path to converted DXF file or None if conversion failed
        """
        if output_path is None:
            output_path = dwg_path.with_suffix('.dxf')

        # Try ODA File Converter
        try:
            logger.info(f"Attempting DWG to DXF conversion: {dwg_path}")

            result = subprocess.run(
                [
                    'ODAFileConverter',
                    str(dwg_path.parent),
                    str(output_path.parent),
                    'ACAD2018',
                    'DXF',
                    '0',
                    '1',
                    str(dwg_path.name)
                ],
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0 and output_path.exists():
                logger.info(f"DWG converted successfully: {output_path}")
                return output_path

        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"ODA converter not available: {e}")

        logger.error(f"Failed to convert DWG file: {dwg_path}")
        return None

    @staticmethod
    def read_dwg(dwg_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read DWG file.

        Args:
            dwg_path: Path to DWG file

        Returns:
            Dictionary with DWG data or None
        """
        # Try conversion to DXF first
        dxf_path = DWGHandler.convert_to_dxf(dwg_path)

        if dxf_path and HAS_EZDXF:
            try:
                from ..io.dxf_parser import DXFParser
                parser = DXFParser()
                return parser.parse(str(dxf_path))
            except Exception as e:
                logger.error(f"Failed to parse converted DXF: {e}")

        return None


class FreeCADHandler:
    """
    Handler for FreeCAD files (.FCStd, .FCbak).

    FreeCAD files are ZIP archives containing XML and BREP data.
    """

    @staticmethod
    def is_freecad_file(file_path: Path) -> bool:
        """Check if file is a FreeCAD file."""
        return file_path.suffix.lower() in ['.fcstd', '.fcbak']

    @staticmethod
    def extract_freecad_archive(fcstd_path: Path, extract_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Extract FreeCAD archive.

        Args:
            fcstd_path: Path to .FCStd or .FCbak file
            extract_dir: Directory to extract to

        Returns:
            Path to extracted directory
        """
        import zipfile

        if extract_dir is None:
            extract_dir = Path(tempfile.mkdtemp(prefix='freecad_'))

        try:
            with zipfile.ZipFile(fcstd_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            logger.info(f"FreeCAD file extracted: {extract_dir}")
            return extract_dir

        except Exception as e:
            logger.error(f"Failed to extract FreeCAD file: {e}")
            return None

    @staticmethod
    def read_document_xml(fcstd_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read FreeCAD document XML.

        Args:
            fcstd_path: Path to .FCStd file

        Returns:
            Dictionary with document metadata
        """
        import xml.etree.ElementTree as ET

        extract_dir = FreeCADHandler.extract_freecad_archive(fcstd_path)
        if not extract_dir:
            return None

        doc_xml_path = extract_dir / 'Document.xml'
        if not doc_xml_path.exists():
            return None

        try:
            tree = ET.parse(doc_xml_path)
            root = tree.getroot()

            metadata = {
                'objects': [],
                'properties': {}
            }

            # Extract objects
            for obj in root.findall('.//Object'):
                obj_data = {
                    'name': obj.get('name'),
                    'type': obj.get('type'),
                    'properties': {}
                }

                for prop in obj.findall('.//Property'):
                    prop_name = prop.get('name')
                    prop_value = prop.text
                    obj_data['properties'][prop_name] = prop_value

                metadata['objects'].append(obj_data)

            logger.info(f"FreeCAD document parsed: {len(metadata['objects'])} objects")
            return metadata

        except Exception as e:
            logger.error(f"Failed to parse FreeCAD XML: {e}")
            return None

    @staticmethod
    def convert_to_step(fcstd_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Convert FreeCAD file to STEP format using FreeCAD CLI.

        Args:
            fcstd_path: Input FreeCAD file
            output_path: Output STEP file path

        Returns:
            Path to STEP file or None
        """
        if output_path is None:
            output_path = fcstd_path.with_suffix('.step')

        # Create conversion script
        script_content = f"""
import FreeCAD
import Part
import Import

doc = FreeCAD.open("{fcstd_path}")
objects = [obj for obj in doc.Objects if hasattr(obj, 'Shape')]

if objects:
    shapes = [obj.Shape for obj in objects]
    compound = Part.makeCompound(shapes)
    Import.export([compound], "{output_path}")
    print("Export successful")
else:
    print("No exportable objects found")
"""

        script_path = Path(tempfile.mktemp(suffix='.py'))
        script_path.write_text(script_content)

        try:
            result = subprocess.run(
                ['freecadcmd', str(script_path)],
                capture_output=True,
                timeout=60
            )

            if result.returncode == 0 and output_path.exists():
                logger.info(f"FreeCAD file converted to STEP: {output_path}")
                script_path.unlink(missing_ok=True)
                return output_path

            logger.warning(f"FreeCAD conversion output: {result.stdout.decode()}")

        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"FreeCAD conversion failed: {e}")

        finally:
            script_path.unlink(missing_ok=True)

        return None


class SketchUpHandler:
    """
    Handler for SketchUp files (.skp).

    SketchUp is a proprietary format. This handler provides:
    1. Conversion using SketchUp API
    2. Conversion using external tools
    3. Cloud-based conversion services
    """

    @staticmethod
    def check_sketchup_available() -> bool:
        """Check if SketchUp or conversion tools are available."""
        try:
            result = subprocess.run(
                ['sketchup', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def convert_to_dae(skp_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Convert SketchUp file to COLLADA (.dae) format.

        Args:
            skp_path: Input .skp file
            output_path: Output .dae file path

        Returns:
            Path to DAE file or None
        """
        if output_path is None:
            output_path = skp_path.with_suffix('.dae')

        # Try using SketchUp exporters or external tools
        # This is a placeholder for actual implementation

        logger.warning("SketchUp conversion not yet implemented")
        logger.info("Recommended: Use SketchUp's built-in export or Trimble Connect API")

        return None

    @staticmethod
    def get_metadata(skp_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from SketchUp file.

        Args:
            skp_path: Path to .skp file

        Returns:
            Dictionary with metadata
        """
        # SketchUp files have a specific binary format
        # This would require a proper parser or external tool

        return {
            'filename': skp_path.name,
            'size_bytes': skp_path.stat().st_size,
            'format': 'SketchUp',
            'note': 'Full parsing requires SketchUp SDK or conversion tools'
        }


class AdvancedFormatConverter:
    """
    Unified interface for advanced format conversion.
    """

    @staticmethod
    def convert_to_common_format(
        input_path: Path,
        target_format: str = 'step'
    ) -> Optional[Path]:
        """
        Convert any supported format to a common format (STEP, STL, OBJ).

        Args:
            input_path: Input file path
            target_format: Target format ('step', 'stl', 'obj', 'dxf')

        Returns:
            Path to converted file or None
        """
        suffix = input_path.suffix.lower()

        logger.info(f"Converting {suffix} to {target_format}: {input_path}")

        # DWG files
        if suffix == '.dwg':
            if target_format == 'dxf':
                return DWGHandler.convert_to_dxf(input_path)
            else:
                # Convert to DXF first, then to target
                dxf_path = DWGHandler.convert_to_dxf(input_path)
                if dxf_path:
                    return AdvancedFormatConverter._convert_dxf_to_target(dxf_path, target_format)

        # FreeCAD files
        elif suffix in ['.fcstd', '.fcbak']:
            if target_format == 'step':
                return FreeCADHandler.convert_to_step(input_path)
            # Other formats would require additional conversion steps

        # SketchUp files
        elif suffix == '.skp':
            dae_path = SketchUpHandler.convert_to_dae(input_path)
            if dae_path:
                return AdvancedFormatConverter._convert_collada_to_target(dae_path, target_format)

        logger.warning(f"No conversion path available for {suffix} -> {target_format}")
        return None

    @staticmethod
    def _convert_dxf_to_target(dxf_path: Path, target_format: str) -> Optional[Path]:
        """Convert DXF to target format."""
        # This would use appropriate converters (e.g., CADQuery, FreeCAD, etc.)
        logger.info(f"DXF to {target_format} conversion not yet implemented")
        return None

    @staticmethod
    def _convert_collada_to_target(dae_path: Path, target_format: str) -> Optional[Path]:
        """Convert COLLADA to target format."""
        # This would use mesh processing libraries like Trimesh
        try:
            import trimesh

            mesh = trimesh.load(str(dae_path))
            output_path = dae_path.with_suffix(f'.{target_format}')

            mesh.export(str(output_path))
            logger.info(f"COLLADA converted to {target_format}: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"COLLADA conversion failed: {e}")
            return None

    @staticmethod
    def get_supported_formats() -> Dict[str, List[str]]:
        """
        Get list of supported formats and their capabilities.

        Returns:
            Dictionary mapping format to list of capabilities
        """
        return {
            'DWG': ['read', 'convert_to_dxf'],
            'FCStd': ['read', 'convert_to_step', 'extract_metadata'],
            'FCbak': ['read', 'convert_to_step', 'extract_metadata'],
            'SKP': ['metadata_only', 'convert_to_dae (limited)'],
            'DXF': ['read', 'write', 'convert'],
            'STEP': ['read', 'write'],
            'STL': ['read', 'write'],
            'OBJ': ['read', 'write'],
            'IGES': ['read (limited)'],
            'BREP': ['read (limited)']
        }


def get_format_info(file_path: Path) -> Dict[str, Any]:
    """
    Get information about a file format.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with format information
    """
    suffix = file_path.suffix.lower().lstrip('.')

    format_info = {
        'dwg': {
            'name': 'AutoCAD Drawing',
            'description': 'Proprietary 2D/3D CAD format by Autodesk',
            'capabilities': ['read_via_conversion', 'requires_external_tools']
        },
        'fcstd': {
            'name': 'FreeCAD Document',
            'description': 'Open-source parametric 3D CAD format',
            'capabilities': ['read', 'extract_metadata', 'convert_to_step']
        },
        'fcbak': {
            'name': 'FreeCAD Backup',
            'description': 'FreeCAD backup file (same as .FCStd)',
            'capabilities': ['read', 'extract_metadata', 'convert_to_step']
        },
        'skp': {
            'name': 'SketchUp',
            'description': '3D modeling format by Trimble',
            'capabilities': ['metadata', 'requires_external_conversion']
        }
    }

    return format_info.get(suffix, {
        'name': f'{suffix.upper()} file',
        'description': 'Format information not available',
        'capabilities': []
    })
