"""
Photorealistic Rendering Module

Provides photorealistic rendering capabilities with lighting, materials,
and camera controls for CAD models.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from ..cad.adam_connector import AdamNewConnector

logger = logging.getLogger(__name__)


@dataclass
class Material:
    """Material properties for rendering."""
    name: str
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8)  # RGB 0-1
    metallic: float = 0.0  # 0=dielectric, 1=metal
    roughness: float = 0.5  # 0=smooth, 1=rough
    opacity: float = 1.0  # 0=transparent, 1=opaque
    reflectivity: float = 0.5
    emission: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # Emissive color

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'color': self.color,
            'metallic': self.metallic,
            'roughness': self.roughness,
            'opacity': self.opacity,
            'reflectivity': self.reflectivity,
            'emission': self.emission
        }


@dataclass
class Light:
    """Light source configuration."""
    type: str  # 'directional', 'point', 'spot', 'ambient', 'hdri'
    position: Optional[Tuple[float, float, float]] = None
    direction: Optional[Tuple[float, float, float]] = None
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    cast_shadows: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'type': self.type,
            'color': self.color,
            'intensity': self.intensity,
            'cast_shadows': self.cast_shadows
        }
        if self.position:
            data['position'] = self.position
        if self.direction:
            data['direction'] = self.direction
        return data


@dataclass
class Camera:
    """Camera configuration."""
    position: Tuple[float, float, float] = (10.0, 10.0, 10.0)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 45.0  # Field of view in degrees
    aspect_ratio: float = 16.0 / 9.0
    near_clip: float = 0.1
    far_clip: float = 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'position': self.position,
            'target': self.target,
            'up': self.up,
            'fov': self.fov,
            'aspect_ratio': self.aspect_ratio,
            'near_clip': self.near_clip,
            'far_clip': self.far_clip
        }


class MaterialLibrary:
    """Library of predefined materials."""

    MATERIALS = {
        'aluminum': Material(
            name='Aluminum',
            color=(0.913, 0.921, 0.925),
            metallic=1.0,
            roughness=0.3
        ),
        'steel': Material(
            name='Steel',
            color=(0.77, 0.77, 0.77),
            metallic=1.0,
            roughness=0.4
        ),
        'brass': Material(
            name='Brass',
            color=(0.86, 0.75, 0.42),
            metallic=1.0,
            roughness=0.35
        ),
        'copper': Material(
            name='Copper',
            color=(0.95, 0.64, 0.54),
            metallic=1.0,
            roughness=0.3
        ),
        'gold': Material(
            name='Gold',
            color=(1.0, 0.84, 0.0),
            metallic=1.0,
            roughness=0.25
        ),
        'plastic_white': Material(
            name='White Plastic',
            color=(0.95, 0.95, 0.95),
            metallic=0.0,
            roughness=0.5
        ),
        'plastic_black': Material(
            name='Black Plastic',
            color=(0.1, 0.1, 0.1),
            metallic=0.0,
            roughness=0.6
        ),
        'glass': Material(
            name='Glass',
            color=(0.95, 0.95, 0.95),
            metallic=0.0,
            roughness=0.0,
            opacity=0.3,
            reflectivity=0.9
        ),
        'rubber': Material(
            name='Rubber',
            color=(0.2, 0.2, 0.2),
            metallic=0.0,
            roughness=0.9
        ),
        'wood': Material(
            name='Wood',
            color=(0.55, 0.35, 0.2),
            metallic=0.0,
            roughness=0.7
        )
    }

    @classmethod
    def get(cls, name: str) -> Optional[Material]:
        """Get material by name."""
        return cls.MATERIALS.get(name.lower())

    @classmethod
    def list_materials(cls) -> List[str]:
        """Get list of available materials."""
        return list(cls.MATERIALS.keys())


class LightingPreset:
    """Predefined lighting setups."""

    @staticmethod
    def studio_lighting() -> List[Light]:
        """Three-point studio lighting setup."""
        return [
            # Key light
            Light(
                type='directional',
                direction=(-0.5, -0.5, -0.7),
                intensity=1.5,
                cast_shadows=True
            ),
            # Fill light
            Light(
                type='directional',
                direction=(0.5, -0.3, -0.5),
                intensity=0.5,
                cast_shadows=False
            ),
            # Back light
            Light(
                type='directional',
                direction=(0.0, 0.8, -0.3),
                intensity=0.8,
                cast_shadows=False
            ),
            # Ambient
            Light(
                type='ambient',
                intensity=0.2
            )
        ]

    @staticmethod
    def outdoor_lighting() -> List[Light]:
        """Outdoor daylight lighting."""
        return [
            # Sun
            Light(
                type='directional',
                direction=(0.3, 0.5, -1.0),
                color=(1.0, 0.98, 0.95),
                intensity=2.0,
                cast_shadows=True
            ),
            # Sky ambient
            Light(
                type='ambient',
                color=(0.7, 0.85, 1.0),
                intensity=0.4
            )
        ]

    @staticmethod
    def product_showcase() -> List[Light]:
        """Product photography lighting."""
        return [
            # Main light
            Light(
                type='point',
                position=(5.0, 5.0, 8.0),
                intensity=2.0,
                cast_shadows=True
            ),
            # Rim light
            Light(
                type='point',
                position=(-5.0, 5.0, 3.0),
                intensity=1.2,
                cast_shadows=False
            ),
            # Fill
            Light(
                type='point',
                position=(0.0, -5.0, 5.0),
                intensity=0.6,
                cast_shadows=False
            ),
            # Ambient
            Light(
                type='ambient',
                intensity=0.3
            )
        ]


class CameraPreset:
    """Predefined camera positions."""

    @staticmethod
    def isometric(distance: float = 15.0) -> Camera:
        """Isometric view."""
        pos = distance / np.sqrt(3)
        return Camera(
            position=(pos, pos, pos),
            target=(0.0, 0.0, 0.0),
            fov=35.0
        )

    @staticmethod
    def front(distance: float = 15.0) -> Camera:
        """Front view."""
        return Camera(
            position=(0.0, -distance, 0.0),
            target=(0.0, 0.0, 0.0),
            fov=40.0
        )

    @staticmethod
    def top(distance: float = 15.0) -> Camera:
        """Top view."""
        return Camera(
            position=(0.0, 0.0, distance),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov=40.0
        )

    @staticmethod
    def side(distance: float = 15.0) -> Camera:
        """Side view."""
        return Camera(
            position=(distance, 0.0, 0.0),
            target=(0.0, 0.0, 0.0),
            fov=40.0
        )

    @staticmethod
    def perspective(distance: float = 12.0) -> Camera:
        """Artistic perspective view."""
        return Camera(
            position=(distance * 0.7, -distance * 0.7, distance * 0.5),
            target=(0.0, 0.0, 0.0),
            fov=50.0
        )


class PhotorealisticRenderer:
    """
    Photorealistic rendering engine with material, lighting, and camera controls.
    """

    def __init__(self, adam_api_key: Optional[str] = None, use_mock: bool = False):
        """
        Initialize renderer.

        Args:
            adam_api_key: Adam.new API key for rendering
            use_mock: Use mock mode for testing
        """
        self.adam_connector = AdamNewConnector(api_key=adam_api_key, mock_mode=use_mock)
        self.materials = {}
        self.lights = []
        self.camera = CameraPreset.isometric()

    def set_material(self, part_name: str, material: Material) -> None:
        """
        Set material for a part.

        Args:
            part_name: Name of the part
            material: Material to apply
        """
        self.materials[part_name] = material
        logger.info(f"Material '{material.name}' assigned to part '{part_name}'")

    def add_light(self, light: Light) -> None:
        """Add a light to the scene."""
        self.lights.append(light)
        logger.info(f"Added {light.type} light")

    def set_lighting_preset(self, preset_name: str) -> None:
        """
        Set lighting from preset.

        Args:
            preset_name: 'studio', 'outdoor', or 'product'
        """
        self.lights.clear()

        if preset_name == 'studio':
            self.lights = LightingPreset.studio_lighting()
        elif preset_name == 'outdoor':
            self.lights = LightingPreset.outdoor_lighting()
        elif preset_name == 'product':
            self.lights = LightingPreset.product_showcase()
        else:
            logger.warning(f"Unknown preset: {preset_name}")
            return

        logger.info(f"Applied '{preset_name}' lighting preset ({len(self.lights)} lights)")

    def set_camera(self, camera: Camera) -> None:
        """Set camera configuration."""
        self.camera = camera
        logger.info(f"Camera set to position {camera.position}")

    def set_camera_preset(self, preset_name: str, distance: float = 15.0) -> None:
        """
        Set camera from preset.

        Args:
            preset_name: 'isometric', 'front', 'top', 'side', or 'perspective'
            distance: Camera distance from origin
        """
        if preset_name == 'isometric':
            self.camera = CameraPreset.isometric(distance)
        elif preset_name == 'front':
            self.camera = CameraPreset.front(distance)
        elif preset_name == 'top':
            self.camera = CameraPreset.top(distance)
        elif preset_name == 'side':
            self.camera = CameraPreset.side(distance)
        elif preset_name == 'perspective':
            self.camera = CameraPreset.perspective(distance)
        else:
            logger.warning(f"Unknown camera preset: {preset_name}")
            return

        logger.info(f"Applied '{preset_name}' camera preset")

    def render(
        self,
        model_id: str,
        output_path: Path,
        resolution: Tuple[int, int] = (1920, 1080),
        samples: int = 128,
        denoise: bool = True
    ) -> Optional[Path]:
        """
        Render model with current settings.

        Args:
            model_id: Model ID to render
            output_path: Output image path
            resolution: Image resolution (width, height)
            samples: Number of samples for ray tracing
            denoise: Apply denoising

        Returns:
            Path to rendered image or None
        """
        logger.info(f"Rendering model {model_id}...")

        # Build rendering configuration
        config = {
            'model_id': model_id,
            'resolution': resolution,
            'samples': samples,
            'denoise': denoise,
            'materials': {
                name: mat.to_dict()
                for name, mat in self.materials.items()
            },
            'lights': [light.to_dict() for light in self.lights],
            'camera': self.camera.to_dict()
        }

        # For mock mode or if Adam doesn't support rendering yet,
        # we can create a placeholder or use local rendering
        if self.adam_connector.mock_mode:
            return self._mock_render(config, output_path)

        # In real implementation, this would call Adam.new's rendering API
        logger.warning("Adam.new rendering API integration pending")
        logger.info(f"Render config: {json.dumps(config, indent=2, default=str)}")

        return None

    def _mock_render(self, config: Dict[str, Any], output_path: Path) -> Path:
        """Create mock render for testing."""
        logger.info(f"[MOCK] Rendering with config: {json.dumps(config, indent=2, default=str)}")

        # Create placeholder image
        if HAS_NUMPY:
            try:
                from PIL import Image

                width, height = config['resolution']

                # Create gradient image as placeholder
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                for y in range(height):
                    for x in range(width):
                        img_array[y, x] = [
                            int(255 * x / width),
                            int(255 * y / height),
                            128
                        ]

                img = Image.fromarray(img_array)
                img.save(output_path)

                logger.info(f"[MOCK] Saved placeholder render: {output_path}")
                return output_path

            except Exception as e:
                logger.error(f"Failed to create mock render: {e}")

        return None

    def get_scene_info(self) -> Dict[str, Any]:
        """Get current scene configuration."""
        return {
            'materials': {
                name: mat.to_dict()
                for name, mat in self.materials.items()
            },
            'lights': [light.to_dict() for light in self.lights],
            'camera': self.camera.to_dict()
        }
