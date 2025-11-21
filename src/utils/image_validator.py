"""
Image validation and preprocessing utilities.

This module provides robust image validation and preprocessing
to ensure images are compatible with CAD generation pipelines.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {'PNG', 'JPEG', 'JPG', 'BMP', 'TIFF', 'TIF', 'WEBP'}

# Image constraints
MAX_FILE_SIZE_MB = 20  # Maximum file size in MB
MAX_DIMENSION = 4096  # Maximum width or height
MIN_DIMENSION = 32  # Minimum width or height
MAX_PIXELS = 16777216  # Maximum total pixels (4096x4096)


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


class ImageValidator:
    """Validates and preprocesses images for CAD generation."""

    @staticmethod
    def validate_image(
        image_path: Path,
        check_size: bool = True,
        check_format: bool = True,
        check_corruption: bool = True
    ) -> Dict[str, Any]:
        """
        Validate an image file.

        Args:
            image_path: Path to image file
            check_size: Check file size constraints
            check_format: Check format is supported
            check_corruption: Check image is not corrupted

        Returns:
            Dictionary with validation results and image info

        Raises:
            ImageValidationError: If validation fails

        Example:
            >>> validator = ImageValidator()
            >>> info = validator.validate_image(Path("sketch.png"))
            >>> print(info['width'], info['height'])
        """
        logger.info(f"Validating image: {image_path}")

        # Check file exists
        if not image_path.exists():
            raise ImageValidationError(f"Image file not found: {image_path}")

        # Check file size
        if check_size:
            file_size_mb = image_path.stat().st_size / (1024 * 1024)
            logger.info(f"Image file size: {file_size_mb:.2f} MB")

            if file_size_mb > MAX_FILE_SIZE_MB:
                raise ImageValidationError(
                    f"Image file too large: {file_size_mb:.2f} MB "
                    f"(maximum: {MAX_FILE_SIZE_MB} MB)"
                )

        try:
            # Open image with PIL
            with Image.open(image_path) as img:
                # Get basic info
                width, height = img.size
                format_name = img.format
                mode = img.mode

                logger.info(f"Image info - Size: {width}x{height}, Format: {format_name}, Mode: {mode}")

                # Check format
                if check_format and format_name not in SUPPORTED_FORMATS:
                    raise ImageValidationError(
                        f"Unsupported image format: {format_name}. "
                        f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
                    )

                # Check dimensions
                if width < MIN_DIMENSION or height < MIN_DIMENSION:
                    raise ImageValidationError(
                        f"Image too small: {width}x{height} "
                        f"(minimum: {MIN_DIMENSION}x{MIN_DIMENSION})"
                    )

                if width > MAX_DIMENSION or height > MAX_DIMENSION:
                    logger.warning(
                        f"Image very large: {width}x{height}. "
                        f"Will be resized to fit {MAX_DIMENSION}x{MAX_DIMENSION}"
                    )

                total_pixels = width * height
                if total_pixels > MAX_PIXELS:
                    logger.warning(
                        f"Image has {total_pixels} pixels, exceeds {MAX_PIXELS}. "
                        "Will be resized."
                    )

                # Check for corruption by attempting to load pixel data
                if check_corruption:
                    try:
                        img.load()
                        logger.info("Image integrity check passed")
                    except Exception as e:
                        raise ImageValidationError(
                            f"Image appears corrupted: {str(e)}"
                        )

                return {
                    'valid': True,
                    'width': width,
                    'height': height,
                    'format': format_name,
                    'mode': mode,
                    'file_size_mb': image_path.stat().st_size / (1024 * 1024),
                    'total_pixels': total_pixels
                }

        except ImageValidationError:
            raise
        except Exception as e:
            raise ImageValidationError(
                f"Failed to validate image: {str(e)}"
            )

    @staticmethod
    def preprocess_image(
        image_path: Path,
        output_path: Optional[Path] = None,
        max_size: Tuple[int, int] = (2048, 2048),
        target_format: str = 'PNG',
        ensure_rgb: bool = True
    ) -> Path:
        """
        Preprocess image for CAD generation.

        Operations:
        - Convert to RGB if needed
        - Resize if too large
        - Convert to standard format
        - Optimize quality

        Args:
            image_path: Input image path
            output_path: Output path (uses temp if None)
            max_size: Maximum dimensions (width, height)
            target_format: Output format (PNG, JPEG, etc.)
            ensure_rgb: Convert to RGB mode

        Returns:
            Path to preprocessed image

        Example:
            >>> preprocessed = ImageValidator.preprocess_image(
            ...     Path("input.jpg"),
            ...     max_size=(1024, 1024)
            ... )
        """
        logger.info(f"Preprocessing image: {image_path}")

        try:
            with Image.open(image_path) as img:
                # Store original info
                original_size = img.size
                original_format = img.format

                # Convert to RGB if needed
                if ensure_rgb and img.mode not in ('RGB', 'RGBA'):
                    logger.info(f"Converting from {img.mode} to RGB")
                    img = img.convert('RGB')
                elif img.mode == 'RGBA' and ensure_rgb:
                    # Handle transparency by adding white background
                    logger.info("Converting RGBA to RGB with white background")
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                    img = background

                # Resize if needed
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    logger.info(f"Resizing from {img.size} to fit {max_size}")
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized to {img.size}")

                # Determine output path
                if output_path is None:
                    output_path = image_path.parent / f"{image_path.stem}_preprocessed.{target_format.lower()}"

                # Save with optimization
                save_kwargs = {
                    'format': target_format,
                    'optimize': True
                }

                if target_format == 'PNG':
                    save_kwargs['compress_level'] = 6
                elif target_format in ('JPEG', 'JPG'):
                    save_kwargs['quality'] = 95

                img.save(output_path, **save_kwargs)

                output_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Preprocessed image saved: {output_path} "
                    f"({original_size} -> {img.size}, "
                    f"{original_format} -> {target_format}, "
                    f"{output_size_mb:.2f} MB)"
                )

                return output_path

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}", exc_info=True)
            raise ImageValidationError(f"Failed to preprocess image: {str(e)}")

    @staticmethod
    def convert_to_opencv_compatible(image_path: Path) -> Path:
        """
        Convert image to OpenCV-compatible format.

        OpenCV works best with:
        - Standard formats (PNG, JPEG, BMP)
        - RGB or BGR color space
        - No transparency issues

        Args:
            image_path: Input image path

        Returns:
            Path to OpenCV-compatible image
        """
        logger.info(f"Converting to OpenCV-compatible format: {image_path}")

        try:
            with Image.open(image_path) as img:
                # Convert RGBA to RGB with white background
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Save as PNG (best for OpenCV)
                output_path = image_path.parent / f"{image_path.stem}_opencv.png"
                img.save(output_path, 'PNG', optimize=True)

                logger.info(f"OpenCV-compatible image saved: {output_path}")
                return output_path

        except Exception as e:
            logger.error(f"OpenCV conversion failed: {e}", exc_info=True)
            raise ImageValidationError(f"Failed to convert for OpenCV: {str(e)}")

    @staticmethod
    def get_image_info(image_path: Path) -> Dict[str, Any]:
        """
        Get detailed image information.

        Args:
            image_path: Path to image

        Returns:
            Dictionary with image metadata
        """
        try:
            with Image.open(image_path) as img:
                return {
                    'path': str(image_path),
                    'filename': image_path.name,
                    'size': img.size,
                    'width': img.size[0],
                    'height': img.size[1],
                    'format': img.format,
                    'mode': img.mode,
                    'file_size_bytes': image_path.stat().st_size,
                    'file_size_mb': image_path.stat().st_size / (1024 * 1024),
                    'total_pixels': img.size[0] * img.size[1],
                    'has_transparency': img.mode in ('RGBA', 'LA', 'P'),
                    'dpi': img.info.get('dpi', (72, 72))
                }
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return {'error': str(e)}
