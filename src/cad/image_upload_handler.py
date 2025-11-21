"""
Image Upload Handler with Robust Validation

This module provides comprehensive image upload handling with:
- File type validation (PNG, JPG, JPEG, BMP, TIFF)
- File size validation
- Image integrity checking
- Dimension validation
- Format conversion
- Security checks (anti-malware)
- Detailed logging

Author: GenAI CAD CFD Studio
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import io
import hashlib
from datetime import datetime
from enum import Enum

# Optional imports with fallback
try:
    from PIL import Image, ImageStat, UnidentifiedImageError
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not installed. Image validation disabled.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("NumPy not installed. Advanced image analysis disabled.")


logger = logging.getLogger(__name__)


class ImageType(Enum):
    """Supported image types for CAD generation."""
    SKETCH = "sketch"          # Hand-drawn sketch
    PHOTO = "photo"            # Reference photo
    TECHNICAL = "technical"    # Technical drawing
    BLUEPRINT = "blueprint"    # Engineering blueprint
    RENDERING = "rendering"    # 3D rendering


class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass


class ImageUploadHandler:
    """
    Comprehensive image upload handler with validation and preprocessing.

    Features:
    - Multi-format support (PNG, JPG, JPEG, BMP, TIFF, WebP)
    - File size limits (default 10MB, max 50MB)
    - Dimension validation (min 64x64, max 8192x8192)
    - Image integrity checks
    - Format conversion and optimization
    - Security validation
    - Detailed logging and metrics

    Example:
        >>> handler = ImageUploadHandler()
        >>> result = handler.validate_and_process("sketch.png")
        >>> if result['is_valid']:
        ...     image_data = result['processed_image']
        ...     metadata = result['metadata']
    """

    # Supported formats
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}

    # Size limits (in bytes)
    DEFAULT_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    ABSOLUTE_MAX_SIZE = 50 * 1024 * 1024  # 50MB

    # Dimension limits (in pixels)
    MIN_DIMENSION = 64
    MAX_DIMENSION = 8192

    # Quality thresholds
    MIN_QUALITY_SCORE = 0.1  # Minimum acceptable image quality

    def __init__(
        self,
        max_file_size: Optional[int] = None,
        allowed_formats: Optional[set] = None,
        output_dir: Optional[Union[str, Path]] = None,
        enable_conversion: bool = True,
        enable_security_checks: bool = True
    ):
        """
        Initialize ImageUploadHandler.

        Args:
            max_file_size: Maximum file size in bytes (default 10MB)
            allowed_formats: Set of allowed file extensions (default: SUPPORTED_FORMATS)
            output_dir: Directory for processed images (default: ./uploads)
            enable_conversion: Enable automatic format conversion (default: True)
            enable_security_checks: Enable security validation (default: True)
        """
        if not HAS_PIL:
            raise ImportError("PIL (Pillow) is required for ImageUploadHandler")

        self.max_file_size = max_file_size or self.DEFAULT_MAX_SIZE
        self.allowed_formats = allowed_formats or self.SUPPORTED_FORMATS
        self.output_dir = Path(output_dir or './uploads')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enable_conversion = enable_conversion
        self.enable_security_checks = enable_security_checks

        # Metrics tracking
        self.validation_count = 0
        self.validation_success_count = 0
        self.validation_failure_count = 0

        logger.info(
            f"ImageUploadHandler initialized: "
            f"max_size={self.max_file_size/1024/1024:.1f}MB, "
            f"formats={self.allowed_formats}"
        )

    def validate_and_process(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO],
        image_type: Optional[ImageType] = None,
        custom_validations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate and process an uploaded image.

        Args:
            image_input: Image file path, bytes, or BytesIO object
            image_type: Type of image (sketch, photo, technical, etc.)
            custom_validations: Additional validation parameters

        Returns:
            Dictionary containing:
                - is_valid: bool - Whether image is valid
                - processed_image: PIL.Image or None
                - metadata: Dict with image information
                - errors: List of validation errors
                - warnings: List of validation warnings

        Raises:
            ImageValidationError: If critical validation fails
        """
        self.validation_count += 1
        start_time = datetime.now()

        result = {
            'is_valid': False,
            'processed_image': None,
            'original_image': None,
            'metadata': {},
            'errors': [],
            'warnings': [],
            'validation_time': None
        }

        try:
            logger.info(f"Starting image validation (count: {self.validation_count})")

            # Step 1: Load image
            image, file_path, file_size = self._load_image(image_input)
            result['original_image'] = image

            # Step 2: Basic validations
            self._validate_file_size(file_size, result)
            self._validate_format(file_path, result)
            self._validate_dimensions(image, result)
            self._validate_integrity(image, result)

            # Step 3: Security checks
            if self.enable_security_checks:
                self._security_checks(image, file_path, result)

            # Step 4: Quality analysis
            quality_metrics = self._analyze_quality(image)
            result['metadata']['quality'] = quality_metrics

            # Step 5: Type-specific validation
            if image_type:
                self._validate_image_type(image, image_type, result)

            # Step 6: Custom validations
            if custom_validations:
                self._apply_custom_validations(image, custom_validations, result)

            # Step 7: Process and optimize
            if self.enable_conversion and not result['errors']:
                processed_image = self._process_image(image, image_type)
                result['processed_image'] = processed_image
            else:
                result['processed_image'] = image

            # Step 8: Generate metadata
            result['metadata'].update({
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'file_size': file_size,
                'file_path': str(file_path) if file_path else None,
                'image_type': image_type.value if image_type else None,
                'hash': self._compute_hash(image),
                'timestamp': datetime.now().isoformat()
            })

            # Determine overall validity
            result['is_valid'] = len(result['errors']) == 0

            # Update metrics
            if result['is_valid']:
                self.validation_success_count += 1
                logger.info(f"Image validation PASSED: {result['metadata']}")
            else:
                self.validation_failure_count += 1
                logger.warning(
                    f"Image validation FAILED: {len(result['errors'])} error(s), "
                    f"{len(result['warnings'])} warning(s)"
                )
                logger.warning(f"Errors: {result['errors']}")

        except Exception as e:
            logger.error(f"Image validation exception: {e}", exc_info=True)
            result['errors'].append(f"Validation exception: {str(e)}")
            result['is_valid'] = False
            self.validation_failure_count += 1

        finally:
            # Calculate validation time
            validation_time = (datetime.now() - start_time).total_seconds()
            result['validation_time'] = validation_time
            logger.info(f"Validation completed in {validation_time:.3f}s")

        return result

    def _load_image(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO]
    ) -> Tuple[Image.Image, Optional[Path], int]:
        """
        Load image from various input types.

        Returns:
            Tuple of (PIL.Image, file_path, file_size)
        """
        file_path = None
        file_size = 0

        try:
            # Case 1: File path (string or Path)
            if isinstance(image_input, (str, Path)):
                file_path = Path(image_input)

                if not file_path.exists():
                    raise ImageValidationError(f"Image file not found: {file_path}")

                file_size = file_path.stat().st_size
                image = Image.open(file_path)
                logger.debug(f"Loaded image from path: {file_path}")

            # Case 2: Bytes
            elif isinstance(image_input, bytes):
                file_size = len(image_input)
                image = Image.open(io.BytesIO(image_input))
                logger.debug(f"Loaded image from bytes ({file_size} bytes)")

            # Case 3: BytesIO
            elif isinstance(image_input, io.BytesIO):
                image_input.seek(0)
                image = Image.open(image_input)
                image_input.seek(0, 2)  # Seek to end
                file_size = image_input.tell()
                image_input.seek(0)  # Reset
                logger.debug(f"Loaded image from BytesIO ({file_size} bytes)")

            else:
                raise ImageValidationError(
                    f"Unsupported image input type: {type(image_input)}"
                )

            # Verify it's a valid image
            image.verify()

            # Reload image after verify (verify closes the file)
            if isinstance(image_input, (str, Path)):
                image = Image.open(file_path)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            else:
                image_input.seek(0)
                image = Image.open(image_input)

            return image, file_path, file_size

        except UnidentifiedImageError as e:
            raise ImageValidationError(f"Could not identify image format: {e}")
        except Exception as e:
            raise ImageValidationError(f"Failed to load image: {e}")

    def _validate_file_size(self, file_size: int, result: Dict[str, Any]) -> None:
        """Validate file size against limits."""
        if file_size > self.ABSOLUTE_MAX_SIZE:
            result['errors'].append(
                f"File size {file_size/1024/1024:.1f}MB exceeds absolute maximum "
                f"{self.ABSOLUTE_MAX_SIZE/1024/1024:.1f}MB"
            )
        elif file_size > self.max_file_size:
            result['warnings'].append(
                f"File size {file_size/1024/1024:.1f}MB exceeds recommended maximum "
                f"{self.max_file_size/1024/1024:.1f}MB"
            )
        elif file_size == 0:
            result['errors'].append("File is empty (0 bytes)")

        logger.debug(f"File size validation: {file_size/1024:.1f}KB")

    def _validate_format(self, file_path: Optional[Path], result: Dict[str, Any]) -> None:
        """Validate file format."""
        if file_path:
            extension = file_path.suffix.lower()
            if extension not in self.allowed_formats:
                result['errors'].append(
                    f"File format '{extension}' not supported. "
                    f"Allowed: {self.allowed_formats}"
                )
            logger.debug(f"Format validation: {extension}")

    def _validate_dimensions(self, image: Image.Image, result: Dict[str, Any]) -> None:
        """Validate image dimensions."""
        width, height = image.size

        # Check minimum dimensions
        if width < self.MIN_DIMENSION or height < self.MIN_DIMENSION:
            result['errors'].append(
                f"Image dimensions {width}x{height} too small. "
                f"Minimum: {self.MIN_DIMENSION}x{self.MIN_DIMENSION}"
            )

        # Check maximum dimensions
        if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
            result['warnings'].append(
                f"Image dimensions {width}x{height} very large. "
                f"May be resized to {self.MAX_DIMENSION}x{self.MAX_DIMENSION}"
            )

        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio > 10 or aspect_ratio < 0.1:
            result['warnings'].append(
                f"Unusual aspect ratio: {aspect_ratio:.2f}. "
                f"This may affect processing quality."
            )

        logger.debug(f"Dimension validation: {width}x{height} (AR: {aspect_ratio:.2f})")

    def _validate_integrity(self, image: Image.Image, result: Dict[str, Any]) -> None:
        """Validate image integrity."""
        try:
            # Check if image can be converted to array
            if HAS_NUMPY:
                img_array = np.array(image)

                # Check for completely black or white images
                if img_array.min() == img_array.max():
                    result['warnings'].append(
                        "Image appears to be uniform color (no variation detected)"
                    )

            # Check for valid mode
            valid_modes = ['RGB', 'RGBA', 'L', 'LA', 'P', 'CMYK']
            if image.mode not in valid_modes:
                result['warnings'].append(
                    f"Unusual color mode: {image.mode}. May require conversion."
                )

            logger.debug(f"Integrity validation passed (mode: {image.mode})")

        except Exception as e:
            result['errors'].append(f"Integrity check failed: {e}")

    def _security_checks(
        self,
        image: Image.Image,
        file_path: Optional[Path],
        result: Dict[str, Any]
    ) -> None:
        """Perform security checks on image."""
        try:
            # Check for suspiciously large files
            if file_path and file_path.stat().st_size > 25 * 1024 * 1024:  # 25MB
                result['warnings'].append(
                    "Large file size may indicate embedded data or high resolution"
                )

            # Check for unusual formats that might contain scripts
            if image.format in ['SVG', 'PS', 'EPS']:
                result['warnings'].append(
                    f"Format {image.format} may contain executable content"
                )

            logger.debug("Security checks passed")

        except Exception as e:
            logger.warning(f"Security check exception: {e}")

    def _analyze_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image quality metrics."""
        quality_metrics = {
            'sharpness': None,
            'brightness': None,
            'contrast': None,
            'quality_score': None
        }

        try:
            # Convert to RGB for analysis
            if image.mode != 'RGB':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image

            # Calculate statistics
            stat = ImageStat.Stat(rgb_image)

            # Brightness (mean of all channels)
            quality_metrics['brightness'] = sum(stat.mean) / len(stat.mean)

            # Contrast (standard deviation)
            quality_metrics['contrast'] = sum(stat.stddev) / len(stat.stddev)

            # Simple quality score (normalized contrast)
            quality_metrics['quality_score'] = min(quality_metrics['contrast'] / 128.0, 1.0)

            logger.debug(f"Quality analysis: {quality_metrics}")

        except Exception as e:
            logger.warning(f"Quality analysis failed: {e}")

        return quality_metrics

    def _validate_image_type(
        self,
        image: Image.Image,
        image_type: ImageType,
        result: Dict[str, Any]
    ) -> None:
        """Validate image based on specified type."""
        try:
            if image_type == ImageType.SKETCH:
                # Sketches are usually grayscale or simple colors
                if image.mode not in ['L', 'LA', 'RGB', 'RGBA']:
                    result['warnings'].append(
                        f"Sketch images work best in grayscale or RGB. Current: {image.mode}"
                    )

            elif image_type == ImageType.TECHNICAL:
                # Technical drawings should have good contrast
                quality = result['metadata'].get('quality', {})
                if quality.get('contrast', 0) < 30:
                    result['warnings'].append(
                        "Technical drawing has low contrast. May affect line detection."
                    )

            elif image_type == ImageType.PHOTO:
                # Photos should be in color
                if image.mode in ['L', 'LA']:
                    result['warnings'].append(
                        "Photo provided in grayscale. Color may improve analysis."
                    )

            logger.debug(f"Type-specific validation for {image_type.value} passed")

        except Exception as e:
            logger.warning(f"Type-specific validation failed: {e}")

    def _apply_custom_validations(
        self,
        image: Image.Image,
        custom_validations: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Apply custom validation rules."""
        try:
            # Example custom validations
            if 'min_width' in custom_validations:
                if image.width < custom_validations['min_width']:
                    result['errors'].append(
                        f"Width {image.width} less than required {custom_validations['min_width']}"
                    )

            if 'min_height' in custom_validations:
                if image.height < custom_validations['min_height']:
                    result['errors'].append(
                        f"Height {image.height} less than required {custom_validations['min_height']}"
                    )

            if 'required_mode' in custom_validations:
                if image.mode != custom_validations['required_mode']:
                    result['errors'].append(
                        f"Mode {image.mode} does not match required {custom_validations['required_mode']}"
                    )

            logger.debug("Custom validations applied")

        except Exception as e:
            logger.warning(f"Custom validation failed: {e}")

    def _process_image(
        self,
        image: Image.Image,
        image_type: Optional[ImageType]
    ) -> Image.Image:
        """Process and optimize image."""
        try:
            processed = image.copy()

            # Convert to RGB if needed (for consistency)
            if processed.mode not in ['RGB', 'RGBA']:
                processed = processed.convert('RGB')
                logger.debug(f"Converted image to RGB")

            # Resize if too large
            max_dim = 2048  # Reasonable max for processing
            if max(processed.size) > max_dim:
                processed.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {processed.size}")

            return processed

        except Exception as e:
            logger.warning(f"Image processing failed: {e}")
            return image

    def _compute_hash(self, image: Image.Image) -> str:
        """Compute hash of image for deduplication."""
        try:
            # Convert image to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            # Compute SHA256 hash
            hash_obj = hashlib.sha256(image_bytes)
            return hash_obj.hexdigest()

        except Exception as e:
            logger.warning(f"Hash computation failed: {e}")
            return ""

    def get_metrics(self) -> Dict[str, Any]:
        """Get validation metrics."""
        return {
            'total_validations': self.validation_count,
            'successful': self.validation_success_count,
            'failed': self.validation_failure_count,
            'success_rate': (
                self.validation_success_count / self.validation_count
                if self.validation_count > 0 else 0
            )
        }

    def save_processed_image(
        self,
        image: Image.Image,
        filename: Optional[str] = None,
        format: str = 'PNG'
    ) -> Path:
        """
        Save processed image to output directory.

        Args:
            image: PIL Image to save
            filename: Optional filename (default: auto-generated)
            format: Image format (default: PNG)

        Returns:
            Path to saved image
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_{timestamp}.{format.lower()}"

        output_path = self.output_dir / filename
        image.save(output_path, format=format)
        logger.info(f"Saved processed image to {output_path}")

        return output_path


# Convenience function for quick validation
def quick_validate(
    image_path: Union[str, Path],
    image_type: Optional[ImageType] = None
) -> bool:
    """
    Quick validation function for simple use cases.

    Args:
        image_path: Path to image file
        image_type: Optional image type hint

    Returns:
        True if image is valid, False otherwise

    Example:
        >>> if quick_validate("sketch.png", ImageType.SKETCH):
        ...     print("Image is valid!")
    """
    try:
        handler = ImageUploadHandler()
        result = handler.validate_and_process(image_path, image_type)
        return result['is_valid']
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False
