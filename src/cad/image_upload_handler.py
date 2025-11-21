"""
Image Upload Handler with Robust Preprocessing

This module handles image uploads with comprehensive validation, preprocessing,
and error handling for CAD generation workflows.

Features:
- Format validation (JPEG, PNG, BMP, TIFF, WebP)
- Automatic format conversion to RGB
- Image resizing and optimization
- Dual support for PIL and OpenCV
- Graceful fallback handling
- Comprehensive logging
"""

import logging
from typing import Optional, Tuple, Union, Dict, Any
from pathlib import Path
import io
import base64
from enum import Enum

# Optional imports with fallback handling
try:
    from PIL import Image
    import PIL.ImageOps
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL/Pillow not installed. Image processing capabilities limited.")

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("OpenCV not installed. Using PIL as primary image processor.")


logger = logging.getLogger(__name__)


class ImageFormat(Enum):
    """Supported image formats"""
    JPEG = "JPEG"
    PNG = "PNG"
    BMP = "BMP"
    TIFF = "TIFF"
    WEBP = "WebP"


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class ImageUploadHandler:
    """
    Handles image uploads with validation, preprocessing, and conversion.

    Supports both PIL and OpenCV backends with automatic fallback.
    """

    # Maximum dimensions for uploaded images
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096

    # Maximum file size (20MB)
    MAX_FILE_SIZE = 20 * 1024 * 1024

    # Default target size for processing
    DEFAULT_TARGET_SIZE = (1024, 1024)

    # Supported formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        maintain_aspect_ratio: bool = True,
        convert_to_rgb: bool = True,
        quality: int = 95
    ):
        """
        Initialize the image upload handler.

        Args:
            target_size: Target size for resizing (width, height). None to skip resizing.
            maintain_aspect_ratio: Whether to maintain aspect ratio when resizing
            convert_to_rgb: Whether to convert images to RGB mode
            quality: JPEG quality for output (1-100)
        """
        self.target_size = target_size or self.DEFAULT_TARGET_SIZE
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.convert_to_rgb = convert_to_rgb
        self.quality = max(1, min(100, quality))

        # Check available backends
        self.backend = self._determine_backend()
        logger.info(f"ImageUploadHandler initialized with backend: {self.backend}")

    def _determine_backend(self) -> str:
        """Determine which image processing backend to use"""
        if HAS_PIL and HAS_CV2:
            return "PIL_PRIMARY"  # Prefer PIL for better format support
        elif HAS_PIL:
            return "PIL_ONLY"
        elif HAS_CV2:
            return "CV2_ONLY"
        else:
            raise ImportError(
                "Neither PIL/Pillow nor OpenCV is installed. "
                "Please install at least one: pip install pillow OR pip install opencv-python"
            )

    def validate_image(
        self,
        image_data: Union[bytes, str, Path],
        check_size: bool = True
    ) -> Tuple[bool, str]:
        """
        Validate image format and basic properties.

        Args:
            image_data: Image data as bytes, file path, or Path object
            check_size: Whether to check file size limits

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Handle different input types
            if isinstance(image_data, (str, Path)):
                file_path = Path(image_data)

                # Check file exists
                if not file_path.exists():
                    return False, f"File not found: {file_path}"

                # Check file extension
                if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                    return False, f"Unsupported format: {file_path.suffix}"

                # Check file size
                if check_size:
                    file_size = file_path.stat().st_size
                    if file_size > self.MAX_FILE_SIZE:
                        return False, f"File too large: {file_size / (1024*1024):.2f}MB (max: 20MB)"

                with open(file_path, 'rb') as f:
                    image_bytes = f.read()
            elif isinstance(image_data, bytes):
                image_bytes = image_data

                # Check size
                if check_size and len(image_bytes) > self.MAX_FILE_SIZE:
                    return False, f"File too large: {len(image_bytes) / (1024*1024):.2f}MB (max: 20MB)"
            else:
                return False, "Invalid image data type"

            # Try to open and validate the image
            if HAS_PIL:
                try:
                    img = Image.open(io.BytesIO(image_bytes))

                    # Check dimensions
                    if img.width > self.MAX_WIDTH or img.height > self.MAX_HEIGHT:
                        return False, f"Image dimensions too large: {img.width}x{img.height} (max: {self.MAX_WIDTH}x{self.MAX_HEIGHT})"

                    # Verify it's a valid image by trying to load it
                    img.verify()

                    return True, "Image is valid"

                except Exception as e:
                    return False, f"Invalid image file: {str(e)}"

            elif HAS_CV2:
                try:
                    # Decode image
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

                    if img is None:
                        return False, "Failed to decode image"

                    # Check dimensions
                    height, width = img.shape[:2]
                    if width > self.MAX_WIDTH or height > self.MAX_HEIGHT:
                        return False, f"Image dimensions too large: {width}x{height} (max: {self.MAX_WIDTH}x{self.MAX_HEIGHT})"

                    return True, "Image is valid"

                except Exception as e:
                    return False, f"Invalid image file: {str(e)}"

            return False, "No image processing backend available"

        except Exception as e:
            logger.error(f"Error validating image: {e}", exc_info=True)
            return False, f"Validation error: {str(e)}"

    def preprocess_image(
        self,
        image_data: Union[bytes, str, Path],
        resize: bool = True
    ) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Preprocess image with validation, conversion, and resizing.

        Args:
            image_data: Image data as bytes, file path, or Path object
            resize: Whether to resize the image

        Returns:
            Tuple of (processed_image, metadata)
            processed_image is PIL.Image or numpy.ndarray depending on backend
        """
        try:
            # Validate first
            is_valid, error_msg = self.validate_image(image_data)
            if not is_valid:
                logger.error(f"Image validation failed: {error_msg}")
                raise ImageProcessingError(error_msg)

            # Load image
            if isinstance(image_data, (str, Path)):
                with open(image_data, 'rb') as f:
                    image_bytes = f.read()
            else:
                image_bytes = image_data

            # Process based on backend
            if "PIL" in self.backend:
                return self._preprocess_pil(image_bytes, resize)
            elif "CV2" in self.backend:
                return self._preprocess_cv2(image_bytes, resize)
            else:
                raise ImageProcessingError("No image processing backend available")

        except ImageProcessingError:
            raise
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}", exc_info=True)
            raise ImageProcessingError(f"Preprocessing failed: {str(e)}")

    def _preprocess_pil(
        self,
        image_bytes: bytes,
        resize: bool
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Preprocess image using PIL/Pillow"""
        try:
            # Open image
            img = Image.open(io.BytesIO(image_bytes))
            original_format = img.format
            original_mode = img.mode
            original_size = img.size

            logger.info(f"Loaded image: format={original_format}, mode={original_mode}, size={original_size}")

            # Convert to RGB if needed
            if self.convert_to_rgb and img.mode != 'RGB':
                logger.info(f"Converting from {img.mode} to RGB")
                if img.mode == 'RGBA':
                    # Handle transparency by adding white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                    img = background
                else:
                    img = img.convert('RGB')

            # Auto-orient based on EXIF data
            try:
                img = PIL.ImageOps.exif_transpose(img)
            except Exception as e:
                logger.debug(f"EXIF orientation not available or failed: {e}")

            # Resize if needed
            if resize and self.target_size:
                img = self._resize_pil(img)

            # Prepare metadata
            metadata = {
                'original_format': original_format,
                'original_mode': original_mode,
                'original_size': original_size,
                'processed_mode': img.mode,
                'processed_size': img.size,
                'backend': 'PIL'
            }

            logger.info(f"Image preprocessed successfully: {metadata}")
            return img, metadata

        except Exception as e:
            logger.error(f"PIL preprocessing error: {e}", exc_info=True)
            raise ImageProcessingError(f"PIL processing failed: {str(e)}")

    def _resize_pil(self, img: Image.Image) -> Image.Image:
        """Resize PIL image"""
        if self.maintain_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            img.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        else:
            # Resize to exact dimensions
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)

        return img

    def _preprocess_cv2(
        self,
        image_bytes: bytes,
        resize: bool
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess image using OpenCV"""
        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise ImageProcessingError("Failed to decode image with OpenCV")

            original_shape = img.shape
            channels = img.shape[2] if len(img.shape) == 3 else 1

            logger.info(f"Loaded image: shape={original_shape}, channels={channels}")

            # Convert to RGB if needed
            if self.convert_to_rgb:
                if channels == 4:  # RGBA
                    # Handle transparency
                    alpha = img[:, :, 3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    # Add white background where transparent
                    mask = alpha < 255
                    img[mask] = [255, 255, 255]
                elif channels == 1:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # OpenCV uses BGR, convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize if needed
            if resize and self.target_size:
                img = self._resize_cv2(img)

            # Prepare metadata
            metadata = {
                'original_shape': original_shape,
                'original_channels': channels,
                'processed_shape': img.shape,
                'backend': 'OpenCV'
            }

            logger.info(f"Image preprocessed successfully: {metadata}")
            return img, metadata

        except Exception as e:
            logger.error(f"OpenCV preprocessing error: {e}", exc_info=True)
            raise ImageProcessingError(f"OpenCV processing failed: {str(e)}")

    def _resize_cv2(self, img: np.ndarray) -> np.ndarray:
        """Resize OpenCV image"""
        height, width = img.shape[:2]
        target_width, target_height = self.target_size

        if self.maintain_aspect_ratio:
            # Calculate new size maintaining aspect ratio
            aspect = width / height
            if width > height:
                new_width = target_width
                new_height = int(target_width / aspect)
            else:
                new_height = target_height
                new_width = int(target_height * aspect)

            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Resize to exact dimensions
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LANCZOS4)

        return img

    def image_to_bytes(
        self,
        image: Union[Image.Image, np.ndarray],
        format: str = 'PNG'
    ) -> bytes:
        """
        Convert processed image back to bytes.

        Args:
            image: PIL Image or numpy array
            format: Output format (PNG, JPEG, etc.)

        Returns:
            Image as bytes
        """
        try:
            if isinstance(image, Image.Image):
                # PIL Image
                buffer = io.BytesIO()
                image.save(buffer, format=format, quality=self.quality)
                return buffer.getvalue()

            elif isinstance(image, np.ndarray):
                # OpenCV image
                if format.upper() == 'JPEG':
                    ext = '.jpg'
                    params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                else:
                    ext = '.png'
                    params = []

                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                success, encoded = cv2.imencode(ext, image, params)
                if not success:
                    raise ImageProcessingError("Failed to encode image")

                return encoded.tobytes()

            else:
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            logger.error(f"Error converting image to bytes: {e}", exc_info=True)
            raise ImageProcessingError(f"Conversion to bytes failed: {str(e)}")

    def image_to_base64(
        self,
        image: Union[Image.Image, np.ndarray],
        format: str = 'PNG'
    ) -> str:
        """
        Convert image to base64 encoded string.

        Args:
            image: PIL Image or numpy array
            format: Output format

        Returns:
            Base64 encoded string
        """
        try:
            image_bytes = self.image_to_bytes(image, format)
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}", exc_info=True)
            raise ImageProcessingError(f"Conversion to base64 failed: {str(e)}")

    def save_processed_image(
        self,
        image: Union[Image.Image, np.ndarray],
        output_path: Union[str, Path],
        format: Optional[str] = None
    ) -> bool:
        """
        Save processed image to file.

        Args:
            image: PIL Image or numpy array
            output_path: Output file path
            format: Output format (auto-detected from extension if None)

        Returns:
            True if successful
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Auto-detect format from extension
            if format is None:
                format = output_path.suffix[1:].upper()
                if format == 'JPG':
                    format = 'JPEG'

            if isinstance(image, Image.Image):
                image.save(output_path, format=format, quality=self.quality)
            elif isinstance(image, np.ndarray):
                # Convert RGB to BGR for OpenCV
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image)
            else:
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")

            logger.info(f"Image saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving image: {e}", exc_info=True)
            return False


# Convenience functions
def validate_uploaded_image(image_data: Union[bytes, str, Path]) -> Tuple[bool, str]:
    """
    Quick validation function for uploaded images.

    Args:
        image_data: Image data as bytes, file path, or Path object

    Returns:
        Tuple of (is_valid, error_message)
    """
    handler = ImageUploadHandler()
    return handler.validate_image(image_data)


def preprocess_uploaded_image(
    image_data: Union[bytes, str, Path],
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Quick preprocessing function for uploaded images.

    Args:
        image_data: Image data as bytes, file path, or Path object
        target_size: Target size for resizing

    Returns:
        Tuple of (processed_image, metadata)
    """
    handler = ImageUploadHandler(target_size=target_size)
    return handler.preprocess_image(image_data)
