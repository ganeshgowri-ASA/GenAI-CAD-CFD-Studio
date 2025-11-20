"""
Sketch interpretation using computer vision.

This module provides tools to analyze hand-drawn or digital sketches
and extract geometric information for CAD generation.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import io


class SketchInterpreter:
    """
    Interpret sketches using OpenCV computer vision.

    Capabilities:
    - Edge detection using Canny algorithm
    - Contour extraction and analysis
    - Shape recognition (rectangles, circles, polygons)
    - Conversion to CAD-compatible geometry
    """

    def __init__(self):
        """Initialize the SketchInterpreter."""
        self.image = None
        self.original_image = None
        self.edges = None
        self.contours = None

    def load_image(self, filepath: str) -> np.ndarray:
        """
        Load an image from file.

        Args:
            filepath: Path to image file

        Returns:
            Numpy array containing the image

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded

        Example:
            >>> interpreter = SketchInterpreter()
            >>> image = interpreter.load_image("sketch.png")
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(filepath)

            if image is None:
                raise ValueError(f"Could not load image from {filepath}")

            # Store both original and working copy
            self.original_image = image.copy()
            self.image = image

            return image

        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Load an image from bytes.

        Args:
            image_bytes: Image data as bytes

        Returns:
            Numpy array containing the image

        Example:
            >>> with open("sketch.png", "rb") as f:
            ...     image = interpreter.load_image_from_bytes(f.read())
        """
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image from bytes")

        self.original_image = image.copy()
        self.image = image

        return image

    def detect_edges(
        self,
        image: Optional[np.ndarray] = None,
        canny_threshold1: int = 50,
        canny_threshold2: int = 150,
        use_adaptive_threshold: bool = True
    ) -> np.ndarray:
        """
        Detect edges in the image using Canny edge detection.

        Args:
            image: Image array (uses loaded image if None)
            canny_threshold1: Lower threshold for Canny
            canny_threshold2: Upper threshold for Canny
            use_adaptive_threshold: Use adaptive thresholding preprocessing

        Returns:
            Binary edge image

        Example:
            >>> interpreter.load_image("sketch.png")
            >>> edges = interpreter.detect_edges()
        """
        if image is None:
            if self.image is None:
                raise ValueError("No image loaded. Call load_image() first.")
            image = self.image

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if use_adaptive_threshold:
            # Apply adaptive thresholding for better edge detection
            # This helps with varying lighting conditions
            adaptive = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )

            # Apply Canny edge detection on thresholded image
            edges = cv2.Canny(adaptive, canny_threshold1, canny_threshold2)
        else:
            # Direct Canny edge detection
            edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

        # Apply morphological operations to connect broken edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        self.edges = edges
        return edges

    def extract_contours(
        self,
        edges: Optional[np.ndarray] = None,
        min_area: float = 100.0
    ) -> List[np.ndarray]:
        """
        Extract contours from edge image.

        Args:
            edges: Edge image (uses detected edges if None)
            min_area: Minimum contour area to include

        Returns:
            List of contours as numpy arrays

        Example:
            >>> edges = interpreter.detect_edges()
            >>> contours = interpreter.extract_contours(edges)
        """
        if edges is None:
            if self.edges is None:
                raise ValueError("No edges detected. Call detect_edges() first.")
            edges = self.edges

        # Find contours
        contours, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE  # Compress contours
        )

        # Filter by area
        filtered_contours = [
            cnt for cnt in contours
            if cv2.contourArea(cnt) >= min_area
        ]

        # Sort by area (largest first)
        filtered_contours = sorted(
            filtered_contours,
            key=cv2.contourArea,
            reverse=True
        )

        self.contours = filtered_contours
        return filtered_contours

    def contour_to_geometry(
        self,
        contours: Optional[List[np.ndarray]] = None
    ) -> List[Dict]:
        """
        Convert contours to geometric descriptions.

        Detects basic shapes:
        - Rectangle/Square
        - Circle/Ellipse
        - Triangle
        - Polygon (arbitrary number of sides)

        Args:
            contours: List of contours (uses extracted contours if None)

        Returns:
            List of dictionaries describing detected shapes:
            {
                'type': str (rectangle, circle, triangle, polygon),
                'points': list of (x, y) tuples,
                'area': float,
                'perimeter': float,
                'center': (x, y) tuple,
                'properties': dict (shape-specific properties)
            }

        Example:
            >>> geometries = interpreter.contour_to_geometry()
            >>> geometries[0]['type']  # 'rectangle'
        """
        if contours is None:
            if self.contours is None:
                raise ValueError("No contours extracted. Call extract_contours() first.")
            contours = self.contours

        geometries = []

        for contour in contours:
            # Calculate basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Approximate contour to polygon
            epsilon = 0.04 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Detect shape based on number of vertices
            vertices = len(approx)

            geometry = {
                'area': area,
                'perimeter': perimeter,
                'center': (cx, cy),
                'vertices': vertices,
                'points': approx.reshape(-1, 2).tolist(),
                'properties': {}
            }

            # Classify shape
            if vertices == 3:
                geometry['type'] = 'triangle'

            elif vertices == 4:
                # Check if rectangle or square
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h != 0 else 0

                geometry['type'] = 'rectangle'
                geometry['properties'] = {
                    'width': w,
                    'height': h,
                    'aspect_ratio': aspect_ratio,
                    'is_square': 0.9 <= aspect_ratio <= 1.1
                }

            elif vertices > 8:
                # Likely a circle or ellipse
                # Check circularity
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                if circularity > 0.8:
                    # Fit circle or ellipse
                    if len(contour) >= 5:
                        ellipse = cv2.fitEllipse(contour)
                        (cx, cy), (MA, ma), angle = ellipse

                        geometry['type'] = 'circle' if abs(MA - ma) < MA * 0.1 else 'ellipse'
                        geometry['center'] = (int(cx), int(cy))
                        geometry['properties'] = {
                            'major_axis': MA,
                            'minor_axis': ma,
                            'angle': angle,
                            'radius': (MA + ma) / 4,  # Average radius
                            'circularity': circularity
                        }
                else:
                    geometry['type'] = 'polygon'
                    geometry['properties'] = {
                        'sides': vertices,
                        'circularity': circularity
                    }

            else:
                # Generic polygon
                geometry['type'] = 'polygon'
                geometry['properties'] = {
                    'sides': vertices
                }

            geometries.append(geometry)

        return geometries

    def visualize_detection(
        self,
        image: Optional[np.ndarray] = None,
        contours: Optional[List[np.ndarray]] = None,
        geometries: Optional[List[Dict]] = None
    ) -> np.ndarray:
        """
        Create a visualization with detected shapes annotated.

        Args:
            image: Base image (uses original if None)
            contours: Contours to draw (uses extracted if None)
            geometries: Geometric descriptions to annotate (optional)

        Returns:
            Annotated image as numpy array

        Example:
            >>> interpreter.load_image("sketch.png")
            >>> interpreter.detect_edges()
            >>> interpreter.extract_contours()
            >>> geometries = interpreter.contour_to_geometry()
            >>> annotated = interpreter.visualize_detection(geometries=geometries)
            >>> cv2.imwrite("annotated.png", annotated)
        """
        if image is None:
            if self.original_image is None:
                raise ValueError("No image loaded. Call load_image() first.")
            image = self.original_image.copy()
        else:
            image = image.copy()

        if contours is None:
            if self.contours is None:
                # Try to extract contours
                if self.edges is not None:
                    self.extract_contours()
            contours = self.contours

        # Draw contours
        if contours is not None:
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

        # Annotate with geometry information
        if geometries is not None:
            for i, geom in enumerate(geometries):
                # Draw center point
                center = geom['center']
                cv2.circle(image, center, 5, (0, 0, 255), -1)

                # Add label
                label = f"{geom['type']} #{i+1}"
                cv2.putText(
                    image,
                    label,
                    (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

                # Add properties for specific shapes
                if geom['type'] == 'rectangle':
                    props = geom['properties']
                    text = f"W:{props['width']} H:{props['height']}"
                    cv2.putText(
                        image,
                        text,
                        (center[0] + 10, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )
                elif geom['type'] == 'circle':
                    props = geom['properties']
                    text = f"R:{props['radius']:.1f}"
                    cv2.putText(
                        image,
                        text,
                        (center[0] + 10, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )

        return image

    def save_visualization(
        self,
        output_path: str,
        image: Optional[np.ndarray] = None
    ) -> None:
        """
        Save visualization to file.

        Args:
            output_path: Path to save the image
            image: Image to save (creates visualization if None)

        Example:
            >>> interpreter.save_visualization("output.png")
        """
        if image is None:
            image = self.visualize_detection()

        cv2.imwrite(output_path, image)

    def get_cad_specifications(self) -> Dict:
        """
        Get CAD specifications from detected geometry.

        Returns:
            Dictionary suitable for CAD generation with detected shapes and dimensions

        Example:
            >>> specs = interpreter.get_cad_specifications()
            >>> specs['shapes'][0]['type']  # 'rectangle'
        """
        if self.contours is None:
            raise ValueError("No contours available. Process an image first.")

        geometries = self.contour_to_geometry()

        return {
            'num_shapes': len(geometries),
            'shapes': geometries,
            'image_dimensions': {
                'width': self.image.shape[1] if self.image is not None else 0,
                'height': self.image.shape[0] if self.image is not None else 0
            }
        }
