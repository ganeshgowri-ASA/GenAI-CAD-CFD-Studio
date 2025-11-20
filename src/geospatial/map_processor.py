"""
Map processing utilities for geospatial data handling.

Provides tools for loading, saving, transforming, and clipping geospatial data.
"""

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from typing import Union
import pyproj


class MapProcessor:
    """
    Handles geospatial data processing operations.

    Provides methods for loading/saving GeoJSON files, coordinate reference
    system (CRS) transformations, and spatial clipping operations.
    """

    def __init__(self):
        """Initialize the MapProcessor."""
        pass

    def load_geojson(self, filepath: str) -> gpd.GeoDataFrame:
        """
        Load a GeoJSON file into a GeoDataFrame.

        Args:
            filepath: Path to the GeoJSON file

        Returns:
            GeoDataFrame containing the loaded geospatial data

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
        """
        try:
            gdf = gpd.read_file(filepath)
            return gdf
        except Exception as e:
            raise ValueError(f"Failed to load GeoJSON from {filepath}: {str(e)}")

    def save_geojson(self, gdf: gpd.GeoDataFrame, filepath: str) -> None:
        """
        Save a GeoDataFrame to a GeoJSON file.

        Args:
            gdf: GeoDataFrame to save
            filepath: Output file path

        Raises:
            ValueError: If the GeoDataFrame is invalid or save fails
        """
        try:
            gdf.to_file(filepath, driver='GeoJSON')
        except Exception as e:
            raise ValueError(f"Failed to save GeoJSON to {filepath}: {str(e)}")

    def transform_crs(
        self,
        gdf: gpd.GeoDataFrame,
        from_crs: Union[str, int],
        to_crs: Union[str, int]
    ) -> gpd.GeoDataFrame:
        """
        Transform GeoDataFrame from one CRS to another.

        Args:
            gdf: Input GeoDataFrame
            from_crs: Source CRS (EPSG code or proj string)
            to_crs: Target CRS (EPSG code or proj string)

        Returns:
            GeoDataFrame transformed to the target CRS

        Example:
            >>> gdf_wgs84 = processor.transform_crs(gdf, 'EPSG:3857', 'EPSG:4326')
        """
        # Set the source CRS if not already set
        if gdf.crs is None:
            gdf = gdf.set_crs(from_crs)
        elif str(gdf.crs) != str(from_crs):
            # If CRS is already set but different, still set it
            gdf = gdf.set_crs(from_crs, allow_override=True)

        # Transform to target CRS
        transformed_gdf = gdf.to_crs(to_crs)
        return transformed_gdf

    def clip_to_boundary(
        self,
        gdf: gpd.GeoDataFrame,
        boundary: Union[Polygon, MultiPolygon, gpd.GeoDataFrame]
    ) -> gpd.GeoDataFrame:
        """
        Clip a GeoDataFrame to a boundary polygon.

        Args:
            gdf: Input GeoDataFrame to clip
            boundary: Boundary polygon or GeoDataFrame defining the clip region

        Returns:
            GeoDataFrame clipped to the boundary

        Raises:
            ValueError: If CRS mismatch between gdf and boundary
        """
        # Convert boundary to GeoDataFrame if it's a geometry
        if isinstance(boundary, (Polygon, MultiPolygon)):
            boundary_gdf = gpd.GeoDataFrame(
                geometry=[boundary],
                crs=gdf.crs
            )
        else:
            boundary_gdf = boundary

        # Check CRS compatibility
        if gdf.crs != boundary_gdf.crs:
            raise ValueError(
                f"CRS mismatch: GeoDataFrame has {gdf.crs}, "
                f"boundary has {boundary_gdf.crs}. "
                f"Transform to same CRS first."
            )

        # Perform clipping
        clipped_gdf = gpd.clip(gdf, boundary_gdf)
        return clipped_gdf

    def get_crs_info(self, gdf: gpd.GeoDataFrame) -> dict:
        """
        Get information about the GeoDataFrame's CRS.

        Args:
            gdf: Input GeoDataFrame

        Returns:
            Dictionary containing CRS information
        """
        if gdf.crs is None:
            return {"crs": None, "units": None, "proj": None}

        return {
            "crs": str(gdf.crs),
            "units": gdf.crs.axis_info[0].unit_name if gdf.crs.axis_info else None,
            "proj": gdf.crs.to_proj4() if hasattr(gdf.crs, 'to_proj4') else None,
        }

    def calculate_area(self, gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """
        Calculate the area of geometries in a GeoDataFrame.

        For accurate area calculations, the GeoDataFrame should be in a
        projected CRS (not geographic lat/lon).

        Args:
            gdf: Input GeoDataFrame

        Returns:
            GeoSeries containing area values

        Note:
            If CRS is geographic (degrees), area will be in square degrees
            which is not meaningful. Use a projected CRS for accurate results.
        """
        return gdf.geometry.area

    def calculate_bounds(self, gdf: gpd.GeoDataFrame) -> tuple:
        """
        Calculate the bounding box of a GeoDataFrame.

        Args:
            gdf: Input GeoDataFrame

        Returns:
            Tuple of (minx, miny, maxx, maxy)
        """
        bounds = gdf.total_bounds
        return tuple(bounds)
