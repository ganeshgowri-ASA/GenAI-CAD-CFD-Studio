"""
Layout optimization for Solar PV installations.

Provides tools for generating optimal solar panel layouts within site boundaries,
considering spacing constraints, coverage, and capacity objectives.
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, Point, box
from shapely.strtree import STRtree
from scipy.optimize import differential_evolution
from typing import Tuple, List, Dict, Optional, Callable
import warnings


class LayoutOptimizer:
    """
    Optimizes solar PV panel layouts for maximum efficiency.

    Handles grid generation, spatial optimization, and capacity calculations
    for solar panel installations.
    """

    def __init__(self):
        """Initialize the LayoutOptimizer."""
        self.rtree_index = None

    def generate_grid_layout(
        self,
        site_polygon: Polygon,
        module_dims: Tuple[float, float],
        spacing: Tuple[float, float],
        orientation: float = 0.0,
        return_all: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Generate a regular grid layout of solar modules within a site boundary.

        Args:
            site_polygon: Shapely polygon defining the site boundary
            module_dims: Tuple of (width, length) of each module in meters
            spacing: Tuple of (row_spacing, column_spacing) in meters
            orientation: Rotation angle in degrees (0 = North, clockwise positive)
            return_all: If True, return all generated modules; if False, only
                       those within site boundary

        Returns:
            GeoDataFrame with columns:
                - geometry: Polygon for each module
                - module_id: Unique identifier for each module
                - row: Row index in the grid
                - col: Column index in the grid
                - orientation: Orientation angle in degrees
                - center_x: X coordinate of module center
                - center_y: Y coordinate of module center

        Example:
            >>> optimizer = LayoutOptimizer()
            >>> site = Polygon([(0,0), (100,0), (100,100), (0,100)])
            >>> layout = optimizer.generate_grid_layout(
            ...     site, module_dims=(2.0, 1.0), spacing=(0.5, 0.5)
            ... )
        """
        width, length = module_dims
        row_spacing, col_spacing = spacing

        # Get site bounds
        minx, miny, maxx, maxy = site_polygon.bounds

        # Calculate grid dimensions
        step_x = width + col_spacing
        step_y = length + row_spacing

        # Generate grid points
        x_points = np.arange(minx, maxx, step_x)
        y_points = np.arange(miny, maxy, step_y)

        modules = []
        module_id = 0

        for row_idx, y in enumerate(y_points):
            for col_idx, x in enumerate(x_points):
                # Create module rectangle centered at (x, y)
                module_box = box(
                    x - width / 2,
                    y - length / 2,
                    x + width / 2,
                    y + length / 2
                )

                # Apply rotation if specified
                if orientation != 0.0:
                    module_box = self._rotate_polygon(
                        module_box, orientation, (x, y)
                    )

                # Check if module is within site boundary
                if return_all or site_polygon.contains(module_box):
                    modules.append({
                        'geometry': module_box,
                        'module_id': module_id,
                        'row': row_idx,
                        'col': col_idx,
                        'orientation': orientation,
                        'center_x': x,
                        'center_y': y
                    })
                    module_id += 1
                elif site_polygon.intersects(module_box):
                    # Optionally include partially intersecting modules
                    intersection = site_polygon.intersection(module_box)
                    if intersection.area / module_box.area > 0.9:  # 90% threshold
                        modules.append({
                            'geometry': module_box,
                            'module_id': module_id,
                            'row': row_idx,
                            'col': col_idx,
                            'orientation': orientation,
                            'center_x': x,
                            'center_y': y
                        })
                        module_id += 1

        # Create GeoDataFrame
        if not modules:
            # Return empty GeoDataFrame with proper schema
            return gpd.GeoDataFrame(
                columns=['module_id', 'row', 'col', 'orientation',
                        'center_x', 'center_y', 'geometry']
            )

        gdf = gpd.GeoDataFrame(modules, crs='EPSG:4326')

        # Build spatial index for fast collision detection
        self._build_spatial_index(gdf)

        return gdf

    def _rotate_polygon(
        self,
        polygon: Polygon,
        angle: float,
        origin: Tuple[float, float]
    ) -> Polygon:
        """
        Rotate a polygon around a point.

        Args:
            polygon: Polygon to rotate
            angle: Rotation angle in degrees (clockwise positive)
            origin: Point to rotate around (x, y)

        Returns:
            Rotated polygon
        """
        from shapely import affinity
        return affinity.rotate(polygon, angle, origin=origin)

    def _build_spatial_index(self, gdf: gpd.GeoDataFrame) -> None:
        """
        Build Rtree spatial index for fast collision detection.

        Args:
            gdf: GeoDataFrame containing module geometries
        """
        geometries = gdf.geometry.tolist()
        self.rtree_index = STRtree(geometries)

    def optimize_layout(
        self,
        site: Polygon,
        module_dims: Tuple[float, float],
        objectives: Dict[str, any],
        constraints: Optional[Dict[str, any]] = None
    ) -> gpd.GeoDataFrame:
        """
        Optimize module layout based on specified objectives.

        Args:
            site: Site boundary polygon
            module_dims: Module dimensions (width, length)
            objectives: Dictionary of optimization objectives:
                - 'maximize_count': bool - Maximize number of modules
                - 'minimize_cable_length': bool - Minimize total cable length
                - 'optimize_tilt': bool - Optimize tilt angle
                - 'target_orientation': float - Preferred orientation (degrees)
            constraints: Optional constraints dictionary:
                - 'min_spacing': Tuple[float, float] - Minimum spacing
                - 'max_spacing': Tuple[float, float] - Maximum spacing
                - 'fixed_orientation': float - Fixed orientation if specified

        Returns:
            Optimized GeoDataFrame of module layout

        Note:
            Uses differential evolution algorithm for multi-objective optimization.
        """
        # Set default constraints
        if constraints is None:
            constraints = {}

        min_spacing = constraints.get('min_spacing', (0.5, 0.5))
        max_spacing = constraints.get('max_spacing', (3.0, 3.0))
        fixed_orientation = constraints.get('fixed_orientation', None)

        # Define optimization bounds
        # [row_spacing, col_spacing, orientation]
        if fixed_orientation is not None:
            bounds = [
                (min_spacing[0], max_spacing[0]),  # row_spacing
                (min_spacing[1], max_spacing[1]),  # col_spacing
            ]
            use_orientation = False
        else:
            bounds = [
                (min_spacing[0], max_spacing[0]),  # row_spacing
                (min_spacing[1], max_spacing[1]),  # col_spacing
                (0, 360),  # orientation
            ]
            use_orientation = True

        # Define objective function
        def objective_function(params):
            if use_orientation:
                row_sp, col_sp, orient = params
            else:
                row_sp, col_sp = params
                orient = fixed_orientation if fixed_orientation is not None else 0.0

            spacing = (row_sp, col_sp)

            # Generate layout
            layout = self.generate_grid_layout(
                site, module_dims, spacing, orientation=orient
            )

            if len(layout) == 0:
                return 1e10  # Penalty for invalid layout

            # Multi-objective optimization
            score = 0.0

            if objectives.get('maximize_count', True):
                # Negative because we're minimizing
                score -= len(layout) * 1000

            if objectives.get('minimize_cable_length', False):
                cable_length = self._estimate_cable_length(layout)
                score += cable_length

            if objectives.get('optimize_tilt', False) and 'target_orientation' in objectives:
                target = objectives['target_orientation']
                orientation_penalty = abs(orient - target)
                score += orientation_penalty * 10

            return score

        # Run optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42,
                maxiter=50,
                popsize=15,
                atol=0.01,
                tol=0.01
            )

        # Generate optimal layout
        if use_orientation:
            optimal_spacing = (result.x[0], result.x[1])
            optimal_orientation = result.x[2]
        else:
            optimal_spacing = (result.x[0], result.x[1])
            optimal_orientation = fixed_orientation if fixed_orientation is not None else 0.0

        optimal_layout = self.generate_grid_layout(
            site, module_dims, optimal_spacing, orientation=optimal_orientation
        )

        return optimal_layout

    def _estimate_cable_length(self, layout: gpd.GeoDataFrame) -> float:
        """
        Estimate total cable length for a layout.

        Simple estimation based on connecting modules in a grid pattern.

        Args:
            layout: GeoDataFrame of module layout

        Returns:
            Estimated total cable length
        """
        if len(layout) == 0:
            return 0.0

        # Simple estimation: sum of distances between adjacent modules
        centers = layout[['center_x', 'center_y']].values
        total_length = 0.0

        for i in range(len(centers) - 1):
            dist = np.sqrt(
                (centers[i + 1][0] - centers[i][0]) ** 2 +
                (centers[i + 1][1] - centers[i][1]) ** 2
            )
            total_length += dist

        return total_length

    def calculate_coverage(self, site: Polygon, layout: gpd.GeoDataFrame) -> float:
        """
        Calculate the coverage percentage of modules over the site area.

        Args:
            site: Site boundary polygon
            layout: GeoDataFrame of module layout

        Returns:
            Coverage percentage (0-100)

        Example:
            >>> coverage = optimizer.calculate_coverage(site, layout)
            >>> print(f"Site coverage: {coverage:.2f}%")
        """
        if len(layout) == 0:
            return 0.0

        site_area = site.area
        modules_area = layout.geometry.area.sum()

        coverage_pct = (modules_area / site_area) * 100.0
        return min(coverage_pct, 100.0)  # Cap at 100%

    def calculate_total_capacity(
        self,
        layout: gpd.GeoDataFrame,
        module_power: float
    ) -> float:
        """
        Calculate total power capacity of the layout.

        Args:
            layout: GeoDataFrame of module layout
            module_power: Power rating of each module in kW

        Returns:
            Total capacity in kW

        Example:
            >>> total_kw = optimizer.calculate_total_capacity(layout, 0.4)
            >>> print(f"Total capacity: {total_kw:.2f} kW")
        """
        num_modules = len(layout)
        total_capacity = num_modules * module_power
        return total_capacity

    def check_collisions(self, layout: gpd.GeoDataFrame) -> List[Tuple[int, int]]:
        """
        Check for collisions between modules in the layout.

        Args:
            layout: GeoDataFrame of module layout

        Returns:
            List of tuples containing module_id pairs that collide

        Note:
            Uses Rtree spatial index for efficient collision detection.
        """
        if self.rtree_index is None:
            self._build_spatial_index(layout)

        collisions = []
        for idx, row in layout.iterrows():
            geom = row.geometry
            module_id = row.module_id

            # Query spatial index
            possible_matches_idx = self.rtree_index.query(geom)

            for match_idx in possible_matches_idx:
                match_geom = layout.iloc[match_idx].geometry
                match_id = layout.iloc[match_idx].module_id

                if module_id != match_id and geom.intersects(match_geom):
                    # Avoid duplicate pairs
                    if (match_id, module_id) not in collisions:
                        collisions.append((module_id, match_id))

        return collisions

    def get_layout_statistics(
        self,
        site: Polygon,
        layout: gpd.GeoDataFrame,
        module_power: float
    ) -> Dict[str, float]:
        """
        Get comprehensive statistics for a layout.

        Args:
            site: Site boundary polygon
            layout: GeoDataFrame of module layout
            module_power: Power rating per module in kW

        Returns:
            Dictionary containing:
                - num_modules: Number of modules
                - total_capacity_kw: Total capacity in kW
                - coverage_percent: Site coverage percentage
                - avg_spacing_x: Average spacing in x direction
                - avg_spacing_y: Average spacing in y direction
                - site_area: Total site area
                - modules_area: Total area covered by modules
        """
        stats = {
            'num_modules': len(layout),
            'total_capacity_kw': self.calculate_total_capacity(layout, module_power),
            'coverage_percent': self.calculate_coverage(site, layout),
            'site_area': site.area,
            'modules_area': layout.geometry.area.sum() if len(layout) > 0 else 0.0,
        }

        # Calculate average spacing
        if len(layout) > 1:
            centers = layout[['center_x', 'center_y']].values
            x_diffs = np.diff(np.sort(np.unique(centers[:, 0])))
            y_diffs = np.diff(np.sort(np.unique(centers[:, 1])))

            stats['avg_spacing_x'] = np.mean(x_diffs) if len(x_diffs) > 0 else 0.0
            stats['avg_spacing_y'] = np.mean(y_diffs) if len(y_diffs) > 0 else 0.0
        else:
            stats['avg_spacing_x'] = 0.0
            stats['avg_spacing_y'] = 0.0

        return stats
