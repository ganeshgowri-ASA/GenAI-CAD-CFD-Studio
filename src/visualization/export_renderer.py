"""
High-quality export and rendering utilities for PyVista visualizations.

Provides off-screen rendering, animation generation, and batch rendering
capabilities for production-quality outputs.
"""

import pyvista as pv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import warnings


class ExportRenderer:
    """
    High-quality export renderer for PyVista visualizations.

    Handles off-screen rendering, animation generation, batch rendering,
    and export to various formats with advanced rendering options.
    """

    def __init__(self):
        """Initialize export renderer."""
        pass

    def render_high_quality(
        self,
        mesh: Union[pv.DataSet, List[pv.DataSet]],
        output_file: Union[str, Path],
        resolution: Tuple[int, int] = (3840, 2160),
        **kwargs
    ) -> None:
        """
        Render high-quality image with advanced options.

        Args:
            mesh: PyVista mesh or list of meshes to render
            output_file: Output file path (.png, .jpg, .svg)
            resolution: Image resolution (width, height), default 4K
            **kwargs: Additional rendering options
                - anti_aliasing: bool, default True
                - ambient_occlusion: bool, default True
                - background: color, default 'white'
                - camera_position: str or tuple, default 'iso'
                - lighting: bool, default True
                - shadows: bool, default True
                - scalars: str, scalar field name
                - cmap: str, colormap
                - show_edges: bool, default False
                - smooth_shading: bool, default True
                - color: mesh color, default 'lightblue'
                - opacity: float, default 1.0
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create off-screen plotter
        plotter = pv.Plotter(
            window_size=resolution,
            off_screen=True
        )

        # Configure background
        background = kwargs.get('background', 'white')
        plotter.set_background(background)

        # Handle single mesh or list of meshes
        meshes = mesh if isinstance(mesh, list) else [mesh]

        # Add meshes
        for m in meshes:
            mesh_kwargs = {
                'show_edges': kwargs.get('show_edges', False),
                'smooth_shading': kwargs.get('smooth_shading', True),
                'opacity': kwargs.get('opacity', 1.0),
            }

            if 'scalars' in kwargs and kwargs['scalars'] in m.array_names:
                mesh_kwargs['scalars'] = kwargs['scalars']
                mesh_kwargs['cmap'] = kwargs.get('cmap', 'jet')
                mesh_kwargs['show_scalar_bar'] = kwargs.get('show_scalar_bar', True)
            else:
                mesh_kwargs['color'] = kwargs.get('color', 'lightblue')

            plotter.add_mesh(m, **mesh_kwargs)

        # Configure lighting
        if kwargs.get('lighting', True):
            plotter.add_light(pv.Light(
                position=(10, 10, 10),
                light_type='camera',
                intensity=0.5
            ))
            plotter.add_light(pv.Light(
                position=(-10, -10, 10),
                light_type='camera',
                intensity=0.3
            ))

        # Enable shadows
        if kwargs.get('shadows', True):
            plotter.enable_shadows()

        # Enable anti-aliasing
        if kwargs.get('anti_aliasing', True):
            plotter.enable_anti_aliasing('msaa')

        # Enable ambient occlusion (SSAO)
        if kwargs.get('ambient_occlusion', True):
            try:
                plotter.enable_ssao(radius=10, bias=0.5)
            except Exception:
                warnings.warn("SSAO not available in this PyVista version")

        # Set camera position
        camera_pos = kwargs.get('camera_position', 'iso')
        plotter.camera_position = camera_pos
        plotter.reset_camera()

        # Render and save
        transparent = kwargs.get('transparent_background', False)
        plotter.screenshot(
            filename=str(output_file),
            transparent_background=transparent,
            return_img=False
        )

        plotter.close()
        print(f"High-quality image saved to: {output_file}")

    def create_animation(
        self,
        meshes_sequence: List[pv.DataSet],
        output_video: Union[str, Path],
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        **kwargs
    ) -> None:
        """
        Generate MP4/GIF animation from mesh sequence.

        Args:
            meshes_sequence: List of meshes representing animation frames
            output_video: Output video file path (.mp4, .gif)
            fps: Frames per second
            resolution: Video resolution (width, height)
            **kwargs: Additional rendering options
                - background: color, default 'white'
                - camera_position: str or tuple, default 'iso'
                - orbit: bool, rotate camera during animation
                - scalars: str, scalar field name
                - cmap: str, colormap
                - show_edges: bool, default False
        """
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Create off-screen plotter
        plotter = pv.Plotter(
            window_size=resolution,
            off_screen=True
        )

        plotter.set_background(kwargs.get('background', 'white'))

        # Open movie file
        plotter.open_movie(str(output_video), framerate=fps)

        # Determine if orbit camera
        orbit = kwargs.get('orbit', False)
        n_frames = len(meshes_sequence)

        # Render each frame
        for idx, mesh in enumerate(meshes_sequence):
            plotter.clear()

            # Add mesh
            mesh_kwargs = {
                'show_edges': kwargs.get('show_edges', False),
                'smooth_shading': True,
            }

            if 'scalars' in kwargs and kwargs['scalars'] in mesh.array_names:
                mesh_kwargs['scalars'] = kwargs['scalars']
                mesh_kwargs['cmap'] = kwargs.get('cmap', 'jet')
                mesh_kwargs['show_scalar_bar'] = kwargs.get('show_scalar_bar', True)
            else:
                mesh_kwargs['color'] = kwargs.get('color', 'lightblue')

            plotter.add_mesh(mesh, **mesh_kwargs)

            # Set camera
            if orbit:
                # Rotate camera around object
                angle = 360 * idx / n_frames
                plotter.camera_position = [
                    (np.cos(np.radians(angle)) * 3, np.sin(np.radians(angle)) * 3, 2),
                    (0, 0, 0),
                    (0, 0, 1)
                ]
            else:
                plotter.camera_position = kwargs.get('camera_position', 'iso')
                plotter.reset_camera()

            # Write frame
            plotter.write_frame()

        # Close and finalize video
        plotter.close()
        print(f"Animation saved to: {output_video}")

    def batch_render(
        self,
        mesh_list: List[Union[pv.DataSet, Tuple[pv.DataSet, str]]],
        views: List[str],
        output_dir: Union[str, Path],
        resolution: Tuple[int, int] = (1920, 1080),
        **kwargs
    ) -> List[Path]:
        """
        Render multiple views of multiple meshes.

        Args:
            mesh_list: List of meshes or (mesh, name) tuples
            views: List of view positions ('iso', 'xy', 'xz', 'yz', etc.)
            output_dir: Output directory for rendered images
            resolution: Image resolution
            **kwargs: Additional rendering options (passed to render_high_quality)

        Returns:
            List of generated file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # Process each mesh
        for mesh_idx, mesh_item in enumerate(mesh_list):
            # Extract mesh and name
            if isinstance(mesh_item, tuple):
                mesh, name = mesh_item
            else:
                mesh = mesh_item
                name = f"mesh_{mesh_idx:03d}"

            # Render each view
            for view in views:
                output_file = output_dir / f"{name}_{view}.png"

                # Render with specified view
                self.render_high_quality(
                    mesh,
                    output_file,
                    resolution=resolution,
                    camera_position=view,
                    **kwargs
                )

                generated_files.append(output_file)

        print(f"Batch rendering complete. Generated {len(generated_files)} images.")
        return generated_files

    def create_turntable_animation(
        self,
        mesh: pv.DataSet,
        output_video: Union[str, Path],
        n_frames: int = 120,
        fps: int = 30,
        resolution: Tuple[int, int] = (1920, 1080),
        elevation: float = 30.0,
        **kwargs
    ) -> None:
        """
        Create turntable (360-degree rotation) animation.

        Args:
            mesh: PyVista mesh to render
            output_video: Output video file path
            n_frames: Number of frames for full rotation
            fps: Frames per second
            resolution: Video resolution
            elevation: Camera elevation angle in degrees
            **kwargs: Additional rendering options
        """
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Create off-screen plotter
        plotter = pv.Plotter(
            window_size=resolution,
            off_screen=True
        )

        plotter.set_background(kwargs.get('background', 'white'))

        # Add mesh once
        mesh_kwargs = {
            'show_edges': kwargs.get('show_edges', False),
            'smooth_shading': True,
        }

        if 'scalars' in kwargs and kwargs['scalars'] in mesh.array_names:
            mesh_kwargs['scalars'] = kwargs['scalars']
            mesh_kwargs['cmap'] = kwargs.get('cmap', 'jet')
            mesh_kwargs['show_scalar_bar'] = kwargs.get('show_scalar_bar', True)
        else:
            mesh_kwargs['color'] = kwargs.get('color', 'lightblue')

        plotter.add_mesh(mesh, **mesh_kwargs)

        # Calculate camera orbit
        center = mesh.center
        bounds = mesh.bounds
        radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 1.5

        # Open movie
        plotter.open_movie(str(output_video), framerate=fps)

        # Generate frames
        for frame in range(n_frames):
            angle = 360 * frame / n_frames

            # Calculate camera position
            x = center[0] + radius * np.cos(np.radians(angle))
            y = center[1] + radius * np.sin(np.radians(angle))
            z = center[2] + radius * np.sin(np.radians(elevation))

            plotter.camera_position = [
                (x, y, z),
                center,
                (0, 0, 1)
            ]

            plotter.write_frame()

        plotter.close()
        print(f"Turntable animation saved to: {output_video}")

    def export_interactive_html(
        self,
        mesh: pv.DataSet,
        output_file: Union[str, Path],
        **kwargs
    ) -> None:
        """
        Export interactive HTML visualization.

        Args:
            mesh: PyVista mesh to export
            output_file: Output HTML file path
            **kwargs: Additional options
                - scalars: str, scalar field name
                - cmap: str, colormap
                - show_edges: bool
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create plotter
        plotter = pv.Plotter()

        # Add mesh
        mesh_kwargs = {
            'show_edges': kwargs.get('show_edges', False),
        }

        if 'scalars' in kwargs and kwargs['scalars'] in mesh.array_names:
            mesh_kwargs['scalars'] = kwargs['scalars']
            mesh_kwargs['cmap'] = kwargs.get('cmap', 'jet')
            mesh_kwargs['show_scalar_bar'] = True
        else:
            mesh_kwargs['color'] = kwargs.get('color', 'lightblue')

        plotter.add_mesh(mesh, **mesh_kwargs)
        plotter.camera_position = 'iso'
        plotter.reset_camera()

        # Export to HTML
        plotter.export_html(str(output_file))
        plotter.close()

        print(f"Interactive HTML saved to: {output_file}")

    def create_comparison_image(
        self,
        meshes: List[Tuple[pv.DataSet, str]],
        output_file: Union[str, Path],
        resolution: Tuple[int, int] = (1920, 1080),
        layout: str = "horizontal",
        **kwargs
    ) -> None:
        """
        Create side-by-side comparison image.

        Args:
            meshes: List of (mesh, title) tuples
            output_file: Output file path
            resolution: Image resolution
            layout: Layout type ('horizontal', 'vertical', 'grid')
            **kwargs: Additional rendering options
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        n_meshes = len(meshes)

        # Determine subplot layout
        if layout == "horizontal":
            shape = (1, n_meshes)
        elif layout == "vertical":
            shape = (n_meshes, 1)
        elif layout == "grid":
            n_cols = int(np.ceil(np.sqrt(n_meshes)))
            n_rows = int(np.ceil(n_meshes / n_cols))
            shape = (n_rows, n_cols)
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Create plotter with subplots
        plotter = pv.Plotter(
            shape=shape,
            window_size=resolution,
            off_screen=True
        )

        plotter.set_background(kwargs.get('background', 'white'))

        # Add each mesh to subplot
        for idx, (mesh, title) in enumerate(meshes):
            if layout == "horizontal":
                plotter.subplot(0, idx)
            elif layout == "vertical":
                plotter.subplot(idx, 0)
            else:  # grid
                row = idx // shape[1]
                col = idx % shape[1]
                plotter.subplot(row, col)

            plotter.add_text(title, position='upper_edge', font_size=14)

            mesh_kwargs = {
                'show_edges': kwargs.get('show_edges', False),
                'smooth_shading': True,
            }

            if 'scalars' in kwargs and kwargs['scalars'] in mesh.array_names:
                mesh_kwargs['scalars'] = kwargs['scalars']
                mesh_kwargs['cmap'] = kwargs.get('cmap', 'jet')
            else:
                mesh_kwargs['color'] = kwargs.get('color', 'lightblue')

            plotter.add_mesh(mesh, **mesh_kwargs)
            plotter.camera_position = kwargs.get('camera_position', 'iso')
            plotter.reset_camera()

        # Link views if requested
        if kwargs.get('link_views', True) and n_meshes > 1:
            plotter.link_views()

        # Render and save
        plotter.screenshot(str(output_file))
        plotter.close()

        print(f"Comparison image saved to: {output_file}")
