"""
Plotly-based 2D charting utilities for CAD/CFD analysis.

Provides professional 2D visualizations for convergence analysis, residuals,
statistics, and other analytical data using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple


class PlotlyCharts:
    """
    2D visualization utilities using Plotly for CAD/CFD analysis data.

    Handles convergence plots, residual tracking, statistical visualizations,
    and custom analytical charts.
    """

    def __init__(self, theme: str = "plotly_white"):
        """
        Initialize Plotly charts with default theme.

        Args:
            theme: Plotly template ('plotly', 'plotly_white', 'plotly_dark', 'seaborn', 'simple_white')
        """
        self.theme = theme

    def plot_residuals(
        self,
        iteration_data: Union[pd.DataFrame, Dict[str, List]],
        log_scale: bool = True,
        title: str = "CFD Residuals Convergence"
    ) -> go.Figure:
        """
        Create line chart of CFD residuals over iterations.

        Args:
            iteration_data: DataFrame or dict with iteration data
                Format: {
                    'iteration': [1, 2, 3, ...],
                    'continuity': [1e-3, 5e-4, 1e-4, ...],
                    'x-velocity': [...],
                    'y-velocity': [...],
                    'energy': [...],
                }
            log_scale: Use logarithmic y-axis
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        # Convert to DataFrame if dict
        if isinstance(iteration_data, dict):
            df = pd.DataFrame(iteration_data)
        else:
            df = iteration_data

        # Create figure
        fig = go.Figure()

        # Identify residual columns (all except 'iteration')
        residual_cols = [col for col in df.columns if col.lower() != 'iteration']

        # Add trace for each residual
        for col in residual_cols:
            fig.add_trace(go.Scatter(
                x=df['iteration'] if 'iteration' in df.columns else df.index,
                y=df[col],
                mode='lines+markers',
                name=col.replace('_', ' ').title(),
                line=dict(width=2),
                marker=dict(size=4)
            ))

        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Residual",
            yaxis_type="log" if log_scale else "linear",
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1
            ),
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        return fig

    def plot_convergence(
        self,
        history: Union[pd.DataFrame, Dict[str, List]],
        metrics: Optional[List[str]] = None,
        title: str = "Convergence History"
    ) -> go.Figure:
        """
        Create convergence plot for multiple metrics.

        Args:
            history: DataFrame or dict with convergence history
                Format: {
                    'iteration': [1, 2, 3, ...],
                    'metric1': [...],
                    'metric2': [...],
                }
            metrics: List of metric names to plot (None = all)
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        # Convert to DataFrame if dict
        if isinstance(history, dict):
            df = pd.DataFrame(history)
        else:
            df = history

        # Determine metrics to plot
        if metrics is None:
            metrics = [col for col in df.columns if col.lower() != 'iteration']

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        for idx, metric in enumerate(metrics):
            if metric not in df.columns:
                continue

            use_secondary = idx >= len(metrics) // 2

            fig.add_trace(
                go.Scatter(
                    x=df['iteration'] if 'iteration' in df.columns else df.index,
                    y=df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(width=2),
                    marker=dict(size=4)
                ),
                secondary_y=use_secondary
            )

        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            hovermode='x unified',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )

        fig.update_xaxes(title_text="Iteration", showgrid=True)
        fig.update_yaxes(title_text="Metric Value", showgrid=True, secondary_y=False)
        fig.update_yaxes(title_text="Secondary Metrics", showgrid=True, secondary_y=True)

        return fig

    def plot_statistics(
        self,
        data_dict: Dict[str, Union[float, int]],
        chart_type: str = "bar",
        title: str = "Statistics Overview"
    ) -> go.Figure:
        """
        Create statistical visualizations (bar, pie, etc.).

        Args:
            data_dict: Dictionary of statistics
                Format: {'metric1': value1, 'metric2': value2, ...}
            chart_type: Type of chart ('bar', 'pie', 'horizontal_bar')
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        labels = list(data_dict.keys())
        values = list(data_dict.values())

        if chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=values,
                    text=values,
                    texttemplate='%{text:.2f}',
                    textposition='auto',
                    marker=dict(
                        color=values,
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            ])

            fig.update_layout(
                title=title,
                xaxis_title="Metric",
                yaxis_title="Value",
                template=self.theme,
                height=500
            )

        elif chart_type == "horizontal_bar":
            fig = go.Figure(data=[
                go.Bar(
                    y=labels,
                    x=values,
                    text=values,
                    texttemplate='%{text:.2f}',
                    textposition='auto',
                    orientation='h',
                    marker=dict(
                        color=values,
                        colorscale='Viridis',
                        showscale=True
                    )
                )
            ])

            fig.update_layout(
                title=title,
                xaxis_title="Value",
                yaxis_title="Metric",
                template=self.theme,
                height=max(400, len(labels) * 40)
            )

        elif chart_type == "pie":
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo='label+percent',
                    hovertemplate='%{label}<br>%{value:.2f}<br>%{percent}',
                    marker=dict(
                        line=dict(color='white', width=2)
                    )
                )
            ])

            fig.update_layout(
                title=title,
                template=self.theme,
                height=500
            )

        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

        return fig

    def plot_shadow_heatmap(
        self,
        shadow_hours: Union[np.ndarray, List[List]],
        x_labels: Optional[List] = None,
        y_labels: Optional[List] = None,
        title: str = "Shadow Hours Heatmap"
    ) -> go.Figure:
        """
        Create heatmap visualization for solar/shadow analysis.

        Args:
            shadow_hours: 2D array of shadow hour data
            x_labels: Labels for x-axis (e.g., months)
            y_labels: Labels for y-axis (e.g., hours of day)
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        # Convert to numpy array if needed
        if isinstance(shadow_hours, list):
            shadow_hours = np.array(shadow_hours)

        # Generate default labels if not provided
        if x_labels is None:
            x_labels = [f"Col {i}" for i in range(shadow_hours.shape[1])]
        if y_labels is None:
            y_labels = [f"Row {i}" for i in range(shadow_hours.shape[0])]

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=shadow_hours,
            x=x_labels,
            y=y_labels,
            colorscale='YlOrRd',
            colorbar=dict(title="Hours"),
            hovertemplate='%{x}<br>%{y}<br>Hours: %{z:.1f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time Period",
            yaxis_title="Location",
            template=self.theme,
            height=500
        )

        return fig

    def plot_comparison(
        self,
        data: Dict[str, Union[pd.DataFrame, Dict]],
        x_col: str,
        y_col: str,
        title: str = "Comparison Plot"
    ) -> go.Figure:
        """
        Create comparison plot for multiple datasets.

        Args:
            data: Dictionary of datasets
                Format: {
                    'Dataset1': DataFrame or dict,
                    'Dataset2': DataFrame or dict,
                }
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()

        for name, dataset in data.items():
            # Convert to DataFrame if dict
            if isinstance(dataset, dict):
                df = pd.DataFrame(dataset)
            else:
                df = dataset

            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode='lines+markers',
                name=name,
                line=dict(width=2),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            template=self.theme,
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="right",
                x=1
            )
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        return fig

    def plot_3d_scatter(
        self,
        data: Union[pd.DataFrame, Dict],
        x: str,
        y: str,
        z: str,
        color: Optional[str] = None,
        size: Optional[str] = None,
        title: str = "3D Scatter Plot"
    ) -> go.Figure:
        """
        Create 3D scatter plot.

        Args:
            data: DataFrame or dict with data
            x: Column name for x-axis
            y: Column name for y-axis
            z: Column name for z-axis
            color: Column name for color coding (optional)
            size: Column name for marker size (optional)
            title: Chart title

        Returns:
            go.Figure: Plotly figure object
        """
        # Convert to DataFrame if dict
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data

        # Prepare marker configuration
        marker_config = {}
        if color and color in df.columns:
            marker_config['color'] = df[color]
            marker_config['colorscale'] = 'Viridis'
            marker_config['showscale'] = True
            marker_config['colorbar'] = dict(title=color.replace('_', ' ').title())

        if size and size in df.columns:
            marker_config['size'] = df[size]
            marker_config['sizemode'] = 'diameter'
            marker_config['sizeref'] = 2. * max(df[size]) / (40.**2)
        else:
            marker_config['size'] = 5

        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=df[x],
            y=df[y],
            z=df[z],
            mode='markers',
            marker=marker_config,
            text=df.index if isinstance(df.index, pd.Index) else None,
            hovertemplate=f'<b>{x}</b>: %{{x:.2f}}<br>'
                         f'<b>{y}</b>: %{{y:.2f}}<br>'
                         f'<b>{z}</b>: %{{z:.2f}}<br>'
                         '<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=x.replace('_', ' ').title(),
                yaxis_title=y.replace('_', ' ').title(),
                zaxis_title=z.replace('_', ' ').title(),
            ),
            template=self.theme,
            height=600
        )

        return fig

    def plot_contour(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        title: str = "Contour Plot",
        n_contours: int = 20
    ) -> go.Figure:
        """
        Create 2D contour plot.

        Args:
            x: X coordinates (1D or 2D array)
            y: Y coordinates (1D or 2D array)
            z: Z values (2D array)
            title: Chart title
            n_contours: Number of contour levels

        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure(data=go.Contour(
            x=x if x.ndim == 1 else x[0],
            y=y if y.ndim == 1 else y[:, 0],
            z=z,
            colorscale='Jet',
            contours=dict(
                start=z.min(),
                end=z.max(),
                size=(z.max() - z.min()) / n_contours,
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            colorbar=dict(title="Value"),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            template=self.theme,
            height=500,
            width=600
        )

        return fig

    def create_dashboard(
        self,
        figures: List[Tuple[go.Figure, str]],
        layout: str = "grid"
    ) -> go.Figure:
        """
        Combine multiple figures into a dashboard.

        Args:
            figures: List of (figure, title) tuples
            layout: Layout type ('grid', 'vertical', 'horizontal')

        Returns:
            go.Figure: Combined dashboard figure
        """
        n_figs = len(figures)

        if layout == "grid":
            n_cols = 2
            n_rows = (n_figs + 1) // 2
        elif layout == "vertical":
            n_rows = n_figs
            n_cols = 1
        elif layout == "horizontal":
            n_rows = 1
            n_cols = n_figs
        else:
            raise ValueError(f"Unknown layout: {layout}")

        # Create subplots
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[title for _, title in figures]
        )

        # Add each figure to subplot
        for idx, (source_fig, title) in enumerate(figures):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1

            # Add all traces from source figure
            for trace in source_fig.data:
                fig.add_trace(trace, row=row, col=col)

        fig.update_layout(
            template=self.theme,
            height=400 * n_rows,
            showlegend=True
        )

        return fig
