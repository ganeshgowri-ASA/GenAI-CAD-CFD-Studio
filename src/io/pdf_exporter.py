"""
PDF Export for CAD Models

Exports generated CAD models and technical drawings to PDF format
with proper formatting, dimensions, and metadata.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import io

# Check for PDF generation libraries
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logging.warning("reportlab not installed. PDF export disabled.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)


class PDFExporter:
    """
    Export CAD models and technical drawings to PDF.

    Features:
    - Model preview images
    - Technical specifications
    - Dimension tables
    - Metadata and generation info
    - Multi-view layouts
    """

    def __init__(self, page_size=A4):
        """
        Initialize PDF exporter.

        Args:
            page_size: PDF page size (default: A4)
        """
        if not HAS_REPORTLAB:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")

        self.page_size = page_size
        self.styles = getSampleStyleSheet()

        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30
        ))

        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12
        ))

    def export_model(
        self,
        output_path: Path,
        model_data: Dict[str, Any],
        preview_images: Optional[List[Path]] = None,
        title: str = "CAD Model Export",
        include_metadata: bool = True
    ) -> Path:
        """
        Export CAD model to PDF.

        Args:
            output_path: Output PDF file path
            model_data: Model data including parameters, specifications, etc.
            preview_images: List of preview image paths
            title: Document title
            include_metadata: Include generation metadata

        Returns:
            Path to generated PDF

        Example:
            >>> exporter = PDFExporter()
            >>> exporter.export_model(
            ...     Path("output.pdf"),
            ...     model_data={'type': 'box', 'length': 100},
            ...     preview_images=[Path("preview.png")]
            ... )
        """
        logger.info(f"Exporting model to PDF: {output_path}")

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )

        # Build content
        story = []

        # Title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # Generation timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"<i>Generated: {timestamp}</i>", self.styles['Normal']))
        story.append(Spacer(1, 24))

        # Preview images
        if preview_images:
            story.append(Paragraph("Model Previews", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            for img_path in preview_images:
                if img_path.exists():
                    try:
                        # Calculate image size to fit page
                        img_width = 6 * inch
                        img = RLImage(str(img_path), width=img_width, height=None, kind='proportional')
                        story.append(img)
                        story.append(Spacer(1, 12))
                    except Exception as e:
                        logger.warning(f"Failed to add image {img_path}: {e}")

            story.append(Spacer(1, 24))

        # Technical Specifications
        story.append(Paragraph("Technical Specifications", self.styles['CustomHeading']))
        story.append(Spacer(1, 12))

        specs_data = self._build_specifications_table(model_data)
        if specs_data:
            table = Table(specs_data, colWidths=[2.5*inch, 3.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
            ]))
            story.append(table)
            story.append(Spacer(1, 24))

        # Metadata
        if include_metadata and 'metadata' in model_data:
            story.append(Paragraph("Generation Metadata", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            metadata_data = self._build_metadata_table(model_data['metadata'])
            if metadata_data:
                table = Table(metadata_data, colWidths=[2.5*inch, 3.5*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 10),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
                ]))
                story.append(table)

        # Build PDF
        doc.build(story)

        logger.info(f"PDF exported successfully: {output_path}")
        return output_path

    def _build_specifications_table(self, model_data: Dict[str, Any]) -> List[List[str]]:
        """Build specifications table data."""
        data = [['Parameter', 'Value']]

        # Extract key parameters
        params = model_data.get('parameters', {})

        # Common parameters
        param_display_names = {
            'type': 'Object Type',
            'object_type': 'Object Type',
            'length': 'Length (mm)',
            'width': 'Width (mm)',
            'height': 'Height (mm)',
            'radius': 'Radius (mm)',
            'diameter': 'Diameter (mm)',
            'thickness': 'Thickness (mm)',
            'material': 'Material',
            'units': 'Units',
            'engine': 'Generation Engine',
        }

        for key, display_name in param_display_names.items():
            if key in params and params[key] is not None:
                value = params[key]
                # Format value
                if isinstance(value, float):
                    value = f"{value:.2f}"
                data.append([display_name, str(value)])

        # Add any other parameters
        for key, value in params.items():
            if key not in param_display_names and value is not None:
                display_name = key.replace('_', ' ').title()
                if isinstance(value, float):
                    value = f"{value:.2f}"
                elif isinstance(value, (list, dict)):
                    continue  # Skip complex structures
                data.append([display_name, str(value)])

        return data if len(data) > 1 else []

    def _build_metadata_table(self, metadata: Dict[str, Any]) -> List[List[str]]:
        """Build metadata table data."""
        data = [['Metadata', 'Value']]

        # Format metadata
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                display_name = key.replace('_', ' ').title()
                if isinstance(value, float):
                    value = f"{value:.4f}"
                data.append([display_name, str(value)])

        return data if len(data) > 1 else []

    def export_technical_drawing(
        self,
        output_path: Path,
        drawing_data: Dict[str, Any],
        views: Optional[List[Tuple[str, Path]]] = None,
        title: str = "Technical Drawing"
    ) -> Path:
        """
        Export technical drawing with multiple views to PDF.

        Args:
            output_path: Output PDF file path
            drawing_data: Drawing specifications and dimensions
            views: List of (view_name, image_path) tuples
            title: Document title

        Returns:
            Path to generated PDF
        """
        logger.info(f"Exporting technical drawing to PDF: {output_path}")

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=self.page_size,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=18
        )

        story = []

        # Title
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # Drawing number and date
        drawing_number = drawing_data.get('drawing_number', 'DWG-001')
        revision = drawing_data.get('revision', 'A')
        story.append(Paragraph(f"Drawing Number: {drawing_number} Rev. {revision}", self.styles['Normal']))
        story.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal']))
        story.append(Spacer(1, 24))

        # Multiple views
        if views:
            for view_name, view_path in views:
                if view_path.exists():
                    story.append(Paragraph(f"{view_name} View", self.styles['CustomHeading']))
                    story.append(Spacer(1, 12))

                    try:
                        img = RLImage(str(view_path), width=6.5*inch, height=None, kind='proportional')
                        story.append(img)
                        story.append(Spacer(1, 24))
                    except Exception as e:
                        logger.warning(f"Failed to add view {view_name}: {e}")

        # Dimensions table
        if 'dimensions' in drawing_data:
            story.append(Paragraph("Dimensions", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))

            dim_data = [['Feature', 'Dimension', 'Tolerance', 'Notes']]
            for dim in drawing_data['dimensions']:
                dim_data.append([
                    dim.get('feature', ''),
                    dim.get('value', ''),
                    dim.get('tolerance', '±0.1'),
                    dim.get('notes', '')
                ])

            table = Table(dim_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.lightgrey])
            ]))
            story.append(table)

        doc.build(story)

        logger.info(f"Technical drawing PDF exported: {output_path}")
        return output_path


def create_pdf_from_model(
    model_result,
    output_path: Path,
    preview_images: Optional[List[Path]] = None
) -> Optional[Path]:
    """
    Convenience function to create PDF from model generation result.

    Args:
        model_result: CADGenerationResult object
        output_path: Output PDF path
        preview_images: List of preview image paths

    Returns:
        Path to generated PDF or None if export failed
    """
    if not HAS_REPORTLAB:
        logger.error("reportlab not available for PDF export")
        return None

    try:
        exporter = PDFExporter()

        model_data = {
            'parameters': model_result.parameters,
            'metadata': model_result.metadata,
            'message': model_result.message
        }

        return exporter.export_model(
            output_path,
            model_data,
            preview_images=preview_images,
            title="CAD Model - Technical Documentation"
        )

    except Exception as e:
        logger.error(f"PDF export failed: {e}", exc_info=True)
        return None
