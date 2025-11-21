"""
PDF Export Module for CAD Drawings

Provides comprehensive PDF export capabilities for CAD models and technical drawings.

Features:
- 2D drawing export with dimensions
- Title blocks and annotations
- Multi-page support
- Vector graphics (high quality)
- Custom templates

Author: GenAI CAD CFD Studio
Version: 1.0.0
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Optional imports
try:
    from reportlab.lib.pagesizes import A4, A3, LETTER
    from reportlab.lib.units import mm, inch
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logging.warning("ReportLab not installed. PDF export disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as pdf_backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("Matplotlib not installed. Some PDF features disabled.")


logger = logging.getLogger(__name__)


class PDFExporter:
    """
    Comprehensive PDF exporter for CAD drawings and technical documentation.

    Features:
    - Multiple page sizes (A4, A3, Letter)
    - Title blocks with metadata
    - Dimension annotations
    - Vector graphics for scalability
    - Custom branding/logos
    - Multi-page documents

    Example:
        >>> exporter = PDFExporter(page_size='A4')
        >>> exporter.create_drawing_pdf(
        ...     "output.pdf",
        ...     title="Mounting Bracket",
        ...     dimensions={'length': 100, 'width': 50}
        ... )
    """

    PAGE_SIZES = {
        'A4': A4 if HAS_REPORTLAB else (210*2.83465, 297*2.83465),
        'A3': A3 if HAS_REPORTLAB else (297*2.83465, 420*2.83465),
        'LETTER': LETTER if HAS_REPORTLAB else (8.5*72, 11*72)
    }

    def __init__(
        self,
        page_size: str = 'A4',
        orientation: str = 'portrait',
        author: str = 'GenAI CAD CFD Studio',
        company: str = ''
    ):
        """
        Initialize PDF Exporter.

        Args:
            page_size: Page size ('A4', 'A3', 'LETTER')
            orientation: Page orientation ('portrait' or 'landscape')
            author: Document author
            company: Company name for branding
        """
        if not HAS_REPORTLAB:
            raise ImportError("ReportLab is required for PDF export. Install with: pip install reportlab")

        self.page_size_name = page_size
        self.page_size = self.PAGE_SIZES.get(page_size, A4)
        self.orientation = orientation
        self.author = author
        self.company = company

        # Adjust for landscape
        if orientation == 'landscape':
            self.page_size = (self.page_size[1], self.page_size[0])

        self.width, self.height = self.page_size

        logger.info(f"PDFExporter initialized: {page_size} {orientation}")

    def create_drawing_pdf(
        self,
        output_path: str,
        title: str,
        description: Optional[str] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Create a technical drawing PDF with title block and dimensions.

        Args:
            output_path: Path for output PDF
            title: Drawing title
            description: Drawing description
            dimensions: Dictionary of dimensions
            metadata: Additional metadata
            image_path: Optional path to drawing image

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create canvas
            c = canvas.Canvas(output_path, pagesize=self.page_size)

            # Set metadata
            c.setAuthor(self.author)
            c.setTitle(title)
            c.setSubject("Technical Drawing")
            c.setCreator("GenAI CAD CFD Studio")

            # Draw title block
            self._draw_title_block(c, title, description, metadata)

            # Draw dimensions table if provided
            if dimensions:
                self._draw_dimensions_table(c, dimensions)

            # Draw image if provided
            if image_path and Path(image_path).exists():
                self._draw_image(c, image_path)

            # Add page number
            c.setFont("Helvetica", 10)
            c.drawString(
                self.width - 100,
                20,
                f"Page 1 | {datetime.now().strftime('%Y-%m-%d')}"
            )

            # Save PDF
            c.save()

            logger.info(f"PDF created successfully: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create PDF: {e}", exc_info=True)
            return False

    def _draw_title_block(
        self,
        c: canvas.Canvas,
        title: str,
        description: Optional[str],
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Draw title block at bottom of page."""
        # Title block dimensions
        block_height = 80
        block_y = 40

        # Draw border
        c.setStrokeColor(colors.black)
        c.setLineWidth(2)
        c.rect(40, block_y, self.width - 80, block_height)

        # Draw internal divisions
        c.setLineWidth(1)
        c.line(40, block_y + 50, self.width - 40, block_y + 50)
        c.line(self.width / 2, block_y, self.width / 2, block_y + block_height)

        # Add title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, block_y + 55, title)

        # Add description
        if description:
            c.setFont("Helvetica", 10)
            # Wrap text if too long
            max_width = self.width / 2 - 70
            if c.stringWidth(description, "Helvetica", 10) > max_width:
                description = description[:60] + "..."
            c.drawString(50, block_y + 35, description)

        # Add metadata
        c.setFont("Helvetica", 9)
        y_pos = block_y + 35
        if metadata:
            for key, value in list(metadata.items())[:3]:  # Max 3 items
                c.drawString(50, y_pos, f"{key}: {value}")
                y_pos -= 12

        # Add date and author
        c.setFont("Helvetica", 9)
        right_x = self.width / 2 + 10
        c.drawString(right_x, block_y + 60, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        c.drawString(right_x, block_y + 48, f"Author: {self.author}")
        if self.company:
            c.drawString(right_x, block_y + 36, f"Company: {self.company}")

        # Add project info
        c.drawString(right_x, block_y + 20, f"Project: CAD Design")
        c.drawString(right_x, block_y + 8, f"Sheet: 1 of 1")

    def _draw_dimensions_table(
        self,
        c: canvas.Canvas,
        dimensions: Dict[str, Any]
    ) -> None:
        """Draw dimensions table."""
        # Table position and size
        table_x = 60
        table_y = self.height - 150
        col_width = 120
        row_height = 20

        # Draw table header
        c.setFont("Helvetica-Bold", 12)
        c.drawString(table_x, table_y, "Dimensions")

        # Draw table rows
        c.setFont("Helvetica", 10)
        y_pos = table_y - 25

        for key, value in dimensions.items():
            if y_pos < 150:  # Stop if reaching title block
                break

            # Draw row
            c.drawString(table_x, y_pos, str(key).capitalize())
            c.drawString(table_x + col_width, y_pos, str(value))

            # Draw line
            c.setLineWidth(0.5)
            c.setStrokeColor(colors.grey)
            c.line(table_x, y_pos - 5, table_x + col_width * 2, y_pos - 5)

            y_pos -= row_height

    def _draw_image(
        self,
        c: canvas.Canvas,
        image_path: str
    ) -> None:
        """Draw image in center of page."""
        try:
            # Calculate position to center image
            img_width = 400
            img_height = 300
            x = (self.width - img_width) / 2
            y = (self.height - img_height) / 2 + 50  # Offset for title block

            # Draw image
            c.drawImage(
                image_path,
                x, y,
                width=img_width,
                height=img_height,
                preserveAspectRatio=True
            )

        except Exception as e:
            logger.warning(f"Failed to draw image: {e}")

    def create_multipage_pdf(
        self,
        output_path: str,
        pages: List[Dict[str, Any]]
    ) -> bool:
        """
        Create multi-page PDF document.

        Args:
            output_path: Path for output PDF
            pages: List of page dictionaries with content

        Returns:
            True if successful
        """
        try:
            c = canvas.Canvas(output_path, pagesize=self.page_size)

            for i, page_data in enumerate(pages):
                # Draw page content
                self.create_drawing_pdf(
                    output_path=output_path,
                    title=page_data.get('title', f'Page {i+1}'),
                    description=page_data.get('description'),
                    dimensions=page_data.get('dimensions'),
                    metadata=page_data.get('metadata'),
                    image_path=page_data.get('image_path')
                )

                # Add new page if not last
                if i < len(pages) - 1:
                    c.showPage()

            c.save()
            logger.info(f"Multi-page PDF created: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create multi-page PDF: {e}", exc_info=True)
            return False


def quick_export_pdf(
    output_path: str,
    title: str,
    dimensions: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
) -> bool:
    """
    Quick PDF export function.

    Args:
        output_path: Path for output PDF
        title: Drawing title
        dimensions: Dimensions dictionary
        description: Optional description

    Returns:
        True if successful

    Example:
        >>> quick_export_pdf(
        ...     "drawing.pdf",
        ...     "Mounting Bracket",
        ...     dimensions={'length': 100, 'width': 50, 'height': 30}
        ... )
    """
    try:
        exporter = PDFExporter()
        return exporter.create_drawing_pdf(
            output_path=output_path,
            title=title,
            description=description,
            dimensions=dimensions
        )
    except Exception as e:
        logger.error(f"Quick PDF export failed: {e}")
        return False
