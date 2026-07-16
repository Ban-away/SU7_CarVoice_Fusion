"""PDF parser interface with mock fallback.

Provides a uniform ``parse`` method that returns extracted text and
metadata from a PDF file.  When the real parsing library is unavailable,
a mock implementation is used that returns the filename as a placeholder.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import fitz  # PyMuPDF

    _HAS_PYMUPDF = True
except ImportError:  # pragma: no cover
    _HAS_PYMUPDF = False

try:
    from pdfplumber import PDF as PDFPlumber

    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False


# ---------------------------------------------------------------------------
# PDFParser
# ---------------------------------------------------------------------------

class PDFParser:
    """Parse PDF files into text pages with metadata.

    Tries PyMuPDF (fitz) first, then pdfplumber, and falls back to a
    mock that returns a placeholder.

    Parameters:
        use_ocr: When True, attempt OCR on image-based pages
            (PyMuPDF only; pdfplumber ignores this flag).
    """

    def __init__(self, use_ocr: bool = False) -> None:
        self._use_ocr = use_ocr

        if _HAS_PYMUPDF:
            logger.info("PDFParser using PyMuPDF (fitz)")
        elif _HAS_PDFPLUMBER:
            logger.info("PDFParser using pdfplumber")
        else:
            logger.info(
                "Neither PyMuPDF nor pdfplumber installed; "
                "PDFParser running in mock mode"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Parse a PDF file into a list of page dicts.

        Each dict contains:
        - ``page``: page number (1-based)
        - ``text``: extracted text content
        - ``images``: list of image info dicts (may be empty)

        Args:
            file_path: Path to the PDF file.

        Returns:
            A list of page dicts, one per page.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if _HAS_PYMUPDF:
            return self._parse_pymupdf(path)
        elif _HAS_PDFPLUMBER:
            return self._parse_pdfplumber(path)
        else:
            return self._parse_mock(path)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _parse_pymupdf(self, path: Path) -> list[dict[str, Any]]:
        """Parse using PyMuPDF."""
        doc = fitz.open(str(path))
        pages: list[dict[str, Any]] = []
        try:
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                images: list[dict] = []
                # Collect embedded image metadata
                for img_index, img in enumerate(page.get_image_info()):
                    images.append(
                        {
                            "index": img_index,
                            "bbox": list(img.get("bbox", [])),
                            "size": img.get("size", 0),
                        }
                    )
                pages.append(
                    {
                        "page": page_num,
                        "text": text.strip(),
                        "images": images,
                    }
                )
        finally:
            doc.close()

        logger.info("Parsed %s: %d pages (PyMuPDF)", path.name, len(pages))
        return pages

    def _parse_pdfplumber(self, path: Path) -> list[dict[str, Any]]:
        """Parse using pdfplumber."""
        pages: list[dict[str, Any]] = []
        with PDFPlumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                pages.append(
                    {
                        "page": page_num,
                        "text": text.strip(),
                        "images": [],
                    }
                )

        logger.info("Parsed %s: %d pages (pdfplumber)", path.name, len(pages))
        return pages

    def _parse_mock(self, path: Path) -> list[dict[str, Any]]:
        """Mock parser: returns filename as placeholder text."""
        logger.warning(
            "Mock PDF parse for %s — install PyMuPDF or pdfplumber for real parsing",
            path.name,
        )
        return [
            {
                "page": 1,
                "text": f"[Mock PDF content for: {path.name}]",
                "images": [],
            }
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def available_backends() -> list[str]:
        """Return the list of currently available parsing backends."""
        backends: list[str] = []
        if _HAS_PYMUPDF:
            backends.append("pymupdf")
        if _HAS_PDFPLUMBER:
            backends.append("pdfplumber")
        if not backends:
            backends.append("mock")
        return backends
