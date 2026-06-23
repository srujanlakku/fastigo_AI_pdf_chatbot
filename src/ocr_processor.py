import os
import tempfile
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image

from .config import PDF_OCR_ENABLED
from .logger import logger


def run_ocr_for_page(page) -> str:
    if not PDF_OCR_ENABLED:
        logger.info("OCR disabled by configuration; skipping page.")
        return ""

    temp_path: Optional[Path] = None
    try:
        pix = page.get_pixmap(dpi=300)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(pix.tobytes(output="png"))
            temp_file.flush()
        with Image.open(temp_path) as image:
            text = pytesseract.image_to_string(image)
            return text or ""
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract OCR is not installed or not available on PATH.")
        return ""
    except Exception as exc:
        logger.error("OCR processing failed: %s", exc)
        return ""
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove OCR temp file %s: %s", temp_path, exc)
