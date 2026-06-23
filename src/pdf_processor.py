import fitz
from typing import Any, Dict, List

from .config import PDF_OCR_ENABLED
from .logger import logger
from .ocr_processor import run_ocr_for_page
from .utils import safe_filename


def extract_pdf_text(file: Any) -> List[Dict[str, Any]]:
    documents = []
    file_name = safe_filename(getattr(file, "name", "uploaded.pdf"))
    try:
        file_bytes = file.getvalue() if hasattr(file, "getvalue") else file.read()
        if not file_bytes:
            raise ValueError("The uploaded PDF file is empty.")
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.page_count == 0:
                raise ValueError("The uploaded PDF contains no pages.")
            logger.info("Processing PDF: %s with %s pages", file_name, pdf.page_count)
            for page_idx in range(pdf.page_count):
                page = pdf.load_page(page_idx)
                raw_text = page.get_text("text") or ""
                if not raw_text.strip() and PDF_OCR_ENABLED:
                    raw_text = run_ocr_for_page(page)

                documents.append(
                    {
                        "file_name": file_name,
                        "page_number": page_idx + 1,
                        "text": raw_text.strip(),
                        "metadata": {
                            "file_name": file_name,
                            "page_number": page_idx + 1,
                        },
                    }
                )
    except fitz.FileDataError as exc:
        logger.error("Corrupted PDF file: %s", exc)
        raise ValueError("The uploaded PDF file is corrupted or not a valid PDF.") from exc
    except ValueError:
        raise
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        raise ValueError("Unable to process the uploaded PDF. Please check that it is a valid PDF file.") from exc

    if not any(doc.get("text") for doc in documents):
        ocr_hint = " Enable OCR or upload a text-based PDF." if not PDF_OCR_ENABLED else ""
        raise ValueError(f"No extractable text was found in {file_name}.{ocr_hint}")
    return documents
