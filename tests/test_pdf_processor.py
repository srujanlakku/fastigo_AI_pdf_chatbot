import io

import pytest
from src.pdf_processor import extract_pdf_text


def test_extract_pdf_text_valid_pdf(tmp_path):
    from reportlab.pdfgen import canvas

    file_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(file_path))
    c.drawString(100, 750, "Hello world")
    c.save()

    with open(file_path, "rb") as f:
        result = extract_pdf_text(f)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["file_name"] == "test.pdf"
    assert result[0]["page_number"] == 1
    assert "Hello world" in result[0]["text"]


def test_extract_pdf_text_corrupted_pdf():
    file_obj = io.BytesIO(b"not-a-pdf")
    with pytest.raises(ValueError, match="corrupted or not a valid PDF"):
        extract_pdf_text(file_obj)
