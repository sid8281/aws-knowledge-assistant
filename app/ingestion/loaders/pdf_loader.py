"""Extract raw text from a PDF file using pypdf."""

from pypdf import PdfReader


def load_pdf(file_path: str) -> str:
    """Read every page of a PDF and return the concatenated text."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)
