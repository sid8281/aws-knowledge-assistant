"""Routes a file to the correct loader based on its extension."""

import os
from app.ingestion.loaders.pdf_loader import load_pdf
from app.ingestion.loaders.html_loader import load_html
from app.ingestion.loaders.txt_loader import load_txt, load_csv_rows
from app.ingestion.loaders.docx_loader import load_docx
from app.ingestion.loaders.pptx_loader import load_pptx


def load_document(file_path: str) -> str:
    """Return the raw text of any supported document type."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".html", ".htm"):
        return load_html(file_path)
    elif ext == ".txt":
        return load_txt(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    elif ext == ".pptx":
        return load_pptx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
