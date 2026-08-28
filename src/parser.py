from typing import BinaryIO, Union

import pymupdf as fitz


def extract_text_from_pdf(pdf_file: Union[BinaryIO, bytes, bytearray]) -> str:
    """Extract text from every page of a PDF and return it as one string.

    Accepts a Streamlit UploadedFile, another file-like object, or raw bytes.
    Returns an empty string if the PDF has no extractable text.
    Raises ValueError if the file cannot be read as a PDF.
    """
    try:
        if isinstance(pdf_file, (bytes, bytearray)):
            pdf_bytes = bytes(pdf_file)
        else:
            # Streamlit may have already read the buffer; rewind first.
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            pdf_bytes = pdf_file.read()
    except Exception as exc:
        raise ValueError("Could not read the uploaded file.") from exc

    if not pdf_bytes:
        return ""

    try:
        # open(..., stream=...) reads the PDF from memory, not from a path.
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_texts = [page.get_text() for page in document]
    except Exception as exc:
        raise ValueError("Could not parse this PDF. Please try another file.") from exc

    return "\n".join(page_texts).strip()
