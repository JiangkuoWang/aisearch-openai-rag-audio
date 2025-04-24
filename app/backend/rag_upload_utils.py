import logging
from typing import List
import io
from PyPDF2 import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

def extract_text(filename: str, raw: bytes) -> str:
    """
    Extracts plain text from a file based on its filename extension.
    Requires additional libraries like PyPDF2, python-docx.
    """
    logger.info(f"Extracting text from {filename}...")
    # Placeholder: Implement actual extraction based on file type
    # Example for plain text:
    if filename.lower().endswith(".txt"):
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            logger.warning(f"Could not decode {filename} as UTF-8, trying latin-1")
            return raw.decode('latin-1') # Fallback
    elif filename.lower().endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n".join(pages)
        except Exception as e:
            logger.error(f"Error extracting PDF text from {filename}: {e}")
            return ""
    elif filename.lower().endswith((".docx")):
        try:
            doc = Document(io.BytesIO(raw))
            paras = [p.text for p in doc.paragraphs if p.text]
            return "\n".join(paras)
        except Exception as e:
            logger.error(f"Error extracting DOCX text from {filename}: {e}")
            return ""
    else:
        logger.warning(f"Unsupported file type for extraction: {filename}")
        return ""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a specified size with overlap.
    """
    logger.info(f"Chunking text (length {len(text)})...")
    if not text:
        return []
    # Placeholder: Implement a simple chunking strategy
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap # Move start pointer with overlap
        if start >= len(text): # Avoid infinite loop if overlap >= chunk_size
            break
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks
