from pathlib import Path
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw_docs"
TEXT_DIR = ROOT / "data" / "texts"
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def ingest_pdf(doc_id: str, file_bytes: bytes) -> Path:
    """Save a PDF and extract its text to a .txt file."""
    pdf_path = RAW_DIR / f"{doc_id}.pdf"
    pdf_path.write_bytes(file_bytes)

    reader = PdfReader(str(pdf_path))
    text = ""
    for i, page in enumerate(reader.pages, start=1):
        text += (page.extract_text() or "") + f"\n--- PAGE {i} END ---\n"

    out_path = TEXT_DIR / f"{doc_id}.txt"
    out_path.write_text(text, encoding="utf-8")

    return out_path
