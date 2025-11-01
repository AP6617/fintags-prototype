import os
from pathlib import Path
from pypdf import PdfReader

# folders (adjust if you changed structure)
ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw_docs"
OUT_DIR = ROOT / "data" / "texts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text(pdf_path: Path, out_path: Path):
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        parts.append(t + f"\n--- PAGE {i} END ---\n")
    out_path.write_text("".join(parts), encoding="utf-8")
    return len(reader.pages)

def main():
    pdfs = sorted([p for p in PDF_DIR.glob("*.pdf")])
    if not pdfs:
        print(f" No PDFs found in: {PDF_DIR}")
        return

    print(f"Found {len(pdfs)} PDF(s) in {PDF_DIR}")
    ok, fail = 0, 0
    for pdf in pdfs:
        out = OUT_DIR / (pdf.stem + ".txt")
        try:
            pages = extract_text(pdf, out)
            print(f" {pdf.name}  →  {out.name}  ({pages} pages)")
            ok += 1
        except Exception as e:
            print(f" {pdf.name}  →  ERROR: {e}")
            fail += 1

    print("-" * 60)
    print(f"Done. Converted: {ok}, Failed: {fail}. Output folder: {OUT_DIR}")

if __name__ == "__main__":
    main()
