# pdf_highlight.py
# Soft-blue full-sentence highlights + compact hover popups ("Concept • Trend").

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import fitz  # PyMuPDF

_SOFT_BLUE_STROKE = (0.10, 0.40, 0.95)
_SOFT_BLUE_FILL   = (0.85, 0.92, 1.00)

def _add_hover_popup(page: fitz.Page, rect: fitz.Rect, label: str) -> None:
    # Attach a small popup to the highlight rect
    popup = page.add_popup_annot(rect, label)
    popup.set_flags(0)  # no huge sticky note
    popup.update()

def _add_highlight(page: fitz.Page, rect: fitz.Rect) -> fitz.Annot:
    hl = page.add_highlight_annot(rect)
    hl.set_colors(stroke=_SOFT_BLUE_STROKE, fill=_SOFT_BLUE_FILL)
    hl.update()
    return hl

def _rect_from_span(span: Dict) -> fitz.Rect:
    # Accept either 'rect' or 'bbox' key, or x0,y0,x1,y1
    if "rect" in span and isinstance(span["rect"], (list, tuple)):
        x0, y0, x1, y1 = span["rect"]
    elif "bbox" in span and isinstance(span["bbox"], (list, tuple)):
        x0, y0, x1, y1 = span["bbox"]
    else:
        x0, y0, x1, y1 = span["x0"], span["y0"], span["x1"], span["y1"]
    # Slightly expand for nicer rounded ends
    return fitz.Rect(x0 - 0.8, y0 - 0.6, x1 + 0.8, y1 + 0.6)

def highlight_pdf(input_pdf: str, hits: List[Dict], out_pdf: Optional[str] = None) -> str:
    """
    Parameters
    ----------
    input_pdf : path to source PDF
    hits : list of dicts with keys:
        - page (int, 0-based)
        - rect / bbox / (x0,y0,x1,y1)
        - concept (str)
        - trend (str | None)
        - sentence (str)  # optional, not shown in tooltip
    out_pdf : optional output path; if None, write next to input

    Returns
    -------
    str : path to written PDF
    """
    doc = fitz.open(input_pdf)
    for h in hits:
        try:
            pg = int(h.get("page", 0))
            if not (0 <= pg < len(doc)):
                continue
            page = doc[pg]
            rect = _rect_from_span(h)
            concept = (h.get("concept") or "").strip()
            trend = (h.get("trend") or "").strip()
            label = concept + (f" • {trend}" if trend else "")

            hl = _add_highlight(page, rect)
            # Attach a compact popup (hover reveals)
            _add_hover_popup(page, hl.rect, label or "Metric")
        except Exception:
            # Don't crash on a single bad span
            continue

    out_path = out_pdf or input_pdf.replace(".pdf", ".highlighted.pdf")
    doc.save(out_path, deflate=True)
    doc.close()
    return out_path
