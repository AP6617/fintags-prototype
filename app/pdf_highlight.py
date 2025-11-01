import fitz  # PyMuPDF
import hashlib
from typing import List, Dict

def _hash_annot(page_no: int, rect, text: str) -> str:
    key = f"{page_no}:{text}:{round(rect.x0,2)}-{round(rect.y0,2)}-{round(rect.x1,2)}-{round(rect.y1,2)}"
    return hashlib.md5(key.encode()).hexdigest()

def _find_sentence_rects(page: fitz.Page, sentence: str):
    rects = []
    quads = page.search_for(sentence, quads=True)
    if not quads:
        norm = " ".join(sentence.split())
        quads = page.search_for(norm, quads=True)
    if not quads:
        quads = page.search_for(sentence, quads=True, flags=fitz.TEXT_DEHYPHENATE | fitz.TEXT_PRESERVE_LIGATURES)
    for q in quads:
        rects.append(fitz.Rect(q.rect))
    return rects

def _add_note(page: fitz.Page, rect: fitz.Rect, label: str):
    box_w = min(200, max(60, 8 * len(label)))
    box_h = 14
    x0 = rect.x0
    y0 = max(0, rect.y0 - box_h - 4)
    box = fitz.Rect(x0, y0, x0 + box_w, y0 + box_h)
    annot = page.add_freetext_annot(
        box,
        label,
        fontsize=8,
        fontname="helv",
        text_color=(1,1,1),
        fill_color=(0,0,0),
        align=fitz.TEXT_ALIGN_LEFT,
        rotate=0
    )
    annot.set_border(width=0)
    return annot

def highlight_sentences_with_notes(in_pdf: str, out_pdf: str, sentences: List[str], notes: List[str]):
    doc = fitz.open(in_pdf)
    dedup = set()
    meta = []
    for page_no in range(len(doc)):
        page = doc[page_no]
        for i, sent in enumerate(sentences):
            rects = _find_sentence_rects(page, sent)
            for rect in rects:
                key = _hash_annot(page_no, rect, sent)
                if key in dedup:
                    continue
                dedup.add(key)
                hl = page.add_highlight_annot(rect)
                hl.set_colors(stroke=(0,0,1))
                hl.update()
                label = notes[i] if i < len(notes) else ""
                _add_note(page, rect, label)
                meta.append({"page": page_no+1, "sentence": sent, "note": label})
    doc.save(out_pdf, incremental=False, encryption=fitz.PDF_ENCRYPT_KEEP)
    doc.close()
    return meta
