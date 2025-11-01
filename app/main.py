from __future__ import annotations
import re
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import torch
from sentence_transformers import CrossEncoder

# ==============================
# HIGHLIGHT STYLE (soft blue, hover-only tooltip)
# ==============================
HIGHLIGHT_FILL = (0.73, 0.86, 1.00)  # light blue
HIGHLIGHT_STROKE = None
HIGHLIGHT_OPACITY = 0.28

# ==============================
# PATHS & GLOBALS
# ==============================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_uploads"; DATA_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "outputs"; OUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = ROOT / "models"
FINBERT_PATH = MODELS_DIR / "finbert_crossenc"

MAX_PAGES = 400
MAX_SENTENCES = 6000
CE_BATCH_SIZE = 64

# ==============================
# APP + CROSS-ENCODER
# ==============================
app = FastAPI(title="FinTags – Backend (Dynamic Keywords)")

device = "cuda" if torch.cuda.is_available() else "cpu"
_xenc = CrossEncoder(
    str(FINBERT_PATH) if FINBERT_PATH.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=device
)

# ==============================
# VALUE / UNITS PARSING
# ==============================
UNIT_SCALE = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

MONEY_RE = re.compile(
    r"""(?P<cur>[$£€])?\s?
        (?P<min>\d[\d,]*\.?\d*)
        (?:\s?[–\-]\s?(?P<max>\d[\d,]*\.?\d*))?
        \s?(?P<unit>k|m|mn|b|bn|thousand|million|billion)?\b[.)]?
    """,
    re.I | re.VERBOSE,
)

PCT_RE = re.compile(
    r"""(?P<pmin>\d[\d,]*\.?\d*)
        (?:\s?[–\-]\s?(?P<pmax>\d[\d,]*\.?\d*))?
        \s?%[.)]?
    """,
    re.I | re.VERBOSE,
)

YEAR_ONLY  = re.compile(r"^\s*(?:FY|CY)?\s*(19|20)\d{2}(?:E)?\s*\.?\s*$", re.I)
YEAR_TOKEN = re.compile(r"\b(?:FY|CY)?(19|20)\d{2}(?:E)?\b", re.I)
PAGE_FOOT  = re.compile(r"^\s*(page\s+\d+(\s+of\s+\d+)?)\s*$", re.I)

# word-number support
_WORD_NUM = {
    "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,
    "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,
    "sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,
    "twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
}
def _word_to_number(token_seq: str) -> float | None:
    toks = [t for t in re.split(r"[\s-]+", token_seq.lower().strip()) if t]
    if not toks: return None
    total = 0
    for t in toks:
        if t not in _WORD_NUM: return None
        total += _WORD_NUM[t]
    return float(total)

WORDNUM_PERCENT_RE = re.compile(r"\b([A-Za-z-]+)\s*(?:percent|%)\b", re.I)
WORDNUM_MONEY_RE   = re.compile(r"\b([A-Za-z-]+)\s*(billion|million|thousand|k)\b", re.I)

def _to_float(s: str) -> float:
    return float(s.replace(",", ""))

def _scale(val: float, unit: str | None) -> float:
    if not unit:
        return val
    u = unit.lower()
    return val * UNIT_SCALE.get(u, 1)

def _human_money(v: float) -> str:
    if v >= 1_000_000_000: return f"{v/1_000_000_000:.1f}B"
    if v >= 1_000_000:     return f"{v/1_000_000:.1f}M"
    if v >= 1_000:         return f"{v/1_000:.1f}K"
    return f"{v:g}"

def pick_best_value(text: str) -> Tuple[str, str | None, str, str]:
    """
    Returns:
      value_text, normalized, unit, value_type
    """
    t = text

    # percent (numeric)
    mp = PCT_RE.search(t)
    if mp and not YEAR_TOKEN.search(mp.group(0)):
        pmin = float(mp.group("pmin").replace(",", ""))
        pmax = mp.group("pmax")
        if pmax:
            pmaxf = float(pmax.replace(",", ""))
            return f"{pmin:g}–{pmaxf:g}%", f"{pmaxf:g}", "%", "percent"
        return f"{pmin:g}%", f"{pmin:g}", "%", "percent"

    # money (numeric)
    mm = MONEY_RE.search(t)
    if mm and not YEAR_TOKEN.search(mm.group(0)):
        cur = mm.group("cur") or "$"
        vmin = _to_float(mm.group("min"))
        vmax = mm.group("max")
        unit = mm.group("unit")
        if vmax:
            vmaxf = _to_float(vmax)
            vmin_s = _scale(vmin, unit)
            vmax_s = _scale(vmaxf, unit)
            return (f"{cur}{_human_money(vmin_s)}–{cur}{_human_money(vmax_s)}",
                    str(int(round(vmax_s))), cur, "money")
        v = _scale(vmin, unit)
        return f"{cur}{_human_money(v)}", str(int(round(v))), cur, "money"

    # EPS
    eps = re.search(r"\b(\d+\.\d{1,3})\s*(?:per\s+share|eps)\b", t, re.I)
    if eps and not YEAR_TOKEN.search(eps.group(0)):
        v = eps.group(1)
        return v, v, "$", "eps"

    # word-number %  / money
    wp = WORDNUM_PERCENT_RE.search(t)
    if wp and not YEAR_TOKEN.search(wp.group(0)):
        n = _word_to_number(wp.group(1))
        if n is not None:
            return f"{int(n)}%", f"{int(n)}", "%", "percent"

    wm = WORDNUM_MONEY_RE.search(t)
    if wm and not YEAR_TOKEN.search(wm.group(0)):
        n = _word_to_number(wm.group(1))
        unit = wm.group(2).lower()
        if n is not None:
            scaled = _scale(float(n), unit)
            return f"${_human_money(scaled)}", str(int(round(scaled))), "$", "money"

    return "", None, "", "other"

# ==============================
# DEFAULT KEYWORDS + DYNAMIC STORE
# ==============================
DEFAULT_KEYWORDS: Dict[str, List[str]] = {
    "Revenue": ["revenue", "sales", "turnover", "top line"],
    "Net Income": ["net income", "net profit", "net earnings", "profit after tax"],
    "EBITDA": ["ebitda", "ebit", "operating profit", "operating earnings"],
    "EPS": ["earnings per share", "eps", "diluted eps"],
    "Cash Flow": ["cash flow", "free cash flow", "operating cash flow"],
    "CapEx": ["capital expenditures", "capex", "ppe", "property plant and equipment"],
    "Gross Margin": ["gross margin", "operating margin", "margin"],
    "Debt": ["debt", "borrowings", "loans", "interest expense"],
    "Guidance": ["guidance", "outlook", "expects", "forecast", "growth of"],
    "Opex": ["opex", "operating expenses", "sg&a", "r&d"],
    # Example built-ins for market topics:
    "Stock Price": ["stock price", "share price", "stock value", "share value", "market price", "stock prices", "share prices"],
}

CURRENT_KEYWORDS: Dict[str, List[str]] = copy.deepcopy(DEFAULT_KEYWORDS)

def _expand_synonyms_base(term: str) -> List[str]:
    """Handy finance-focused expansions and variants (lowercased)."""
    t = term.lower().strip()

    # seed map for common finance synonyms
    seeds: Dict[str, List[str]] = {
        "stock price": ["share price", "stock value", "share value", "market price", "stock prices", "share prices"],
        "market cap": ["market capitalization", "market capitalisation", "market-cap"],
        "operating expenses": ["opex", "operating expense", "operating costs", "operating cost"],
        "research and development": ["research & development", "r&d"],
        "sales": ["revenue", "turnover", "top line"],
        "net income": ["net profit", "net earnings", "profit after tax"],
        "earnings per share": ["eps", "diluted eps"],
        "free cash flow": ["fcf", "free cashflows", "free cash flows"],
        "capital expenditures": ["capex", "capital expense", "capital expenses"],
    }

    out = {t}

    # seed hits
    for k, vals in seeds.items():
        if t == k or t in vals:
            out.add(k)
            out.update(vals)

    # hyphen <-> space variants
    out.update({t.replace("-", " "), t.replace(" ", "-")})

    # pluralize last token naïvely (if it looks pluralizable)
    toks = t.split()
    if toks:
        last = toks[-1]
        plurals = []
        if not last.endswith("s"):
            if last.endswith("y") and len(last) > 2 and last[-2] not in "aeiou":
                plurals.append(last[:-1] + "ies")
            elif last.endswith(("s", "x", "z", "ch", "sh")):
                plurals.append(last + "es")
            else:
                plurals.append(last + "s")
        for p in plurals:
            out.add(" ".join(toks[:-1] + [p]))
            out.add("-".join(toks[:-1] + [p]))

    # & <-> and variants
    if " and " in t:
        out.add(t.replace(" and ", " & "))
    if " & " in t:
        out.add(t.replace(" & ", " and "))

    # compact variants for e.g. "r&d" / "R & D"
    if t in {"r&d", "r & d"}:
        out.update({"r&d", "r & d", "research & development", "research and development"})

    return sorted({x.strip() for x in out if x.strip()})

def _build_term_patterns(terms: List[str]) -> List[str]:
    """Build robust regex patterns (case-insensitive) for all terms."""
    pats: List[str] = []
    for raw in terms:
        t = raw.strip()
        if not t:
            continue
        # special handling for SG&A, R&D style
        if t.lower() in {"sg&a", "sga"}:
            pats.append(r"s\s*g\s*&\s*a")
            pats.append(r"sg&a")
            pats.append(r"sga")
            continue
        if t.lower() in {"r&d", "research & development", "research and development"}:
            pats.extend([r"r\s*&\s*d", r"research\s*&\s*development", r"research\s+and\s+development"])
            continue
        # normal phrase (escape, allow flexible whitespace)
        esc = re.escape(t).replace(r"\ ", r"\s+")
        pats.append(esc)
    return pats

def _rebuild_term_re() -> re.Pattern:
    all_terms: List[str] = []
    for terms in CURRENT_KEYWORDS.values():
        all_terms.extend(terms)
    patterns = _build_term_patterns(all_terms)
    if not patterns:
        # fallback to something that never matches
        return re.compile(r"$^")
    big = r"\b(" + "|".join(patterns) + r")\b"
    return re.compile(big, re.I)

# global term regex used in filters
TERM_RE = _rebuild_term_re()

def _keyword_concept(sentence: str) -> str:
    s = sentence.lower()
    scores: Dict[str, int] = {}
    for concept, terms in CURRENT_KEYWORDS.items():
        for t in terms:
            if re.search(rf"\b{re.escape(t)}\b", s, re.I):
                scores[concept] = scores.get(concept, 0) + 1
    return max(scores, key=scores.get) if scores else "General"

# ==============================
# TREND + FILTERS
# ==============================
TREND_UP   = re.compile(r"\b(increase|increased|grew|rose|growth|improved|higher|expanded|up)\b", re.I)
TREND_DOWN = re.compile(r"\b(decrease|decreased|declined|fell|lower|reduced|down|shrank|contraction)\b", re.I)

def detect_trend(text: str) -> str:
    s = text.lower()
    if TREND_UP.search(s):   return "Increase"
    if TREND_DOWN.search(s): return "Decrease"
    # blank when unclear
    return ""

def is_heading(line: str) -> bool:
    if not line: return True
    l = line.strip()
    if len(l) < 20: return True
    first = (l.splitlines() or [""])[0]
    if first.isupper(): return True
    if re.search(r"\b(consolidated|outlook|additional metrics|item\s+\d+|part\s+[ivx]+)\b", first, re.I):
        return True
    return False

_ALPHA_TOK = re.compile(r"[A-Za-z]")

def _alpha_token_count(s: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z]+", s)])

def qualifies_sentence(s: str) -> bool:
    if is_heading(s): return False
    if YEAR_ONLY.match(s): return False
    if PAGE_FOOT.match(s): return False
    if _alpha_token_count(s) < 3: return False  # avoid bare numbers / page refs
    contains_value = bool(
        re.search(r"[$£€]\s?\d", s) or
        re.search(r"\b\d[\d,]*\.?\d*\s?(b|bn|m|mn|k|million|billion|thousand)\b", s, re.I) or
        PCT_RE.search(s) or WORDNUM_PERCENT_RE.search(s) or WORDNUM_MONEY_RE.search(s)
    )
    return (TERM_RE.search(s) is not None) and contains_value

# ==============================
# SPLITTING (., ; and multi-metrics)
# ==============================
SPLIT_SENT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    base = [x.strip() for x in SPLIT_SENT.split(text.strip()) if x.strip()]
    out: List[str] = []
    for s in base:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        if len(parts) == 1 and (len(re.findall(r"[$£€]\s?\d", s)) + len(re.findall(r"\d[\d,]*\.?\d*\s?%", s))) >= 2:
            more = [p.strip() for p in re.split(r"\.\s+", s) if p.strip()]
            out.extend(more)
        else:
            out.extend(parts)
    return out

# ==============================
# EXTRACTION
# ==============================
def extract_pdf_fast(pdf_bytes: bytes, limit_pages: int = MAX_PAGES) -> List[Dict[str, Any]]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n = min(len(doc), limit_pages)
    for i in range(n):
        page = doc[i]
        txt = page.get_text("text").replace("A$", "$").replace("US$", "$")
        pages.append({"page": i + 1, "text": txt})
    doc.close()
    return pages

def build_candidates(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for p in pages:
        for s in split_sentences(p["text"]):
            if qualifies_sentence(s):
                cands.append({"page": p["page"], "sentence": s})
    if len(cands) > MAX_SENTENCES:
        def quick_score(x: str) -> int:
            sc = 0
            sc += len(TERM_RE.findall(x))
            if re.search(r"[$£€]\s?\d", x) or re.search(r"\d[\d,]*\.?\d*\s?%", x) or WORDNUM_PERCENT_RE.search(x) or WORDNUM_MONEY_RE.search(x):
                sc += 2
            return sc
        cands.sort(key=lambda c: quick_score(c["sentence"]), reverse=True)
        cands = cands[:MAX_SENTENCES]
    return cands

@torch.no_grad()
def rerank_batched(cands: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not cands: return []
    pairs = [(query, c["sentence"]) for c in cands]
    scores: List[float] = []
    for i in range(0, len(pairs), CE_BATCH_SIZE):
        chunk = pairs[i:i + CE_BATCH_SIZE]
        s = _xenc.predict(chunk)
        scores.extend([float(x) for x in s])
    mn, mx = float(min(scores)), float(max(scores))
    denom = (mx - mn + 1e-9)
    for c, s in zip(cands, scores):
        c["ce"] = s
        c["ce_norm"] = (s - mn) / denom
    cands.sort(key=lambda x: x["ce_norm"], reverse=True)
    return cands

def unique_by_sentence(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, int]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["Sentence"].strip(), r["Page"])
        if key in seen: continue
        seen.add(key)
        out.append(r)
    return out

def _concept_for_sentence(sentence: str) -> str:
    # concept via CURRENT_KEYWORDS, fallback heuristic
    c = _keyword_concept(sentence)
    if c == "General":
        # light fallback to previous static mapping style if needed
        pass
    return c

def process_document(file_name: str, pdf_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cands = build_candidates(pdf_pages)
    ranked = rerank_batched(cands, "Extract key financial metrics with values and trends.")
    rows: List[Dict[str, Any]] = []
    for c in ranked:
        sent = c["sentence"]
        value_text, norm_val, unit, _ = pick_best_value(sent)
        if not value_text:
            continue
        concept = _concept_for_sentence(sent)
        trend = detect_trend(sent)  # "" when unclear
        conf = round(0.7 * c["ce_norm"] + 0.3, 3)
        rows.append({
            "Document": file_name,
            "Concept": concept,
            "Sentence": sent,
            "Value": value_text,
            "Normalized_Value": norm_val if norm_val is not None else "",
            "Currency/Unit": unit if unit else ("%" if "%" in value_text else "$"),
            "Trend": trend,  # may be ""
            "Confidence": conf,
            "Page": c["page"],
        })
    rows = unique_by_sentence(rows)
    rows.sort(key=lambda x: (x["Page"], x["Concept"]))
    return rows

# ==============================
# HIGHLIGHTER (hover-only tooltip, dash-robust anchoring)
# ==============================
def _norm(s: str) -> str:
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

_PUNCT_STRIP = re.compile(r"^[^\w$£€%]+|[^\w$£€%]+$")

def _strip_tok(t: str) -> str:
    return _PUNCT_STRIP.sub("", t)

def _find_sentence_rect(page, sentence: str, word_cache) -> fitz.Rect | None:
    # 1) exact sentence
    rects = page.search_for(sentence)
    if rects:
        r = rects[0]
        return fitz.Rect(r.x0, r.y0, r.x1 + 6, r.y1)

    # 2) normalized sentence
    norm_sent = _norm(sentence)
    rects = page.search_for(norm_sent)
    if rects:
        r = rects[0]
        return fitz.Rect(r.x0, r.y0, r.x1 + 6, r.y1)

    # 3) token span fallback
    words = page.get_text("words") if not word_cache["words"] else word_cache["words"]
    toks = [(_strip_tok(_norm(w[4])), w) for w in words] if not word_cache["toks"] else word_cache["toks"]
    word_cache["words"], word_cache["toks"] = words, toks
    sent_tokens = [_strip_tok(x) for x in _norm(sentence).split(" ") if _strip_tok(x)]
    if sent_tokens and toks:
        n = len(sent_tokens)
        i = 0
        while i <= len(toks) - n:
            ok = True
            for j in range(n):
                if toks[i + j][0] != sent_tokens[j]:
                    ok = False
                    break
            if ok:
                x0 = min(words[i + k][0] for k in range(n))
                y0 = min(words[i + k][1] for k in range(n))
                x1 = max(words[i + k][2] for k in range(n))
                y1 = max(words[i + k][3] for k in range(n))
                return fitz.Rect(x0, y0 + 0.5, x1 + 6, y1 - 0.5)
            i += 1

    # 4) year-safe anchor on value fragment (dash-robust)
    frag = None
    m = re.search(
        r"([$£€]\s?\d[\d,\.]*\s?(?:b|bn|m|mn|k|million|billion|thousand)?|\d[\d,\.]*\s?%)",
        norm_sent, re.I,
    )
    if m and not YEAR_TOKEN.search(m.group(0)):
        frag = m.group(0)
    else:
        wm = WORDNUM_MONEY_RE.search(norm_sent)
        if wm and not YEAR_TOKEN.search(wm.group(0)):
            frag = wm.group(0)
        else:
            wp = WORDNUM_PERCENT_RE.search(norm_sent)
            if wp and not YEAR_TOKEN.search(wp.group(0)):
                frag = wp.group(0)

    if not frag:
        return None

    # dash-robust variants: 29.0–29.8 / 6–9%
    def _frag_variants(s: str) -> List[str]:
        if not s:
            return []
        cand = {s}
        if "-" in s:
            cand.add(s.replace("-", "–"))
            cand.add(s.replace("-", "—"))
        if "–" in s:
            cand.add(s.replace("–", "-"))
            cand.add(s.replace("–", "—"))
        if "—" in s:
            cand.add(s.replace("—", "-"))
            cand.add(s.replace("—", "–"))
        out = set()
        for c in cand:
            out.add(re.sub(r"\s*[–—-]\s*", "-", c))
            out.add(re.sub(r"\s*[–—-]\s*", "–", c))
            out.add(re.sub(r"\s*[–—-]\s*", "—", c))
        return list(out)

    hits = []
    for v in _frag_variants(frag):
        hits = page.search_for(v)
        if hits:
            break
    if not hits:
        return None

    anchor = hits[0]
    rect = fitz.Rect(anchor.x0 - 160, anchor.y0 - 1.0, anchor.x1 + 280, anchor.y1 + 1.0)
    rect.intersect(page.rect)
    return fitz.Rect(rect.x0, rect.y0 + 0.5, rect.x1 + 6, rect.y1 - 0.5)

def highlight_pdf(file_name: str, rows: List[Dict[str, Any]]) -> str:
    pdf_path = DATA_DIR / file_name
    doc = fitz.open(pdf_path.as_posix())

    for page in doc:
        pnum = page.number + 1
        local = [r for r in rows if r["Page"] == pnum]
        if not local:
            continue

        words = page.get_text("words")
        toks = [(_strip_tok(_norm(w[4])), w) for w in words] if words else []
        word_cache = {"words": words, "toks": toks}

        for r in local:
            rect = _find_sentence_rect(page, r["Sentence"], word_cache)
            if not rect:
                continue

            ann = page.add_highlight_annot(rect)
            ann.set_colors(stroke=HIGHLIGHT_STROKE, fill=HIGHLIGHT_FILL)
            ann.set_opacity(HIGHLIGHT_OPACITY)

            # Hover-only tooltip; small content (concept or concept • trend)
            tip = r["Concept"] if not r["Trend"] else f"{r['Concept']} • {r['Trend']}"
            ann.set_info({"title": "", "content": tip})
            try: ann.set_popup(None)
            except Exception: pass
            try: ann.set_open(False)
            except Exception: pass
            ann.update()

    out = OUT_DIR / f"{Path(file_name).stem}.highlighted.pdf"
    doc.save(out.as_posix())
    doc.close()
    return out.name

# ==============================
# API – PROCESS
# ==============================
class ProcessResp(BaseModel):
    message: str
    file: str
    csv: str
    highlighted_pdf: str
    preview_sentences: List[str]

@app.post("/process/upload", response_model=ProcessResp)
async def process_upload(file: UploadFile = File(...)):
    raw = await file.read()
    (DATA_DIR / file.filename).write_bytes(raw)

    pages = extract_pdf_fast(raw)
    rows = process_document(file.filename, pages)

    csv_name = f"{Path(file.filename).stem}_fintags.csv"
    pd.DataFrame(
        rows,
        columns=["Concept", "Sentence", "Value", "Normalized_Value", "Currency/Unit", "Trend", "Confidence", "Page"],
    ).to_csv(OUT_DIR / csv_name, index=False, encoding="utf-8")

    pdf_out = highlight_pdf(file.filename, rows)

    return {
        "message": "Processed",
        "file": file.filename,
        "csv": csv_name,
        "highlighted_pdf": pdf_out,
        "preview_sentences": [r["Sentence"] for r in rows[:50]],
    }

@app.get("/download/{filename}")
def download(filename: str):
    fp = OUT_DIR / filename
    if not fp.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    media = "application/pdf" if filename.lower().endswith(".pdf") else "text/csv"
    return FileResponse(path=fp.as_posix(), filename=filename, media_type=media)

@app.get("/")
def root():
    return {"status": "ok"}

# ==============================
# API – KEYWORD CONFIG (NEW)
# ==============================
class KeywordUpdate(BaseModel):
    add: Dict[str, List[str]] | None = None          # {"Stock Price": ["stock price", "stock value"]}
    remove_concepts: List[str] | None = None         # ["Opex", "Debt"]
    remove_terms: List[str] | None = None            # ["turnover", "operating earnings"]
    clear_defaults: bool = False                     # start from empty if True
    auto_synonyms: bool = True                       # expand synonyms/variants for added terms

@app.get("/config/keywords")
def get_keywords():
    return {"keywords": CURRENT_KEYWORDS}

@app.post("/config/keywords")
def update_keywords(update: KeywordUpdate):
    global CURRENT_KEYWORDS, TERM_RE

    # start set
    if update.clear_defaults:
        CURRENT_KEYWORDS = {}
    else:
        # work on a copy to be safe
        CURRENT_KEYWORDS = copy.deepcopy(CURRENT_KEYWORDS)

    # remove whole concepts
    if update.remove_concepts:
        for c in update.remove_concepts:
            CURRENT_KEYWORDS.pop(c, None)

    # remove specific terms (from any concept)
    if update.remove_terms:
        rmset = {t.lower().strip() for t in update.remove_terms}
        for c, terms in list(CURRENT_KEYWORDS.items()):
            CURRENT_KEYWORDS[c] = [t for t in terms if t.lower().strip() not in rmset]

    # add concepts/terms (+ optional synonym expansion)
    if update.add:
        for concept, terms in update.add.items():
            if concept not in CURRENT_KEYWORDS:
                CURRENT_KEYWORDS[concept] = []
            for t in terms:
                variants = _expand_synonyms_base(t) if update.auto_synonyms else [t]
                for v in variants:
                    if v not in CURRENT_KEYWORDS[concept]:
                        CURRENT_KEYWORDS[concept].append(v)

    # cleanup empties
    for c in list(CURRENT_KEYWORDS.keys()):
        if not CURRENT_KEYWORDS[c]:
            CURRENT_KEYWORDS.pop(c)

    # rebuild term regex used by filters
    TERM_RE = _rebuild_term_re()

    return {"message": "updated", "keywords": CURRENT_KEYWORDS}
