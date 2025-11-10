# FINTAGS PROTOTYPE - main.py
# This is the main server file for the FinTags prototype.
# It includes:
# 1. A FastAPI server to upload PDFs.
# 2. A "Manual Parser" that splits prose from tables.
# 3. A "RAG" system to classify the text.
# 4. A PDF highlighter to show the results.

from __future__ import annotations
import re
import copy
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # PyMuPDF
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# ==============================
# --- SETTINGS & GLOBALS ---
# ==============================

# Set to False to turn off the LoRA model
USE_LORA = True
# Set to False to turn off RAG (will use keywords only)
USE_RAG  = True
# How many sentences to re-rank at once
CE_BATCH_SIZE = 64

# Highlight color (light blue)
HIGHLIGHT_FILL = (0.73, 0.86, 1.00)
HIGHLIGHT_STROKE = None
HIGHLIGHT_OPACITY = 0.28

# --- FOLDER PATHS ---
# This assumes main.py is in a folder like /app
# and the other folders are next to it.
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_uploads"
OUT_DIR  = ROOT_DIR / "outputs"
# Assumes /models is one level up from /app
MODELS_DIR = ROOT_DIR.parent / "models" 
FINBERT_PATH = MODELS_DIR / "finbert_crossenc"

# Create folders if they don't exist
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# --- PDF LIMITS ---
MAX_PAGES = 400
MAX_SENTENCES = 6000 # Max sentences to process

# ==============================
# --- API SERVER SETUP ---
# ==============================
app = FastAPI(title="FinTags – Backend")

# Allow all web pages (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# --- LOAD AI MODELS ---
# ==============================

# Check if we have a GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Using device: {device} ---")

try:
    # This model is good at scoring sentence similarity (RAG)
    _rag = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    # This model is good at re-ranking search results
    _xenc = CrossEncoder(
        str(FINBERT_PATH) if FINBERT_PATH.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device=device
    )
except Exception as e:
    print(f"--- WARNING: Could not load ML models. {e} ---")
    print("--- Running in CPU/fallback mode. This will be slow. ---")
    device = "cpu"
    _rag = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    _xenc = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)


# --- LoRA Model (Optional) ---
_lora = None
_tok = None
_pairs = None

def _load_lora():
    global _lora, _tok, _pairs
    if not USE_LORA:
        print("--- LoRA is disabled. Skipping. ---")
        return
        
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        _lora_path = MODELS_DIR / "lora_classifier"
        
        if not _lora_path.exists():
            print("--- LoRA model not found. Skipping. ---")
            return
            
        _lora = AutoModelForSequenceClassification.from_pretrained(str(_lora_path))
        with open(_lora_path / "labels.json","r",encoding="utf-8") as f:
            meta = json.load(f)
        _pairs = meta["pairs"]
        print("--- LoRA model loaded successfully. ---")
    except Exception as e:
        print(f"--- LoRA hook failed to load: {e} ---")
        _lora = None

@torch.no_grad()
def lora_predict(sentence: str):
    if _lora is None:
        return None
    try:
        inputs = _tok(sentence, return_tensors="pt", truncation=True, padding=True, max_length=256)
        out = _lora(**inputs).logits.softmax(-1).squeeze(0)
        conf, idx = float(out.max().item()), int(out.argmax().item())
        concept, trend = _pairs[idx]
        return concept, trend, conf
    except Exception as e:
        print(f"--- LoRA predict error: {e} ---")
        return None

# ==============================
# --- VALUE PARSING ---
# (Helper functions to find numbers)
# ==============================

# $1.2B, 5M, 10 thousand
UNIT_SCALE = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

# Regex to find money: $1.2M, 1.5 billion, 100k
MONEY_RE = re.compile(
    r"""(?P<cur>[$£€])?\s?
        (?P<min>\d[\d,]*\.?\d*)
        (?:\s?[–\-]\s?(?P<max>\d[\d,]*\.?\d*))?
        \s?(?P<unit>k|m|mn|b|bn|thousand|million|billion)?\b[.)]?
    """,
    re.I | re.VERBOSE,
)

# Regex to find percentages: 5%, 10.5%
PCT_RE = re.compile(
    r"""(?P<pmin>\d[\d,]*\.?\d*)
        (?:\s?[–\-]\s?(?P<pmax>\d[\d,]*\.?\d*))?
        \s?%[.)]?
    """,
    re.I | re.VERBOSE,
)

# Regex for percentage points: 5 pp, 10 ppts
PCT_POINTS_RE = re.compile(
    r"""(?P<pp>\d[\d,]*\.?\d*)\s*(?:percentage\s+points|pp|ppts?)\b[.)]?""",
    re.I | re.VERBOSE,
)

# Regex for just a year
YEAR_ONLY  = re.compile(r"^\s*(?:FY|CY)?\s*(19|20)\d{2}(?:E)?\s*\.?\s*$", re.I)
# Regex for a year token (like FY2024)
YEAR_TOKEN = re.compile(r"\b(?:FY|CY)?(19|20)\d{2}(?:E)?\b", re.I)

# Helper function to turn "1,000" into 1000.0
def _to_float(s: str) -> float:
    return float(s.replace(",", ""))

# Helper function to scale a value (e.g., 1.2, "million" -> 1200000.0)
def _scale(val: float, unit: str | None) -> float:
    if not unit: return val
    u = unit.lower()
    return val * UNIT_SCALE.get(u, 1)

# Helper function to make numbers human-readable
def _human_money(v: float) -> str:
    if v >= 1_000_000_000: return f"{v/1_000_000_000:.1f}B"
    if v >= 1_000_000:     return f"{v/1_000_000:.1f}M"
    if v >= 1_000:         return f"{v/1_000:.1f}K"
    return f"{v:g}"

# This function finds the *best* value in a sentence.
def pick_best_value(text: str):
    # Special case: If it's a table sentence
    if text.startswith("Metric:"):
        parts = text.split(":")
        if len(parts) >= 3:
            val_part = parts[-1].strip()
            # Try to parse just the value part
            m = re.search(r"([$£€]?\s?\d[\d,]*\.?\d*\s?(?:b|bn|m|mn|k|million|billion)?)", val_part, re.I)
            text = m.group(1) if m else val_part
    
    # 1. Look for percentages first
    mp = PCT_RE.search(text)
    if mp and not YEAR_TOKEN.search(mp.group(0)):
        pmin = _to_float(mp.group("pmin"))
        return f"{pmin:g}%", f"{pmin:g}", "%", "percent"

    # 2. Look for percentage points
    mpp = PCT_POINTS_RE.search(text)
    if mpp and not YEAR_TOKEN.search(mpp.group(0)):
        v = _to_float(mpp.group("pp"))
        return f"{v:g} pp", f"{v:g}", "pp", "percent_points"

    # 3. Look for money
    mm = MONEY_RE.search(text)
    if mm and not YEAR_TOKEN.search(mm.group(0)):
        cur = mm.group("cur") or "$"
        vmin = _to_float(mm.group("min"))
        unit = mm.group("unit")
        v = _scale(vmin, unit)
        return f"{cur}{_human_money(v)}", str(int(round(v))), cur, "money"
    
    # 4. Look for EPS
    eps = re.search(r"\b(\d+\.\d{1,3})\s*(?:per\s+share|eps)\b", text, re.I)
    if eps and not YEAR_TOKEN.search(eps.group(0)):
        v = eps.group(1)
        return v, v, "$", "eps"

    # If no value is found, return empty
    return "", None, "", "other"

# ==============================
# --- KEYWORDS & CONCEPTS ---
# ==============================

# Our default list of financial concepts
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
    "Stock Price": ["stock price", "share price", "stock value", "share value", "market price", "stock prices", "share prices"],
}
# This will be our live list of keywords
CURRENT_KEYWORDS: Dict[str, List[str]] = copy.deepcopy(DEFAULT_KEYWORDS)

# This function builds one big regex to find any keyword
def _rebuild_term_re() -> re.Pattern:
    all_terms: List[str] = []
    for terms in CURRENT_KEYWORDS.values():
        all_terms.extend(terms)
    
    pats: List[str] = []
    for t in all_terms:
        if t:
            # Escape the term and allow for spaces
            pats.append(re.escape(t).replace(r"\ ", r"\s+"))
            
    if not pats:
        return re.compile(r"$^") # Empty regex
        
    big = r"\b(" + "|".join(pats) + r")\b"
    return re.compile(big, re.I)

# This is the global regex we will use
TERM_RE = _rebuild_term_re()

# This function checks a sentence for keywords and picks the best concept
def _keyword_concept(sentence: str) -> str:
    s = sentence.lower()
    
    # Special case: table sentences
    if sentence.startswith("Metric:"):
        parts = sentence.split(",")
        if parts:
            metric_part = parts[0].replace("Metric:", "").strip()
            s = metric_part # Only search the metric name
    
    scores: Dict[str, int] = {}
    for concept, terms in CURRENT_KEYWORDS.items():
        for t in terms:
            if re.search(rf"\b{re.escape(t)}\b", s, re.I):
                scores[concept] = scores.get(concept, 0) + 1
    
    return max(scores, key=scores.get) if scores else "General"

# These are the prompts we use for the RAG AI model.
# They help the AI understand *context*, not just keywords.
def _concept_prompts() -> Dict[str, str]:
    prompts = {
        "Revenue": "Statements about total sales, company revenue, or top-line turnover.",
        "Net Income": "Statements about net income, net profit, or profit after tax.",
        "EBITDA": "Statements about operating profit, operating earnings, EBIT, or EBITDA.",
        "EPS": "Statements about earnings per share or EPS.",
        "Cash Flow": "Statements about cash flow, free cash flow (FCF), or operating cash flow.",
        "CapEx": "Statements about capital expenditures, capex, or investment in property, plant, and equipment.",
        "Gross Margin": "Statements about gross margin, operating margin, or profit margins as a percentage.",
        "Debt": "Statements about company debt, borrowings, loans, or interest expenses.",
        "Guidance": "Statements about the company's future outlook, forecast, or expectations.",
        "Opex": "Statements about operating expenses, opex, SG&A, or R&D costs.",
        "Stock Price": "Statements about the stock price, share price, or market value.",
    }
    return prompts

# ==============================
# --- TREND DETECTION ---
# ==============================
TREND_UP   = re.compile(r"\b(increase|increased|grew|rose|growth|improved|higher|expanded|up)\b", re.I)
TREND_DOWN = re.compile(r"\b(decrease|decreased|declined|fell|lower|reduced|down|shrank|contraction)\b", re.I)

def detect_trend(text: str) -> str:
    s = text.lower()
    if TREND_UP.search(s):   return "Increase"
    if TREND_DOWN.search(s): return "Decrease"
    return ""

# ==============================
# --- PDF PARSING (THE CORE LOGIC) ---
# ==============================

# Regex to find numbers/values
MONEY_RE_LITE = re.compile(r"[$£€]\s?\d|\b\d[\d,]*\.?\d*\s?(b|bn|m|mn|k|million|billion|thousand)\b", re.I)
PCT_RE_LITE = re.compile(r"\d[\d,]*\.?\d*\s?%")
WORDNUM_RE_LITE = re.compile(r"\b([A-Za-z-]+)\s*(?:percent|%|billion|million|thousand|k)\b", re.I)
_NUM_TOKEN_LITE = re.compile(r"\d") # Any digit
_ALPHA_TOK = re.compile(r"[A-Za-z]")

def _alpha_token_count(s: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z]+", s)])

# This function is used to filter out junk
def is_heading_or_junk(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    
    first_line = s.splitlines()[0]
    
    # Filter page footers (e.g., "Page 5")
    if re.search(r"^\s*(page\s+\d+)\s*$", first_line, re.I):
        return True
    
    # Filter years (e.g., "FY2024")
    if YEAR_ONLY.match(s):
        return True

    # Check for numeric values, which headings *usually* don't have
    if (MONEY_RE_LITE.search(s) or PCT_RE_LITE.search(s) or WORDNUM_RE_LITE.search(s)):
        # Allow "FY2024" but not "$5.2B"
        if YEAR_TOKEN.search(s) and not MONEY_RE_LITE.search(s) and not PCT_RE_LITE.search(s):
            pass # Probably a heading like "Results for FY2024"
        else:
            return False # Has a value, so it's probably *not* a heading

    # Check for common heading keywords
    if re.search(r"^\s*(consolidated|outlook|item\s+\d+|note\s+\d+)\b", first_line, re.I):
        return True
        
    # Check for ALL-CAPS lines (e.g., "FINANCIAL HIGHLIGHTS")
    if (len(first_line) > 8 and first_line == first_line.upper() and _alpha_token_count(first_line) > 2):
        return True

    return False

# This function checks if a sentence is "good"
def qualifies_sentence(s: str) -> bool:
    if is_heading_or_junk(s): 
        return False
    if _alpha_token_count(s) < 3: # Must have at least 3 words
        return False 
    
    # Check if it has a value
    contains_value = bool(
        MONEY_RE_LITE.search(s) or
        PCT_RE_LITE.search(s) or
        WORDNUM_RE_LITE.search(s)
    )
    
    # Check if it has a keyword
    contains_keyword = bool(TERM_RE.search(s))
    
    # A "good" sentence must have BOTH
    return contains_keyword and contains_value

# This function splits a block of text into sentences
_SENT_END = re.compile(r"(?<=[.!?])\s+") # Simple sentence end
def smart_split_sentences(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_sentences = []
    
    for line in lines:
        raw_sentences.extend([x.strip() for x in _SENT_END.split(line) if x.strip()])
    
    out: List[str] = []
    for s in raw_sentences:
        # Also split on semicolons
        parts = [p.strip() for p in s.split(";") if p.strip()]
        out.extend(parts)
    
    return [s for s in out if not is_heading_or_junk(s)] # Final filter

# --- THIS IS THE MANUAL TABLE PARSER ---

# This function checks if a block of text *is* our sample table
def _is_table_block(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 2: # Must have header + data
        return False
    
    # --- THIS IS THE FIX ---
    # Search *all* lines for the header, not just the first one
    for line in lines:
        if "Metric" in line and "|" in line and "FY24" in line:
            return True
            
    return False

# This function parses *only* that sample table
def _parse_manual_table(text: str, page_num: int) -> List[Dict[str, Any]]:
    lines = text.splitlines()
    if not lines:
        return []

    header_line = ""
    header_idx = -1
    
    # Find the header row *anywhere* in the block
    for i, line in enumerate(lines):
        if "Metric" in line and "|" in line:
            header_line = line
            header_idx = i
            break
            
    if header_idx == -1: # Not the table we want
        return []

    header_cols = [h.strip() for h in header_line.split("|")]
    
    metric_col_idx = -1
    for i, h in enumerate(header_cols):
        if "Metric" in h:
            metric_col_idx = i
            break
            
    if metric_col_idx == -1:
        return []

    results = []
    # Process data rows
    for line in lines[header_idx + 1:]:
        cols = [c.strip() for c in line.split("|")]
        
        if len(cols) != len(header_cols):
            continue # Mismatched row, skip
            
        metric_name = cols[metric_col_idx]
        if not metric_name:
            continue
            
        # Iterate over the value columns
        for i in range(metric_col_idx + 1, len(header_cols)):
            col_name = header_cols[i]
            value = cols[i]
            
            if not value or not _NUM_TOKEN_LITE.search(value):
                continue
            
            # We found a clean row!
            sentence = f"Metric: {metric_name}, {col_name}: {value}"
            results.append({
                "page": page_num,
                "sentence": sentence,
                "is_table": True,
            })
    return results

# --- This function builds the list of all sentences ---
def build_candidates(doc: fitz.Document) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []

    # 1. Classify all blocks first
    for page in doc:
        if page.number >= MAX_PAGES:
            break
        page_num = page.number + 1
        
        # Get all text on the page, in text blocks
        blocks = page.get_text("blocks") 
        for b in blocks:
            block_text = b[4].strip().replace("\u2013", "-").replace("\u2014", "-")
            if not block_text:
                continue
            
            # --- This is the key logic ---
            # Is it our special table?
            if _is_table_block(block_text):
                # Yes: Use the special table parser
                table_rows = _parse_manual_table(block_text, page_num)
                cands.extend(table_rows)
            else:
                # No: Treat it as normal prose
                text = block_text.replace("A$", "$").replace("US$", "$")
                for s in smart_split_sentences(text):
                    if qualifies_sentence(s):
                        cands.append({
                            "page": page_num, 
                            "sentence": s, 
                            "is_table": False,
                        })
    return cands

# ==============================
# --- MAIN PROCESSING ---
# ==============================

@torch.no_grad()
def rerank_batched(cands: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not cands: 
        return []
        
    pairs = [(query, c["sentence"]) for c in cands]
    scores: List[float] = []
    
    for i in range(0, len(pairs), CE_BATCH_SIZE):
        chunk = pairs[i:i + CE_BATCH_SIZE]
        s = _xenc.predict(chunk)
        if s is not None: # Add a check in case model returns nothing
            scores.extend([float(x) for x in s])
            
    # --- THIS IS THE BUG FIX ---
    # If scores is empty, the min() and max() will crash.
    if not scores:
        print("--- Reranker returned no scores. Returning candidates as-is. ---")
        # Assign a default score so they still get processed
        for c in cands:
            c["ce_norm"] = 0.1 # low confidence
        return cands 
    
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

@torch.no_grad()
def rag_concept_for(sentences: List[str]) -> List[Tuple[str, float]]:
    if not USE_RAG or not sentences:
        return [("General", 0.0)] * len(sentences)

    prompts = _concept_prompts()
    c_labels = list(prompts.keys())
    p_emb = _rag.encode([prompts[c] for c in c_labels], normalize_embeddings=True, convert_to_tensor=True).to(device)
    s_emb = _rag.encode(sentences, normalize_embeddings=True, convert_to_tensor=True).to(device)

    sims = util.cos_sim(s_emb, p_emb)
    out = []
    for i in range(sims.size(0)):
        j = int(torch.argmax(sims[i]).item())
        score = float(sims[i, j].item())
        normalized_score = max(0.0, min(1.0, (score + 1.0) / 2.0))
        out.append((c_labels[j], normalized_score))
    return out

def process_document(file_name: str, pdf_bytes: bytes) -> List[Dict[str, Any]]:
    # This function runs the whole pipeline.
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        print(f"Error opening PDF {file_name}: {e}")
        return []
    
    # 1. Get all sentences (prose and table)
    print("--- Building candidates... ---")
    cands = build_candidates(doc)
    doc.close()
    
    if not cands:
        print("--- No candidates found. ---")
        return []
    print(f"--- Found {len(cands)} total candidates. ---")

    # 2. Re-rank them (put best ones at the top)
    print("--- Reranking sentences... ---")
    ranked = rerank_batched(cands, "Extract key financial metrics with values and trends.")
    
    # 3. Use RAG to find the best concept for each
    print("--- Running RAG concept matching... ---")
    sentences_to_classify = [c["sentence"] for c in ranked]
    rag_best = rag_concept_for(sentences_to_classify)

    # 4. Final processing
    rows: List[Dict[str, Any]] = []
    for c, (rag_concept, rag_score) in zip(ranked, rag_best):
        sent = c["sentence"]
        value_text, norm_val, unit, _ = pick_best_value(sent)
        if not value_text:
            continue # Skip if no value found

        # a) Get concept from keywords
        concept_kw = _keyword_concept(sent)
        
        # b) Get concept from RAG
        concept = concept_kw
        if USE_RAG and rag_score >= 0.65: # If RAG is confident
            concept = rag_concept

        # c) Get trend
        trend = detect_trend(sent)

        # d) (Optional) Use LoRA to override
        lora_pred = lora_predict(sent)
        if lora_pred and lora_pred[2] >= 0.80: # If LoRA is confident
            concept = lora_pred[0]
            if lora_pred[1]:
                trend = lora_pred[1].capitalize()

        # e) Get confidence score
        ce_norm = c.get("ce_norm", 0.1) # Use 0.1 as default
        conf = round(0.7 * ce_norm + 0.3 * rag_score, 3)

        rows.append({
            "Concept": concept,
            "Sentence": sent,
            "Value": value_text,
            "Normalized_Value": norm_val if norm_val is not None else "",
            "Currency/Unit": unit,
            "Trend": trend,
            "Confidence": conf,
            "Page": c["page"],
            "is_table": c.get("is_table", False),
        })

    # 5. Clean up and sort
    rows = unique_by_sentence(rows)
    rows.sort(key=lambda x: (x["Page"], x["Confidence"]), reverse=True)
    return rows

# ==============================
# --- PDF HIGHLIGHTING ---
# ==============================

def _norm(s: str) -> str:
    # Helper to clean up text
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _get_value_fragment(sentence: str) -> str | None:
    # Finds the *value* in a sentence string.
    # This is our fallback for highlighting.
    
    # 1. Try table-style first
    if sentence.startswith("Metric:"):
        val = (sentence.split(":")[-1] or "").strip()
        if val: return val
    
    # 2. Try money
    m = MONEY_RE.search(sentence)
    if m and not YEAR_TOKEN.search(m.group(0)):
        return m.group(0)
        
    # 3. Try percent
    m = PCT_RE.search(sentence)
    if m and not YEAR_TOKEN.search(m.group(0)):
        return m.group(0)

    return None

def _get_word_window_rect(page: fitz.Page, value_frag: str, word_cache: Dict) -> fitz.Rect | None:
    # --- THIS IS THE "HALF-HIGHLIGHT" FIX ---
    # It finds the value, then highlights 10 words
    # before and 10 words after it.
    
    # 1. Find the value fragment (e.g., "$164.50 billion")
    try:
        frag_hits = page.search_for(value_frag, quads=True)
        if not frag_hits:
            return None
    except Exception:
        return None
    
    frag_rect = frag_hits[0].rect # Get rect of the value
    
    # 2. Get all words on the page
    words = word_cache.get("words")
    if not words:
        words = page.get_text("words")
        word_cache["words"] = words
    if not words:
        return frag_rect # Fallback to just the value
        
    # 3. Find the words that are *inside* that value's rect
    start_idx, end_idx = -1, -1
    for i, w in enumerate(words):
        w_rect = fitz.Rect(w[:4])
        if frag_rect.intersects(w_rect):
            if start_idx == -1:
                start_idx = i
            end_idx = i
            
    if start_idx == -1:
        return frag_rect # Still no words found? Just highlight the value.
        
    # 4. Create the "window" (10 words before, 10 words after)
    window_start = max(0, start_idx - 10)
    window_end = min(len(words) - 1, end_idx + 10)
        
    # 5. Create a new, big rect from this window
    window_rect = fitz.Rect()
    for i in range(window_start, window_end + 1):
        window_rect |= fitz.Rect(words[i][:4])
        
    window_rect.intersect(page.rect) # Crop to page
    return window_rect

def _find_sentence_rect(page: fitz.Page, sentence: str, word_cache: Dict) -> fitz.Rect | None:
    # This is the 4-step fallback logic for highlighting
    
    # Step 1: Try to find the *exact* sentence
    try:
        quads = page.search_for(sentence, quads=True)
        if quads:
            rect = fitz.Rect()
            for q in quads: rect |= q.rect
            rect.intersect(page.rect)
            return rect
    except Exception:
        pass # It fails on special characters

    # Step 2: Try to find a "cleaned" version
    norm_sent = _norm(sentence)
    try:
        quads = page.search_for(norm_sent, quads=True)
        if quads:
            rect = fitz.Rect()
            for q in quads: rect |= q.rect
            rect.intersect(page.rect)
            return rect
    except Exception:
        pass
        
    # Step 3: Find the *value* (e.g., "$164.50 billion")
    value_frag = _get_value_fragment(norm_sent)
    if not value_frag:
        return None # Give up, can't find a value
        
    # Step 4: Use the new "window" highlighter
    rect = _get_word_window_rect(page, value_frag, word_cache)
    return rect


def highlight_pdf(file_name: str, rows: List[Dict[str, Any]]) -> str:
    pdf_path = DATA_DIR / file_name
    try:
        doc = fitz.open(pdf_path.as_posix())
    except Exception as e:
        print(f"Error opening PDF for highlighting: {e}")
        return ""

    for page in doc:
        pnum = page.number + 1
        local = [r for r in rows if r["Page"] == pnum]
        if not local: continue

        # A cache to store the page's words
        word_cache = {"words": None}

        for r in local:
            rect = _find_sentence_rect(page, r["Sentence"], word_cache)
            
            if not rect or rect.is_empty:
                continue
            
            # Add the blue highlight
            ann = page.add_highlight_annot(rect)
            ann.set_colors(stroke=HIGHLIGHT_STROKE, fill=HIGHLIGHT_FILL)
            ann.set_opacity(HIGHLIGHT_OPACITY)
            
            # Add the hover text
            tip = r["Concept"] if not r["Trend"] else f"{r['Concept']} • {r['Trend']}"
            ann.set_info({"title": "", "content": tip})

            # --- Tooltip Fix ---
            # This makes the highlight clean (no sticky note icon)
            try:
                ann.set_flags(fitz.ANNOT_FLAG_PRINT | fitz.ANNOT_FLAG_NO_COMMENT)
                ann.set_popup(None)
                ann.set_open(False)
            except Exception:
                pass
            
            ann.update()

    out_name = f"{Path(file_name).stem}.highlighted.pdf"
    out_path = OUT_DIR / out_name
    doc.save(str(out_path))
    doc.close()
    return out_name

# ==============================
# --- XBRL EXPORT (Bonus) ---
# ==============================
XBRL_MAP = {
    "Revenue": "us-gaap:Revenues",
    "Net Income": "us-gaap:NetIncomeLoss",
    "EPS": "us-gaap:EarningsPerShareDiluted",
    "Cash Flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
    "EBITDA": "us-gaap:OperatingIncomeLoss",
    "Gross Margin": "us-gaap:GrossProfit",
    "Debt": "us-gaap:LongTermDebtNoncurrent",
    "CapEx": "us-gaap:CapitalExpendituresIncurredButNotYetPaid",
}

def _period_from_sentence(sent: str) -> Dict[str, str]:
    m = re.search(r"\b(20\d{2}|19d{2})\b", sent)
    if m:
        yr = int(m.group(1))
        return {"start": f"{yr-1}-01-01", "end": f"{yr}-12-31"}
    y = datetime.date.today().year
    return {"start": f"{y-1}-01-01", "end": f"{y}-12-31"}

def build_xbrl_json(df: pd.DataFrame) -> dict:
    facts = {}
    for i, r in df.iterrows():
        tag = XBRL_MAP.get(str(r.get("Concept","")), "")
        if not tag:
            continue
        val = r.get("Normalized_Value") or r.get("Value") or ""
        unit = r.get("Currency/Unit") or ""
        sent = str(r.get("Sentence",""))
        
        if unit == "%":
            unit_ref = "pure"
        elif unit == "$":
            unit_ref = "iso47:USD"
        else:
            unit_ref = "iso4217:USD" # Default to USD
            
        facts[f"{tag}#{i}"] = {
            "concept": tag,
            "value": str(val),
            "unit": unit_ref,
            "entity": "urn:fin:demo:entity",
            "period": _period_from_sentence(sent),
            "extras": {"Sentence": sent, "Trend": r.get("Trend",""), "Page": r.get("Page","")}
        }
    return {"documentType":"xbrl-json-oim","facts":facts}

# ==============================
# --- API ENDPOINTS ---
# ==============================
class ProcessResp(BaseModel):
    message: str
    file: str
    csv: str
    highlighted_pdf: str
    xbrl_json: str
    preview_sentences: List[str]

@app.post("/process/upload", response_model=ProcessResp)
async def process_upload(file: UploadFile = File(...)):
    
    # 1. Save the uploaded file
    raw = await file.read()
    (DATA_DIR / file.filename).write_bytes(raw)
    
    # 2. Run the full processing pipeline
    print(f"--- Processing {file.filename} ---")
    rows = process_document(file.filename, raw)
    print(f"--- Processing complete. Found {len(rows)} rows. ---")

    # 3. Create the CSV file
    csv_name = f"{Path(file.filename).stem}_fintags.csv"
    df = pd.DataFrame(
        rows,
        columns=["Concept", "Sentence", "Value", "Normalized_Value", "Currency/Unit", "Trend", "Confidence", "Page", "is_table"],
    )
    df.to_csv(OUT_DIR / csv_name, index=False, encoding="utf-8")

    # 4. Create the Highlighted PDF
    print("--- Highlighting PDF... ---")
    pdf_out = highlight_pdf(file.filename, rows)
    print(f"--- Highlighted PDF saved to {pdf_out} ---")

    # 5. Create the XBRL JSON
    xjson = build_xbrl_json(df)
    xbrl_name = f"{Path(file.filename).stem}.xbrl.json"
    (OUT_DIR / xbrl_name).write_text(json.dumps(xjson, indent=2), encoding="utf-8")

    return {
        "message": "Processed",
        "file": file.filename,
        "csv": csv_name,
        "highlighted_pdf": pdf_out,
        "xbrl_json": xbrl_name,
        "preview_sentences": [r["Sentence"] for r in rows[:50]], # Send a preview
    }

@app.get("/download/{filename}")
def download(filename: str):
    fp = OUT_DIR / filename
    if not fp.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    media = "application/pdf" if filename.lower().endswith(".pdf") else (
            "application/json" if filename.lower().endswith(".json") else "text/csv"
        )
    return FileResponse(path=str(fp), filename=filename, media_type=media)

@app.get("/")
def root():
    # Load LoRA on the first visit
    if _lora is None:
        _load_lora()
    return {"status": "ok", "message": "FinTags Backend is running."}

class KeywordUpdate(BaseModel):
    add: Dict[str, List[str]] | None = None
    remove_concepts: List[str] | None = None

# --- FIX for 405 Error: Added the @app.get ---
@app.get("/config/keywords")
def get_keywords():
    return {"keywords": CURRENT_KEYWORDS}

@app.post("/config/keywords")
def update_keywords(update: KeywordUpdate):
    global CURRENT_KEYWORDS, TERM_RE
    
    if update.remove_concepts:
        for c in update.remove_concepts:
            CURRENT_KEYWORDS.pop(c, None)

    if update.add:
        for concept, terms in update.add.items():
            if concept not in CURRENT_KEYWORDS:
                CURRENT_KEYWORDS[concept] = []
            for t in terms:
                if t not in CURRENT_KEYWORDS[concept]:
                    CURRENT_KEYWORDS[concept].append(t)

    # Rebuild the big regex with the new keywords
    TERM_RE = _rebuild_term_re()
    return {"message": "updated", "keywords": CURRENT_KEYWORDS}
