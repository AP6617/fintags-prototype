# app/value_extractor.py
import re
from typing import Optional, Tuple, Dict

# Magnitudes
_MAG = {
    "billion": 1_000_000_000, "bn": 1_000_000_000,
    "million": 1_000_000, "mn": 1_000_000,
    "thousand": 1_000, "k": 1_000
}

# Normalize dash
_DASH = r"[-–—]"

# $ ranges like $29.0-$29.8 billion  OR  29.0-29.8 billion (no $ but with billion/million)
USD_RANGE_RE = re.compile(
    rf"""
    (?P<prefix>\$)?\s?
    (?P<v1>\d[\d,]*\.?\d*)
    \s*{_DASH}\s*
    (?P<v2>\d[\d,]*\.?\d*)
    \s*(?P<mag>billion|bn|million|mn|thousand|k)?
    """, re.IGNORECASE | re.VERBOSE
)

# plain $ single value with optional magnitude word
USD_SINGLE_RE = re.compile(
    r"""
    (?P<prefix>\$)\s?
    (?P<v>\d[\d,]*\.?\d*)
    \s*(?P<mag>billion|bn|million|mn|thousand|k)?
    """, re.IGNORECASE | re.VERBOSE
)

# word-magnitude single value without $  e.g. 13.2b, 5.0 billion
USD_WORDVAL_RE = re.compile(
    r"""
    (?P<v>\d[\d,]*\.?\d*)\s*
    (?P<mag>billion|bn|million|mn|thousand|k)\b
    """, re.IGNORECASE | re.VERBOSE
)

# percent ranges like 6-9%  or 6 – 9 %
PCT_RANGE_RE = re.compile(
    rf"""
    (?P<p1>\d[\d,]*\.?\d*)\s*%?
    \s*{_DASH}\s*
    (?P<p2>\d[\d,]*\.?\d*)\s*%
    """, re.IGNORECASE | re.VERBOSE
)

# single percent like 8% or 0.5 %
PCT_SINGLE_RE = re.compile(
    r"(?P<p>\d[\d,]*\.?\d*)\s?%", re.IGNORECASE
)

BPS_RE = re.compile(r"(?P<bps>\d[\d,]{0,4})\s?bps\b", re.IGNORECASE)

def _to_float(x: str) -> float:
    return float(x.replace(",", ""))

def _apply_mag(v: float, mag: Optional[str]) -> float:
    if not mag:
        return v
    m = mag.lower()
    return v * _MAG.get(m, 1.0)

def extract_best_value(sentence: str) -> Dict[str, Optional[str]]:
    """
    Returns a single 'best' value found in the sentence for CSV display:
    - prefers $ range center, else $ single, else % range center, else % single, else bps
    Also returns the raw 'value_text' and 'unit'.
    """
    s = sentence

    # 1) $ range
    m = USD_RANGE_RE.search(s)
    if m:
        v1 = _apply_mag(_to_float(m.group("v1")), m.group("mag"))
        v2 = _apply_mag(_to_float(m.group("v2")), m.group("mag"))
        center = (v1 + v2) / 2.0
        return {"value": f"{center:.6g}", "unit": "$", "value_text": m.group(0).strip()}

    # 2) $ single with $ prefix
    m = USD_SINGLE_RE.search(s)
    if m:
        val = _apply_mag(_to_float(m.group("v")), m.group("mag"))
        return {"value": f"{val:.6g}", "unit": "$", "value_text": m.group(0).strip()}

    # 3) word-magnitude like 13.2b
    m = USD_WORDVAL_RE.search(s)
    if m:
        val = _apply_mag(_to_float(m.group("v")), m.group("mag"))
        return {"value": f"{val:.6g}", "unit": "$", "value_text": m.group(0).strip()}

    # 4) % range
    m = PCT_RANGE_RE.search(s)
    if m:
        p1 = _to_float(m.group("p1"))
        p2 = _to_float(m.group("p2"))
        center = (p1 + p2) / 2.0
        return {"value": f"{center:.6g}", "unit": "%", "value_text": m.group(0).strip()}

    # 5) % single
    m = PCT_SINGLE_RE.search(s)
    if m:
        p = _to_float(m.group("p"))
        return {"value": f"{p:.6g}", "unit": "%", "value_text": m.group(0).strip()}

    # 6) bps
    m = BPS_RE.search(s)
    if m:
        b = _to_float(m.group("bps"))
        return {"value": f"{b:.6g}", "unit": "bps", "value_text": m.group(0).strip()}

    return {"value": "", "unit": "", "value_text": ""}
