# app/trend_detector.py
import re
from typing import Dict, Optional

# --- patterns ---------------------------------------------------------------

# Word-boundary variants to avoid "grownups" -> "up"
_UP_WORDS = r"\b(increase|increased|rise|rose|grow|growth|higher|improve|improved|up|widened)\b"
_DOWN_WORDS = r"\b(decrease|decreased|decline|declined|lower|fell|down|drop|dropped|deteriorated|narrowed)\b"
_STABLE_WORDS = r"\b(stable|flat|unchanged|steady|in line with)\b"

# “up 16%”, “decreased $2.9 billion”, “increase of 30 bps”, “declined by 5%”
_DELTA_SIMPLE = re.compile(
    r"\b(?:(up|down)|increase(?:d)?|decrease(?:d)?|rose|fell)\b"
    r"(?:\s+(?:by|of))?\s+"
    r"([$\-]?\d[\d,]*(?:\.\d+)?\s*(?:k|m|bn|billion|million|thousand|bps|%))",
    re.I,
)

# “from 10% to 12%”, “from $1.0b to $1.2b”
_DELTA_FROM_TO = re.compile(
    r"\bfrom\s+([$\-]?\d[\d,]*(?:\.\d+)?\s*(?:k|m|bn|billion|million|thousand|bps|%))\s+"
    r"(?:to|->)\s+([$\-]?\d[\d,]*(?:\.\d+)?\s*(?:k|m|bn|billion|million|thousand|bps|%))",
    re.I,
)

# Period hint
_PERIOD = re.compile(r"\b(yoy|year over year|qoq|quarter over quarter)\b", re.I)

# Metric aliases (expandable)
_METRICS = [
    ("revenue", ["revenue", "sales", "top line"]),
    ("eps", ["eps", "earnings per share"]),
    ("net income", ["net income", "net profit"]),
    ("operating income", ["operating income", "income from operations"]),
    ("margin", ["margin", "operating margin", "gross margin"]),
    ("expenses", ["expenses", "cost", "opex", "cogs"]),
    ("cash flow", ["cash flow", "free cash flow", "fcf"]),
    ("capex", ["capex", "capital expenditures"]),
    ("guidance", ["guidance", "outlook"]),
]

def _guess_metric(s: str) -> str:
    s = s.lower()
    for name, aliases in _METRICS:
        if any(a in s for a in aliases):
            return name
    return ""

def _period_hint(s: str) -> str:
    m = _PERIOD.search(s)
    if not m:
        return ""
    token = m.group(0).lower()
    if token.startswith("yoy") or "year over year" in token:
        return "YoY"
    if token.startswith("qoq") or "quarter over quarter" in token:
        return "QoQ"
    return ""

def _delta_phrase(text: str) -> Optional[str]:
    """Return a concise delta phrase if we can find one."""
    # from-to has priority (more informative)
    m2 = _DELTA_FROM_TO.search(text)
    if m2:
        return f"from {m2.group(1)} to {m2.group(2)}"
    m1 = _DELTA_SIMPLE.search(text)
    if m1:
        return m1.group(0)
    return None

def detect(sentence: str) -> Optional[Dict[str, str]]:
    """
    Detect trend direction/metric/period/delta from a sentence.
    Returns:
      {
        "trend_metric": "...",
        "trend_direction": "Increase|Decrease|Stable",
        "trend_period": "YoY|QoQ|",
        "trend_delta_text": "...",
        "trend_confidence": "high|med|low"
      }
    or None if no directional signal.
    """
    if not sentence or not sentence.strip():
        return None

    s = sentence.lower()

    # Direction (with stability)
    direction = ""
    if re.search(_STABLE_WORDS, s):
        direction = "Stable"
    elif re.search(_UP_WORDS, s):
        direction = "Increase"
    elif re.search(_DOWN_WORDS, s):
        direction = "Decrease"
    else:
        return None  # no trend language => skip

    metric = _guess_metric(s)
    period = _period_hint(s)
    delta = _delta_phrase(sentence) or ""

    # crude confidence: explicit delta + explicit metric = high; one of them = med; else low
    has_explicit_delta = bool(delta)
    has_metric = bool(metric)
    if has_explicit_delta and has_metric:
        conf = "high"
    elif has_explicit_delta or has_metric:
        conf = "med"
    else:
        conf = "low"

    return {
        "trend_metric": metric,
        "trend_direction": direction,
        "trend_period": period,
        "trend_delta_text": delta,
        "trend_confidence": conf,
    }
