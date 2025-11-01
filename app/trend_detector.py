# app/trend_detector.py
import re
from typing import Dict, Optional

_UP = r"(increase|increased|rise|rose|higher|grew|growth|up)"
_DOWN = r"(decrease|decreased|decline|declined|lower|fell|down|drop|dropped)"

def _delta_phrase(text: str) -> Optional[str]:
    # captures "up 16%" or "decreased $2.9 billion" or "up $3.2b YoY"
    m = re.search(r"((up|down|increase[sd]?|decrease[sd]?|rose|fell)\s+(?:\$?\d[\d,]*\.?\d*\s*(?:k|m|bn|billion|million|thousand|tn|t)?|\d+\.?\d*\s*%|[0-9]+\s*bps))", text, re.I)
    return m.group(1) if m else None

def detect(sentence: str) -> Optional[Dict[str, str]]:
    s = sentence.lower()
    direction = ""
    if re.search(_UP, s):
        direction = "up"
    elif re.search(_DOWN, s):
        direction = "down"
    else:
        return None

    # metric guess
    metric = ""
    if "revenue" in s: metric = "revenue"
    elif "eps" in s or "earnings per share" in s: metric = "eps"
    elif "income from operations" in s or "operating income" in s: metric = "operating income"
    elif "net income" in s: metric = "net income"
    elif "margin" in s: metric = "margin"
    elif "cost" in s or "expenses" in s: metric = "expenses"

    # period hints
    period = ""
    if "yoy" in s or "year over year" in s: period = "YoY"
    elif "qoq" in s or "quarter over quarter" in s: period = "QoQ"

    delta = _delta_phrase(sentence) or ""

    return {
        "trend_metric": metric,
        "trend_direction": direction,
        "trend_period": period,
        "trend_delta_text": delta
    }
