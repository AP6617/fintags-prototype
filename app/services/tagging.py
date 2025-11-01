# app/services/tagging.py
import re
from typing import List, Dict, Any, Optional
from app.services.indexer import query_topk, list_collections
from app.services.finbert_crossenc import relevance_score

# ---------------- Regex patterns ----------------
MONEY_RE = re.compile(
    r"(?P<cur>[$€£])\s?(?P<num>\d[\d,\. ,]*)\s*(?P<mult>million|billion|m|bn)?",
    re.I,
)
PCT_RE = re.compile(r"(?P<num>-?\d[\d\.]*)\s?%")
EPS_RE = re.compile(
    r"(?:eps|earnings\s+per\s+share)[^\d$€£]{0,10}(?:[$€£]\s?)?(?P<num>\d[\d\.]*)"
    r"|(?:[$€£]\s?)?(?P<num2>\d[\d\.]*)\s*(?:per\s+share)",
    re.I,
)

# ---------------- Concept anchors ----------------
ANCHORS = {
    "total revenue": ["revenue", "sales", "turnover", "net sales"],
    "net income": ["net income", "net earnings", "profit", "loss attributable"],
    "earnings per share": ["eps", "earnings per share", "diluted eps", "basic eps"],
    "shares issued": ["shares issued", "shares outstanding", "common shares"],
    "operating expenses": ["operating expenses", "opex", "operating costs"],
    "total liabilities": ["total liabilities"],
}

# ---------------- Utilities ----------------
def _clean_number_str(s: str) -> float:
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return float("nan")

def normalize_money(text: str) -> Dict[str, Any]:
    values, currencies, previews, spans = [], [], [], []
    for m in MONEY_RE.finditer(text):
        cur = (m.group("cur") or "").strip()
        raw = (m.group("num") or "").strip()
        mult = (m.group("mult") or "").lower().strip()
        base = _clean_number_str(raw)
        if mult in ("million", "m"):
            base *= 1_000_000
        elif mult in ("billion", "bn"):
            base *= 1_000_000_000
        if base == base:
            values.append(base)
            currencies.append(cur)
            previews.append(m.group(0).strip())
            spans.append(m.span())
    return {
        "preview": ("money: " + ", ".join(previews[:5])) if previews else "",
        "values": values,
        "currencies": currencies,
        "spans": spans,
    }

def normalize_percent(text: str) -> Dict[str, Any]:
    values, previews, spans = [], [], []
    for m in PCT_RE.finditer(text):
        val = _clean_number_str(m.group("num"))
        if val == val:
            values.append(val)
            previews.append(m.group(0).strip())
            spans.append(m.span())
    return {
        "preview": ("percent: " + ", ".join(previews[:5])) if previews else "",
        "values": values,
        "spans": spans,
    }

def detect_eps(text: str) -> Optional[float]:
    m = EPS_RE.search(text)
    if not m:
        return None
    raw = m.group("num") or m.group("num2")
    if not raw:
        return None
    val = _clean_number_str(raw)
    return val if val == val else None

def extract_values_preview(text: str) -> str:
    money = normalize_money(text)["preview"]
    pct = normalize_percent(text)["preview"]
    bits = [b for b in [money, pct] if b]
    return " | ".join(bits)[:160]

# ---------------- Value selection ----------------
def _find_anchor_positions(text: str, terms: List[str]) -> List[int]:
    text_l = text.lower()
    pos = []
    for t in terms:
        i = 0
        t_l = t.lower()
        while True:
            j = text_l.find(t_l, i)
            if j == -1:
                break
            pos.append(j)
            i = j + len(t_l)
    return pos

def pick_best_value(concept: str, paragraph: str,
                    money_info: Dict[str, Any],
                    pct_info: Dict[str, Any]) -> Dict[str, Any]:
    ckey = concept.lower().strip()

    # EPS special
    if "earnings per share" in ckey:
        eps_val = detect_eps(paragraph)
        if eps_val is not None:
            return {"best_value": eps_val, "best_value_type": "eps"}

    anchors = []
    for key, terms in ANCHORS.items():
        if key in ckey:
            anchors = terms
            break
    if not anchors:
        anchors = [ckey]

    anchor_pos = _find_anchor_positions(paragraph, anchors)
    candidates = []
    for val, (s, e) in zip(money_info["values"], money_info["spans"]):
        d = min((abs(s - a) for a in anchor_pos), default=1e6)
        candidates.append(("money", val, d))
    for val, (s, e) in zip(pct_info["values"], pct_info["spans"]):
        d = min((abs(s - a) for a in anchor_pos), default=1e6)
        candidates.append(("percent", val, d))

    if candidates:
        candidates.sort(key=lambda x: x[2])
        typ, val, _ = candidates[0]
        return {"best_value": val, "best_value_type": typ}

    return {"best_value": None, "best_value_type": None}

# ---------------- Main tagging ----------------
def tag_concepts(
    concepts: List[str],
    doc_ids: Optional[List[str]] = None,
    k: int = 3,
    min_score: float = 0.0,
    top1: bool = False,
    min_ce: float = 0.0,  # ✅ added support
) -> List[Dict[str, Any]]:
    if not doc_ids:
        doc_ids = list_collections()

    rows: List[Dict[str, Any]] = []
    for concept in concepts:
        for doc in doc_ids:
            res = query_topk(concept, doc, k=k)
            for hit in res["hits"]:
                if hit["score"] < min_score:
                    continue
                para = hit["paragraph"]

                # FinBERT relevance score
                ce = relevance_score(concept, para)
                if ce < min_ce:
                    continue

                money_info = normalize_money(para)
                pct_info = normalize_percent(para)
                best = pick_best_value(concept, para, money_info, pct_info)

                rows.append({
                    "doc": doc,
                    "concept": concept,
                    "rank": hit["rank"],
                    "score": hit["score"],
                    "finbert_ce_score": round(ce, 3),
                    "values": extract_values_preview(para),
                    "money_norm": money_info["values"],
                    "currency": money_info["currencies"],
                    "percent_norm": pct_info["values"],
                    "best_value": best["best_value"],
                    "best_value_type": best["best_value_type"],
                    "source_paragraph": para[:700] + ("..." if len(para) > 700 else "")
                })

    if top1:
        best_rows = {}
        for r in rows:
            key = (r["doc"], r["concept"])
            if key not in best_rows or r["score"] > best_rows[key]["score"]:
                best_rows[key] = r
        rows = list(best_rows.values())

    rows.sort(key=lambda r: (r["concept"].lower(), r["doc"].lower(), -r["score"]))
    return rows
