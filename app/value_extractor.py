import re
from typing import Tuple, Optional

UNIT_SCALE = {
    "k": 1_000, "thousand": 1_000,
    "m": 1_000_000, "mn": 1_000_000, "million": 1_000_000,
    "b": 1_000_000_000, "bn": 1_000_000_000, "billion": 1_000_000_000,
}

MONEY_RE = re.compile(
    r"""(?P<cur>[$£€])?\s?
        (?P<min>\d[\d,]*\.?\d*)
        (?:\s?[–\-]\s?(?P<max>\d[\d,]*\.?\d*))?
        \s?(?P<unit>k|m|mn|b|bn|thousand|million|billion)?\b
    """, re.IGNORECASE | re.VERBOSE)

PCT_RE = re.compile(
    r"""(?P<pmin>\d[\d,]*\.?\d*)
        (?:\s?[–\-]\s?(?P<pmax>\d[\d,]*\.?\d*))?
        \s?%""", re.IGNORECASE | re.VERBOSE)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

def _to_float(s: str) -> float:
    return float(s.replace(",", ""))

def _scale(val: float, unit: Optional[str]) -> float:
    if not unit:
        return val
    unit = unit.lower()
    return val * UNIT_SCALE.get(unit, 1)

def pick_best_value(text: str) -> Tuple[str, Optional[str], str, str]:
    """
    Returns:
      value_text (human string, e.g., "$29.8B" or "6–9%")
      normalized (string integer/float w/o commas; for ranges, the UPPER bound)
      currency_or_unit ("$", "£", "€", or "%")
      value_type ("money" | "percent" | "eps" | "other")
    """
    t = text

    mp = PCT_RE.search(t)
    if mp:
        pmin = float(mp.group("pmin").replace(",", ""))
        pmax = mp.group("pmax")
        if pmax:
            pmaxf = float(pmax.replace(",", ""))
            val_text = f"{pmin:g}–{pmaxf:g}%"
            normalized = f"{pmaxf:g}"
        else:
            val_text = f"{pmin:g}%"
            normalized = f"{pmin:g}"
        return val_text, normalized, "%", "percent"

    mm = MONEY_RE.search(t)
    if mm:
        cur = mm.group("cur") or "$"
        vmin = _to_float(mm.group("min"))
        vmax = mm.group("max")
        unit = mm.group("unit")

        def human(x: float) -> str:
            if x >= 1_000_000_000: return f"{x/1_000_000_000:.1f}B"
            if x >= 1_000_000:     return f"{x/1_000_000:.1f}M"
            if x >= 1_000:         return f"{x/1_000:.1f}K"
            return f"{x:g}"

        if vmax:
            vmaxf = _to_float(vmax)
            vmin_s = _scale(vmin, unit)
            vmax_s = _scale(vmaxf, unit)
            val_text = f"{cur}{human(vmin_s)}–{cur}{human(vmax_s)}"
            normalized = str(int(round(vmax_s)))  # upper bound
            return val_text, normalized, cur, "money"
        else:
            v = _scale(vmin, unit)
            val_text = f"{cur}{human(v)}"
            normalized = str(int(round(v)))
            return val_text, normalized, cur, "money"

    eps = re.search(r"\b(\d+\.\d{1,3})\s*(?:per\s+share|eps)\b", t, re.I)
    if eps:
        v = eps.group(1)
        return v, v, "$", "eps"

    return "", None, "", "other"
