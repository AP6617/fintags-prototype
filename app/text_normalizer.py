# app/text_normalizer.py
import re
from typing import List

_SENT_END = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z(])")

def split_sentences(text: str) -> List[str]:
    # normalize spaces
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return []
    parts = re.split(_SENT_END, t)
    return [p.strip() for p in parts if p.strip()]

# Keep sentences that are likely financial (has number + finance words)
_FIN_WORDS = ("revenue","income","eps","earnings","margin","cost","expenses","operating",
              "cash","share","tax","capex","capital","dividend","liabilities","assets",
              "equity","sales","guidance","growth","yoy","qoq","year over year","quarter over quarter",
              "family of apps","reality labs")

def looks_financial_sentence(s: str) -> bool:
    sl = s.lower()
    if not any(w in sl for w in _FIN_WORDS):
        return False
    if not re.search(r"\d", s):
        # allow purely financial wording sometimes
        return any(k in sl for k in ("revenue","eps","net income","operating income","margin"))
    return True
