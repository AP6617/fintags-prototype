# app/text_normalizer.py
import re

# keep your existing YEAR regex elsewhere if you import it there

# Normalize fancy dashes to '-' and join "Guidance:" with its continuation.
_DASHES = re.compile(r"[–—-]")
_MULTI_SPACE = re.compile(r"\s{2,}")
_HYPH_SPLIT_FIX = re.compile(r"(\w)-\s+(\w)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

def normalize_page_text(raw: str) -> str:
    if not raw:
        return ""
    s = raw.replace("\r", " ").replace("\n", " ")
    s = _DASHES.sub("-", s)                  # normalize en/em dashes
    s = _HYPH_SPLIT_FIX.sub(r"\1\2", s)      # fix hyphenated wraps
    s = _MULTI_SPACE.sub(" ", s).strip()
    return s

def split_into_sentences(text: str):
    """
    Split, but stitch 'Guidance:' and similar labels with the following sentence.
    Also avoids splitting on semicolons so multi-metric lines stay together.
    """
    if not text:
        return []

    # first split by sentence enders
    parts = _SPLIT_ON_ENDERS(text)

    # stitch label-like prefixes to next chunk
    stitched = []
    i = 0
    while i < len(parts):
        cur = parts[i].strip()
        if cur.endswith(":") and i + 1 < len(parts):
            nxt = parts[i + 1].strip()
            stitched.append(f"{cur} {nxt}")
            i += 2
        else:
            stitched.append(cur)
            i += 1

    # final cleanup & drop empties
    return [p.strip() for p in stitched if p.strip()]

def _SPLIT_ON_ENDERS(text: str):
    # don’t split on semicolons; we want full multi-metric sentences highlighted
    # split on . ! ? followed by space + capital/(
    return _SENT_SPLIT.split(text)
