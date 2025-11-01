# app/ingestion_plus.py
import io, re, json
from typing import List, Dict, Any, Tuple
import pdfplumber
import pandas as pd

# -----------------------------
# 1) Normalise numbers / spacing
# -----------------------------
def normalise_numbers(text: str) -> str:
    # unify currency tokens
    text = text.replace("A$", "$").replace("US$", "$").replace("AUD", "$").replace("USD", "$")

    # remove thousand separators: 1,234 -> 1234  (but keep decimals)
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', text)

    # insert space between number + unit stuck together (e.g., 100m -> 100 m; 5bn -> 5 bn)
    text = re.sub(r'(\d)(?=(k|m|b|bn|million|billion|thousand)\b)', r'\1 ', text, flags=re.I)

    # insert space between currency and number if stuck: $1000 -> $ 1000
    text = re.sub(r'([$€£])(?=(\d))', r'\1 ', text)

    # fix very long digit runs by adding thin spaces every 3 (only if > 6 digits)
    def _chunk_digits(m):
        s = m.group(0)
        if len(s) <= 6: return s
        rev = s[::-1]
        grouped = ' '.join([rev[i:i+3] for i in range(0, len(rev), 3)])[::-1]
        return grouped
    text = re.sub(r'\b\d{7,}\b', _chunk_digits, text)

    return text

# -----------------------------------
# 2) Extract text + tables page-by-page
# -----------------------------------
def extract_pdf_with_tables(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Returns:
      {
        'pages': [
           {'page_num': 1, 'text': '...cleaned...', 'tables': [df1_as_dict, df2_as_dict, ...]},
           ...
        ],
        'table_markdown': {1: ['|...|','|...|'], 2: [...]}
      }
    """
    out_pages: List[Dict[str, Any]] = []
    table_md_map: Dict[int, List[str]] = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # layout-aware text (tolerances help keep word gaps)
            text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            text = normalise_numbers(text)

            # try to extract tables
            page_tables = []
            md_list = []
            try:
                tables = page.extract_tables()
                for tbl in tables or []:
                    # turn raw list-of-rows into DataFrame safely
                    if not tbl: 
                        continue
                    # pad ragged rows
                    maxlen = max(len(r) for r in tbl)
                    padded = [r + [""]*(maxlen-len(r)) for r in tbl]
                    df = pd.DataFrame(padded)
                    # promote first row to header if it looks like headers
                    if df.shape[0] >= 2:
                        header_candidate = df.iloc[0].tolist()
                        # simple heuristic: >= 2 non-empty cells -> treat as header
                        if sum(1 for c in header_candidate if str(c).strip()) >= 2:
                            df.columns = [str(c).strip() or f"col_{j}" for j, c in enumerate(header_candidate)]
                            df = df.iloc[1:].reset_index(drop=True)
                    page_tables.append(df)

                    # markdown version for context
                    md = df.to_markdown(index=False)
                    md_list.append(md)
            except Exception:
                # table extraction is best-effort; ignore failures per page
                pass

            if md_list:
                table_md_map[i] = md_list

            out_pages.append({
                "page_num": i,
                "text": text,
                "tables": [df.to_dict(orient="records") for df in page_tables]
            })

    return {"pages": out_pages, "table_markdown": table_md_map}

# -----------------------------------
# 3) Build context chunks with tables
# -----------------------------------
def build_chunks_with_tables(pages: List[Dict[str, Any]], max_chars: int = 1200) -> List[Dict[str, Any]]:
    """
    Mixes normal text chunks with short table snapshots so retrieval sees both.
    Each chunk includes page number metadata.
    """
    chunks: List[Dict[str, Any]] = []

    for p in pages:
        pg = p["page_num"]
        txt = (p["text"] or "").strip()

        # split text into paragraphs, then chunk
        paras = [t.strip() for t in re.split(r'\n\s*\n', txt) if t and t.strip()]
        buf = ""
        for para in paras:
            if len(buf) + len(para) + 1 <= max_chars:
                buf += ("\n" if buf else "") + para
            else:
                if buf:
                    chunks.append({"text": buf.strip(), "page": pg, "type": "text"})
                buf = para
        if buf:
            chunks.append({"text": buf.strip(), "page": pg, "type": "text"})

        # add tables as small context chunks (markdown, trimmed if huge)
        # keep first ~40 lines to prevent massive chunks
        for tbl in p.get("tables", []):
            df = pd.DataFrame(tbl)
            if df.empty:
                continue
            md = df.to_markdown(index=False)
            md_lines = md.splitlines()
            if len(md_lines) > 40:
                md = "\n".join(md_lines[:40]) + "\n… (trimmed)"
            table_chunk = f"TABLE (page {pg}):\n{md}"
            chunks.append({"text": table_chunk, "page": pg, "type": "table"})

    return chunks

# -----------------------------------
# 4) Build a per-concept synthesis
# -----------------------------------
DEFAULT_CONCEPTS = {
    "revenue": ["revenue","total revenue","sales","net sales","turnover"],
    "net_income": ["net income","profit after tax","net profit","profit attributable"],
    "eps": ["earnings per share","eps","basic eps","diluted eps","statutory eps"],
    "ebitda": ["ebitda","earnings before interest tax depreciation amortisation","earnings before interest, tax, depreciation and amortization"]
}

def make_synthesis(chunks: List[Dict[str, Any]], concepts: Dict[str, List[str]] = None, max_hits_per_concept: int = 5):
    concepts = concepts or DEFAULT_CONCEPTS
    syn = {}
    for key, synonyms in concepts.items():
        key_hits: List[Tuple[int, str]] = []
        patt = re.compile(r'(' + '|'.join([re.escape(s) for s in synonyms]) + r')', re.I)
        for ch in chunks:
            if ch["type"] not in ("text","table"):
                continue
            txt = ch["text"]
            if patt.search(txt):
                # take a short snippet (first line / sentence)
                line = txt.strip().splitlines()[0]
                snippet = line if len(line) <= 220 else line[:217] + "…"
                key_hits.append((ch["page"], snippet))
                if len(key_hits) >= max_hits_per_concept:
                    break
        if key_hits:
            syn[key] = [{"page": p, "snippet": s} for p, s in key_hits]
    return syn
