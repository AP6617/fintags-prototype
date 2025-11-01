import os, re, hashlib
from typing import List, Tuple, Dict
import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

_NUM_MODEL = None
_DB_CLIENT = None

SENT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")
PERCENT_RE = re.compile(r"(?<![A-Za-z])([-+]?\d{1,3}(?:[.,]\d{1,2})?)\s?%")
BPS_RE = re.compile(r"([-+]?\d{1,4})\s?bps\b", re.I)
USD_RE = re.compile(r"\$\s?([\d.,]+)\s*(billion|bn|million|mn|thousand|k)?", re.I)
PERIOD_RE = re.compile(
    r"\b(Q[1-4]\s*(FY\s*)?\d{4}|FY\s*\d{4}|for the (three|six|nine|twelve) months ended\s*[A-Za-z]+\s*\d{1,2},?\s*\d{4}|year ended\s*[A-Za-z]+\s*\d{1,2},?\s*\d{4})",
    re.I,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

UNIT_MAP = {
    "usd": "USD",
    "%": "%",
    "bps": "bps",
    "usd/share": "USD/share",
    "none": "none",
}

MAG_MAP = {
    "billion": 1_000_000_000, "bn": 1_000_000_000,
    "million": 1_000_000, "mn": 1_000_000,
    "thousand": 1_000, "k": 1_000
}

def safe_mkdirs(paths: List[str]):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def compute_document_hash(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]

def _get_model():
    global _NUM_MODEL
    if _NUM_MODEL is None:
        _NUM_MODEL = SentenceTransformer(SENT_MODEL_NAME)
    return _NUM_MODEL

def _get_client(persist_dir: str):
    global _DB_CLIENT
    if _DB_CLIENT is None:
        _DB_CLIENT = chromadb.Client(Settings(persist_directory=persist_dir, anonymized_telemetry=False))
    return _DB_CLIENT

def ensure_chroma_collection(persist_dir: str, collection_name: str):
    client = _get_client(persist_dir)
    try:
        col = client.get_collection(collection_name)
    except Exception:
        col = client.create_collection(collection_name)
    return col

def extract_pdf_sentences(pdf_path: str) -> Tuple[List[str], Dict[int, int]]:
    sentences: List[str] = []
    page_index: Dict[int, int] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=1.5, y_tolerance=1.0) or ""
            text = re.sub(r"\s{2,}", " ", text)
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = text.replace("\n", " ")
            for s in SENT_SPLIT.split(text.strip()):
                s_clean = s.strip()
                if not s_clean:
                    continue
                sentences.append(s_clean)
                page_index[len(sentences) - 1] = pno
    return sentences, page_index

def sentences_to_chunks(sentences: List[str], max_chars: int = 900, overlap: int = 120) -> List[str]:
    chunks: List[str] = []
    cur = []
    cur_len = 0
    for s in sentences:
        if cur_len + len(s) + 1 > max_chars:
            if cur:
                chunks.append(" ".join(cur))
                spill = []
                spill_len = 0
                for t in reversed(cur):
                    if spill_len + len(t) + 1 <= overlap:
                        spill.insert(0, t)
                        spill_len += len(t) + 1
                    else:
                        break
                cur = spill
                cur_len = sum(len(x) + 1 for x in spill)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def embed_chunks(collection, doc_id: str, chunks: List[str]):
    model = _get_model()
    existing = set(collection.get(ids=None)["ids"] or [])
    new_ids, new_texts = [], []
    for i, ch in enumerate(chunks):
        cid = f"{doc_id}_{i}"
        if cid in existing:
            continue
        new_ids.append(cid)
        new_texts.append(ch)
    if new_ids:
        embs = model.encode(new_texts, batch_size=64, show_progress_bar=False).tolist()
        collection.add(ids=new_ids, documents=new_texts, embeddings=embs, metadatas=[{"i": i} for i in range(len(new_ids))])

def search_chunks(collection, queries: List[str], top_k: int = 10):
    model = _get_model()
    q_emb = model.encode(queries, batch_size=8, show_progress_bar=False).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    hits = []
    for docs, ids in zip(res.get("documents", []), res.get("ids", [])):
        if not docs:
            continue
        for d, i in zip(docs, ids):
            hits.append({"id": i, "chunk": d})
    return hits

def detect_period(sentence: str) -> str:
    m = PERIOD_RE.search(sentence)
    return m.group(0) if m else ""

def parse_numeric(text: str) -> Tuple[float, str, str]:
    m = USD_RE.search(text)
    if m:
        raw = m.group(1)
        mag = m.group(2).lower() if m.group(2) else None
        v = float(str(raw).replace(",", ""))
        if mag in MAG_MAP:
            v *= MAG_MAP[mag]
        return v, UNIT_MAP["usd"], m.group(0)
    m = PERCENT_RE.search(text)
    if m:
        try:
            v = float(m.group(1).replace(",", "."))
        except Exception:
            v = None
        return v if v is not None else float("nan"), UNIT_MAP["%"], m.group(0)
    m = BPS_RE.search(text)
    if m:
        return float(m.group(1)), UNIT_MAP["bps"], m.group(0)
    m = re.search(r"\$\s?([\d.,]+)\s*/\s*share|\$\s?([\d.,]+)\s+per\s+share", text, re.I)
    if m:
        val = m.group(1) or m.group(2)
        return float(val.replace(",", "")), UNIT_MAP["usd/share"], m.group(0)
    return float("nan"), UNIT_MAP["none"], ""
