from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parents[2]
TEXT_DIR = ROOT / "data" / "texts"
DB_DIR = ROOT / "data" / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

def chunk_text(text: str, max_chars=900, overlap=120) -> List[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] + " " + p).strip() if overlap > 0 else p
    if buf:
        chunks.append(buf)
    return chunks

def build_collection(doc_id: str, model_name="sentence-transformers/all-MiniLM-L6-v2") -> int:
    """Create/replace a Chroma collection for this doc. Returns number of chunks."""
    txt_path = TEXT_DIR / f"{doc_id}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"No text file for {doc_id}. Ingest first.")
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    try:
        client.delete_collection(doc_id)  # replace if exists
    except Exception:
        pass

    coll = client.create_collection(name=doc_id, embedding_function=ef)
    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metas = [{"doc_id": doc_id, "chunk_id": i} for i in range(len(chunks))]
    coll.add(documents=chunks, ids=ids, metadatas=metas)
    return len(chunks)

def list_collections() -> list:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return [c.name for c in client.list_collections()]

def query_topk(concept: str, doc_id: str, k: int = 3) -> Dict[str, Any]:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    coll = client.get_collection(doc_id)
    res = coll.query(query_texts=[concept], n_results=k, include=["documents", "distances"])
    hits = []
    docs = res["documents"][0]
    dists = res["distances"][0]
    for i, (para, dist) in enumerate(zip(docs, dists), start=1):
        hits.append({"rank": i, "score": round(1 - float(dist), 3), "paragraph": para})
    return {"doc_id": doc_id, "concept": concept, "hits": hits}
