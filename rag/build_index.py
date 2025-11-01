import os, uuid
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "texts"
DB_DIR = ROOT / "data" / "chroma_db"
DB_DIR.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, max_chars=900, overlap=120):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = (buf[-overlap:] + " " + p).strip()
    if buf:
        chunks.append(buf)
    return chunks

def build_collection(client, name: str, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(name=name, embedding_function=ef)

# ---------- main ----------
def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))

    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"⚠️ No .txt files in {TEXT_DIR}. Run the PDF->text step first.")
        return

    print(f"Found {len(txt_files)} text file(s). Building indexes in {DB_DIR}...")
    for txt in txt_files:
        doc_id = txt.stem  # e.g., us_steel_2024
        print(f"• {doc_id} ...")
        text = load_text(txt)
        chunks = chunk_text(text, max_chars=900, overlap=120)
        coll = build_collection(client, name=doc_id)

        ids = [f"{doc_id}-{i}-{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk_id": i} for i in range(len(chunks))]
        coll.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"  added {len(chunks)} chunks ✅")

    print("Done. You can now query the DB.")

if __name__ == "__main__":
    main()
