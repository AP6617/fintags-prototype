from pathlib import Path
import chromadb

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chroma_db"

def search(query: str, doc_id: str = None, k: int = 3):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if doc_id:
        names = [doc_id]
    else:
        # query all collections
        names = [c.name for c in client.list_collections()]

    all_hits = []
    for name in names:
        coll = client.get_collection(name)
        res = coll.query(query_texts=[query], n_results=k, include=["distances", "metadatas", "documents"])
        for i in range(len(res["documents"][0])):
            all_hits.append({
                "doc": name,
                "rank": i+1,
                "distance": float(res["distances"][0][i]),
                "chunk": res["documents"][0][i][:500] + ("..." if len(res["documents"][0][i]) > 500 else "")
            })
    # sort by distance (lower = better)
    all_hits.sort(key=lambda x: x["distance"])
    return all_hits[:k]

if __name__ == "__main__":
    # try some finance queries
    for q in ["net income 2024", "total revenue", "shares issued", "earnings per share"]:
        print(f"\n=== QUERY: {q} ===")
        for hit in search(q, doc_id=None, k=3):
            print(f"[{hit['doc']}] d={hit['distance']:.3f} :: {hit['chunk']}\n")
