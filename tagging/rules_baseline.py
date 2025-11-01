import re
import csv
from pathlib import Path
import chromadb

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "data" / "chroma_db"
EXPORTS_DIR = ROOT / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Edit your starter concepts here ---
CONCEPTS = [
    "Net Income",
    "Total Revenue",
    "Earnings Per Share",
    "Shares Issued",
    "Operating Expenses",
    "Current Assets",
    "Total Liabilities",
]

# --- regex helpers ---
MONEY_RE = re.compile(r"(?:[$€£]\s?\d[\d,\.]*\s*(?:million|billion|m|bn)?)", re.I)
PCT_RE   = re.compile(r"\b\d[\d,\.]*\s?%")
NUM_RE   = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")

def extract_values(text: str):
    """Return a short preview of monetary/percent/number values in the paragraph."""
    money = MONEY_RE.findall(text)
    pct   = PCT_RE.findall(text)
    nums  = NUM_RE.findall(text)
    # de-duplicate, keep short
    values = []
    if money: values.append("money: " + ", ".join(sorted(set(money))[:5]))
    if pct:   values.append("percent: " + ", ".join(sorted(set(pct))[:5]))
    if nums:  values.append("number: " + ", ".join(sorted(set(nums))[:5]))
    return " | ".join(values)[:160]

def query_concept(client, collection_name: str, concept: str, k: int = 3):
    coll = client.get_collection(collection_name)
    res = coll.query(
        query_texts=[concept],
        n_results=k,
        include=["distances", "documents", "metadatas"]
    )
    hits = []
    docs = res["documents"][0]
    dists = res["distances"][0]
    for i, (doc_text, dist) in enumerate(zip(docs, dists), start=1):
        hits.append({
            "rank": i,
            "distance": float(dist),
            "paragraph": doc_text,
            "values_preview": extract_values(doc_text)
        })
    return hits

def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collections = [c.name for c in client.list_collections()]
    if not collections:
        print(f"⚠️ No Chroma collections found in {DB_DIR}. Run RAG build step first.")
        return

    out_csv = EXPORTS_DIR / "baseline_tagging_output.csv"
    rows = []

    print(f"Collections found: {collections}")
    for concept in CONCEPTS:
        for name in collections:
            hits = query_concept(client, name, concept, k=3)
            for h in hits:
                para = h["paragraph"]
                short_para = para[:500] + ("..." if len(para) > 500 else "")
                rows.append({
                    "doc": name,
                    "concept": concept,
                    "score": round(1 - h["distance"], 3),  # higher = better
                    "values": h["values_preview"],
                    "source_paragraph": short_para
                })
        print(f"✓ processed concept: {concept}")

    # write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc", "concept", "score", "values", "source_paragraph"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved: {out_csv}  (rows: {len(rows)})")
    print("Tip: open it in Excel to filter by concept/doc/score.")

if __name__ == "__main__":
    main()
