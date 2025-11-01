import csv, json, random
from pathlib import Path

RAW = Path("exports/fintags_tags.csv")
OUT = Path("exports/finbert_train.jsonl")

if not RAW.exists():
    raise FileNotFoundError(f"{RAW} not found. Run /tag/export/csv first.")

rows = list(csv.DictReader(open(RAW, encoding="utf-8")))
random.shuffle(rows)

samples = []
for r in rows:
    try:
        score = float(r.get("score", 0))
    except ValueError:
        score = 0
    label = 1.0 if score >= 0.6 else 0.0
    samples.append({
        "concept": r["concept"],
        "paragraph": r["source_paragraph"],
        "label": label
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print(f"✅ Saved {len(samples)} training examples to {OUT}")
