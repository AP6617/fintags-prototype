# training/build_fintags_train.py
import pandas as pd
from pathlib import Path

OUT_DIR = Path("../outputs")
TRAIN_OUT = Path("./fintags_train.csv")
LABELS_OUT = Path("./labels.json")

# Use your latest, most-correct CSVs
files = sorted(OUT_DIR.glob("*_fintags.csv"))
rows = []
for f in files:
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        sent = str(r.get("Sentence","")).strip()
        concept = str(r.get("Concept","")).strip()
        trend = str(r.get("Trend","")).strip()  # may be ""
        if not sent: 
            continue
        rows.append({"sentence": sent, "concept": concept, "trend": trend})

# small negative sampling (sentences w/o numbers/concepts are not present in CSV, so skip)
# If you have a plain-text dump, you can add negatives with is_negative=1.

train = pd.DataFrame(rows).drop_duplicates()
pairs = sorted({(c, t) for c, t in zip(train["concept"], train["trend"])})
import json
LABELS_OUT.write_text(json.dumps({"pairs": list(pairs)}, indent=2), encoding="utf-8")
train.to_csv(TRAIN_OUT, index=False)
print(f"Saved {len(train)} rows to {TRAIN_OUT}")
print(f"Saved labels to {LABELS_OUT}")
