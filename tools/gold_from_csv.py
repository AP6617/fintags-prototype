import csv, json, argparse
from pathlib import Path

def build_gold_skeleton(csv_path: Path, out_jsonl: Path):
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sent = r["sentence"].strip()
            kw = r["keyword"].strip().lower()
            if not kw:
                kw = "auto"
            rows.append({"sentence": sent, "keywords": [kw], "label": True})  # default True; adjust later

    # de-dup while preserving order
    seen = set(); uniq = []
    for row in rows:
        key = (row["sentence"], tuple(row["keywords"]))
        if key not in seen:
            seen.add(key); uniq.append(row)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in uniq:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {out_jsonl} with {len(uniq)} items; edit 'label' as needed.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Your produced CSV path")
    ap.add_argument("--out", default="gold.jsonl", help="Output JSONL path")
    args = ap.parse_args()
    build_gold_skeleton(Path(args.csv), Path(args.out))
