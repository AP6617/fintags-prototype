import csv, json
from pathlib import Path

# Evaluate sentence-level tagging vs a gold jsonl:
# each line: {"sentence": "...", "keywords": ["revenue","eps"], "label": true}
def evaluate(pred_csv: Path, gold_jsonl: Path, out_html: Path):
    gold = []
    for line in gold_jsonl.read_text(encoding="utf-8").splitlines():
        gold.append(json.loads(line))
    preds = []
    with pred_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds.append({"sentence": row["sentence"].strip(), "keyword": row["keyword"].lower()})

    P = {(p["sentence"], p["keyword"]) for p in preds}
    G = {(g["sentence"].strip(), kw.lower())
         for g in gold for kw in g.get("keywords", [])
         if g.get("label", True)}

    tp = len(P & G); fp = len(P - G); fn = len(G - P)
    prec = tp / (tp + fp) if tp + fp else 0
    rec  = tp / (tp + fn) if tp + fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0

    out_html.write_text(f"""
    <h2>FinTags Evaluation</h2>
    <p>Precision: {prec:.3f} &nbsp; Recall: {rec:.3f} &nbsp; F1: {f1:.3f}</p>
    <h3>False Positives ({fp})</h3>
    <ul>{"".join(f"<li>{s} [{k}]</li>" for (s,k) in (P-G))}</ul>
    <h3>False Negatives ({fn})</h3>
    <ul>{"".join(f"<li>{s} [{k}]</li>" for (s,k) in (G-P))}</ul>
    """, encoding="utf-8")
    return {"precision":prec, "recall":rec, "f1":f1}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv", help="Path to produced CSV")
    ap.add_argument("gold_jsonl", help="Path to gold JSONL")
    ap.add_argument("--out", default="eval_report.html", help="HTML report path")
    args = ap.parse_args()
    res = evaluate(Path(args.pred_csv), Path(args.gold_jsonl), Path(args.out))
    print(res)
