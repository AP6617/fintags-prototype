# training/train_lora_classifier.py  (compat version for older transformers)
import os, json, torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

# --------- Inputs ---------
train_csv = "fintags_train.csv"
labels_json = "labels.json"
base_model_name = "distilbert-base-uncased"

# --------- Load labels & data ---------
pairs = json.load(open(labels_json, "r", encoding="utf-8"))["pairs"]
if not pairs:
    raise ValueError("labels.json has no 'pairs'. Build it with build_fintags_train.py first.")

label2id = {f"{c}|||{t}": i for i, (c, t) in enumerate(pairs)}

df = pd.read_csv(train_csv).fillna("")
if df.empty:
    raise ValueError("No rows in fintags_train.csv — confirm your CSVs got generated.")

def lbl(row):
    key = f"{row['concept']}|||{row['trend']}"
    return label2id.get(key, 0)

df["label"] = df.apply(lbl, axis=1)
df = df[["sentence", "label"]].dropna().drop_duplicates()
if len(df) < 20:
    print(f"[warn] Very small dataset ({len(df)} rows). Training will still run but results may be weak.")

tok = AutoTokenizer.from_pretrained(base_model_name)

def tok_map(batch):
    return tok(batch["sentence"], truncation=True, padding="max_length", max_length=256)

ds = Dataset.from_pandas(df, preserve_index=False).map(tok_map, batched=True)
# Older API: do a simple split ourselves
split = int(max(1, round(0.1 * len(ds))))
eval_ds = ds.select(range(split)) if split < len(ds) else ds.select([])
train_ds = ds.select(range(split, len(ds))) if split < len(ds) else ds

# --------- Base model ---------
base = AutoModelForSequenceClassification.from_pretrained(
    base_model_name, num_labels=len(pairs)
)

# --------- LoRA target modules for DistilBERT ---------
DEFAULT_TARGETS = ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
present = set()
for name, _ in base.named_modules():
    for t in DEFAULT_TARGETS:
        if t in name:
            present.add(t)
if not present:
    present = set(DEFAULT_TARGETS)
print(f"[info] Using LoRA target modules: {sorted(present)}")

peft_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=sorted(present),
)

model = get_peft_model(base, peft_cfg)

# --------- Training args (compatible) ---------
use_fp16 = torch.cuda.is_available()
args = TrainingArguments(
    output_dir="./lora_out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=3e-5,
    logging_steps=50,
    fp16=use_fp16,
    save_steps=500,         # older API
    eval_steps=500,         # older API; we’ll call evaluate() manually too
    save_total_limit=1,
)

def compute_metrics(pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {"acc": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, average="macro"))}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds if len(eval_ds) else None,
    tokenizer=tok,
    compute_metrics=compute_metrics if len(eval_ds) else None,
)

trainer.train()

metrics = {}
if len(eval_ds):
    metrics = trainer.evaluate()
    print("[eval]", metrics)

# --------- Save final LoRA adapter ---------
SAVE_DIR = "../models/lora_classifier"
os.makedirs(SAVE_DIR, exist_ok=True)
trainer.save_model(SAVE_DIR)
with open(os.path.join(SAVE_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump({"pairs": pairs}, f, indent=2)
print(f"[ok] Saved LoRA model + labels.json to {SAVE_DIR}")
