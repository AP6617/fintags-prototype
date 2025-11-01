from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from tqdm import tqdm
import json, random, os

# paths
DATA_FILE = "exports/finbert_train.jsonl"
SAVE_DIR = "models/finbert_crossenc"

# load data
samples = []
for line in open(DATA_FILE, encoding="utf-8"):
    j = json.loads(line)
    samples.append(InputExample(texts=[j["concept"], j["paragraph"]], label=float(j["label"])))

random.shuffle(samples)
train_loader = DataLoader(samples, batch_size=8, shuffle=True)

# load base FinBERT
model = CrossEncoder("yiyanghkust/finbert-pretrain", num_labels=1)

# train
print("Fine-tuning FinBERT on", len(samples), "pairs …")
model.fit(train_dataloader=train_loader, epochs=1, warmup_steps=100, optimizer_params={'lr':2e-5})

# save
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(SAVE_DIR)
print("✅ saved fine-tuned model to", SAVE_DIR)
