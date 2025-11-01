# app/models.py
from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    util = None

class ModelRegistry:
    """
    - Embeddings: all-MiniLM-L6-v2
    - Reranker: FinBERT cross-encoder if available, else MiniLM cross-encoder,
      else cosine similarity on embeddings as a fallback.
    - Ready for future LoRA (see attach_lora_if_available).
    """
    def __init__(self):
        self.embed_model = None
        self.rerank_model = None
        self.device = os.environ.get("DEVICE", "cpu")

    def load(self):
        # Embedding model (fast, strong baseline)
        if SentenceTransformer is not None:
            self.embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        # Try FinBERT-style cross-encoder first (if user installed one), then common MiniLM CE
        self.rerank_model = None
        if CrossEncoder is not None:
            tried = [
                # Put any FinBERT cross-encoder(s) you have locally/privately first
                # ("<your-finbert-crossencoder>",),
                ("cross-encoder/ms-marco-MiniLM-L-6-v2",),
                ("cross-encoder/ms-marco-MiniLM-L-12-v2",),
            ]
            for (name,) in tried:
                try:
                    self.rerank_model = CrossEncoder(name, device=self.device)
                    break
                except Exception:
                    continue

    def embed(self, sentences: List[str]) -> np.ndarray:
        if self.embed_model is None:
            raise RuntimeError("Embedding model not loaded.")
        return np.asarray(self.embed_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False))

    def rerank(self, query: str, candidates: List[str]) -> List[Tuple[int, float]]:
        """
        Returns list of (index, score) sorted DESC by score.
        """
        if self.rerank_model is not None:
            pairs = [[query, c] for c in candidates]
            scores = self.rerank_model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
            order = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return order

        # Fallback: cosine similarity using embeddings (if CrossEncoder not available)
        if self.embed_model is not None and util is not None:
            q_emb = self.embed([query])[0]
            c_emb = self.embed(candidates)
            sims = util.cos_sim(q_emb, c_emb).cpu().numpy().flatten()
            order = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
            return order

        # Last resort: lexical score (length-weighted)
        q = query.lower()
        scores = []
        for i, s in enumerate(candidates):
            score = sum(1 for tok in q.split(",") if tok.strip() and tok.strip() in s.lower())
            scores.append((i, float(score)))
        return sorted(scores, key=lambda x: x[1], reverse=True)

def attach_lora_if_available(registry: ModelRegistry) -> None:
    """
    Hook for future numeracy LoRA / adapters.
    No-op by default to keep runtime stable.
    """
    # If you have an adapter, load/apply it here.
    return
