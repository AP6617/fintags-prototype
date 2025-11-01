# app/services/finbert_crossenc.py
from sentence_transformers import CrossEncoder
import os

# Path where your trained model was saved by train_finbert_crossenc.py
MODEL_PATH = "models/finbert_crossenc"

# cached singleton
_model: CrossEncoder | None = None

def get_model() -> CrossEncoder:
    """
    Load the fine-tuned FinBERT cross-encoder once and cache it.
    """
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                f"Fine-tuned FinBERT not found at {MODEL_PATH}. Train it first."
            )
        _model = CrossEncoder(MODEL_PATH)
    return _model

def relevance_score(concept: str, paragraph: str) -> float:
    """
    Return the relevance score (higher = more relevant) from the cross-encoder.
    """
    model = get_model()
    score = model.predict([(concept, paragraph)])[0]
    return float(score)
