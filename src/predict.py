"""Raw-text inference. This is the attack surface: every attack calls predict_proba."""
import json
from pathlib import Path

import numpy as np
import torch

from src.data import Vocab, encode
from src.model import SentimentLSTM
from src.preprocess import clean_text

DEFAULT_ARTIFACTS = "artifacts/v0"
_default = None  # lazily loaded (model, vocab) for DEFAULT_ARTIFACTS


def load_model(artifact_dir: str = DEFAULT_ARTIFACTS, device: str = "cpu"):
    """Rebuild SentimentLSTM from config.json and load model.pt. Returns (model, vocab)."""
    artifact_dir = Path(artifact_dir)
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    vocab = Vocab.load(artifact_dir / "vocab.json")
    model = SentimentLSTM(config["vocab_size"], embedding_dim=config["embedding_dim"],
                          hidden_dim=config["hidden_dim"], pad_idx=config["pad_idx"])
    # weights_only=True: tensors only, no arbitrary pickle code execution on load.
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location=device, weights_only=True))
    return model.to(device).eval(), vocab


def predict_proba(texts: list[str], model=None, vocab=None, batch_size: int = 64) -> np.ndarray:
    """P(positive) for each RAW review: clean_text -> tokenizer -> vocab -> model -> sigmoid."""
    global _default
    if model is None or vocab is None:
        if _default is None:
            _default = load_model()
        model, vocab = _default
    if isinstance(texts, str):
        raise TypeError("predict_proba expects a list of strings, not a single string")

    device = next(model.parameters()).device
    model.eval()
    probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [encode(clean_text(t), vocab) for t in texts[start:start + batch_size]]
            logits = model(torch.tensor(batch, dtype=torch.long, device=device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)
