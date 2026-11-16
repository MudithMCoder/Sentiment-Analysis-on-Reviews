"""The vectorized SentimentLSTM.forward must match the original per-row forward on the v0 checkpoint."""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ARTIFACTS = Path("artifacts/v0")
pytestmark = pytest.mark.skipif(not (ARTIFACTS / "model.pt").exists(),
                                reason="run `python -m src.train` first")
TOL = 1e-5


def old_forward(self, token_ids):
    # Verbatim copy of the original SentimentLSTM.forward (reference implementation).
    # Remove left-padding before packing so padding cannot affect the LSTM state.
    sequences = [self.embedding(row[row != self.pad_idx]) for row in token_ids]
    sequences = [seq if len(seq) else self.embedding.weight.new_zeros((1, self.embedding.embedding_dim))
                 for seq in sequences]
    packed = nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
    _, (hidden, _) = self.lstm(packed)
    return self.classifier(hidden[-1]).squeeze(1)


@pytest.fixture(scope="module")
def model_vocab():
    from src.predict import load_model
    return load_model(ARTIFACTS)


def max_diff(model, token_ids):
    with torch.no_grad():
        return (old_forward(model, token_ids) - model(token_ids)).abs().max().item()


def left_padded(lengths, vocab_size, max_len, gen):
    rows = torch.zeros(len(lengths), max_len, dtype=torch.long)
    for i, n in enumerate(lengths):
        if n:
            rows[i, max_len - n:] = torch.randint(1, vocab_size, (n,), generator=gen)  # 1 = <unk>
    return rows


def test_random_batches_match(model_vocab):
    from src.data import MAX_LEN
    model, vocab = model_vocab
    gen = torch.Generator().manual_seed(0)
    for _ in range(5):
        lengths = [0, 1, MAX_LEN] + torch.randint(0, MAX_LEN + 1, (61,), generator=gen).tolist()
        assert max_diff(model, left_padded(lengths, len(vocab), MAX_LEN, gen)) < TOL


def test_real_test_reviews_match(model_vocab):
    from src.data import encode, load_dataframe
    from src.preprocess import clean_text
    model, vocab = model_vocab
    test_idx = json.loads((ARTIFACTS / "splits.json").read_text(encoding="utf-8"))["test"][:64]
    reviews = load_dataframe(clean=False).loc[test_idx, "review"]
    token_ids = torch.tensor([encode(clean_text(r), vocab) for r in reviews])
    assert max_diff(model, token_ids) < TOL
