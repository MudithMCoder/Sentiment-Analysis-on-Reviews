import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.pad_idx = pad_idx

    def forward(self, token_ids):
        # Remove left-padding before packing so padding cannot affect the LSTM state.
        valid = token_ids != self.pad_idx
        lengths = valid.sum(1).clamp(min=1)  # all-pad rows -> one (zero) vector, as before
        # Stable sort moves real tokens to the front in original order (== row[row != pad]).
        order = torch.argsort((~valid).int(), dim=1, stable=True)
        # Mask zeroes pad positions regardless of the <pad> embedding row's contents.
        emb = self.embedding(token_ids.gather(1, order)) * valid.gather(1, order).unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1]).squeeze(1)
