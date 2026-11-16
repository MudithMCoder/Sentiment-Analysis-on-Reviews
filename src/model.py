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
        sequences = [self.embedding(row[row != self.pad_idx]) for row in token_ids]
        sequences = [seq if len(seq) else self.embedding.weight.new_zeros((1, self.embedding.embedding_dim))
                     for seq in sequences]
        packed = nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1]).squeeze(1)
