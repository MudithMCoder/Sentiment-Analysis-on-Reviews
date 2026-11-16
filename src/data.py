"""Data loading, split, tokenization, vocabulary and batching (Pipeline.ipynb cells 1, 8, 9, 10)."""
import json
from collections import Counter
from functools import partial

import pandas as pd
import spacy
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from src.preprocess import clean_text, label_map

MAX_LEN    = 300
BATCH_SIZE = 64
MAX_TOKENS = 15_000
SPECIALS   = ["<pad>", "<unk>"]
RANDOM_STATE = 42


#  LOAD + CLEAN

def load_dataframe(csv_path: str = "Data/IMDB.csv", clean: bool = True) -> pd.DataFrame:
    """Read CSV, drop duplicate reviews, encode labels, clean text. Index = original CSV row."""
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['review'])
    df["sentiment"] = df["sentiment"].map(label_map)
    if clean:
        df["review"] = df["review"].apply(clean_text)
    return df


#  TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15)

def split(df: pd.DataFrame, random_state: int = RANDOM_STATE):
    X = df["review"]
    y = df["sentiment"]

    # Step 1: Carve out 70% train, 30% temp (val + test combined)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=random_state,
        stratify=y          # preserves 50/50 class balance in every split
    )

    # Split the 30% temp evenly → 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}


def split_indices(splits) -> dict[str, list[int]]:
    """Original CSV row indices per split, so later phases reuse identical sets."""
    return {name: [int(i) for i in X.index] for name, (X, _) in splits.items()}


#  TOKENIZATION

# Load spaCy's small English model.
nlp = spacy.load(
    "en_core_web_sm",
    disable=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler"]
)

def tokenizer(text: str) -> list[str]:
    """
    Tokenize a cleaned review string using spaCy's rule-based tokenizer.
    """
    return [token.text for token in nlp.tokenizer(text)]


#  VOCABULARY CONSTRUCTION

class Vocab:
    """
    Lightweight vocabulary that mirrors TorchText's Vocab interface so that
    nothing downstream needs to change:
    """
    def __init__(self, token_to_idx: dict[str, int]):
        self._token_to_idx  = token_to_idx
        self._default_index = 0           # overridden by set_default_index

    def __getitem__(self, token: str) -> int:
        return self._token_to_idx.get(token, self._default_index)

    def __call__(self, tokens: list[str]) -> list[int]:
        return [self[t] for t in tokens]

    def __len__(self) -> int:
        return len(self._token_to_idx)

    def set_default_index(self, index: int):
        self._default_index = index

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._token_to_idx, f)

    @classmethod
    def load(cls, path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            vocab = cls(json.load(f))
        vocab.set_default_index(vocab["<unk>"])
        return vocab


def build_vocab_from_texts(
    texts,
    tokenizer_fn,
    max_tokens: int     = 15_000,
    specials: list[str] = ["<pad>", "<unk>"]
) -> Vocab:
    """
    Build a Vocab by counting token frequencies across all texts and
    retaining only the most frequent ones.

    """
    counter = Counter()
    for text in texts:
        counter.update(tokenizer_fn(text))

    # Reserve slots for special tokens
    n_regular       = max_tokens - len(specials)
    frequent_tokens = [tok for tok, _ in counter.most_common(n_regular)]

    token_to_idx: dict[str, int] = {}
    for tok in specials:           # guarantees <pad>=0, <unk>=1
        token_to_idx[tok] = len(token_to_idx)
    for tok in frequent_tokens:
        if tok not in token_to_idx:   # guards against a special appearing in text
            token_to_idx[tok] = len(token_to_idx)

    return Vocab(token_to_idx)


def build_vocab(X_train) -> Vocab:
    vocab = build_vocab_from_texts(X_train, tokenizer, max_tokens=MAX_TOKENS, specials=SPECIALS)
    # Without this line, any OOV token raises a KeyError at inference time.
    vocab.set_default_index(vocab["<unk>"])
    return vocab


#  NUMERICALIZATION

def text_pipeline(text: str, vocab: Vocab) -> list[int]:
    """Tokenize a cleaned string and convert tokens to vocab indices."""
    return vocab(tokenizer(text))


def encode(text: str, vocab: Vocab) -> list[int]:
    """Numericalize, keep the first MAX_LEN tokens, left-pad shorter sequences."""
    indices = text_pipeline(text, vocab)
    if len(indices) >= MAX_LEN:
        indices = indices[:MAX_LEN]
    else:
        pad_size = MAX_LEN - len(indices)
        indices  = [vocab["<pad>"]] * pad_size + indices
    return indices


#  DYNAMIC MINI-BATCHING WITH PADDING

class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts  = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch: list[tuple], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    texts, labels    = zip(*batch)
    padded_sequences = [encode(text, vocab) for text in texts]

    text_tensor  = torch.tensor(padded_sequences, dtype=torch.long)
    label_tensor = torch.tensor(labels,           dtype=torch.float)
    return text_tensor, label_tensor


def make_loaders(splits, vocab: Vocab, seed: int):
    collate = partial(collate_fn, vocab=vocab)
    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(IMDBDataset(X, y), batch_size=BATCH_SIZE, shuffle=(name == "train"),
                         collate_fn=collate, generator=generator if name == "train" else None)
        for name, (X, y) in splits.items()
    }
