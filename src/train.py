"""Train the v0 SentimentLSTM (Pipeline.ipynb cell 11) and save reproducible artifacts.

Usage: python -m src.train [--data Data/IMDB.csv] [--out artifacts/v0] [--seed 42]
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import (BATCH_SIZE, MAX_LEN, MAX_TOKENS, RANDOM_STATE, SPECIALS,
                      build_vocab, load_dataframe, make_loaders, split, split_indices)
from src.model import SentimentLSTM
from src.utils import set_seed

# Hyperparameters are fixed on purpose: v0 must match the notebook.
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
LR = 1e-3
EPOCHS = 5


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run("status", "--porcelain")
    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(status) if status is not None else None}


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data/IMDB.csv")
    parser.add_argument("--out", default="artifacts/v0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.data)
    splits = split(df, random_state=RANDOM_STATE)
    vocab = build_vocab(splits["train"][0])
    loaders = make_loaders(splits, vocab, seed=args.seed)
    train_loader, val_loader = loaders["train"], loaders["val"]
    pad_idx = vocab["<pad>"]
    print(f"Train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,} | "
          f"test {len(loaders['test'].dataset):,} | vocab {len(vocab):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentLSTM(len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                          pad_idx=pad_idx).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log, best_val_loss, best_epoch, best_state = [], float("inf"), None, None
    for epoch in range(EPOCHS):
        start = time.perf_counter()
        model.train()
        train_loss = 0.0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                logits = model(texts)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                correct += ((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item()

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / len(val_loader.dataset)
        seconds = time.perf_counter() - start
        log.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss,
                    "val_accuracy": val_accuracy, "seconds": seconds})
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"train loss: {train_loss:.4f} | "
              f"val loss: {val_loss:.4f} | "
              f"val accuracy: {val_accuracy:.4f} | "
              f"time: {seconds:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, out / "model.pt")
    vocab.save(out / "vocab.json")
    write_json(out / "splits.json", split_indices(splits))
    write_json(out / "train_log.json", log)
    write_json(out / "config.json", {
        "model": "SentimentLSTM",
        "vocab_size": len(vocab),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "pad_idx": pad_idx,
        "max_len": MAX_LEN,
        "max_tokens": MAX_TOKENS,
        "specials": SPECIALS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
        "epochs": EPOCHS,
        "split": {"train": 0.70, "val": 0.15, "test": 0.15, "stratified": True,
                  "random_state": RANDOM_STATE},
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "device": str(device),
        "data": args.data,
        **git_info(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
    })
    print(f"Training complete on {device}. Best epoch {best_epoch} "
          f"(val loss {best_val_loss:.4f}) saved to {out / 'model.pt'}")


if __name__ == "__main__":
    main()
