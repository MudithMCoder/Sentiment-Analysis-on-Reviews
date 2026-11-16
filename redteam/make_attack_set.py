"""Build the frozen attack set: correctly classified v0 test reviews, balanced per class.

Usage: python -m redteam.make_attack_set [--artifacts artifacts/v0] [--out redteam/attack_set.json]
                                         [--n-per-class 250] [--seed 42]
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from src.data import MAX_LEN, load_dataframe, tokenizer
from src.predict import load_model, predict_proba
from src.preprocess import clean_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts/v0")
    parser.add_argument("--data", default="Data/IMDB.csv")
    parser.add_argument("--out", default="redteam/attack_set.json")
    parser.add_argument("--n-per-class", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    artifacts = Path(args.artifacts)

    test_idx = json.loads((artifacts / "splits.json").read_text(encoding="utf-8"))["test"]
    test_df = load_dataframe(args.data, clean=False).loc[test_idx]
    model, vocab = load_model(artifacts)
    probs = predict_proba(test_df["review"].tolist(), model, vocab)
    labels = test_df["sentiment"].to_numpy(dtype=int)
    correct = (probs >= 0.5).astype(int) == labels

    rng = np.random.default_rng(args.seed)
    chosen, n_correct = [], {}
    for label in (0, 1):
        pool = np.flatnonzero(correct & (labels == label))
        n_correct[label] = len(pool)
        if len(pool) < args.n_per_class:
            raise SystemExit(f"Only {len(pool)} correct reviews with label {label}")
        chosen.extend(rng.choice(pool, size=args.n_per_class, replace=False).tolist())
    chosen.sort(key=lambda i: test_idx[i])

    items = []
    for i in chosen:
        raw = test_df["review"].iloc[i]
        items.append({
            "row": int(test_idx[i]),                          # CSV row index
            "label": int(labels[i]),                          # 1 = positive, 0 = negative
            "p_v0": float(probs[i]),                          # v0 P(positive)
            "n_tokens": len(tokenizer(clean_text(raw))),      # pipeline tokens before truncation
            "n_words_raw": len(raw.split()),
        })

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    n_long = sum(it["n_tokens"] > MAX_LEN for it in items)
    out = {
        "model_tag": "v0-baseline",
        "artifacts": str(artifacts),
        "git_commit": commit,
        "seed": args.seed,
        "n_per_class": args.n_per_class,
        "test_size": len(test_idx),
        "n_correct_per_class": {"negative": n_correct[0], "positive": n_correct[1]},
        "n_longer_than_max_len": n_long,
        "max_len": MAX_LEN,
        "items": items,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"Correct on test: negative {n_correct[0]}, positive {n_correct[1]} (of {len(test_idx)})")
    print(f"Attack set: {len(items)} reviews ({args.n_per_class} per class) -> {args.out}")
    print(f"Longer than {MAX_LEN} pipeline tokens: {n_long} / {len(items)}")


if __name__ == "__main__":
    main()
