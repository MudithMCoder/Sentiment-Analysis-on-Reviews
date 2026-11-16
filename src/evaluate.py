"""Test-set metrics for a saved model (Pipeline.ipynb cell 12, without plots).

Usage: python -m src.evaluate [--artifacts artifacts/v0] [--data Data/IMDB.csv]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.data import load_dataframe
from src.predict import load_model, predict_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts/v0")
    parser.add_argument("--data", default="Data/IMDB.csv")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)

    test_idx = json.loads((artifacts / "splits.json").read_text(encoding="utf-8"))["test"]
    # Raw reviews: predict_proba cleans them exactly once, like the notebook.
    test_df = load_dataframe(args.data, clean=False).loc[test_idx]

    model, vocab = load_model(artifacts)
    y_prob = predict_proba(test_df["review"].tolist(), model, vocab)
    y_true = test_df["sentiment"].to_numpy(dtype=int)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": cm.tolist(),  # rows=true [neg, pos], cols=predicted
    }

    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=["negative", "positive"], zero_division=0))
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"{k:<10}: {metrics[k]:.4f}")
    print("Confusion matrix (rows=true [negative, positive], columns=predicted):")
    print(np.array(cm))

    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {artifacts / 'metrics.json'}")


if __name__ == "__main__":
    main()
