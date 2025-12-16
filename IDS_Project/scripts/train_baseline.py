from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

from src.utils import load_npz, ensure_dir, save_json, set_seed

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "confusion_matrix": cm}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--model", choices=["logreg"], default="logreg")
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.results_dir)

    train = load_npz(str(Path(args.data_dir) / "train.npz"))
    val = load_npz(str(Path(args.data_dir) / "val.npz"))
    test = load_npz(str(Path(args.data_dir) / "test.npz"))

    X_tr, y_tr = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_te, y_te = test["X"], test["y"]

    cw = None if args.class_weight == "none" else "balanced"

    clf = LogisticRegression(
        max_iter=500,
        solver="saga",
        n_jobs=-1,
        class_weight=cw,
    )

    clf.fit(X_tr, y_tr)

    def eval_split(name, X, y):
        pred = clf.predict(X)
        metrics = compute_metrics(y, pred)
        metrics["split"] = name
        return metrics

    out = {
        "model": "logreg",
        "class_weight": args.class_weight,
        "seed": args.seed,
        "val": eval_split("val", X_val, y_val),
        "test": eval_split("test", X_te, y_te),
    }

    tag = f"baseline_{args.model}_{args.class_weight}"
    save_json(str(Path(args.results_dir) / f"{tag}_metrics.json"), out)
    print("Saved:", Path(args.results_dir) / f"{tag}_metrics.json")

if __name__ == "__main__":
    main()
