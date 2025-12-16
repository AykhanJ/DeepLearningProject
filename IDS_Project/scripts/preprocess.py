from __future__ import annotations
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.nslkdd import CATEGORICAL, read_nslkdd_txt, split_features_labels
from src.utils import ensure_dir, save_json, set_seed

def build_preprocessor(X):
    numeric_cols = [c for c in X.columns if c not in CATEGORICAL]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", StandardScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre, numeric_cols

def save_split(path: Path, X: np.ndarray, y: np.ndarray):
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw", help="Directory with KDDTrain+.txt and KDDTest+.txt")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))

    train_df = read_nslkdd_txt(str(raw_dir / "KDDTrain+.txt"))
    test_df  = read_nslkdd_txt(str(raw_dir / "KDDTest+.txt"))

    X_train_full, y_train_full = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=y_train_full
    )

    pre, numeric_cols = build_preprocessor(X_train_full)
    pre.fit(X_tr)

    X_tr_t = pre.transform(X_tr)
    X_val_t = pre.transform(X_val)
    X_test_t = pre.transform(X_test)

    save_split(out_dir / "train.npz", X_tr_t, y_tr.to_numpy())
    save_split(out_dir / "val.npz", X_val_t, y_val.to_numpy())
    save_split(out_dir / "test.npz", X_test_t, y_test.to_numpy())

    joblib.dump(pre, out_dir / "preprocessor.joblib")

    meta = {
        "categorical_cols": CATEGORICAL,
        "numeric_cols": numeric_cols,
        "train_rows": int(X_tr_t.shape[0]),
        "val_rows": int(X_val_t.shape[0]),
        "test_rows": int(X_test_t.shape[0]),
        "feature_dim": int(X_tr_t.shape[1]),
        "val_size": args.val_size,
        "seed": args.seed
    }
    save_json(str(out_dir / "meta.json"), meta)
    print("Saved processed data to:", out_dir)

if __name__ == "__main__":
    main()
