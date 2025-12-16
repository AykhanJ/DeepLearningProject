from __future__ import annotations
from typing import List, Tuple

import pandas as pd

# 41 features + label + difficulty (per NSL-KDD .txt files)
COLUMNS: List[str] = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
    "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count",
    "serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET_COL = "label"
DROP_COLS = ["difficulty"]

def read_nslkdd_txt(path: str) -> pd.DataFrame:
    """Read NSL-KDD .txt (CSV-like, no header)."""
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df

def make_binary_label(df: pd.DataFrame) -> pd.Series:
    """Map 'normal' to 0 and everything else to 1 (attack). Handles 'normal.' too."""
    y_raw = df[TARGET_COL].astype(str).str.strip()
    y = (y_raw != "normal") & (y_raw != "normal.")
    return y.astype(int)

def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = make_binary_label(df)
    X = df.drop(columns=[TARGET_COL] + DROP_COLS, errors="ignore")
    return X, y
