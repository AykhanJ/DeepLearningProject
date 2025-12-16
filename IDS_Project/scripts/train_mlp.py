from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt

from src.utils import load_npz, ensure_dir, save_json, set_seed

class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: list[int], dropout: float):
        super().__init__()
        layers = []
        prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def metrics_from_logits(y_true: np.ndarray, logits: np.ndarray) -> dict:
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "confusion_matrix": cm}

@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits, all_y = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits)
    y = np.concatenate(all_y)
    return y, logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", default="256,128,64", help="Comma-separated hidden sizes")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--imbalance", choices=["none", "pos_weight"], default="pos_weight")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.results_dir)

    train = load_npz(str(Path(args.data_dir) / "train.npz"))
    val = load_npz(str(Path(args.data_dir) / "val.npz"))
    test = load_npz(str(Path(args.data_dir) / "test.npz"))

    X_tr, y_tr = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_te, y_te = test["X"], test["y"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_tr_t = torch.from_numpy(X_tr)
    y_tr_t = torch.from_numpy(y_tr).float()
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val).float()
    X_te_t = torch.from_numpy(X_te)
    y_te_t = torch.from_numpy(y_te).float()

    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=args.batch_size, shuffle=False)

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    model = MLP(d_in=X_tr.shape[1], hidden=hidden, dropout=args.dropout).to(device)

    # imbalance handling via pos_weight in BCEWithLogitsLoss
    if args.imbalance == "pos_weight":
        n_pos = float((y_tr == 1).sum())
        n_neg = float((y_tr == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    else:
        pos_weight = None

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_f1": []}
    best_f1 = -1.0
    best_path = Path(args.results_dir) / "mlp_best.pt"
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = total_loss / max(n, 1)
        y_val_np, val_logits = eval_model(model, val_loader, device=device)
        val_metrics = metrics_from_logits(y_val_np, val_logits)
        val_f1 = val_metrics["f1"]

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "config": vars(args)}, best_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    # Load best for final eval
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    y_val_np, val_logits = eval_model(model, val_loader, device=device)
    y_te_np, te_logits = eval_model(model, te_loader, device=device)

    out = {
        "model": "mlp",
        "imbalance": args.imbalance,
        "seed": args.seed,
        "hidden": hidden,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs_ran": len(history["train_loss"]),
        "val": metrics_from_logits(y_val_np, val_logits),
        "test": metrics_from_logits(y_te_np, te_logits),
    }

    save_json(str(Path(args.results_dir) / f"mlp_{args.imbalance}_metrics.json"), out)
    save_json(str(Path(args.results_dir) / "mlp_history.json"), history)

    # training curves figure
    fig_path = Path(args.results_dir) / "mlp_training_curve.pdf"
    plt.figure()
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train loss")
    plt.plot(range(1, len(history["val_f1"]) + 1), history["val_f1"], label="val F1")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print("Saved:", Path(args.results_dir) / f"mlp_{args.imbalance}_metrics.json")
    print("Saved:", fig_path)

if __name__ == "__main__":
    main()
