#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ------------------------------
#  Sequence CNN Model
# ------------------------------
class SeqCNN(nn.Module):
    def __init__(self, motif_k=20, num_filters=16):
        super().__init__()
        self.conv = nn.Conv1d(4, num_filters, kernel_size=motif_k)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x_seq):
        x = F.relu(self.conv(x_seq))
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).view(-1)


# ------------------------------
#  Training Loop
# ------------------------------
def train(model, loader, opt, device):
    model.train()
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).float()

        opt.zero_grad()
        preds = model(xb)
        loss = F.binary_cross_entropy(preds, yb)
        loss.backward()
        opt.step()

        total += loss.item()

    return total / len(loader)


# ------------------------------
#  Evaluation (loss + metrics)
# ------------------------------
def eval_with_loss(model, X, y, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        preds = model(X_t).cpu().numpy()

    loss = F.binary_cross_entropy(
        torch.tensor(preds), torch.tensor(y, dtype=torch.float32)
    ).item()

    auroc = roc_auc_score(y, preds)
    auprc = average_precision_score(y, preds)

    return loss, auroc, auprc, preds


# ------------------------------
#  MAIN
# ------------------------------
def main(args):

    # Load dataset
    data = np.load(args.data)
    X_seq = data["X_seq"]
    y = data["y"]

    print("Loaded dataset:", X_seq.shape, y.shape)

    # Train/val/test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X_seq, y, test_size=0.4, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=args.batch,
        shuffle=True
    )

    model = SeqCNN(motif_k=args.kernel, num_filters=args.filters).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # History dict
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": []
    }

    # Training loop
    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, train_loader, opt, device)
        val_loss, val_auroc, val_auprc, _ = eval_with_loss(model, X_val, y_val, device)

        print(f"Epoch {epoch:02d} | "
              f"train_loss={tr_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_AUROC={val_auroc:.4f} | "
              f"val_AUPRC={val_auprc:.4f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)

    # Final test evaluation
    test_loss, test_auroc, test_auprc, test_preds = eval_with_loss(model, X_test, y_test, device)
    print("\n=== TEST PERFORMANCE ===")
    print("Loss:", test_loss)
    print("AUROC:", test_auroc)
    print("AUPRC:", test_auprc)

    # ------------------------------------------
    # Confusion matrix
    # ------------------------------------------
    y_pred_bin = (test_preds > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred_bin)
    print("\nConfusion Matrix:\n", cm)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix – Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()

    # ------------------------------------------
    # ROC curve
    # ------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, test_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={test_auroc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Test Set")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=200)
    plt.close()

    # ------------------------------------------
    # PR curve
    # ------------------------------------------
    prec, rec, _ = precision_recall_curve(y_test, test_preds)
    plt.figure()
    plt.plot(rec, prec, label=f"AUPRC={test_auprc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve – Test Set")
    plt.legend()
    plt.savefig("pr_curve.png", dpi=200)
    plt.close()

    # ------------------------------------------
    # Loss curves ONLY (proper plot)
    # ------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=200)
    plt.close()

    # Save model + history
    torch.save(model.state_dict(), args.out)
    np.savez("training_history.npz", **history)

    print("\nSaved model →", args.out)
    print("Saved plots → loss_curve.png, roc_curve.png, pr_curve.png, confusion_matrix.png")
    print("Saved history → training_history.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kernel", type=int, default=10)
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--out", type=str, default="seqcnn.pt")
    args = p.parse_args()
    main(args)
