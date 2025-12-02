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
#  Fusion CNN (Seq + Struct)
# ------------------------------
class FusionCNN(nn.Module):
    def __init__(self, seq_kernel=10, seq_filters=16, struct_filters=8):
        super().__init__()

        # Sequence branch
        self.seq_conv = nn.Conv1d(4, seq_filters, kernel_size=seq_kernel)
        self.seq_pool = nn.AdaptiveMaxPool1d(1)

        # Structural branch
        self.struct_conv = nn.Conv1d(1, struct_filters, kernel_size=3, padding=1)
        self.struct_pool = nn.AdaptiveMaxPool1d(1)

        # Fusion-MLP
        self.fc = nn.Sequential(
            nn.Linear(seq_filters + struct_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq, x_struct):
        # Sequence CNN
        s = F.relu(self.seq_conv(x_seq))
        s = self.seq_pool(s).squeeze(-1)

        # Structural CNN
        st = F.relu(self.struct_conv(x_struct))
        st = self.struct_pool(st).squeeze(-1)

        # Concatenate
        x = torch.cat([s, st], dim=1)
        return torch.sigmoid(self.fc(x)).view(-1)


# ------------------------------
#  EVALUATION
# ------------------------------
def evaluate(model, Xs, Xst, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(
            torch.tensor(Xs, dtype=torch.float32).to(device),
            torch.tensor(Xst, dtype=torch.float32).to(device)
        ).cpu().numpy()

    return (
        roc_auc_score(y, preds),
        average_precision_score(y, preds),
        preds
    )


# ------------------------------
#  MAIN
# ------------------------------
def main(args):

    # Load dataset
    data = np.load(args.data)
    X_seq = data["X_seq"]
    X_struct = data["X_struct"]
    y = data["y"]

    # Reshape structures → (N, 1, S)
    X_struct = X_struct[:, None, :]

    print("Loaded:", X_seq.shape, X_struct.shape, y.shape)

    # Split
    Xs_tr, Xs_tmp, Xst_tr, Xst_tmp, y_tr, y_tmp = train_test_split(
        X_seq, X_struct, y, test_size=0.4, stratify=y, random_state=42
    )
    Xs_val, Xs_te, Xst_val, Xst_te, y_val, y_te = train_test_split(
        Xs_tmp, Xst_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xs_tr, dtype=torch.float32),
            torch.tensor(Xst_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        ),
        batch_size=args.batch,
        shuffle=True
    )

    # Model
    model = FusionCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # History
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    # ------------------------------
    # Training Loop
    # ------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0

        for xb, xstb, yb in train_loader:
            xb = xb.to(device)
            xstb = xstb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            preds = model(xb, xstb)
            loss = F.binary_cross_entropy(preds, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # Validation metrics
        val_auroc, val_auprc, val_preds = evaluate(model, Xs_val, Xst_val, y_val, device)
        val_loss = F.binary_cross_entropy(
            torch.tensor(val_preds, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ).item()

        print(f"Epoch {epoch:02d} | loss={total_loss/len(train_loader):.4f} "
              f"| val loss={val_loss:.4f} | val AUROC={val_auroc:.4f} | val AUPRC={val_auprc:.4f}")

        # Save to history
        history["epoch"].append(epoch)
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)

    # ------------------------------
    # Test Evaluation
    # ------------------------------
    test_auroc, test_auprc, test_preds = evaluate(model, Xs_te, Xst_te, y_te, device)

    print("\n=== TEST PERFORMANCE ===")
    print("AUROC:", test_auroc)
    print("AUPRC:", test_auprc)

    # ------------------------------
    # Confusion Matrix
    # ------------------------------
    from sklearn.metrics import confusion_matrix
    y_pred_bin = (test_preds > 0.5).astype(int)
    cm = confusion_matrix(y_te, y_pred_bin)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title("Confusion Matrix – Test")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig("fusion_confusion_matrix.png", dpi=200)
    plt.close()

    # ------------------------------
    # ROC Curve
    # ------------------------------
    fpr, tpr, _ = roc_curve(y_te, test_preds)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve (AUROC={test_auroc:.3f})")
    plt.tight_layout()
    plt.savefig("fusion_roc_curve.png", dpi=200)
    plt.close()

    # ------------------------------
    # PR Curve
    # ------------------------------
    prec, rec, _ = precision_recall_curve(y_te, test_preds)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve (AUPRC={test_auprc:.3f})")
    plt.tight_layout()
    plt.savefig("fusion_pr_curve.png", dpi=200)
    plt.close()

    # ------------------------------
    # Training Curves
    # ------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    #plt.plot(history["epoch"], history["val_auroc"], label="Val AUROC")
    #plt.plot(history["epoch"], history["val_auprc"], label="Val AUPRC")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.title("Training Curves – Fusion CNN")
    plt.tight_layout()
    plt.savefig("fusion_training_curves.png", dpi=200)
    plt.close()

    # ------------------------------
    # Save model + history
    # ------------------------------
    torch.save(model.state_dict(), args.out)
    print("Saved model →", args.out)

    np.savez("fusion_training_history.npz",
             epoch=np.array(history["epoch"]),
             train_loss=np.array(history["train_loss"]),
             val_loss=np.array(history["val_loss"]),
             val_auroc=np.array(history["val_auroc"]),
             val_auprc=np.array(history["val_auprc"]))

    print("Saved history → fusion_training_history.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="fusioncnn.pt")
    args = p.parse_args()
    main(args)
