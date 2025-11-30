#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split


# ------------------------------
#  Sequence CNN Model
# ------------------------------
class SeqCNN(nn.Module):
    def __init__(self, motif_k=10, num_filters=16):
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
#  Evaluation Loop
# ------------------------------
def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
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
    X_seq = data["X_seq"]        # (N, 4, 30)
    y = data["y"]

    print("Loaded dataset:", X_seq.shape, y.shape)

    # Split
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

    # Training
    for epoch in range(1, args.epochs + 1):
        tr_loss = train(model, train_loader, opt, device)
        val_auroc, val_auprc, _ = evaluate(model, X_val, y_val, device)

        print(f"Epoch {epoch:02d} | loss={tr_loss:.4f} | val AUROC={val_auroc:.4f} | val AUPRC={val_auprc:.4f}")

    # Final test
    test_auroc, test_auprc, test_preds = evaluate(model, X_test, y_test, device)
    print("\n=== TEST PERFORMANCE ===")
    print("AUROC:", test_auroc)
    print("AUPRC:", test_auprc)

    # Save model
    torch.save(model.state_dict(), args.out)
    print("Saved model →", args.out)


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
