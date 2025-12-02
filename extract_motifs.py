#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logomaker
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------
#  Model must match training
# ------------------------------
class SeqCNN(nn.Module):
    def __init__(self, motif_k=10, num_filters=16):
        super().__init__()
        self.conv = nn.Conv1d(4, num_filters, kernel_size=motif_k)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x)).view(-1)


# ------------------------------
#  Convert subsequences → PWM
# ------------------------------
def build_pwm(seqs):
    """
    seqs: list of strings, equal length (kernel size)
    """
    L = len(seqs[0])
    counts = np.zeros((4, L))

    map_idx = {'A':0, 'C':1, 'G':2, 'T':3}

    for s in seqs:
        for i, b in enumerate(s):
            if b in map_idx:
                counts[map_idx[b], i] += 1

    pwm = counts / counts.sum(axis=0, keepdims=True)
    pwm[np.isnan(pwm)] = 0.25  # handle empty columns
    return pwm


# ------------------------------
#  Plot PWM using logomaker
# ------------------------------
def save_logo(pwm, out_png):
    df = pd.DataFrame(
        pwm.T,
        columns=['A', 'C', 'G', 'T']
    )
    df_ic = logomaker.transform_matrix(df,
                                       from_type='probability',
                                       to_type='information')

    plt.figure(figsize=(12, 3))
    logomaker.Logo(df_ic,
                   color_scheme='classic',
                   vpad=0.05)
    plt.title("Motif (activation-based PWM)")
    plt.xlabel("Position")
    plt.ylabel("Information (bits)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ------------------------------
#  MAIN
# ------------------------------
def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    data = np.load(args.data)
    X_seq = data["X_seq"]   # (N, 4, 30)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = SeqCNN(motif_k=args.kernel, num_filters=args.filters)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    print("Model loaded.")
    print("Extracting motifs using ACTIVATIONS...")

    X = torch.tensor(X_seq, dtype=torch.float32).to(device)

    # Get conv weights
    conv = model.conv
    num_filters = args.filters
    K = args.kernel

    # Compute convolution activations (before ReLU)
    with torch.no_grad():
        activ = conv(X)  # shape: (N, filters, L_out)

    activ = activ.cpu().numpy()
    X_np = X_seq  # numpy version

    for f in range(num_filters):
        print(f"Processing filter {f}...")

        # Get max activation over sequence for each example
        f_activ = activ[:, f, :]  # (N, L_out)
        max_per_seq = f_activ.max(axis=1)

        # Pick top K sequences
        top_idx = np.argsort(-max_per_seq)[:args.topk]

        subseqs = []
        for idx in top_idx:
            seq_onehot = X_np[idx]   # (4, 30)
            act_map = f_activ[idx]   # length ~ L_out

            pos = act_map.argmax()
            start = pos
            end = pos + K
            if end > seq_onehot.shape[1]:
                continue

            # Convert back to string
            oh = seq_onehot[:, start:end]  # (4, K)
            bases = np.argmax(oh, axis=0)
            inv = {0:'A', 1:'C', 2:'G', 3:'T'}
            seq = "".join(inv[b] for b in bases)
            subseqs.append(seq)

        if len(subseqs) == 0:
            print(f"Filter {f}: No subsequences found (skip).")
            continue

        pwm = build_pwm(subseqs)
        out_png = os.path.join(args.outdir, f"filter_{f}_activation_pwm.png")
        save_logo(pwm, out_png)

        print(f"Saved → {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="motifs/")
    p.add_argument("--kernel", type=int, default=10)
    p.add_argument("--filters", type=int, default=16)
    p.add_argument("--topk", type=int, default=200)

    args = p.parse_args()
    main(args)