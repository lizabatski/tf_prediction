import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# PWM LOADING + SCORING
# ============================================================

def load_pwm(path, tf_name="CTCF"):
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            if parts[0].upper() != tf_name.upper():
                continue

            L = int(parts[1])
            A = [float(x) for x in parts[2].rstrip(",").split(",")]
            C = [float(x) for x in parts[3].rstrip(",").split(",")]
            G = [float(x) for x in parts[4].rstrip(",").split(",")]
            T = [float(x) for x in parts[5].rstrip(",").split(",")]

            pwm = np.array([A, C, G, T], dtype=float)
            assert pwm.shape == (4, L)
            return pwm

    raise ValueError(f"PWM for TF {tf_name} not found in {path}")


def score_pwm(onehot, pwm):
    L_seq = onehot.shape[1]
    L_pwm = pwm.shape[1]

    best = -np.inf
    for i in range(L_seq - L_pwm + 1):
        window = onehot[:, i:i+L_pwm]
        score = np.sum(window * pwm)
        if score > best:
            best = score
    return best


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_logreg(data_path, pwm_path, tf_name, mode):
    print("\n========================")
    print(f"Training mode: {mode}")
    print("========================\n")

    # ---------------------------------------------------
    # Load dataset
    # ---------------------------------------------------
    data = np.load(data_path)
    X_seq = data["X_seq"]             # (N,4,L)
    X_struct = data["X_struct"]       # (N,10)
    y = data["y"]

    print("Loaded dataset:")
    print("  X_seq shape:   ", X_seq.shape)
    print("  X_struct shape:", X_struct.shape)
    print("  y shape:        ", y.shape)

    # ---------------------------------------------------
    # PWM score computation (only if required)
    # ---------------------------------------------------
    if mode in ("pwm_only", "struct_pwm"):
        print("\nLoading PWM…")
        pwm = load_pwm(pwm_path, tf_name=tf_name)
        print("Computing PWM scores…")
        pwm_scores = np.array([score_pwm(x, pwm) for x in X_seq]).reshape(-1, 1)
        print("PWM scores shape:", pwm_scores.shape)

    # ---------------------------------------------------
    # Feature assembly
    # ---------------------------------------------------
    if mode == "struct_only":
        X = X_struct
        feature_names = [
            "Buckle_mean", "Buckle_std",
            "MGW_mean", "MGW_std",
            "Opening_mean", "Opening_std",
            "ProT_mean", "ProT_std",
            "Roll_mean", "Roll_std",
        ]

    elif mode == "pwm_only":
        X = pwm_scores
        feature_names = ["PWM_score"]

    elif mode == "struct_pwm":
        X = np.concatenate([X_struct, pwm_scores], axis=1)
        feature_names = [
            "Buckle_mean", "Buckle_std",
            "MGW_mean", "MGW_std",
            "Opening_mean", "Opening_std",
            "ProT_mean", "ProT_std",
            "Roll_mean", "Roll_std",
            "PWM_score"
        ]

    print("Final X shape:", X.shape)

    # ---------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    aucs = []
    auprcs = []

    fold = 1

    for train_idx, test_idx in kf.split(X):
        print(f"\n===== Fold {fold} =====")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # impute missing
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # train LR
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        # evaluate
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob > 0.5).astype(int)

        acc = accuracy_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        auprc = average_precision_score(y_test, prob)

        accs.append(acc)
        aucs.append(auc)
        auprcs.append(auprc)

        print(f"Accuracy: {acc:.4f}")
        print(f"AUROC:    {auc:.4f}")
        print(f"AUPRC:    {auprc:.4f}")

        # ---------- SAVE CONFUSION MATRIX PNG ----------
        cm = confusion_matrix(y_test, pred)

        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap="Blues")
        plt.title(f"Confusion Matrix – Fold {fold} – {mode}")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Add text inside the squares
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]),
                        ha="center", va="center", color="black", fontsize=12)

        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"confusion_fold{fold}_{mode}.png")
        plt.close()

        # save ROC curve
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Fold {fold} – {mode}")
        plt.tight_layout()
        plt.savefig(f"roc_fold{fold}_{mode}.png")
        plt.close()

        # save PR curve
        prec, rec, _ = precision_recall_curve(y_test, prob)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Fold {fold} – {mode}")
        plt.tight_layout()
        plt.savefig(f"pr_fold{fold}_{mode}.png")
        plt.close()

        fold += 1

    # ---------------------------------------------------
    # Summary
    # ---------------------------------------------------
    print("\n===== CV SUMMARY =====")
    print(f"Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}")
    print(f"AUROC:    mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}")
    print(f"AUPRC:    mean={np.mean(auprcs):.4f}, std={np.std(auprcs):.4f}")

    # ---------------------------------------------------
    # Train final model
    # ---------------------------------------------------
    print("\nTraining final model on full dataset…")

    final_imputer = SimpleImputer(strategy="mean")
    X_imp = final_imputer.fit_transform(X)

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_imp)

    final_model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
    final_model.fit(X_scaled, y)

    # ---------- FINAL CONFUSION MATRIX ----------
    final_prob = final_model.predict_proba(X_scaled)[:, 1]
    final_pred = (final_prob > 0.5).astype(int)
    final_cm = confusion_matrix(y, final_pred)

    plt.figure(figsize=(4,4))
    plt.imshow(final_cm, cmap="Blues")
    plt.title(f"Confusion Matrix – Final Model – {mode}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add text to squares
    for i in range(final_cm.shape[0]):
        for j in range(final_cm.shape[1]):
            plt.text(j, i, str(final_cm[i, j]),
                    ha="center", va="center", color="black", fontsize=12)

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"confusion_final_{mode}.png")
    plt.close()


    # save artifacts
    out_prefix = f"logreg_{mode}_{tf_name}"

    pickle.dump(final_model, open(out_prefix + "_model.pkl", "wb"))
    pickle.dump(final_scaler, open(out_prefix + "_scaler.pkl", "wb"))
    pickle.dump(final_imputer, open(out_prefix + "_imputer.pkl", "wb"))

    print(f"\nSaved model:   {out_prefix}_model.pkl")
    print(f"Saved scaler:  {out_prefix}_scaler.pkl")
    print(f"Saved imputer: {out_prefix}_imputer.pkl")

    # print feature coefficients
    print("\n===== Feature Coefficients =====")
    for name, coef in zip(feature_names, final_model.coef_[0]):
        print(f"{name:18s}: {coef:+.4f}")


# ============================================================
# CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument("--pwm", required=False, default="data/factorbook/factorbookMotifPwm.txt")
    parser.add_argument("--tf", required=False, default="CTCF", help="TF name for PWM")
    parser.add_argument("--mode", required=False,
                        choices=["struct_only", "pwm_only", "struct_pwm"],
                        default="struct_pwm")

    args = parser.parse_args()

    train_logreg(
        data_path=args.data,
        pwm_path=args.pwm,
        tf_name=args.tf,
        mode=args.mode
    )
