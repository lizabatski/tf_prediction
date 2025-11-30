 #!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier


# ------------------------------------------------------------
#  PWM loading and scoring (same as logistic regression script)
# ------------------------------------------------------------
def load_pwm(path, target_tf):
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] != target_tf:
                continue

            L = int(parts[1])

            A = [float(x) for x in parts[2].strip().rstrip(",").split(",")]
            C = [float(x) for x in parts[3].strip().rstrip(",").split(",")]
            G = [float(x) for x in parts[4].strip().rstrip(",").split(",")]
            T = [float(x) for x in parts[5].strip().rstrip(",").split(",")]

            pwm = np.array([A, C, G, T], dtype=float)
            assert pwm.shape == (4, L)
            return pwm

    raise ValueError(f"PWM for TF {target_tf} not found in {path}")


def score_pwm(onehot, pwm):
    L_seq = onehot.shape[1]
    L_pwm = pwm.shape[1]

    max_score = -np.inf
    for start in range(L_seq - L_pwm + 1):
        score = np.sum(onehot[:, start:start + L_pwm] * pwm)
        max_score = max(max_score, score)

    return max_score


# ------------------------------------------------------------
#  MAIN TRAIN FUNCTION
# ------------------------------------------------------------
def train_xgboost(feature_set, data_path, pwm_path, tf_name):
    print("\n=======================================")
    print(f"Training XGBoost model ({feature_set})")
    print("=======================================\n")

    # -------------------------------
    # Load dataset
    # -------------------------------
    data = np.load(data_path)
    X_struct = data["X_struct"]
    X_seq = data["X_seq"]
    y = data["y"]

    print("Dataset loaded.")
    print("  X_struct:", X_struct.shape)
    print("  X_seq:", X_seq.shape)
    print("  y:", y.shape)

    # -------------------------------
    # PWM scoring (if needed)
    # -------------------------------
    if feature_set in ["pwm_only", "struct_pwm"]:
        print(f"\nLoading PWM for TF = {tf_name}")
        pwm = load_pwm(pwm_path, tf_name)

        print("Computing PWM scores...")
        pwm_scores = np.array([score_pwm(X_seq[i], pwm) for i in range(len(X_seq))])
        pwm_scores = pwm_scores.reshape(-1, 1)
        print("  pwm_scores:", pwm_scores.shape)

    # -------------------------------
    # Build feature matrix
    # -------------------------------
    if feature_set == "struct_only":
        X = X_struct
        feature_names = [f"shape_{i}" for i in range(X_struct.shape[1])]
        model_name = "xgb_struct_only.json"
        imputer_name = "imputer_struct_only.pkl"
        scaler_name = "scaler_struct_only.pkl"

    elif feature_set == "pwm_only":
        X = pwm_scores
        feature_names = ["pwm_score"]
        model_name = "xgb_pwm_only.json"
        imputer_name = "imputer_pwm_only.pkl"
        scaler_name = "scaler_pwm_only.pkl"

    elif feature_set == "struct_pwm":
        X = np.concatenate([X_struct, pwm_scores], axis=1)
        feature_names = [
            *[f"shape_{i}" for i in range(X_struct.shape[1])],
            "pwm_score"
        ]
        model_name = "xgb_struct_pwm.json"
        imputer_name = "imputer_struct_pwm.pkl"
        scaler_name = "scaler_struct_pwm.pkl"

    print("\nFinal feature matrix:", X.shape)

    # -------------------------------
    # 5-fold CV
    # -------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accs, aurocs, auprcs = [], [], []

    fold = 1
    for train_idx, test_idx in kf.split(X):
        print(f"\n===== Fold {fold} =====")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -------------------------------
        # Train XGBoost
        # -------------------------------
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)

        accs.append(acc)
        aurocs.append(auroc)
        auprcs.append(auprc)

        print(f"Fold {fold} Accuracy: {acc:.4f}")
        print(f"Fold {fold} AUROC:    {auroc:.4f}")
        print(f"Fold {fold} AUPRC:    {auprc:.4f}")

        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auroc:.3f}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"XGBoost ROC – Fold {fold}")
        plt.legend()
        plt.savefig(f"xgb_roc_fold{fold}_{feature_set}.png", dpi=200)
        plt.close()

        # PR plot
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(rec, prec, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"XGBoost PR – Fold {fold}")
        plt.legend()
        plt.savefig(f"xgb_pr_fold{fold}_{feature_set}.png", dpi=200)
        plt.close()

        fold += 1

    # -------------------------------
    # Summary
    # -------------------------------
    print("\n===== 5-FOLD SUMMARY =====")
    print(f"Accuracy: mean={np.mean(accs):.4f}, std={np.std(accs):.4f}")
    print(f"AUROC:    mean={np.mean(aurocs):.4f}, std={np.std(aurocs):.4f}")
    print(f"AUPRC:    mean={np.mean(auprcs):.4f}, std={np.std(auprcs):.4f}")

    # -------------------------------
    # Train final model on all data
    # -------------------------------
    print("\nTraining final XGBoost model on full dataset...")

    final_imputer = SimpleImputer(strategy="mean")
    X_imputed = final_imputer.fit_transform(X)

    final_scaler = StandardScaler()
    X_scaled = final_scaler.fit_transform(X_imputed)

    final_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42
    )

    final_model.fit(X_scaled, y)

    print("\nComputing final confusion matrix...")

    from sklearn.metrics import confusion_matrix

    y_prob_final = final_model.predict_proba(X_scaled)[:, 1]
    y_pred_final = (y_prob_final > 0.5).astype(int)

    cm = confusion_matrix(y, y_pred_final)
    tn, fp, fn, tp = cm.ravel()

    print("\nFinal confusion matrix:")
    print(cm)

    # ----- Clean Figure -----
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow([[0, 0], [0, 0]], cmap="Blues")  # Blank heatmap background

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["0", "1"], fontsize=12)
    ax.set_yticklabels(["0", "1"], fontsize=12)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix – Final Model – {feature_set}", fontsize=14)

    # Annotate counts
    ax.text(0, 0, str(tn), ha="center", va="center", fontsize=16)
    ax.text(1, 0, str(fp), ha="center", va="center", fontsize=16)
    ax.text(0, 1, str(fn), ha="center", va="center", fontsize=16)
    ax.text(1, 1, str(tp), ha="center", va="center", fontsize=16)

    plt.tight_layout()
    plt.savefig(f"xgb_confusion_final_{feature_set}.png", dpi=300)
    plt.close()

    print(f"Saved clean confusion matrix → xgb_confusion_final_{feature_set}.png")

    # Save
    final_model.save_model(model_name)
    pickle.dump(final_imputer, open(imputer_name, "wb"))
    pickle.dump(final_scaler, open(scaler_name, "wb"))

    print(f"\nSaved model → {model_name}")
    print(f"Saved imputer → {imputer_name}")
    print(f"Saved scaler → {scaler_name}")

    print("\n=== Feature Importances ===")
    for name, imp in zip(feature_names, final_model.feature_importances_):
        print(f"{name:20s}: {imp:.4f}")


# ------------------------------------------------------------
#  ENTRY POINT
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode",
                        type=str,
                        choices=["struct_only", "pwm_only", "struct_pwm"],
                        default="struct_pwm")

    parser.add_argument("--data",
                        type=str,
                        required=True,
                        help="Path to dataset .npz file")

    parser.add_argument("--pwm",
                        type=str,
                        required=True,
                        help="Path to PWM txt file")

    parser.add_argument("--tf",
                        type=str,
                        required=True,
                        help="TF name for PWM scanning")

    args = parser.parse_args()

    train_xgboost(args.mode, args.data, args.pwm, args.tf)
