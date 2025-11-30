import matplotlib.pyplot as plt
import numpy as np

def plot_clean_confusion_matrix(tn, fp, fn, tp, title, savepath):
    """
    Draw a clean confusion matrix without the weird -0.5 axis range.
    """

    # Matrix in normal layout
    cm = np.array([[tn, fp],
                   [fn, tp]])

    classes = ["0", "1"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")

    # Title
    plt.title(title, fontsize=14)

    # Axis labels
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)

    # Tick labels
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)

    # Print numbers inside the matrix
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     fontsize=14, color="black")

    # Add colorbar
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()


# =======================
# Example usage:
# =======================
if __name__ == "__main__":
    plot_clean_confusion_matrix(
    tn=1157,
    fp=11,
    fn=6,
    tp=513,
    title="Confusion Matrix – Final Model – struct + PWM",
    savepath="clean_confusion_struct_pwm.png"
)
