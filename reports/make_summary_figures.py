"""Summary figures for the README.

The numbers below are copied from the notebooks named in the comments.
Run from the repo root:  python reports/make_summary_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREY, RED = "#3b78b0", "#e8872e", "#9a9a9a", "#c0392b"


def results_at_a_glance():
    # AUC (mean of per-fold AUCs), development folds
    rows = [
        ("Seizure vs the rest\n(amplitude only, notebook 09)", 0.997, BLUE),
        ("Seizure vs classes 2 and 3\n(shape features only, notebook 11)", 0.994, BLUE),
        ("Class 2 vs 3, whole-segment features\n(15 shape features, logistic, notebook 12)", 0.749, ORANGE),
        ("Class 2 vs 3, one chunk at a time\n(best honest model, notebook 07)", 0.635, GREY),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axvspan(0.32, 0.66, color="#dddddd", alpha=0.7, label="what chance can produce (shuffled labels, widest range across tasks): 0.32 to 0.66")
    ax.axvline(0.5, color="black", lw=0.8)
    for i, (label, auc, colour) in enumerate(rows[::-1]):
        ax.barh(i, auc - 0.3, left=0.3, color=colour, height=0.55)
        ax.text(auc + 0.01, i, f"{auc:.3f}", va="center", fontsize=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=9)
    ax.set_xlim(0.3, 1.09)
    ax.set_xlabel("AUC (1.0 = perfect, 0.5 = coin flip)")
    ax.set_title("How well can each question be answered?", fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "results_at_a_glance.png", dpi=150)
    plt.close()


def svm_lead_vanished():
    # SVM AUC minus logistic AUC on the 15 shape features (notebook 14)
    labels = ["saved folds", "fresh folds\n(seed 1000)", "fresh folds\n(seed 2000)", "fresh folds\n(seed 3000)"]
    gaps = [0.044, 0.019, 0.012, 0.011]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, gaps, color=[ORANGE, BLUE, BLUE, BLUE], width=0.55)
    ax.axhline(0.03, color=RED, ls="--", lw=1.2)
    ax.text(3.45, 0.0315, "our rule: needs more than 0.03", color=RED, fontsize=8, ha="right", va="bottom")
    for i, g in enumerate(gaps):
        ax.text(i, g + 0.001, f"+{g:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 0.056)
    ax.set_ylabel("SVM AUC minus logistic AUC")
    ax.set_title("The SVM's lead did not replicate", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "svm_lead_vanished.png", dpi=150)
    plt.close()


def leak_in_one_picture():
    pair_colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#16a085"]

    def square(ax, x, y, colour):
        ax.add_patch(Rectangle((x, y), 0.8, 0.8, facecolor=colour, edgecolor="black", lw=0.8))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, title in zip(axes, ["Plain split by segment (leaks)", "Split by whole twin group (honest)"]):
        ax.set_xlim(-0.3, 11.3)
        ax.set_ylim(-0.2, 5.6)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.add_patch(Rectangle((0, 0.3), 6.2, 3.7, fill=False, lw=1.5, edgecolor=GREY))
        ax.add_patch(Rectangle((7, 0.3), 4, 3.7, fill=False, lw=1.5, edgecolor=GREY))
        ax.text(3.1, 4.25, "TRAINING", ha="center", fontsize=10, weight="bold")
        ax.text(9, 4.25, "VALIDATION", ha="center", fontsize=10, weight="bold")

    # left: twins end up on both sides
    ax = axes[0]
    train_left = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (0, 1), (1, 1), (2, 1)]   # (pair, twin) already in training
    # training: one twin of pairs 0-5 and the second twin of pairs 0-2; validation: second twin of pairs 3-5 plus ...
    train_items = [(p, t) for p, t in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (4, 0), (5, 0)]]
    val_items = [(3, 1), (4, 1), (5, 1)]
    pos_train = {}
    for k, (p, t) in enumerate(train_items):
        x, y = 0.3 + (k % 5) * 1.15, 2.7 - (k // 5) * 1.3
        square(ax, x, y, pair_colours[p]); pos_train[(p, t)] = (x + 0.4, y + 0.4)
    for k, (p, t) in enumerate(val_items):
        x, y = 7.4 + k * 1.15, 2.7
        square(ax, x, y, pair_colours[p])
        tx, ty = pos_train[(p, 0)]
        ax.plot([tx, x + 0.4], [ty, y + 0.4], color=RED, ls="--", lw=1.2)
    ax.text(5.5, 5.0, "each colour = one recording filed twice", ha="center", fontsize=9, style="italic")
    ax.text(5.5, -0.15, "A validation segment's twin sits in training,\nso the model has seen its answer", ha="center", va="top", fontsize=9, color=RED)

    # right: whole pairs stay together
    ax = axes[1]
    train_pairs = [0, 1, 2, 3]
    val_pairs = [4, 5]
    for k, p in enumerate(train_pairs):
        for t in range(2):
            x, y = 0.3 + (k * 2 + t) % 5 * 1.15, 2.7 - ((k * 2 + t) // 5) * 1.3
            square(ax, x, y, pair_colours[p])
    for k, p in enumerate(val_pairs):
        for t in range(2):
            x, y = 7.4 + (k * 2 + t) % 3 * 1.15, 2.7 - ((k * 2 + t) // 3) * 1.3
            square(ax, x, y, pair_colours[p])
    ax.text(5.5, 5.0, "each colour = one recording filed twice", ha="center", fontsize=9, style="italic")
    ax.text(5.5, -0.15, "Twins always travel together,\nso validation is truly unseen", ha="center", va="top", fontsize=9, color="#1e8449")

    plt.tight_layout()
    plt.savefig(OUT / "leak_in_one_picture.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    results_at_a_glance()
    svm_lead_vanished()
    leak_in_one_picture()
    print("saved to", OUT)
