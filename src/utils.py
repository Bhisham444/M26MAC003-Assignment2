"""
Utility functions for logging, plotting publication-quality figures,
and exporting results for the CVPR report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # Non-interactive headless backend
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set clean aesthetic style suitable for IEEE/CVPR publications
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def setup_logger(log_file_path: str):
    """
    Creates a simple tee logger that prints to console and appends to file.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    f = open(log_file_path, "w", encoding="utf-8")
    
    def log(msg: str):
        print(msg)
        f.write(msg + "\n")
        f.flush()
        
    return log, f


def plot_q1_oob_disagreement_margin(
    B_list: list,
    oob_accs: list,
    disagreements: list,
    margins: list,
    val_accs: list,
    save_path: str = "figures/q1_oob_disagreement_margin.png"
):
    """
    Plots OOB Accuracy, Tree Disagreement, and Vote Margin against ensemble size B.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(7, 4.2), dpi=300)

    color_oob = "#1f77b4"
    color_val = "#2ca02c"
    color_dis = "#d62728"
    color_mar = "#9467bd"

    # Left axis: Accuracies and Margin
    ax1.plot(B_list, val_accs, marker="o", color=color_val, label="Validation Accuracy")
    ax1.plot(B_list, oob_accs, marker="s", color=color_oob, label="OOB Accuracy")
    ax1.plot(B_list, margins, marker="^", color=color_mar, linestyle="--", label="Mean Vote Margin")
    ax1.set_xlabel("Ensemble Size $B$ (Number of Trees)")
    ax1.set_ylabel("Accuracy / Vote Margin", color="black")
    ax1.set_xscale("log")
    ax1.set_xticks(B_list)
    ax1.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax1.grid(True)
    ax1.set_ylim(-0.05, 1.05)

    # Right axis: Disagreement
    ax2 = ax1.twinx()
    ax2.plot(B_list, disagreements, marker="D", color=color_dis, linestyle="-.", label="Pairwise Disagreement")
    ax2.set_ylabel("Pairwise Disagreement", color=color_dis)
    ax2.tick_params(axis="y", labelcolor=color_dis)
    ax2.set_ylim(-0.05, 1.05)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", framealpha=0.9)

    plt.title("Q1: Ensemble Dynamics vs. Bagging Size $B$")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_q1_error_faces(
    corrected_cases: list,
    failed_cases: list,
    save_path: str = "figures/q1_error_faces.png"
):
    """
    Plots the 4 diagnostic face cases: 2 where bagging corrects the tree,
    and 2 where both models are wrong.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.5), dpi=300)

    # Rows: 0 -> images, 1 -> vote histograms
    all_cases = corrected_cases + failed_cases
    titles = [
        "Corrected #1", "Corrected #2",
        "Both Wrong #1", "Both Wrong #2"
    ]

    for col, case in enumerate(all_cases):
        # Image
        img = case["image"].reshape(64, 64)
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].axis("off")
        status = "Corrected" if col < 2 else "Both Wrong"
        axes[0, col].set_title(
            f"{titles[col]}\nTrue: {case['y_true']} | Tree: {case['tree_pred']}\nBag: {case['bag_pred']}",
            fontsize=9
        )

        # Vote distribution bar plot
        votes = case["votes"] # (40,)
        top_classes = np.argsort(votes)[::-1][:5]
        top_votes = votes[top_classes]
        bars = axes[1, col].bar(range(len(top_classes)), top_votes, color="#3b528b", width=0.6)
        axes[1, col].set_xticks(range(len(top_classes)))
        axes[1, col].set_xticklabels([f"C{c}" for c in top_classes], fontsize=8)
        axes[1, col].set_ylabel("Vote Count", fontsize=8)
        axes[1, col].set_ylim(0, max(max(votes) * 1.15, 10))
        axes[1, col].grid(True, axis="y")

        # Highlight true class if present
        for bar_idx, c in enumerate(top_classes):
            if c == case["y_true"]:
                bars[bar_idx].set_color("#2ca02c")
            elif c == case["tree_pred"]:
                bars[bar_idx].set_color("#d62728")

    axes[1, 0].set_title("Vote Dist (Green=True, Red=Tree)", fontsize=8, loc="left")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_q2_diagnostics(
    rounds: list,
    weighted_errors: list,
    learner_weights: list,
    max_weights: list,
    top10_weights: list,
    save_path: str = "figures/q2_diagnostics.png"
):
    """
    Plots the 4 diagnostic curves over SAMME AdaBoost rounds:
    1. Weighted training error epsilon_t
    2. Learner weight alpha_t
    3. Maximum sample weight max_i w_t(i)
    4. Total weight of top-10 heaviest training images
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), dpi=300)

    # 1. Weighted error
    axes[0, 0].plot(rounds, weighted_errors, color="#d62728", marker="o", markersize=4)
    axes[0, 0].set_xlabel("Boosting Round $t$")
    axes[0, 0].set_ylabel(r"Weighted Error $\epsilon_t$")
    axes[0, 0].set_title(r"Weighted Training Error $\epsilon_t$")
    axes[0, 0].grid(True)

    # 2. Learner weight alpha_t
    axes[0, 1].plot(rounds, learner_weights, color="#1f77b4", marker="s", markersize=4)
    axes[0, 1].set_xlabel("Boosting Round $t$")
    axes[0, 1].set_ylabel(r"Learner Weight $\alpha_t$")
    axes[0, 1].set_title(r"Weak Learner Weight $\alpha_t$")
    axes[0, 1].grid(True)

    # 3. Maximum sample weight
    axes[1, 0].plot(rounds, max_weights, color="#ff7f0e", marker="^", markersize=4)
    axes[1, 0].set_xlabel("Boosting Round $t$")
    axes[1, 0].set_ylabel(r"Max Sample Weight $\max_i w_t(i)$")
    axes[1, 0].set_title(r"Maximum Sample Weight $\max_i w_t(i)$")
    axes[1, 0].grid(True)

    # 4. Total weight of 10 heaviest training images
    axes[1, 1].plot(rounds, top10_weights, color="#2ca02c", marker="D", markersize=4)
    axes[1, 1].set_xlabel("Boosting Round $t$")
    axes[1, 1].set_ylabel("Cumulative Weight")
    axes[1, 1].set_title("Total Weight of Top 10 Heaviest Images")
    axes[1, 1].grid(True)

    plt.suptitle("Q2: SAMME AdaBoost Training Dynamics", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_q2_heavy_sample(
    face_image: np.ndarray,
    sample_idx: int,
    true_label: int,
    weight_history: list,
    is_misclassified_history: list,
    save_path: str = "figures/q2_heavy_sample_history.png"
):
    """
    Plots the heavily weighted training sample and its weight trajectory across rounds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=300, gridspec_kw={"width_ratios": [1, 2]})

    # Face image
    ax1.imshow(face_image.reshape(64, 64), cmap="gray")
    ax1.set_title(f"Heaviest Face Sample #{sample_idx}\n(Class {true_label})", fontsize=10)
    ax1.axis("off")

    # Weight trajectory
    rounds = np.arange(1, len(weight_history) + 1)
    ax2.plot(rounds, weight_history, color="#d62728", marker="o", markersize=3, label="Sample Weight $w_t(i)$")
    
    # Mark rounds where sample was misclassified
    misc_rounds = [r for r, m in zip(rounds, is_misclassified_history) if m]
    misc_weights = [w for w, m in zip(weight_history, is_misclassified_history) if m]
    ax2.scatter(misc_rounds, misc_weights, color="black", s=25, zorder=5, label="Misclassified by $h_t$")

    ax2.set_xlabel("Boosting Round $t$")
    ax2.set_ylabel("Weight $w_t(i)$")
    ax2.set_title("Weight Progression Across Boosting Rounds")
    ax2.grid(True)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_q2_masked_comparison_face(
    clean_img: np.ndarray,
    masked_img: np.ndarray,
    sample_idx: int,
    y_true: int,
    bag_pred_clean: int,
    bag_pred_mask: int,
    boost_pred_clean: int,
    boost_pred_mask: int,
    save_path: str = "figures/q2_masked_comparison_face.png"
):
    """
    Plots the face that responds differently between bagging and boosting to the masked region.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.8), dpi=300)

    ax1.imshow(clean_img.reshape(64, 64), cmap="gray")
    ax1.set_title(f"Clean Face #{sample_idx} (True: {y_true})\nBag: {bag_pred_clean} | Boost: {boost_pred_clean}", fontsize=9)
    ax1.axis("off")

    ax2.imshow(masked_img.reshape(64, 64), cmap="gray")
    ax2.set_title(f"Masked Face #{sample_idx} (True: {y_true})\nBag: {bag_pred_mask} | Boost: {boost_pred_mask}", fontsize=9)
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_q3_bias_variance_depth(
    depth_labels: list,
    tree_bias: list,
    tree_var: list,
    bag_bias: list,
    bag_var: list,
    save_path: str = "figures/q3_bias_variance_depth.png"
):
    """
    Plots Bias^2 and Variance across depths for Single Tree vs 100-Tree Bagging.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=300)

    x = np.arange(len(depth_labels))
    width = 0.35

    # 1. Bias Squared
    ax1.bar(x - width/2, tree_bias, width, label="Single Tree", color="#1f77b4")
    ax1.bar(x + width/2, bag_bias, width, label="100-Tree Bagging", color="#2ca02c")
    ax1.set_xticks(x)
    ax1.set_xticklabels(depth_labels)
    ax1.set_xlabel("Tree Depth")
    ax1.set_ylabel(r"$\mathrm{Bias}^2$")
    ax1.set_title(r"$\mathrm{Bias}^2$ vs Tree Depth")
    ax1.grid(True, axis="y")
    ax1.legend()

    # 2. Variance
    ax2.bar(x - width/2, tree_var, width, label="Single Tree", color="#d62728")
    ax2.bar(x + width/2, bag_var, width, label="100-Tree Bagging", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(depth_labels)
    ax2.set_xlabel("Tree Depth")
    ax2.set_ylabel("Variance")
    ax2.set_title("Variance vs Tree Depth")
    ax2.grid(True, axis="y")
    ax2.legend()

    plt.suptitle("Q3: Bias-Variance Decomposition across Depths", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")


def plot_q3_decomposition_bars(
    models: list,
    biases: list,
    variances: list,
    save_path: str = "figures/q3_decomposition_bars.png"
):
    """
    Stacked bar chart of Bias^2 and Variance for Selected Tree, Bagging, and Boosting.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    x = np.arange(len(models))
    width = 0.45

    ax.bar(x, biases, width, label=r"$\mathrm{Bias}^2$", color="#3b528b", edgecolor="black")
    ax.bar(x, variances, width, bottom=biases, label="Variance", color="#5dc863", edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(r"Total Error ($\mathrm{Bias}^2 + \mathrm{Variance}$)")
    ax.set_title(r"Bias-Variance Decomposition on Fixed Test Set")
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    # Add numeric labels on bars
    for i in range(len(models)):
        total = biases[i] + variances[i]
        ax.text(i, total + 0.02, f"Total: {total:.3f}\n(Bias²:{biases[i]:.3f}, Var:{variances[i]:.3f})",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {save_path}")
