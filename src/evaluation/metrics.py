"""
Evaluation utilities: plots and result tables for all phases.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)


def plot_roc_curve(y_true, y_scores_dict: dict, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUROC=0.50)")

    for name, scores in y_scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auroc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUROC={auroc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Hallucination Predictor")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved ROC curve: {save_path}")
    plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Hallucinated", "Correct"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved confusion matrix: {save_path}")
    plt.show()
    return fig


def plot_confidence_vs_accuracy(
    confidence_scores: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10,
    save_path: str = None,
):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_acc = [], []

    for i in range(n_bins):
        mask = (confidence_scores >= bins[i]) & (confidence_scores < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_acc.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(bin_centers, bin_acc, "o-", label="Model")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Accuracy")
    ax.set_title("Confidence vs Accuracy (Calibration)")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved calibration plot: {save_path}")
    plt.show()
    return fig


def plot_ablation_heatmap(
    importance_matrix: np.ndarray,
    save_path: str = None,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        importance_matrix,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        annot=False,
        xticklabels=[f"H{i}" for i in range(importance_matrix.shape[1])],
        yticklabels=[f"L{i}" for i in range(importance_matrix.shape[0])],
    )
    ax.set_title("Causal Head Importance (Accuracy Drop on Ablation)")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved ablation heatmap: {save_path}")
    plt.show()
    return fig


def plot_logit_lens(
    hidden_states: np.ndarray,
    model,
    tokenizer,
    token_idx: int = -1,
    top_k: int = 5,
    save_path: str = None,
):
    import torch

    lm_head   = model.lm_head
    num_layers = hidden_states.shape[0]

    top_tokens_per_layer = []
    for layer in range(num_layers):
        h = torch.tensor(hidden_states[layer, token_idx]).unsqueeze(0)
        with torch.no_grad():
            logits = lm_head(h)[0]
        probs = torch.softmax(logits, dim=-1)
        top_idx   = probs.topk(top_k).indices.numpy()
        top_probs = probs.topk(top_k).values.numpy()
        top_tokens_per_layer.append([
            (tokenizer.decode([idx]).strip(), float(p))
            for idx, p in zip(top_idx, top_probs)
        ])

    print(f"\nLogit Lens (position={token_idx}, top-{top_k} per layer)")
    print(f"{'Layer':<8}" + "".join(f"  {i+1}{'':>12}" for i in range(top_k)))
    print("-" * (8 + top_k * 14))
    for layer, tokens in enumerate(top_tokens_per_layer):
        row = f"L{layer:<7}"
        for tok, prob in tokens:
            row += f"  {tok[:10]:<10}({prob:.3f})"
        print(row)

    top1_probs  = [tokens[0][1] for tokens in top_tokens_per_layer]
    top1_tokens = [tokens[0][0] for tokens in top_tokens_per_layer]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(num_layers), top1_probs)
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{i}\n{t}" for i, t in enumerate(top1_tokens)], fontsize=7)
    ax.set_ylabel("Top-1 Probability")
    ax.set_title("Logit Lens: Top-1 Prediction Probability per Layer")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved logit lens plot: {save_path}")
    plt.show()
    return fig


def plot_layer_probing_curve(
    layer_results: list,
    save_path: str = None,
):
    """
    Plot supervised layer probe AUROC vs. transformer depth with 95% CI.
    """
    layers     = [r["layer"]      for r in layer_results]
    aurocs     = [r["mean_auroc"] for r in layer_results]
    ci_low     = [r["ci_low"]     for r in layer_results]
    ci_high    = [r["ci_high"]    for r in layer_results]
    peak_layer = int(np.argmax(aurocs))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(layers, ci_low, ci_high, alpha=0.2, color="steelblue", label="95% CI")
    ax.plot(layers, aurocs, "o-", color="steelblue", linewidth=2, markersize=6,
            label="Probe AUROC")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax.axvline(peak_layer, color="crimson", linestyle=":", linewidth=1.5,
               label=f"Peak: Layer {peak_layer} (AUROC={aurocs[peak_layer]:.3f})")
    ax.scatter([peak_layer], [aurocs[peak_layer]], color="crimson", zorder=5, s=80)

    xticklabels = ["Emb"] + [f"L{i}" for i in range(1, len(layers))]
    ax.set_xticks(layers)
    ax.set_xticklabels(xticklabels, fontsize=8)
    ax.set_xlabel("Transformer Layer (0 = Embedding, 1–12 = Attention Blocks)")
    ax.set_ylabel("Probe AUROC (5-fold CV)")
    ax.set_title("Layer-wise Representation Probing: Where is the Hallucination Signal?")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved layer probing curve: {save_path}")
    plt.show()
    return fig


def plot_subspace_vs_supervised(
    layer_results: list,
    subspace_results: list,
    save_path: str = None,
):
    """
    Overlay: supervised dense probe AUROC vs. unsupervised SVD subspace AUROC.

    This is the HaloScope-inspired comparison: does the model's own geometric
    structure (unsupervised) reproduce what a trained probe finds (supervised)?
    Agreement = the signal is intrinsic to the representation geometry.
    """
    layers   = [r["layer"]      for r in layer_results]
    sup_auc  = [r["mean_auroc"] for r in layer_results]
    unsup_auc = [r["subspace_auroc"] for r in subspace_results]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, sup_auc,   "o-", color="steelblue",  linewidth=2.5, markersize=7,
            label="Supervised LR Probe (5-fold CV)")
    ax.plot(layers, unsup_auc, "s--", color="darkorange", linewidth=2.5, markersize=7,
            label="Unsupervised SVD Subspace Score")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.50)")

    sup_peak   = int(np.argmax(sup_auc))
    unsup_peak = int(np.argmax(unsup_auc))
    ax.axvline(sup_peak,   color="steelblue",  linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(unsup_peak, color="darkorange",  linestyle=":", linewidth=1.2, alpha=0.7)

    ax.set_xticks(layers)
    ax.set_xticklabels(["Emb"] + [f"L{i}" for i in range(1, len(layers))], fontsize=8)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(
        "Supervised vs. Unsupervised: Does Geometry Reproduce the Trained Probe?\n"
        "(HaloScope-style SVD subspace vs. standard LR probe)",
        fontsize=11
    )
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved subspace comparison plot: {save_path}")
    plt.show()
    return fig


def plot_component_comparison(
    ffn_results: list,
    attn_results: list,
    supervised_results: list = None,
    save_path: str = None,
):
    """
    Compare FFN component probe AUROC vs. Attention component probe AUROC per layer.

    This is the ReDeEP-style component decomposition for closed-book QA.
    Shading between curves shows which component dominates at each layer.
    """
    layers     = [r["layer"]      for r in ffn_results]
    ffn_auc    = [r["mean_auroc"] for r in ffn_results]
    attn_auc   = [r["mean_auroc"] for r in attn_results]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Fill between to highlight dominance
    ax.fill_between(layers,
                    [min(f, a) for f, a in zip(ffn_auc, attn_auc)],
                    [max(f, a) for f, a in zip(ffn_auc, attn_auc)],
                    alpha=0.12, color="gray")

    ax.plot(layers, ffn_auc,  "o-", color="#d62728", linewidth=2.5, markersize=7,
            label="FFN Output (Parametric Memory)")
    ax.plot(layers, attn_auc, "s-", color="#1f77b4", linewidth=2.5, markersize=7,
            label="Attention Output (Context Composition)")

    if supervised_results is not None:
        sup_auc = [r["mean_auroc"] for r in supervised_results]
        # Residual stream results have 13 layers (incl. embedding); components have 12
        sup_auc_blocks = sup_auc[1:]  # skip embedding layer
        ax.plot(layers, sup_auc_blocks, "--", color="black", linewidth=1.5, alpha=0.5,
                label="Full Hidden State (reference)")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance (0.50)")

    ffn_peak  = int(np.argmax(ffn_auc))
    attn_peak = int(np.argmax(attn_auc))
    ax.scatter([ffn_peak],  [ffn_auc[ffn_peak]],  color="#d62728", zorder=5, s=150, marker="*")
    ax.scatter([attn_peak], [attn_auc[attn_peak]], color="#1f77b4", zorder=5, s=150, marker="*")

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in range(len(layers))], fontsize=8)
    ax.set_xlabel("Transformer Block (Layer 0–11)")
    ax.set_ylabel("Probe AUROC (5-fold CV)")
    ax.set_title(
        "Component Decomposition: FFN vs. Attention — Which Drives Hallucination?\n"
        "(ReDeEP-style for Closed-Book QA on TruthfulQA)",
        fontsize=11
    )
    ax.set_ylim(0.3, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved component comparison plot: {save_path}")
    plt.show()
    return fig


def plot_token_position_heatmap(
    position_results: list,
    positions: list = None,
    save_path: str = None,
):
    """
    2D heatmap: [num_layers × num_token_positions] AUROC.

    Rows = transformer layers (embedding to L12)
    Columns = token positions (mean-pool, first, last-3, last-2, last)

    This is a novel visualization: no paper in the layer-probing literature
    shows the joint layer × position AUROC matrix. Mean-pooling collapses
    the positional dimension and hides where in the sequence the signal lives.
    """
    if positions is None:
        positions = [0, -3, -2, -1]

    num_layers = len(position_results)
    col_keys   = ["mean_pool"] + positions
    col_labels = ["Mean\nPool", "First\nToken", "Last-3", "Last-2", "Last\nToken"]

    # Build matrix [num_layers, num_positions]
    matrix = np.zeros((num_layers, len(col_keys)))
    for i, r in enumerate(position_results):
        for j, key in enumerate(col_keys):
            matrix[i, j] = r.get(key, 0.5)

    row_labels = ["Emb"] + [f"L{i}" for i in range(1, num_layers)]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=0.85)
    plt.colorbar(im, ax=ax, label="Probe AUROC")

    # Annotate cells
    for i in range(num_layers):
        for j in range(len(col_keys)):
            val = matrix[i, j]
            color = "white" if val < 0.5 or val > 0.78 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(col_keys)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(
        "Token-Position × Layer AUROC Heatmap\n"
        "Where in the sequence is the hallucination signal?",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved token position heatmap: {save_path}")
    plt.show()
    return fig


def plot_dla_comparison(
    summary: dict,
    save_path: str = None,
):
    """
    Two-panel DLA comparison: FFN and Attention contributions per layer,
    correct vs hallucinated. The difference curve shows which layers
    actively drive the divergence in prediction.
    """
    num_layers   = summary["num_layers"]
    c_ffn        = summary["correct_mean_ffn_dla"]
    h_ffn        = summary["hallucinated_mean_ffn"]
    c_attn       = summary["correct_mean_attn_dla"]
    h_attn       = summary["hallucinated_mean_attn"]
    ffn_diff     = summary["ffn_dla_diff"]
    attn_diff    = summary["attn_dla_diff"]
    peak_ffn     = summary["peak_ffn_diff_layer"]
    peak_attn    = summary["peak_attn_diff_layer"]

    layers       = list(range(num_layers))
    layer_labels = [f"L{i}" for i in range(num_layers)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # FFN DLA: correct vs hallucinated
    ax = axes[0, 0]
    ax.plot(layers, c_ffn, color="#1B4F8A", linewidth=2, marker="o",
            markersize=5, label=f"Correct (n={summary['n_correct']})")
    ax.plot(layers, h_ffn, color="#C0392B", linewidth=2, marker="s",
            markersize=5, label=f"Hallucinated (n={summary['n_hallucinated']})")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("FFN Direct Logit Attribution", fontweight="bold")
    ax.set_ylabel("Mean DLA (logit units)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Attn DLA: correct vs hallucinated
    ax = axes[0, 1]
    ax.plot(layers, c_attn, color="#1B4F8A", linewidth=2, marker="o", markersize=5,
            label=f"Correct")
    ax.plot(layers, h_attn, color="#C0392B", linewidth=2, marker="s", markersize=5,
            label=f"Hallucinated")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title("Attention Direct Logit Attribution", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # FFN diff (correct - hallucinated)
    ax = axes[1, 0]
    colors = ["#C0392B" if l == peak_ffn else "#5D6D7E" for l in layers]
    ax.bar(layer_labels, ffn_diff, color=colors)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"FFN DLA Difference (Correct − Hallucinated)\nPeak: L{peak_ffn}",
                 fontweight="bold")
    ax.set_ylabel("DLA difference (logit units)")
    ax.set_xlabel("Layer")
    ax.spines[["top", "right"]].set_visible(False)

    # Attn diff
    ax = axes[1, 1]
    colors = ["#C0392B" if l == peak_attn else "#5D6D7E" for l in layers]
    ax.bar(layer_labels, attn_diff, color=colors)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"Attention DLA Difference (Correct − Hallucinated)\nPeak: L{peak_attn}",
                 fontweight="bold")
    ax.set_xlabel("Layer")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Direct Logit Attribution: Which Components Drive Correct vs Hallucinated Predictions?",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved DLA comparison: {save_path}")
    plt.show()
    return fig


def plot_logit_lens_divergence(
    summary: dict,
    save_path: str = None,
):
    """
    Two-panel logit lens plot:
      Top: mean generated-token probability per layer, correct vs hallucinated
      Bottom: separation (|correct - hallucinated|) with divergence layer marked
    """
    num_layers   = summary["num_layers"]
    correct_mean = summary["correct_mean_prob"]
    hall_mean    = summary["hallucinated_mean_prob"]
    separation   = summary["layer_separation"]
    div_layer    = summary["divergence_layer"]

    layers      = list(range(num_layers))
    layer_labels = ["Emb"] + [f"L{i}" for i in range(1, num_layers)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Panel 1: probability curves
    ax1.plot(layers, correct_mean,  color="#1B4F8A", linewidth=2.5,
             marker="o", markersize=6, label=f"Correct (n={summary['n_correct']})")
    ax1.plot(layers, hall_mean,     color="#C0392B", linewidth=2.5,
             marker="s", markersize=6, label=f"Hallucinated (n={summary['n_hallucinated']})")
    ax1.axvline(div_layer, color="#E67E22", linewidth=1.5, linestyle="--",
                label=f"Max divergence: L{div_layer}")
    ax1.set_ylabel("Mean P(generated token)", fontsize=12)
    ax1.set_title(
        "Logit Lens: How Does the Model's Prediction Evolve Layer by Layer?\n"
        "Mean probability of the generated token at each intermediate layer",
        fontsize=11, fontweight="bold"
    )
    ax1.legend(fontsize=10)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: separation
    ax2.bar(layers, separation, color=["#C0392B" if l == div_layer else "#5D6D7E"
                                        for l in layers])
    ax2.axvline(div_layer, color="#E67E22", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("|Correct − Hallucinated|", fontsize=12)
    ax2.set_xlabel("Transformer Layer", fontsize=12)
    ax2.set_title("Layer-wise Separation Between Correct and Hallucinated Samples",
                  fontsize=11)
    ax2.set_xticks(layers)
    ax2.set_xticklabels(layer_labels, fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved logit lens divergence plot: {save_path}")
    plt.show()
    return fig


def plot_steering_layer_sweep(
    layer_results: dict,
    save_path: str = None,
):
    """
    Bar chart: AUROC improvement from steering across all transformer layers.
    Peak should align with the layer-probing peak (L9), independently validating
    the cascade finding.
    """
    results  = layer_results["layer_results"]
    peak_layer = layer_results["peak_layer"]

    layers      = [r["layer"] for r in results]
    improvements = [r["improvement"] for r in results]
    labels_x    = ["Emb"] + [f"L{i}" for i in range(1, len(results))]
    colors      = ["#C0392B" if l == peak_layer else "#1B4F8A" for l in layers]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels_x, improvements, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Transformer Layer", fontsize=13)
    ax.set_ylabel("AUROC Improvement from Steering", fontsize=13)
    ax.set_title(
        "Steering Layer Sweep — Which Layer's Truthfulness Direction Is Most Informative?\n"
        "(Independent validation: peak should match layer-probing peak at L9)",
        fontsize=11, fontweight="bold"
    )

    # Annotate peak
    peak_idx = layers.index(peak_layer)
    ax.annotate(
        f"Peak L{peak_layer}\n+{layer_results['peak_improvement']:.4f}",
        xy=(peak_idx, improvements[peak_idx]),
        xytext=(peak_idx + 1.2, improvements[peak_idx] + 0.003),
        fontsize=10, color="#C0392B", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#C0392B"),
    )

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved steering layer sweep: {save_path}")
    plt.show()
    return fig


def plot_steering_curve(
    results: dict,
    save_path: str = None,
):
    """
    Plot AUROC vs. steering alpha — the activation steering experiment result.

    Shows how probe AUROC changes as we inject increasing amounts of the
    truthfulness direction into the residual stream at Layer 9. The optimal
    alpha and degradation pattern reveal the causal geometry of the signal.
    """
    alphas         = results["alphas"]
    found_aurocs   = results["found_aurocs"]
    random_aurocs  = results["random_aurocs"]
    baseline_auroc = results["baseline_auroc"]
    best_alpha     = results["best_alpha"]
    best_auroc     = results["best_auroc"]
    layer_idx      = results.get("layer_idx", 9)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline
    ax.axhline(baseline_auroc, color="#888888", linewidth=1.2,
               linestyle="--", label=f"Baseline (α=0)  AUROC={baseline_auroc:.4f}", zorder=1)

    # Random direction (null baseline)
    ax.plot(alphas, random_aurocs, color="#95A5A6", linewidth=2.0,
            marker="s", markersize=7, linestyle=":", zorder=2,
            label="Random orthogonal direction (null baseline)")

    # Found (truthfulness) direction
    ax.plot(alphas, found_aurocs, color="#1B4F8A", linewidth=2.5,
            marker="o", markersize=8, zorder=3, label="Found truthfulness direction")

    # Best point
    ax.scatter([best_alpha], [best_auroc], color="#C0392B", s=130,
               zorder=5, label=f"Best: α={best_alpha}  AUROC={best_auroc:.4f}")

    improvement = best_auroc - baseline_auroc
    if improvement > 0:
        ax.annotate(
            f"+{improvement:.4f}",
            xy=(best_alpha, best_auroc),
            xytext=(best_alpha + max(alphas) * 0.05, best_auroc + 0.003),
            fontsize=11, color="#C0392B", fontweight="bold",
        )

    ax.set_xlabel("Steering Strength (α)", fontsize=13)
    ax.set_ylabel("Probe AUROC (held-out test set)", fontsize=13)
    ax.set_title(
        f"Activation Steering at Layer {layer_idx}: Found Direction vs. Random Baseline\n"
        "Does the truthfulness direction encode class-discriminative structure?",
        fontsize=11, fontweight="bold"
    )
    ax.set_xticks(alphas)
    all_vals = found_aurocs + random_aurocs
    ax.set_ylim(min(all_vals) - 0.03, max(all_vals) + 0.04)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved steering curve: {save_path}")
    plt.show()
    return fig


def plot_attention_patterns(summary: dict, layer_idx: int = 8, save_path: str = None):
    """
    Phase 11: 3-panel attention pattern comparison plot.

    Panel 1: Per-head entropy — correct vs hallucinated
    Panel 2: Per-head combined discrimination AUROC
    Panel 3: Zone-last3 attention mass — correct vs hallucinated
    """
    n_heads  = summary["n_heads"]
    heads    = np.arange(n_heads)
    c_ent    = summary["correct_mean_entropy"]
    h_ent    = summary["hallucinated_mean_entropy"]
    aurocs   = summary["combined_aurocs"]
    c_last3  = summary["correct_mean_zone_last3"]
    h_last3  = summary["hallucinated_mean_zone_last3"]
    top_head = summary["top_head"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Phase 11 — Attention Pattern Analysis at L{layer_idx}\n"
        "From last-token position: entropy, discrimination, zone attention",
        fontsize=12, fontweight="bold"
    )

    w = 0.35

    # Panel 1: Entropy
    ax = axes[0]
    ax.bar(heads - w/2, c_ent,  w, color="#1B4F8A", alpha=0.8, label="Correct")
    ax.bar(heads + w/2, h_ent,  w, color="#C0392B", alpha=0.8, label="Hallucinated")
    ax.axvline(top_head, color="orange", linestyle="--", linewidth=1.5,
               label=f"Top head H{top_head}")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Mean attention entropy")
    ax.set_title("Attention Entropy per Head", fontweight="bold")
    ax.set_xticks(heads)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Discrimination AUROC
    ax = axes[1]
    colors = ["#E67E22" if i == top_head else "#7F8C8D" for i in heads]
    ax.bar(heads, aurocs, color=colors, alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance (0.50)")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Combined AUROC")
    ax.set_title("Per-Head Discrimination AUROC\n(entropy + zone_last3)", fontweight="bold")
    ax.set_xticks(heads)
    ax.set_ylim(0.45, max(aurocs) + 0.05)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 3: Zone-last3 attention mass
    ax = axes[2]
    ax.bar(heads - w/2, c_last3, w, color="#1B4F8A", alpha=0.8, label="Correct")
    ax.bar(heads + w/2, h_last3, w, color="#C0392B", alpha=0.8, label="Hallucinated")
    ax.axvline(top_head, color="orange", linestyle="--", linewidth=1.5,
               label=f"Top head H{top_head}")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Mean attention mass on last-3 tokens")
    ax.set_title("Attention to '\\nA:' Zone (last 3 tokens)", fontweight="bold")
    ax.set_xticks(heads)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved attention pattern plot: {save_path}")
    plt.show()
    return fig


def plot_head_dla(summary: dict, save_path: str = None):
    """
    Phase 12: Per-head DLA contribution at the target layer.

    Two panels:
      Left:  absolute DLA (correct vs hallucinated per head)
      Right: relative DLA difference % per head
    """
    n_heads  = summary["n_heads"]
    layer    = summary["layer_idx"]
    heads    = np.arange(n_heads)
    c_dla    = summary["correct_mean_head_dla"]
    h_dla    = summary["hallucinated_mean_head_dla"]
    diff     = summary["head_dla_diff"]
    rel      = summary["head_dla_rel_diff_pct"]
    top_abs  = summary["top_head_abs"]
    top_rel  = summary["top_head_rel"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Phase 12 — Head-Level DLA at L{layer}\n"
        "Which attention heads drive the correct-vs-hallucinated logit difference?",
        fontsize=12, fontweight="bold"
    )

    w = 0.35

    # Panel 1: Absolute DLA
    ax = axes[0]
    ax.bar(heads - w/2, c_dla, w, color="#1B4F8A", alpha=0.85, label="Correct")
    ax.bar(heads + w/2, h_dla, w, color="#C0392B", alpha=0.85, label="Hallucinated")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.axvline(top_abs, color="orange", linestyle="--", linewidth=1.5,
               label=f"Top abs H{top_abs}")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Mean DLA (logit units)")
    ax.set_title(f"Absolute Head DLA at L{layer}", fontweight="bold")
    ax.set_xticks(heads)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: Relative difference
    ax = axes[1]
    bar_colors = ["#E67E22" if i == top_rel else
                  ("#1B4F8A" if rel[i] >= 0 else "#C0392B") for i in heads]
    ax.bar(heads, rel, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Head index")
    ax.set_ylabel("Relative DLA diff % (C−H) / magnitude")
    ax.set_title("Relative Head DLA Difference (%)\n"
                 "Normalised by per-head output magnitude", fontweight="bold")
    ax.set_xticks(heads)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved head DLA plot: {save_path}")
    plt.show()
    return fig


def plot_logit_lens_gold(summary: dict, save_path: str = None):
    """
    Phase 9 (gold-token update): dual logit lens plot.

    Panel 1: generated-token probability (original null result)
    Panel 2: gold-token probability across layers (new divergence finding)
    """
    num_layers = summary["num_layers"]
    layers     = np.arange(num_layers)
    has_gold   = "correct_gold_mean_prob" in summary

    n_panels = 2 if has_gold else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "Phase 9 — Logit Lens: Layer-by-Layer Token Probability",
        fontsize=12, fontweight="bold"
    )

    # Panel 1: generated token (existing)
    ax = axes[0]
    ax.plot(layers, summary["correct_mean_prob"],      color="#1B4F8A",
            linewidth=2, marker="o", markersize=4, label="Correct")
    ax.plot(layers, summary["hallucinated_mean_prob"], color="#C0392B",
            linewidth=2, marker="s", markersize=4, label="Hallucinated")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean P(generated token)")
    ax.set_title("Generated-Token Probability\n(model's own output token)", fontweight="bold")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: gold token (new)
    if has_gold:
        ax = axes[1]
        div_layer = summary.get("gold_divergence_layer", 0)
        ax.plot(layers, summary["correct_gold_mean_prob"],      color="#1B4F8A",
                linewidth=2, marker="o", markersize=4, label="Correct")
        ax.plot(layers, summary["hallucinated_gold_mean_prob"], color="#C0392B",
                linewidth=2, marker="s", markersize=4, label="Hallucinated")
        ax.axvline(div_layer, color="#E67E22", linestyle="--", linewidth=1.5,
                   label=f"Divergence L{div_layer}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean P(gold answer first token)")
        ax.set_title(
            "Gold-Token Probability\n(correct answer's first token)",
            fontweight="bold"
        )
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, fontsize=8)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved gold logit lens plot: {save_path}")
    plt.show()
    return fig


def plot_dla_relative(summary: dict, save_path: str = None):
    """
    Plot relative DLA differences (normalised by per-layer magnitude).
    Companion to plot_dla_comparison — this is the scale-invariant view.
    """
    num_layers = summary["num_layers"]
    layers     = np.arange(num_layers)
    attn_rel   = summary["attn_dla_rel_pct"]
    ffn_rel    = summary["ffn_dla_rel_pct"]

    peak_attn = summary.get("peak_attn_rel_layer", int(np.argmax(np.abs(attn_rel))))
    peak_ffn  = summary.get("peak_ffn_rel_layer",  int(np.argmax(np.abs(ffn_rel))))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Relative DLA Difference (Correct − Hallucinated) / Magnitude × 100\n"
        "Scale-invariant view — removes later-layer magnitude dominance",
        fontsize=11, fontweight="bold"
    )

    for ax, rel, peak, label, color in [
        (axes[0], attn_rel, peak_attn, "Attention", "#1B4F8A"),
        (axes[1], ffn_rel,  peak_ffn,  "FFN",       "#27AE60"),
    ]:
        bar_cols = [color if i != peak else "#E67E22" for i in range(num_layers)]
        ax.bar(layers, rel, color=bar_cols, alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Relative DLA diff (%)")
        ax.set_title(f"{label} Relative DLA\n(peak at L{peak}: {rel[peak]:+.1f}%)",
                     fontweight="bold")
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved relative DLA plot: {save_path}")
    plt.show()
    return fig


def print_results_table(results: dict):
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'AUROC':>8} {'F1':>8} {'Accuracy':>10}")
    print("=" * 60)
    for name, metrics in results.items():
        auroc = f"{metrics['auroc']:.4f}" if metrics.get('auroc') is not None else "  N/A"
        f1    = f"{metrics['f1']:.4f}"    if metrics.get('f1')    is not None else "  N/A"
        acc   = f"{metrics['accuracy']:.4f}" if metrics.get('accuracy') is not None else "  N/A"
        print(f"{name:<25} {auroc:>8} {f1:>8} {acc:>10}")
    print("=" * 60)
