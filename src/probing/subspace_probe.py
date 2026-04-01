"""
SVD-based unsupervised hallucination subspace identification.

Methodology: HaloScope (NeurIPS 2024 Spotlight) — Du et al.
"HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection"

Core Insight
------------
Truthful and hallucinated completions occupy geometrically distinct regions of
the activation space. SVD factorization of the activation matrix identifies the
principal directions of variance. If hallucinated outputs cluster in a specific
subspace, the top singular vectors capture that structure — WITHOUT using any
labels during score computation.

The approach:
1. Collect mean-pooled hidden states at a given layer: X ∈ R^[N × 768]
2. Standardize and compute SVD: X = U @ diag(S) @ Vt
3. Project each sample onto the top-k right singular vectors: coords = X @ Vt[:k].T
4. Membership score = L2 norm of projection — how much a sample "belongs" to
   the dominant subspace
5. Evaluate: AUROC of this score against labels (labels used only for evaluation,
   never during score computation)

Why this matters vs. supervised probing
----------------------------------------
Supervised LR probe: requires labels to TRAIN — can't generalize to unlabeled data.
SVD subspace score: requires labels only to EVALUATE — the score itself is unsupervised.
A strong AUROC from an unsupervised score is evidence that the model's own geometry
separates truthful from hallucinated representations, independent of our supervision.

This directly responds to the MIND (ACL 2024) / HaloScope (NeurIPS 2024) gap in the
layer-wise probing literature: does the hallucination signal have geometric structure,
or is it only linearly separable when trained with labels?
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import List, Dict


def compute_subspace_scores(
    activations: list,
    layer_idx: int,
    k: int = 10,
    use_last_token: bool = False,
) -> np.ndarray:
    """
    Project activations onto the top-k singular vectors and return membership scores.

    Args:
        activations : list of activation dicts from extract_activations()
        layer_idx   : transformer layer to probe (0=embedding, 1..12=blocks)
        k           : number of top singular vectors (subspace rank)
        use_last_token: if True, use last-token representation; otherwise mean-pool

    Returns:
        scores : [N] float array — higher = stronger membership in top-k subspace
    """
    if use_last_token:
        X = np.stack([
            act["hidden_states"][layer_idx][-1]  # last token position
            for act in activations
        ])
    else:
        X = np.stack([
            act["hidden_states"][layer_idx].mean(axis=0)  # mean-pool over seq
            for act in activations
        ])  # [N, hidden_dim]

    # Standardize: zero-mean, unit-variance per dimension
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # [N, hidden_dim]

    # Truncated SVD via numpy (exact for our sizes; ~0.5s for N=400, D=768)
    # X_scaled ≈ U @ diag(S) @ Vt   where Vt[i] = i-th right singular vector
    _, _, Vt = np.linalg.svd(X_scaled, full_matrices=False)  # Vt: [min(N,D), D]

    # Project each sample onto the top-k subspace
    projections = X_scaled @ Vt[:k].T  # [N, k]

    # Membership score = Euclidean norm of projection in the k-dim subspace
    scores = np.linalg.norm(projections, axis=1)  # [N]

    return scores


def evaluate_subspace_at_layer(
    activations: list,
    labels: list,
    layer_idx: int,
    k: int = 10,
) -> Dict:
    """
    Compute AUROC of unsupervised SVD subspace score at a given layer.

    The score is computed WITHOUT labels. AUROC uses labels only for evaluation.
    We try both directions (high-score = hallucinated, or high-score = correct)
    and keep whichever is stronger — we don't assume which class clusters in the
    dominant subspace.

    Returns dict with layer, subspace_auroc, k, direction, n_used.
    """
    scores = compute_subspace_scores(activations, layer_idx, k)
    y = np.array(labels)

    auroc_pos = roc_auc_score(y, scores)   # higher score → correct
    auroc_neg = roc_auc_score(y, -scores)  # higher score → hallucinated

    if auroc_pos >= auroc_neg:
        auroc, direction = auroc_pos, "high=correct"
    else:
        auroc, direction = auroc_neg, "high=hallucinated"

    return {
        "layer":           layer_idx,
        "subspace_auroc":  round(auroc, 4),
        "direction":       direction,
        "k":               k,
        "n_used":          len(y),
    }


def probe_all_layers_subspace(
    activations: list,
    labels: list,
    k: int = 10,
) -> List[Dict]:
    """
    Run unsupervised SVD subspace scoring across all transformer layers.

    Args:
        activations : list of activation dicts
        labels      : ground-truth labels (1=correct, 0=hallucinated) — evaluation only
        k           : SVD subspace rank

    Returns:
        List of result dicts, one per layer, ordered layer 0 → num_layers.
    """
    num_layers = activations[0]["hidden_states"].shape[0]  # 13 for GPT-2

    print(f"SVD Subspace Probing (k={k}): {num_layers} layers, N={len(labels)}")
    print(f"  [Unsupervised scores — labels used only for AUROC evaluation]")
    print("-" * 62)

    results = []
    for layer_idx in range(num_layers):
        r = evaluate_subspace_at_layer(activations, labels, layer_idx, k)
        results.append(r)
        lbl = "embed" if layer_idx == 0 else f"L{layer_idx:2d}   "
        print(
            f"  {lbl}  Subspace AUROC = {r['subspace_auroc']:.4f}"
            f"  [{r['direction']}]"
        )

    peak = max(results, key=lambda r: r["subspace_auroc"])
    print("-" * 62)
    print(f"  Peak: Layer {peak['layer']} -> Subspace AUROC {peak['subspace_auroc']:.4f}")

    return results
