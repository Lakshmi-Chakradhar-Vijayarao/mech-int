"""
Layer-wise representation probing — supervised, sparse, and token-position variants.

Three probing modes:

1. SUPERVISED DENSE PROBE (original)
   Mean-pool hidden states → LR probe with 5-fold CV.
   Answers: WHICH layer encodes the most hallucination-predictive information?

2. SPARSE LASSO PROBE (new)
   L1-regularized LR at the peak layer.
   Answers: WHICH DIMENSIONS at the peak layer carry the signal?
   (Moves from "where in the network" to "which neurons")
   Connects to SAE-based work (Anthropic 2024) — finding monosemantic features.

3. TOKEN POSITION PROBE (new)
   Probe each of [first, middle, last-3, last-2, last] token positions separately.
   Produces a [num_layers × num_positions] 2D AUROC heatmap.
   Answers: WHERE IN THE SEQUENCE does the hallucination signal concentrate?
   (Novel: not in any of the cited papers; mean-pooling hides this information)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score
from typing import List, Dict


# ── Internal builders ────────────────────────────────────────────────────────

def _build_dense_probe() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])


def _build_sparse_probe(C: float = 0.1) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1", C=C, solver="liblinear", max_iter=1000, random_state=42
        )),
    ])


def _safe_cv(y: np.ndarray, n_splits: int) -> int:
    actual = min(n_splits, min(int(y.sum()), int((y == 0).sum())))
    return max(2, actual)


# ── 1. Dense layer probe (original) ─────────────────────────────────────────

def probe_layer(
    activations: list,
    labels: list,
    layer_idx: int,
    n_splits: int = 5,
) -> Dict:
    """
    Train a dense LR probe on mean-pooled hidden states at one layer.

    Returns dict: layer, mean_auroc, std_auroc, mean_acc, std_acc, ci_low, ci_high
    """
    X = np.stack([
        act["hidden_states"][layer_idx].mean(axis=0)
        for act in activations
    ])  # [N, hidden_dim]
    y = np.array(labels)

    probe  = _build_dense_probe()
    skf    = StratifiedKFold(n_splits=_safe_cv(y, n_splits), shuffle=True, random_state=42)
    scoring = {
        "auroc":    make_scorer(roc_auc_score, response_method="predict_proba"),
        "accuracy": "accuracy",
    }

    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)

    mean_auroc = float(cv["test_auroc"].mean())
    std_auroc  = float(cv["test_auroc"].std())

    return {
        "layer":      layer_idx,
        "mean_auroc": mean_auroc,
        "std_auroc":  std_auroc,
        "mean_acc":   float(cv["test_accuracy"].mean()),
        "std_acc":    float(cv["test_accuracy"].std()),
        "ci_low":     max(0.0, mean_auroc - 1.96 * std_auroc),
        "ci_high":    min(1.0, mean_auroc + 1.96 * std_auroc),
    }


def probe_all_layers(
    activations: list,
    labels: list,
    n_splits: int = 5,
) -> List[Dict]:
    """Run dense layer probing across all transformer layers."""
    num_layers = activations[0]["hidden_states"].shape[0]
    n = len(labels)

    actual_splits = _safe_cv(np.array(labels), n_splits)
    if actual_splits != n_splits:
        print(f"  [Note] Reduced CV folds to {actual_splits} due to small class size.")

    print(f"Layer-wise probing: {num_layers} layers, {actual_splits}-fold CV, N={n}")
    print(f"Class balance — Correct: {sum(labels)}, Hallucinated: {n - sum(labels)}")
    print("-" * 55)

    results = []
    for layer_idx in range(num_layers):
        r = probe_layer(activations, labels, layer_idx, actual_splits)
        results.append(r)
        label = "embed" if layer_idx == 0 else f"L{layer_idx:2d}   "
        print(
            f"  {label}  AUROC = {r['mean_auroc']:.4f} ± {r['std_auroc']:.4f}"
            f"  [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        )

    peak = max(results, key=lambda r: r["mean_auroc"])
    print("-" * 55)
    print(f"  Peak: Layer {peak['layer']} → AUROC {peak['mean_auroc']:.4f}")

    return results


# ── 2. Sparse Lasso probe ────────────────────────────────────────────────────

def probe_layer_sparse(
    activations: list,
    labels: list,
    layer_idx: int,
    C: float = 0.1,
) -> Dict:
    """
    L1-regularized probe at a specific layer to find sparse predictive dimensions.

    The non-zero coefficients identify which of the 768 hidden dimensions carry
    the hallucination signal at this layer. This is the neuron-level mechanistic
    finding — moving from "which layer" (dense probe) to "which dimensions".

    Strong sparsity (few non-zero dims) = tight, interpretable signal.
    High density (many non-zero dims) = distributed representation.

    Args:
        C: inverse regularization strength. Lower = sparser. 0.1 is a good start.

    Returns dict with: layer, C, n_nonzero, sparsity, top20_dims, auroc_train
    """
    X = np.stack([
        act["hidden_states"][layer_idx].mean(axis=0)
        for act in activations
    ])  # [N, 768]
    y = np.array(labels)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        penalty="l1", C=C, solver="liblinear", max_iter=1000, random_state=42
    )
    clf.fit(X_scaled, y)

    coef       = clf.coef_[0]                              # [768]
    nonzero    = np.where(np.abs(coef) > 1e-6)[0]
    top20_dims = nonzero[np.argsort(np.abs(coef[nonzero]))[::-1]][:20].tolist()

    # In-sample AUROC (indicative — no CV needed here, we care about the dimensions)
    y_prob     = clf.predict_proba(X_scaled)[:, 1]
    auroc_train = float(roc_auc_score(y, y_prob))

    # Cross-validated test AUROC (5-fold) on the full dataset
    from sklearn.pipeline import Pipeline as _P
    skf_cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring_s = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    sparse_pipe = _P([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            penalty="l1", C=C, solver="liblinear", max_iter=1000, random_state=42
        )),
    ])
    cv_res2      = cross_validate(sparse_pipe, X, y, cv=skf_cv, scoring=scoring_s)
    auroc_cv     = float(cv_res2["test_auroc"].mean())
    auroc_cv_std = float(cv_res2["test_auroc"].std())

    return {
        "layer":        layer_idx,
        "C":            C,
        "n_nonzero":    int(len(nonzero)),
        "hidden_dim":   int(len(coef)),
        "sparsity":     round(1.0 - len(nonzero) / len(coef), 4),
        "top20_dims":   top20_dims,
        "coef":         coef.tolist(),
        "auroc_train":  round(auroc_train, 4),
        "auroc_cv":     round(auroc_cv, 4),
        "auroc_cv_std": round(auroc_cv_std, 4),
    }


# ── 3. Token position probe ──────────────────────────────────────────────────

POSITION_NAMES = {
    0:  "First token",
    -3: "Last-3",
    -2: "Last-2",
    -1: "Last token",
}


def probe_layer_token_positions(
    activations: list,
    labels: list,
    layer_idx: int,
    positions: list = None,
    n_splits: int = 5,
) -> Dict:
    """
    Probe individual token positions at a single layer.

    Instead of mean-pooling the sequence, we probe each position separately.
    This reveals WHERE IN THE SEQUENCE the hallucination signal concentrates.

    For decoder-only models (GPT-2), the hypothesis is: the last token position
    carries the most signal (it accumulates all context and is where the next
    token is predicted from). But this is a testable claim.

    Args:
        positions: list of token positions to probe (default: [0, -3, -2, -1])
                   Positions are clipped to valid range per sample.

    Returns dict: {pos: auroc, ...} for each requested position, plus 'mean_pool'
    """
    if positions is None:
        positions = [0, -3, -2, -1]

    y = np.array(labels)
    actual_splits = _safe_cv(y, n_splits)

    results = {"layer": layer_idx, "mean_pool": None}

    # Mean-pool baseline
    X_mean = np.stack([act["hidden_states"][layer_idx].mean(axis=0) for act in activations])
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}
    cv = cross_validate(_build_dense_probe(), X_mean, y, cv=skf, scoring=scoring)
    results["mean_pool"] = round(float(cv["test_auroc"].mean()), 4)

    # Per-position probing
    for pos in positions:
        pos_vecs = []
        valid_indices = []
        for idx, act in enumerate(activations):
            hs = act["hidden_states"][layer_idx]  # [seq_len, hidden_dim]
            seq_len = hs.shape[0]
            # Clip position to valid range
            if pos >= 0:
                safe_pos = min(pos, seq_len - 1)
            else:
                safe_pos = max(-seq_len, pos)
            pos_vecs.append(hs[safe_pos])
            valid_indices.append(idx)

        X_pos = np.stack(pos_vecs)  # [N, hidden_dim]
        cv_pos = cross_validate(_build_dense_probe(), X_pos, y, cv=skf, scoring=scoring)
        results[pos] = round(float(cv_pos["test_auroc"].mean()), 4)

    return results


def probe_token_positions_all_layers(
    activations: list,
    labels: list,
    positions: list = None,
    n_splits: int = 5,
) -> List[Dict]:
    """
    Probe token positions across all layers → produces a [num_layers × num_positions]
    2D AUROC matrix, visualized as a heatmap.

    This is novel: no paper in the layer-probing literature shows this 2D view.
    Mean pooling hides whether the signal is concentrated at specific positions.
    """
    if positions is None:
        positions = [0, -3, -2, -1]

    num_layers = activations[0]["hidden_states"].shape[0]
    print(f"Token-Position Probing: {num_layers} layers × {len(positions)+1} positions, N={len(labels)}")

    results = []
    for layer_idx in range(num_layers):
        r = probe_layer_token_positions(activations, labels, layer_idx, positions, n_splits)
        results.append(r)
        lbl = "embed" if layer_idx == 0 else f"L{layer_idx:2d}"
        pos_str = "  ".join([f"pos{p}={r[p]:.3f}" for p in positions])
        print(f"  {lbl:<8}  mean={r['mean_pool']:.3f}  {pos_str}")

    return results
