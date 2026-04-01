"""
Activation Steering — Representation-level intervention via direction injection.

Methodology: Representation Engineering (Zou et al., 2023)
"Representation Engineering: A Top-Down Approach to AI Transparency"

Core Idea
---------
If the hallucination signal lives as a direction in the residual stream,
we can test this causally: shift test representations along that direction
and measure whether probe AUROC improves.

Experiment Design
-----------------
For each transformer layer L:
  1. Compute "truthfulness direction" from TRAIN hidden states:
         direction_L = mean(correct[L]) - mean(hallucinated[L])
         direction_L = direction_L / ||direction_L||
  2. Apply manual steering to TEST hidden states:
         X_test_steered = X_test[L] + alpha * direction_L
  3. Evaluate probe (trained on unsteered TRAIN) on steered TEST
  4. Repeat with a RANDOM orthogonal direction (control baseline)

Findings:
  - If the found direction improves probe AUROC above the random baseline,
    the direction encodes class-discriminative structure beyond chance
  - If L9 shows the highest improvement, it validates L9 as the critical layer
  - The alpha curve shows the optimal injection strength and degradation

This is an important distinction from naive circular steering:
  - Probe trained on UNSTEERED train data
  - Direction found on TRAIN set (no test leakage)
  - Evaluated on HELD-OUT test set
  - Compared against random direction (controls for "any steering helps")
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from typing import List, Dict


# ── Direction computation ─────────────────────────────────────────────────────

def compute_steering_direction(
    activations: list,
    labels: list,
    layer_idx: int,
    use_last_token: bool = True,
) -> np.ndarray:
    """
    Compute the unit-norm truthfulness direction at a given layer.

    direction = mean(correct_hidden_states[L]) - mean(hallucinated_hidden_states[L])

    Args:
        activations   : list of activation dicts (train set only — no leakage)
        labels        : ground-truth labels (1=correct, 0=hallucinated)
        layer_idx     : residual stream layer index
        use_last_token: if True, use last-token hidden state; else mean-pool

    Returns:
        direction: np.ndarray [hidden_dim] — unit-norm steering vector
    """
    correct_vecs, hallucinated_vecs = [], []

    for act, label in zip(activations, labels):
        hs  = act["hidden_states"][layer_idx]   # [seq_len, hidden_dim]
        vec = hs[-1] if use_last_token else hs.mean(axis=0)
        (correct_vecs if label == 1 else hallucinated_vecs).append(vec)

    direction = np.mean(correct_vecs, axis=0) - np.mean(hallucinated_vecs, axis=0)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Direction near zero — check label balance.")
    return direction / norm


def _random_orthogonal_direction(direction: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Generate a random unit vector orthogonal to `direction`.
    Used as the null-hypothesis control baseline.
    """
    rng = np.random.default_rng(seed)
    random_vec = rng.standard_normal(direction.shape)
    # Gram-Schmidt: subtract projection onto direction
    random_vec -= np.dot(random_vec, direction) * direction
    random_vec /= np.linalg.norm(random_vec)
    return random_vec


# ── Steering on extracted representations ────────────────────────────────────

def _steer_representations(
    X: np.ndarray,
    direction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shift every sample by alpha * direction. Shape [N, hidden_dim]."""
    return X + alpha * direction[np.newaxis, :]


def _probe_auroc(X_train, y_train, X_test, y_test) -> float:
    """Train a logistic regression probe and return test AUROC."""
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)
    clf    = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_tr, y_train)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return float(roc_auc_score(y_test, y_prob))


# ── Full steering experiment ──────────────────────────────────────────────────

def run_steering_experiment(
    activations: list,
    labels: list,
    layer_idx: int = 9,
    alphas: List[float] = None,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Dict:
    """
    Steering experiment: found direction vs. random direction across alpha sweep.

    Protocol (no data leakage):
      1. Stratified 70/30 split
      2. Compute found direction on TRAIN hidden states at `layer_idx`
      3. Compute random orthogonal direction (null baseline)
      4. For each alpha:
           a. Steer TEST representations along found direction
           b. Steer TEST representations along random direction
           c. Evaluate probe (trained on unsteered TRAIN) on both steered TEST sets

    Returns dict with alphas, found_aurocs, random_aurocs, baseline_auroc,
    best_alpha, best_auroc, improvement, direction, train/test indices.
    """
    if alphas is None:
        alphas = [0, 5, 10, 15, 20, 30, 40, 50]

    y = np.array(labels)

    # ── 1. Stratified split ───────────────────────────────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))

    train_acts   = [activations[i] for i in train_idx]
    train_labels = y[train_idx]
    test_acts    = [activations[i] for i in test_idx]
    test_labels  = y[test_idx]

    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"Train — correct: {train_labels.sum()}  hallucinated: {(train_labels==0).sum()}")
    print(f"Test  — correct: {test_labels.sum()}  hallucinated: {(test_labels==0).sum()}")

    # ── 2. Build raw representation matrices ─────────────────────────────────
    X_train = np.stack([
        act["hidden_states"][layer_idx][-1] for act in train_acts
    ])
    X_test = np.stack([
        act["hidden_states"][layer_idx][-1] for act in test_acts
    ])

    # ── 3. Compute steering directions ───────────────────────────────────────
    found_dir  = compute_steering_direction(train_acts, train_labels.tolist(), layer_idx)
    random_dir = _random_orthogonal_direction(found_dir, seed=42)

    print(f"\nSteering at Layer {layer_idx} (last token)")
    print(f"  Found direction norm  : {np.linalg.norm(found_dir):.6f}")
    print(f"  Random direction norm : {np.linalg.norm(random_dir):.6f}")
    print(f"  Orthogonality check   : {abs(np.dot(found_dir, random_dir)):.6f} (should be ~0)")

    # ── 4. Alpha sweep ────────────────────────────────────────────────────────
    found_aurocs  = []
    random_aurocs = []

    print(f"\n{'Alpha':>7}  {'Found Dir AUROC':>16}  {'Random Dir AUROC':>17}  {'Delta':>7}")
    print("  " + "-" * 54)

    for alpha in alphas:
        X_test_found  = _steer_representations(X_test, found_dir,  alpha)
        X_test_random = _steer_representations(X_test, random_dir, alpha)

        auroc_found  = _probe_auroc(X_train, train_labels, X_test_found,  test_labels)
        auroc_random = _probe_auroc(X_train, train_labels, X_test_random, test_labels)

        found_aurocs.append(auroc_found)
        random_aurocs.append(auroc_random)

        delta = auroc_found - auroc_random
        sign  = "+" if delta >= 0 else ""
        print(f"  {alpha:>5}       {auroc_found:.4f}            {auroc_random:.4f}       "
              f"{sign}{delta:.4f}")

    baseline_auroc = found_aurocs[0]   # alpha=0 — same as random at 0
    best_idx       = int(np.argmax(found_aurocs))
    best_alpha     = alphas[best_idx]
    best_auroc     = found_aurocs[best_idx]
    improvement    = best_auroc - baseline_auroc

    print("  " + "-" * 54)
    print(f"  Baseline AUROC     : {baseline_auroc:.4f}")
    print(f"  Best found-dir     : {best_auroc:.4f}  at alpha={best_alpha}")
    print(f"  Net improvement    : {'+' if improvement>=0 else ''}{improvement:.4f}")

    max_random = max(random_aurocs[1:]) if len(random_aurocs) > 1 else random_aurocs[0]
    print(f"  Best random-dir    : {max_random:.4f}")
    margin = best_auroc - max_random
    print(f"  Found vs Random    : {'+' if margin>=0 else ''}{margin:.4f}  "
          f"({'found direction is more informative' if margin > 0.005 else 'marginal difference'})")

    if improvement > 0.01 and margin > 0.005:
        print("\n  Interpretation: The found truthfulness direction encodes")
        print("  class-discriminative structure that generalises from train to test,")
        print("  and outperforms a random orthogonal direction. This validates that")
        print("  the direction is a stable feature of GPT-2's representation geometry.")
    elif improvement > 0.005:
        print("\n  Interpretation: Modest improvement from found direction.")
        print("  The signal is present but the geometry is not sharply separated.")
    else:
        print("\n  Interpretation: Steering produces negligible improvement,")
        print("  consistent with the sparse, scale-limited signal seen in probing.")

    return {
        "alphas":          alphas,
        "found_aurocs":    found_aurocs,
        "random_aurocs":   random_aurocs,
        "baseline_auroc":  baseline_auroc,
        "best_alpha":      best_alpha,
        "best_auroc":      best_auroc,
        "improvement":     round(improvement, 4),
        "direction":       found_dir,
        "layer_idx":       layer_idx,
        "train_indices":   train_idx.tolist(),
        "test_indices":    test_idx.tolist(),
    }


def run_steering_layer_sweep(
    activations: list,
    labels: list,
    alphas: List[float] = None,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Dict:
    """
    Run steering experiment across ALL layers to find which layer's direction
    gives the highest AUROC improvement.

    The peak layer from steering should match the peak layer from probing (L9),
    providing independent validation of the cascade finding.

    Returns dict: layer_results (list of per-layer dicts), peak_layer, peak_improvement.
    """
    if alphas is None:
        alphas = [0, 10, 20, 30]   # reduced sweep for speed

    num_layers = activations[0]["hidden_states"].shape[0]
    y          = np.array(labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(sss.split(np.zeros(len(y)), y))

    train_acts   = [activations[i] for i in train_idx]
    train_labels = y[train_idx]
    test_acts    = [activations[i] for i in test_idx]
    test_labels  = y[test_idx]

    print(f"Layer sweep steering: {num_layers} layers, alphas={alphas}, N={len(labels)}")
    print(f"\n  {'Layer':<8}  {'Baseline':>9}  {'Best Steered':>13}  {'Improvement':>12}")
    print("  " + "-" * 48)

    layer_results = []

    for layer_idx in range(num_layers):
        X_train = np.stack([act["hidden_states"][layer_idx][-1] for act in train_acts])
        X_test  = np.stack([act["hidden_states"][layer_idx][-1] for act in test_acts])

        found_dir = compute_steering_direction(train_acts, train_labels.tolist(), layer_idx)

        aurocs = []
        for alpha in alphas:
            X_te_s  = _steer_representations(X_test, found_dir, alpha)
            aurocs.append(_probe_auroc(X_train, train_labels, X_te_s, test_labels))

        baseline   = aurocs[0]
        best       = max(aurocs)
        best_alpha = alphas[int(np.argmax(aurocs))]
        improvement = best - baseline

        lbl = "embed" if layer_idx == 0 else f"L{layer_idx}"
        sign = "+" if improvement >= 0 else ""
        print(f"  {lbl:<8}  {baseline:.4f}     {best:.4f}       {sign}{improvement:.4f}")

        layer_results.append({
            "layer":       layer_idx,
            "baseline":    baseline,
            "best_auroc":  best,
            "best_alpha":  best_alpha,
            "improvement": round(improvement, 4),
            "all_aurocs":  aurocs,
        })

    peak = max(layer_results, key=lambda r: r["improvement"])
    print("  " + "-" * 48)
    print(f"  Peak steering layer: L{peak['layer']}  "
          f"(improvement +{peak['improvement']:.4f}  best AUROC {peak['best_auroc']:.4f})")

    return {
        "layer_results":  layer_results,
        "peak_layer":     peak["layer"],
        "peak_improvement": peak["improvement"],
        "alphas":         alphas,
    }
