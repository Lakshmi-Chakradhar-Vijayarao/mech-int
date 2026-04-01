"""
Phase 4: Layer-wise Representation Probing — the mechanistic core of MECH-INT.

Three probing analyses in one script:
  A. Dense supervised probe (original) — WHICH LAYER?
  B. Sparse Lasso probe at peak layer   — WHICH DIMENSIONS?
  C. Token-position probe               — WHERE IN THE SEQUENCE?

Outputs:
  results/logs/layer_probe_results.npy      — dense probe per-layer AUROC
  results/logs/sparse_probe_results.npy     — sparse probe for peak layer
  results/logs/token_position_results.npy   — position × layer AUROC matrix
  results/plots/layer_probing_curve.png
  results/plots/token_position_heatmap.png

Usage:
    python experiments/run_layer_probing.py
"""

import sys
import numpy as np
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probing.layer_probe import (
    probe_all_layers,
    probe_layer_sparse,
    probe_token_positions_all_layers,
)
from src.evaluation.metrics import (
    plot_layer_probing_curve,
    plot_token_position_heatmap,
)

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
LOGS_DIR    = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 4: Layer-wise Representation Probing ===\n")

    # --- Load activations and labels ---
    act_path = DATA_DIR / "activations.pkl"
    if not act_path.exists():
        raise FileNotFoundError(
            f"activations.pkl not found at {act_path}.\n"
            "Run: python experiments/run_extraction.py first."
        )
    with open(act_path, "rb") as f:
        activations = pickle.load(f)

    labels_path = DATA_DIR / "labels.npy"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.npy not found at {labels_path}.\n"
            "Run: python experiments/run_extraction.py first."
        )
    labels = np.load(labels_path).tolist()

    print(f"Loaded {len(activations)} prompts")
    print(f"Hidden state shape: {activations[0]['hidden_states'].shape}\n")

    # ══════════════════════════════════════════════════════════════
    # A. Dense supervised probe (which layer carries the signal?)
    # ══════════════════════════════════════════════════════════════
    print("── A. Dense Layer Probe ─────────────────────────────────")
    results = probe_all_layers(activations, labels, n_splits=5)

    np.save(LOGS_DIR / "layer_probe_results.npy", results)
    print(f"Saved: {LOGS_DIR / 'layer_probe_results.npy'}")

    aurocs     = [r["mean_auroc"] for r in results]
    peak_layer = int(np.argmax(aurocs))
    peak_auroc = aurocs[peak_layer]

    print(f"\n  Embedding layer AUROC : {aurocs[0]:.4f}")
    print(f"  Peak layer (L{peak_layer}) AUROC  : {peak_auroc:.4f}")
    print(f"  Final layer AUROC     : {aurocs[-1]:.4f}")

    plot_layer_probing_curve(
        results,
        save_path=str(PLOTS_DIR / "layer_probing_curve.png"),
    )

    # ══════════════════════════════════════════════════════════════
    # B. Sparse Lasso probe at peak layer (which dimensions?)
    # ══════════════════════════════════════════════════════════════
    print(f"\n── B. Sparse Lasso Probe at Peak Layer (L{peak_layer}) ──────")
    sparse_result = probe_layer_sparse(activations, labels, layer_idx=peak_layer, C=0.1)

    np.save(LOGS_DIR / "sparse_probe_results.npy", sparse_result)
    print(f"  Non-zero dimensions : {sparse_result['n_nonzero']} / {sparse_result['hidden_dim']}")
    print(f"  Sparsity            : {sparse_result['sparsity']:.1%}")
    print(f"  Train AUROC (in-sample) : {sparse_result['auroc_train']:.4f}  "
          f"(NOTE: in-sample, not generalisation estimate)")
    print(f"  CV AUROC (5-fold)   : {sparse_result['auroc_cv']:.4f} "
          f"± {sparse_result['auroc_cv_std']:.4f}  ← use this for claims")
    print(f"  Top-10 dim indices  : {sparse_result['top20_dims'][:10]}")

    if sparse_result['n_nonzero'] < 50:
        print(f"\n  Tight mechanistic finding: only {sparse_result['n_nonzero']} of 768")
        print("  dimensions are predictive. The hallucination signal is concentrated.")
    elif sparse_result['n_nonzero'] < 200:
        print(f"\n  Moderate sparsity: {sparse_result['n_nonzero']} dims are predictive.")
    else:
        print(f"\n  Dense representation: {sparse_result['n_nonzero']} dims are active.")
        print("  The signal is distributed — no single neuron cluster dominates.")

    # ══════════════════════════════════════════════════════════════
    # C. Token position probing (where in the sequence?)
    # ══════════════════════════════════════════════════════════════
    print("\n── C. Token-Position Probing (Layer × Position AUROC) ────")
    positions = [0, -3, -2, -1]
    pos_results = probe_token_positions_all_layers(
        activations, labels, positions=positions, n_splits=5
    )

    np.save(LOGS_DIR / "token_position_results.npy", pos_results)
    print(f"Saved: {LOGS_DIR / 'token_position_results.npy'}")

    plot_token_position_heatmap(
        pos_results,
        positions=positions,
        save_path=str(PLOTS_DIR / "token_position_heatmap.png"),
    )

    # Find the best (layer, position) pair
    best_auroc = 0.0
    best_layer = 0
    best_pos   = "mean_pool"
    for r in pos_results:
        for key in ["mean_pool"] + positions:
            if r.get(key, 0) > best_auroc:
                best_auroc = r[key]
                best_layer = r["layer"]
                best_pos   = key

    pos_name = "mean-pool" if best_pos == "mean_pool" else f"position {best_pos}"
    print(f"\n  Best (layer, position): Layer {best_layer}, {pos_name}  "
          f"→ AUROC {best_auroc:.4f}")

    if best_pos == -1 or best_pos == "mean_pool":
        print("  Last-token (or mean-pool) is most predictive — expected for GPT-2,")
        print("  as the last token accumulates full context for next-token prediction.")
    else:
        print(f"  Unexpected: {pos_name} is most predictive at Layer {best_layer}.")
        print("  This may indicate early-layer positional encoding carries signal.")

    print("\n=== Phase 4 Complete ===")
    print("Next: python experiments/run_subspace_probing.py")


if __name__ == "__main__":
    main()
