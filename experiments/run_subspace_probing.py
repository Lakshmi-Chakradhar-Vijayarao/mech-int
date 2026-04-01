"""
Phase 6: SVD Subspace Probing — Unsupervised Hallucination Subspace Identification.

Inspired by HaloScope (NeurIPS 2024 Spotlight).
Uses saved activations — no GPT-2 re-run needed. Fast (~10s on CPU).

This answers the question: does the hallucination signal have intrinsic geometric
structure in activation space, independent of any training supervision?

A strong AUROC from an unsupervised SVD subspace score means the model's own
representation geometry separates truthful from hallucinated outputs — the signal
is "baked in" to the activation space, not just linearly imposed by the probe.

Usage:
    python experiments/run_subspace_probing.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probing.subspace_probe import probe_all_layers_subspace
from src.evaluation.metrics import plot_subspace_vs_supervised

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
LOGS_DIR    = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 6: SVD Subspace Probing (Unsupervised) ===\n")

    # --- Load saved activations ---
    act_path = DATA_DIR / "activations.pkl"
    if not act_path.exists():
        print(f"ERROR: {act_path} not found. Run run_extraction.py first.")
        return
    with open(act_path, "rb") as f:
        activations = pickle.load(f)

    labels_path = DATA_DIR / "labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run run_extraction.py first.")
        return
    labels = np.load(labels_path).tolist()

    print(f"Loaded {len(activations)} activations, {sum(labels)} correct / "
          f"{len(labels) - sum(labels)} hallucinated\n")

    # --- Run subspace probing ---
    subspace_results = probe_all_layers_subspace(activations, labels, k=10)

    # --- Save results ---
    np.save(LOGS_DIR / "subspace_results.npy", subspace_results)
    print(f"\nSaved subspace results → {LOGS_DIR / 'subspace_results.npy'}")

    # --- Summary stats ---
    aurocs     = [r["subspace_auroc"] for r in subspace_results]
    peak_layer = int(np.argmax(aurocs))
    print(f"\n--- SVD Subspace Summary ---")
    print(f"  Embedding layer AUROC : {aurocs[0]:.4f}")
    print(f"  Peak layer           : L{peak_layer}  (AUROC {aurocs[peak_layer]:.4f})")
    print(f"  Final layer AUROC    : {aurocs[-1]:.4f}")

    if aurocs[peak_layer] > 0.60:
        print("\n  Unsupervised signal confirmed: model's geometry separates")
        print("  truthful from hallucinated representations without supervision.")
    else:
        print("\n  Weak unsupervised signal: geometric structure is not strongly")
        print("  separating. Supervised probe may still work if labels are informative.")

    # --- Plot: subspace vs supervised (if supervised results exist) ---
    supervised_path = LOGS_DIR / "layer_probe_results.npy"
    if supervised_path.exists():
        supervised_results = np.load(supervised_path, allow_pickle=True).tolist()
        plot_subspace_vs_supervised(
            supervised_results,
            subspace_results,
            save_path=str(PLOTS_DIR / "subspace_vs_supervised.png"),
        )
        print(f"Saved comparison plot → {PLOTS_DIR / 'subspace_vs_supervised.png'}")
    else:
        print("\n  [Tip] Run run_layer_probing.py first, then re-run this script")
        print("  to generate the supervised vs. unsupervised comparison plot.")

    print("\nDone. Proceed to run_component_probing.py")


if __name__ == "__main__":
    main()
