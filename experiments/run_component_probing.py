"""
Phase 7: Component Decomposition — FFN vs Attention Probing.

Inspired by ReDeEP (ICLR 2025), adapted for closed-book QA.
Uses saved activations (requires updated run_extraction.py with component hooks).
Fast (~2–3 min on CPU for 12 layers × 2 components × 5-fold CV).

Residual stream decomposition:
  hidden_state[l+1] = hidden_state[l] + attn_output[l] + ffn_output[l]

We probe attn_output[l] and ffn_output[l] separately at each layer.

Research question: for closed-book factual QA hallucination, which component
carries the hallucination signal — FFN (parametric memory failure) or
Attention (context composition failure)?

ReDeEP (RAG setting) found: FFN dominates.
Our closed-book setting tests the same question independently.

Usage:
    python experiments/run_component_probing.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probing.component_probe import probe_all_layers_components
from src.evaluation.metrics import plot_component_comparison

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
LOGS_DIR    = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 7: Component Decomposition (FFN vs Attention) ===\n")

    # --- Load saved activations ---
    act_path = DATA_DIR / "activations.pkl"
    if not act_path.exists():
        print(f"ERROR: {act_path} not found. Run run_extraction.py first.")
        return
    with open(act_path, "rb") as f:
        activations = pickle.load(f)

    # Verify component outputs are present
    if "ffn_outputs" not in activations[0]:
        print("ERROR: Component outputs (ffn_outputs, attn_outputs) not found in activations.")
        print("The activations were saved with an older version of activations.py.")
        print("Re-run: python experiments/run_extraction.py")
        return

    labels_path = DATA_DIR / "labels.npy"
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found. Run run_extraction.py first.")
        return
    labels = np.load(labels_path).tolist()

    print(f"Loaded {len(activations)} activations, {sum(labels)} correct / "
          f"{len(labels) - sum(labels)} hallucinated")
    print(f"Component shapes: FFN {activations[0]['ffn_outputs'].shape}, "
          f"Attn {activations[0]['attn_outputs'].shape}\n")

    # --- Run component probing ---
    component_results = probe_all_layers_components(activations, labels, n_splits=5)

    # --- Save results ---
    np.save(LOGS_DIR / "component_results.npy", component_results)
    print(f"\nSaved component results → {LOGS_DIR / 'component_results.npy'}")

    # --- Summary ---
    ffn_results  = component_results["ffn"]
    attn_results = component_results["attn"]

    ffn_aurocs  = [r["mean_auroc"] for r in ffn_results]
    attn_aurocs = [r["mean_auroc"] for r in attn_results]
    peak_ffn    = int(np.argmax(ffn_aurocs))
    peak_attn   = int(np.argmax(attn_aurocs))
    ffn_wins    = sum(f >= a for f, a in zip(ffn_aurocs, attn_aurocs))

    print(f"\n--- Component Decomposition Summary ---")
    print(f"  FFN  peak: Layer {peak_ffn}  (AUROC {ffn_aurocs[peak_ffn]:.4f})")
    print(f"  Attn peak: Layer {peak_attn}  (AUROC {attn_aurocs[peak_attn]:.4f})")
    print(f"  FFN dominates in {ffn_wins}/12 layers")
    print(f"  Attn dominates in {12 - ffn_wins}/12 layers")

    if ffn_wins >= 7:
        print("\n  Finding: FFN component carries stronger hallucination signal.")
        print("  Interpretation: Closed-book hallucination is primarily a PARAMETRIC")
        print("  RECALL FAILURE — the feedforward memory retrieval is the weak link.")
        print("  Consistent with ReDeEP's finding for RAG (same mechanism, different context).")
    elif ffn_wins <= 5:
        print("\n  Finding: Attention component carries stronger hallucination signal.")
        print("  Interpretation: Closed-book hallucination is primarily an ATTENTION")
        print("  COMPOSITION FAILURE — the model fails to properly integrate token context.")
        print("  NOVEL FINDING: Distinct from RAG hallucination (where FFN dominates).")
    else:
        print("\n  Finding: Mixed — both components contribute roughly equally.")
        print("  Interpretation: Hallucination is distributed across both mechanisms.")

    # --- Plot ---
    supervised_path = LOGS_DIR / "layer_probe_results.npy"
    supervised_results = None
    if supervised_path.exists():
        supervised_results = np.load(supervised_path, allow_pickle=True).tolist()

    plot_component_comparison(
        ffn_results,
        attn_results,
        supervised_results=supervised_results,
        save_path=str(PLOTS_DIR / "component_comparison.png"),
    )
    print(f"Saved component comparison plot → {PLOTS_DIR / 'component_comparison.png'}")

    print("\nDone. All phases complete. Launch dashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
