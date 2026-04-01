"""
Phase 11: Attention Pattern Analysis — which L8 heads discriminate hallucinations?

The relative DLA analysis revealed L8 attention has an 80% per-unit difference
between correct and hallucinated samples. This experiment unpacks that signal
at head resolution, examining entropy and positional attention mass for all
12 heads at L8 (no model reload — uses saved attention weight tensors).

Outputs:
  results/logs/attention_pattern_results.npy
  results/plots/attention_patterns_L8.png

Usage:
    python experiments/run_attention_patterns.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.attention_patterns import (
    compute_attention_stats_all,
    summarise_attention_patterns,
)
from src.evaluation.metrics import plot_attention_patterns

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
LOGS_DIR    = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LAYER_IDX = 8   # primary target: 80% relative DLA signal found here


def main():
    print("=== Phase 11: Attention Pattern Analysis ===\n")

    act_path    = DATA_DIR / "activations.pkl"
    labels_path = DATA_DIR / "labels.npy"

    if not act_path.exists():
        raise FileNotFoundError("Run run_extraction.py first.")

    with open(act_path, "rb") as f:
        activations = pickle.load(f)
    labels = np.load(labels_path).tolist()

    print(f"Loaded {len(activations)} samples")
    print(f"Attention tensor shape: {activations[0]['attentions'].shape}")
    print(f"Analysing layer L{LAYER_IDX} (80% relative DLA signal)\n")

    # ── Compute per-sample attention stats ───────────────────────────────────
    print("Computing attention statistics from last-token position...")
    stats = compute_attention_stats_all(activations, layer_idx=LAYER_IDX)

    # ── Group summary ─────────────────────────────────────────────────────────
    print("\nAttention pattern comparison (correct vs hallucinated):")
    summary = summarise_attention_patterns(stats, labels)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = LOGS_DIR / "attention_pattern_results.npy"
    np.save(save_path, summary)
    print(f"\nSaved: {save_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = str(PLOTS_DIR / f"attention_patterns_L{LAYER_IDX}.png")
    plot_attention_patterns(summary, layer_idx=LAYER_IDX, save_path=plot_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    top_h = summary["top_head"]
    top_auroc = summary["combined_aurocs"][top_h]
    ent_diff  = summary["entropy_diff"]

    print("\n=== Phase 11 Summary ===")
    print(f"  Top discriminative head : H{top_h}  (AUROC {top_auroc:.4f})")
    print(f"  Entropy diff C−H (H{top_h}) : {ent_diff[top_h]:+.4f}"
          f"  ({'correct more diffuse' if ent_diff[top_h] > 0 else 'correct more focused'})")
    print(f"  Mean entropy diff across all heads: {ent_diff.mean():+.4f}")

    if top_auroc > 0.58:
        print(f"\n  Finding: H{top_h} at L{LAYER_IDX} shows meaningful discrimination.")
        print("  This is consistent with the 80% relative DLA signal — one or a few")
        print("  specific heads are driving the attention-level hallucination signal.")
    else:
        print("\n  Finding: No single head strongly dominates.")
        print("  The L8 attention signal is distributed across multiple heads.")

    print("\n=== Phase 11 Complete ===")
    print("Next: python experiments/run_head_dla.py")


if __name__ == "__main__":
    main()
