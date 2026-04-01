"""
Phase 12: Head-Level Direct Logit Attribution at L8.

Decomposes the L8 attention DLA (80% relative signal, from Phase 10) into
per-head contributions. Identifies which of the 12 GPT-2 attention heads at
L8 are individually responsible for the correct-vs-hallucinated difference.

Connects to Phase 7 (attention head ablation): ablation used accuracy-drop
(near-zero signal). Head-level DLA uses the logit-difference metric which
is 50x more sensitive — these two results are not contradictory.

Outputs:
  results/logs/head_dla_L8_results.npy
  results/plots/head_dla_L8.png

Usage:
    python experiments/run_head_dla.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.load_model import load_gpt2
from src.analysis.head_dla import (
    compute_head_dla_all,
    summarise_head_dla,
)
from src.evaluation.metrics import plot_head_dla

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
LOGS_DIR    = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LAYER_IDX = 8


def main():
    print("=== Phase 12: Head-Level DLA at L8 ===\n")

    act_path    = DATA_DIR / "activations.pkl"
    labels_path = DATA_DIR / "labels.npy"

    if not act_path.exists():
        raise FileNotFoundError("Run run_extraction.py first.")

    with open(act_path, "rb") as f:
        activations = pickle.load(f)
    labels = np.load(labels_path).tolist()

    print(f"Loaded {len(activations)} samples\n")

    model, tokenizer, device = load_gpt2()
    print()

    # ── Compute per-head DLA ──────────────────────────────────────────────────
    print(f"Computing head-level DLA at L{LAYER_IDX}...")
    print("(Requires one forward pass per sample for LN1 + V projection)\n")
    results = compute_head_dla_all(activations, model, device, layer_idx=LAYER_IDX)

    # ── Summarise ─────────────────────────────────────────────────────────────
    summary = summarise_head_dla(results, labels, layer_idx=LAYER_IDX)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = LOGS_DIR / f"head_dla_L{LAYER_IDX}_results.npy"
    np.save(save_path, summary)
    print(f"\nSaved: {save_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = str(PLOTS_DIR / f"head_dla_L{LAYER_IDX}.png")
    plot_head_dla(summary, save_path=plot_path)

    # ── Summary print ─────────────────────────────────────────────────────────
    top_abs = summary["top_head_abs"]
    top_rel = summary["top_head_rel"]
    rel_pct = summary["head_dla_rel_diff_pct"]

    print("\n=== Phase 12 Summary ===")
    print(f"  Top head (abs diff) : H{top_abs}  "
          f"diff={summary['head_dla_diff'][top_abs]:+.4f}")
    print(f"  Top head (rel diff) : H{top_rel}  "
          f"rel={rel_pct[top_rel]:+.1f}%")

    if abs(rel_pct[top_rel]) > 20:
        print(f"\n  Strong head-level signal: H{top_rel} at L{LAYER_IDX} shows "
              f"{rel_pct[top_rel]:+.1f}% relative DLA difference.")
        print("  This head is the primary mechanistic contributor to the")
        print("  L8 attention hallucination signal.")
    else:
        print("\n  Distributed signal: no single head dominates strongly.")
        print("  The L8 attention signal is spread across multiple heads.")

    # Cross-check with layer-level DLA
    layer_dlas = [r["layer_dla"] for r in results]
    correct_layer = np.mean([dla for dla, l in zip(layer_dlas, labels) if l == 1])
    hall_layer    = np.mean([dla for dla, l in zip(layer_dlas, labels) if l == 0])
    print(f"\n  Layer-total DLA (sum of heads):  C={correct_layer:.4f}  H={hall_layer:.4f}")
    print(f"  Layer attn DLA diff (from Phase 10): +0.5327")
    print(f"  Head-decomp sum diff: {correct_layer - hall_layer:+.4f}")
    print("  (Should approximately match — difference is due to shared c_proj bias)")

    print("\n=== Phase 12 Complete ===")


if __name__ == "__main__":
    main()
