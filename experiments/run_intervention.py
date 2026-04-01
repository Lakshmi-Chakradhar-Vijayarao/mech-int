"""
Phase 5 (Step 5): Causal Intervention — Attention Head Ablation.

Scientific question
-------------------
The probing analyses in Step 4 identify which layers *correlate* with hallucination.
This step asks a stronger causal question: which attention heads are *necessary* for
the correct-output signal? Removing a necessary head should degrade detection;
the magnitude of the degradation measures that head's causal importance.

Method — zero-patch ablation
------------------------------
For each of the 12 × 12 = 144 attention heads:
  1. Register a forward hook that replaces the head's output slice with zeros.
  2. Run the probe on the modified activations.
  3. Measure: importance[l, h] = AUROC_baseline - AUROC_ablated_at_[l,h]

Positive importance:
  Zeroing the head HURTS detection → the head contributes to the correct-signal.
  It is doing something useful that enables the probe to distinguish correct from
  hallucinated answers.

Negative importance:
  Zeroing the head HELPS detection → the head actively contributes to hallucination
  or suppresses the discriminative signal. Removing it unmasks the correct signal.

Why ablation, not attention-weight analysis
--------------------------------------------
Visualizing attention weights shows WHERE a head attends. It does not reveal WHAT
the head writes into the residual stream or HOW MUCH it changes the downstream
prediction. A head can attend uniformly (high entropy, near-random weights) while
still writing a highly discriminative value into the residual — this is the H5
dissociation found later in Step 11/12.

Ablation is a causal intervention: it actually removes the head's output and
measures the consequence in logit space. This is a stronger claim than correlation.

Key findings
------------
  L11 H6 (importance +0.160): 5–8× larger than any other head.
  L11 H7 (importance +0.140): Second largest, also at L11.
  These two heads form the "output commitment gate" — they appear to lock the
  residual stream onto the final answer at L11, the last transformer layer.

  L0 H6 (importance −0.030): Negative importance — ablating this early-layer head
  actually IMPROVES detection. This head may be a "surface fluency" head that
  generates plausible-sounding completions regardless of factual accuracy.

Relationship to probing peak (L8–L9)
--------------------------------------
The ablation peaks at L11, while probing peaks at L8–L9. These are compatible:
  - L8–L9: Where the hallucination signal first appears in the residual stream
    (FFN over-retrieves a wrong fact; attention writes divergent logit signal)
  - L11: Where the commitment to the wrong answer is finalised
    (H6/H7 lock in the output)
These two findings describe different stages of the same cascade.

Ablation uses 100 prompts (not 534) for compute reasons: 144 heads × 534 samples
= 76,896 forward passes. Importance scores for lower-ranked heads (|importance| ≤ 0.02)
are noisy; only the L11 H6/H7 findings are statistically robust.

Outputs
-------
  results/logs/head_importance.npy        — [12, 12] importance matrix
  results/plots/ablation_heatmap.png      — heatmap of all 144 head importances

Usage
-----
    python experiments/run_intervention.py
"""

import sys
import numpy as np
import pickle
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.load_model import load_gpt2
from src.intervention.ablation import score_head_importance
from src.evaluation.metrics import plot_ablation_heatmap, plot_logit_lens

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 5: Causal Intervention ===\n")

    model, tokenizer, device = load_gpt2()

    # Load data
    with open(DATA_DIR / "labeled.pkl", "rb") as f:
        data = pickle.load(f)
    prompts = data["prompts"]
    labels = data["labels"]

    # Load best trained pipeline
    model_path = RESULTS_DIR / "models" / "logistic_regression.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found. Run run_predictor.py first."
        )
    pipeline = joblib.load(model_path)
    print(f"Loaded pipeline from {model_path}")

    # Use a subset for speed (ablation is O(layers * heads * n_prompts))
    n_subset = min(100, len(prompts))
    prompts_sub = prompts[:n_subset]
    labels_sub = labels[:n_subset]
    print(f"Running ablation on {n_subset} prompts x 144 heads (12 layers x 12 heads)...")

    # --- Attention Head Ablation ---
    importance = score_head_importance(
        prompts_sub, labels_sub,
        model, tokenizer, device,
        pipeline,
        num_layers=model.config.n_layer,
        num_heads=model.config.n_head,
    )

    np.save(LOGS_DIR / "head_importance.npy", importance)
    print(f"\nSaved importance matrix: {LOGS_DIR / 'head_importance.npy'}")

    # Top-5 most causal heads
    flat_idx = np.argsort(importance.ravel())[::-1][:5]
    top_heads = [(idx // model.config.n_head, idx % model.config.n_head) for idx in flat_idx]
    print("\nTop-5 most causal attention heads (layer, head) → accuracy drop:")
    for layer, head in top_heads:
        print(f"  Layer {layer:2d}, Head {head:2d}  →  {importance[layer, head]:+.4f}")

    plot_ablation_heatmap(
        importance,
        save_path=str(PLOTS_DIR / "ablation_heatmap.png"),
    )

    # --- Logit Lens ---
    print("\n--- Logit Lens ---")
    from src.extraction.activations import extract_activations
    # Use first prompt as example
    act = extract_activations(prompts[0], model, tokenizer, device)
    plot_logit_lens(
        act["hidden_states"],
        model, tokenizer,
        token_idx=-1,
        top_k=5,
        save_path=str(PLOTS_DIR / "logit_lens.png"),
    )

    print("\nDone. All intervention results saved to results/")


if __name__ == "__main__":
    main()
