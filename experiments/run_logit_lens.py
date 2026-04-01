"""
Phase 9 (Step 9): Logit Lens — Reading the Model's "Evolving Opinion" Layer by Layer.

Scientific question
-------------------
Can we watch the hallucination happen in real time across the transformer's layers?
By projecting each layer's hidden state through the unembedding matrix, we can read
what token the model would predict if processing stopped at that layer — a layer-by-layer
"movie" of the model's belief evolution.

Method — the logit lens (Nostalgebraist, 2020)
-----------------------------------------------
For each layer l and sequence position t:
  intermediate_logits[l, t] = W_unembed @ LayerNorm(h[l, t, :])
  intermediate_probs[l, t]  = softmax(intermediate_logits[l, t])

This shows the model's "instantaneous prediction" at every depth. For a correctly
answered question, the right answer token typically emerges in the probability
distribution around L8–L9 and stays there. For hallucinated questions, the right
answer token probability diverges downward at that same layer.

Two tracking modes
------------------
  1. Generated-token tracking (null control):
     At each layer, find argmax(intermediate_probs[l, -1, :]).
     This tracks when the model first "commits" to the token it ultimately generates.
     For hallucinated samples, the WRONG token is already the top prediction from
     the earliest layers (L1). This gives no useful diagnostic information — the
     model's decision is made very early and never changes. This is the null control.

  2. Gold-token tracking (the signal):
     For each sample, identify the first token of the reference correct answer.
     At each layer l, track intermediate_probs[l, -1, gold_token_id].
     - Correct samples: gold-token probability rises steadily through layers
     - Hallucinated samples: gold-token probability diverges DOWNWARD at L8

     This is a novel extension of the standard logit lens. Prior work (Nostalgebraist
     2020) tracks only the generated token. Our key insight: the generated-token view
     is a null result because the model's "chosen" token dominates from layer 1.
     The gold-token view asks the genuinely informative question: "when does the
     model give up on the correct answer?"

Why this matters
----------------
Gold-token divergence at L8 is independent confirmation of the mechanistic core —
reached without any probe, any classifier, or any attribution method. It uses only
the model's own unembedding matrix applied to its own intermediate representations.

This convergence (probing peaks at L8/L9, DLA peaks at L8/L9, logit lens diverges at L8)
across completely independent methods is the strongest evidence in the project for
the L8–L9 mechanistic core.

Coverage note
-------------
Not all 534 samples have a usable gold token. The correct answers from TruthfulQA
may be multi-word (e.g., "Canberra") or use words outside GPT-2's tokenizer vocabulary
in a way that maps to a single clean token ID. ~400/534 samples yield usable gold
token matches.

Outputs
-------
  results/logs/logit_lens_results.npy      — dict with gold/generated tracking data
  results/plots/logit_lens_divergence.png  — gold-token probability curves by class

Usage
-----
    python experiments/run_logit_lens.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.load_model import load_gpt2
from src.analysis.logit_lens import (
    compute_logit_lens,
    summarise_logit_lens,
    build_gold_token_map,
)
from src.evaluation.metrics import plot_logit_lens_divergence, plot_logit_lens_gold

DATA_DIR    = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
LOGS_DIR    = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 9: Logit Lens (+ Gold-Token Extension) ===\n")

    act_path    = DATA_DIR / "activations.pkl"
    labels_path = DATA_DIR / "labels.npy"
    labeled_path = DATA_DIR / "labeled.pkl"
    raw_path    = Path("data/raw/truthfulqa_raw.pkl")

    if not act_path.exists():
        raise FileNotFoundError("Run run_extraction.py first.")

    with open(act_path, "rb") as f:
        activations = pickle.load(f)
    labels = np.load(labels_path).tolist()

    print(f"Loaded {len(activations)} samples\n")

    model, tokenizer, device = load_gpt2()
    print()

    # ── Build gold token IDs (one per sample) ─────────────────────────────
    gold_token_ids = None
    if labeled_path.exists() and raw_path.exists():
        print("Building gold-token map from TruthfulQA correct_answers...")
        with open(labeled_path, "rb") as f:
            labeled = pickle.load(f)
        import pickle as _pk
        with open(raw_path, "rb") as f:
            raw_dataset = _pk.load(f)
        gold_map = build_gold_token_map(labeled["prompts"], raw_dataset, tokenizer)
        gold_token_ids = [gold_map.get(p) for p in labeled["prompts"]]
        n_matched = sum(1 for g in gold_token_ids if g is not None)
        print(f"  Gold tokens resolved: {n_matched}/{len(gold_token_ids)}\n")
    else:
        print("  (Gold-token mapping skipped: labeled.pkl or raw dataset not found)\n")

    # ── Compute logit lens ────────────────────────────────────────────────
    print("Computing logit lens across all layers...")
    lens_results = compute_logit_lens(
        activations, model, device, gold_token_ids=gold_token_ids
    )

    summary = summarise_logit_lens(lens_results, labels)

    np.save(LOGS_DIR / "logit_lens_results.npy", summary)
    print(f"\nSaved: {LOGS_DIR / 'logit_lens_results.npy'}")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_logit_lens_divergence(
        summary, save_path=str(PLOTS_DIR / "logit_lens_divergence.png")
    )

    if "correct_gold_mean_prob" in summary:
        plot_logit_lens_gold(
            summary, save_path=str(PLOTS_DIR / "logit_lens_gold.png")
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Logit Lens Summary ===")
    print(f"  Generated-token divergence layer : L{summary['divergence_layer']}")
    print(f"  Generated-token max separation   : {summary['layer_separation'].max():.4f}")
    if "gold_divergence_layer" in summary:
        print(f"  Gold-token divergence layer      : L{summary['gold_divergence_layer']}")
        print(f"  Gold-token max separation        : "
              f"{summary['gold_layer_separation'].max():.4f}")
        print(f"  Gold-matched samples             : {summary['gold_matched_n']}")
    print(f"  N correct         : {summary['n_correct']}")
    print(f"  N hallucinated    : {summary['n_hallucinated']}")

    print("\n=== Phase 9 Complete ===")
    print("Next: python experiments/run_logit_attribution.py")


if __name__ == "__main__":
    main()
