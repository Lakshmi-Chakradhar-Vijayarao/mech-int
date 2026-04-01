"""
Phase 2-3: Activation Extraction + Feature Engineering.

Scientific question
-------------------
To probe GPT-2's internal representations, we need all component outputs captured
during a single forward pass per prompt. This script runs that forward pass and saves
every internal signal needed by every downstream analysis (Steps 3–12).

What is captured and why
------------------------
  hidden_states [13, seq_len, 768]
      The residual stream at every layer. h[0] = embedding; h[1]-h[12] = transformer
      layer outputs. These are the input to all probing analyses (Steps 4A/4B/4C).
      Because h[l] = h[l-1] + Attn[l] + FFN[l], the final hidden state is a linear
      sum — the mathematical basis for Direct Logit Attribution (Step 10).

  attentions [12, 12, seq_len, seq_len]
      Attention weight tensors at each layer, each head. Used in Step 11 (attention
      pattern analysis) and Step 12 (head-level DLA via V-projection).

  logits [seq_len, vocab_size]
      The model's output logit distribution at each sequence position. Used in
      Step 3 (surface feature engineering: entropy, confidence gap, logit variance).

  attn_outputs [12, seq_len, 768]   ← captured via forward hooks
      The attention sub-block output BEFORE residual addition. This is what
      MultiHeadAttn(LN1(h[l-1])) writes into the stream. Used in Step 7 (component
      decomposition, ReDeEP-style): probing Attn[l] vs FFN[l] independently.

  ffn_outputs [12, seq_len, 768]    ← captured via forward hooks
      The FFN sub-block output BEFORE residual addition. This is FFN(LN2(h_mid)).
      High FFN probe AUROC at L8 = parametric memory (factual recall) failure.

Forward hook mechanism
----------------------
The HuggingFace GPT-2 model natively returns hidden_states and attentions via
output_hidden_states=True / output_attentions=True. Component-level outputs
(attn_outputs, ffn_outputs) require explicit hooks because they are intermediate
values not exposed in the standard API:

  block.attn   → attention sub-block (MultiHeadAttention)
  block.mlp    → FFN sub-block (two Linear layers + GELU)

A hook registered on block.attn fires AFTER attention computation, BEFORE the
output is added to the residual stream — exactly the decomposition we need.

Feature engineering (Phase 3)
------------------------------
Six scalar features per prompt are extracted from the logits to form the baseline
surface predictor (Step 3): mean_entropy, max_entropy, logit_variance,
confidence_gap, attention_entropy, activation_norm.

Outputs
-------
  data/processed/activations.pkl     ~2.1 GB — gitignored (required for all probing)
  data/processed/features.npy        [N, 6]  — surface features for Step 3
  data/processed/labels.npy          [N]     — binary labels (1=correct, 0=hallucinated)

Usage
-----
    python experiments/run_extraction.py
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.load_model import load_gpt2
from src.extraction.activations import extract_activations
from src.features.engineer import build_feature_matrix, FEATURE_NAMES

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_labeled_data():
    """Load prompts and labels from data/processed/labeled.npy"""
    labeled_path = DATA_DIR / "labeled.pkl"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"Labeled data not found at {labeled_path}. "
            "Run: python experiments/prepare_data.py first."
        )
    with open(labeled_path, "rb") as f:
        data = pickle.load(f)
    return data["prompts"], data["labels"]


def main():
    print("=== Phase 2: Activation Extraction ===\n")

    model, tokenizer, device = load_gpt2()

    prompts, labels = load_labeled_data()
    print(f"Loaded {len(prompts)} labeled prompts")
    print(f"  Correct:      {sum(labels)}")
    print(f"  Hallucinated: {len(labels) - sum(labels)}\n")

    # --- Extract activations ---
    print("Extracting activations (this takes a few minutes)...")
    activations = []
    for prompt in tqdm(prompts, desc="Forward passes"):
        act = extract_activations(prompt, model, tokenizer, device)
        activations.append(act)

    # --- Save raw activations ---
    raw_path = DATA_DIR / "activations.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump(activations, f)
    print(f"\nSaved raw activations: {raw_path}")

    # --- Spot-check shapes ---
    sample = activations[0]
    print(f"\nActivation shapes for first prompt:")
    print(f"  hidden_states:  {sample['hidden_states'].shape}")
    print(f"  attentions:     {sample['attentions'].shape}")
    print(f"  logits:         {sample['logits'].shape}")

    # --- Build feature matrix ---
    print("\n=== Phase 3: Feature Engineering ===\n")
    X, y = build_feature_matrix(activations, labels)
    print(f"Feature matrix: {X.shape}  (samples x features)")
    print(f"Labels:         {y.shape}")
    print(f"\nFeature names: {FEATURE_NAMES}")
    print(f"Feature stats (mean per feature):\n  {X.mean(axis=0)}")

    # --- Save feature matrix ---
    np.save(DATA_DIR / "features.npy", X)
    np.save(DATA_DIR / "labels.npy", y)
    print(f"\nSaved features → {DATA_DIR / 'features.npy'}")
    print(f"Saved labels   → {DATA_DIR / 'labels.npy'}")
    print("\nDone. Ready for run_predictor.py")


if __name__ == "__main__":
    main()
