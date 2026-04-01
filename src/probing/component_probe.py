"""
FFN vs Attention component decomposition probing.

Methodology: ReDeEP (ICLR 2025) — adapted for closed-book QA.
"ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability"

Core Insight
------------
The transformer residual stream is a sum of contributions:

  hidden_state[l+1] = hidden_state[l]
                    + attn_output[l]    ← Attention: context composition
                    + ffn_output[l]     ← FFN: parametric memory retrieval

ReDeEP (ICLR 2025) showed that in RAG hallucination:
  - Hallucination occurs when FFN over-emphasizes parametric knowledge
  - Copying Heads (attention) fail to integrate retrieved context

For closed-book QA (TruthfulQA, our setting), there is NO external context.
The research question becomes:

  Is hallucination primarily a PARAMETRIC RECALL FAILURE (FFN signal is
  more predictive) or an ATTENTION COMPOSITION FAILURE (attention signal
  is more predictive)?

This distinction is novel for closed-book settings — ReDeEP only tested RAG.

Method
------
For each transformer layer l (0–11 for GPT-2):
  1. Extract mean-pooled FFN output:  ffn_vec[l]   ∈ R^768
  2. Extract mean-pooled Attn output: attn_vec[l]  ∈ R^768
  3. Train separate LR probes with 5-fold CV
  4. Compare AUROC: whichever is higher tells us the dominant failure mode

The 2D comparison curve — FFN AUROC vs. Attention AUROC vs. layer depth —
is the component decomposition finding of this project.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score
from typing import List, Dict


def _build_probe() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
    ])


def probe_component_at_layer(
    activations: list,
    labels: list,
    layer_idx: int,
    component: str,
    n_splits: int = 5,
) -> Dict:
    """
    Train a probe on one component (FFN or attention) at a given transformer block.

    Args:
        activations : list of activation dicts from extract_activations()
        labels      : ground-truth labels (1=correct, 0=hallucinated)
        layer_idx   : transformer block index 0..11 (component layers, not residual layers)
        component   : 'ffn' or 'attn'
        n_splits    : CV folds

    Returns dict: layer, component, mean_auroc, std_auroc, ci_low, ci_high
    """
    key = "ffn_outputs" if component == "ffn" else "attn_outputs"

    if key not in activations[0]:
        raise RuntimeError(
            f"'{key}' not found in activations. "
            "Re-run experiments/run_extraction.py with the updated activations.py."
        )

    # Mean-pool over sequence: [seq_len, hidden_dim] → [hidden_dim]
    X = np.stack([
        act[key][layer_idx].mean(axis=0)
        for act in activations
    ])  # [N, hidden_dim]
    y = np.array(labels)

    actual_splits = min(n_splits, min(int(y.sum()), int((y == 0).sum())))
    actual_splits = max(2, actual_splits)

    probe = _build_probe()
    skf   = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    scoring = {"auroc": make_scorer(roc_auc_score, response_method="predict_proba")}

    cv = cross_validate(probe, X, y, cv=skf, scoring=scoring)
    mean_auroc = float(cv["test_auroc"].mean())
    std_auroc  = float(cv["test_auroc"].std())

    return {
        "layer":      layer_idx,
        "component":  component,
        "mean_auroc": round(mean_auroc, 4),
        "std_auroc":  round(std_auroc, 4),
        "ci_low":     round(max(0.0, mean_auroc - 1.96 * std_auroc), 4),
        "ci_high":    round(min(1.0, mean_auroc + 1.96 * std_auroc), 4),
    }


def probe_all_layers_components(
    activations: list,
    labels: list,
    n_splits: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Probe both FFN and attention outputs at each transformer layer.

    Returns dict with keys 'ffn' and 'attn', each a list of per-layer results.
    Covers 12 transformer blocks (GPT-2 has no FFN/Attn at embedding layer).

    This produces the component decomposition comparison:
      "At the peak layer, which component carries more hallucination signal?"

    In RAG settings (ReDeEP): FFN dominates (parametric over-reliance).
    For closed-book QA: we test the same question fresh.
    """
    if "ffn_outputs" not in activations[0]:
        raise RuntimeError(
            "Component outputs not found in activations. "
            "Re-run experiments/run_extraction.py with the updated activations.py."
        )

    num_layers = activations[0]["ffn_outputs"].shape[0]  # 12 for GPT-2
    n = len(labels)

    print(f"Component Probing (FFN vs Attention): {num_layers} layers, N={n}")
    print(f"  [Comparing parametric recall (FFN) vs context composition (Attn)]")
    print(f"\n  {'Layer':<8}  {'FFN AUROC':>10}  {'Attn AUROC':>11}  {'Dominant':>10}")
    print("  " + "-" * 46)

    ffn_results  = []
    attn_results = []

    for layer_idx in range(num_layers):
        r_ffn  = probe_component_at_layer(activations, labels, layer_idx, "ffn",  n_splits)
        r_attn = probe_component_at_layer(activations, labels, layer_idx, "attn", n_splits)
        ffn_results.append(r_ffn)
        attn_results.append(r_attn)

        dominant = "FFN  ★" if r_ffn["mean_auroc"] >= r_attn["mean_auroc"] else "Attn ★"
        print(
            f"  L{layer_idx:<7}  {r_ffn['mean_auroc']:.4f}      "
            f"{r_attn['mean_auroc']:.4f}      {dominant}"
        )

    # Summary
    ffn_aurocs  = [r["mean_auroc"] for r in ffn_results]
    attn_aurocs = [r["mean_auroc"] for r in attn_results]
    peak_ffn    = int(np.argmax(ffn_aurocs))
    peak_attn   = int(np.argmax(attn_aurocs))
    ffn_wins    = sum(f >= a for f, a in zip(ffn_aurocs, attn_aurocs))

    print("  " + "-" * 46)
    print(f"  FFN  peak: Layer {peak_ffn}  (AUROC {ffn_aurocs[peak_ffn]:.4f})")
    print(f"  Attn peak: Layer {peak_attn}  (AUROC {attn_aurocs[peak_attn]:.4f})")
    print(f"  FFN dominates in {ffn_wins}/{num_layers} layers")

    if ffn_wins > num_layers // 2:
        print("\n  Interpretation: Hallucination signal is primarily in FFN outputs.")
        print("  → Closed-book hallucination resembles parametric recall failure")
        print("    (consistent with ReDeEP's finding for RAG, but for different reasons).")
    else:
        print("\n  Interpretation: Hallucination signal is primarily in Attention outputs.")
        print("  → Closed-book hallucination is an attention composition failure,")
        print("    distinct from RAG hallucination (where FFN dominates in ReDeEP).")

    return {"ffn": ffn_results, "attn": attn_results}
