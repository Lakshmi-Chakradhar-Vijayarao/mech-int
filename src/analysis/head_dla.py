"""
Phase 12: Head-Level Direct Logit Attribution — per-head contribution at L8.

The layer-level DLA showed L8 attention has an 80% relative difference between
correct and hallucinated samples. This module decomposes that layer-level signal
into individual head contributions, identifying *which* of the 12 heads at L8
are responsible.

Derivation
----------
GPT-2 attention output for layer L at position -1:

  attn_out[-1] = concat(head_0[-1], ..., head_11[-1]) @ W_O + b_O

where W_O = c_proj.weight  ∈ R^[hidden_dim, hidden_dim]  (Conv1D convention)
and b_O is shared across heads (cancels in correct − hallucinated difference).

For head h:
  head_out_h[-1] = attn_weights[L, h, -1, :] @ V_h          [head_dim]
  contribution_h = head_out_h[-1] @ W_O[h*hd : (h+1)*hd, :] [hidden_dim]
  head_dla_h     = dot(w_unembed[generated_id], contribution_h)

where V_h = LN1(hidden_states[L]) @ W_V_h + b_V_h,
and W_V_h is extracted from the combined c_attn weight matrix.

GPT-2 c_attn weight layout (Conv1D: [in_dim, 3*hidden_dim]):
  [:, 0       : hidden_dim]   → W_Q   (all heads interleaved)
  [:, hidden_dim : 2*hidden_dim] → W_K
  [:, 2*hidden_dim : 3*hidden_dim] → W_V

Within each Q/K/V block heads are stored as:
  head h occupies columns  h*head_dim : (h+1)*head_dim

Note on LayerNorm
-----------------
The computation uses LN1(hidden_states[L]) as input to c_attn, which requires
running the LayerNorm (non-linear). This is exact, not an approximation.
The final DLA uses lm_head WITHOUT the final LayerNorm (pre-ln_f), consistent
with the layer-level DLA in logit_attribution.py.
"""

import numpy as np
import torch
from typing import List, Dict


_HEAD_DIM = 64    # GPT-2: hidden_dim 768 / 12 heads = 64
_HIDDEN   = 768


def compute_head_dla_single(
    act: dict,
    model,
    device: torch.device,
    layer_idx: int = 8,
) -> Dict:
    """
    Compute per-head DLA contribution at one layer for a single sample.

    Returns dict:
        generated_id  int             — token the model generated
        head_dla      np.ndarray [12] — signed DLA contribution per head
        layer_dla     float           — total layer attn DLA (sum of heads + bias)
        head_dim      int             — head dimension (64 for GPT-2)
    """
    block     = model.transformer.h[layer_idx]
    lm_head   = model.lm_head

    # ── Determine generated token ─────────────────────────────────────────────
    hidden_states = act["hidden_states"]          # [13, seq, 768]
    final_h       = torch.tensor(
        hidden_states[-1, -1:, :], dtype=torch.float32
    ).to(device)
    final_logits  = lm_head(block.ln_2(final_h) if False else
                            model.transformer.ln_f(final_h))[0]
    generated_id  = int(final_logits.argmax().item())

    w_unembed = lm_head.weight[generated_id].detach().cpu().numpy()  # [768]

    # ── Compute V_h: project layer input through W_V ──────────────────────────
    # Input to attention = LN1(hidden_states[layer_idx])
    h_in = torch.tensor(
        hidden_states[layer_idx], dtype=torch.float32
    ).unsqueeze(0).to(device)                     # [1, seq, 768]

    with torch.no_grad():
        ln1_out = block.ln_1(h_in)                # [1, seq, 768]
        qkv     = block.attn.c_attn(ln1_out)      # [1, seq, 3*768]

    # Split into Q, K, V — each [1, seq, 768]
    _, _, v_all = qkv.split(_HIDDEN, dim=-1)
    v_all_np = v_all[0].detach().cpu().numpy()    # [seq, 768]

    # W_O (c_proj): Conv1D weight [768, 768]  (in_dim × out_dim)
    W_O = block.attn.c_proj.weight.detach().cpu().numpy()  # [768, 768]

    # Attention weights at this layer: [n_heads, seq, seq]
    attn_w = act["attentions"][layer_idx]         # [12, seq, seq]

    # ── Per-head DLA ──────────────────────────────────────────────────────────
    head_dla = np.zeros(12, dtype=np.float64)

    for h in range(12):
        lo, hi = h * _HEAD_DIM, (h + 1) * _HEAD_DIM

        # V_h for head h: slice the V block  [seq, head_dim]
        v_h = v_all_np[:, lo:hi]

        # Head output at last position: attn_weights[h, -1, :] @ V_h  [head_dim]
        head_out_last = attn_w[h, -1, :] @ v_h   # [head_dim]

        # Project through W_O slice for this head: [head_dim, 768]
        W_O_h         = W_O[lo:hi, :]             # [head_dim, 768]
        contribution  = head_out_last @ W_O_h     # [768]

        head_dla[h] = float(np.dot(w_unembed, contribution))

    # Layer total (sum of heads, ignoring shared bias)
    layer_dla = float(head_dla.sum())

    return {
        "generated_id": generated_id,
        "head_dla":     head_dla,
        "layer_dla":    layer_dla,
        "head_dim":     _HEAD_DIM,
    }


def compute_head_dla_all(
    activations: list,
    model,
    device: torch.device,
    layer_idx: int = 8,
) -> List[Dict]:
    """Compute head-level DLA for all samples at a given layer."""
    results = []
    for i, act in enumerate(activations):
        r = compute_head_dla_single(act, model, device, layer_idx)
        results.append(r)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(activations)} samples...")
    return results


def summarise_head_dla(
    results: List[Dict],
    labels: list,
    layer_idx: int = 8,
) -> Dict:
    """
    Aggregate head-level DLA into correct vs hallucinated group profiles.

    For each of the 12 heads, computes:
      - Mean DLA contribution for correct samples
      - Mean DLA contribution for hallucinated samples
      - Difference (correct − hallucinated)
      - Relative difference (diff / mean_magnitude × 100)

    Returns summary dict identifying the top discriminative head.
    """
    n_heads = 12

    correct_dla = np.stack([r["head_dla"] for r, l in zip(results, labels) if l == 1])
    hall_dla    = np.stack([r["head_dla"] for r, l in zip(results, labels) if l == 0])

    c_mean  = correct_dla.mean(axis=0)     # [12]
    h_mean  = hall_dla.mean(axis=0)        # [12]
    diff    = c_mean - h_mean              # [12]

    mag     = (np.abs(c_mean) + np.abs(h_mean)) / 2
    rel_diff = np.where(mag > 1e-6, diff / mag * 100, 0.0)

    top_abs = int(np.argmax(np.abs(diff)))
    top_rel = int(np.argmax(np.abs(rel_diff)))

    print(f"\n  Head-Level DLA at L{layer_idx}  "
          f"(C={correct_dla.shape[0]}, H={hall_dla.shape[0]})")
    print(f"\n  {'Head':<6}  {'C_DLA':>9}  {'H_DLA':>9}  {'Diff':>9}  "
          f"{'Magnitude':>10}  {'Rel%':>7}")
    print("  " + "-" * 58)
    for h in range(n_heads):
        sign = "+" if diff[h] >= 0 else ""
        abs_star = " A" if h == top_abs else ""
        rel_star = " R" if h == top_rel else ""
        star     = abs_star or rel_star
        print(f"  H{h:<5}  {c_mean[h]:>9.4f}  {h_mean[h]:>9.4f}  "
              f"{sign}{diff[h]:>8.4f}  {mag[h]:>10.4f}  "
              f"{rel_diff[h]:>+7.1f}%{star}")

    print("  " + "-" * 58)
    print(f"  Top head (abs diff) : H{top_abs}  diff={diff[top_abs]:+.4f}")
    print(f"  Top head (rel diff) : H{top_rel}  rel={rel_diff[top_rel]:+.1f}%")

    return {
        "layer_idx":                layer_idx,
        "n_heads":                  n_heads,
        "correct_mean_head_dla":    c_mean,
        "hallucinated_mean_head_dla": h_mean,
        "head_dla_diff":            diff,
        "head_dla_magnitude":       mag,
        "head_dla_rel_diff_pct":    rel_diff,
        "top_head_abs":             top_abs,
        "top_head_rel":             top_rel,
        "n_correct":                correct_dla.shape[0],
        "n_hallucinated":           hall_dla.shape[0],
    }
