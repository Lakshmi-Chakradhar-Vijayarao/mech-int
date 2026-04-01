"""
Direct Logit Attribution (DLA) — decompose the final prediction into
additive contributions from every attention and FFN component.

Methodology: Elhage et al. (2021) "A Mathematical Framework for Transformer Circuits"
Extended here to compare DLA profiles for correct vs hallucinated samples.

Core Idea
---------
The residual stream is a pure sum:

  hidden_states[-1][-1] = embedding[-1]
                        + sum_{l=0}^{11}  attn_output[l][-1]
                        + sum_{l=0}^{11}  ffn_output[l][-1]

The final logit for token t is (approximately, ignoring LayerNorm nonlinearity):

  logit(t) ≈ W_unembed[t] · hidden_states[-1][-1]
           = W_unembed[t] · embedding[-1]
           + sum_l  W_unembed[t] · attn_output[l][-1]    ← attn DLA
           + sum_l  W_unembed[t] · ffn_output[l][-1]     ← FFN DLA

Each term is a scalar — the signed contribution of that component to the
logit for the generated token.

  Positive DLA → component pushed the model toward the generated token
  Negative DLA → component pushed the model away from the generated token

For CORRECT samples:   positive FFN DLA at L8-9 → FFN is retrieving the right fact
For HALLUCINATED samples: FFN DLA at L8-9 may be pushing toward the wrong token

The difference in DLA profiles between correct and hallucinated samples
quantifies WHERE in the computational graph the failure originates.

Note on LayerNorm
-----------------
The final logit uses lm_head(LayerNorm(h[-1])), and LayerNorm is not linear.
The DLA here is an approximation (using h[-1] directly, before LayerNorm).
This is standard in the field and gives the correct qualitative picture.
For exact attribution one would need to linearise the LayerNorm.
"""

import numpy as np
import torch
from typing import List, Dict


def compute_dla_single(
    act: dict,
    model,
    device: torch.device,
) -> Dict:
    """
    Compute DLA for a single sample.

    Returns dict with:
        generated_id   : int — the token the model generated (argmax at final layer)
        embedding_dla  : float — embedding contribution
        attn_dla       : np.ndarray [num_layers] — per-layer attention DLA
        ffn_dla        : np.ndarray [num_layers] — per-layer FFN DLA
        total_check    : float — should ≈ logit(generated_id) from model
    """
    lm_head = model.lm_head   # [vocab_size, hidden_dim]

    hidden_states = act["hidden_states"]   # [num_layers+1, seq_len, hidden_dim]
    attn_outputs  = act["attn_outputs"]    # [num_layers, seq_len, hidden_dim]
    ffn_outputs   = act["ffn_outputs"]     # [num_layers, seq_len, hidden_dim]
    num_layers    = attn_outputs.shape[0]

    # Determine generated token from final layer, last position
    final_h  = torch.tensor(hidden_states[-1, -1:, :], dtype=torch.float32).to(device)
    final_h_norm = model.transformer.ln_f(final_h)
    final_logits = lm_head(final_h_norm)[0]
    generated_id = int(final_logits.argmax().item())

    # Unembedding direction for the generated token: [hidden_dim]
    w_unembed = lm_head.weight[generated_id].detach().cpu().numpy()   # [hidden_dim]

    # Embedding DLA (last token position)
    emb_vec     = hidden_states[0, -1, :]      # [hidden_dim] — raw embedding
    emb_dla     = float(np.dot(w_unembed, emb_vec))

    # Per-layer component DLA (last token position)
    attn_dla = np.array([
        float(np.dot(w_unembed, attn_outputs[l, -1, :]))
        for l in range(num_layers)
    ])

    ffn_dla = np.array([
        float(np.dot(w_unembed, ffn_outputs[l, -1, :]))
        for l in range(num_layers)
    ])

    total_check = emb_dla + attn_dla.sum() + ffn_dla.sum()

    return {
        "generated_id":  generated_id,
        "embedding_dla": emb_dla,
        "attn_dla":      attn_dla,
        "ffn_dla":       ffn_dla,
        "total_check":   total_check,
    }


def compute_dla_all(
    activations: list,
    model,
    device: torch.device,
) -> List[Dict]:
    """
    Compute DLA for all samples. Returns list of per-sample dicts.
    """
    return [compute_dla_single(act, model, device) for act in activations]


def summarise_dla(
    dla_results: List[Dict],
    labels: list,
) -> Dict:
    """
    Aggregate DLA into mean profiles for correct vs hallucinated samples.

    Returns dict:
        num_layers             : int
        correct_mean_attn_dla  : np.ndarray [num_layers]
        correct_mean_ffn_dla   : np.ndarray [num_layers]
        hallucinated_mean_attn : np.ndarray [num_layers]
        hallucinated_mean_ffn  : np.ndarray [num_layers]
        attn_dla_diff          : np.ndarray [num_layers]  correct - hallucinated
        ffn_dla_diff           : np.ndarray [num_layers]  correct - hallucinated
        peak_ffn_diff_layer    : int — layer with largest |FFN DLA difference|
        peak_attn_diff_layer   : int — layer with largest |Attn DLA difference|
    """
    num_layers = len(dla_results[0]["attn_dla"])

    correct_attn = np.stack([r["attn_dla"] for r, l in zip(dla_results, labels) if l == 1])
    correct_ffn  = np.stack([r["ffn_dla"]  for r, l in zip(dla_results, labels) if l == 1])
    hall_attn    = np.stack([r["attn_dla"] for r, l in zip(dla_results, labels) if l == 0])
    hall_ffn     = np.stack([r["ffn_dla"]  for r, l in zip(dla_results, labels) if l == 0])

    c_attn_mean = correct_attn.mean(axis=0)
    c_ffn_mean  = correct_ffn.mean(axis=0)
    h_attn_mean = hall_attn.mean(axis=0)
    h_ffn_mean  = hall_ffn.mean(axis=0)

    attn_diff = c_attn_mean - h_attn_mean
    ffn_diff  = c_ffn_mean  - h_ffn_mean

    print(f"\n  {'Layer':<8}  {'FFN DLA (C)':>12}  {'FFN DLA (H)':>12}  {'Diff':>8}")
    print("  " + "-" * 46)
    for l in range(num_layers):
        sign = "+" if ffn_diff[l] >= 0 else ""
        print(f"  L{l:<7}  {c_ffn_mean[l]:>12.4f}  {h_ffn_mean[l]:>12.4f}  "
              f"{sign}{ffn_diff[l]:.4f}")

    peak_ffn  = int(np.argmax(np.abs(ffn_diff)))
    peak_attn = int(np.argmax(np.abs(attn_diff)))

    # ── Relative DLA: diff / mean_magnitude × 100 ────────────────────────────
    attn_mag = (np.abs(c_attn_mean) + np.abs(h_attn_mean)) / 2
    ffn_mag  = (np.abs(c_ffn_mean)  + np.abs(h_ffn_mean))  / 2

    attn_rel = np.where(attn_mag > 1e-6, attn_diff / attn_mag * 100, 0.0)
    ffn_rel  = np.where(ffn_mag  > 1e-6, ffn_diff  / ffn_mag  * 100, 0.0)

    peak_ffn_rel  = int(np.argmax(np.abs(ffn_rel)))
    peak_attn_rel = int(np.argmax(np.abs(attn_rel)))

    print(f"\n  Peak FFN  difference (abs): L{peak_ffn}  diff={ffn_diff[peak_ffn]:+.4f}")
    print(f"  Peak Attn difference (abs): L{peak_attn}  diff={attn_diff[peak_attn]:+.4f}")
    print(f"  Peak FFN  difference (rel): L{peak_ffn_rel}  "
          f"rel={ffn_rel[peak_ffn_rel]:+.1f}%")
    print(f"  Peak Attn difference (rel): L{peak_attn_rel}  "
          f"rel={attn_rel[peak_attn_rel]:+.1f}%")

    print(f"\n  {'Layer':<8}  {'ATN_rel%':>9}  {'FFN_rel%':>9}")
    print("  " + "-" * 32)
    for l in range(num_layers):
        a_mark = " <-" if l == peak_attn_rel else ""
        f_mark = " <-" if l == peak_ffn_rel  else ""
        print(f"  L{l:<7}  {attn_rel[l]:>+8.1f}%  {ffn_rel[l]:>+8.1f}%{a_mark}{f_mark}")

    return {
        "num_layers":             num_layers,
        "correct_mean_attn_dla":  c_attn_mean,
        "correct_mean_ffn_dla":   c_ffn_mean,
        "hallucinated_mean_attn": h_attn_mean,
        "hallucinated_mean_ffn":  h_ffn_mean,
        "attn_dla_diff":          attn_diff,
        "ffn_dla_diff":           ffn_diff,
        "attn_dla_magnitude":     attn_mag,
        "ffn_dla_magnitude":      ffn_mag,
        "attn_dla_rel_pct":       attn_rel,
        "ffn_dla_rel_pct":        ffn_rel,
        "peak_ffn_diff_layer":    peak_ffn,
        "peak_attn_diff_layer":   peak_attn,
        "peak_ffn_rel_layer":     peak_ffn_rel,
        "peak_attn_rel_layer":    peak_attn_rel,
        "n_correct":              correct_attn.shape[0],
        "n_hallucinated":         hall_attn.shape[0],
    }
