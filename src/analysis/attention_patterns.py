"""
Phase 11: Attention Pattern Analysis — characterise head behaviour at L8.

At L8 the relative DLA analysis showed 80% per-unit signal difference between
correct and hallucinated samples.  This module unpacks *which heads* at L8
drive that signal, and *how* — by examining what each head attends to from
the final token position (where the next-token prediction originates).

Three complementary statistics per head, all derived from the saved
`attentions` tensor (no model reload required):

  1. Attention entropy
       H = -sum_j  a_j * log(a_j)   for the distribution over source tokens
       Low H  → head focuses sharply on one or a few tokens
       High H → head spreads attention broadly (diffuse)
       Hypothesis: correct samples show lower entropy at L8 (sharper focus
       on the question entity), hallucinated samples show higher entropy.

  2. Positional zone mass
       Zone fractions of attention from last position:
         zone_first  = a[0]              (the Q/BOS token)
         zone_middle = sum(a[1:-3])      (question body)
         zone_last3  = sum(a[-3:])       (\\nA: boundary tokens)
       Hypothesis: correct samples direct more attention toward the
       question body (the factual content zone).

  3. Peak position (normalised)
       argmax(a) / (seq_len - 1)  in [0,1].
       0 = first token, 1 = last token.

Per-head discrimination AUROC is computed for entropy and zone_last3,
identifying which specific heads carry the most label-predictive signal.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Dict

_EPS = 1e-9


# ── Per-sample statistics ─────────────────────────────────────────────────────

def _entropy(p: np.ndarray) -> float:
    q = np.clip(p, _EPS, 1.0)
    return float(-np.sum(q * np.log(q)))


def compute_attention_stats_single(act: dict, layer_idx: int) -> Dict:
    """
    Compute attention statistics from the last token position for one sample.

    Args:
        act       : activation dict containing 'attentions'
                    shape [n_layers, n_heads, seq_len, seq_len]
        layer_idx : transformer block to analyse

    Returns dict:
        entropy        [n_heads]  Shannon entropy of last-pos attention
        zone_first     [n_heads]  fraction attending to token 0
        zone_middle    [n_heads]  fraction attending to tokens 1..(seq-4)
        zone_last3     [n_heads]  fraction attending to last 3 tokens
        peak_pos_norm  [n_heads]  normalised position (0–1) of peak weight
        seq_len        int
    """
    attn = act["attentions"][layer_idx]   # [n_heads, seq_len, seq_len]
    n_heads, seq_len, _ = attn.shape

    last = attn[:, -1, :]                 # [n_heads, seq_len]  (from last pos)

    entropy       = np.array([_entropy(last[h]) for h in range(n_heads)])
    zone_first    = last[:, 0].copy()
    zone_last3    = last[:, -3:].sum(axis=1) if seq_len >= 3 else last.sum(axis=1)
    zone_middle   = last[:, 1:-3].sum(axis=1) if seq_len > 4 else np.zeros(n_heads)
    peak_idx      = last.argmax(axis=1).astype(float)
    peak_pos_norm = peak_idx / max(seq_len - 1, 1)

    return {
        "entropy":       entropy,
        "zone_first":    zone_first,
        "zone_middle":   zone_middle,
        "zone_last3":    zone_last3,
        "peak_pos_norm": peak_pos_norm,
        "seq_len":       seq_len,
    }


def compute_attention_stats_all(
    activations: list,
    layer_idx: int,
) -> List[Dict]:
    """Compute attention stats for every sample at a given layer."""
    return [compute_attention_stats_single(act, layer_idx) for act in activations]


# ── Group-level summary ───────────────────────────────────────────────────────

def summarise_attention_patterns(
    stats: List[Dict],
    labels: list,
) -> Dict:
    """
    Compare attention statistics between correct and hallucinated groups.

    For each head computes:
      - Mean entropy difference (correct − hallucinated)
      - Mean zone_last3 difference
      - Per-head AUROC using entropy as discriminative score
      - Per-head AUROC using zone_last3 as discriminative score

    The head with the highest combined AUROC is the primary suspect for
    the 80% relative DLA difference found at L8.

    Returns comprehensive summary dict.
    """
    n_heads    = len(stats[0]["entropy"])
    labels_arr = np.array(labels)
    n_correct  = int((labels_arr == 1).sum())
    n_hall     = int((labels_arr == 0).sum())

    # ── Stack per-group arrays ────────────────────────────────────────────────
    def _stack(key, label_val):
        return np.stack([s[key] for s, l in zip(stats, labels) if l == label_val])

    c_ent   = _stack("entropy",       1)   # [n_correct, n_heads]
    h_ent   = _stack("entropy",       0)
    c_last3 = _stack("zone_last3",    1)
    h_last3 = _stack("zone_last3",    0)
    c_mid   = _stack("zone_middle",   1)
    h_mid   = _stack("zone_middle",   0)
    c_peak  = _stack("peak_pos_norm", 1)
    h_peak  = _stack("peak_pos_norm", 0)

    c_ent_mean  = c_ent.mean(axis=0)       # [n_heads]
    h_ent_mean  = h_ent.mean(axis=0)
    ent_diff    = c_ent_mean - h_ent_mean  # + → correct more diffuse

    c_last3_mean = c_last3.mean(axis=0)
    h_last3_mean = h_last3.mean(axis=0)

    # ── Per-head AUROC (entropy and zone_last3 as scores) ────────────────────
    all_ent   = np.stack([s["entropy"]    for s in stats])   # [N, n_heads]
    all_last3 = np.stack([s["zone_last3"] for s in stats])   # [N, n_heads]

    head_ent_aurocs   = np.zeros(n_heads)
    head_last3_aurocs = np.zeros(n_heads)

    for h in range(n_heads):
        for scores_arr, out in [
            (all_ent[:, h],   head_ent_aurocs),
            (all_last3[:, h], head_last3_aurocs),
        ]:
            try:
                a_pos = roc_auc_score(labels_arr, scores_arr)
                a_neg = roc_auc_score(labels_arr, -scores_arr)
                out[h] = max(a_pos, a_neg)
            except Exception:
                out[h] = 0.5

    combined_aurocs = np.maximum(head_ent_aurocs, head_last3_aurocs)
    top_head        = int(combined_aurocs.argmax())

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n  Layer attention pattern comparison  "
          f"(N_correct={n_correct}, N_hall={n_hall})")
    print(f"\n  {'Head':<6}  {'C_ent':>7}  {'H_ent':>7}  {'Ent_diff':>9}  "
          f"{'Ent_AUC':>8}  {'L3_AUC':>8}  {'Best':>6}")
    print("  " + "-" * 62)
    for h in range(n_heads):
        sign = "+" if ent_diff[h] >= 0 else ""
        star = " ★" if h == top_head else ""
        print(f"  H{h:<5}  {c_ent_mean[h]:>7.3f}  {h_ent_mean[h]:>7.3f}  "
              f"{sign}{ent_diff[h]:>8.3f}  {head_ent_aurocs[h]:>8.4f}  "
              f"{head_last3_aurocs[h]:>8.4f}  {combined_aurocs[h]:>6.4f}{star}")

    print("  " + "-" * 62)
    print(f"  Top head: H{top_head}  AUROC {combined_aurocs[top_head]:.4f}")
    overall_ent_diff = (c_ent_mean - h_ent_mean).mean()
    sign = "+" if overall_ent_diff >= 0 else ""
    print(f"  Mean entropy diff (C-H): {sign}{overall_ent_diff:.4f}")

    return {
        "n_heads":                    n_heads,
        "correct_mean_entropy":       c_ent_mean,
        "hallucinated_mean_entropy":  h_ent_mean,
        "entropy_diff":               ent_diff,
        "head_entropy_aurocs":        head_ent_aurocs,
        "head_last3_aurocs":          head_last3_aurocs,
        "combined_aurocs":            combined_aurocs,
        "top_head":                   top_head,
        "correct_mean_zone_last3":    c_last3_mean,
        "hallucinated_mean_zone_last3": h_last3_mean,
        "correct_mean_zone_middle":   c_mid.mean(axis=0),
        "hallucinated_mean_zone_middle": h_mid.mean(axis=0),
        "correct_mean_peak_pos":      c_peak.mean(axis=0),
        "hallucinated_mean_peak_pos": h_peak.mean(axis=0),
        "n_correct":                  n_correct,
        "n_hallucinated":             n_hall,
    }
