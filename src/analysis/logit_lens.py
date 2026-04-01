"""
Logit Lens — Layer-by-layer token prediction from intermediate hidden states.

Methodology: Nostalgebraist (2020) "interpreting GPT: the logit lens"
Extended here to visualise the hallucination cascade.

Core Idea
---------
At every transformer layer, the residual stream already contains a partial
representation of the final answer. By projecting each layer's hidden state
through the final LayerNorm + unembedding matrix, we can ask:

  "If the model stopped HERE — what token would it predict?"

For CORRECT samples: the right token should appear early and stay stable.
For HALLUCINATED samples: the right token may briefly appear at early layers,
then fade and be replaced by the wrong token — exactly at L8-9 (the cascade).

This transforms our probing finding ("signal peaks at L9") into a direct
visualisation of how the model's belief evolves layer by layer.

Method
------
For each layer L and each sample:
  1. Take hidden_state[L]  ∈ R^[seq_len, 768]
  2. Apply the final LayerNorm: h_norm = ln_f(hidden_state[L])
  3. Project through unembedding: logits = h_norm @ W_unembed.T  ∈ R^[seq_len, vocab]
  4. Softmax → probabilities
  5. Record the probability assigned to the model's own generated token
     (what the model ultimately output) at the last-token position

Two views:
  A. Per-sample trace: probability of the generated token at each layer
     → shows when hallucinated samples "commit" to the wrong token
  B. Group comparison: mean prob of generated token, correct vs hallucinated
     → shows at which layer the two groups diverge

Gold-token extension (Phase 11 update)
---------------------------------------
The original "generated_prob" tracks the model's own output token — giving a
null divergence because both groups commit equally to their own output.
The gold-token track compares probability of the *correct answer's first token*
across layers.  For hallucinated samples this gold token is different from the
generated token, so the two curves genuinely diverge at the layer where the
model's belief shifts from correct to wrong.

Pass `gold_token_ids` (one int per sample, or None to skip) to enable.
"""

import numpy as np
import torch
from typing import List, Dict, Optional


def build_gold_token_map(
    prompts: list,
    raw_dataset,
    tokenizer,
) -> dict:
    """
    Map each prompt to the first token of its TruthfulQA gold correct answer.

    Args:
        prompts     : list of "Q: {question}\\nA:" strings
        raw_dataset : HuggingFace Dataset (truthful_qa generation split)
        tokenizer   : GPT-2 tokenizer

    Returns:
        dict mapping prompt → gold_token_id (int)
        Prompts with no match or no tokenisable answer map to None.
    """
    # Build question → correct_answers lookup
    q_to_answers = {item["question"]: item["correct_answers"]
                    for item in raw_dataset}

    result = {}
    for prompt in prompts:
        # Extract question from "Q: {question}\nA:"
        q = prompt[3:].split("\nA:")[0].strip()
        answers = q_to_answers.get(q)
        if not answers:
            result[prompt] = None
            continue

        # Tokenise each correct answer, collect first token
        first_tokens = []
        for ans in answers:
            ans = ans.strip()
            if not ans:
                continue
            ids = tokenizer.encode(ans)
            if ids:
                first_tokens.append(ids[0])

        if not first_tokens:
            result[prompt] = None
            continue

        # Use the most common first token
        from collections import Counter
        result[prompt] = Counter(first_tokens).most_common(1)[0][0]

    matched = sum(1 for v in result.values() if v is not None)
    print(f"  Gold token mapping: {matched}/{len(prompts)} prompts matched")
    return result


def compute_logit_lens(
    activations: list,
    model,
    device: torch.device,
    gold_token_ids: Optional[List[Optional[int]]] = None,
) -> List[Dict]:
    """
    Compute logit-lens token probabilities for all samples across all layers.

    For each sample, at each layer, records:
      - top1_token_id    : the most probable next token at that layer
      - top1_prob        : its probability
      - generated_prob   : probability of the model's actual generated token
                           (the token with highest prob at the FINAL layer)

    Args:
        activations    : list of activation dicts from extract_activations()
        model          : GPT2LMHeadModel (frozen, used only for ln_f + lm_head)
        device         : torch device
        gold_token_ids : optional list (one per sample) of gold answer first-token
                         IDs.  When provided each result dict also contains
                         `gold_prob` [num_layers] and `gold_id` int.

    Returns:
        List of per-sample dicts, each containing:
            layer_probs    : np.ndarray [num_layers, vocab_size]
            top1_ids       : np.ndarray [num_layers]
            top1_probs     : np.ndarray [num_layers]
            generated_prob : np.ndarray [num_layers] — prob of model's output token
            generated_id   : int
            gold_prob      : np.ndarray [num_layers] — prob of gold answer token
                             (only present when gold_token_ids is provided)
            gold_id        : int or None
    """
    ln_f    = model.transformer.ln_f
    lm_head = model.lm_head

    if gold_token_ids is None:
        gold_token_ids = [None] * len(activations)

    results = []

    for act, gold_id in zip(activations, gold_token_ids):
        hidden_states = act["hidden_states"]  # [num_layers, seq_len, hidden_dim]
        num_layers    = hidden_states.shape[0]

        # Determine generated token = top-1 at the FINAL layer, last token position
        final_hs  = torch.tensor(hidden_states[-1, -1:, :], dtype=torch.float32).to(device)
        final_norm = ln_f(final_hs)                   # [1, hidden_dim]
        final_logits = lm_head(final_norm)[0]          # [vocab_size]
        generated_id = int(final_logits.argmax().item())

        vocab_size     = lm_head.weight.shape[0]
        layer_probs    = np.zeros((num_layers, vocab_size), dtype=np.float32)
        top1_ids       = np.zeros(num_layers, dtype=np.int32)
        top1_probs     = np.zeros(num_layers, dtype=np.float32)
        generated_prob = np.zeros(num_layers, dtype=np.float32)
        gold_prob      = np.zeros(num_layers, dtype=np.float32) if gold_id is not None else None

        for l in range(num_layers):
            hs     = torch.tensor(hidden_states[l, -1:, :], dtype=torch.float32).to(device)
            norm   = ln_f(hs)
            logits = lm_head(norm)[0]

            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            layer_probs[l]    = probs
            top1_ids[l]       = int(probs.argmax())
            top1_probs[l]     = float(probs.max())
            generated_prob[l] = float(probs[generated_id])
            if gold_prob is not None:
                gold_prob[l] = float(probs[gold_id])

        entry = {
            "layer_probs":    layer_probs,
            "top1_ids":       top1_ids,
            "top1_probs":     top1_probs,
            "generated_prob": generated_prob,
            "generated_id":   generated_id,
            "gold_id":        gold_id,
        }
        if gold_prob is not None:
            entry["gold_prob"] = gold_prob

        results.append(entry)

    return results


def summarise_logit_lens(
    lens_results: List[Dict],
    labels: list,
) -> Dict:
    """
    Aggregate logit lens results into group-level statistics.

    Computes mean generated-token probability per layer, separately for
    correct (label=1) and hallucinated (label=0) samples.

    The divergence point — where the two curves separate — is the layer where
    the model commits to correct vs. wrong answers differently.

    Returns dict with generated-token and (if available) gold-token statistics:
        num_layers               : int
        correct_mean_prob        : np.ndarray [num_layers]  (generated token)
        hallucinated_mean_prob   : np.ndarray [num_layers]
        divergence_layer         : int
        layer_separation         : np.ndarray [num_layers]
        -- gold-token fields (present only when gold_prob is in results) --
        correct_gold_mean_prob   : np.ndarray [num_layers]
        hallucinated_gold_mean_prob : np.ndarray [num_layers]
        gold_divergence_layer    : int
        gold_layer_separation    : np.ndarray [num_layers]
        gold_matched_n           : int  samples with a valid gold token
    """
    num_layers = len(lens_results[0]["generated_prob"])
    labels_arr = np.array(labels)

    correct_probs = np.stack([
        r["generated_prob"] for r, l in zip(lens_results, labels) if l == 1
    ])
    hall_probs = np.stack([
        r["generated_prob"] for r, l in zip(lens_results, labels) if l == 0
    ])

    correct_mean      = correct_probs.mean(axis=0)
    hallucinated_mean = hall_probs.mean(axis=0)
    separation        = np.abs(correct_mean - hallucinated_mean)
    divergence_layer  = int(separation.argmax())

    result = {
        "num_layers":             num_layers,
        "correct_mean_prob":      correct_mean,
        "hallucinated_mean_prob": hallucinated_mean,
        "divergence_layer":       divergence_layer,
        "layer_separation":       separation,
        "n_correct":              int((labels_arr == 1).sum()),
        "n_hallucinated":         int((labels_arr == 0).sum()),
    }

    # ── Gold-token analysis (only when gold_prob is available) ────────────────
    has_gold = all("gold_prob" in r for r in lens_results)
    if has_gold:
        # Filter to samples that had a successfully matched gold token
        c_gold = np.stack([
            r["gold_prob"] for r, l in zip(lens_results, labels)
            if l == 1 and r.get("gold_id") is not None
        ])
        h_gold = np.stack([
            r["gold_prob"] for r, l in zip(lens_results, labels)
            if l == 0 and r.get("gold_id") is not None
        ])
        gold_matched = sum(
            1 for r in lens_results if r.get("gold_id") is not None
        )

        if c_gold.size > 0 and h_gold.size > 0:
            c_gold_mean   = c_gold.mean(axis=0)
            h_gold_mean   = h_gold.mean(axis=0)
            gold_sep      = np.abs(c_gold_mean - h_gold_mean)
            gold_div_layer = int(gold_sep.argmax())

            print(f"\n  Gold-token divergence layer: L{gold_div_layer}  "
                  f"(max separation {gold_sep.max():.4f})")
            print(f"  Gold-matched samples: {gold_matched}/{len(labels)}")

            result.update({
                "correct_gold_mean_prob":      c_gold_mean,
                "hallucinated_gold_mean_prob": h_gold_mean,
                "gold_divergence_layer":       gold_div_layer,
                "gold_layer_separation":       gold_sep,
                "gold_matched_n":              gold_matched,
            })

    return result
