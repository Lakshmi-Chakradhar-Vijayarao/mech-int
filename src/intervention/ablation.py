"""
Causal intervention: attention head ablation.

Zero out specific attention heads during inference and measure
the impact on model correctness — establishing causality beyond correlation.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import GPT2LMHeadModel


def _register_ablation_hooks(
    model: GPT2LMHeadModel,
    heads_to_ablate: List[Tuple[int, int]],
) -> List:
    """
    Register forward hooks that zero out specified (layer, head) attention outputs.

    Args:
        model: GPT2LMHeadModel
        heads_to_ablate: list of (layer_idx, head_idx) tuples

    Returns:
        list of hook handles (call handle.remove() to clean up)
    """
    handles = []
    # Group by layer for efficiency
    layer_heads: Dict[int, List[int]] = {}
    for layer_idx, head_idx in heads_to_ablate:
        layer_heads.setdefault(layer_idx, []).append(head_idx)

    for layer_idx, head_list in layer_heads.items():
        attn_module = model.transformer.h[layer_idx].attn

        def make_hook(heads):
            def hook(module, input, output):
                # output[0] is the attention output tensor [batch, seq_len, hidden_dim]
                # GPT-2 splits hidden_dim into num_heads * head_dim
                hidden_dim = output[0].shape[-1]
                num_heads = module.num_heads
                head_dim = hidden_dim // num_heads

                out = output[0].clone()
                for h in heads:
                    out[:, :, h * head_dim:(h + 1) * head_dim] = 0.0

                return (out,) + output[1:]
            return hook

        handle = attn_module.register_forward_hook(make_hook(head_list))
        handles.append(handle)

    return handles


def run_with_ablation(
    prompt: str,
    model: GPT2LMHeadModel,
    tokenizer,
    device: torch.device,
    heads_to_ablate: List[Tuple[int, int]],
    max_length: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Run inference with specified heads zeroed out.

    Returns same dict as extract_activations().
    """
    from src.extraction.activations import extract_activations

    handles = _register_ablation_hooks(model, heads_to_ablate)
    try:
        result = extract_activations(prompt, model, tokenizer, device, max_length)
    finally:
        for h in handles:
            h.remove()

    return result


def score_head_importance(
    prompts: List[str],
    labels: List[int],
    model: GPT2LMHeadModel,
    tokenizer,
    device: torch.device,
    pipeline,
    num_layers: int = 12,
    num_heads: int = 12,
) -> np.ndarray:
    """
    Compute causal importance of each attention head by measuring
    accuracy drop when that head is ablated.

    Args:
        prompts: list of input prompts
        labels: ground truth labels (1=correct, 0=hallucinated)
        pipeline: trained sklearn Pipeline from classifier.py
        num_layers, num_heads: GPT-2 architecture dims

    Returns:
        importance_matrix: np.ndarray [num_layers, num_heads]
            Values = accuracy_before - accuracy_after ablating that head.
            Higher = more causally important.
    """
    from src.extraction.activations import batch_extract
    from src.features.engineer import build_feature_matrix

    # Baseline accuracy (no ablation)
    base_activations = batch_extract(prompts, model, tokenizer, device)
    X_base, y = build_feature_matrix(base_activations, labels)
    baseline_acc = (pipeline.predict(X_base) == y).mean()
    print(f"Baseline accuracy: {baseline_acc:.4f}")

    importance = np.zeros((num_layers, num_heads), dtype=np.float32)

    total = num_layers * num_heads
    done = 0
    for layer in range(num_layers):
        for head in range(num_heads):
            abl_activations = []
            for prompt in prompts:
                abl = run_with_ablation(
                    prompt, model, tokenizer, device,
                    heads_to_ablate=[(layer, head)]
                )
                abl_activations.append(abl)

            X_abl, _ = build_feature_matrix(abl_activations, labels)
            abl_acc = (pipeline.predict(X_abl) == y).mean()
            importance[layer, head] = baseline_acc - abl_acc

            done += 1
            if done % 20 == 0:
                print(f"  Ablated {done}/{total} heads...")

    return importance
