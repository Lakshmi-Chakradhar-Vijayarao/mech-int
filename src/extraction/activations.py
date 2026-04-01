"""
Extract hidden states, attention weights, logits, and component outputs from GPT-2.

Component outputs (ffn_outputs, attn_outputs) are captured via forward hooks:
  - attn_outputs: attention block output BEFORE residual addition [12, seq_len, 768]
  - ffn_outputs:  MLP/FFN block output BEFORE residual addition   [12, seq_len, 768]

These enable ReDeEP-style component decomposition: which part of the residual
stream — parametric memory (FFN) or context composition (attention) — carries
the hallucination signal?
"""

import torch
import numpy as np
from typing import Dict


def extract_activations(
    prompt: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
) -> Dict[str, np.ndarray]:
    """
    Run a forward pass and extract internal activations for a prompt.

    Returns dict with:
        hidden_states  -> np.ndarray [num_layers+1, seq_len, hidden_dim]
        attentions     -> np.ndarray [num_layers, num_heads, seq_len, seq_len]
        logits         -> np.ndarray [seq_len, vocab_size]
        input_ids      -> np.ndarray [seq_len]
        attn_outputs   -> np.ndarray [num_layers, seq_len, hidden_dim]
            Attention block output before residual addition (per transformer block).
        ffn_outputs    -> np.ndarray [num_layers, seq_len, hidden_dim]
            FFN/MLP output before residual addition (per transformer block).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = inputs["input_ids"].to(device)

    # --- Register hooks to capture component outputs ---
    _attn_outputs = {}
    _ffn_outputs  = {}

    def make_attn_hook(layer_idx):
        def hook(module, input, output):
            # output is a tuple; output[0] is [batch, seq_len, hidden_dim]
            _attn_outputs[layer_idx] = output[0][0].detach().cpu().numpy()
        return hook

    def make_ffn_hook(layer_idx):
        def hook(module, input, output):
            # output is [batch, seq_len, hidden_dim]
            _ffn_outputs[layer_idx] = output[0].detach().cpu().numpy()
        return hook

    handles = []
    for i, block in enumerate(model.transformer.h):
        handles.append(block.attn.register_forward_hook(make_attn_hook(i)))
        handles.append(block.mlp.register_forward_hook(make_ffn_hook(i)))

    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()

    # hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, hidden_dim]
    hidden_states = np.stack(
        [hs[0].cpu().numpy() for hs in outputs.hidden_states], axis=0
    )  # [num_layers+1, seq_len, hidden_dim]

    # attentions: tuple of num_layers tensors, each [1, num_heads, seq_len, seq_len]
    attentions = np.stack(
        [att[0].cpu().numpy() for att in outputs.attentions], axis=0
    )  # [num_layers, num_heads, seq_len, seq_len]

    logits = outputs.logits[0].cpu().numpy()  # [seq_len, vocab_size]

    # Stack component outputs: num_layers x seq_len x hidden_dim
    num_layers = len(model.transformer.h)
    attn_out_stack = np.stack(
        [_attn_outputs[i] for i in range(num_layers)], axis=0
    )  # [num_layers, seq_len, hidden_dim]
    ffn_out_stack = np.stack(
        [_ffn_outputs[i] for i in range(num_layers)], axis=0
    )  # [num_layers, seq_len, hidden_dim]

    return {
        "hidden_states": hidden_states,
        "attentions":    attentions,
        "logits":        logits,
        "input_ids":     input_ids[0].cpu().numpy(),
        "attn_outputs":  attn_out_stack,
        "ffn_outputs":   ffn_out_stack,
    }


def batch_extract(
    prompts: list,
    model,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
) -> list:
    """
    Extract activations for a list of prompts.
    Returns list of dicts (one per prompt).
    """
    results = []
    for prompt in prompts:
        result = extract_activations(prompt, model, tokenizer, device, max_length)
        results.append(result)
    return results
