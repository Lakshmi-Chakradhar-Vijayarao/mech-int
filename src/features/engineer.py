"""
Feature engineering: convert raw activations into a fixed-length feature vector.

Each prompt gets one feature vector (6 scalar features) + one label.
"""

import numpy as np
from typing import Dict


def _token_entropy(logits: np.ndarray) -> np.ndarray:
    """
    Compute per-token entropy from logits.

    Args:
        logits: [seq_len, vocab_size]
    Returns:
        entropies: [seq_len]
    """
    # Numerically stable softmax
    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)  # [seq_len]
    return entropy, probs


def _attention_entropy(attentions: np.ndarray) -> float:
    """
    Mean entropy of attention weight distributions.

    Args:
        attentions: [num_layers, num_heads, seq_len, seq_len]
    Returns:
        scalar mean attention entropy
    """
    # attentions already sum to 1 across last dim (softmax output)
    attn = np.clip(attentions, 1e-12, 1.0)
    entropy = -np.sum(attn * np.log(attn), axis=-1)  # [layers, heads, seq_len]
    return float(entropy.mean())


def compute_features(activation: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute a fixed-length feature vector from one prompt's activations.

    Features:
        0: mean_entropy          - avg token-level entropy
        1: max_entropy           - max token-level entropy
        2: logit_variance        - variance of top-50 logits at last token
        3: confidence_gap        - top1_prob - top2_prob at last token
        4: attention_entropy     - mean entropy over all attention heads/layers
        5: mean_activation_norm  - mean L2 norm of hidden states (last 4 layers)

    Args:
        activation: dict from extract_activations()

    Returns:
        features: np.ndarray of shape [6]
    """
    logits = activation["logits"]          # [seq_len, vocab_size]
    hidden = activation["hidden_states"]   # [num_layers+1, seq_len, hidden_dim]
    attn = activation["attentions"]        # [num_layers, num_heads, seq_len, seq_len]

    entropy, probs = _token_entropy(logits)

    # 0: mean entropy
    mean_entropy = float(entropy.mean())

    # 1: max entropy
    max_entropy = float(entropy.max())

    # 2: logit variance (top-50 logits at the last token position)
    last_logits = logits[-1]  # [vocab_size]
    top50_idx = np.argpartition(last_logits, -50)[-50:]
    logit_variance = float(last_logits[top50_idx].var())

    # 3: confidence gap (last token)
    last_probs = probs[-1]  # [vocab_size]
    sorted_probs = np.sort(last_probs)[::-1]
    confidence_gap = float(sorted_probs[0] - sorted_probs[1])

    # 4: attention entropy
    attention_entropy = _attention_entropy(attn)

    # 5: mean activation norm (last 4 layers, averaged over seq and hidden)
    last4 = hidden[-4:]  # [4, seq_len, hidden_dim]
    norms = np.linalg.norm(last4, axis=-1)  # [4, seq_len]
    mean_activation_norm = float(norms.mean())

    return np.array([
        mean_entropy,
        max_entropy,
        logit_variance,
        confidence_gap,
        attention_entropy,
        mean_activation_norm,
    ], dtype=np.float32)


FEATURE_NAMES = [
    "mean_entropy",
    "max_entropy",
    "logit_variance",
    "confidence_gap",
    "attention_entropy",
    "mean_activation_norm",
]


def build_feature_matrix(activations: list, labels: list) -> tuple:
    """
    Build feature matrix X and label vector y from a list of activations.

    Args:
        activations: list of dicts from batch_extract()
        labels: list of ints (1=correct, 0=hallucinated), same length

    Returns:
        X: np.ndarray [n_samples, n_features]
        y: np.ndarray [n_samples]
    """
    X = np.stack([compute_features(a) for a in activations], axis=0)
    y = np.array(labels, dtype=np.int32)
    return X, y
