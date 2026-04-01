"""
Load GPT-2 and tokenizer with hidden states + attention output enabled.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_gpt2(model_name: str = "gpt2", device: str = None) -> tuple:
    """
    Load GPT-2 model and tokenizer.

    Returns:
        model: GPT2LMHeadModel in eval mode, on device
        tokenizer: GPT2Tokenizer with pad token set
        device: torch.device used
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"   # Apple Silicon GPU (M1/M2/M3)
        else:
            device = "cpu"
    device = torch.device(device)

    # Optimize CPU thread usage on MacBook Air
    if device.type == "cpu":
        import os
        torch.set_num_threads(os.cpu_count() or 4)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
    )
    model.eval()
    model.to(device)

    print(f"Loaded {model_name} on {device}")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Heads:  {model.config.n_head}")
    print(f"  Hidden: {model.config.n_embd}")

    return model, tokenizer, device


if __name__ == "__main__":
    model, tokenizer, device = load_gpt2()
    print("Model loaded successfully.")
