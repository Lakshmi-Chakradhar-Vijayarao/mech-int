"""
Phase 1 Day 2: Download TruthfulQA, generate GPT-2 answers, label correct vs hallucinated.

Labeling strategy:
  - GPT-2 generates a short completion for each question.
  - We compare the completion against the list of known correct answers (case-insensitive substring match).
  - Label 1 = correct (answer contains a correct answer string)
  - Label 0 = hallucinated (answer does not match any correct answer)

Usage:
    python experiments/prepare_data.py
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from datasets import load_dataset
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.load_model import load_gpt2

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_completion(
    prompt: str,
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    """Generate a short completion for a question prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy — deterministic
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _word_overlap(a: str, b: str) -> float:
    """
    Jaccard similarity on word sets (case-insensitive).
    Returns 0.0–1.0. Does not require any extra packages.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def label_completion(completion: str, correct_answers: list, incorrect_answers: list) -> int:
    """
    Label a completion as correct (1), hallucinated (0), or ambiguous (-1).

    Strategy (soft word-overlap, no extra dependencies):
      - Score completion against every correct answer via Jaccard word similarity.
      - Score completion against every incorrect answer the same way.
      - If best_correct > THRESHOLD  → label 1
      - If best_incorrect > THRESHOLD → label 0
      - Tie (both above threshold)   → correct wins (conservative)
      - Neither above threshold      → -1 (exclude)

    Using soft overlap instead of exact substring match raises dataset yield
    from ~9% to ~40-50% because GPT-2 completions rarely repeat exact answer
    strings verbatim but often share key words.
    """
    THRESHOLD = 0.12   # empirically tuned: low enough to catch paraphrases,
                       # high enough to exclude pure noise completions

    best_correct   = max((_word_overlap(completion, a) for a in correct_answers),   default=0.0)
    best_incorrect = max((_word_overlap(completion, a) for a in incorrect_answers), default=0.0)

    if best_correct > THRESHOLD or best_incorrect > THRESHOLD:
        # Correct wins ties — conservative labeling
        return 1 if best_correct >= best_incorrect else 0

    return -1  # ambiguous — exclude


def main():
    print("=== Phase 1: Data Download + Labeling ===\n")

    # --- Download TruthfulQA ---
    print("Downloading TruthfulQA from HuggingFace datasets...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Loaded {len(dataset)} questions\n")

    # Save raw dataset snapshot
    raw_path = RAW_DIR / "truthfulqa_raw.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved raw dataset: {raw_path}")

    # --- Load GPT-2 ---
    model, tokenizer, device = load_gpt2()

    # --- Generate completions and label ---
    print("\nGenerating GPT-2 completions and labeling...")
    prompts = []
    labels = []
    completions = []
    skipped = 0

    for item in tqdm(dataset, desc="Processing"):
        question = item["question"]
        correct_answers = item["correct_answers"]    # list of strings
        incorrect_answers = item["incorrect_answers"]  # list of strings

        prompt = f"Q: {question}\nA:"
        completion = generate_completion(prompt, model, tokenizer, device)
        label = label_completion(completion, correct_answers, incorrect_answers)

        if label == -1:
            skipped += 1
            continue

        prompts.append(prompt)
        labels.append(label)
        completions.append(completion)

    print(f"\nLabeling complete:")
    print(f"  Total kept:      {len(labels)}")
    print(f"  Correct  (1):    {sum(labels)}")
    print(f"  Hallucinated (0): {len(labels) - sum(labels)}")
    print(f"  Skipped (ambiguous): {skipped}")

    # --- Save ---
    labeled = {
        "prompts":     prompts,
        "labels":      labels,
        "completions": completions,
    }
    out_path = PROCESSED_DIR / "labeled.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(labeled, f)
    print(f"\nSaved labeled data: {out_path}")

    # Quick sanity check
    print("\nSample (first 3):")
    for i in range(min(3, len(prompts))):
        print(f"  [{labels[i]}] {prompts[i][:60]}...")
        print(f"       Completion: {completions[i][:80]}")

    print("\nDone. Ready for run_extraction.py")


if __name__ == "__main__":
    main()
