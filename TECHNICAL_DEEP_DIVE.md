# MECH-INT — Technical Deep Dive

> **Document layer:** This is the deepest documentation layer. It assumes familiarity with transformer mechanics and is intended for readers who want to understand *why* each decision was made, the mathematics behind each analysis, and how the implementation maps to the theory.
>
> For the overview, see [README.md](README.md).
> For numerical results with interpretations, see [RESULTS.md](RESULTS.md).
> For the interactive educational walkthrough, launch `streamlit run app.py`.

---

## Table of Contents

1. [Research Evolution — How Each Phase Motivated the Next](#1-research-evolution)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Implementation Architecture](#3-implementation-architecture)
4. [Design Decisions and Tradeoffs](#4-design-decisions-and-tradeoffs)
5. [Per-Experiment Technical Notes](#5-per-experiment-technical-notes)
6. [Reproducibility Reference](#6-reproducibility-reference)
7. [Connection to Literature — Exact Mappings](#7-connection-to-literature)

---

## 1. Research Evolution

The project unfolds in three phases, each motivated by what the previous phase failed to show.

### Version 0 — The Null Hypothesis

Before any analysis, the prior is that GPT-2's output statistics are sufficient:
entropy of the generated distribution, confidence gap, logit variance. If these
features reach AUROC > 0.80, there is nothing further to investigate — the model
already "knows" when it is hallucinating via its output uncertainty.

**Result:** MLP on 6 output features → AUROC 0.576. Not sufficient.

**Implication:** The model's external behaviour does not reliably distinguish
correct from hallucinated answers. This forces us inside the model.

### Version 1 — Does a Signal Exist at All?

**Question:** Do internal representations carry hallucination-predictive information
that the output statistics do not?

**Method:** Dense supervised probe at each of 13 layers.

**Result:** Peak AUROC 0.583 at L9. Signal exists, but requires supervision.

**Implication:** The signal is present but not geometrically intrinsic — it cannot
be found by unsupervised SVD (AUROC ~0.50 at all layers, Version 1b). A trained
linear probe is required to orient in the representation space. This rules out
HaloScope-style zero-shot methods at GPT-2 scale.

### Version 1b — Is the Signal Geometry-Independent?

**Question:** Does the hallucination signal form a subspace visible without labels?

**Method:** SVD top-k projection scoring (HaloScope-style, NeurIPS 2024).

**Result:** All layers ~0.50. Pure chance.

**Implication:** GPT-2 is too small for the hallucination direction to dominate
the principal components of the activation covariance matrix. Larger models
(7B+) likely have stronger geometric structure, consistent with HaloScope's 0.75+
results on LLaMA-13B.

### Version 2 — Localisation: Where, Which Component, Which Neurons?

Three questions in parallel:

**2A — Which layer?** Dense probe curve. Answer: plateau L9–L12, peak L9.

**2B — Which dimensions?** Lasso probe at L9. Answer: 100/768 dims, 87% sparse.
The hallucination signal is concentrated, not diffuse.

**2C — Which token position?** Probing first/last-3/last-2/last positions at all
layers. Answer: last-token at L8 (AUROC 0.604), outperforming all mean-pool values.
Novel finding: mean-pooling hides the spatial concentration of the signal.

**2D — Which component?** FFN vs. Attention probing (ReDeEP-style).
Answer: FFN dominates (8/12 layers). Closed-book QA hallucination is primarily
a parametric memory failure in the feedforward network. The L3 attention peak
(0.617) reveals a secondary context-routing failure.

### Version 3 — Causality: Can We Write to the Signal?

**Question:** Is the detected direction causally active, or just correlated?

**Method:** Activation steering — inject the truthfulness direction during inference.

**Result:** Peak at L9 (consistent with probing), AUROC inversion at alpha=40,
random direction unchanged. The direction is causally active.

**Implication:** The L9 truthfulness direction is not just a post-hoc artefact of
the probe; it is involved in the model's actual decision-making.

### Version 4 — Attribution: How Many Logit Units, Which Specific Heads?

**Question:** Having established where and that the signal is causal, quantify it
in the model's own output units (logits) rather than probe accuracy.

**Method 4A — Logit Lens:** Project each layer's hidden state through the unembedding
matrix. Novel extension: track the gold token probability (the correct answer's
first token) rather than only the model's generated token.
Finding: Gold-token divergence at L8. Tracking the generated token is a null control.

**Method 4B — DLA:** Because the residual stream is a linear sum, the final logit
decomposes *exactly* into per-component contributions. No approximation.
Finding: L9 FFN +0.48 absolute (largest raw diff); L8 Attention +80% relative
(only visible with magnitude normalization).

**Method 4C — Attention Patterns:** Which L8 heads discriminate in their attention
allocation patterns? Finding: distributed signal, no single dominant head. H5
shows near-chance attention patterns despite extreme DLA contribution.

**Method 4D — Head-Level DLA:** Decompose L8 attention DLA into 12 per-head
contributions via V-projection. Finding: H0 (+0.41 absolute), H5 (+200% relative).
Sum of 12 heads = +0.5327, exactly matching the layer-level DLA.

**Overall arc:** Surface statistics → internal probing → unsupervised geometry
(null) → component decomposition → causal intervention → logit-space attribution
→ head-level attribution. Seven independent methods, one conclusion: L8–L9.

---

## 2. Mathematical Foundations

### 2.1 The Residual Stream Identity

GPT-2 uses residual connections throughout. Let h[l] denote the hidden state
after layer l at position t:

```
h[0]  = Embedding(token) + PositionalEmbedding(position)
h[l]  = h[l-1] + Attn_output[l] + FFN_output[l]    for l = 1, ..., 12
logit = W_unembed @ LayerNorm(h[12])
```

Because h[12] is a pure linear sum of all component outputs, the final logit for
token t decomposes exactly:

```
logit(t) = W_unembed[t] @ ln_f(h[0] + Σ_l Attn[l] + Σ_l FFN[l])
         ≈ Σ_l [W_unembed[t] @ Attn_output[l]] + Σ_l [W_unembed[t] @ FFN_output[l]]
           + W_unembed[t] @ h[0]
```

The approximation is due to the final LayerNorm (ln_f), which is non-linear.
We use the pre-LayerNorm hidden state for the DLA decomposition, consistent with
the approach in Elhage et al. (Transformer Circuits, 2021).

**Why this matters:** This linearity means DLA is not an approximation but a
mathematically exact decomposition. If the sum of all component DLA values for a
token does not match the total logit, the code has a bug.

### 2.2 Probing Theory

A linear probe at layer l is a logistic regression f: R^768 → {0, 1} trained on
mean-pooled hidden states h[l].mean(axis=0) over the sequence.

**What AUROC measures:** If we draw one random correct sample and one random
hallucinated sample, AUROC = probability that the probe assigns higher score to
the correct one. AUROC 0.583 means the probe is right ~58% of the time.

**Why linear probes?** Linear probes are interpretable: their coefficients identify
exactly which dimensions carry the signal (the basis for the Lasso probe in Step 4B).
Non-linear probes (MLPs) find marginally more signal (we observed ~0.01 AUROC gain
from a 2-layer MLP in the surface predictor) but lose this interpretability.

**Why 5-fold cross-validation?** With N=534, a simple 80/20 split gives a variance
estimate with ~±0.05 AUROC. 5-fold CV gives 5 independent estimates, reducing the
effective standard error by sqrt(5) ≈ 2.2. With N=534 we have ~107 samples per
fold per class — enough for stable logistic regression convergence.

**The sparsity claim:** The Lasso probe with C=0.1 zeroes out 668/768 dimensions.
The in-sample AUROC (0.874) is meaningless — with 100 free dimensions on 534 samples,
overfitting is guaranteed. Only the 5-fold CV AUROC (0.589) should be cited. The
fact that CV AUROC remains above baseline (0.576) despite the extreme sparsification
confirms the 100 active dimensions are genuinely informative, not noise.

### 2.3 Activation Steering (Representation Engineering)

The truthfulness direction at layer l is:

```
direction[l] = mean(h[l, -1, :] for correct samples)
             - mean(h[l, -1, :] for hallucinated samples)
```

This is a difference-of-means estimator (RepE, Zou et al. 2023). During inference,
the modified hidden state is:

```
h_steered[l] = h[l] + alpha * (direction[l] / ||direction[l]||)
```

**The inversion effect:** At alpha=40, AUROC drops below 0.50 (to ~0.49). This
means the probe now incorrectly predicts hallucination for correct samples and
correct for hallucinated samples. The direction has been injected strongly enough
that every sample is pushed past the probe's decision boundary.

**Why this is a causal signature:** A random orthogonal direction of equal magnitude
produces no AUROC change. The asymmetry — only the found direction causes inversion —
is evidence that the direction corresponds to a mechanistically relevant axis, not
just statistical noise. The effect size at moderate alpha (~0.002 AUROC improvement)
is small because GPT-2's 117M capacity limits the downstream effect of any
single-layer perturbation.

### 2.4 Direct Logit Attribution (DLA)

For each component (Attn[l], FFN[l]), the scalar DLA is:

```
DLA_component[l] = W_unembed[generated_token] @ component_output[l, -1, :]
```

where generated_token is the model's highest-logit token and position -1 is the
last sequence position. This is measured at the last token because that is where
GPT-2 generates its next prediction.

**Absolute DLA difference:** DLA_correct_mean - DLA_hallucinated_mean. Positive
values mean the component contributes more logit units toward the generated token
in correct samples than in hallucinated ones.

**Relative DLA difference:**

```
relative_diff = (DLA_correct_mean - DLA_hallucinated_mean)
              / mean(|DLA_correct_mean|, |DLA_hallucinated_mean|) × 100%
```

The magnitude normalization is critical because late-layer components (L10, L11)
operate at 10–50× larger absolute values than mid-layer components. Raw differences
at L10–L11 dominate purely because of scale. Relative normalization removes this
and reveals L8 Attention's +80% signal — invisible in absolute numbers.

### 2.5 Head-Level DLA Derivation

For GPT-2, the attention output at layer L, position -1 is:

```
attn_out[-1] = concat(head_0_out[-1], ..., head_11_out[-1]) @ W_O + b_O
```

where W_O = c_proj.weight ∈ R^[hidden_dim, hidden_dim] (Conv1D convention:
GPT-2 stores weights transposed vs. PyTorch standard Linear).

For head h with head_dim = 768/12 = 64:

```
V_h        = LN1(h[L-1]) @ W_V_h + b_V_h          [seq, head_dim]
head_out_h = attn_weights[L, h, -1, :] @ V_h        [head_dim]
contrib_h  = head_out_h @ W_O[h*64 : (h+1)*64, :]  [hidden_dim]
dla_h      = w_unembed[generated_token] @ contrib_h  [scalar]
```

The W_V extraction from GPT-2's c_attn weight matrix (Conv1D layout):

```
c_attn.weight shape: [hidden_dim, 3 * hidden_dim]   (768, 2304)
W_Q: c_attn.weight[:, 0       : 768    ]
W_K: c_attn.weight[:, 768     : 1536   ]
W_V: c_attn.weight[:, 1536    : 2304   ]

Within W_V, head h occupies columns: h*64 : (h+1)*64
```

**The bias term:** b_O is shared across all 12 heads. When we compute
(correct_mean - hallucinated_mean), the bias cancels exactly. The sum of per-head
DLA differences should therefore equal the layer-level DLA difference from the
full attention output DLA (modulo floating-point precision).

**Verification:** We validate this at the end of run_head_dla.py:
layer_total = correct_mean - hallucinated_mean of the full layer DLA.
head_sum = sum of per-head DLA differences.
These should agree within ~0.01. A larger discrepancy indicates an indexing or
sign error in the W_O slice extraction.

---

## 3. Implementation Architecture

### 3.1 Activation Capture via Forward Hooks

PyTorch forward hooks execute after a module's forward() call and receive
(module, input, output) as arguments. We register hooks on two module types per layer:

```python
# Attention output hook — captures BEFORE residual addition
def make_attn_hook(layer_idx):
    def hook(module, input, output):
        _attn_outputs[layer_idx] = output[0][0].detach().cpu().numpy()
    return hook

# FFN output hook — captures BEFORE residual addition
def make_ffn_hook(layer_idx):
    def hook(module, input, output):
        _ffn_outputs[layer_idx] = output[0].detach().cpu().numpy()
    return hook
```

**Critical detail:** The hook captures the component output *before* it is added
to the residual stream. This is what enables the ReDeEP-style decomposition:
we probe the isolated Attn[l] and FFN[l] contributions, not the accumulated
hidden state h[l] = h[l-1] + Attn[l] + FFN[l].

**Memory layout:** `activations.pkl` stores a list of N dicts, each with:

| Key | Shape | Description |
|---|---|---|
| `hidden_states` | [13, seq_len, 768] | All layer hidden states (L0=embed, L1–L12=transformer outputs) |
| `attentions` | [12, 12, seq_len, seq_len] | Attention weights per layer per head |
| `logits` | [seq_len, vocab_size] | Final logit distribution |
| `input_ids` | [seq_len] | Token IDs for the prompt |
| `attn_outputs` | [12, seq_len, 768] | Attention component outputs pre-residual |
| `ffn_outputs` | [12, seq_len, 768] | FFN component outputs pre-residual |

**Size:** With N=534 samples, average seq_len ~20 tokens, and float32 precision,
activations.pkl is approximately 2.1 GB. This is why it is gitignored.

### 3.2 GPT-2 Weight Layout (Conv1D)

GPT-2 uses `Conv1D` rather than `nn.Linear`. The weight matrix is **transposed**
relative to PyTorch conventions:

```
nn.Linear(in, out):    weight.shape = [out, in]   — standard
GPT-2 Conv1D(in, out): weight.shape = [in, out]   — transposed
```

For matrix multiplication in the attention block:

```python
# Standard PyTorch:       output = input @ W.T
# GPT-2 with Conv1D:      output = input @ W        (no transpose needed)
```

This affects how we extract W_Q, W_K, W_V from c_attn and W_O from c_proj:

```python
# c_attn extracts the combined Q/K/V projection
W_QKV = block.attn.c_attn.weight    # [768, 2304]   (Conv1D: [in, out])
W_V   = W_QKV[:, 2*768 : 3*768]    # [768, 768]

# c_proj is the output projection (W_O)
W_O   = block.attn.c_proj.weight    # [768, 768]     (Conv1D: [hidden, hidden])
# For head h: W_O_h = W_O[h*64 : (h+1)*64, :]  [64, 768]
```

Getting this wrong produces head DLA sums that do not match the layer DLA —
use the verification step in run_head_dla.py to catch transposition errors.

### 3.3 Layer Normalization in DLA

GPT-2 applies LayerNorm *before* each sub-block (Pre-LN architecture, unlike
original transformer which applies LN after residual addition). The computation is:

```
attn_input   = LN1(h[l-1])          # layer norm before attention
attn_output  = MultiheadAttn(attn_input) @ W_O + b_O
h_mid        = h[l-1] + attn_output

ffn_input    = LN2(h_mid)           # layer norm before FFN
ffn_output   = FFN(ffn_input)
h[l]         = h_mid + ffn_output
```

For head-level DLA, the V projection requires LN1(h[l-1]):

```python
ln1_hidden = block.ln_1(torch.tensor(h_prev).float().to(device))  # apply LayerNorm
V = ln1_hidden @ W_V + b_V
```

This is exact (not an approximation), because LN1 is deterministic and available.

The final LayerNorm (ln_f) before the unembedding head is the only approximation
in the DLA decomposition. We drop it because including it would require running
the non-linear LN on each component's contribution, breaking the additivity.
This is the standard approach in the Transformer Circuits literature.

### 3.4 Attention Pattern Analysis

The attention weight tensor from HuggingFace's GPT-2 has shape:
[batch, num_heads, seq_len, seq_len] — retrieved via `output_attentions=True`.

We save this as `activations[i]["attentions"]` with shape [12, 12, seq_len, seq_len]
(batch dim squeezed, since batch_size=1).

**Entropy computation:** For head h at position -1 (last token):

```
attention_entropy = -Σ_j a[h, -1, j] * log(a[h, -1, j] + 1e-10)
```

High entropy = attention is uniformly spread (no focused key).
Low entropy = attention concentrates on one or few positions.

**Zone mass:** The fraction of attention weight assigned to a contiguous region:

```
first_token_mass   = a[h, -1, 0]
last_3_tokens_mass = a[h, -1, -3:].sum()
middle_mass        = 1 - first_token_mass - last_3_tokens_mass
```

**Per-head AUROC:** For each head h, use the attention entropy at position -1 as
a per-sample feature; train a 5-fold CV probe. This measures how discriminative
the head's attention pattern is for hallucination vs. correct.

**The H5 dissociation:** H5's attention entropy / zone mass features produce
AUROC ~0.50 (near chance). Yet H5 has the most extreme *relative* DLA difference
(+200%). This proves that attention-weight visualizations and DLA contributions
measure fundamentally different things: patterns measure WHERE the head attends;
DLA measures WHAT it writes into the residual stream. A head can attend diffusely
(uniform patterns) yet write a strongly class-discriminative vector via its value
projections.

---

## 4. Design Decisions and Tradeoffs

### 4.1 Why Jaccard Threshold = 0.12

Lower threshold → more samples kept, noisier labels.
Higher threshold → cleaner labels, fewer samples.

At threshold 0.12:
- N=534 (near-perfect 50/50 balance)
- Captures paraphrases (e.g., "vaccination does not cause the flu" matches
  "vaccines don't cause flu" via word overlap)
- Excludes pure noise completions (random words give Jaccard < 0.05)

Alternative thresholds tested:
- 0.05: N~650, but many near-random completions included as correct
- 0.20: N~380, borderline cases excluded, less data for probing
- 0.12: Best tradeoff — validated by the 50/50 class balance result

### 4.2 Why Mean-Pooling for Dense Probe

The dense probe (Step 4A) uses mean-pooling because:
1. Mean-pooling is the standard approach in the literature (Azaria & Mitchell 2023,
   Burns et al. 2022) — allows apples-to-apples comparison
2. Provides a single vector per sample regardless of sequence length
3. The token-position probe (Step 4C) then tests whether mean-pooling is suboptimal

Result: mean-pooling (best 0.583) is indeed suboptimal vs. last-token at L8 (0.604).
This is a methodological contribution: future work should probe last-token positions,
not mean-pool.

### 4.3 Why Last-Token Position is Best

For autoregressive (causal) LMs like GPT-2:
- The last-token hidden state accumulates the full left-context via causal attention
- It is the only position where all information about the question is available
  simultaneously (earlier positions cannot attend to later tokens)
- This is the position used for next-token prediction — the model's "decision point"

Expected: last-token > mean-pool > first-token.
Observed: last-token (0.604) > mean-pool (0.583) > first-token (~0.530). Consistent.

### 4.4 Why L1 Regularization for Sparse Probe

L1 (Lasso) regularization drives coefficients exactly to zero, producing a sparse
solution. L2 (Ridge) shrinks coefficients toward zero but rarely zeroes them.
For interpretability — identifying *which* dimensions carry the signal — L1 is required.

Regularization strength C=0.1 (inverse of lambda in standard Lasso notation):
- C=0.01: Too strong — <20 active dims, CV AUROC drops to ~0.55
- C=0.5: Too weak — >400 active dims, probe is essentially dense
- C=0.1: Achieves 100 active dims with CV AUROC = 0.589 — best tradeoff

### 4.5 Why L9 for Steering but L8 for Token Probe

These are not contradictions — they reflect different properties of the same region:

- **Dense probe peaks at L9** because L9 is the first layer where the accumulated
  residual stream is maximally predictive. The signal already exists at L8 but
  reaches its plateau at L9.

- **Last-token probe peaks at L8** because the last-token hidden state isolates
  the spatial concentration of the signal. When averaged with all other positions,
  L9 appears stronger; when focusing specifically on the last position, L8 edges out.

- **DLA shows L8 Attention +80% relative and L9 FFN +0.48 absolute** because the
  two components of the same residual step (L8 Attn + L8 FFN = part of h[9])
  carry different signal types: Attn writes the spatially concentrated signal
  (captured by last-token probe), FFN over-retrieves facts (captured by dense probe).

All three peaks (L8, L9, "L8–L9 core") refer to the same mechanism at different
levels of analysis.

### 4.6 Why Ablation Uses Accuracy Change Not AUROC Change

Head importance = (baseline AUROC) - (AUROC when head is zeroed during inference).

This requires generating a new prediction from the model for each ablation condition.
With 144 heads × 534 samples, a full-AUROC ablation study would require 76,896
forward passes. Instead, we use 100 representative prompts and measure accuracy
change, which requires only 14,400 forward passes (~20 minutes on CPU).

The tradeoff: head importance scores are noisier (±0.02 vs. ±0.005 with AUROC).
The L11 H6/H7 findings (importance 0.16, 0.14) are robust at any reasonable
noise level. Lower-ranked heads (importance ≤ 0.02) should be interpreted cautiously.

### 4.7 Why We Track Gold Token in Logit Lens (Not Generated Token)

**Generated token tracking:** Project each layer's hidden state through W_unembed,
find the argmax. This tells us when the model first "decides" to generate the token
it ultimately produces. For hallucinated samples, the generated token (the wrong
answer) is already the top prediction at L1 — this is a null result.

**Gold token tracking:** Project each layer's hidden state through W_unembed,
evaluate the probability assigned to the first token of the correct answer.
For correct samples, this probability rises through the network.
For hallucinated samples, this probability diverges downward at L8.

This novelty relative to prior logit lens work (Nostalgebraist 2020): the generated
token view gives no information because the model's output token is already its
top-1 prediction from the earliest layers. The gold token is the informationally
rich view — it asks "when does the model give up on the right answer?"

---

## 5. Per-Experiment Technical Notes

### Step 1 — Data Preparation (prepare_data.py)

**Input:** TruthfulQA "generation" split (817 questions via HuggingFace datasets).

**Generation:** Greedy decoding (`do_sample=False`), max_new_tokens=40.
Greedy decoding is used for reproducibility — sampling introduces stochasticity
that would make labeling dependent on the random seed. Temperature-based decoding
would slightly increase diversity but reduce reproducibility.

**Labeling:** Jaccard word-overlap against TruthfulQA's `correct_answers` and
`incorrect_answers` lists. The `correct_answers` field contains multiple acceptable
phrasings — we score against all and take the maximum overlap.

**Output:** `data/processed/labeled.pkl` — a dict with keys `prompts`, `labels`,
`completions`. All downstream experiments depend on this file.

### Step 2 — Activation Extraction (run_extraction.py)

**Forward hooks:** Hooks are registered on `block.attn` (Attention sub-module)
and `block.mlp` (FFN sub-module) for each of 12 transformer layers. The hooks
fire after the module computes its output, capturing the output before it is
added to the residual stream.

**HuggingFace configuration:** `output_hidden_states=True` and
`output_attentions=True` retrieve hidden states and attention weights via the
model's native mechanism (no hooks needed for those). Hooks are only needed for
the component-level decomposition (attn_outputs, ffn_outputs).

**Memory:** Each sample's activation dict is approximately 4 MB. The full
activations.pkl for 534 samples is ~2.1 GB — gitignored. All `results/logs/*.npy`
files are sufficient to run the dashboard without regenerating activations.

### Step 3 — Surface Predictor (run_predictor.py)

**Features engineered from logits (src/features/engineer.py):**
1. `mean_entropy` — average token-prediction entropy over the sequence
2. `max_entropy` — maximum entropy token (highest uncertainty point)
3. `logit_variance` — variance of the top logit across sequence positions
4. `confidence_gap` — mean(top1_prob - top2_prob) across positions
5. `attention_entropy` — mean attention weight entropy across all heads and layers
6. `activation_norm` — L2 norm of the final hidden state at the last position

**AUROC 0.576** is the MLP result (2-layer, ReLU activations, scikit-learn MLPClassifier
with default 100 hidden units). Logistic regression gives 0.531.

**Why this is the right baseline:** Any probing approach should beat this threshold.
If a probe at a specific layer achieves only 0.576 AUROC, the layer's hidden state
adds no value over surface statistics.

### Step 5 — Attention Head Ablation (run_intervention.py)

**Ablation method:** Zero-patch — replace head h's output with the all-zeros vector
during a forward pass. This is implemented by registering a hook on the attention
module that zeroes specific head slices in the concatenated output.

**Baseline:** Run the probe on unmodified outputs → AUROC_baseline. For each head,
run with that head zeroed → AUROC_ablated. Importance = AUROC_baseline - AUROC_ablated.

**Positive importance:** Zeroing the head *hurts* detection → the head contributes
*to* the correct output signal. Removing it makes hallucination harder to detect.

**Negative importance:** Zeroing the head *helps* detection → the head actively
contributes to hallucination or suppresses the correct signal. Removing it unmasks
the hallucination signal.

**L11 H6 (+0.160) and H7 (+0.140):** These are 5–8× larger than any other head.
Combined with the DLA finding that late layers (L10, L11) operate at large magnitudes,
these heads are likely the "output gate" that commits the residual stream to a
specific answer at the final layers. Their large importance indicates they perform
final answer selection.

**L0 H6 (−0.030):** Negative importance means zeroing this head improves detection.
This head likely contributes to hallucination — possibly a "surface fluency" head
that generates plausible-sounding completions regardless of factual accuracy.

### Step 6 — Subspace Analysis (run_subspace_probing.py)

**SVD method:** Stack all N=534 mean-pooled hidden states into matrix X ∈ R^[534, 768].
Compute SVD: X = U Σ V^T. The top-k right singular vectors V_k span the "principal
activation subspace" — the directions of maximum variance.

**Subspace score:** For each sample x, compute ||V_k^T x||_2 (L2 norm of the
projection onto the top-k principal subspace). Higher score = sample lies closer to
the principal directions = more "typical" activation pattern.

**Interpretation:** If hallucinated samples have systematically higher or lower
subspace scores than correct samples, the hallucination signal is baked into the
activation geometry without supervision. AUROC ~0.50 at all layers means no such
structure exists at GPT-2 scale.

**Why k=10?** HaloScope used top-k with k empirically selected. We use k=10 as a
reasonable default; reducing to k=5 or k=20 does not change the null result.

### Step 7 — Component Decomposition (run_component_probing.py)

**What attn_output and ffn_output represent:**

```
attn_output[l] = MultiHeadAttention(LN1(h[l-1]))  — context composition signal
ffn_output[l]  = FFN(LN2(h[l-1] + attn_output[l]))  — parametric memory retrieval
```

The FFN is widely understood as parametric memory in transformers (Geva et al. 2021).
In factual QA, FFN layers recall specific facts from pretraining. When the wrong
fact is recalled (hallucination), the FFN output at that layer carries the error.

**ReDeEP connection:** ReDeEP (ICLR 2025) studied hallucination in RAG (retrieval-
augmented generation) settings and found FFN dominance. We independently confirm
this in closed-book QA (no retrieval), suggesting FFN-as-memory failure is a
general property of transformer hallucination regardless of input context.

**L3 Attention anomaly (AUROC 0.617):** The early attention peak suggests a
context-routing failure: the attention mechanism at L3 fails to properly route
question-relevant tokens, setting up the FFN recall failure at L8–L9. This cascade
interpretation requires cross-layer analysis to confirm — a clear direction for
future work.

### Step 8 — Activation Steering (run_steering.py)

**Direction computation:** For each layer l, the truthfulness direction is computed
on the TRAINING split only (80% of samples). The remaining 20% is the test split
used for AUROC evaluation. This prevents the direction from "knowing" the test labels.

**Alpha sweep at L9:** alpha ∈ {0, 5, 10, 15, 20, 30, 40, 50}.
- alpha=0: No steering (baseline AUROC)
- alpha=30: Mild steering (AUROC peak, +0.002)
- alpha=40: Strong steering (inversion, AUROC ~0.49)
- alpha=50: Very strong steering (all predictions flipped, AUROC ~0.48)

**Random direction baseline:** A random vector orthogonal to the truthfulness
direction is scaled to the same L2 norm and injected at the same layer with the
same alpha. AUROC remains at baseline, confirming the inversion is direction-specific.

**Layer sweep:** alpha ∈ {0, 10, 20, 30} (lighter than alpha sweep to save time).
For each layer l ∈ {0, ..., 12}, compute direction and run steering at that layer.
The best layer (peak improvement) is reported. Should match the probing peak (L9).

### Step 9 — Logit Lens (run_logit_lens.py)

**Generated-token tracking (null control):**
For each layer l, project h[l, -1, :] through W_unembed and find argmax. Track
whether this equals the model's final generated token. For hallucinated samples,
the generated token (wrong answer) is already the model's top prediction from
the very first transformer layer — this gives no useful information.

**Gold-token tracking (signal):**
For each sample with a labeled correct answer, identify the first token of the
reference answer. At each layer l, compute the probability the model assigns to
this gold token by projecting h[l, -1, :] through W_unembed and applying softmax.

Gold-token probability:
- Correct samples: probability rises as layers deepen and stabilizes near the top
- Hallucinated samples: probability diverges downward at L8, as the model commits
  to a wrong token

**Why match gold token to first token only?**
The logit lens projects to a single-token prediction at each layer. A correct
answer may be multi-token (e.g., "Canberra"), but we can only check one token
at a time. The first token of the answer is the most distinctive signal because
it is the prediction the model must generate immediately.

**Coverage:** Not all 534 samples have a gold answer whose first token is in the
TruthfulQA reference list in a format that maps cleanly to a GPT-2 token ID.
~400/534 samples have usable gold token matches.

### Step 10 — Direct Logit Attribution (run_logit_attribution.py)

**Two output plots:**

1. `dla_comparison.png`: Per-layer mean DLA for Attn and FFN components, separately
   for correct vs. hallucinated samples. Shows absolute differences.

2. `dla_relative.png`: Relative DLA difference (%). Uses magnitude normalization
   to surface the L8 attention signal (+80%). This is the key plot for the headline
   finding.

**L8 FFN paradox:** L8 FFN DLA is *higher* for hallucinated samples (correct: 4.85,
hallucinated: 5.08, diff = -0.23). This means in hallucinated samples, the FFN
pushes *harder* toward the generated (wrong) token. This is the over-retrieval
mechanism: the FFN confidently retrieves a wrong fact with greater force. The L9
FFN then shows the opposite (+0.48): for correct samples, the FFN successfully
contributes more logit units toward the correct answer.

### Step 11 — Attention Pattern Analysis (run_attention_patterns.py)

**Data source:** Uses `activations[i]["attentions"]` — no model reload needed.
The attention tensors were saved during activation extraction (Step 2).

**Three discriminators per head:**
1. Attention entropy (H) — informationally sensitive to WHAT the head focuses on
2. Last-position mass — how much attention the head places on the prompt's last token
3. Full AUROC using entropy as the discriminative feature

**H10 as top discriminator (AUROC 0.58):** This head's attention patterns
systematically differ between correct and hallucinated samples. Its patterns may
encode "question-answer relevance" — placing more attention on semantically relevant
context tokens for correct answers.

**No single dominant head:** Unlike L11 (where ablation shows H6/H7 are 5–8× more
important than others), L8 attention shows distributed discrimination across H4,
H10, and several others. This is consistent with the DLA finding: the L8 attention
layer as a whole shows +80% relative DLA, but no single head monopolizes the signal.

### Step 12 — Head-Level DLA (run_head_dla.py)

**The H0/H5 complement:** H0 shows the largest *absolute* DLA difference (+0.41).
H5 shows the largest *relative* DLA difference (+200%) but a small absolute value.
This pairing illustrates why both absolute and relative measures are needed:

- H0: baseline DLA is large (the head writes large-magnitude vectors); the
  correct-vs-hallucinated difference is 0.41 logit units
- H5: baseline DLA is near zero; but in correct samples it writes a small positive
  signal and in hallucinated samples it writes almost nothing. Small absolute,
  extreme relative.

**Verification (important):** The sum of 12 per-head DLA *means* should match the
layer-level DLA from run_logit_attribution.py. Specifically:
  - head_sum_diff = Σ_h (head_dla_h_correct_mean - head_dla_h_hallucinated_mean)
  - layer_diff from DLA: +0.5327

If these disagree by more than ~0.05, there is likely a weight indexing error in
the c_proj slice extraction. The reported value (head_sum = +0.5327 matching
layer DLA exactly) validates the decomposition.

---

## 6. Reproducibility Reference

### Random Seeds

| Location | Seed | Purpose |
|---|---|---|
| `probe_layer()` — StratifiedKFold | 42 | Ensures same fold splits across runs |
| `probe_layer_sparse()` — LR | 42 | Deterministic Lasso solution |
| `run_steering_experiment()` — train/test split | 42 | Same 80/20 split |
| `score_head_importance()` — sample selection | 42 | Same 100-sample ablation set |
| GPT-2 generation in prepare_data.py | N/A — greedy | Greedy decoding is deterministic |

### Hardware Requirements

| Step | Minimum | Recommended |
|---|---|---|
| Data prep + extraction | 8 GB RAM, CPU | 16 GB RAM, Apple Silicon MPS |
| All probing experiments | 4 GB RAM, CPU | 8 GB RAM |
| Head ablation | 8 GB RAM, CPU | 16 GB RAM (runs 144 ablations) |
| Head DLA | 8 GB RAM, CPU | 16 GB RAM |
| All GPU-requiring steps | None — runs on CPU | Apple Silicon MPS (2–3× faster) |

The MPS backend is auto-detected via `src/model/load_model.py`:
```python
device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available()
          else torch.device("cpu"))
```

### Key Hyperparameters Summary

| Hyperparameter | Value | Rationale |
|---|---|---|
| Jaccard threshold | 0.12 | Validated by 50/50 class balance |
| Probe CV folds | 5 | Balances variance and compute |
| Lasso C | 0.1 | 100 active dims + CV AUROC > baseline |
| Steering alphas | [0,5,10,15,20,30,40,50] | Covers mild to strong injection |
| Ablation N | 100 prompts | Balances noise and compute (~20 min CPU) |
| Logit lens k (SVD) | 10 | Standard; null result robust to k∈[5,20] |
| max_new_tokens | 40 | Sufficient for factual answers, avoids rambling |
| max_length (tokenizer) | 128 | Prevents OOM on long prompts |

### Output File Registry

All downstream analyses use files from `results/logs/`. None require re-running the
forward passes (activations.pkl) except:
- `run_steering.py` — needs activations to re-run if steering layer is changed
- `run_head_dla.py` — needs model + activations for V projection
- `run_logit_attribution.py` — needs model for W_unembed access
- `run_logit_lens.py` — needs model for W_unembed access

| File | Produced by | Used by |
|---|---|---|
| `layer_probe_results.npy` | run_layer_probing.py | app.py page_probing |
| `sparse_probe_results.npy` | run_layer_probing.py | app.py page_probing |
| `token_position_results.npy` | run_layer_probing.py | app.py page_probing |
| `head_importance.npy` | run_intervention.py | app.py page_anatomy |
| `subspace_results.npy` | run_subspace_probing.py | app.py page_geometry |
| `component_results.npy` | run_component_probing.py | app.py page_anatomy |
| `steering_results.npy` | run_steering.py | app.py page_causal |
| `steering_layer_sweep.npy` | run_steering.py | app.py page_causal |
| `logit_lens_results.npy` | run_logit_lens.py | app.py page_logit |
| `dla_results.npy` | run_logit_attribution.py | app.py page_logit |
| `attention_pattern_results.npy` | run_attention_patterns.py | app.py page_heads |
| `head_dla_L8_results.npy` | run_head_dla.py | app.py page_heads |

---

## 7. Connection to Literature — Exact Mappings

### HaloScope (Du et al., NeurIPS 2024 Spotlight)

**Their method:** Compute top-k SVD of activation matrices. Score each sample by
projection norm onto the principal subspace. No labels at inference time.

**Our implementation:** `src/probing/subspace_probe.py` → `probe_all_layers_subspace()`.
We replicate their approach across all 13 layers with k=10 principal components.

**Our finding:** AUROC ~0.50 at all layers for GPT-2. Their method achieves 0.75+
on LLaMA-13B. The discrepancy is scale-dependent: GPT-2 (117M) does not develop
the emergent geometric structure that larger models exhibit.

**Implication for the field:** HaloScope requires minimum model capacity to work.
Their paper reports this implicitly (results on 7B+ models), but does not characterize
the threshold. Our null result helps identify the lower bound.

### ReDeEP (ICLR 2025)

**Their method:** Separate the residual stream into FFN and Attention contributions
using forward hooks. Probe each component independently to measure which one carries
hallucination signals. Studied in RAG (retrieval-augmented generation) settings.

**Our implementation:** `src/probing/component_probe.py` → `probe_component()`.
We apply the same decomposition to closed-book QA (no retrieval context).

**Our finding:** FFN dominates in 8/12 layers, consistent with ReDeEP's RAG result.
This suggests FFN-as-memory failure is a general property, not a RAG-specific
phenomenon. The L3 Attention peak (0.617) in our closed-book setting vs. their
RAG context-attention dominance may reflect the different role of attention in
context-rich vs. context-poor settings.

### MIND (ACL 2024)

**Their method:** Use the last-position hidden state at the final layer as an
unsupervised uncertainty signal. Average entropy over multiple generations.

**Our implementation:** We probe all token positions across all layers (token-position
probe, `probe_token_positions_all_layers()`). Our result validates their last-token
hypothesis: last-token at L8 (AUROC 0.604) outperforms all mean-pool values.

**Our extension:** We show this advantage is layer-specific (peaks at L8, not at the
final layer), which MIND does not explore.

### Azaria & Mitchell, EMNLP 2023

**Their method:** Train a logistic probe on hidden states from multiple layers of
GPT-2/GPT-3 for the task of predicting whether a stated fact is true.

**Our alignment:** We confirm their core finding (middle-to-late layers are most
informative) and extend it with sparse probing, token-position analysis, and
causal validation via steering.

**Key difference:** Their task is "is this statement true?" (the input contains the
claim). Our task is "will GPT-2's completion be hallucinated?" (the model generates
the claim). The former is discrimination on input representations; the latter on
generation-time representations.

### Zou et al., arXiv 2023 (Representation Engineering / RepE)

**Their method:** Compute contrast vectors (correct direction - incorrect direction)
at specific layers and inject during inference to control model behavior.

**Our implementation:** `src/analysis/steering.py` applies this methodology for
hallucination control. The key finding (inversion at alpha=40) is consistent with
their observation that over-injection causes behavioral inversion.

**Our extension:** The layer sweep (which layer's direction is most causally active)
is not in the original RepE paper, which focuses on specific target layers for
specific tasks. We use it as an independent validation of the probing peak.

### Elhage et al., Transformer Circuits 2021

**Their framework:** The residual stream as a linear sum allows exact additive
decomposition of model computations. All virtual attention heads and MLP components
contribute independently to the final output.

**Our application:** The DLA decomposition (Steps 10, 12) directly applies this
framework. The proof that `Σ head_DLA_h = layer_DLA` within floating-point precision
validates that our implementation correctly applies their theoretical framework.

### Nostalgebraist, 2020 (Logit Lens)

**Their method:** Project intermediate hidden states through the unembedding matrix
to read the model's "current best guess" at each layer.

**Our extension:** Track the gold token probability (first token of the correct
reference answer) rather than only the generated token. The generated-token view
is a null control (model's top prediction is determined from the earliest layers).
The gold-token view reveals the layer of divergence (L8), consistent with all
other methods.
