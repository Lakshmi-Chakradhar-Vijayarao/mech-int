# Can a Language Model Know When It's Wrong?

### Detecting, Localising, and Steering Hallucinations from Internal Activations

**Lakshmi Chakradhar Vijayarao**

> *GPT-2 is answering a question. Before you read its answer, can you look inside the model and predict whether it's about to hallucinate? This project says: yes — and then goes further: it pinpoints exactly where, which component, which specific attention heads, which dimensions, and proves the direction is causal by intervening on it.*

[![Live Demo](https://img.shields.io/badge/Live%20Demo-mech--int.streamlit.app-1565c0?logo=streamlit)](https://lakshmi-chakradhar-vijayarao-mech-int-app-3enga8.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-mech--int-1565c0?logo=github)](https://github.com/Lakshmi-Chakradhar-Vijayarao/mech-int)
[![Companion Project](https://img.shields.io/badge/Companion-HaRP%20(Qwen%203B)-4527a0?logo=github)](https://github.com/Lakshmi-Chakradhar-Vijayarao/harp)

---

## Live Dashboard

**Deployed:** [https://lakshmi-chakradhar-vijayarao-mech-int-app-3enga8.streamlit.app/](https://lakshmi-chakradhar-vijayarao-mech-int-app-3enga8.streamlit.app/)

The interactive dashboard walks through the complete 12-step research pipeline as a narrative — from motivation to head-level attribution. Every chart, table, and finding is explained in plain English alongside the technical result.

---

## The Problem

Large language models (LLMs) hallucinate — they confidently produce factually wrong answers with exactly the same fluency and tone as when they are correct. This is one of the most critical unsolved problems in AI safety and deployment.

Most detection approaches check the *output* — comparing the answer to a reference, running another model to verify it, or sampling multiple times and checking consistency. All of these happen **after** the model has already generated the wrong answer.

**This project takes a different approach: look inside the model while it's thinking.**

Every transformer layer produces a hidden state — a vector of numbers representing what the model has "processed" at that depth. The central question is:

> **Do hallucinated outputs leave a detectable fingerprint in the model's internal activations — before the first output token is ever generated?**

If yes, we can build monitors that intercept hallucinations upstream of the output. And we can go further: identify *which layer*, *which component*, *which dimensions*, and *which attention heads* are responsible — turning a detection problem into a mechanistic explanation.

---

## What This Project Does

A complete 12-step mechanistic interpretability pipeline across three phases:

```
PHASE I — DETECTION        PHASE II — LOCALISATION          PHASE III — ATTRIBUTION
Does the signal exist?  →  Where / what / which component?  →  Can we steer it? Attribute it?
    Steps 3–4                      Steps 5–8                         Steps 9–12
```

| Step | Question | Method | Key Result |
|---|---|---|---|
| 1 · Data | — | Download TruthfulQA, generate GPT-2 answers, Jaccard label | **534 labeled samples (50/50 balance)** |
| 2 · Extraction | — | Forward hooks: capture all hidden states, FFN/Attn outputs, attention weights | `activations.pkl` (2.1 GB) |
| 3 · Surface Predictor | *Do output statistics detect hallucination?* | Train classifier on 6 logit-based features, 5-fold CV | **AUROC 0.576 — weak signal; motivates Step 4** |
| 4A · Layer Probe | *Which layer holds the signal?* | LR probe at each of 13 layers, 5-fold CV | **Plateau L9–L12, peak AUROC 0.583 at L9** |
| 4B · Sparse Probe | *Which of the 768 dimensions carry the signal?* | L1-regularized (Lasso) probe at peak layer | **100 / 768 dims active (87% sparse); CV AUROC 0.589** |
| 4C · Token Probe | *Which token position is most informative?* | Probe each position separately across all layers | **Last-token at L8: AUROC 0.604 — best single point** |
| 5 · Ablation | *Which attention heads are causally necessary?* | Zero each of 144 heads, measure AUROC change | **L11 H6 (+0.16) and H7 (+0.14) dominate — 5–8× others** |
| 6 · Subspace | *Is the signal baked into the unsupervised geometry?* | SVD projection scoring (HaloScope-style, NeurIPS 2024) | **~0.50 at all layers — no intrinsic geometry at GPT-2 scale** |
| 7 · Component | *FFN or Attention — which component fails?* | Probe FFN vs. Attn outputs separately (ReDeEP-style, ICLR 2025) | **FFN dominates: 8/12 layers, peak AUROC 0.605 at L8** |
| 8 · Steering | *Is the direction causally active?* | Inject truthfulness direction at each layer; found vs. random control | **L9 peak independently confirmed; α=40 inverts signal — causal signature** |
| 9 · Logit Lens | *How does the prediction evolve layer by layer?* | Project each layer's hidden state through W_unembed; gold-token tracking | **Gold-token divergence at L8 — consistent with all other peaks** |
| 10 · DLA | *Which component contributes how many logit units?* | Direct Logit Attribution: additive decomposition of the final logit | **L8 Attention: +80% relative DLA — strongest normalized signal in the project** |
| 11 · Attn Patterns | *How do individual heads allocate attention?* | Entropy + zone mass + per-head AUROC from saved attention weights | **Distributed signal at L8; H10 top discriminator (0.58); no dominant head** |
| 12 · Head DLA | *Which specific head at L8 drives the attention DLA?* | Decompose L8 DLA into 12 per-head contributions via V-projection | **H0: +0.41 abs; H5: +200% relative; 12-head sum = +0.5327 (exact)** |

---

## Key Results

### Seven methods converge on L8–L9 as the mechanistic core

| Method | Peak Layer | Key Number |
|---|---|---|
| Dense layer probe (Step 4A) | L9 | AUROC 0.583 — plateau L9–L12 |
| Token-position probe (Step 4C) | L8 × last token | AUROC 0.604 — best single probe point |
| FFN component probe (Step 7) | L8 | AUROC 0.605 — peak FFN layer |
| Activation steering layer sweep (Step 8) | L9 | Peak without any probe |
| DLA absolute (Step 10) | L9 FFN | +0.48 logit units — largest raw difference |
| DLA relative (Step 10) | L8 Attention | +80% relative — strongest normalized signal |
| Gold-token logit lens (Step 9) | L8 | Gold-token probability diverges here |

### The three-stage hallucination cascade

```
Stage 1 (L3)           Stage 2 (L8–L9)  ★ CORE          Stage 3 (L11)
Early context       →  FFN over-retrieval              →  Output commitment
routing failure        + Attn DLA peak (+80%)             L11 H6/H7 dominate
Attn peak 0.617        Wrong fact recalled from weights    Importance 5–8× others
```

### Headlines

- **Best AUROC: 0.604** — L8 last-token probe (vs. 0.576 surface baseline)
- **Mechanistic core: L8–L9** — confirmed by 7 independent methods
- **Signal sparsity: 100 / 768 dims** (87%) — concentrated, not diffuse
- **L8 Attention +80% relative DLA** — only visible with normalized attribution
- **H5 dissociation** — near-chance attention patterns, +200% relative DLA contribution; pattern ≠ contribution
- **Subspace null result** — SVD scoring fails at GPT-2 scale; supervision is required
- **Causal proof** — steering inversion at α=40 is a strong causal signature

---

## Architecture & Design

### Why GPT-2

GPT-2 (117M, decoder-only) was chosen as the study subject for four reasons:
1. **No instruction tuning / RLHF** — internals directly reflect the base LM objective, cleaner for mechanistic study
2. **CPU-runnable** — full pipeline on a MacBook in ~75 minutes, no GPU required
3. **Established probing literature** — Azaria & Mitchell (2023), Burns et al. (2022) used GPT-2; results are comparable
4. **Acknowledged limitation** — AUROC values are modest (0.55–0.61); the methodology is designed to scale to 7B+ models

### Why logistic regression probes

Linear probes are used throughout because their coefficients are directly interpretable — they identify *which dimensions* carry the signal (the basis for the sparse probe in Step 4B). Non-linear probes find marginally more signal but lose this interpretability.

### Why the residual stream decomposition is exact

GPT-2 uses residual connections: `h[l] = h[l-1] + Attn_output[l] + FFN_output[l]`. Because the final hidden state is a pure linear sum, the final logit decomposes *exactly* into per-component contributions — no approximation. This is the mathematical basis for DLA (Step 10) and head-level DLA (Step 12).

### Why relative DLA matters

Late transformer layers operate at much larger absolute magnitudes than mid layers (because the residual stream accumulates contributions). Raw DLA differences at L10–L11 can be 50× larger than at L8, even if L8 carries the more discriminative signal. Relative DLA normalizes by mean magnitude:

```
relative_diff = (correct_mean − hallucinated_mean) / mean_magnitude × 100%
```

This surfaces L8 attention's +80% signal — invisible in the absolute numbers.

---

## Repository Structure

```
mech-int/
├── app.py                              # Streamlit educational dashboard (11 pages)
├── src/
│   ├── model/load_model.py             # GPT-2 loader — auto-selects MPS > CUDA > CPU
│   ├── features/engineer.py            # 6 logit-based uncertainty features
│   ├── extraction/activations.py       # Forward hooks — captures all hidden states + attn weights
│   ├── predictor/classifier.py         # LR + MLP on surface features, 5-fold CV
│   ├── probing/
│   │   ├── layer_probe.py              # Dense probe, Lasso probe (CV), token-position probe
│   │   ├── subspace_probe.py           # SVD subspace scoring (HaloScope-style)
│   │   └── component_probe.py          # FFN vs. Attention decomposition (ReDeEP-style)
│   ├── intervention/ablation.py        # Head ablation — zero-patch each of 144 heads
│   ├── analysis/
│   │   ├── steering.py                 # Activation steering + layer sweep
│   │   ├── logit_lens.py               # Logit lens + gold-token probability tracking
│   │   ├── logit_attribution.py        # Direct Logit Attribution (absolute + relative)
│   │   ├── attention_patterns.py       # Entropy, zone mass, per-head AUROC
│   │   └── head_dla.py                 # Head-level DLA via V-projection at L8
│   └── evaluation/metrics.py           # All visualizations
├── experiments/                        # One script per pipeline step (run in order)
│   ├── prepare_data.py                 # Step 1 — data + labeling
│   ├── run_extraction.py               # Step 2 — forward pass + hook capture
│   ├── run_predictor.py                # Step 3 — surface feature classifier
│   ├── run_layer_probing.py            # Steps 4A/4B/4C — dense + sparse + token probing
│   ├── run_intervention.py             # Step 5 — attention head ablation
│   ├── run_subspace_probing.py         # Step 6 — SVD subspace scoring
│   ├── run_component_probing.py        # Step 7 — FFN vs. Attn decomposition
│   ├── run_steering.py                 # Step 8 — activation steering + layer sweep
│   ├── run_logit_lens.py               # Step 9 — logit lens + gold-token tracking
│   ├── run_logit_attribution.py        # Step 10 — Direct Logit Attribution
│   ├── run_attention_patterns.py       # Step 11 — attention pattern analysis
│   └── run_head_dla.py                 # Step 12 — head-level DLA decomposition
├── data/
│   └── processed/                      # labeled.pkl, features.npy, labels.npy
│                                       # activations.pkl (2.1 GB — gitignored)
├── results/
│   ├── plots/                          # All figures (16 PNG files)
│   └── logs/                           # Raw numpy results per analysis (*.npy)
├── RESULTS.md                          # Full numerical results + interpretations
└── requirements.txt
```

---

## Setup & Running

**Requirements:** Python 3.9+. Runs on CPU or Apple Silicon MPS — no GPU needed.

```bash
git clone https://github.com/Lakshmi-Chakradhar-Vijayarao/mech-int.git
cd mech-int
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Run the full pipeline (in order):**

```bash
python experiments/prepare_data.py            # ~15 min  — download TruthfulQA, label completions
python experiments/run_extraction.py          # ~15 min  — GPT-2 forward passes + hook captures
python experiments/run_predictor.py           # ~3 min   — surface feature classifier
python experiments/run_layer_probing.py       # ~90 sec  — dense + sparse (CV) + token-position probing
python experiments/run_intervention.py        # ~20 min  — attention head ablation (144 heads)
python experiments/run_subspace_probing.py    # ~10 sec  — SVD subspace scoring
python experiments/run_component_probing.py   # ~3 min   — FFN vs. Attn component probing
python experiments/run_steering.py            # ~5 min   — activation steering + layer sweep
python experiments/run_logit_lens.py          # ~2 min   — logit lens + gold-token tracking
python experiments/run_logit_attribution.py   # ~2 min   — DLA (absolute + relative)
python experiments/run_attention_patterns.py  # ~1 min   — attention entropy + per-head AUROC
python experiments/run_head_dla.py            # ~3 min   — head-level DLA decomposition at L8
```

**Launch the interactive dashboard:**

```bash
streamlit run app.py
# Opens at http://localhost:8501
# Deployed: https://lakshmi-chakradhar-vijayarao-mech-int-app-3enga8.streamlit.app/
```

**Note:** `data/processed/activations.pkl` (~2.1 GB) and `data/processed/labeled.pkl` are gitignored because of size. All downstream result files in `results/logs/*.npy` and `results/plots/*.png` are committed and sufficient to run the dashboard without re-running the pipeline.

---

## Timing

| Step | Time (CPU) |
|---|---|
| Data prep + extraction | ~30 min |
| Surface classifier | ~3 min |
| Layer + sparse + token probing | ~90 sec |
| Head ablation | ~20 min |
| SVD subspace | ~10 sec |
| Component decomposition | ~3 min |
| Activation steering | ~5 min |
| Logit lens | ~2 min |
| Direct logit attribution | ~2 min |
| Attention patterns | ~1 min |
| Head-level DLA | ~3 min |
| **Total** | **~75 min** |

Hardware: tested on MacBook Air (Apple Silicon M-series). Auto-detects MPS > CUDA > CPU.

---

## Methodology in Plain English

**Probe:** A tiny logistic regression classifier trained on a frozen model's hidden states. If it can predict whether the output is correct, that layer's representation encodes the information. The model is never modified — only read from.

**AUROC:** Area Under the ROC Curve. 0.50 = random chance. 1.00 = perfect. 0.58 means the probe correctly ranks correct above hallucinated in ~58% of pairs.

**Residual stream:** Each layer adds its output to a running sum: `h[l] = h[l-1] + Attn[l] + FFN[l]`. Because this is linear, the final logit decomposes *exactly* into additive contributions — the basis for DLA.

**SVD subspace scoring:** Singular value decomposition finds the directions of maximum variance. If hallucinated and correct samples occupy different subspaces, detection is possible without labels. The fact that this fails for GPT-2 is itself a finding.

**Activation steering:** Instead of reading representations (probing), we write to them — injecting a direction vector at a specific layer during inference. If AUROC changes, the direction is causally active.

**Direct Logit Attribution:** Because the residual stream is a linear sum, the final logit decomposes exactly: `logit(t) = Σ W_unembed[t] · component_output[l]`. This turns "which layer matters" (probing) into "by how many logit units" (attribution).

**Relative DLA:** Normalizes by mean magnitude to surface mid-layer signals: `(C_mean − H_mean) / mean_magnitude × 100%`. L8 attention's +80% is invisible in raw DLA numbers.

**Head-level DLA:** Each head's contribution: `head_out_h = attn_weights[h,-1,:] @ V_h @ W_O_h`. Summing 12 heads reproduces the layer DLA exactly — allows attribution to individual heads.

---

## Related Work

| Paper | Venue | What We Implement / Compare Against |
|---|---|---|
| HaloScope (Du et al.) | NeurIPS 2024 Spotlight | SVD subspace membership scoring — we replicate and find it scale-dependent |
| ReDeEP | ICLR 2025 | FFN vs. Attention residual stream decomposition — we confirm FFN dominance in closed-book QA |
| LLM-Check | NeurIPS 2024 | Internal state eigenvalue analysis — framing for mid-late layer signal |
| MIND | ACL 2024 | Unsupervised last-position hidden state — we confirm last-token advantage |
| Azaria & Mitchell | EMNLP 2023 | Foundation: probing hidden states for truthfulness on GPT-2 |
| Burns et al. (CCS) | NeurIPS 2022 | Unsupervised elicitation of latent knowledge |
| Zou et al. (RepE) | arXiv 2023 | Representation Engineering — activation steering methodology |
| Elhage et al. | Transformer Circuits 2021 | Mathematical framework for Direct Logit Attribution |
| Nostalgebraist | Blog 2020 | Logit lens — projecting intermediate hidden states through the unembedding matrix |

---

## Limitations

- **GPT-2 is small (117M).** AUROC values are modest (0.55–0.61). Larger models (7B+) show stronger separation. The SVD null result is likely scale-dependent and should not be generalised.
- **Jaccard labeling is noisy.** Word overlap is a soft proxy for factual correctness. Semantic or model-based labeling would improve quality.
- **N=534 is moderate.** Confidence intervals are wide (±0.04–0.08). More data would tighten head-level estimates especially.
- **Sparse probe CV AUROC is 0.589, not 0.874.** The in-sample figure reflects overfitting to N=534. Only the 5-fold CV number (0.589) should be cited as a generalisation estimate.
- **Steering effect size is small.** AUROC improvement ~0.002 at moderate α. The causal structure is real; the magnitude is limited by model capacity.
- This is a **probing and attribution study**, not a deployment-ready detector.

---

## Future Directions

| Direction | Why it matters | Difficulty |
|---|---|---|
| Run on LLaMA-7B / Mistral-7B | Test whether L8–L9 findings generalise; subspace method should succeed at 7B | Medium |
| Semantic labeling (NLI / GPT-4) | Replace Jaccard with model-based labeling for cleaner labels | Low |
| Sparse autoencoder (SAE) on peak dims | Map the 100 active dimensions to human-interpretable features (Anthropic 2024 style) | High |
| Real-time inference monitor | Deploy the L8 last-token probe as a lightweight inference-time hallucination flag | Low |
| Multi-dataset validation | Test on FEVER, Natural Questions, BioASQ to check domain generality | Medium |
| Cross-layer causal tracing | Activation patching between correct/hallucinated pairs to trace exact information flow | High |


---

## Companion Project — HaRP & The Two-Part Arc

MECH-INT establishes the mechanistic foundation at GPT-2 scale. **[HaRP](https://github.com/Lakshmi-Chakradhar-Vijayarao/harp)** is the follow-up: it applies the same hidden-state geometry insight to Qwen 2.5 3B (25× larger), adds a full governance pipeline, and reports honest OOF-corrected results across 28 experiments.

| | MECH-INT | HaRP |
|---|---|---|
| **Model** | GPT-2 (117M) | Qwen 2.5 3B |
| **Parameters** | 117M | 3B (25× larger) |
| **Data** | 534 TruthfulQA samples | 700 TruthfulQA samples |
| **Goal** | Mechanistic understanding | Detection + governance |
| **Best AUROC** | 0.604 (L8 last-token probe) | 0.775 (honest OOF, 5-fold CV) |
| **Geometry signal** | Absent (SVD ~0.50 — scale too small) | Present at L32 (+0.198 over token signals) |
| **Token entropy** | AUROC 0.576 — near random | AUROC 0.573 — near random |
| **Signal linearity** | Yes — LR sufficient | Yes — LR 0.775 > MLP 0.733 |
| **Primary failure mode** | FFN parametric over-retrieval | 35% confident hallucinations (low entropy, wrong) |
| **Steering** | Causal structure real, small effect | Universally fails |
| **Output** | 12-step interpretability pipeline | 28-experiment governance system |

### What Both Projects Prove Together

| Question | MECH-INT Answer | HaRP Answer |
|---|---|---|
| Does internal signal exist? | Yes — L8–L9, causally verified (steering inversion at α=40) | Yes — L32, +0.198 over all baselines |
| Is token entropy sufficient? | No — AUROC 0.576 | No — AUROC 0.573 |
| Is the signal linear? | Yes — LR probe sufficient | Yes — LR outperforms MLP at N=700 |
| Is geometry unsupervised? | No — SVD null at 117M (scale-dependent) | Partially — supervised probe required |
| Primary failure mode? | FFN over-retrieval from parametric memory | FFN-driven confident hallucinations (35% of samples) |
| Can we steer it away? | Structurally real, small effect (~0.002 AUROC) | Universally fails — null at all alpha values |
| Can we govern it? | Not at 117M — signal too weak | Yes — ACCEPT/REGENERATE/ABSTAIN at AUROC 0.775 |

### The Two-Part Argument

**In one sentence:** Hallucination is a parametric retrieval failure that writes itself into the residual stream at the right layer — invisible to output probabilities, detectable through hidden-state geometry, and manageable through a calibrated routing policy once you reach sufficient model scale.

MECH-INT answers: **WHERE** is the signal? **WHAT** is the mechanism? **HOW** does it propagate?
HaRP answers: **HOW DO WE USE** the signal? **HOW DO WE GOVERN** it? **HOW DO WE CALIBRATE** it?

### The Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED PIPELINE                                 │
│                                                                     │
│  1. MECHANISTIC PROFILING (from MECH-INT)                           │
│     • Probing sweep → identify optimal layer automatically           │
│     • FFN vs. Attention decomposition → understand failure mode      │
│     • Sparse dimension map → 100 dims instead of 2048               │
│     • Steering validation → confirm signal is causal, not artifact  │
│                  ↓                                                  │
│  2. FEATURE EXTRACTION (from HaRP)                                  │
│     • Extract hidden states at identified optimal layer              │
│     • Compute OOF Group C geometry features (4 features)            │
│     • Add token signals (Group A) + consistency (Group B)           │
│                  ↓                                                  │
│  3. FAILURE ESTIMATION (from HaRP)                                  │
│     • Logistic Regression on combined features                       │
│     • 5-fold OOF to prevent leakage (critical — Exp 09b finding)    │
│     • Calibration check (ECE, temperature scaling)                  │
│                  ↓                                                  │
│  4. GOVERNANCE POLICY (from HaRP)                                   │
│     • Optimize α*, β* thresholds on validation set                  │
│     • Route: ACCEPT / REGENERATE / ABSTAIN                           │
│                  ↓                                                  │
│  5. MECHANISTIC EXPLANATION (from MECH-INT)                         │
│     • For flagged queries: run DLA to identify responsible heads     │
│     • Quadrant taxonomy maps failure mode to mechanism               │
│     • Output: "Confident hallucination — FFN over-retrieval at L32, │
│               primary heads: H19, H24 — route to ABSTAIN"           │
└─────────────────────────────────────────────────────────────────────┘
```

What each project contributes to the merge:

**From MECH-INT:**
- Layer identification methodology — makes HaRP's L32 choice principled, not empirical
- Head-level attribution — explains *why* a query was flagged, not just *that* it was
- Causal validation via steering — confirms the signal is real, not a probe artifact
- Sparse dimension map — enables 100-dim feature extraction instead of full 2048

**From HaRP:**
- OOF computation pattern — prevents the leakage bug (−0.187 AUROC inflation) in any nested probe pipeline
- Governance policy layer — converts a research AUROC into an operational decision
- Calibration framework — ECE 0.039, threshold optimization, CI-aware evaluation
- Four-quadrant taxonomy — maps mechanistic findings (FFN over-retrieval → confident hallucination quadrant) to routing decisions

**HaRP live dashboard:** [https://harpfind.streamlit.app/](https://harpfind.streamlit.app/)

---

## License

MIT
