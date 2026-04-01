# MECH-INT — Research Value, Breadth, and Future Scope

**Project:** Mechanistic Interpretability of GPT-2 Hallucination
**Status:** Complete
**Position in arc:** Project 1 of 5 — lays the causal foundation

---

## Why This Project Matters

MECH-INT is the only project in the arc that asks *how* hallucination happens,
not just *whether* it can be detected. Every downstream project (HaRP, GEOM-PROOF,
FAIL-CHAIN, GUARDIAN) builds detectors and governance systems — but they are all
correlational. MECH-INT provides the causal mechanism underneath.

The finding — hallucination localizes to L8–L9 in GPT-2 via FFN over-retrieval,
not attention misfiring — is architecturally significant because it tells you
*where to look* and *what mechanism to target* in any LLM. The layer depth (33–38%
of network) predates the ~89% depth signal found in HaRP, suggesting the causal
source and the predictive signal live at different depths. That gap is itself
a finding no other project in the arc addresses.

---

## Core Contributions

| Contribution | What it establishes |
|---|---|
| Hallucination localizes to L8–L9 (33–38% depth) | The causal source is early-to-mid network, not late |
| FFN over-retrieval, not attention | The mechanism is feed-forward memorization gone wrong, not attention pattern failure |
| Causal intervention via activation patching | Proves the localization is causal, not correlational |
| L8 direct logit attribution (DLA) | Shows the wrong token is being promoted by specific FFN components |
| Head importance sweep | Identifies which attention heads *suppress* vs *amplify* hallucination |

---

## Mathematical and Architectural Power

**The depth gap theorem (informal):**
MECH-INT finds the causal source at ~35% depth.
HaRP finds the predictive signal at ~89% depth (L32/36 in Qwen 2.5 3B).
GEOM-PROOF finds Fisher J peaks at 69–79% depth across architectures.

This creates a three-layer model of hallucination:
1. **Generation** (35% depth): FFN over-retrieval creates the hallucinated token
2. **Propagation** (35%→89%): the error propagates and crystallizes through the
   residual stream
3. **Manifestation** (89% depth): the error is detectable as geometric separation
   between correct and wrong representations

No paper has formally characterized this three-layer structure. It is implicit in
the arc but never stated. This is a publishable theoretical contribution requiring
no new experiments — only synthesis.

**The FFN intervention target:**
The direct logit attribution (DLA) at L8 identifies specific FFN components
(neurons, not heads) whose activation promotes wrong tokens. These are the
surgical intervention targets. If you can *suppress* these components at inference
time (activation patching, steering vectors, or structured pruning), you could
reduce hallucination at the source — not just detect it post-hoc.

---

## Open Threads With Genuine Future Value

### OT-1: The Depth Gap — Source vs Signal
**Question:** Why does the causal source (L8–L9, ~35%) differ from the predictive
signal (L32, ~89%)? What happens in between?

**Hypothesis:** The residual stream integrates the hallucination signal as it
propagates. Early layers generate the wrong content; middle layers fail to suppress
it; late layers encode the semantic consequence (which is what geometry detects).

**Why it matters:** Locating the suppression failure in the middle layers would
identify the intervention point with the best cost-efficiency tradeoff — earlier
than L32 (cheaper to patch) but later than L8 (less collateral damage).

**What to do:** Run layer-sweep activation patching from the clean run at every
layer between L8 and L32. Plot the "recovery curve" — at which layer does patching
the FFN output restore correct output? The layer where recovery probability first
exceeds 0.5 is the effective propagation boundary.

### OT-2: Cross-Architecture Causal Transfer
**Question:** Does FFN over-retrieval localize at ~35% depth in Qwen 2.5 3B and
Mistral 7B, or is 35% a GPT-2 artifact?

**Why it matters:** If depth-normalized localization transfers, the three-layer
model (generate → propagate → manifest) becomes architecture-universal. If it
doesn't, architectures with different tokenization or MLP designs (e.g., SwiGLU
vs GELU) create different causal structures.

**What to do:** Re-run MECH-INT's DLA sweep on Qwen 2.5 3B using the hidden states
already captured in HaRP. This requires no new generation — HaRP's `hidden_states.npz`
contains all 36 layers. Compute per-layer DLA and compare the localization depth.

### OT-3: FFN Neuron Fingerprinting Across Questions
**Question:** Are the same FFN neurons at L8 responsible for hallucination across
different questions, or does the responsible neuron set change per question?

**Why it matters:** If the responsible neurons are universal (same neurons fire
for all hallucinated responses), you can build a sparse hallucination detector
(check just those neurons) with near-zero overhead. If they are question-specific,
you need the full geometric approach (HaRP, GEOM-PROOF).

**Why this hasn't been answered:** The current MECH-INT analysis averages over
questions. Per-question FFN attribution was not computed.

### OT-4: Sycophancy as FFN Misfire
**Question:** Is sycophancy (model changes answer when user pushes back) driven by
the same L8 FFN over-retrieval mechanism as factual hallucination?

**Hypothesis:** Sycophancy activates different FFN components — not the "stored
wrong fact" pathway but the "social compliance" pathway. Distinguishing these
mechanistically would separate factual hallucination from sycophancy at the
architectural level, informing which governance tools apply to which failure type.

### OT-5: The L8 Signal as a Real-Time Alarm
**What it enables:** The L8 DLA signal can be computed after just 8 layers of
the forward pass. For a 24-layer model (GPT-2 Medium), that's 33% of inference
cost. If the L8 signal predicts final-output hallucination with AUROC > 0.70,
you could abort generation after 8 layers — a 67% compute saving for failed
responses.

**Connection to FAIL-CHAIN:** FAIL-CHAIN's intra-step convergence hypothesis
(does the persona direction signal converge before step 1 generation is complete?)
is the pipeline-level version of this question. MECH-INT's L8 alarm is the
token-level version.

---

## Connection to the PhD Arc

| Downstream project | How MECH-INT informs it |
|---|---|
| HaRP | The L8 causal source motivates checking the L32 predictive signal — different layers for different purposes |
| GEOM-PROOF | The FFN mechanism explains why Fisher J peaks at 69–79% depth: the causal error crystallizes into geometric separation over that depth range |
| FAIL-CHAIN | The depth gap (35% source, 89% detection) reappears as an inter-step gap: step 1 generates the error, step 3 is where geometry detects it. The pipeline is the architecture analog of the residual stream |
| GUARDIAN | The L8 alarm is a prototype for GUARDIAN's step-1 early exit — abort generation at 33% of forward passes based on the causal signal |

---

## The One Sentence This Project Adds to the World

> *Hallucination in GPT-2 is not a failure of attention — it is a failure of
> feed-forward memory at 33–38% network depth, and this causal source can be
> located, characterized, and targeted without training any detector.*
