# References — MECH-INT

*Papers cited or directly compared against in this project.*

---

## Core Methodology

**[1] Elhage et al. — A Mathematical Framework for Transformer Circuits**
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, et al.
*Transformer Circuits Thread*, 2021.
https://transformer-circuits.pub/2021/mathematical-framework/index.html
> Foundation for Direct Logit Attribution (DLA) and residual stream decomposition used throughout this project.

**[2] Nostalgebraist — Interpreting GPT: The Logit Lens**
*Blog post*, 2020.
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
> Logit lens: projecting intermediate hidden states through the unembedding matrix. Used in Stage 7 of our pipeline.

**[3] Zou et al. — Representation Engineering: A Top-Down Approach to AI Transparency (RepE)**
Andy Zou, Long Phan, Sarah Chen, James Zou, et al.
*arXiv:2310.01405*, 2023.
> Difference-of-means estimator for steering vectors. Used to derive the hallucination direction and test activation steering (Stage 12). We validate the causal signature at L9.

---

## Probing & Hidden-State Detection

**[4] Azaria & Mitchell — The Internal State of an LLM Knows When It's Lying (SAPLMA)**
Amos Azaria, Tom Mitchell.
*EMNLP Findings*, 2023.
> First systematic hidden-state probing for LLM truthfulness on GPT-2 medium. Mean-pooling approach adopted from this work.

**[5] Burns et al. — Discovering Latent Knowledge in Language Models Without Supervision (CCS)**
Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt.
*NeurIPS*, 2022.
> Unsupervised elicitation of latent knowledge. Establishes that linear probes find concentrated directions — consistent with our 87% sparsity finding (100/768 active dims).

---

## Related Detectors (Benchmarked / Compared)

**[6] Du et al. — HaloScope: Probing for Hallucinations in Large Vision-Language Models (HaloScope)**
Yijun Du, Jingyi Zhang, et al.
*NeurIPS 2024 Spotlight*.
> SVD subspace membership scoring. We replicate on GPT-2 Medium and find scale-dependent failure: method achieves AUROC 0.75+ on LLaMA-13B but near-chance on GPT-2 (too small).

**[7] ReDeEP — Detecting Hallucinations in LLMs with Residual Stream and FFN Decomposition**
*ICLR*, 2025.
> FFN vs. attention residual stream decomposition in RAG settings. Our finding (FFN dominant in 8/12 layers) is directly consistent — suggests FFN over-retrieval generalises beyond RAG.

**[8] Sriramanan et al. — LLM-Check: Investigating Detection of Hallucinations in Large Language Models (LLM-Check)**
*NeurIPS*, 2024.
> Internal-state eigenvalue analysis. Our finding that peak signal is at mid-to-late layers (L8–L9) is consistent across architectures.

**[9] MIND — Multifaceted In-context No-code Detection**
*ACL*, 2024.
> Unsupervised last-position hidden state for hallucination detection. Directly corroborates our finding that the last token position is most informative.

---

## Dataset

**[10] Lin et al. — TruthfulQA: Measuring How Models Mimic Human Falsehoods**
Stephanie Lin, Jacob Hilton, Owain Evans.
*ACL*, 2022.
> Primary dataset: 817 questions across 38 categories. We use a curated 534-sample subset with binary correctness labels.

---

*Total: 10 references*
