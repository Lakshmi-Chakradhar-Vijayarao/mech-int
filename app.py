"""
MECH-INT: Mechanistic Interpretability of GPT-2 Hallucination
===============================================================
An educational research dashboard walking through a complete
12-stage mechanistic interpretability pipeline — from output-level
surface features to component-level attribution and activation
steering — on GPT-2 (117M) and TruthfulQA.

This is Project 1 of a 5-project PhD research arc on LLM hallucination
governance by Lakshmi Chakradhar Vijayarao.

Run:   streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MECH-INT · Mechanistic Interpretability of GPT-2 Hallucination",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Design tokens ───────────────────────────────── */
:root {
    --primary:    #1565c0;
    --primary-lt: #1976d2;
    --primary-bg: #e3f2fd;
    --green:      #2e7d32;
    --green-bg:   #e8f5e9;
    --amber:      #bf360c;
    --amber-bg:   #fbe9e7;
    --red:        #b71c1c;
    --red-bg:     #ffebee;
    --teal:       #00695c;
    --teal-bg:    #e0f2f1;
    --slate:      #37474f;
    --slate-bg:   #eceff1;
    --bg:         #f4f7fb;
    --surface:    #ffffff;
    --text:       #0d1b2e;
    --text-muted: #6b7280;
    --border:     #d0dff0;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* ── Callout boxes ───────────────────────────────── */
.finding-box {
    background: linear-gradient(135deg, #e3f2fd 0%, #e8eaf6 100%);
    border-left: 5px solid var(--primary);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(21,101,192,0.10);
    font-size: 0.93rem;
}
.good-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 5px solid var(--green);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(46,125,50,0.10);
    font-size: 0.93rem;
}
.warn-box {
    background: linear-gradient(135deg, #fbe9e7 0%, #fff8e1 100%);
    border-left: 5px solid var(--amber);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(191,54,12,0.10);
    font-size: 0.93rem;
}
.null-box {
    background: linear-gradient(135deg, #ffebee 0%, #fce4ec 100%);
    border-left: 5px solid var(--red);
    padding: 1.1rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(183,28,28,0.10);
    font-size: 0.93rem;
}
.analogy-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #e0f2f1 100%);
    border-left: 5px solid var(--teal);
    padding: 0.9rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    font-style: italic;
    box-shadow: 0 2px 8px rgba(0,105,92,0.10);
    font-size: 0.90rem;
}
.limit-box {
    background: linear-gradient(135deg, #eceff1 0%, #f5f5f5 100%);
    border-left: 5px solid var(--slate);
    padding: 1.0rem 1.4rem;
    border-radius: 0 10px 10px 0;
    margin: 0.9rem 0;
    box-shadow: 0 2px 8px rgba(55,71,79,0.08);
    font-size: 0.91rem;
}

/* ── Pills ───────────────────────────────────────── */
.step-pill {
    display: inline-block;
    background: linear-gradient(90deg, #1565c0 0%, #1976d2 100%);
    color: white;
    padding: 0.28rem 0.9rem;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
    box-shadow: 0 2px 6px rgba(21,101,192,0.28);
}
.exp-pill {
    display: inline-block;
    background: linear-gradient(90deg, #0d47a1 0%, #1565c0 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    margin-right: 0.3rem;
    box-shadow: 0 2px 6px rgba(13,71,161,0.25);
}
.novel-pill {
    display: inline-block;
    background: linear-gradient(90deg, #00695c 0%, #00897b 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(0,105,92,0.25);
}
.arc-pill {
    display: inline-block;
    background: linear-gradient(90deg, #4527a0 0%, #5e35b1 100%);
    color: white;
    padding: 0.22rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 6px rgba(69,39,160,0.25);
}

/* ── Stat card ───────────────────────────────────── */
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--primary);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(21,101,192,0.07);
    transition: box-shadow 0.2s, transform 0.15s;
}
.stat-card:hover {
    box-shadow: 0 8px 28px rgba(21,101,192,0.13);
    transform: translateY(-2px);
}
.stat-number {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--primary);
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-number-red    { color: var(--red); }
.stat-number-green  { color: var(--green); }
.stat-number-indigo { color: #1565c0; }
.stat-number-amber  { color: var(--amber); }
.stat-label {
    font-size: 0.76rem;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-top: 0.35rem;
    line-height: 1.3;
}

/* ── General card ────────────────────────────────── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1.1rem 1.3rem;
    border-radius: 10px;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Phase badge ─────────────────────────────────── */
.phase-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(90deg, #0a1929 0%, #1565c0 100%);
    color: #e3f2fd;
    border-radius: 8px;
    padding: 0.38rem 1rem;
    font-size: 0.80rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 8px rgba(21,101,192,0.22);
}

/* ── Feature chips ───────────────────────────────── */
.feature-chip {
    display: inline-block;
    background: var(--primary-bg);
    color: var(--primary);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.15rem;
    font-family: 'SFMono-Regular', Consolas, monospace;
}

/* ── Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0d1e 0%, #0d1a3a 45%, #1a2f6e 80%, #1565c0 100%);
}
section[data-testid="stSidebar"] * { color: #e3f2fd !important; }
section[data-testid="stSidebar"] .stButton button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e3f2fd !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 0.35rem 0.7rem !important;
    margin-bottom: 0.1rem !important;
    transition: background 0.15s !important;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: rgba(255,255,255,0.15) !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

/* ── Typography ──────────────────────────────────── */
h1 { color: #0a1929; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; }
h2 { color: #1565c0; font-weight: 700; }
h3 { color: #00695c; font-weight: 600; }

h1::after {
    content: '';
    display: block;
    width: 56px;
    height: 4px;
    background: linear-gradient(90deg, #1565c0, #42a5f5);
    border-radius: 2px;
    margin-top: 0.4rem;
}

/* ── Progress bar ────────────────────────────────── */
.prog-bar-bg {
    background: rgba(255,255,255,0.15);
    border-radius: 8px;
    height: 6px;
    overflow: hidden;
    margin: 0.3rem 0 0.7rem;
}
.prog-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #1565c0 0%, #42a5f5 100%);
    border-radius: 8px;
}

/* ── Table & misc ────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }
.stDataFrame thead th { background: var(--primary-bg) !important; color: var(--primary) !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
PLOTS_DIR = _HERE / "results" / "plots"


# ── Helpers ───────────────────────────────────────────────────────────────────
def plot_img(name, caption=None, width=None):
    p = PLOTS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=(width is None))
    else:
        st.info(f"Plot not yet generated: {name} — run the corresponding experiment first.")

def pill(text):
    st.markdown(f'<span class="step-pill">{text}</span>', unsafe_allow_html=True)

def exp_pill(text):
    st.markdown(f'<span class="exp-pill">{text}</span>', unsafe_allow_html=True)

def novel_pill(text):
    st.markdown(f'<span class="novel-pill">{text}</span>', unsafe_allow_html=True)

def arc_pill(text):
    st.markdown(f'<span class="arc-pill">{text}</span>', unsafe_allow_html=True)

def finding(text):
    st.markdown(f'<div class="finding-box">{text}</div>', unsafe_allow_html=True)

def good(text):
    st.markdown(f'<div class="good-box">{text}</div>', unsafe_allow_html=True)

def warn(text):
    st.markdown(f'<div class="warn-box">{text}</div>', unsafe_allow_html=True)

def null(text):
    st.markdown(f'<div class="null-box">{text}</div>', unsafe_allow_html=True)

def analogy(text):
    st.markdown(f'<div class="analogy-box"><b>Analogy:</b> {text}</div>', unsafe_allow_html=True)

def limit(text):
    st.markdown(f'<div class="limit-box"><b>Limitation:</b> {text}</div>', unsafe_allow_html=True)

def stat(number, label, color=""):
    st.markdown(
        f'<div class="stat-card"><div class="stat-number {color}">{number}</div>'
        f'<div class="stat-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def phase(text):
    st.markdown(f'<div class="phase-badge">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def page_brief():
    st.title("Research Brief — What This Work Is and Why It Matters")
    st.markdown(
        "##### Plain English introduction — no prior technical knowledge required. "
        "For anyone curious about how AI goes wrong and how we can look inside to find out why."
    )

    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1929,#1565c0);color:#e3f2fd;
    border-radius:12px;padding:1.3rem 1.6rem;margin:0.5rem 0 1.4rem;
    box-shadow:0 4px 20px rgba(21,101,192,0.25);line-height:1.7;">
    <b style="font-size:1.1rem;">The one-sentence version:</b><br>
    When an AI confidently gives you a wrong answer, it sounds exactly like when it gives you a right one —
    this project asks: can we look <em>inside</em> the AI's computations, layer by layer, to find out
    <em>where</em> and <em>how</em> the error is being manufactured?
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: The Hallucination Problem ──────────────────────────────────
    st.subheader("1 — What Is Hallucination, and Why Is It Hard to Catch?")
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        A language model generates text one word at a time. At each step it asks: *"Given everything
        I've seen so far, what word comes next?"* It does this by running the input through a series
        of internal processing stages — called **transformer layers** — and at the end picking the
        most likely next word.

        **The problem:** this process can produce confidently wrong text. The model might write a
        plausible-sounding but completely fabricated answer — and because the phrasing is fluent and
        the tone is certain, a reader has no easy way to tell. This is called **hallucination**.

        **Why doesn't checking the model's confidence help?**

        The most common approach is to check how "spread out" the model's predictions are — called
        *output entropy*. If the model is unsure (high entropy), flag the answer. If it seems certain
        (low entropy), trust it.

        This completely fails for the most dangerous case: **confident hallucination** — when the
        model is wrong *and* certain. Entropy is low. The model passes every surface-level check.
        A standard filter would accept it and show it to a user.
        """)
    with col2:
        st.markdown("""
        <div style="background:#fff8f8;border:2px solid #c62828;border-radius:12px;
        padding:1.2rem;margin-top:0.5rem;">
        <div style="font-weight:800;color:#c62828;font-size:1rem;margin-bottom:0.7rem;">
        A real example from this project</div>
        <div style="font-size:0.88rem;color:#37474f;line-height:1.6;">
        <b>Question:</b> Which vitamin deficiency causes night blindness?<br><br>
        <b>Ground truth:</b> Vitamin A<br><br>
        <b>GPT-2 output (confident, wrong):</b><br>
        <em>"Night blindness is caused by a deficiency in Vitamin D, which is essential for
        retinal rod cell function..."</em><br><br>
        <b>Model's confidence:</b> High — low entropy, high top-1 probability<br><br>
        <span style="color:#c62828;font-weight:700;">An entropy filter would have accepted this.</span>
        </div></div>
        """, unsafe_allow_html=True)

    # ── Section 2: What Mechanistic Interpretability Is ──────────────────────
    st.markdown("---")
    st.subheader("2 — What Is Mechanistic Interpretability?")
    st.markdown("""
    Standard AI evaluation looks at outputs. **Mechanistic interpretability** looks at the computation
    itself — the mathematical operations happening inside the model as it processes a question.

    Think of it like the difference between judging a car by its speed (output evaluation) versus
    opening the hood to study the engine (mechanistic interpretability). One tells you *what* the
    car does; the other tells you *how* and *why*.

    A transformer model like GPT-2 processes every input through **12 successive transformer layers**.
    Each layer performs two operations: an **attention** step (the model decides what parts of the input
    to focus on) and a **feed-forward** step (the model updates its internal representation of the answer).
    Mechanistic interpretability asks: which of these steps is contributing to the wrong answer,
    and at which layer?
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#e3f2fd;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:#1565c0;">Step 1</div>
        <div style="font-weight:700;color:#1565c0;margin:0.4rem 0;">Token-level signals</div>
        <div style="font-size:0.83rem;color:#37474f;">
        Measure what comes out of the model: entropy, confidence, top token probability.
        Fast and simple — but blind to confident hallucination.
        </div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:#2e7d32;">Step 2</div>
        <div style="font-weight:700;color:#2e7d32;margin:0.4rem 0;">Layer-level probing</div>
        <div style="font-size:0.83rem;color:#37474f;">
        Extract hidden states at each of the 12 layers. Train a detector at each layer.
        Find <em>where</em> in the network hallucination information concentrates.
        </div></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff8e1;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:2rem;font-weight:800;color:#bf360c;">Step 3</div>
        <div style="font-weight:700;color:#bf360c;margin:0.4rem 0;">Component-level attribution</div>
        <div style="font-size:0.83rem;color:#37474f;">
        Break each layer into its attention heads and feed-forward network.
        Find <em>which components</em> cause the hallucination — and whether we can steer them.
        </div></div>
        """, unsafe_allow_html=True)

    # ── Section 3: Why GPT-2? ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3 — Why GPT-2? Small Is Interpretable")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        Modern AI models have billions of parameters — GPT-4 is estimated at over one trillion.
        Studying exactly what each component does in a model that large is computationally and
        conceptually intractable.

        **GPT-2 (117 million parameters, 12 layers)** is the right starting point because:

        - Every attention head can be individually analyzed (there are only 96 of them total)
        - Every layer can be probed in a few minutes on a laptop
        - The mechanistic findings are more interpretable: when something is found, it is tractable
          to trace back to specific components
        - Null results are informative: if geometry-based detection fails at 117M, we can argue it
          is a scale-dependent phenomenon — and this motivates the HaRP project (Project 2) at 3B parameters

        This project is explicitly designed as a **foundation study**. The goal is not to build a
        production hallucination detector. The goal is to understand the mechanism.
        """)
    with col2:
        good("""
        <b>Why small models are informative:</b><br><br>
        MECH-INT found that the hallucination signal in GPT-2 is distributed — it does not
        concentrate in a single layer or component. This is a scale-dependent null result:
        at 117M parameters, the model lacks the representational capacity to encode a clean
        truth direction.<br><br>
        This directly motivates HaRP (Project 2): scale up to 3B (Qwen 2.5), apply the same
        probing pipeline, and measure whether a truth geometry emerges. It does — AUROC lifts
        from 0.58 (MECH-INT/GPT-2) to 0.776 (HaRP/Qwen 3B).
        """)

    analogy(
        "Studying hallucination in GPT-2 is like studying disease mechanisms in a simple organism "
        "like a fruit fly before moving to humans. The fly's biology is simpler, every cell is "
        "identifiable, and the findings establish the baseline for what changes at scale."
    )

    # ── Section 4: What this project found ───────────────────────────────────
    st.markdown("---")
    st.subheader("4 — What This Project Actually Found")
    col1, col2 = st.columns(2)
    with col1:
        findings_list = [
            ("✅", "Surface confidence fails as a hallucination detector",
             "Logistic regression on 6 entropy/logit features: AUROC 0.531 ± 0.059; "
             "MLP: 0.576 ± 0.068 — barely above the random baseline of 0.447. "
             "The model's output distribution cannot distinguish hallucination from correct answers reliably."),
            ("✅", "Hallucination signal peaks at Layer 8–9, not the final layer",
             "Layer-by-layer probing of all 13 representation stages shows AUROC rising through early layers, "
             "peaking at L9 (0.583 mean-pooled) and L8 last-token (0.604). The final layer is not the most informative."),
            ("✅", "Last-token at Layer 8 outperforms mean-pooling",
             "The token that determines what gets generated next (last position) carries more hallucination "
             "signal than the average over all token positions. AUROC: 0.604 vs 0.583."),
            ("✅", "Only 100 of 768 dimensions carry the hallucination signal",
             "A sparse L1 probe at L9 (C=0.1) achieves AUROC 0.589 using only 100 of 768 features (87% sparsity). "
             "The signal is low-dimensional — concentrated in a small subspace of the hidden state."),
            ("✅", "FFN dominates hallucination causation in 8 of 12 stages",
             "Direct Logit Attribution assigns causal credit to attention vs. feed-forward network at each layer. "
             "FFN is dominant in 8 of 12 analysis stages, peaking at L8 AUROC 0.605."),
            ("✅", "Attention head L8 shows +80% relative difference (correct vs. hallucinated)",
             "The normalized DLA signal at L8 attention shows the largest relative gap between correct and "
             "hallucinated samples — the strongest normalized signal despite being weaker in absolute terms."),
        ]
        for icon, title, detail in findings_list:
            st.markdown(
                f'<div style="border:1px solid #c8e6c9;border-radius:10px;padding:0.9rem 1rem;'
                f'margin-bottom:0.6rem;background:#f9fffe;">'
                f'<span style="font-size:0.72rem;font-weight:700;color:#2e7d32;">{icon}</span><br>'
                f'<b style="font-size:0.90rem;color:#0a1929;">{title}</b><br>'
                f'<span style="font-size:0.82rem;color:#546e7a;line-height:1.5;">{detail}</span></div>',
                unsafe_allow_html=True,
            )
    with col2:
        null_list = [
            ("⚠", "Activation steering at α=40 only achieves P(correct) = 0.494",
             "Injecting the truth vector into the residual stream at L8–L9 barely exceeds random chance "
             "(0.50). The model's wrong beliefs are not easily overwritten by patching the activation stream "
             "at this scale. This null result is honestly reported."),
            ("⚠", "Three distinct stages emerge but do not converge to a single causal site",
             "Stage 1 (context routing, L3 Attn 0.617), Stage 2 (FFN over-retrieval, L8–L9), and "
             "Stage 3 (output commitment, L11 H6/H7) are identifiable but distributed. "
             "There is no single 'hallucination neuron' to ablate."),
            ("⚠", "AUROC ceiling of 0.604 — insufficient for production detection",
             "The best result on GPT-2 (0.604 last-token L8) is well below what a practical detector "
             "needs (>0.80 minimum). This is expected at 117M parameters — see HaRP (Project 2) for "
             "what happens at 3B where AUROC reaches 0.776."),
            ("⚠", "All experiments run on MacBook Air Apple Silicon — zero cloud compute",
             "This is both a constraint and a feature. Every finding here is reproducible by anyone "
             "with a laptop. No API keys, no GPU clusters, no cloud costs. The tradeoff is that "
             "larger models and larger datasets were not feasible."),
        ]
        for icon, title, detail in null_list:
            st.markdown(
                f'<div style="border:1px solid #ffcdd2;border-radius:10px;padding:0.9rem 1rem;'
                f'margin-bottom:0.6rem;background:#fff8f8;">'
                f'<span style="font-size:0.72rem;font-weight:700;color:#c62828;">{icon} Honest limit</span><br>'
                f'<b style="font-size:0.90rem;color:#0a1929;">{title}</b><br>'
                f'<span style="font-size:0.82rem;color:#546e7a;line-height:1.5;">{detail}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Section 5: Where to go next ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("5 — Where to Go Next in This Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#e3f2fd;border-radius:10px;padding:1rem;">
        <b style="color:#1565c0;">If you want the full pipeline</b><br><br>
        <span style="font-size:0.85rem;">
        → <b>The 12-Stage Pipeline</b> — overview of all analysis steps<br>
        → <b>Surface Predictor</b> — why entropy fails<br>
        → <b>Layer Probing</b> — where the signal lives
        </span></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:10px;padding:1rem;">
        <b style="color:#2e7d32;">If you want the mechanistic findings</b><br><br>
        <span style="font-size:0.85rem;">
        → <b>Logit Lens</b> — when the error gets locked in<br>
        → <b>Direct Logit Attribution</b> — FFN vs Attention<br>
        → <b>Attention Head Analysis</b> — two heads that commit
        </span></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff8e1;border-radius:10px;padding:1rem;">
        <b style="color:#bf360c;">If you want the research arc</b><br><br>
        <span style="font-size:0.85rem;">
        → <b>Steering &amp; Connections</b> — what failed and what comes next<br>
        → How this feeds into HaRP, GEOM-PROOF, FAIL-CHAIN, GUARDIAN
        </span></div>
        """, unsafe_allow_html=True)


def page_pipeline():
    st.title("The 12-Stage Analysis Pipeline")
    phase("Full arc — Surface → Hidden States → Component-Level")

    st.markdown("""
    MECH-INT is a **12-stage mechanistic interpretability pipeline** applied to GPT-2 (117M) on TruthfulQA.
    Each stage is a discrete analysis step with a defined hypothesis, method, and result.
    Together they form a logical chain from the model's output surface down to individual attention heads.
    """)

    # ASCII pipeline diagram
    st.markdown("""
    <div style="background:#0a1929;color:#90caf9;font-family:'SFMono-Regular',Consolas,monospace;
    font-size:0.80rem;line-height:1.7;padding:1.3rem 1.6rem;border-radius:12px;margin:0.5rem 0 1.2rem;
    overflow-x:auto;white-space:pre;">
  INPUT: TruthfulQA (534 samples, 50/50 correct/hallucinated, Jaccard ≥ 0.12 labeling)
        |
        v
  +-------------------------------------------------------------+
  |  STAGE 1: SURFACE (Steps 1-3)                               |
  |  What comes out of the model?                               |
  |                                                             |
  |  Step 1  -- Data construction + Jaccard labeling (>=0.12)  |
  |  Step 2  -- Baseline: random predictor AUROC 0.447          |
  |  Step 3  -- Surface predictor: 6 entropy/logit features     |
  |             LR AUROC 0.531 +/-0.059                         |
  |             MLP AUROC 0.576 +/-0.068                        |
  +------------------------+------------------------------------+
                           | Surface fails -> go deeper
                           v
  +-------------------------------------------------------------+
  |  STAGE 2: HIDDEN STATES (Steps 4A-5)                        |
  |  What do the internal representations reveal?               |
  |                                                             |
  |  Step 4A -- Layer-by-layer mean-pooled probing (L0-L12)    |
  |             Peak: L9 AUROC 0.583 +/- 0.050                  |
  |  Step 4B -- Token-position probe: last-token at L8          |
  |             AUROC 0.604  <-- headline number                |
  |  Step 4C -- Sparse L1 probe (C=0.1): 100/768 dims          |
  |             AUROC 0.589, 87% sparsity                       |
  |  Step 5  -- Logit lens: when does the error commit?         |
  |             DLA difference baseline AUROC: 0.576            |
  +------------------------+------------------------------------+
                           | Where? -> How?
                           v
  +-------------------------------------------------------------+
  |  STAGE 3: COMPONENT-LEVEL (Steps 6-12)                      |
  |  Which transformer components cause the error?              |
  |                                                             |
  |  Step 6  -- Direct Logit Attribution: FFN vs Attention      |
  |             FFN dominates 8/12 stages; peak L8 AUROC 0.605  |
  |  Step 7  -- Relative DLA: L8 Attn +80% relative diff        |
  |  Step 8  -- Head-level ablation at L11 H6/H7                |
  |  Step 9  -- Attention patterns analysis (L8)                |
  |  Step 10 -- Activation steering experiment                  |
  |             alpha=40 -> P(correct)=0.494 (near-random)      |
  |  Step 11 -- Steering layer sweep                            |
  |  Step 12 -- Three-stage causal model synthesis              |
  +-------------------------------------------------------------+
        |
        v
  THREE-STAGE HALLUCINATION MODEL:
    Stage 1: Context routing  (L3 Attn AUROC 0.617)
    Stage 2: FFN over-retrieval  (L8-L9, FFN dominant)
    Stage 3: Output commitment  (L11 H6/H7)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("All 12 Steps at a Glance")

    steps = [
        ("Step 1",  "Data Construction",         "#1565c0", "#e3f2fd",
         "534 samples from TruthfulQA. Labels by Jaccard word-overlap ≥ 0.12 against ground truth. "
         "266 correct / 268 hallucinated — near-perfect 50/50 balance.",
         "Surface"),
        ("Step 2",  "Random Baseline",            "#1565c0", "#e3f2fd",
         "Empirical random predictor on the 50/50 split. Measured AUROC: 0.447. "
         "Sets the floor for all subsequent experiments.",
         "Surface"),
        ("Step 3",  "Surface Predictor",          "#bf360c", "#fbe9e7",
         "Logistic regression and MLP on 6 entropy/logit features. "
         "LR AUROC 0.531 ± 0.059; MLP AUROC 0.576 ± 0.068. Above chance — but insufficient. "
         "Motivates hidden-state analysis.",
         "Surface"),
        ("Step 4A", "Mean-Pooled Layer Probing",  "#2e7d32", "#e8f5e9",
         "Linear probe on mean-pooled hidden states at each of 13 layers (L0–L12). "
         "Peak: L9 AUROC 0.583 ± 0.050.",
         "Hidden States"),
        ("Step 4B", "Token-Position Probe",       "#2e7d32", "#e8f5e9",
         "Probe specifically the last-token hidden state at L8 (most informative position). "
         "AUROC 0.604 — the headline detection number for this project.",
         "Hidden States"),
        ("Step 4C", "Sparse L1 Probe",            "#2e7d32", "#e8f5e9",
         "L1-regularized sparse probe at L9 (C=0.1). Uses only 100 of 768 dimensions (87% sparsity). "
         "AUROC 0.589 — signal is concentrated in a low-dimensional subspace.",
         "Hidden States"),
        ("Step 5",  "Logit Lens",                 "#2e7d32", "#e8f5e9",
         "Project intermediate hidden states through the unembedding matrix at each layer. "
         "DLA difference baseline AUROC: 0.576. Shows when the model commits to the wrong token.",
         "Hidden States"),
        ("Step 6",  "Direct Logit Attribution",   "#00695c", "#e0f2f1",
         "Decompose each layer's contribution to the final logit into FFN and Attention components. "
         "FFN dominates 8 of 12 attribution stages. FFN peak AUROC at L8: 0.605.",
         "Component"),
        ("Step 7",  "Relative DLA",               "#00695c", "#e0f2f1",
         "Normalize DLA by component magnitude to find relative differences. "
         "L8 Attention shows +80% relative difference between correct and hallucinated samples — "
         "the strongest normalized signal despite being weaker in absolute terms.",
         "Component"),
        ("Step 8",  "Attention Head Ablation",    "#00695c", "#e0f2f1",
         "Ablate individual attention heads at each layer and measure impact on hallucination rate. "
         "L11 H6 and H7 identified as output-commitment heads.",
         "Component"),
        ("Step 9",  "Attention Pattern Analysis", "#00695c", "#e0f2f1",
         "Visualize full attention pattern matrices at L8. "
         "Correct vs. hallucinated samples show different attention concentration patterns.",
         "Component"),
        ("Step 10", "Activation Steering",        "#b71c1c", "#ffebee",
         "Inject the truth direction vector into the residual stream at optimal alpha. "
         "Alpha=40 achieves P(correct)=0.494 — near chance. Null result, honestly reported.",
         "Component"),
        ("Step 11", "Steering Layer Sweep",       "#b71c1c", "#ffebee",
         "Test steering injection across all layers and alpha values. "
         "No layer-alpha combination substantially improves over baseline.",
         "Component"),
        ("Step 12", "Three-Stage Synthesis",      "#4527a0", "#ede7f6",
         "Integrate all component findings into a unified three-stage causal model: "
         "Context routing (L3) → FFN over-retrieval (L8–L9) → Output commitment (L11 H6/H7).",
         "Synthesis"),
    ]

    stage_colors = {"Surface": "#1565c0", "Hidden States": "#2e7d32",
                    "Component": "#00695c", "Synthesis": "#4527a0"}

    for step_label, title, color, bg, desc, stage in steps:
        sc = stage_colors.get(stage, "#1565c0")
        st.markdown(
            f'<div style="border-left:4px solid {color};background:{bg};'
            f'border-radius:0 10px 10px 0;padding:0.75rem 1.1rem;margin-bottom:0.4rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span style="font-size:0.72rem;font-weight:700;color:{color};">{step_label}</span>'
            f'<span style="font-size:0.70rem;font-weight:700;color:{sc};background:{sc}22;'
            f'padding:0.1rem 0.5rem;border-radius:12px;">{stage}</span></div>'
            f'<b style="font-size:0.91rem;color:#0a1929;">{title}</b><br>'
            f'<span style="font-size:0.81rem;color:#546e7a;">{desc}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("What Converges on Layers 8–9")
    col1, col2, col3 = st.columns(3)
    with col1:
        finding(
            "<b>Layer probing (Steps 4A/4B):</b><br>"
            "Mean-pooled probe peaks at L9 (0.583). "
            "Last-token probe peaks at L8 (0.604). "
            "Both point to the same mid-to-late network region."
        )
    with col2:
        finding(
            "<b>Direct Logit Attribution (Step 6):</b><br>"
            "FFN at L8 achieves AUROC 0.605 — the strongest single-component signal. "
            "The feed-forward network at this layer is doing something causally important."
        )
    with col3:
        finding(
            "<b>Relative DLA (Step 7):</b><br>"
            "Attention at L8 shows +80% relative difference. "
            "Even though attention is weaker in absolute terms, "
            "it shows the sharpest normalized gap between correct and hallucinated."
        )

    limit(
        "All 12 steps run on GPT-2 (117M), TruthfulQA (534 samples), MacBook Air Apple Silicon. "
        "Results may not generalize to larger models — this is a feature, not a bug. "
        "The explicit purpose is to establish a mechanistic baseline at small scale."
    )


def page_surface():
    st.title("Surface Predictor — Step 3")
    exp_pill("Step 3 — Surface Feature Extraction")
    phase("Stage 1 — Surface Analysis")

    st.markdown("""
    **Scientific question:** Can we detect hallucination from GPT-2's output distribution alone —
    without ever looking inside the model? This is the approach most deployed systems use.
    The honest answer establishes the floor for everything that follows.
    """)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: stat("0.531 ± 0.059", "LR AUROC (6 features)", "stat-number-amber")
    with c2: stat("0.576 ± 0.068", "MLP AUROC (6 features)", "stat-number-amber")
    with c3: stat("0.447", "Random Baseline AUROC", "stat-number-red")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("roc_curve.png", "ROC curve — surface predictor vs random baseline")
    with col2:
        st.subheader("The 6 Surface Features")
        features = [
            ("output_entropy",
             "Shannon entropy over the full output token probability distribution. "
             "Low entropy = confident model. High entropy = uncertain model."),
            ("max_token_prob",
             "Probability of the single most likely next token. "
             "High value = model is narrowing in on one answer."),
            ("top1_logit",
             "Raw logit score for the top-ranked token before softmax. "
             "Captures the unnormalized prediction strength."),
            ("logit_gap",
             "Difference between top-1 and top-2 logit scores. "
             "Large gap = more decisive prediction."),
            ("logit_variance",
             "Variance across the top-k logit scores. "
             "High variance = uneven probability mass distribution."),
            ("first_token_entropy",
             "Entropy computed from only the first generated token's distribution. "
             "Captures how confident the model is at the very start of its answer."),
        ]
        for fname, desc in features:
            st.markdown(
                f'<div class="card" style="padding:0.5rem 0.8rem;margin-bottom:0.4rem;">'
                f'<code style="color:var(--primary);font-size:0.82rem;">{fname}</code><br>'
                f'<span style="font-size:0.82rem;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Why 0.576 Is Above Chance — But Not Enough")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        AUROC 0.576 means: given a random correct sample and a random hallucinated sample,
        the surface predictor ranks the correct one higher about **57.6% of the time**.
        Compare to the random baseline of 0.447 — so it is above chance and learning a real signal.

        **Why is this not enough?**

        The 0.576 figure is achieved primarily on *epistemically uncertain* hallucinations —
        cases where the model genuinely doesn't know and its entropy reflects that uncertainty.

        For **confident hallucinations** — the most dangerous case — entropy is *low*. The model
        is fully committed to the wrong answer. All 6 surface features are uninformative in this
        regime: entropy is low, top1 probability is high, logit gap is large. Every surface signal
        says "trust me" — and the model is wrong.
        """)
    with col2:
        null(
            "<b>The structural failure:</b><br><br>"
            "Surface features measure <em>how certain</em> the model is — not whether it is right. "
            "A confident hallucination produces the same surface signature as a confident correct answer. "
            "These two cases are indistinguishable from the output distribution alone. "
            "You must look inside the model to separate them — which is exactly what Steps 4–12 do."
        )
        finding(
            "<b>Why this failure motivates MECH-INT:</b><br><br>"
            "The MLP achieves 0.576 with 6 hand-crafted features. This is the best a surface-only "
            "approach can do on GPT-2/TruthfulQA. Every subsequent experiment in this project asks: "
            "<em>can we do better by looking at what's happening inside?</em> The answer is yes — "
            "but only modestly (peak 0.604 at L8 last-token). The improvement is real but small, "
            "which is itself a mechanistic finding about GPT-2's representational capacity."
        )

    st.markdown("---")
    st.subheader("Result Summary Table")
    rows = [
        ("Random predictor (empirical)", "0.447", "±0.000", "Coin flip on 50/50 split"),
        ("LR — 6 surface features",      "0.531", "±0.059", "Above chance; learns marginal signal"),
        ("MLP — 6 surface features",     "0.576", "±0.068", "Best surface result; large variance"),
        ("Layer 8 last-token probe",      "0.604", "±~0.05", "Step 4B — internal representation"),
        ("FFN L8 DLA",                    "0.605", "±~0.05", "Step 6 — component attribution"),
    ]
    df = pd.DataFrame(rows, columns=["Method", "AUROC", "Std", "Notes"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    warn(
        "<b>All AUROC values for GPT-2 are modest.</b> This is expected and scientifically meaningful. "
        "At 117M parameters, the model does not have sufficient representational capacity to cleanly "
        "encode a linear truth direction — a finding that directly motivated HaRP (Project 2), "
        "which uses Qwen 2.5 3B and achieves AUROC 0.776 with full geometry signals."
    )

    plot_img("confusion_matrix.png", "Confusion matrix — surface predictor at default threshold")


def page_probing():
    st.title("Layer Probing — Steps 4A / 4B / 4C")
    exp_pill("Step 4A — Mean-Pooled Probing")
    exp_pill("Step 4B — Token-Position Probe")
    exp_pill("Step 4C — Sparse Probe")
    phase("Stage 2 — Hidden State Analysis")

    st.markdown("""
    **Scientific question:** At which layer does a linear probe best discriminate correct answers
    from hallucinations? This is the core hidden-state analysis — it tells us *where* in the network
    the hallucination information is encoded, and how much of the hidden space it occupies.
    """)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat("L9", "Peak Layer (Mean-Pooled)")
    with c2: stat("0.583 ± 0.050", "L9 Mean-Pooled AUROC", "stat-number-indigo")
    with c3: stat("0.604", "L8 Last-Token AUROC", "stat-number-green")
    with c4: stat("100 / 768", "Active Dims (Sparse Probe)", "stat-number-indigo")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("layer_probing_curve.png", "Probing AUROC by transformer layer — peaks at L8–L9")
    with col2:
        st.subheader("Layer-by-Layer AUROC Table")
        layer_data = [
            ("0 (Embed)",  0.501, "±0.062", "Near random — no processing yet"),
            ("L1",         0.508, "±0.058", "Minimal signal"),
            ("L2",         0.519, "±0.055", "Early pattern detection"),
            ("L3",         0.534, "±0.051", "Context routing begins"),
            ("L4",         0.541, "±0.050", "Gradual rise"),
            ("L5",         0.549, "±0.049", "Mid-network"),
            ("L6",         0.554, "±0.049", "Consistent rise"),
            ("L7",         0.561, "±0.050", "Approaching peak region"),
            ("L8",         0.577, "±0.051", "Peak region — DLA peak (L8 last-token: 0.604)"),
            ("L9",         0.583, "±0.050", "Mean-pooled peak"),
            ("L10",        0.572, "±0.052", "Slight post-peak drop"),
            ("L11",        0.565, "±0.053", "Output commitment heads here"),
            ("L12 (final)",0.558, "±0.055", "Final layer — not most informative"),
        ]
        df_layers = pd.DataFrame(
            [(r[0], f"{r[1]:.3f}", r[2], r[3]) for r in layer_data],
            columns=["Layer", "AUROC", "Std", "Notes"]
        )
        st.dataframe(df_layers, hide_index=True, use_container_width=True)
        finding(
            "<b>Key finding:</b> AUROC rises monotonically through L0–L9, peaks at L9 (mean-pooled), "
            "then slightly decreases. The final layer (L12) is not the most informative — "
            "hallucination information is maximally encoded in the mid-to-late network."
        )

    st.markdown("---")
    st.subheader("Step 4B: Token-Position Analysis")
    col1, col2 = st.columns([2, 3])
    with col1:
        plot_img("token_position_heatmap.png",
                 "Token position × layer AUROC heatmap — last-token at L8 is the peak")
    with col2:
        st.subheader("Why Last-Token Matters")
        st.markdown("""
        In GPT-2's autoregressive architecture, the **last token position** is special: it is
        the position that attends to the full input context and directly determines the next
        generated token. Its hidden state is the representation that gets projected through
        the unembedding matrix to produce the output distribution.

        Mean-pooling over all token positions (Step 4A) dilutes this signal by including
        positions that represent question words, punctuation, and context tokens — none of which
        directly determine what the model will say next.

        **The result:** Last-token probe at L8 achieves AUROC 0.604 vs. 0.583 for mean-pooled
        at L9. The improvement is modest (+0.021) but consistent — and it points to L8 as the
        critical layer for last-token hallucination encoding.
        """)
        good(
            "<b>The 'write-in' phenomenon:</b> From L9 onward, probing AUROC plateaus. "
            "This is consistent with the idea that the model finalizes its answer decision at L8–L9 "
            "and subsequent layers primarily format and commit the output — they do not add new "
            "discriminative information about correctness."
        )
        st.markdown("""
        **Practical implication:** In any hidden-state-based hallucination detector for GPT-2,
        the last-token hidden state at L8 is the optimal feature. This is the single number this
        project contributes to the mechanistic interpretability literature on GPT-2.
        """)

    st.markdown("---")
    st.subheader("Step 4C: Sparse Probe — How Many Dimensions Carry the Signal?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        A dense linear probe uses all 768 dimensions of GPT-2's hidden state. This risks fitting
        noise. An **L1-regularized sparse probe** (C=0.1, logistic regression with L1 penalty)
        selects only the most predictive dimensions.

        **Result at L9 (C=0.1):**
        - AUROC: **0.589**
        - Active dimensions: **100 / 768** (87% sparsity)
        - Compared to dense probe at L9: 0.583

        The sparse probe *matches or slightly exceeds* the dense probe while using only 13% of
        the available dimensions. This tells us the hallucination signal is:
        1. **Low-dimensional** — concentrated in a small subspace of R^768
        2. **Robust** — the 100 active dimensions are truly informative, not noise
        3. **Interpretable** — we can inspect exactly which dimensions contribute
        """)
    with col2:
        plot_img("calibration_curve.png", "Calibration curve — probe confidence vs. actual accuracy")
        finding(
            "<b>Sparsity finding:</b> 100/768 active dimensions (87% sparsity) achieving "
            "AUROC 0.589 means the hallucination signal in GPT-2's L9 hidden state occupies "
            "a low-dimensional subspace. This is consistent with the sparse coding hypothesis "
            "in mechanistic interpretability — features are not distributed uniformly across "
            "all dimensions but concentrated in a small subset."
        )

    limit(
        "All probing results use 5-fold cross-validation on 534 samples — standard deviations "
        "of ±0.05 are non-trivial. The sparse probe's active dimensions were identified on the "
        "full dataset; a fully OOF-corrected version would select features on training folds only. "
        "All results are specific to GPT-2 (768-dimensional hidden states, 12 layers). "
        "At larger models (2048-dim, 36 layers), the signal structure is expected to be richer — "
        "see HaRP (Project 2)."
    )


def page_logit_lens():
    st.title("Logit Lens — Step 5")
    exp_pill("Step 5 — Logit Lens & Commitment Analysis")
    phase("Stage 2 — When Does the Error Get Locked In?")

    st.markdown("""
    **Scientific question:** At which layer does GPT-2 "decide" to produce the wrong token?
    The logit lens lets us project intermediate hidden states through the final unembedding matrix
    to see what the model would output if processing stopped at each layer.
    """)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: stat("0.576", "DLA Difference Baseline AUROC", "stat-number-indigo")
    with c2: stat("L8–L9", "Error Commitment Region")
    with c3: stat("12", "Layers Analyzed (L0–L11)")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        plot_img("logit_lens_gold.png",
                 "Logit lens — gold token probability across layers (correct vs. hallucinated)")
    with col2:
        st.subheader("The Logit Lens Method")
        st.markdown("""
        The **logit lens** (nostalgebraist, 2020) applies the model's unembedding matrix W_U
        to the intermediate hidden state at each layer, after layer normalization. At each layer l,
        this gives the probability distribution the model *would* produce if it stopped processing there.

        By tracking the **gold-token probability** (the probability assigned to the ground-truth
        correct answer), we can see:

        - **Correct samples:** Gold token probability rises through early layers, stabilizes high.
        - **Hallucinated samples:** Gold token probability rises early but then *drops* as the model
          locks in the wrong answer in mid-to-late layers.

        **The commitment region (L8–L9):** This is where the divergence between correct and
        hallucinated trajectories is largest. The error is not present at L1–L3 — it gets
        introduced in the L4–L8 region and becomes irreversible by L9.
        """)
        finding(
            "<b>Key finding:</b> By the time GPT-2 reaches L8–L9, the wrong token's logit has "
            "risen above the correct token's logit, and this ordering is maintained through L12. "
            "This is mechanistically consistent with the FFN over-retrieval hypothesis: the "
            "feed-forward network at L8–L9 is writing in an incorrect fact from its memorized associations."
        )

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        plot_img("logit_lens_divergence.png",
                 "Logit lens divergence: correct minus hallucinated gold-token probability per layer")
    with col2:
        st.subheader("The DLA Difference Baseline")
        st.markdown("""
        Direct Logit Attribution (DLA) measures how much each component contributes to the final
        logit difference between the correct and hallucinated token.

        The **DLA difference baseline** of AUROC 0.576 is achieved by using the aggregate
        logit difference as a hallucination score. It is computed before decomposing into FFN
        vs. Attention components.

        **Interpretation:** AUROC 0.576 from the total logit difference aligns closely with the surface
        MLP predictor result (0.576) — confirming that the logit-level signal is consistent
        but limited. The gain from component decomposition (FFN peak 0.605) shows that
        attributing to specific components reveals more discriminative signal.
        """)
        analogy(
            "The logit lens is like reading a draft document at each editing stage. You can see "
            "when a wrong piece of information was introduced — not by looking at the final version, "
            "but by reading version 4, 5, 6, and seeing the error appear at version 5 and persist."
        )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        plot_img("subspace_vs_supervised.png",
                 "Subspace analysis vs. supervised probe — comparing signal sources")
    with col2:
        st.subheader("Connection to Subspace Analysis")
        st.markdown("""
        The logit lens reveals *when* the error commits. The subspace analysis reveals *where*
        in the representation space it lives.

        The comparison between subspace projection (unsupervised) and supervised probing shows:
        - The subspace approach (projecting onto the principal component of hallucinated-vs-correct
          hidden state differences) achieves lower AUROC than the supervised probe.
        - This confirms that the hallucination signal in GPT-2 is not a clean, single linear
          direction — it is distributed and requires a supervised classifier to extract.

        **This is the key difference from larger models:** In Qwen 2.5 3B (HaRP, Project 2),
        the first principal component correlates strongly with the truth label. In GPT-2 (117M),
        the correlation is weak. Scale matters for geometry.
        """)
        warn(
            "<b>Scale dependence:</b> The logit lens commitment region (L8–L9) is GPT-2-specific. "
            "In models with more layers, the commitment region is expected to shift toward later "
            "relative depths. The absolute layer number is less meaningful than the relative depth "
            "within the network (L8–L9 is approximately 67–75% of the way through GPT-2)."
        )


def page_dla():
    st.title("Direct Logit Attribution — Steps 6 & 7")
    exp_pill("Step 6 — Component Attribution")
    exp_pill("Step 7 — Relative DLA")
    phase("Stage 3 — Which Components Cause Hallucination?")

    st.markdown("""
    **Scientific question:** The layer-probing results point to L8–L9. But which *part* of
    those layers is causally responsible? Each transformer layer has two components:
    an **attention sublayer** and a **feed-forward network (FFN)** sublayer.
    Direct Logit Attribution decomposes the causal credit between them.
    """)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: stat("8 / 12", "Stages where FFN Dominates", "stat-number-indigo")
    with c2: stat("0.605", "FFN Peak AUROC (L8)", "stat-number-green")
    with c3: stat("+80%", "Attn L8 Relative Difference", "stat-number-amber")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("component_comparison.png",
                 "Component comparison — FFN vs. Attention AUROC across all layers")
    with col2:
        st.subheader("What Is Direct Logit Attribution?")
        st.markdown("""
        DLA decomposes the model's final logit prediction into additive contributions from each
        component at each layer. For GPT-2, the residual stream at each layer is updated by:

        1. **Multi-head Attention** — reads from previous positions and updates the residual stream
        2. **Feed-Forward Network (FFN)** — applies two linear transformations with a nonlinearity;
           often described as the model's "factual memory" or "key-value store"

        For each sample, DLA measures how much each component shifts the logit toward the correct
        vs. hallucinated token. The difference between correct and hallucinated samples becomes
        the discrimination signal.

        **Why this matters:** If FFN dominates, the error is in the model's memorized associations
        (what it "knows" about the world). If Attention dominates, the error is in how the model
        routes and combines context (what it "attends to" from the input). These imply different
        interventions.
        """)
        finding(
            "<b>FFN dominance finding:</b> FFN contributes more to hallucination causation than "
            "Attention in 8 of 12 analysis stages. This is consistent with the <em>FFN over-retrieval</em> "
            "hypothesis: GPT-2 is retrieving a memorized but incorrect fact from its FFN memory banks "
            "at layers L8–L9, overriding the context-based information that Attention is routing."
        )

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        plot_img("dla_comparison.png",
                 "DLA comparison — absolute FFN vs. Attention contribution by layer")
    with col2:
        plot_img("dla_relative.png",
                 "Relative DLA — normalized by component magnitude; Attention at L8 shows +80%")

    st.markdown("---")
    st.subheader("Step 7: Why Relative DLA Reveals Something Different")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Absolute DLA** measures the raw difference in logit contributions between correct and
        hallucinated samples. FFN wins here because it makes large, decisive contributions to
        the final logit.

        **Relative DLA** divides the DLA difference by the component's own contribution magnitude.
        This normalizes away size effects and measures: *proportionally, how much does this
        component change its behavior between correct and hallucinated samples?*

        The result is striking: **L8 Attention shows a +80% relative difference** — the largest
        normalized gap of any component. This means:
        - In absolute terms: Attention at L8 makes a moderate contribution
        - In relative terms: Attention at L8 behaves *very differently* for correct vs. hallucinated samples

        **Interpretation:** The attention heads at L8 are not the primary logit-writers (FFN is),
        but they are *highly sensitive* to whether the model is about to hallucinate. They may be
        acting as a **routing gate** — deciding which FFN memory gets activated at L8–L9.
        """)
    with col2:
        finding(
            "<b>The attention-as-gate hypothesis:</b><br><br>"
            "If L8 attention is relatively more differentiated (correct vs. hallucinated) than any other "
            "component, while L8 FFN has the highest absolute AUROC — this is consistent with attention "
            "acting as a selector that routes the FFN computation. "
            "Correct samples route attention to the right context, so FFN retrieves the right fact. "
            "Hallucinated samples route attention incorrectly, so FFN retrieves a plausible but wrong fact. "
            "The two processes (routing by Attention, retrieval by FFN) are coupled — which is why "
            "both show strong hallucination signal at L8."
        )
        analogy(
            "FFN is the library. Attention is the librarian who decides which shelf to go to. "
            "When the librarian points to the wrong shelf (hallucinated attention routing), "
            "the library returns the wrong book (FFN retrieves wrong fact). "
            "The librarian's relative change in behavior is more diagnostic than the library's size."
        )

    limit(
        "DLA assumes a linear decomposition of logit contributions. In transformer models, "
        "attention and FFN interact non-linearly through layer normalization and residual connections "
        "— the DLA decomposition is an approximation. The 'FFN dominates' finding should be "
        "interpreted as 'FFN makes larger absolute contributions to logit differences on average', "
        "not as a claim of strict causal isolation."
    )


def page_attention():
    st.title("Attention Head Analysis — Steps 8 & 9")
    exp_pill("Step 8 — Head Ablation")
    exp_pill("Step 9 — Attention Pattern Analysis")
    phase("Stage 3 — Which Specific Heads Commit the Error?")

    st.markdown("""
    **Scientific question:** DLA identified L8–L9 as the critical region and FFN as the dominant
    contributor. But within attention, are there specific heads that lock in the hallucination?
    Head-level analysis zooms in from layers to individual attention heads.
    """)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1: stat("L11 H6 / H7", "Output-Commitment Heads")
    with c2: stat("144", "Total Attention Heads (12 layers × 12 heads)")
    with c3: stat("L8", "Primary DLA Attribution Layer")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        plot_img("ablation_heatmap.png",
                 "Ablation heatmap — impact of zeroing each head on hallucination rate (layer × head)")
    with col2:
        st.subheader("The Ablation Method")
        st.markdown("""
        **Head ablation** tests: what happens to the model's predictions when we zero out a
        specific attention head? If ablating a head changes the hallucination rate significantly,
        that head is causally involved.

        For each of the 144 attention heads:
        1. Zero out the head's output by masking its attention weights
        2. Run inference on all 534 samples
        3. Measure: does the rate of hallucination increase or decrease?

        **Heads that increase correct answers when ablated:** These heads were *contributing to*
        hallucination — removing their influence helps.

        **Heads that decrease correct answers when ablated:** These heads were *supporting*
        correct answers — removing them hurts.
        """)
        finding(
            "<b>L11 H6 and H7 — output commitment heads:</b><br><br>"
            "These two heads at Layer 11 (the penultimate layer) show the strongest sensitivity "
            "to ablation in the direction of output commitment. When ablated together, they disrupt "
            "the model's ability to finalize its answer selection — the hallucination rate changes "
            "most at this ablation. They appear to function as the final gate before the output "
            "projection, locking in the token that will be generated."
        )

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        plot_img("head_dla_L8.png",
                 "Head-level DLA at Layer 8 — per-head contribution to correct vs. hallucinated logit")
    with col2:
        plot_img("attention_patterns_L8.png",
                 "Attention patterns at Layer 8 — correct (top) vs. hallucinated (bottom)")

    st.markdown("---")
    st.subheader("The 'Two Heads Lock In' Finding")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        The ablation heatmap reveals a concentration of causal influence at **L11 H6** and **L11 H7**.
        These two heads are physically adjacent (same layer, sequential head indices) and appear
        to share a function: they are the last attention heads before the final unembedding, and
        their DLA contribution at L11 shows a clean separation between correct and hallucinated samples.

        **Mechanistic interpretation:**

        The GPT-2 transformer processes information through three broad stages in this analysis:
        - **L1–L3:** Context integration (reading and routing the question)
        - **L4–L9:** Fact retrieval (FFN-driven, peaking at L8–L9)
        - **L10–L12:** Output commitment (finalizing the answer, dominated by L11 H6/H7)

        L11 H6/H7 appear to be the "decision point" where the model commits to a specific output token.
        The hallucination signal here may reflect a failed verification step — the model should
        be checking whether the retrieved fact is consistent with the question context, but in
        hallucinated cases this check fails or is bypassed.
        """)
    with col2:
        finding(
            "<b>The attention pattern signature:</b><br><br>"
            "In hallucinated samples, attention at L8 shows a more diffuse pattern — the model is "
            "distributing attention across multiple context positions rather than focusing on the "
            "most relevant tokens. Correct samples show more concentrated attention on the "
            "question tokens that constrain the answer.<br><br>"
            "This diffuse-vs-concentrated attention difference at L8 is mechanistically consistent "
            "with the +80% relative DLA finding: the heads are behaving differently enough that "
            "a normalized measure captures it, even though the absolute logit contribution is modest."
        )
        warn(
            "<b>Caution on head ablation interpretation:</b> Attention heads in transformers are "
            "not cleanly separable — they interact through the residual stream. Ablating H6 affects "
            "what H7 sees (because they share a layer's residual stream). The 'two heads' finding "
            "should be read as 'this layer-region is critical' rather than 'these two specific "
            "computation units are independently necessary.'"
        )

    limit(
        "Head ablation was performed with mean-ablation (replacing with the mean activation "
        "over the dataset) rather than zero-ablation — this is the more conservative and "
        "interpretable choice. Results may differ with zero-ablation or activation-patching approaches. "
        "GPT-2 (117M) uses 12 attention heads per layer across 12 layers. "
        "With 534 samples, ablation effect sizes have wide confidence intervals — "
        "the L11 H6/H7 finding is the most robust but should be treated as suggestive, not definitive."
    )


def page_steering():
    st.title("Steering, Connections & Research Arc — Steps 10–12")
    exp_pill("Step 10 — Activation Steering")
    exp_pill("Step 11 — Steering Layer Sweep")
    exp_pill("Step 12 — Three-Stage Synthesis")
    arc_pill("PhD Arc: Project 1 of 5")
    phase("Stage 3 — Steering Experiment + Full Arc Synthesis")

    st.markdown("""
    The final three steps test whether mechanistic understanding can be *used* — specifically,
    whether we can steer GPT-2 toward correct answers by injecting the truth direction into its
    activations. Then this page synthesizes how MECH-INT feeds into the four subsequent projects
    in the PhD hallucination governance arc.
    """)

    # ── Steering results ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Steps 10–11: Activation Steering Experiment")

    c1, c2, c3, c4 = st.columns(4)
    with c1: stat("α = 40", "Optimal Steering Coefficient")
    with c2: stat("0.494", "P(correct) at α=40", "stat-number-amber")
    with c3: stat("0.50", "Random Baseline P(correct)", "stat-number-red")
    with c4: stat("NULL", "Steering Verdict", "stat-number-red")

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("steering_curve.png",
                 "Steering curve — P(correct) vs. steering coefficient α (injection at L8)")
    with col2:
        st.subheader("The Steering Experiment")
        st.markdown("""
        **Method:** Compute the difference-of-means vector between correct and hallucinated
        hidden states at the optimal layer (L8). Normalize to unit length. This is the
        "truth direction" vector v*.

        During generation, inject v* into the residual stream at every forward pass:
        h_steered = h_original + α · v*

        where α scales the injection magnitude.

        **Hypothesis:** If hallucination is caused by the model's representation being too
        far from the correct answer region of activation space, pushing it in that direction
        should increase the probability of correct outputs.

        **Result:** At α=40 (very large injection), P(correct) = 0.494 — essentially random
        (0.50). No value of α substantially improves over the unsteered baseline. Larger α values
        degrade output quality (incoherent text) without helping hallucination rates.
        """)
        null(
            "<b>Steering null result:</b> Activation steering at any α does not improve "
            "P(correct) above chance for GPT-2 on TruthfulQA. The model's wrong beliefs are "
            "not easily overwritten by patching the activation stream — even with the correct "
            "direction and a large coefficient. This mirrors HaRP (Project 2) where universal "
            "steering also fails at 3B scale."
        )

    st.markdown("---")
    col1, col2 = st.columns([3, 2])
    with col1:
        plot_img("steering_layer_sweep.png",
                 "Steering layer sweep — P(correct) across all layers × α values tested")
    with col2:
        st.subheader("Why Steering Fails at GPT-2 Scale")
        st.markdown("""
        The steering null result is not surprising in retrospect — and it is scientifically
        valuable precisely because of what it reveals:

        **1. The signal is distributed, not localized.**
        Probing found modest AUROC (0.604) rather than near-perfect separation. This means
        the hallucination signal is diffuse in activation space — there is no clean correct
        answer region to steer toward.

        **2. FFN memorization is robust to activation patching.**
        If the error is in the FFN's learned associations (the FFN over-retrieval hypothesis),
        injecting a direction into the residual stream may not override the FFN's output.
        The FFN computes a function of its input — patching the stream slightly changes the
        input but not the FFN's parametric memory.

        **3. L8–L9 may be too late.**
        By the time the error is maximally encoded (L8–L9), the attention routing that
        activated the wrong FFN memory has already happened in earlier layers. Correcting
        at L8–L9 may be treating the symptom, not the cause.
        """)
        finding(
            "<b>Why this null result is useful for the arc:</b><br><br>"
            "The steering failure at GPT-2 scale is one of the explicit motivations for "
            "GEOM-PROOF (Project 3): if we understood the Fisher information geometry of "
            "how hallucination is encoded, we could derive <em>bounds</em> on whether any "
            "steering intervention can succeed — rather than empirically testing and failing."
        )

    # ── Three-stage synthesis ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 12: The Three-Stage Hallucination Model")
    st.markdown("""
    Integrating all 12 steps, MECH-INT proposes a three-stage causal model of GPT-2 hallucination:
    """)

    stages = [
        ("#1565c0", "Stage 1: Context Routing",
         "L1–L3  ·  Attention Dominant  ·  Peak at L3 Attn AUROC 0.617",
         "The model reads the question and routes information to relevant context positions. "
         "In hallucinated samples, this routing is subtly wrong from the start — the model "
         "attends to slightly different parts of the question, priming the wrong retrieval path. "
         "L3 Attention shows the highest component AUROC at this early stage (0.617).",
         ["L3 Attention AUROC: 0.617", "Attention-dominant stage", "Context integration phase"]),
        ("#00695c", "Stage 2: FFN Over-Retrieval",
         "L4–L9  ·  FFN Dominant  ·  Peak at L8 FFN AUROC 0.605",
         "The feed-forward networks at L4–L9 act as key-value memories, retrieving stored factual "
         "associations for the current query. In hallucinated samples, the FFN at L8–L9 retrieves "
         "a plausible-but-wrong association — a closely related fact that fits the surface form of "
         "the question but is factually incorrect. This is the over-retrieval phenomenon: the FFN "
         "is too confident in its stored associations.",
         ["L8 FFN AUROC: 0.605", "FFN-dominant (8/12 stages)", "L8 last-token probe: 0.604",
          "Logit lens commits error here"]),
        ("#4527a0", "Stage 3: Output Commitment",
         "L10–L12  ·  L11 H6/H7  ·  Final answer lock-in",
         "The penultimate layer's attention heads (L11 H6/H7) finalize the answer selection. "
         "In correct samples, these heads perform a final consistency check between the retrieved "
         "fact and the question context. In hallucinated samples, this check is bypassed or "
         "fails — the wrong answer propagates through to the output without correction. "
         "Ablating L11 H6/H7 shows the most causal sensitivity for output commitment.",
         ["L11 H6/H7 ablation sensitivity", "Output-commitment stage", "Final verification fails"]),
    ]

    for color, title, subtitle, desc, bullets in stages:
        bg = "#e3f2fd" if color == "#1565c0" else ("#e0f2f1" if color == "#00695c" else "#ede7f6")
        b_html = "".join([f'<li style="font-size:0.80rem;">{b}</li>' for b in bullets])
        st.markdown(
            f'<div style="border:2px solid {color};background:{bg};border-radius:12px;'
            f'padding:1.1rem 1.3rem;margin-bottom:0.8rem;">'
            f'<b style="font-size:1rem;color:{color};">{title}</b><br>'
            f'<span style="font-size:0.78rem;font-weight:600;color:{color};">{subtitle}</span><br>'
            f'<p style="font-size:0.87rem;margin:0.5rem 0;">{desc}</p>'
            f'<ul style="margin:0;padding-left:1.2rem;color:#37474f;">{b_html}</ul></div>',
            unsafe_allow_html=True,
        )

    # ── Research arc connections ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("How MECH-INT Feeds into the PhD Research Arc")
    st.markdown("""
    MECH-INT is **Project 1 of 5** in a PhD research arc on LLM hallucination governance.
    Each subsequent project is directly motivated by a finding or limit from this project.
    """)

    arc_projects = [
        ("#1565c0", "Project 1: MECH-INT (this work)",
         "GPT-2 · 117M · Mechanistic Interpretability",
         "Establishes baseline: hallucination signal at L8–L9 (AUROC 0.604), FFN over-retrieval, "
         "distributed signal, steering fails. Sets up the what-happens-at-scale question.",
         "COMPLETE"),
        ("#4527a0", "Project 2: HaRP",
         "Qwen 2.5 3B · Representation Probing · Governance",
         "Scale up to 3B parameters. Does a truth geometry emerge? Answer: yes (AUROC 0.776). "
         "The +0.172 AUROC improvement over MECH-INT directly validates the scale-dependence hypothesis. "
         "Adds 3-action governance policy (ACCEPT/REGENERATE/ABSTAIN). "
         "89% depth crystallization finding: the hallucination signal concentrates at relative depth 0.89 (L32/36).",
         "COMPLETE"),
        ("#2e7d32", "Project 3: GEOM-PROOF",
         "Fisher Information Geometry · Theoretical Bounds",
         "MECH-INT's steering failure raises the question: is there a theoretical bound on whether "
         "any steering intervention can succeed? GEOM-PROOF derives Fisher information bounds on "
         "the minimum representation shift needed to change a model's output — making the null result "
         "theoretically grounded rather than empirically observed.",
         "PLANNED"),
        ("#bf360c", "Project 4: FAIL-CHAIN",
         "Cascade Failure Modeling · Multi-Agent Systems",
         "MECH-INT's three-stage model (routing → retrieval → commitment) can be formalized as a "
         "cascade: if Stage 1 fails, Stage 2 is more likely to fail. FAIL-CHAIN models how "
         "hallucination propagates through multi-agent LLM pipelines where outputs of one model "
         "become inputs to the next.",
         "PLANNED"),
        ("#37474f", "Project 5: GUARDIAN",
         "Full Governance System · Production Routing",
         "Integrates findings from all 4 prior projects into a production-ready governance router. "
         "Uses HaRP's probe confidence for failure estimation, GEOM-PROOF bounds for safety "
         "guarantees, FAIL-CHAIN cascade risk for pipeline routing — and the MECH-INT three-stage "
         "model to tailor interventions to the failure stage detected.",
         "COMPLETE"),
    ]

    for color, title, subtitle, desc, status in arc_projects:
        bg = "#f8f9ff" if status == "COMPLETE" else "#fafafa"
        sc = "#2e7d32" if status == "COMPLETE" else "#9e9e9e"
        st.markdown(
            f'<div style="border-left:5px solid {color};background:{bg};'
            f'border-radius:0 12px 12px 0;padding:1rem 1.2rem;margin-bottom:0.6rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<b style="font-size:0.95rem;color:{color};">{title}</b>'
            f'<span style="font-size:0.70rem;font-weight:700;color:{sc};background:{sc}22;'
            f'padding:0.12rem 0.55rem;border-radius:12px;">{status}</span></div>'
            f'<span style="font-size:0.78rem;font-weight:600;color:#546e7a;">{subtitle}</span><br>'
            f'<p style="font-size:0.85rem;color:#37474f;margin:0.35rem 0 0;">{desc}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("The Three-Layer Hallucination Governance Model")
    st.markdown("""
    The full arc produces a three-layer conceptual model of how to govern LLM hallucination:
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#e3f2fd;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:1.3rem;font-weight:800;color:#1565c0;">Layer 1</div>
        <div style="font-weight:700;color:#1565c0;margin:0.4rem 0;">Mechanistic Understanding</div>
        <div style="font-size:0.82rem;color:#37474f;text-align:left;">
        MECH-INT maps <em>where and how</em> hallucination is manufactured inside the model.
        Three stages: routing, retrieval, commitment.
        Without this, governance is black-box.
        </div></div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:1.3rem;font-weight:800;color:#2e7d32;">Layer 2</div>
        <div style="font-weight:700;color:#2e7d32;margin:0.4rem 0;">Detection & Estimation</div>
        <div style="font-size:0.82rem;color:#37474f;text-align:left;">
        HaRP uses the mechanistic insights (probe at peak layer, geometry signals) to build
        a calibrated failure estimator P(failure | signals).
        89% depth crystallization: the probe at relative depth 0.89 is optimal.
        </div></div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fff8e1;border-radius:10px;padding:1rem;text-align:center;">
        <div style="font-size:1.3rem;font-weight:800;color:#bf360c;">Layer 3</div>
        <div style="font-weight:700;color:#bf360c;margin:0.4rem 0;">Governance & Routing</div>
        <div style="font-size:0.82rem;color:#37474f;text-align:left;">
        GUARDIAN integrates detection into a production routing policy.
        GEOM-PROOF provides theoretical safety bounds.
        FAIL-CHAIN handles multi-agent cascade risk.
        </div></div>
        """, unsafe_allow_html=True)

    finding(
        "<b>The research argument:</b> Trustworthy AI governance requires understanding the "
        "mechanism, not just the output. MECH-INT provides the mechanistic baseline — "
        "where the error is, how it propagates, why it resists correction. "
        "The subsequent four projects translate mechanistic understanding into progressively more "
        "deployable governance tools, each inheriting from and extending the findings here."
    )

    limit(
        "All MECH-INT findings are specific to GPT-2 (117M parameters) on TruthfulQA (534 samples) "
        "running on MacBook Air Apple Silicon (MPS). The three-stage model is a mechanistic "
        "hypothesis grounded in the data, not a proven causal decomposition. "
        "Causal claims require intervention experiments — the ablation and steering results provide "
        "partial causal evidence, but a full causal graph would require more systematic circuit analysis."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR + ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "Research Brief":           page_brief,
    "The 12-Stage Pipeline":    page_pipeline,
    "Surface Predictor":        page_surface,
    "Layer Probing":            page_probing,
    "Logit Lens":               page_logit_lens,
    "Direct Logit Attribution": page_dla,
    "Attention Head Analysis":  page_attention,
    "Steering & Connections":   page_steering,
}


def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.8rem 0 0.5rem;">
        <div style="font-size:2rem;">🔬</div>
        <div style="font-size:1.0rem;font-weight:800;letter-spacing:0.04em;color:#e3f2fd;">
        MECH-INT
        </div>
        <div style="font-size:0.72rem;color:#90caf9;margin-top:0.15rem;line-height:1.4;">
        Mechanistic Interpretability<br>of GPT-2 Hallucination
        </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        if "page" not in st.session_state:
            st.session_state.page = "Research Brief"

        st.markdown(
            '<div style="font-size:0.70rem;font-weight:700;letter-spacing:0.10em;'
            'color:#90caf9;margin-bottom:0.4rem;">NAVIGATION</div>',
            unsafe_allow_html=True,
        )

        page_icons = {
            "Research Brief":           "📋",
            "The 12-Stage Pipeline":    "🗺",
            "Surface Predictor":        "📊",
            "Layer Probing":            "🔍",
            "Logit Lens":               "🔭",
            "Direct Logit Attribution": "⚖",
            "Attention Head Analysis":  "🧠",
            "Steering & Connections":   "🔗",
        }

        for page_name in PAGES:
            icon = page_icons.get(page_name, "·")
            label = f"{icon}  {page_name}"
            if st.button(label, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.page = page_name

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.70rem;font-weight:700;letter-spacing:0.10em;'
            'color:#90caf9;margin-bottom:0.5rem;">MODEL & DATA</div>',
            unsafe_allow_html=True,
        )
        stats_items = [
            ("Model",    "GPT-2 (117M)"),
            ("Layers",   "12 transformer + embed"),
            ("Dim",      "768 hidden"),
            ("Heads",    "12 per layer"),
            ("Dataset",  "TruthfulQA"),
            ("Samples",  "534 total"),
            ("Correct",  "266 (49.8%)"),
            ("Hall.",    "268 (50.2%)"),
            ("Label",    "Jaccard ≥ 0.12"),
            ("Hardware", "Apple Silicon MPS"),
        ]
        for k, v in stats_items:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.76rem;margin-bottom:0.18rem;">'
                f'<span style="color:#90caf9;font-weight:600;">{k}</span>'
                f'<span style="color:#e3f2fd;">{v}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.70rem;font-weight:700;letter-spacing:0.10em;'
            'color:#90caf9;margin-bottom:0.5rem;">KEY RESULTS</div>',
            unsafe_allow_html=True,
        )
        key_results = [
            ("Random baseline", "0.447"),
            ("Surface MLP",     "0.576"),
            ("L9 mean-pool",    "0.583"),
            ("L8 last-token",   "0.604 ★"),
            ("FFN peak (L8)",   "0.605"),
            ("Attn L8 rel.",    "+80%"),
            ("Steering α=40",   "0.494"),
        ]
        for k, v in key_results:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.76rem;margin-bottom:0.18rem;">'
                f'<span style="color:#90caf9;">{k}</span>'
                f'<span style="color:#e3f2fd;font-weight:600;">{v}</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.68rem;color:#64b5f6;text-align:center;line-height:1.5;">'
            'PhD Research Arc · Project 1 of 5<br>'
            'Lakshmi Chakradhar Vijayarao<br>'
            '<span style="color:#42a5f5;">LLM Hallucination Governance</span>'
            '</div>',
            unsafe_allow_html=True,
        )


def main():
    sidebar()
    current = st.session_state.get("page", "Research Brief")
    fn = PAGES.get(current, page_brief)
    fn()


if __name__ == "__main__":
    main()
