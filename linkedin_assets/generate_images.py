"""
Generate 4 professional LinkedIn images for MECH-INT project.
All images: 1200 x 627 px (LinkedIn recommended landscape ratio).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent

# ── Shared palette ─────────────────────────────────────────────────────────────
NAVY      = "#0d1b4b"
BLUE      = "#1565c0"
BLUE_LT   = "#1976d2"
BLUE_BG   = "#e3f2fd"
GREEN     = "#2e7d32"
AMBER     = "#e65100"
RED       = "#b71c1c"
PURPLE    = "#6a1b9a"
WHITE     = "#ffffff"
OFFWHITE  = "#f4f6fb"
GREY      = "#607d8b"
LIGHT     = "#cfd8dc"

W, H = 12.0, 6.27   # inches at 100 dpi → 1200 × 627 px

def save(fig, name):
    fig.savefig(OUT / name, dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 1 — TITLE CARD
# ══════════════════════════════════════════════════════════════════════════════
def img_title():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # ── background accent shapes ──────────────────────────────────────────────
    for rx, ry, rr, alpha in [(11.5, 5.8, 3.5, 0.07), (0.5, 0.5, 2.5, 0.05),
                               (10, 1.2, 2.0, 0.06)]:
        circle = plt.Circle((rx, ry), rr, color=BLUE_LT, alpha=alpha)
        ax.add_patch(circle)

    # Horizontal accent line
    ax.plot([0.6, 11.4], [1.1, 1.1], color=BLUE_LT, lw=1.2, alpha=0.4)
    ax.plot([0.6, 4.2],  [1.1, 1.1], color=BLUE_LT, lw=2.5, alpha=0.9)

    # ── main text ─────────────────────────────────────────────────────────────
    ax.text(0.6, 5.55, "MECH-INT",
            fontsize=54, fontweight="black", color=WHITE,
            va="top", ha="left", family="monospace",
            path_effects=[pe.withStroke(linewidth=0, foreground=NAVY)])

    ax.text(0.6, 4.55,
            "Mechanistic Interpretability of LLM Hallucinations",
            fontsize=18, fontweight="bold", color=BLUE_LT,
            va="top", ha="left")

    ax.text(0.6, 3.75,
            "GPT-2  ·  TruthfulQA  ·  534 samples  ·  12 analyses",
            fontsize=12, color="#90caf9", va="top", ha="left")

    # ── the question ──────────────────────────────────────────────────────────
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.55, 1.6), 7.5, 1.55,
        boxstyle="round,pad=0.15", fc=BLUE, ec=BLUE_LT, lw=1.5, alpha=0.25))
    ax.text(1.3, 2.38,
            'Can you predict a hallucination before\n'
            'a single output token is generated?',
            fontsize=14, color=WHITE, va="center", ha="left",
            style="italic", linespacing=1.6)
    ax.text(0.65, 2.38, '"', fontsize=48, color=BLUE_LT,
            va="center", alpha=0.6)

    # ── right side — key numbers ───────────────────────────────────────────────
    stats = [
        ("534",   "labeled samples"),
        ("12",    "analyses"),
        ("0.604", "best AUROC"),
        ("L8–L9", "mechanistic core"),
        ("7",     "methods converge"),
    ]
    xs = [8.6, 9.5, 10.4, 11.0, 11.5]  # not used; use grid
    for i, (num, lbl) in enumerate(stats[:4]):
        col = i % 2
        row = i // 2
        x = 8.9 + col * 1.7
        y = 5.0 - row * 1.55
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.75, y - 0.65), 1.45, 1.22,
            boxstyle="round,pad=0.1", fc=WHITE, ec=BLUE_LT,
            lw=1.0, alpha=0.08))
        ax.text(x, y + 0.18, num, ha="center", va="center",
                fontsize=17, fontweight="black", color=WHITE)
        ax.text(x, y - 0.28, lbl, ha="center", va="center",
                fontsize=7.5, color="#90caf9", fontweight="600")

    # 5th stat — centered on right panel
    ax.add_patch(mpatches.FancyBboxPatch(
        (9.0, 1.25), 1.45, 1.22,
        boxstyle="round,pad=0.1", fc=WHITE, ec=BLUE_LT,
        lw=1.0, alpha=0.08))
    ax.text(9.725, 2.08, "7", ha="center", va="center",
            fontsize=17, fontweight="black", color=WHITE)
    ax.text(9.725, 1.62, "methods converge", ha="center", va="center",
            fontsize=7.5, color="#90caf9", fontweight="600")

    # ── URL ───────────────────────────────────────────────────────────────────
    ax.text(0.6, 0.52, "mech-int.streamlit.app",
            fontsize=10, color=BLUE_LT, va="center", alpha=0.8,
            family="monospace")

    save(fig, "li_01_title.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 2 — THREE-STAGE CASCADE
# ══════════════════════════════════════════════════════════════════════════════
def img_cascade():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(OFFWHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Header
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 5.5), 12, 0.77, boxstyle="square", fc=NAVY, ec=NAVY))
    ax.text(6, 5.88, "The Three-Stage Hallucination Cascade in GPT-2",
            ha="center", va="center", fontsize=16, fontweight="bold",
            color=WHITE)
    ax.text(6, 5.62,
            "Seven independent methods converge on this structure",
            ha="center", va="center", fontsize=10, color="#90caf9")

    # Stage boxes
    stages = [
        (1.4,  "Stage 1",    "Layer 3 — Attention",
         "Context Routing\nFailure",
         ["Attn component AUROC 0.617",
          "Model mis-routes question\ncontext tokens",
          "Wrong tokens receive\nattention weight"],
         AMBER),
        (5.1,  "Stage 2  ★", "Layers 8–9 — FFN + Attention",
         "Parametric Recall\nFailure  (CORE)",
         ["7 independent methods converge",
          "FFN AUROC 0.605  |  Probe 0.604",
          "Attn +80% relative DLA",
          "Steering inversion at L9"],
         BLUE),
        (8.8,  "Stage 3",    "Layer 11 — H6 & H7",
         "Output\nCommitment",
         ["Head ablation: H6 +0.16, H7 +0.14",
          "5–8x importance vs all others",
          "Final answer locked in"],
         GREEN),
    ]

    for cx, stg, sub, title, bullets, color in stages:
        w, h = 3.2, 4.6
        x0, y0 = cx - w/2, 0.55

        # Shadow
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0+0.06, y0-0.06), w, h,
            boxstyle="round,pad=0.15", fc="#00000015", ec="none"))

        # Card
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.15", fc=WHITE, ec=color, lw=2.5))

        # Top colour strip
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y0 + h - 1.05), w, 1.05,
            boxstyle="round,pad=0.0", fc=color, ec=color, lw=0))
        ax.text(cx, y0 + h - 0.32, stg,
                ha="center", va="center", fontsize=12,
                fontweight="black", color=WHITE)
        ax.text(cx, y0 + h - 0.72, sub,
                ha="center", va="center", fontsize=8.5,
                color=WHITE, alpha=0.9)

        # Title
        ax.text(cx, y0 + h - 1.42, title,
                ha="center", va="top", fontsize=11,
                fontweight="bold", color=color, linespacing=1.3)

        # Bullets
        for bi, b in enumerate(bullets):
            ax.text(x0 + 0.22, y0 + h - 2.12 - bi * 0.52, f"•  {b}",
                    va="top", fontsize=8.2, color="#333",
                    linespacing=1.25)

    # Arrows between stages
    for ax_x in [2.8 + 0.0, 6.5 + 0.0]:
        ax.annotate("", xy=(ax_x + 0.42, 2.85), xytext=(ax_x, 2.85),
                    arrowprops=dict(arrowstyle="-|>", color=GREY,
                                    lw=2.5, mutation_scale=18))

    # Footer note
    ax.text(6, 0.25,
            "Stage 2 (L8–L9) is independently confirmed by: "
            "dense probe · token probe · FFN component · steering · DLA · logit lens · head DLA",
            ha="center", va="center", fontsize=8, color=GREY,
            style="italic")

    save(fig, "li_02_cascade.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 3 — KEY RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def img_results():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Header
    ax.text(6, 5.92, "MECH-INT  —  Key Results",
            ha="center", va="center", fontsize=20,
            fontweight="black", color=WHITE)
    ax.plot([1.5, 10.5], [5.6, 5.6], color=BLUE_LT, lw=1.5, alpha=0.5)

    results = [
        # (x, y, number, label, sublabel, color)
        (1.5,  4.55, "0.576", "Surface AUROC",    "output stats only\nnear chance", AMBER),
        (3.8,  4.55, "0.604", "Best Probe AUROC",  "L8 × last token\n+0.028 gain", GREEN),
        (6.1,  4.55, "87%",   "Signal Sparsity",   "100 / 768 dims\nconcentrated", BLUE_LT),
        (8.4,  4.55, "8/12",  "FFN Dominant",      "parametric recall\nfailure", PURPLE),
        (10.6, 4.55, "~0.50", "SVD Subspace",      "no geometry\nat GPT-2 scale", RED),
        (1.5,  2.15, "+80%",  "L8 Attn Rel DLA",   "strongest signal\nin project", GREEN),
        (3.8,  2.15, "+200%", "H5 Head Rel DLA",   "near-chance pattern\nexact contribution", BLUE_LT),
        (6.1,  2.15, "L9",    "Steering Peak",      "causal inversion\nat α = 40", AMBER),
        (8.4,  2.15, "H6/H7", "L11 Ablation",      "5–8x importance\nvs all others", PURPLE),
        (10.6, 2.15, "+0.53", "12-Head DLA Sum",   "exact decomposition\nvalidated", GREEN),
    ]

    for x, y, num, lbl, sub, col in results:
        # Card
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.0, y - 0.9), 2.0, 1.78,
            boxstyle="round,pad=0.12", fc=col, ec=WHITE,
            lw=0.8, alpha=0.18))
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 1.0, y - 0.9), 2.0, 1.78,
            boxstyle="round,pad=0.12", fc="none", ec=col,
            lw=1.5, alpha=0.6))

        ax.text(x, y + 0.44, num,
                ha="center", va="center", fontsize=22,
                fontweight="black", color=col)
        ax.text(x, y - 0.05, lbl,
                ha="center", va="center", fontsize=8,
                fontweight="bold", color=WHITE)
        ax.text(x, y - 0.55, sub,
                ha="center", va="center", fontsize=7,
                color="#90caf9", linespacing=1.3)

    # Row labels
    ax.text(0.22, 4.55, "Detection\n& Probing",
            ha="center", va="center", fontsize=8,
            color="#90caf9", fontweight="bold", rotation=90)
    ax.text(0.22, 2.15, "Attribution\n& Causation",
            ha="center", va="center", fontsize=8,
            color="#90caf9", fontweight="bold", rotation=90)

    ax.plot([0.48, 0.48], [0.75, 5.5], color=BLUE_LT, lw=0.8, alpha=0.3)

    ax.text(6, 0.38, "mech-int.streamlit.app  ·  Full interactive walkthrough with every analysis explained",
            ha="center", va="center", fontsize=9,
            color="#7986cb", style="italic")

    save(fig, "li_03_results.png")


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE 4 — 12-STEP PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def img_pipeline():
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(OFFWHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.27); ax.axis("off")

    # Header band
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 5.5), 12, 0.77, boxstyle="square", fc=NAVY, ec=NAVY))
    ax.text(6, 5.88, "12-Step Mechanistic Interpretability Pipeline",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=WHITE)
    ax.text(6, 5.62,
            "Each step answers one specific question — together they form a complete mechanistic explanation",
            ha="center", va="center", fontsize=9, color="#90caf9")

    phases = [
        ("PHASE I — DETECTION", BLUE,
         [("Step 1", "Data\nPrep",    "#b3e5fc"),
          ("Step 2", "Activation\nExtraction", "#b3e5fc"),
          ("Step 3", "Surface\nPredictor", BLUE_LT),
          ("Step 4A","Layer\nProbe",   BLUE),
          ("Step 4B","Sparse\nProbe",  BLUE),
          ("Step 4C","Token\nPosition",NAVY)]),
        ("PHASE II — LOCALISATION", GREEN,
         [("Step 5", "Head\nAblation", "#a5d6a7"),
          ("Step 6", "SVD\nSubspace",  "#66bb6a"),
          ("Step 7", "Component\nProbe", GREEN),
          ("Step 8", "Activation\nSteering", "#1b5e20")]),
        ("PHASE III — ATTRIBUTION", PURPLE,
         [("Step 9",  "Logit\nLens",   "#ce93d8"),
          ("Step 10", "DLA",           "#ab47bc"),
          ("Step 11", "Attn\nPatterns","#8e24aa"),
          ("Step 12", "Head\nDLA",     PURPLE)]),
    ]

    phase_xs   = [0.25, 4.45, 8.35]
    phase_widths = [4.1, 3.8, 3.55]

    for pi, (ph_name, ph_col, steps) in enumerate(phases):
        px = phase_xs[pi]
        pw = phase_widths[pi]

        # Phase label band
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, 4.62), pw, 0.65,
            boxstyle="round,pad=0.08", fc=ph_col, ec=ph_col, lw=0,
            alpha=0.15))
        ax.add_patch(mpatches.FancyBboxPatch(
            (px, 4.62), pw, 0.65,
            boxstyle="round,pad=0.08", fc="none", ec=ph_col, lw=1.5))
        ax.text(px + pw/2, 4.95, ph_name,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color=ph_col)

        n = len(steps)
        step_w = (pw - 0.15) / n
        for si, (step_num, step_lbl, step_col) in enumerate(steps):
            sx = px + si * step_w + 0.05
            sy = 0.55

            # Step box
            ax.add_patch(mpatches.FancyBboxPatch(
                (sx, sy), step_w - 0.12, 3.85,
                boxstyle="round,pad=0.1", fc=step_col, ec=WHITE,
                lw=1.2, alpha=0.9))

            # Step number top chip
            ax.add_patch(mpatches.FancyBboxPatch(
                (sx + 0.04, sy + 3.35), step_w - 0.20, 0.42,
                boxstyle="round,pad=0.05", fc=WHITE, ec="none",
                lw=0, alpha=0.25))
            ax.text(sx + (step_w - 0.12)/2, sy + 3.57,
                    step_num, ha="center", va="center",
                    fontsize=7.5, fontweight="bold",
                    color=WHITE if step_col not in ["#b3e5fc","#a5d6a7","#ce93d8"] else "#111")

            # Label
            ax.text(sx + (step_w - 0.12)/2, sy + 1.85,
                    step_lbl, ha="center", va="center",
                    fontsize=8, fontweight="bold", linespacing=1.35,
                    color=WHITE if step_col not in ["#b3e5fc","#a5d6a7","#ce93d8"] else "#111")

            # Arrow to next step
            if si < n - 1:
                ax.annotate("", xy=(sx + step_w - 0.06, sy + 1.85),
                            xytext=(sx + step_w - 0.18, sy + 1.85),
                            arrowprops=dict(arrowstyle="-|>",
                                            color=WHITE, lw=1.2,
                                            mutation_scale=8))

        # Arrow to next phase
        if pi < len(phases) - 1:
            mid_y = 3.0
            x_from = px + pw + 0.02
            x_to   = phase_xs[pi+1] - 0.02
            ax.annotate("", xy=(x_to, mid_y), xytext=(x_from, mid_y),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=GREY, lw=2.0,
                                        mutation_scale=14))

    # Question labels at bottom of boxes
    q_labels = {
        "Step 3":  "Does signal\nexist?",
        "Step 4A": "Which\nlayer?",
        "Step 4B": "Which\nneurons?",
        "Step 4C": "Which\nposition?",
        "Step 5":  "Which\nheads?",
        "Step 7":  "FFN or\nAttn?",
        "Step 8":  "Causally\nactive?",
        "Step 10": "How many\nlogit units?",
        "Step 12": "Which\nhead?",
    }

    # Footer
    ax.text(6, 0.24,
            "Full code, results, and educational walkthrough:  mech-int.streamlit.app",
            ha="center", va="center", fontsize=8.5,
            color=GREY, style="italic")

    save(fig, "li_04_pipeline.png")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating LinkedIn images...")
    img_title()
    img_cascade()
    img_results()
    img_pipeline()
    print("Done. All images saved to linkedin_assets/")
