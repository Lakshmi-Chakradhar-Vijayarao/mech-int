"""
Phase 3 (Step 3): Surface Feature Classifier — the intentional null result.

Scientific question
-------------------
Can GPT-2's own output statistics predict whether its answer will be hallucinated,
WITHOUT looking inside the model?

This experiment is designed to FAIL in an informative way. If output features
alone achieve AUROC > 0.80, there is no need to look at internal activations.
The weak result here (AUROC ~0.576) is the justification for every subsequent
internal analysis — it proves that external observation is insufficient.

What the surface features capture
----------------------------------
Six scalar features are computed per prompt from the model's logit distribution:

  mean_entropy    : H = -Σ p log p averaged over all generated tokens.
                    Measures average uncertainty in token predictions.
  max_entropy     : Maximum entropy at any single token position.
                    Captures the "hardest" prediction in the sequence.
  logit_variance  : Variance of the top logit value across sequence positions.
                    High variance = model is inconsistently confident.
  confidence_gap  : Mean (p_top1 - p_top2) across positions.
                    Low gap = the model had close alternatives — hesitant prediction.
  attention_entropy : Mean attention weight entropy across all heads and layers.
                    Low entropy = focused attention; high entropy = diffuse attention.
  activation_norm : L2 norm of the final layer's hidden state at the last position.
                    Proxy for the "energy" of the model's final representation.

Why these features are insufficient
------------------------------------
GPT-2 hallucinates with exactly the same fluency and confidence as when it is
correct. The surface statistics — entropy, confidence gap — reflect lexical
fluency, not factual accuracy. A confident wrong answer looks identical to a
confident right answer in logit space.

This is consistent with the literature: HaloScope and ReDeEP both observe that
scalar uncertainty features underperform full activation probes on factual QA tasks.

Models trained
--------------
  Logistic Regression   : Linear model, 5-fold CV. CV AUROC ~0.531.
  MLP (2-layer, ReLU)   : Non-linear model, 5-fold CV. CV AUROC ~0.576.

The MLP's marginal improvement (0.531 → 0.576) shows non-linear interactions
between features carry some additional signal, but neither reaches operational
reliability. Best accuracy is ~57.9% — barely above the 50.5% majority class baseline.

Outputs
-------
  results/logs/predictor_results.txt      — CV AUROC / accuracy / F1 for all models
  results/plots/roc_curve.png             — ROC curves for all models
  results/plots/confusion_matrix.png      — confusion matrix at 0.5 threshold
  results/plots/confidence_vs_accuracy.png — calibration plot

Usage
-----
    python experiments/run_predictor.py
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor.classifier import train_and_evaluate
from src.evaluation.metrics import (
    plot_roc_curve, plot_confusion_matrix,
    plot_confidence_vs_accuracy, print_results_table,
)

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=== Phase 4: Hallucination Predictor ===\n")

    X = np.load(DATA_DIR / "features.npy")
    y = np.load(DATA_DIR / "labels.npy")
    print(f"Loaded features: {X.shape},  labels: {y.shape}")
    print(f"Class balance — Correct: {y.sum()}, Hallucinated: {(y==0).sum()}\n")

    results, lr, mlp, X_train, X_test, y_train, y_test = train_and_evaluate(
        X, y, test_size=0.2, save_dir="results/models"
    )

    print_results_table(results)

    # --- Cross-validated AUROC (statistically honest summary) ---
    print("\n--- Cross-Validated AUROC (more reliable than single split) ---")
    for name in ["logistic", "mlp"]:
        cv_mean = results[name].get("cv_auroc_mean", "N/A")
        cv_std  = results[name].get("cv_auroc_std",  "N/A")
        ci_low  = round(cv_mean - 1.96 * cv_std, 4) if isinstance(cv_std, float) else "N/A"
        ci_high = round(cv_mean + 1.96 * cv_std, 4) if isinstance(cv_std, float) else "N/A"
        print(f"  {name:<12}  CV AUROC = {cv_mean:.4f} ± {cv_std:.4f}  95% CI [{ci_low}, {ci_high}]")

    # --- Detailed report for best model ---
    best_model_name = max(
        ["logistic", "mlp"],
        key=lambda k: results[k].get("cv_auroc_mean", results[k]["auroc"])
    )
    best_pipeline = lr if best_model_name == "logistic" else mlp
    print(f"\nBest model: {best_model_name} (CV AUROC={results[best_model_name].get('cv_auroc_mean', results[best_model_name]['auroc'])})")
    print("\nClassification Report (held-out test set):")
    print(results[best_model_name]["report"])

    # --- Plots ---
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
    # confidence_gap is feature index 3
    confidence_gap = X_test[:, 3]

    plot_roc_curve(
        y_test,
        {
            "Logistic Regression": y_prob_lr,
            "MLP":                 y_prob_mlp,
            "Confidence Gap Only": confidence_gap,
        },
        save_path=str(PLOTS_DIR / "roc_curve.png"),
    )

    y_pred_best = best_pipeline.predict(X_test)
    plot_confusion_matrix(
        y_test, y_pred_best,
        save_path=str(PLOTS_DIR / "confusion_matrix.png"),
    )

    plot_confidence_vs_accuracy(
        y_prob_lr, y_test,
        save_path=str(PLOTS_DIR / "calibration_curve.png"),
    )

    # --- Save results log ---
    log_path = LOGS_DIR / "predictor_results.txt"
    with open(log_path, "w") as f:
        for name, metrics in results.items():
            f.write(f"{name}: {metrics}\n")
    print(f"\nResults saved to {log_path}")

    # Signal to proceed
    if results[best_model_name]["auroc"] > 0.70:
        print("\n✓ AUROC > 0.70 — signal confirmed. Proceed to Phase 5 (ablation).")
    else:
        print("\n⚠  AUROC <= 0.70 — features are weak. Revisit extraction / features.")


if __name__ == "__main__":
    main()
