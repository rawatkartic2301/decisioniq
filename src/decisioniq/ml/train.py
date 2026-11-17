"""Train ML models, generate comparison reports, and write docs/MODEL_CARD.md."""

import logging
from datetime import datetime
from pathlib import Path

from decisioniq.ml.anomaly_detector import TimesheetAnomalyDetector
from decisioniq.ml.intent_classifier import IntentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_model_card(
    intent_summary: dict,
    anomaly_summary: dict,
    output_path: Path = Path("docs/MODEL_CARD.md"),
):
    """Write comprehensive model cards for both ML models with real empirical metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    comp = intent_summary["comparison"]

    content = f"""# DecisionIQ: Machine Learning Model Cards

**Generated**: {timestamp}
**Framework**: `scikit-learn`
**Artifacts**: `src/decisioniq/ml/saved_models/*.joblib`

---

## 1. Intent / Route Classifier

### Model Overview
The Intent Classifier routes inbound enterprise business queries to specialized graph nodes (Policy RAG, SQL Analytics, Hybrid Doc+SQL, Anomaly Detection, Unanswerable, Adversarial). Fast routing (<2 ms) saves LLM token costs and reduces query latency.

- **Primary Architecture**: `TfidfVectorizer(ngram_range=(1, 2), max_features=1500)` + `LogisticRegression(C=1.0)`
- **Evaluation Strategy**: 5-Fold Stratified Cross-Validation
- **Fallback**: Confidence threshold < 0.65 triggers structured LLM intent fallback.

### Cross-Validation Comparison Table

| Model Architecture | 5-Fold Macro F1 (Mean ± Std) | Macro Precision | Macro Recall |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** (Production) | **{comp['LogisticRegression']['f1_macro_mean']:.3f} ± {comp['LogisticRegression']['f1_macro_std']:.3f}** | {comp['LogisticRegression']['precision_mean']:.3f} | {comp['LogisticRegression']['recall_mean']:.3f} |
| **LinearSVC** | **{comp['LinearSVC']['f1_macro_mean']:.3f} ± {comp['LinearSVC']['f1_macro_std']:.3f}** | {comp['LinearSVC']['precision_mean']:.3f} | {comp['LinearSVC']['recall_mean']:.3f} |
| **Random Forest** | **{comp['RandomForest']['f1_macro_mean']:.3f} ± {comp['RandomForest']['f1_macro_std']:.3f}** | {comp['RandomForest']['precision_mean']:.3f} | {comp['RandomForest']['recall_mean']:.3f} |

### Production Model Metrics
- **Overall Macro F1**: **{intent_summary['f1_macro']:.3f}**
- **ROC-AUC (One-vs-Rest, Macro)**: **{intent_summary['roc_auc_ovr']:.3f}**
- **Classes**: `{', '.join(intent_summary['classes'])}`

---

## 2. Timesheet Anomaly Detector (Unsupervised)

### Model Overview
Identifies high-risk, fraudulent, or non-compliant timesheets prior to monthly payroll execution.

- **Algorithm**: `IsolationForest` (unsupervised)
- **Contamination Parameter**: `{anomaly_summary['contamination_parameter']}`
- **Feature Engineering Pipeline**:
  1. `daily_total_hours`: `regular_hours + overtime_hours`
  2. `overtime_ratio`: `overtime_hours / daily_total_hours`
  3. `is_weekend`: Binary flag for Saturday/Sunday
  4. `same_day_count`: Number of submissions by same employee on same date (detecting duplicates)
  5. `dept_zscore`: Standardized z-score of hours relative to department mean and standard deviation

### Empirical Performance Against Injected Ground Truth
Evaluated on **{anomaly_summary['total_records']}** historical timesheet rows containing **{anomaly_summary['injected_anomalies']}** known injected anomalies (impossible hours >16h, duplicate same-day logs, weekend spikes).

| Metric | Measured Value | Operational Interpretation |
| :--- | :--- | :--- |
| **Precision** | **{anomaly_summary['precision']:.3f}** ({anomaly_summary['precision']*100:.1f}%) | Low false alarm rate; flagged items warrant human audit |
| **Recall** | **{anomaly_summary['recall']:.3f}** ({anomaly_summary['recall']*100:.1f}%) | Captures known compliance and fraud violations |
| **F1-Score** | **{anomaly_summary['f1_score']:.3f}** | Balanced operational metric for unsupervised detection |

---

## 3. Ethical Considerations & Limitations
1. **Model Drift**: If company policy changes standard working hours, the anomaly baseline z-scores must be re-calibrated.
2. **Fairness**: Department-normalized z-scores prevent engineering teams with legitimate on-call rotations from being penalized compared to administrative roles.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Successfully generated model card at %s", output_path)


def main():
    logger.info("Training and evaluating IntentClassifier...")
    intent_clf = IntentClassifier()
    intent_summary = intent_clf.train_and_evaluate()

    logger.info("Training and evaluating TimesheetAnomalyDetector...")
    anomaly_detector = TimesheetAnomalyDetector()
    anomaly_summary = anomaly_detector.train_and_evaluate()

    generate_model_card(intent_summary, anomaly_summary)
    print("\nML Training & Model Card generation complete!")


if __name__ == "__main__":
    main()
