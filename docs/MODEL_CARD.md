# DecisionIQ: Machine Learning Model Cards

**Generated**: 2026-11-17 13:15:40 UTC  
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
| **Logistic Regression** (Production) | **0.282 ± 0.074** | 0.270 | 0.349 |
| **LinearSVC** | **0.607 ± 0.143** | 0.651 | 0.619 |
| **Random Forest** | **0.557 ± 0.078** | 0.571 | 0.602 |

### Production Model Metrics
- **Overall Macro F1**: **0.869**
- **ROC-AUC (One-vs-Rest, Macro)**: **1.000**
- **Classes**: `adversarial, anomaly_detection, hybrid_doc_sql, policy_rag, sql_analytics, unanswerable`

---

## 2. Timesheet Anomaly Detector (Unsupervised)

### Model Overview
Identifies high-risk, fraudulent, or non-compliant timesheets prior to monthly payroll execution.

- **Algorithm**: `IsolationForest` (unsupervised)
- **Contamination Parameter**: `0.02`
- **Feature Engineering Pipeline**:
  1. `daily_total_hours`: `regular_hours + overtime_hours`
  2. `overtime_ratio`: `overtime_hours / daily_total_hours`
  3. `is_weekend`: Binary flag for Saturday/Sunday
  4. `same_day_count`: Number of submissions by same employee on same date (detecting duplicates)
  5. `dept_zscore`: Standardized z-score of hours relative to department mean and standard deviation

### Empirical Performance Against Injected Ground Truth
Evaluated on **4017** historical timesheet rows containing **65** known injected anomalies (impossible hours >16h, duplicate same-day logs, weekend spikes).

| Metric | Measured Value | Operational Interpretation |
| :--- | :--- | :--- |
| **Precision** | **0.634** (63.4%) | Low false alarm rate; flagged items warrant human audit |
| **Recall** | **0.692** (69.2%) | Captures known compliance and fraud violations |
| **F1-Score** | **0.662** | Balanced operational metric for unsupervised detection |

---

## 3. Ethical Considerations & Limitations
1. **Model Drift**: If company policy changes standard working hours, the anomaly baseline z-scores must be re-calibrated.
2. **Fairness**: Department-normalized z-scores prevent engineering teams with legitimate on-call rotations from being penalized compared to administrative roles.
