"""Unit tests for Scikit-Learn Intent Classifier and Timesheet Anomaly Detector."""

from pathlib import Path

import pandas as pd
import pytest

from decisioniq.ml.anomaly_detector import TimesheetAnomalyDetector
from decisioniq.ml.intent_classifier import IntentClassifier


def test_intent_classifier_prediction_and_confidence():
    clf = IntentClassifier()
    # Query clearly about leave policy
    pred, conf = clf.predict_with_confidence("What is our policy on carry-forward leave?")
    assert pred in ["policy_rag", "hybrid_doc_sql"]
    assert 0.0 <= conf <= 1.0

    # Query clearly about SQL aggregation
    pred_sql, conf_sql = clf.predict_with_confidence("How many total employees are based in Pune?")
    assert pred_sql in ["sql_analytics", "hybrid_doc_sql"]
    assert conf_sql > 0.0

    # Query clearly about timesheet anomaly
    pred_anom, conf_anom = clf.predict_with_confidence(
        "Which timesheets look anomalous this month?"
    )
    assert pred_anom == "anomaly_detection"
    assert conf_anom > 0.0


def test_anomaly_detector_feature_engineering():
    sample_df = pd.DataFrame(
        [
            {
                "id": 1,
                "employee_id": 101,
                "name": "Alice Smith",
                "department": "Engineering",
                "location": "Pune",
                "grade": "L2",
                "date": "2024-03-10",
                "project_code": "PRJ-ENG-101",
                "regular_hours": 8.0,
                "overtime_hours": 2.0,
                "is_weekend": 0,
                "is_anomaly": 0,
                "anomaly_reason": None,
            },
            {
                "id": 2,
                "employee_id": 102,
                "name": "Bob Jones",
                "department": "Engineering",
                "location": "Pune",
                "grade": "L2",
                "date": "2024-03-10",
                "project_code": "PRJ-ENG-101",
                "regular_hours": 8.0,
                "overtime_hours": 12.0,  # 20 total hours -> extreme outlier
                "is_weekend": 0,
                "is_anomaly": 1,
                "anomaly_reason": "Impossible daily hours (>16h)",
            },
        ]
    )

    detector = TimesheetAnomalyDetector()
    feat_df = detector.engineer_features(sample_df)

    assert "daily_total_hours" in feat_df.columns
    assert "overtime_ratio" in feat_df.columns
    assert "same_day_count" in feat_df.columns
    assert "dept_zscore" in feat_df.columns

    assert feat_df.loc[0, "daily_total_hours"] == 10.0
    assert feat_df.loc[1, "daily_total_hours"] == 20.0


def test_anomaly_detector_inference():
    db_path = Path("data/db/decisioniq.db")
    if not db_path.exists():
        pytest.skip("Database does not exist for anomaly test.")

    detector = TimesheetAnomalyDetector(db_path=db_path)
    anomalies = detector.detect_anomalies(limit=5)
    assert len(anomalies) > 0
    top = anomalies[0]
    assert "anomaly_score" in top
    assert "is_predicted_anomaly" in top
    assert top["is_predicted_anomaly"] == 1
