"""Unsupervised Timesheet Anomaly Detection using scikit-learn IsolationForest."""

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SAVED_MODELS_DIR = Path("src/decisioniq/ml/saved_models")
ANOMALY_MODEL_FILE = SAVED_MODELS_DIR / "anomaly_detector.joblib"


class TimesheetAnomalyDetector:
    """Feature engineering pipeline and IsolationForest anomaly detector for timesheet auditing."""

    def __init__(
        self,
        db_path: Path = Path("data/db/decisioniq.db"),
        model_file: Optional[Path] = None,
        contamination: float = 0.02,
    ):
        self.db_path = db_path
        self.model_file = model_file or ANOMALY_MODEL_FILE
        self.contamination = contamination
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            "daily_total_hours",
            "overtime_hours",
            "overtime_ratio",
            "is_weekend",
            "same_day_count",
            "dept_zscore",
        ]
        self._load_if_exists()

    def _load_if_exists(self) -> bool:
        if self.model_file.exists():
            try:
                data = joblib.load(self.model_file)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.feature_columns = data["feature_columns"]
                logger.info("Loaded TimesheetAnomalyDetector from %s", self.model_file)
                return True
            except Exception as e:
                logger.warning("Could not load anomaly model from %s: %s", self.model_file, e)
        return False

    def load_data_from_db(self) -> pd.DataFrame:
        """Load timesheets joined with employee metadata from SQLite."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")

        conn = sqlite3.connect(f"file:{self.db_path.resolve()}?mode=ro", uri=True)
        query = """
        SELECT t.id, t.employee_id, e.name, e.department, e.location, e.grade,
               t.date, t.project_code, t.regular_hours, t.overtime_hours,
               t.is_weekend, t.is_anomaly, t.anomaly_reason
        FROM timesheets t
        JOIN employees e ON t.employee_id = e.id;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer domain features: total hours, weekend ratio, same-day duplicate counts, and department z-scores."""
        feat_df = df.copy()

        # 1. Daily total hours
        feat_df["daily_total_hours"] = feat_df["regular_hours"] + feat_df["overtime_hours"]

        # 2. Overtime ratio
        feat_df["overtime_ratio"] = feat_df["overtime_hours"] / (
            feat_df["daily_total_hours"] + 1e-5
        )

        # 3. Duplicate project submissions on the same date by same employee
        same_day_counts = feat_df.groupby(["employee_id", "date"])["id"].transform("count")
        feat_df["same_day_count"] = same_day_counts

        # 4. Department statistical z-scores
        dept_stats = (
            feat_df.groupby("department")["daily_total_hours"].agg(["mean", "std"]).reset_index()
        )
        dept_stats["std"] = dept_stats["std"].replace(0, 1.0)  # avoid div by zero
        feat_df = feat_df.merge(dept_stats, on="department", how="left")
        feat_df["dept_zscore"] = (feat_df["daily_total_hours"] - feat_df["mean"]) / feat_df["std"]
        feat_df["dept_zscore"] = feat_df["dept_zscore"].fillna(0.0)

        return feat_df

    def train_and_evaluate(self) -> Dict[str, Any]:
        """Train unsupervised IsolationForest and evaluate against injected anomaly labels."""
        raw_df = self.load_data_from_db()
        df = self.engineer_features(raw_df)

        X = df[self.feature_columns].values
        X_scaled = self.scaler.fit_transform(X)

        # Initialize and fit IsolationForest
        self.model = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        # Predict (-1 is anomaly, 1 is normal)
        raw_preds = self.model.predict(X_scaled)
        y_preds = (raw_preds == -1).astype(int)  # 1 if anomaly, 0 if normal

        # Ground truth evaluation against injected labels
        y_true = df["is_anomaly"].values
        prec = precision_score(y_true, y_preds, zero_division=0)
        rec = recall_score(y_true, y_preds, zero_division=0)
        f1 = f1_score(y_true, y_preds, zero_division=0)

        # Save artifacts
        self.save()

        summary = {
            "total_records": len(df),
            "injected_anomalies": int(y_true.sum()),
            "predicted_anomalies": int(y_preds.sum()),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "contamination_parameter": self.contamination,
            "top_features": self.feature_columns,
        }
        logger.info(
            "Trained TimesheetAnomalyDetector (Precision: %.3f, Recall: %.3f, F1: %.3f)",
            prec,
            rec,
            f1,
        )
        return summary

    def detect_anomalies(
        self, df_input: Optional[pd.DataFrame] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Run anomaly detection and return top suspicious records ordered by severity."""
        if self.model is None:
            self._load_if_exists()
            if self.model is None:
                self.train_and_evaluate()

        raw_df = df_input if df_input is not None else self.load_data_from_db()
        df = self.engineer_features(raw_df)

        X = df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)
        preds = (self.model.predict(X_scaled) == -1).astype(int)

        df["anomaly_score"] = scores
        df["is_predicted_anomaly"] = preds

        anomalous = df[df["is_predicted_anomaly"] == 1].sort_values(
            by="anomaly_score", ascending=False
        )
        return anomalous.head(limit).to_dict(orient="records")

    def save(self):
        """Serialize model and scaler to disk."""
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
            },
            self.model_file,
        )
        logger.info("Saved anomaly detector artifact to %s", self.model_file)
