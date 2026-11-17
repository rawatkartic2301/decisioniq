"""Timesheet anomaly detection tool for enterprise payroll auditing."""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional


class TimesheetAnomalyTool:
    """Detects and reports anomalous timesheet records (impossible hours, duplicates, spikes)."""

    def __init__(self, db_path: Path = Path("data/db/decisioniq.db")):
        self.db_path = db_path

    def inspect_anomalies(
        self,
        department: Optional[str] = None,
        limit: int = 15,
    ) -> Dict[str, Any]:
        """Fetch identified timesheet anomalies from database."""
        if not self.db_path.exists():
            return {"error": "Database not initialized", "anomalies": []}

        conn = sqlite3.connect(f"file:{self.db_path.resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        try:
            if department:
                cur.execute(
                    """
                SELECT t.id, t.employee_id, e.name, e.department, e.location,
                       t.date, t.project_code, t.regular_hours, t.overtime_hours,
                       (t.regular_hours + t.overtime_hours) as total_hours,
                       t.status, t.anomaly_reason
                FROM timesheets t
                JOIN employees e ON t.employee_id = e.id
                WHERE t.is_anomaly = 1 AND e.department = ?
                ORDER BY total_hours DESC
                LIMIT ?;
                """,
                    (department, limit),
                )
            else:
                cur.execute(
                    """
                SELECT t.id, t.employee_id, e.name, e.department, e.location,
                       t.date, t.project_code, t.regular_hours, t.overtime_hours,
                       (t.regular_hours + t.overtime_hours) as total_hours,
                       t.status, t.anomaly_reason
                FROM timesheets t
                JOIN employees e ON t.employee_id = e.id
                WHERE t.is_anomaly = 1
                ORDER BY total_hours DESC
                LIMIT ?;
                """,
                    (limit,),
                )

            rows = [dict(row) for row in cur.fetchall()]
            return {
                "detected_anomaly_count": len(rows),
                "anomalies": rows,
            }
        finally:
            conn.close()
