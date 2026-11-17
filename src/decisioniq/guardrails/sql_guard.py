"""AST-level SQL Guardrail using sqlglot for safe enterprise analytics."""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

DEFAULT_TABLE_ALLOWLIST: Set[str] = {"employees", "timesheets", "payroll_runs"}
FORBIDDEN_EXPRESSION_TYPES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.Command,
    exp.Pragma,
)


class SQLSecurityError(ValueError):
    """Raised when an unsafe or forbidden SQL query is attempted."""

    pass


class SafeSQLExecutor:
    """Validates SQL using sqlglot AST and executes in read-only SQLite mode."""

    def __init__(
        self,
        db_path: Path = Path("data/db/decisioniq.db"),
        allowed_tables: Optional[Set[str]] = None,
        max_limit: int = 100,
        timeout_seconds: float = 5.0,
    ):
        self.db_path = db_path
        self.allowed_tables = allowed_tables or DEFAULT_TABLE_ALLOWLIST
        self.max_limit = max_limit
        self.timeout_seconds = timeout_seconds

    def validate_and_format_sql(self, sql_query: str) -> str:
        """Parse query with sqlglot, enforce SELECT-only, table allowlist, and LIMIT clause."""
        cleaned = sql_query.strip().rstrip(";")
        if not cleaned:
            raise SQLSecurityError("Empty SQL query provided.")

        try:
            parsed = sqlglot.parse_one(cleaned, read="sqlite")
        except Exception as e:
            raise SQLSecurityError(f"SQL Syntax parsing error: {e}") from e

        # 1. Reject non-SELECT top-level expression
        if not isinstance(parsed, exp.Select):
            raise SQLSecurityError(
                f"Security Violation: Only SELECT statements are permitted. Received: {type(parsed).__name__}"
            )

        # 2. Reject any nested DDL/DML expressions
        for forbidden in FORBIDDEN_EXPRESSION_TYPES:
            if parsed.find(forbidden):
                raise SQLSecurityError(
                    f"Security Violation: Forbidden expression type '{forbidden.__name__}' detected in query."
                )

        # 3. Check table allowlist
        referenced_tables = {t.name.lower() for t in parsed.find_all(exp.Table) if t.name}
        unauthorized = referenced_tables - self.allowed_tables
        if unauthorized:
            raise SQLSecurityError(
                f"Security Violation: Unauthorized table access detected: {unauthorized}. "
                f"Allowed tables: {sorted(self.allowed_tables)}"
            )

        # 4. Enforce or clamp LIMIT clause
        limit_node = parsed.args.get("limit")
        if limit_node is None:
            parsed = parsed.limit(self.max_limit)
        else:
            try:
                current_limit = int(limit_node.expression.this)
                if current_limit > self.max_limit:
                    parsed = parsed.limit(self.max_limit)
            except Exception:
                parsed = parsed.limit(self.max_limit)

        return parsed.sql(dialect="sqlite")

    def execute(self, sql_query: str) -> Tuple[List[Dict[str, Any]], str]:
        """Validate and execute query against read-only database connection.

        Returns:
            Tuple of (rows_as_dicts, validated_sql)
        """
        validated_sql = self.validate_and_format_sql(sql_query)
        start_time = time.time()

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database file not found at {self.db_path}")

        # Open in URI read-only mode to prevent write locks or mutations
        db_uri = f"file:{self.db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=self.timeout_seconds)
        conn.row_factory = sqlite3.Row

        try:
            cur = conn.cursor()
            cur.execute(validated_sql)
            rows = [dict(row) for row in cur.fetchall()]
            elapsed = time.time() - start_time
            logger.info(
                "Executed safe SQL in %.3fs [%d rows]: %s", elapsed, len(rows), validated_sql
            )
            return rows, validated_sql
        finally:
            conn.close()
