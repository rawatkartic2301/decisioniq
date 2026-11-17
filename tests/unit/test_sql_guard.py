"""Unit tests for AST SQL guardrail using sqlglot."""

from pathlib import Path

import pytest

from decisioniq.guardrails.sql_guard import SafeSQLExecutor, SQLSecurityError


def test_select_queries_allowed():
    executor = SafeSQLExecutor(max_limit=50)
    sql = "SELECT id, name FROM employees WHERE location = 'Pune'"
    formatted = executor.validate_and_format_sql(sql)
    assert "LIMIT 50" in formatted or "LIMIT 50" in formatted.upper()
    assert "FROM employees" in formatted


def test_non_select_rejected():
    executor = SafeSQLExecutor()
    with pytest.raises(SQLSecurityError, match="Only SELECT statements are permitted"):
        executor.validate_and_format_sql("DROP TABLE employees")

    with pytest.raises(SQLSecurityError, match="Only SELECT statements are permitted"):
        executor.validate_and_format_sql("DELETE FROM timesheets WHERE id = 1")

    with pytest.raises(SQLSecurityError, match="Only SELECT statements are permitted"):
        executor.validate_and_format_sql("INSERT INTO employees (name) VALUES ('Hacker')")

    with pytest.raises(SQLSecurityError, match="Only SELECT statements are permitted"):
        executor.validate_and_format_sql("UPDATE timesheets SET regular_hours = 20 WHERE id = 1")


def test_table_allowlist_enforced():
    executor = SafeSQLExecutor()
    # Unallowed table 'user_credentials'
    with pytest.raises(SQLSecurityError, match="Unauthorized table access"):
        executor.validate_and_format_sql("SELECT * FROM user_credentials")


def test_forced_limit_injection_and_clamping():
    executor = SafeSQLExecutor(max_limit=100)

    # Missing LIMIT -> automatically injected
    formatted = executor.validate_and_format_sql("SELECT name FROM employees")
    assert "LIMIT 100" in formatted.upper()

    # Limit 500 exceeds max 100 -> clamped
    formatted_clamp = executor.validate_and_format_sql("SELECT name FROM employees LIMIT 500")
    assert "LIMIT 100" in formatted_clamp.upper()

    # Limit 10 is below max 100 -> respected
    formatted_res = executor.validate_and_format_sql("SELECT name FROM employees LIMIT 10")
    assert "LIMIT 10" in formatted_res.upper()


def test_safe_execution_against_database():
    db_file = Path("data/db/decisioniq.db")
    if not db_file.exists():
        pytest.skip("SQLite database not generated yet.")

    executor = SafeSQLExecutor(db_path=db_file)
    rows, validated_sql = executor.execute("SELECT id, name, location FROM employees LIMIT 5")
    assert len(rows) <= 5
    assert "LIMIT 5" in validated_sql.upper()
    if rows:
        assert "name" in rows[0]
