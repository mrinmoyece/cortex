"""`query_data` — the tool that used to lie.

It advertised itself to the model as a working SQL tool and returned an
empty result set with a note. The agent would "query the data", get zero
rows, and report confidently that there were no matching records. A tool
that fakes success is worse than one that is absent: the absent one can be
planned around.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import AsyncMock, patch

import pytest

import cortex.mcp.server as server


@pytest.fixture
def sales_db(tmp_path):
    path = tmp_path / "sales.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE orders (id INTEGER, region TEXT, amount REAL)")
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?)",
        [(1, "emea", 100.0), (2, "emea", 250.0), (3, "apac", 75.0)],
    )
    conn.commit()
    conn.close()
    return str(path)


@pytest.fixture
def configured(sales_db, monkeypatch):
    monkeypatch.setitem(server.DATABASE_ALIASES, "default", sales_db)
    return sales_db


def _sql_returning(statement: str):

    return patch.object(
        server,
        "_to_sql",
        AsyncMock(return_value=statement),
    )


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_a_real_query_returns_real_rows(self, configured):
        with _sql_returning("SELECT region, amount FROM orders WHERE region = 'emea'"):
            out = await server.query_data(natural_language_query="emea orders")
        assert out["row_count"] == 2
        assert out["columns"] == ["region", "amount"]
        assert [r[1] for r in out["rows"]] == [100.0, 250.0]

    @pytest.mark.asyncio
    async def test_the_generated_sql_is_returned_for_inspection(self, configured):
        """An agent answer citing data must be auditable back to the query
        that produced it."""
        with _sql_returning("SELECT COUNT(*) FROM orders"):
            out = await server.query_data(natural_language_query="how many orders")
        assert "SELECT" in out["sql"]

    @pytest.mark.asyncio
    async def test_large_result_sets_are_truncated_and_say_so(self, tmp_path, monkeypatch):
        path = tmp_path / "big.db"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE t (n INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(1000)])
        conn.commit()
        conn.close()
        monkeypatch.setitem(server.DATABASE_ALIASES, "big", str(path))

        with _sql_returning("SELECT n FROM t"):
            out = await server.query_data(natural_language_query="everything", database_alias="big")
        assert out["row_count"] == server.MAX_SQL_ROWS
        assert out["truncated"] is True


class TestRefusals:
    @pytest.mark.asyncio
    async def test_an_unconfigured_alias_refuses_instead_of_inventing(self):
        out = await server.query_data(natural_language_query="anything", database_alias="nope")
        assert out["row_count"] == 0
        assert "Unknown database alias" in out["error"]
        assert "rows" not in out, "a refusal must not look like an empty result set"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM orders",
            "DROP TABLE orders",
            "UPDATE orders SET amount = 0",
            "INSERT INTO orders VALUES (9, 'x', 1)",
            "SELECT 1; DROP TABLE orders",
            "  ATTACH DATABASE '/etc/passwd' AS pwn",
        ],
    )
    async def test_anything_but_a_single_select_is_refused(self, configured, sql):
        """An allowlist of one statement type, not a denylist of keywords:
        denylists lose to `DELETE/**/FROM` and to whatever the next SQLite
        version adds."""
        with _sql_returning(sql):
            out = await server.query_data(natural_language_query="be evil")
        assert out["row_count"] == 0
        assert "Refused" in out["error"]

    @pytest.mark.asyncio
    async def test_the_data_survives_an_attempted_write(self, configured):
        with _sql_returning("DELETE FROM orders"):
            await server.query_data(natural_language_query="delete everything")
        conn = sqlite3.connect(configured)
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3
        conn.close()

    @pytest.mark.asyncio
    async def test_broken_sql_reports_the_error_rather_than_raising(self, configured):
        with _sql_returning("SELECT nonexistent_column FROM orders"):
            out = await server.query_data(natural_language_query="bad query")
        assert out["row_count"] == 0
        assert "Query failed" in out["error"]
