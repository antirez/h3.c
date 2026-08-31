"""T120 (R30): versioned schema, user tables, and password hashing.

The point of this task is that an existing database — the one with the
user's takes in it — must survive the migration byte for byte, and that a
fresh install starts at the latest version.
"""

import sqlite3

import pytest

from app.auth import hash_password, verify_password
from app.db import LATEST_VERSION, Database

OLD_SCHEMA = """
CREATE TABLE jobs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    state         TEXT NOT NULL DEFAULT 'queued',
    prompt        TEXT NOT NULL DEFAULT '',
    params        TEXT NOT NULL,
    argv          TEXT,
    phase         TEXT,
    completed     INTEGER NOT NULL DEFAULT 0,
    total         INTEGER NOT NULL DEFAULT 0,
    progress      REAL NOT NULL DEFAULT 0.0,
    error         TEXT,
    output_path   TEXT,
    log_path      TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    started_at    TEXT,
    finished_at   TEXT
);
CREATE TABLE assets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256        TEXT NOT NULL UNIQUE,
    kind          TEXT NOT NULL,
    filename      TEXT NOT NULL,
    path          TEXT NOT NULL,
    bytes         INTEGER NOT NULL,
    metadata      TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _columns(path, table):
    connection = sqlite3.connect(path)
    names = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
    connection.close()
    return names


def _tables(path):
    connection = sqlite3.connect(path)
    names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    connection.close()
    return names


def _seed_old_database(path):
    """A database as it existed before R30: version 1, real rows inside."""
    connection = sqlite3.connect(path)
    connection.executescript(OLD_SCHEMA)
    connection.execute(
        "INSERT INTO jobs (state, prompt, params) VALUES ('completed', 'p', '{}')"
    )
    connection.execute(
        "INSERT INTO assets (sha256, kind, filename, path, bytes)"
        " VALUES ('a' * 64, 'image', 'f.png', '/data/a/f.png', 10)"
    )
    connection.commit()
    connection.close()


def test_a_fresh_database_starts_at_the_latest_version(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    assert database.schema_version() == LATEST_VERSION
    tables = _tables(tmp_path / "db.sqlite")
    assert {"jobs", "assets", "users", "sessions", "invites"} <= tables
    assert "owner" in _columns(tmp_path / "db.sqlite", "jobs")
    assert "owner" in _columns(tmp_path / "db.sqlite", "assets")
    database.close()


def test_an_existing_database_migrates_without_losing_rows(tmp_path):
    path = tmp_path / "db.sqlite"
    _seed_old_database(path)

    database = Database(path)
    assert database.schema_version() == LATEST_VERSION

    jobs = database.query_all("SELECT * FROM jobs")
    assets = database.query_all("SELECT * FROM assets")
    assert len(jobs) == 1 and jobs[0]["prompt"] == "p"
    assert len(assets) == 1 and assets[0]["filename"] == "f.png"
    # Old rows are ownerless until an admin exists (backfill is a later task).
    assert jobs[0]["owner"] is None and assets[0]["owner"] is None
    database.close()


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "db.sqlite"
    _seed_old_database(path)
    Database(path).close()
    reopened = Database(path)
    assert reopened.schema_version() == LATEST_VERSION
    assert len(reopened.query_all("SELECT * FROM jobs")) == 1
    reopened.close()


def test_session_rows_cascade_with_their_user(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    user_id = database.run(
        "INSERT INTO users (username, password_hash) VALUES (?, ?)",
        ("admin", hash_password("secret")),
    )
    database.run(
        "INSERT INTO sessions (token, user_id, expires_at)"
        " VALUES ('t', ?, datetime('now', '+7 days'))",
        (user_id,),
    )
    database.run("DELETE FROM users WHERE id = ?", (user_id,))
    assert database.query_one("SELECT * FROM sessions WHERE token = 't'") is None
    database.close()


def test_only_admin_and_user_roles_exist(tmp_path):
    database = Database(tmp_path / "db.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        database.run(
            "INSERT INTO users (username, password_hash, role)"
            " VALUES ('x', 'h', 'superuser')"
        )
    database.close()


def test_password_hash_roundtrip():
    stored = hash_password("a long enough password")
    assert stored.startswith("$argon2id$")
    assert verify_password(stored, "a long enough password") is True
    assert verify_password(stored, "the wrong password") is False


def test_a_malformed_hash_is_never_a_match():
    assert verify_password("not a hash", "whatever") is False
    assert verify_password("", "whatever") is False
