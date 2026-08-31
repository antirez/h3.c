"""SQLite storage for jobs, assets, users and sessions.

The worker thread and the request handlers share one connection, so every
statement goes through a lock: sqlite3 allows cross-thread use but not
concurrent use. One writer, one job at a time — no server is needed.

The schema is versioned. Version 1 is the original storage (jobs, assets);
later versions are additive migrations applied in order at open time, so an
existing database is never recreated and never loses rows (R30, T120).
"""

import sqlite3
import threading
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # Version 1 — the original storage.
    """
    CREATE TABLE IF NOT EXISTS jobs (
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

    CREATE TABLE IF NOT EXISTS assets (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        sha256        TEXT NOT NULL UNIQUE,
        kind          TEXT NOT NULL,
        filename      TEXT NOT NULL,
        path          TEXT NOT NULL,
        bytes         INTEGER NOT NULL,
        metadata      TEXT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    # Version 2 — users, sessions, invites, and an owner on what exists (T120).
    # Rows created before R30 stay ownerless until the first admin exists.
    """
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'user'
                      CHECK (role IN ('admin', 'user')),
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
        token         TEXT PRIMARY KEY,
        user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        expires_at    TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);

    CREATE TABLE IF NOT EXISTS invites (
        code          TEXT PRIMARY KEY,
        created_by    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        used_by       INTEGER REFERENCES users(id) ON DELETE SET NULL,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        used_at       TEXT
    );

    ALTER TABLE jobs ADD COLUMN owner INTEGER REFERENCES users(id);
    ALTER TABLE assets ADD COLUMN owner INTEGER REFERENCES users(id);
    """,
    # Version 3 — a login-attempt counter, so rate limiting needs no external
    # service (R30, T121). `at` is unix seconds.
    """
    CREATE TABLE IF NOT EXISTS login_attempts (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL,
        at            INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_attempts_username
        ON login_attempts (username, at);
    """,
    # Version 4 — content dedup becomes per-owner (R30, T122): two people may
    # hold the same file, and neither must see the other's library. The table
    # is rebuilt because SQLite cannot drop the old UNIQUE column constraint;
    # rows are copied across untouched.
    """
    CREATE TABLE assets_r30 (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        sha256        TEXT NOT NULL,
        kind          TEXT NOT NULL,
        filename      TEXT NOT NULL,
        path          TEXT NOT NULL,
        bytes         INTEGER NOT NULL,
        metadata      TEXT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        owner         INTEGER REFERENCES users(id)
    );
    INSERT INTO assets_r30 SELECT * FROM assets;
    DROP TABLE assets;
    ALTER TABLE assets_r30 RENAME TO assets;
    CREATE UNIQUE INDEX IF NOT EXISTS uq_assets_sha_owner
        ON assets (sha256, owner);
    """,
    # Version 5 — the restart sweep must know whether a job's h3 survived the
    # crash: jobs record the pid of the process running them (R27, T105).
    """
    ALTER TABLE jobs ADD COLUMN pid INTEGER;
    """,
]

LATEST_VERSION = len(MIGRATIONS)


class Closed(RuntimeError):
    """The database was closed while a background thread was still writing."""


class Database:
    """Every access is serialized and returns plain rows, never live cursors."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._open = True
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._migrate()

    @property
    def version(self) -> int:
        return self.schema_version()

    def schema_version(self) -> int:
        with self._guard() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
            return int(row[0])

    def _migrate(self) -> None:
        """Apply every migration after the recorded version, in one commit.

        A database without the user_version stamp but with tables already
        present is a version-1 database: stamp it, do not replay it.
        """
        connection = self._connection
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            has_jobs = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
            ).fetchone()
            if has_jobs:
                version = 1
        if version >= LATEST_VERSION:
            connection.execute(f"PRAGMA user_version = {LATEST_VERSION}")
            return
        for migration in MIGRATIONS[version:]:
            connection.executescript(migration)
        connection.execute(f"PRAGMA user_version = {LATEST_VERSION}")
        connection.commit()

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._guard() as connection:
            return connection.execute(sql, params).fetchone()

    def query_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._guard() as connection:
            return connection.execute(sql, params).fetchall()

    def run(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        with self._guard() as connection:
            cursor = connection.execute(sql, params)
            connection.commit()
            return int(cursor.lastrowid or 0)

    def close(self) -> None:
        with self._lock:
            if self._open:
                self._open = False
                self._connection.close()

    def _guard(self):
        database = self

        class _Guard:
            def __enter__(self) -> sqlite3.Connection:
                database._lock.acquire()
                if not database._open:
                    database._lock.release()
                    raise Closed("the database is closed")
                return database._connection

            def __exit__(self, *_: object) -> bool:
                database._lock.release()
                return False

        return _Guard()
