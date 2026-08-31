"""Accounts, sessions, invites and login rate limiting for h3.c Studio (R30).

Passwords are hashed with argon2id (memory-hard, OWASP's current pick); the
hash string carries its parameters, so a future re-tune keeps reading old
hashes. Sessions live in a table, not in a JWT: the only client is the UI's
browser, and a table means a logout or a deleted account invalidates its
sessions immediately.

The administrator does not come from the door: it is declared by the
deployment (R33) — `H3_ADMIN_USERNAME` and `H3_ADMIN_PASSWORD` — and is
created once, on the first start of an empty database. After that the
environment is ignored and passwords are managed from the People tab.
"""

import logging
import re
import secrets
import time

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from .db import Database

log = logging.getLogger(__name__)

_hasher = PasswordHasher()

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128

SESSION_COOKIE = "h3_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600

RATE_LIMIT_WINDOW_SECONDS = 15 * 60
RATE_LIMIT_MAX_FAILURES = 5


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """A malformed or mismatched hash is simply not a match."""
    try:
        return _hasher.verify(password_hash, password)
    except (InvalidHashError, VerificationError):
        return False


def validate_credentials(username: str, password: str) -> list[str]:
    errors = []
    if not USERNAME_RE.match(username):
        errors.append(
            "a username is at most 32 of these characters: letters, "
            "digits, dots, dashes and underscores"
        )
    if len(password) < MIN_PASSWORD_LENGTH or len(password) > MAX_PASSWORD_LENGTH:
        errors.append(
            f"a password is between {MIN_PASSWORD_LENGTH} and "
            f"{MAX_PASSWORD_LENGTH} characters"
        )
    return errors


# ── users ──────────────────────────────────────────────────────────────────

def user_count(db: Database) -> int:
    row = db.query_one("SELECT COUNT(*) AS n FROM users")
    return int(row["n"]) if row else 0


def get_user_by_username(db: Database, username: str):
    return db.query_one("SELECT * FROM users WHERE username = ?", (username,))


def create_user(
    db: Database, username: str, password: str, role: str = "user"
) -> int:
    return db.run(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        (username, hash_password(password), role),
    )


def backfill_ownerless_rows(db: Database, user_id: int) -> None:
    """Whatever existed before accounts did belongs to the first admin (D30.4)."""
    db.run("UPDATE jobs SET owner = ? WHERE owner IS NULL", (user_id,))
    db.run("UPDATE assets SET owner = ? WHERE owner IS NULL", (user_id,))


def bootstrap_admin(db: Database, username: str, password: str) -> int | None:
    """Create the administrator from the deployment configuration (R33).

    Runs on an empty users table only: afterwards the environment has no say,
    and the account is managed like any other (People tab). Returns the id of
    the account it made, or None.
    """
    if user_count(db) > 0:
        return None
    if not password:
        log.warning(
            "no accounts exist and H3_ADMIN_PASSWORD is not set: "
            "nobody can sign in until one is configured"
        )
        return None
    user_id = create_user(db, username, password, "admin")
    backfill_ownerless_rows(db, user_id)
    return user_id


# ── sessions ───────────────────────────────────────────────────────────────

def create_session(db: Database, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    db.run(
        "INSERT INTO sessions (token, user_id, expires_at)"
        " VALUES (?, ?, datetime('now', '+' || ? || ' seconds'))",
        (token, user_id, SESSION_TTL_SECONDS),
    )
    # Housekeeping while we are here: expired sessions do not linger.
    db.run("DELETE FROM sessions WHERE expires_at <= datetime('now')")
    return token


def session_user(db: Database, token: str | None):
    """The user a session token belongs to, or None if it is not valid."""
    if not token:
        return None
    return db.query_one(
        "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id"
        " WHERE s.token = ? AND s.expires_at > datetime('now')",
        (token,),
    )


def delete_session(db: Database, token: str) -> None:
    db.run("DELETE FROM sessions WHERE token = ?", (token,))


def delete_sessions_for_user(db: Database, user_id: int) -> None:
    db.run("DELETE FROM sessions WHERE user_id = ?", (user_id,))


# ── invites ────────────────────────────────────────────────────────────────

def create_invite(db: Database, admin_id: int) -> str:
    code = secrets.token_urlsafe(9)
    db.run(
        "INSERT INTO invites (code, created_by) VALUES (?, ?)",
        (code, admin_id),
    )
    return code


def consume_invite(db: Database, code: str, user_id: int) -> bool:
    """Marks an unused invite as used; False if it does not exist or is gone."""
    row = db.query_one(
        "SELECT code FROM invites WHERE code = ? AND used_at IS NULL", (code,)
    )
    if row is None:
        return False
    db.run(
        "UPDATE invites SET used_by = ?, used_at = datetime('now')"
        " WHERE code = ? AND used_at IS NULL",
        (user_id, code),
    )
    return True


# ── login rate limiting ────────────────────────────────────────────────────

def record_failed_login(db: Database, username: str) -> None:
    db.run(
        "INSERT INTO login_attempts (username, at) VALUES (?, ?)",
        (username, int(time.time())),
    )


def clear_failed_logins(db: Database, username: str) -> None:
    db.run("DELETE FROM login_attempts WHERE username = ?", (username,))


def login_blocked(db: Database, username: str) -> bool:
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM login_attempts"
        " WHERE username = ? AND at >= ?",
        (username, int(time.time()) - RATE_LIMIT_WINDOW_SECONDS),
    )
    return bool(row and row["n"] >= RATE_LIMIT_MAX_FAILURES)
