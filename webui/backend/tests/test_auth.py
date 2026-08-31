"""T121/T128 (R30/R33): the auth endpoints and the door they guard.

What is covered, end to end through the HTTP surface:

- the administrator comes from the deployment configuration, not the door;
- every other account exists only through a single-use invite;
- login sets the cookie, logout kills the session, `me` names the user;
- everything under `/api/*` — lists, SSE streams and media included —
  answers 401 without a valid session;
- five wrong passwords buy a pause, not a lockout forever.
"""

import sqlite3

from conftest import authed_client

from app.config import Settings

PASSWORD = "correct-horse-9"


def _config(tmp_path):
    return Settings(
        binary=tmp_path / "absent", model_dir=tmp_path, data_dir=tmp_path / "data"
    )


def _register(client, username="someone", password=PASSWORD, invite=None):
    payload = {"username": username, "password": password}
    if invite is not None:
        payload["invite"] = invite
    return client.post("/api/auth/register", json=payload)


def _login(client, username="someone", password=PASSWORD):
    return client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )


def _invite(client):
    from app import auth

    db = client.app.state.db
    admin = db.query_one("SELECT id FROM users WHERE username = 'admin'")
    return auth.create_invite(db, admin["id"])


def test_the_administrator_comes_from_the_configuration(tmp_path):
    # The test environment carries H3_ADMIN_USERNAME/H3_ADMIN_PASSWORD, as a
    # .env would in production: the account exists as soon as the app starts.
    with authed_client(_config(tmp_path)) as client:
        response = client.get("/api/auth/me")
        assert response.status_code == 200
        assert response.json() == {"username": "admin", "role": "admin"}


def test_without_a_configured_admin_there_is_no_door_in(tmp_path):
    config = _config(tmp_path)
    config.admin_password = ""
    with _raw_client(config) as client:
        assert _login(client, username="admin").status_code == 401
        # And nobody can invite themselves in: registration needs an invite.
        assert _register(client).status_code == 400


def _raw_client(config):
    from fastapi.testclient import TestClient

    from app.main import create_app

    return TestClient(create_app(config))


def test_registration_needs_an_invite_even_for_the_first_try(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        assert _register(client).status_code == 400
        assert _register(client, invite="not-a-real-code").status_code == 400

        invite = _invite(client)
        response = _register(client, invite=invite)
        assert response.status_code == 201
        assert response.json() == {"username": "someone", "role": "user"}
        # An invite is single-use.
        assert _register(client, username="third", invite=invite).status_code == 400


def test_register_rejects_bad_input(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        invite = _invite(client)
        response = _register(client, username="has spaces", invite=invite)
        assert response.status_code == 422
        assert "errors" in response.json()["detail"]

        response = _register(client, password="short", invite=invite)
        assert response.status_code == 422

        response = _register(client, username="admin", invite=invite)
        assert response.status_code == 422
        assert "taken" in response.text
        # The rejected attempts did not burn the invite.
        assert _register(client, invite=invite).status_code == 201


def test_login_and_logout_lifecycle(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        invite = _invite(client)
        assert _register(client, username="solo", invite=invite).status_code == 201

        response = _login(client, username="solo", password="the wrong one")
        assert response.status_code == 401

        response = _login(client, username="solo")
        assert response.status_code == 200
        assert client.cookies.get("h3_session")

        me = client.get("/api/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "solo"

        assert client.post("/api/auth/logout").status_code == 204
        # The session row is gone: the same cookie buys nothing now.
        assert client.get("/api/auth/me").status_code == 401


def test_the_whole_api_is_behind_the_door(tmp_path):
    with _raw_client(_config(tmp_path)) as anonymous:
        assert anonymous.get("/api/jobs").status_code == 401
        assert anonymous.get("/api/assets").status_code == 401
        assert anonymous.get("/api/system").status_code == 401
        assert anonymous.get("/api/jobs/1/events").status_code == 401
        assert anonymous.get("/api/jobs/1/video").status_code == 401
        assert anonymous.get("/api/jobs/1/poster").status_code == 401
        assert anonymous.get("/api/jobs/1/log").status_code == 401
        assert anonymous.post("/api/jobs", json={}).status_code == 401
        # Health stays open: monitoring must not need an account.
        assert anonymous.get("/api/health").status_code == 200


def test_five_wrong_passwords_buy_a_pause(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        invite = _invite(client)
        assert _register(client, username="solo", invite=invite).status_code == 201
        for _ in range(5):
            bad = _login(client, username="solo", password="wrong one")
            assert bad.status_code == 401
        # Even the right password now has to wait out the window.
        assert _login(client, username="solo").status_code == 429


def test_a_successful_login_clears_the_counter(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        invite = _invite(client)
        assert _register(client, username="solo", invite=invite).status_code == 201
        for _ in range(4):
            bad = _login(client, username="solo", password="wrong one")
            assert bad.status_code == 401
        assert _login(client, username="solo").status_code == 200
        # The slate is clean: four more misses are allowed, not one.
        for _ in range(4):
            bad = _login(client, username="solo", password="wrong one")
            assert bad.status_code == 401
        assert _login(client, username="solo").status_code == 200


def test_what_existed_before_accounts_goes_to_the_configured_admin(tmp_path):
    config = _config(tmp_path)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    database = config.data_dir / "h3.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
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
        INSERT INTO jobs (state, prompt, params) VALUES ('completed', 'x', '{}');
        """
    )
    connection.commit()
    connection.close()

    # The backfill happens with the bootstrap, before anyone signs in.
    with authed_client(config) as client:
        rows = client.app.state.db.query_all("SELECT owner FROM jobs")
        assert len(rows) == 1
        admin = client.app.state.db.query_one(
            "SELECT id FROM users WHERE username = 'admin'"
        )
        assert rows[0]["owner"] == admin["id"]
