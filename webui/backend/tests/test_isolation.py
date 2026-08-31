"""T122 (R30): per-user isolation and account administration.

The rule being verified: what you made is yours. Another person's takes and
uploads answer 404 — not 403 — because a foreign id must reveal nothing.
The administrator sees and manages everything.
"""

import base64
import stat

import pytest
from conftest import authed_client

from app.config import Settings

PASSWORD = "correct-horse-9"

# A real 1x1 PNG, so the upload goes through the same probe as a photo.
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture
def png_file(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(_PNG_BYTES)
    return path

SUCCESS = (
    "#!/bin/sh\n"
    "echo 'denoise 1/1' >&2\n"
    "mkdir -p \"$(dirname \"$0\")/../nope\" 2>/dev/null || true\n"
    "exit 0\n"
)

JOB = {"prompt": "x", "width": 256, "height": 256, "frames": 22, "steps": 2}


def _config(tmp_path):
    binary = tmp_path / "h3"
    binary.write_text(SUCCESS)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return Settings(
        binary=binary, model_dir=tmp_path, data_dir=tmp_path / "data"
    )


def _login_as(client, username, password=PASSWORD):
    response = client.post(
        "/api/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    token = response.cookies.get("h3_session")
    assert token
    return token


def _two_users(client):
    """Register admin (done by the fixture) and a second user; returns
    the session tokens of both, with the client wearing the user's."""
    db = client.app.state.db
    from app import auth

    invite = auth.create_invite(db, 1)
    response = client.post(
        "/api/auth/register",
        json={"username": "utente", "password": PASSWORD, "invite": invite},
    )
    assert response.status_code == 201, response.text
    admin_token = client.cookies.get("h3_session")
    user_token = _login_as(client, "utente")
    return admin_token, user_token


def test_takes_are_invisible_across_users(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        admin_token, user_token = _two_users(client)

        # The user makes a take.
        client.cookies.set("h3_session", user_token)
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        listing = client.get("/api/jobs").json()
        assert [job["id"] for job in listing] == [job_id]

        # The admin sees it, with the owner attached.
        client.cookies.set("h3_session", admin_token)
        assert client.get("/api/jobs").json()[0]["owner"] is not None

        # A third account — made through a fresh invite — sees nothing of it.
        from app import auth

        invite = auth.create_invite(client.app.state.db, 1)
        client.post(
            "/api/auth/register",
            json={"username": "terzo", "password": PASSWORD, "invite": invite},
        )
        terzo_token = _login_as(client, "terzo")
        client.cookies.set("h3_session", terzo_token)
        assert client.get("/api/jobs").json() == []
        for path in (
            f"/api/jobs/{job_id}",
            f"/api/jobs/{job_id}/video",
            f"/api/jobs/{job_id}/poster",
            f"/api/jobs/{job_id}/log",
            f"/api/jobs/{job_id}/events",
        ):
            assert client.get(path).status_code == 404, path
        assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 404
        assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_uploads_are_invisible_across_users(tmp_path, png_file):
    with authed_client(_config(tmp_path)) as client:
        admin_token, user_token = _two_users(client)

        client.cookies.set("h3_session", user_token)
        with png_file.open("rb") as handle:
            created = client.post(
                "/api/assets", files={"file": ("a.png", handle, "image/png")}
            )
        assert created.status_code == 201
        asset_id = created.json()["id"]
        assert [a["id"] for a in client.get("/api/assets").json()] == [asset_id]

        client.cookies.set("h3_session", admin_token)
        assert client.get(f"/api/assets/{asset_id}/file").status_code == 200
        client.cookies.set("h3_session", _login_as(client, "utente"))

        # Same bytes from another account: their own row, not the user's.
        with png_file.open("rb") as handle:
            again = client.post(
                "/api/assets", files={"file": ("a.png", handle, "image/png")}
            )
        assert again.status_code == 201


def test_admin_manages_accounts(tmp_path):
    with authed_client(_config(tmp_path)) as client:
        admin_token, user_token = _two_users(client)
        db = client.app.state.db

        # The list and a fresh invite are admin business.
        client.cookies.set("h3_session", admin_token)
        names = [u["username"] for u in client.get("/api/users").json()]
        assert names == ["admin", "utente"]
        code = client.post("/api/invites").json()["code"]
        assert client.get("/api/invites").json()[0]["code"] == code

        # A user gets a 403 on the same doors.
        client.cookies.set("h3_session", user_token)
        assert client.get("/api/users").status_code == 403
        assert client.post("/api/invites").status_code == 403
        assert client.delete("/api/users/1").status_code == 403

        # Resetting a password kills the old sessions.
        client.cookies.set("h3_session", admin_token)
        response = client.post(
            "/api/users/2/password", json={"password": "a new secret 1"}
        )
        assert response.status_code == 200
        client.cookies.set("h3_session", user_token)
        assert client.get("/api/auth/me").status_code == 401
        new_token = _login_as(client, "utente", password="a new secret 1")
        client.cookies.set("h3_session", new_token)
        assert client.get("/api/auth/me").json()["username"] == "utente"

        # Deleting an account refuses while it still owns things...
        client.cookies.set("h3_session", admin_token)
        db.run(
            "INSERT INTO jobs (state, prompt, params, owner)"
            " VALUES ('completed', 'x', '{}', 2)"
        )
        assert client.delete("/api/users/2").status_code == 409
        db.run("DELETE FROM jobs WHERE owner = 2")
        # ...and refuses self-deletion, then lets go of an empty account.
        assert client.delete("/api/users/1").status_code == 409
        assert client.delete("/api/users/2").status_code == 204
        assert client.delete("/api/users/2").status_code == 404
