import os
import sys
from contextlib import contextmanager
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# The administrator is declared by the deployment (R33): in the tests, the
# deployment is this environment, so every app the tests build bootstraps
# the same admin account at startup.
os.environ.setdefault("H3_ADMIN_USERNAME", "admin")
os.environ.setdefault("H3_ADMIN_PASSWORD", "correct-horse-9")


@contextmanager
def authed_client(config, username="admin", password="correct-horse-9"):
    """The app under test, signed in as its administrator.

    Production requires a session for every API call (R30), so the tests
    enter through the same door the browser does: the administrator comes
    from the configuration (R33), and the client logs in with it.
    """
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(config)) as client:
        response = client.post(
            "/api/auth/login",
            json={"username": username, "password": password},
        )
        assert response.status_code == 200, response.text
        yield client
