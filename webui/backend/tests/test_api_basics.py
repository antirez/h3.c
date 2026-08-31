import json
import os
import stat

import pytest
from conftest import authed_client

from app.config import Settings
from app.system import parse_info

REAL_INFO = """h3-metal 0.1.0-dev
Device: NVIDIA GB10 (sm_121)
  physical memory       121.7 GiB
  recommended GPU set   121.7 GiB
  max GPU buffer        121.7 GiB
  unified memory        yes
Native checkpoint inventory (header-only):
  Qwen3-VL encoder    9 files   398 tensors   15.918 GiB
  FL2VA DiT          42 files  1204 tensors   54.063 GiB
  Ref2VA DiT         42 files  1204 tensors   54.063 GiB
  video VAE           1 files   402 tensors    0.749 GiB
  audio VAE           1 files   190 tensors    0.312 GiB
"""


def _fake_h3(tmp_path, stdout=REAL_INFO, code=0):
    binary = tmp_path / "h3"
    binary.write_text(f"#!/bin/sh\ncat <<'EOF'\n{stdout}EOF\nexit {code}\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return binary


@pytest.fixture
def client(tmp_path):
    config = Settings(
        binary=_fake_h3(tmp_path),
        model_dir=tmp_path / "model",
        data_dir=tmp_path / "data",
    )
    (tmp_path / "model").mkdir()
    with authed_client(config) as client:
        yield client


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_capabilities_serves_the_shared_schema_and_the_plugins(client):
    payload = client.get("/api/capabilities").json()
    shared = json.loads(Settings().schema_path.read_text())
    assert {key: payload[key] for key in shared} == shared
    assert [plugin["name"] for plugin in payload["plugins"]] == ["faceswap"]


def test_system_reports_device_and_components(client):
    payload = client.get("/api/system").json()
    assert payload["available"] is True
    assert payload["device"]["name"] == "NVIDIA GB10"
    assert payload["device"]["architecture"] == "sm_121"
    assert payload["device"]["physical_memory"] == "121.7 GiB"
    assert payload["components"]["FL2VA DiT"]["files"] == 42
    assert payload["components"]["audio VAE"]["gib"] == 0.312
    assert payload["has_ref2va"] is True


def test_system_degrades_when_the_binary_is_missing(tmp_path):
    config = Settings(
        binary=tmp_path / "absent",
        model_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    with authed_client(config) as client:
        payload = client.get("/api/system").json()
    assert payload["available"] is False
    assert "not found" in payload["reason"]


def test_system_degrades_when_h3_fails(tmp_path):
    config = Settings(
        binary=_fake_h3(tmp_path, stdout="h3: cannot map checkpoint\n", code=1),
        model_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    with authed_client(config) as client:
        payload = client.get("/api/system").json()
    assert payload["available"] is False
    assert payload["reason"] == "h3: cannot map checkpoint"


def test_the_database_is_created_with_both_tables(tmp_path, client):
    tables = {
        row["name"]
        for row in client.app.state.db.query_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"jobs", "assets"} <= tables


def test_parse_info_tolerates_the_apple_extra_lines():
    apple = REAL_INFO.replace(
        "  unified memory        yes",
        "  Apple GPU family      9\n  Metal 4               no\n"
        "  unified memory        yes",
    )
    info = parse_info(apple)
    assert info["device"]["Apple_GPU_family"] == "9"
    assert len(info["components"]) == 5


def test_settings_read_h3_prefixed_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("H3_MODEL_DIR", str(tmp_path / "elsewhere"))
    assert Settings().model_dir == tmp_path / "elsewhere"
    os.environ.pop("H3_MODEL_DIR", None)
