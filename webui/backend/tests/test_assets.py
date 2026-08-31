"""Uploads: whitelist, size cap, real ffprobe validation and deduplication."""

import shutil
import subprocess

import pytest
from conftest import authed_client

from app.assets import AssetError, kind_from_suffix
from app.config import Settings

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg and FFprobe are required to build and probe the fixtures",
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *args], check=True, timeout=120
    )


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """One real file of each kind, plus an audio track that is too short."""
    root = tmp_path_factory.mktemp("media")
    image = root / "fox.png"
    _ffmpeg(
        "-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-frames:v", "1", str(image)
    )
    video = root / "clip.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-shortest", "-pix_fmt", "yuv420p", str(video),
    )
    audio = root / "music.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=6", str(audio))
    short_audio = root / "blip.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(short_audio))
    return {
        "image": image,
        "video": video,
        "audio": audio,
        "short_audio": short_audio,
        "mislabelled": _copy(audio, root / "actually_audio.png"),
        "not_media": _write(root / "notes.txt", "hello"),
    }


def _copy(source, target):
    shutil.copyfile(source, target)
    return target


def _write(path, text):
    path.write_text(text)
    return path


@pytest.fixture
def client(tmp_path):
    config = Settings(data_dir=tmp_path / "data", model_dir=tmp_path)
    with authed_client(config) as client:
        yield client


def _post(client, path, name=None):
    with path.open("rb") as handle:
        return client.post("/api/assets", files={"file": (name or path.name, handle)})


def test_suffix_whitelist():
    assert kind_from_suffix(".PNG") == "image"
    assert kind_from_suffix(".mov") == "video"
    assert kind_from_suffix(".flac") == "audio"
    with pytest.raises(AssetError):
        kind_from_suffix(".exe")


def test_upload_image_records_its_geometry(client, media):
    response = _post(client, media["image"])
    assert response.status_code == 201
    body = response.json()
    assert body["kind"] == "image"
    assert body["metadata"]["width"] == 64
    assert body["duplicate"] is False


def test_upload_video_records_duration_and_audio_presence(client, media):
    body = _post(client, media["video"]).json()
    assert body["kind"] == "video"
    assert body["metadata"]["has_audio"] is True
    assert 2.9 <= body["metadata"]["seconds"] <= 3.2


def test_upload_audio_within_the_usable_range_has_no_notes(client, media):
    body = _post(client, media["audio"]).json()
    assert body["kind"] == "audio"
    assert body["metadata"]["notes"] == []


def test_audio_shorter_than_two_seconds_is_stored_but_flagged(client, media):
    body = _post(client, media["short_audio"]).json()
    assert body["kind"] == "audio"
    assert "shorter than the 2 s minimum" in body["metadata"]["notes"][0]


def test_uploading_the_same_bytes_twice_deduplicates(client, media):
    first = _post(client, media["image"]).json()
    second = _post(client, media["image"], name="copy.png").json()
    assert second["duplicate"] is True
    assert second["id"] == first["id"]
    assert len(client.get("/api/assets").json()) == 1


def test_extension_that_lies_about_the_content_is_refused(client, media):
    response = _post(client, media["mislabelled"])
    assert response.status_code == 400
    assert response.json()["detail"] == "the extension says image but the file is audio"


def test_unsupported_extension_is_refused(client, media):
    response = _post(client, media["not_media"])
    assert response.status_code == 400
    assert "unsupported file type" in response.json()["detail"]


def test_oversized_upload_is_refused(tmp_path, media):
    config = Settings(
        data_dir=tmp_path / "data", model_dir=tmp_path, max_upload_bytes=64
    )
    with authed_client(config) as client:
        response = _post(client, media["audio"])
    assert response.status_code == 400
    assert "over the" in response.json()["detail"]


def test_stored_file_can_be_downloaded_again(client, media):
    asset = _post(client, media["image"]).json()
    response = client.get(f"/api/assets/{asset['id']}/file")
    assert response.status_code == 200
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_unknown_asset_is_a_404(client):
    assert client.get("/api/assets/999/file").status_code == 404
