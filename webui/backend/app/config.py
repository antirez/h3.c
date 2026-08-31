"""Runtime configuration, read once from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Every value can be overridden with an H3_ prefixed environment variable."""

    model_config = SettingsConfigDict(env_prefix="H3_", extra="ignore")

    # Path to the h3 binary and to the MiniMax-H3 checkpoint directory.
    binary: Path = REPO_ROOT / "h3"
    model_dir: Path = REPO_ROOT / "MiniMax-H3"
    # Where jobs, uploads and generated media are written.
    data_dir: Path = REPO_ROOT / "webui/backend/data"
    # Canonical option inventory shared with the frontend.
    schema_path: Path = REPO_ROOT / "webui/shared/options.schema.json"
    # Measured phase durations behind the weighted progress bar.
    progress_weights_path: Path = REPO_ROOT / "webui/shared/progress_weights.json"
    # Seconds allowed for `h3 --info`, which only reads checkpoint headers.
    info_timeout: float = 120.0
    # Largest accepted upload, in bytes.
    max_upload_bytes: int = 512 * 1024 * 1024
    ffprobe: str = "ffprobe"
    ffmpeg: str = "ffmpeg"
    # Seconds between SIGTERM and SIGKILL when a job is cancelled.
    kill_grace: float = 10.0
    # Post-processing plugins: an executable path enables the plugin.
    # Nothing is installed or downloaded by this repository.
    faceswap_cmd: str = ""
    # The administrator account, defined by the deployment (R33). An empty
    # password means "not configured": the app starts with no administrator
    # and says so in the log.
    admin_username: str = "admin"
    admin_password: str = ""


@lru_cache
def settings() -> Settings:
    return Settings()
