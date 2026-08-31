"""FastAPI application: health, capabilities and system inventory.

Every `/api/*` route requires a session cookie (R30); the only exceptions
are health and the two account endpoints. See the security note in the
README before exposing the service anywhere: it serves plain HTTP, and TLS
is the reverse proxy's job.
"""

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, ValidationError

from . import assets, auth, media
from .capabilities import load_schema
from .config import Settings, settings
from .db import Database
from .events import job_events
from .jobspec import EstimateRequest, JobSpec, validate
from .postprocess import registry
from .progress import observed_correction
from .runner import JobRunner
from .system import read_system

# Routes that must answer before anyone has an account.
PUBLIC_PATHS = {"/api/health", "/api/auth/login", "/api/auth/register"}


class RegisterRequest(BaseModel):
    username: str
    password: str
    invite: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class PasswordResetRequest(BaseModel):
    password: str


def create_app(config: Settings | None = None) -> FastAPI:
    config = config or settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.db = Database(config.data_dir / "h3.sqlite3")
        auth.bootstrap_admin(
            app.state.db, config.admin_username, config.admin_password
        )
        app.state.runner = JobRunner(app.state.db, config)
        app.state.runner.start()
        yield
        app.state.runner.shutdown()
        app.state.db.close()

    app = FastAPI(title="h3c studio", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def require_session(request: Request, call_next):
        """One door for the whole API: a valid session cookie, or 401.

        The SSE and media routes go through it too — they are just GETs the
        browser sends with the same cookie.
        """
        path = request.url.path
        if path.startswith("/api/") and path not in PUBLIC_PATHS:
            user = auth.session_user(
                app.state.db, request.cookies.get(auth.SESSION_COOKIE)
            )
            if user is None:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "authentication required"},
                )
            request.state.user = user
        return await call_next(request)

    @app.post("/api/auth/register", status_code=201)
    def register(payload: RegisterRequest) -> dict[str, Any]:
        db = app.state.db
        errors = auth.validate_credentials(payload.username, payload.password)
        if auth.get_user_by_username(db, payload.username) is not None:
            errors.append("that username is taken")
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})

        # Accounts are made with invites, full stop (R33): the administrator
        # comes from the server configuration, not from the door.
        invite = db.query_one(
            "SELECT 1 FROM invites WHERE code = ? AND used_at IS NULL",
            (payload.invite or "",),
        )
        if invite is None:
            raise HTTPException(
                status_code=400,
                detail="registration needs an invite from the administrator",
            )

        user_id = auth.create_user(db, payload.username, payload.password)
        auth.consume_invite(db, payload.invite or "", user_id)
        user = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
        return {"username": user["username"], "role": user["role"]}

    @app.post("/api/auth/login")
    def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
        db = app.state.db
        if auth.login_blocked(db, payload.username):
            raise HTTPException(
                status_code=429,
                detail="too many wrong passwords: wait a few minutes",
            )
        user = auth.get_user_by_username(db, payload.username)
        if user is None or not auth.verify_password(
            user["password_hash"], payload.password
        ):
            auth.record_failed_login(db, payload.username)
            raise HTTPException(status_code=401, detail="wrong username or password")
        auth.clear_failed_logins(db, payload.username)
        token = auth.create_session(db, user["id"])
        response.set_cookie(
            auth.SESSION_COOKIE,
            token,
            max_age=auth.SESSION_TTL_SECONDS,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return {"username": user["username"], "role": user["role"]}

    @app.post("/api/auth/logout", status_code=204)
    def logout(request: Request) -> Response:
        token = request.cookies.get(auth.SESSION_COOKIE)
        if token:
            auth.delete_session(app.state.db, token)
        response = Response(status_code=204)
        response.delete_cookie(auth.SESSION_COOKIE, path="/")
        return response

    @app.get("/api/auth/me")
    def me(request: Request) -> dict[str, Any]:
        user = request.state.user
        return {"username": user["username"], "role": user["role"]}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": app.version}

    @app.get("/api/capabilities")
    def capabilities() -> dict[str, Any]:
        return load_schema(config.schema_path) | {
            "plugins": [plugin.as_dict() for plugin in registry(config)]
        }

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return read_system(config.binary, config.model_dir, config.info_timeout)

    @app.post("/api/jobs", status_code=201)
    def create_job(spec: JobSpec, request: Request) -> dict[str, Any]:
        errors, warnings = validate(spec)
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        job = app.state.runner.submit(spec, owner=request.state.user["id"])
        return job | {"warnings": warnings}

    @app.post("/api/jobs/validate")
    def validate_job(spec: JobSpec) -> dict[str, Any]:
        errors, warnings = validate(spec)
        model = app.state.runner.model
        correction, learned_from = observed_correction(app.state.db, model)
        return {
            "errors": errors,
            "warnings": warnings,
            "frames": spec.resolved_frames(),
            "seconds": round(spec.resolved_frames() / 24, 3),
            "estimate_seconds": round(
                sum(seconds for _, seconds in model.plan(spec)) * correction, 1
            ),
            "learned_from": learned_from,
        }

    @app.post("/api/jobs/estimate")
    def estimate(request: EstimateRequest) -> dict[str, Any]:
        """How long this job would take, and how long each alternative would.

        One request labels every choice on screen, so a card can say what it
        costs before it is picked.
        """
        model = app.state.runner.model
        correction, learned_from = observed_correction(app.state.db, model)

        def total(candidate: JobSpec) -> float:
            plan = model.plan(candidate)
            return round(sum(seconds for _, seconds in plan) * correction, 1)

        spec = request.spec
        answered = []
        for override in request.variants:
            try:
                candidate = JobSpec.model_validate(spec.model_dump() | override)
            except ValidationError as error:
                answered.append({"override": override, "error": error.error_count()})
                continue
            answered.append({"override": override, "seconds": total(candidate)})
        return {
            "seconds": total(spec),
            "variants": answered,
            "learned_from": learned_from,
        }

    @app.get("/api/jobs")
    def list_jobs(request: Request, limit: int = 100) -> list[dict[str, Any]]:
        user = request.state.user
        owner = None if user["role"] == "admin" else user["id"]
        return app.state.runner.listing(limit, owner=owner)

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: int, request: Request) -> dict[str, Any]:
        job = _visible_job_or_404(app, job_id, request)
        return job

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int, request: Request) -> dict[str, Any]:
        _visible_job_or_404(app, job_id, request)
        job = app.state.runner.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: int, request: Request) -> Response:
        _visible_job_or_404(app, job_id, request)
        try:
            outcome = app.state.runner.delete(job_id)
        except OSError as failure:
            raise HTTPException(
                status_code=500,
                detail=f"the files of this video could not be removed: {failure}",
            ) from failure
        if outcome is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if outcome == "unfinished":
            raise HTTPException(
                status_code=409, detail="stop this video before deleting it"
            )
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/events")
    async def job_stream(job_id: int, request: Request) -> StreamingResponse:
        _visible_job_or_404(app, job_id, request)
        return StreamingResponse(
            job_events(app.state.runner, job_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: int, request: Request) -> FileResponse:
        job = _visible_job_or_404(app, job_id, request)
        path = Path(job["output_path"] or "")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="this job has no video")
        return FileResponse(path, media_type="video/mp4", filename=f"h3-{job_id}.mp4")

    @app.get("/api/jobs/{job_id}/poster")
    def job_poster(job_id: int, request: Request) -> FileResponse:
        job = _visible_job_or_404(app, job_id, request)
        video = Path(job["output_path"] or "")
        if not video.is_file():
            raise HTTPException(status_code=404, detail="this job has no video")
        poster = video.with_name("poster.jpg")
        if not poster.is_file() and not media.extract_poster(video, poster, config):
            raise HTTPException(status_code=404, detail="cannot build a poster")
        return FileResponse(poster, media_type="image/jpeg")

    @app.get("/api/jobs/{job_id}/preview")
    def job_preview(job_id: int, request: Request) -> FileResponse:
        _visible_job_or_404(app, job_id, request)
        jpeg = media.preview_jpeg(app.state.runner.preview_dir(job_id), config)
        if jpeg is None:
            raise HTTPException(status_code=404, detail="no preview yet")
        return FileResponse(
            jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"}
        )

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: int, request: Request) -> str:
        job = _visible_job_or_404(app, job_id, request)
        path = Path(job["log_path"] or "")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="this job has no log")
        return path.read_text(errors="replace")

    @app.get("/api/assets")
    def list_assets(request: Request) -> list[dict[str, Any]]:
        user = request.state.user
        owner = None if user["role"] == "admin" else user["id"]
        return assets.listing(app.state.db, owner=owner)

    @app.post("/api/assets", status_code=201)
    async def upload_asset(file: UploadFile, request: Request) -> dict[str, Any]:
        filename = Path(file.filename or "").name
        if not filename:
            raise HTTPException(status_code=400, detail="the upload has no file name")
        with tempfile.TemporaryDirectory() as staging:
            staged = Path(staging) / filename
            with staged.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
            try:
                return assets.store(
                    app.state.db,
                    staged,
                    filename,
                    config.data_dir / "assets",
                    config.max_upload_bytes,
                    config.ffprobe,
                    owner=request.state.user["id"],
                )
            except assets.AssetError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

    @app.get("/api/assets/{asset_id}/file")
    def asset_file(asset_id: int, request: Request) -> FileResponse:
        row = _visible_asset_or_404(app, asset_id, request)
        return FileResponse(row["path"], filename=row["filename"])

    # ── account administration (admin only, R30) ────────────────────────

    @app.get("/api/users")
    def list_users(request: Request) -> list[dict[str, Any]]:
        _require_admin(request)
        return [
            {
                "id": row["id"],
                "username": row["username"],
                "role": row["role"],
                "created_at": row["created_at"],
            }
            for row in app.state.db.query_all("SELECT * FROM users ORDER BY id")
        ]

    @app.get("/api/invites")
    def list_invites(request: Request) -> list[dict[str, Any]]:
        _require_admin(request)
        return [
            {
                "code": row["code"],
                "created_at": row["created_at"],
                "used": row["used_at"] is not None,
            }
            for row in app.state.db.query_all(
                "SELECT * FROM invites ORDER BY rowid DESC"
            )
        ]

    @app.post("/api/invites", status_code=201)
    def create_invite(request: Request) -> dict[str, Any]:
        _require_admin(request)
        code = auth.create_invite(app.state.db, request.state.user["id"])
        return {"code": code}

    @app.delete("/api/users/{user_id}", status_code=204)
    def delete_user(user_id: int, request: Request) -> Response:
        _require_admin(request)
        db = app.state.db
        if user_id == request.state.user["id"]:
            raise HTTPException(status_code=409, detail="you cannot delete yourself")
        user = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if user is None:
            raise HTTPException(status_code=404, detail="unknown user")
        jobs = db.query_one(
            "SELECT COUNT(*) AS n FROM jobs WHERE owner = ?", (user_id,)
        )["n"]
        owned_assets = db.query_one(
            "SELECT COUNT(*) AS n FROM assets WHERE owner = ?", (user_id,)
        )["n"]
        if jobs or owned_assets:
            raise HTTPException(
                status_code=409,
                detail="this account still has videos or uploads",
            )
        # Sessions die with the user (ON DELETE CASCADE); the invite trail
        # keeps the name of who used it (ON DELETE SET NULL).
        db.run("DELETE FROM users WHERE id = ?", (user_id,))
        return Response(status_code=204)

    @app.post("/api/users/{user_id}/password")
    def reset_password(
        user_id: int, payload: PasswordResetRequest, request: Request
    ) -> dict[str, Any]:
        _require_admin(request)
        db = app.state.db
        user = db.query_one("SELECT * FROM users WHERE id = ?", (user_id,))
        if user is None:
            raise HTTPException(status_code=404, detail="unknown user")
        errors = []
        if not (
            auth.MIN_PASSWORD_LENGTH
            <= len(payload.password)
            <= auth.MAX_PASSWORD_LENGTH
        ):
            errors.append(
                f"a password is between {auth.MIN_PASSWORD_LENGTH} and "
                f"{auth.MAX_PASSWORD_LENGTH} characters"
            )
        if errors:
            raise HTTPException(status_code=422, detail={"errors": errors})
        db.run(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (auth.hash_password(payload.password), user_id),
        )
        # Old sessions belong to the old secret.
        auth.delete_sessions_for_user(db, user_id)
        return {"username": user["username"]}

    return app


def _visible_job_or_404(app: FastAPI, job_id: int, request: Request) -> dict[str, Any]:
    """A job the caller may not see is indistinguishable from one that does
    not exist: 404, not 403, so a foreign id reveals nothing (R30)."""
    user = request.state.user
    job = app.state.runner.get(job_id)
    if job is None or (user["role"] != "admin" and job["owner"] != user["id"]):
        raise HTTPException(status_code=404, detail="unknown job")
    return job


def _visible_asset_or_404(app: FastAPI, asset_id: int, request: Request):
    user = request.state.user
    row = app.state.db.query_one(
        "SELECT * FROM assets WHERE id = ?", (asset_id,)
    )
    if row is None or (user["role"] != "admin" and row["owner"] != user["id"]):
        raise HTTPException(status_code=404, detail="unknown asset")
    return row


def _require_admin(request: Request) -> None:
    if request.state.user["role"] != "admin":
        raise HTTPException(status_code=403, detail="only the administrator can")


app = create_app()
