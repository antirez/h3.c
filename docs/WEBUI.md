# h3c studio — the web UI

A browser front end for `h3`. Everything the CLI accepts is here: duration,
canvas, sampler, first/last frame anchors, ordered Ref2VA references, the
memory and parity switches, and a live preview of the denoising.

The design direction is documented in the mockups under [`docs/mockup/`](mockup/); `v4.html` is the current one, and `logo.html` shows the mark.

## What you need

- The same prerequisites as `h3` itself: an NVIDIA driver matching CUDA 13, the
  CUDA toolkit, ICU, and FFmpeg/FFprobe 6.1 or newer on `PATH`.
- The MiniMax-H3 checkpoint on disk (about 465 GB). It is never copied into a
  container image; it is mounted read-only.
- Python 3.12 and Node 22 for the local (non-Docker) path.
- For Docker: the NVIDIA Container Toolkit, so the container can see the GPU.

## Docker

`./install.sh` writes the `.env` for you and can fetch the checkpoint; see
[Installing with the script](../README.md#installing-with-the-script). By hand:

```sh
cp .env.example .env
# Set H3_MODEL_DIR to your checkpoint directory, then:
docker compose up --build
```

To check that the container sees the GPU and the checkpoint before waiting on
a first render — it prints the device and a tensor inventory, and nothing else:

```sh
docker compose run --rm --no-deps api /app/h3 --info -d /models
```

The UI is then on <http://127.0.0.1:8080> and the API on
<http://127.0.0.1:8000>. Both bind to the loopback address on purpose — see
*Security* below.

`NVCC_ARCH` in `.env` selects the CUDA architecture `h3` is compiled for;
`sm_121` is the NVIDIA GB10. The image is built for the architecture of the
machine that builds it.

## Running it without Docker

```sh
# 1. Build h3 as usual.
make -j"$(nproc)"

# 2. Backend.
python3 -m venv webui/backend/.venv
webui/backend/.venv/bin/pip install \
  "fastapi>=0.115" "uvicorn[standard]>=0.34" "pydantic-settings>=2.6" \
  python-multipart
H3_MODEL_DIR=./MiniMax-H3 \
  webui/backend/.venv/bin/uvicorn app.main:app \
  --app-dir webui/backend --host 127.0.0.1 --port 8000

# 3. Frontend, in another terminal.
cd webui/frontend
npm install
node scripts/generate-options.mjs
npm run dev
```

Open <http://127.0.0.1:5173>. The dev server proxies `/api` to the backend.

## How it works

```
browser ── /api ──▶ FastAPI ──▶ serial queue ──▶ ./h3 (one process per job)
   ▲                   │                              │
   └── SSE progress ────┘                              └── mp4, log, previews
```

- **One job at a time.** A single GPU with a 27 GB DiT peak: the queue is
  serial by design, and a queued job can be cancelled before it starts.
- **One `./h3` process per job.** Every CLI flag is therefore reachable, and
  cancelling is a signal to the process group. The cost is about 96 seconds of
  DiT load per job on a GB10.
- **Validation mirrors the engine.** The messages the browser shows before you
  submit are copied verbatim from `h3.c`, so a job that the UI accepts is a job
  the engine accepts.
- **Progress is weighted.** Phases are not equal: on the calibration run the
  transformer load took 40.9 s and the denoising 5.1 s. Weights live in
  `webui/shared/progress_weights.json`; regenerate them for your hardware with
  `webui/backend/tools/calibrate_progress.py`.
- **The option inventory has one source.** `webui/shared/options.schema.json`
  feeds both the backend validator and the generated TypeScript module, and a
  test fails if it drifts from `main.c`.

## The interface

Composition reads as a sentence: the prompt, then one line of choices —
length, shape, quality, variation — where each word opens where it stands
and carries its own time estimate. Photos to start from, end on, or keep as
references hang off the prompt as chips; files dropped anywhere on the page
land in the library.

Everything else lives in one panel with three tabs — Picture, Reference
material and Expert — and Expert exposes every CLI option, including the ten
`--use-slower-*` parity flags, with the exact command-line name. Two CLI
options are deliberately absent: `--show` and `--zoom` are terminal graphics
protocols with no meaning in a browser — the live preview uses
`--preview-dir` instead.

Finished videos form a gallery that plays on hover. While a video is being
made, the page shows the frame developing pass by pass, with a weighted
progress bar calibrated on this machine, and generation keeps running while
you compose the next one.

The administrator has a People page of its own: accounts, single-use
invites, password resets and deletions. Uploaded images, clips and
soundtracks stay in a library and can be reused in later jobs, as anchors or
as ordered references.

## Security

**Every call needs an account.** The whole API sits behind a session cookie.
The administrator account is defined on the server — `H3_ADMIN_USERNAME` and
`H3_ADMIN_PASSWORD` in `.env` — and is created once, on the first start of an
empty database; afterwards those values are ignored and the password is
managed from the People page. Every other account is made with a single-use
invite from that tab. Passwords are hashed with argon2id, sessions live in
the database (a logout or a password reset ends them at once), and five wrong
passwords in fifteen minutes pause that username. Videos and uploads belong
to the person who made them; an id that is not yours answers 404, the same
as one that does not exist.

What this does **not** do: there is no TLS. The service serves plain HTTP, so
passwords and cookies travel in the clear on whatever network carries them.
Outside a network you trust, put a TLS-terminating reverse proxy (Caddy,
nginx, Traefik) in front of it; binding to a private address is not a
substitute.

A fresh installation with no `H3_ADMIN_PASSWORD` has no administrator and
no way in: the backend says so in its log at startup. Set the two variables
before the first start.

Everything binds to `127.0.0.1` by default.

**A private overlay network.** With [Tailscale](https://tailscale.com), publish
the UI on the tailnet address instead of the loopback: only your own devices
can then connect, and the tailnet does the authenticating.

```sh
tailscale ip -4                 # e.g. 100.117.213.82
# in .env:
H3_BIND=100.117.213.82
docker compose up -d
```

Bear in mind that the container can only bind that address while Tailscale is
up; `restart: unless-stopped` retries if it is not.

**An SSH tunnel.** No configuration at all, from the other machine:

```sh
ssh -N -L 8080:127.0.0.1:8080 you@the-machine
```

Then open <http://localhost:8080> there.

**Publishing on the LAN** (`H3_BIND=0.0.0.0`) gives everyone on the network
the login screen — including attempts at the administrator password. If you
do it anyway, two things to know: only port 8080 needs publishing, because
nginx proxies `/api`; and a host firewall will not save you — ports published
by Docker are DNAT-ed in the `DOCKER-USER` chain, which `ufw` does not filter,
so a `ufw deny` rule has no effect on them.

## Post-processing

An optional stage can hand the finished video to an external program. No such
program is included, and no model is downloaded: see
[`docs/POSTPROCESSING.md`](POSTPROCESSING.md).
