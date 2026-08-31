# Post-processing plugins

After a video is generated, h3.c Studio can hand it to an external program
before publishing it. That is the whole extension point: a contract, not an
integration.

**This repository contains no models and no weights**, and nothing is ever
fetched at build time or at run time. Every plugin is unavailable until a
runtime is installed and an environment variable points at it.

One URL does appear here, in `install.sh`: `--with-faceswap` fetches the
**FaceFusion runtime** — the program, not the models — and only when you ask
for it on the command line or answer yes to the question. FaceFusion then
downloads its own models, from its own sources, under its own licence. The
default installs nothing.

## The contract

A plugin is an executable. The backend calls it with an argument list — never
through a shell — and waits:

```
$H3_<NAME>_CMD --input /path/to/in.mp4 --output /path/to/out.mp4
```

| Outcome | What the backend does |
| --- | --- |
| exit code `0` and the output file exists | the job's video is replaced by the output |
| exit code `0` but no output file | the job fails; the generated video is kept |
| any non-zero exit code | the job fails with the last line of stderr; the generated video is kept |
| the process does not finish within an hour | the job fails with a timeout |

The generated video is never deleted: a failed post-processing step costs you
the stage, not the render.

Anything the plugin writes to stderr ends up in the job log, so make errors
readable in one line.

## Registered plugins

| Name | Environment variable | Status in this repository |
| --- | --- | --- |
| `faceswap` | `H3_FACESWAP_CMD` | unavailable until you install a runtime |

`GET /api/capabilities` reports the same thing at run time, with the reason,
and the UI shows it disabled. Requesting an unavailable plugin fails the job
instead of silently ignoring it.

## Enabling one, with the installer

`./install.sh --with-faceswap` fetches FaceFusion into `vendor/facefusion`,
runs its own installer, and writes into `.env`:

- `H3_FACESWAP_CMD`, pointing at `scripts/faceswap-facefusion.sh`, the adapter
  that turns the `--input`/`--output` contract into FaceFusion's command line;
- `H3_FACEFUSION_DIR`, where the runtime went.

One value stays yours: `H3_FACESWAP_SOURCE`, the image of the face to put in.
The contract carries only the video, so the adapter reads the face from the
environment, and refuses to run until it is set.

This wires the **local** path, where FaceFusion runs in its own Python
environment on the host. It does not wire the container: the API image carries
neither FaceFusion nor its dependencies, and
[`docker-compose.faceswap.yml`](../docker-compose.faceswap.yml) expects a
runtime you have already made reachable inside the container.

FaceFusion's command line is its own and changes between releases; the adapter
is written against the 3.x `headless-run` and is the one file to adjust if
yours differs.

## Enabling one by hand

Installing a runtime is one configuration step, not a code change:

```sh
# 1. Install the runtime yourself, in its own environment.
# 2. Point the variable at the executable and restart the backend.
export H3_FACESWAP_CMD=/opt/faceswap/run
```

With Docker, use the override file, which mounts your runtime read-only:

```sh
export H3_FACESWAP_DIR=/opt/faceswap
export H3_FACESWAP_CMD=/opt/faceswap/run
docker compose -f docker-compose.yml -f docker-compose.faceswap.yml up
```

## Adding another plugin

Add an entry to `registry()` in `webui/backend/app/postprocess.py` with a name,
a label, a description and an environment variable. Nothing else changes: the
API lists it, and the UI renders it from the API — there is no plugin name
hard-coded in the frontend.

The stage is not specific to faces. Upscaling, frame interpolation or
watermarking fit the same `--input`/`--output` contract.

## Licences and consent

The face-swapping models that were evaluated for this project are licensed for
**non-commercial or research use only**. That is why they are not shipped here
and why no URL is given: checking the licence of what you install is your
responsibility, not this repository's.

**Do not use face replacement on images or videos of real people without their
informed consent.** Depicting someone saying or doing something they did not is
harmful regardless of how good the result looks, and in many jurisdictions it
is illegal. If you cannot obtain consent, do not run the stage.
