"""Server-sent events for one job.

The runner emits from its worker thread; each subscriber owns an asyncio queue
fed through the event loop. The stream ends when the job reaches a terminal
state, so the browser does not need to poll or to guess when to stop.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from .runner import TERMINAL_STATES, JobRunner

HEARTBEAT_SECONDS = 15.0


async def job_events(runner: JobRunner, job_id: int) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def listener(job: dict[str, Any]) -> None:
        if job["id"] == job_id:
            loop.call_soon_threadsafe(queue.put_nowait, job)

    snapshot = runner.get(job_id)
    if snapshot is None:
        yield _event("error", {"detail": "unknown job"})
        return

    runner.add_listener(listener)
    try:
        yield _event("job", snapshot)
        if snapshot["state"] in TERMINAL_STATES:
            return
        while True:
            try:
                job = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                # Keeps proxies from closing an idle stream during a long phase.
                yield ": keep-alive\n\n"
                continue
            yield _event("job", job)
            if job["state"] in TERMINAL_STATES:
                return
    finally:
        runner.remove_listener(listener)


def _event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"
