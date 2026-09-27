from __future__ import annotations

import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

JOBS: dict[str, dict] = {}
_POOL = ThreadPoolExecutor(max_workers=2)


def submit(kind: str, fn) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id,
        "kind": kind,
        "status": "pending",
        "progress": 0.0,
        "stage": "queued",
        "result": None,
        "error": None,
    }
    _POOL.submit(_run, job_id, fn)
    return job_id


def _run(job_id: str, fn) -> None:
    job = JOBS[job_id]
    job["status"] = "running"

    def cb(progress: float, stage: str) -> None:
        job["progress"] = round(min(max(float(progress), 0.0), 1.0), 3)
        if stage:
            job["stage"] = stage

    try:
        job["result"] = fn(cb)
        job["progress"] = 1.0
        job["stage"] = "done"
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["trace"] = traceback.format_exc(limit=8)


def get(job_id: str) -> dict | None:
    return JOBS.get(job_id)
