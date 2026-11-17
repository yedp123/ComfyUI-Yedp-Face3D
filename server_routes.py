from pathlib import Path
import struct
import threading
from typing import Iterable
import uuid

from aiohttp import web
from server import PromptServer

from .face_math import reconstruct_face_mesh

_REGISTERED_EXTENSIONS: set[str] = set()
_DRIVING_BATCH_CACHE: dict[str, list[str]] = {}
_MAX_CACHED_BATCHES = 4

_RENDER_JOBS_LOCK = threading.Lock()
_RENDER_JOBS: dict[str, dict] = {}
_MAX_RENDER_JOBS = 8


def cache_driving_batch(frames: list[str]) -> str:
    """Store encoded driving video frames in memory and return a short batch ID."""
    if len(_DRIVING_BATCH_CACHE) >= _MAX_CACHED_BATCHES:
        oldest_key = next(iter(_DRIVING_BATCH_CACHE))
        _DRIVING_BATCH_CACHE.pop(oldest_key, None)
    batch_id = uuid.uuid4().hex[:12]
    _DRIVING_BATCH_CACHE[batch_id] = frames
    return batch_id


def create_render_job(width: int, height: int) -> tuple[str, dict]:
    """Allocate a thread-safe pending render job entry for browser-to-Python pass streaming."""
    job_id = uuid.uuid4().hex[:16]
    job = {
        "event": threading.Event(),
        "width": int(width),
        "height": int(height),
        "frames": {},
        "metadata": None,
        "error": None,
    }
    with _RENDER_JOBS_LOCK:
        if len(_RENDER_JOBS) >= _MAX_RENDER_JOBS:
            oldest_id = next(iter(_RENDER_JOBS))
            _RENDER_JOBS.pop(oldest_id, None)
        _RENDER_JOBS[job_id] = job
    return job_id, job


def pop_render_job(job_id: str) -> dict | None:
    """Remove and return a render job entry once completed or expired."""
    with _RENDER_JOBS_LOCK:
        return _RENDER_JOBS.pop(job_id, None)


def register_static_routes(package_dir: Path) -> None:
    """Expose js/, models/, assets/, and Face3D API endpoints on PromptServer."""
    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    extension_name = package_dir.name
    if extension_name in _REGISTERED_EXTENSIONS:
        return

    mount_specs: Iterable[tuple[str, Path]] = (
        (f"/extensions/{extension_name}/js", package_dir / "js"),
        (f"/extensions/{extension_name}/models", package_dir / "models"),
        (f"/extensions/{extension_name}/assets", package_dir / "assets"),
    )

    for url_prefix, directory in mount_specs:
        if directory.is_dir():
            prompt_server.routes.static(url_prefix, str(directory))

    @prompt_server.routes.post("/yedp_face3d/reconstruct")
    async def handle_reconstruct(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"Invalid JSON payload: {exc}"}, status=400)

        front = payload.get("front")
        triangles = payload.get("triangles")
        if not front or not front.get("landmarks") or not triangles:
            return web.json_response(
                {"error": "Missing required 'front.landmarks' or 'triangles'."},
                status=400,
            )

        try:
            mesh_data = reconstruct_face_mesh(
                front=front,
                triangles=triangles,
                three_quarter=payload.get("three_quarter"),
                side=payload.get("side"),
                depth_weight=float(payload.get("depth_weight", 1.0)),
            )
            return web.json_response(mesh_data)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @prompt_server.routes.get("/yedp_face3d/driving_batch/{batch_id}")
    async def handle_get_driving_batch(request: web.Request) -> web.Response:
        batch_id = request.match_info.get("batch_id", "")
        frames = _DRIVING_BATCH_CACHE.get(batch_id)
        if frames is None:
            return web.json_response({"error": "Driving batch not found or expired."}, status=404)
        return web.json_response({"batch_id": batch_id, "frames": frames})

    @prompt_server.routes.post("/yedp_face3d/render_chunk/{job_id}")
    async def handle_render_chunk(request: web.Request) -> web.Response:
        job_id = request.match_info.get("job_id", "")
        with _RENDER_JOBS_LOCK:
            job = _RENDER_JOBS.get(job_id)
        if job is None:
            return web.json_response({"error": "Render job not found or expired."}, status=404)

        body = await request.read()
        if len(body) < 16:
            return web.json_response({"error": "Truncated render chunk header."}, status=400)

        start_frame, frame_count, width, height = struct.unpack("<4I", body[:16])
        if width != job["width"] or height != job["height"] or frame_count == 0:
            return web.json_response({"error": "Render chunk dimensions mismatch job."}, status=400)

        frame_stride = width * height * 11
        expected_len = 16 + frame_count * frame_stride
        if len(body) != expected_len:
            return web.json_response(
                {"error": f"Expected {expected_len} bytes, received {len(body)}."},
                status=400,
            )

        mv = memoryview(body)
        with _RENDER_JOBS_LOCK:
            for k in range(frame_count):
                off = 16 + k * frame_stride
                job["frames"][start_frame + k] = bytes(mv[off : off + frame_stride])

        return web.json_response({"ok": True, "received": frame_count})

    @prompt_server.routes.post("/yedp_face3d/render_finish/{job_id}")
    async def handle_render_finish(request: web.Request) -> web.Response:
        job_id = request.match_info.get("job_id", "")
        with _RENDER_JOBS_LOCK:
            job = _RENDER_JOBS.get(job_id)
        if job is None:
            return web.json_response({"error": "Render job not found or expired."}, status=404)

        try:
            payload = await request.json()
        except Exception:
            payload = {}

        job["metadata"] = payload.get("metadata", {})
        job["event"].set()
        return web.json_response({"ok": True})

    @prompt_server.routes.post("/yedp_face3d/render_error/{job_id}")
    async def handle_render_error(request: web.Request) -> web.Response:
        job_id = request.match_info.get("job_id", "")
        with _RENDER_JOBS_LOCK:
            job = _RENDER_JOBS.get(job_id)
        if job is None:
            return web.json_response({"ok": False}, status=404)

        try:
            payload = await request.json()
        except Exception:
            payload = {}

        job["error"] = str(payload.get("error") or "Browser render error.")
        job["event"].set()
        return web.json_response({"ok": True})

    _REGISTERED_EXTENSIONS.add(extension_name)
