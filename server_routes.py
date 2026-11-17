from pathlib import Path
from typing import Iterable
import uuid

from aiohttp import web
from server import PromptServer

from .face_math import reconstruct_face_mesh

_REGISTERED_EXTENSIONS: set[str] = set()
_DRIVING_BATCH_CACHE: dict[str, list[str]] = {}
_MAX_CACHED_BATCHES = 4


def cache_driving_batch(frames: list[str]) -> str:
    """Store encoded driving video frames in memory and return a short batch ID."""
    if len(_DRIVING_BATCH_CACHE) >= _MAX_CACHED_BATCHES:
        oldest_key = next(iter(_DRIVING_BATCH_CACHE))
        _DRIVING_BATCH_CACHE.pop(oldest_key, None)
    batch_id = uuid.uuid4().hex[:12]
    _DRIVING_BATCH_CACHE[batch_id] = frames
    return batch_id


def register_static_routes(package_dir: Path) -> None:
    """Expose js/, models/, and Face3D API endpoints on PromptServer."""
    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    extension_name = package_dir.name
    if extension_name in _REGISTERED_EXTENSIONS:
        return

    mount_specs: Iterable[tuple[str, Path]] = (
        (f"/extensions/{extension_name}/js", package_dir / "js"),
        (f"/extensions/{extension_name}/models", package_dir / "models"),
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

    _REGISTERED_EXTENSIONS.add(extension_name)
