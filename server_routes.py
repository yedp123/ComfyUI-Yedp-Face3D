import asyncio
from pathlib import Path
import struct
import threading
from typing import Iterable
import uuid

from aiohttp import web
import folder_paths
from server import PromptServer

from .face_math import reconstruct_face_mesh
from .uv_baker import bake_uv_texture

_REGISTERED_EXTENSIONS: set[str] = set()

_RENDER_JOBS_LOCK = threading.Lock()
_RENDER_JOBS: dict[str, dict] = {}
_MAX_RENDER_JOBS = 8


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


def _resolve_image_ref_path(image_ref: dict | None) -> Path | None:
    """Safely resolve a persisted ComfyUI image descriptor via folder_paths containment."""
    if not isinstance(image_ref, dict):
        return None
    filename = str(image_ref.get("filename") or "").strip()
    if not filename or "/" in filename or "\\" in filename or Path(filename).name != filename:
        return None
    dir_type = str(image_ref.get("type") or "input").strip()
    base_dir = folder_paths.get_directory_by_type(dir_type)
    if not base_dir:
        return None
    subfolder = str(image_ref.get("subfolder") or "").strip().lstrip("/\\")
    candidate = Path(base_dir) / subfolder / filename if subfolder else Path(base_dir) / filename
    if not folder_paths.is_within_directory(base_dir, str(candidate)):
        raise ValueError("Image reference path escapes allowed folder directory.")
    resolved = candidate.resolve()
    return resolved if resolved.is_file() else None


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

        def _reconstruct_view(raw: dict | None, include_seg_mask: bool = False) -> dict | None:
            if not isinstance(raw, dict):
                return raw
            out = {
                "landmarks": raw.get("landmarks"),
                "matrix": raw.get("matrix"),
                "width": raw.get("width", 512),
                "height": raw.get("height", 512),
            }
            if include_seg_mask:
                if raw.get("seg_mask_png") is not None:
                    out["seg_mask_png"] = raw.get("seg_mask_png")
                if raw.get("seg_mask_rgb") is not None:
                    out["seg_mask_rgb"] = raw.get("seg_mask_rgb")
            return out

        raw_silh_fit = payload.get("enable_silhouette_fit")
        raw_silh_str = payload.get("silhouette_strength")
        try:
            mesh_data = reconstruct_face_mesh(
                front=_reconstruct_view(front, include_seg_mask=False),
                triangles=triangles,
                three_quarter=_reconstruct_view(payload.get("three_quarter"), include_seg_mask=False),
                three_quarter_2=_reconstruct_view(payload.get("three_quarter_2"), include_seg_mask=False),
                side=_reconstruct_view(payload.get("side"), include_seg_mask=True),
                side_2=_reconstruct_view(payload.get("side_2"), include_seg_mask=True),
                depth_weight=float(payload.get("depth_weight", 1.0)),
                enable_reliability_weighting=bool(payload.get("enable_reliability_weighting", True)),
                mirror_transfer_weight=float(payload.get("mirror_transfer_weight", 0.85)),
                enable_anatomical_weighting=bool(payload.get("enable_anatomical_weighting", True)),
                structural_importance=float(payload.get("structural_importance", 1.75)),
                secondary_importance=float(payload.get("secondary_importance", 1.0)),
                fine_detail_importance=float(payload.get("fine_detail_importance", 0.65)),
                unreliable_importance=float(payload.get("unreliable_importance", 0.10)),
                huber_delta_ratio=float(payload.get("huber_delta_ratio", 0.08)),
                enable_regularization=bool(payload.get("enable_regularization", True)),
                base_geometry_anchoring=float(payload.get("base_geometry_anchoring", 0.25)),
                surface_deformation_smoothness=float(payload.get("surface_deformation_smoothness", 0.45)),
                max_local_displacement=float(payload.get("max_local_displacement", 0.18)),
                symmetry_strength=float(payload.get("symmetry_strength", 0.0)),
                enable_silhouette_fit=True if raw_silh_fit is None else bool(raw_silh_fit),
                silhouette_strength=0.75 if raw_silh_str is None else float(raw_silh_str),
            )
            return web.json_response(mesh_data)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

    @prompt_server.routes.post("/yedp_face3d/bake_uv")
    async def handle_bake_uv(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception as exc:
            return web.json_response({"error": f"Invalid JSON payload: {exc}"}, status=400)

        views_raw = payload.get("views")
        if not isinstance(views_raw, dict):
            views_raw = payload
        triangles = payload.get("triangles")
        front_raw = views_raw.get("front")
        if not isinstance(front_raw, dict) or not front_raw.get("landmarks") or not triangles:
            return web.json_response(
                {"error": "Missing required 'front.landmarks' or 'triangles'."},
                status=400,
            )

        views_clean: dict[str, dict] = {}
        try:
            for view_key in ("front", "three_quarter", "three_quarter_2", "side", "side_2"):
                raw = views_raw.get(view_key)
                if not isinstance(raw, dict) or not raw.get("landmarks"):
                    continue
                entry = {
                    "landmarks": raw["landmarks"],
                    "matrix": raw.get("matrix"),
                    "width": int(raw.get("width", 512)),
                    "height": int(raw.get("height", 512)),
                    "seg_mask_png": raw.get("seg_mask_png"),
                }
                resolved_path = _resolve_image_ref_path(raw.get("image_ref"))
                if resolved_path is not None:
                    entry["image_path"] = resolved_path
                if raw.get("image_data_url"):
                    entry["image_data_url"] = raw["image_data_url"]
                views_clean[view_key] = entry

            atlas_size = max(256, min(2048, int(payload.get("atlas_size", 1024))))
            debug_size = max(128, min(1024, int(payload.get("debug_size", 512))))
            enable_color_match = bool(payload.get("enable_color_match", True))
            enable_seam_blend = bool(payload.get("enable_seam_blend", True))
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                bake_uv_texture,
                views_clean,
                triangles,
                atlas_size,
                debug_size,
                enable_color_match,
                enable_seam_blend,
            )
            return web.json_response(result)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=500)

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

        frame_stride = width * height * 14
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
