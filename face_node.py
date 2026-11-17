import asyncio
import base64
import io as py_io
import json
import math
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from comfy_api.latest import io
from server import PromptServer

from .server_routes import cache_driving_batch, create_render_job, pop_render_job

_RENDER_TIMEOUT_SECONDS = 120.0


def _frame_to_data_url(frame_tensor: torch.Tensor, max_side: int = 1024, quality: int = 92) -> str:
    """Convert a single [H, W, C] ComfyUI image tensor to a JPEG data URL on CPU."""
    frame = frame_tensor.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)
    arr = (frame.numpy() * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode="RGB")

    w, h = pil_img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / float(longest)
        pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    buf = py_io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _tensor_to_data_url(image_tensor: torch.Tensor | None, max_side: int = 1024) -> str | None:
    """Convert first frame of a [B, H, W, C] ComfyUI IMAGE tensor to a JPEG data URL on CPU."""
    if image_tensor is None or image_tensor.numel() == 0:
        return None
    return _frame_to_data_url(image_tensor[0], max_side=max_side, quality=92)


def _batch_to_data_urls(video_tensor: torch.Tensor | None, max_side: int = 512) -> list[str]:
    """Encode a [B, H, W, C] driving_video tensor into compact JPEG data URLs on CPU."""
    if video_tensor is None or video_tensor.numel() == 0:
        return []
    return [_frame_to_data_url(video_tensor[i], max_side=max_side, quality=85) for i in range(video_tensor.shape[0])]


def feather_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply separable Gaussian feathering on CPU to a [B, H, W] mask tensor."""
    if radius <= 0:
        return mask.clamp(0.0, 1.0)

    sigma = max(radius / 2.5, 0.5)
    kernel_size = radius * 2 + 1
    coords = torch.arange(kernel_size, dtype=mask.dtype, device=mask.device) - radius
    kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
    kernel_y = kernel_1d.view(1, 1, kernel_size, 1)

    x = mask.unsqueeze(1)
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, kernel_x)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    x = F.conv2d(x, kernel_y)
    return x.squeeze(1).clamp(0.0, 1.0)


def unpack_render_job_tensors(
    job: dict,
    mask_feather: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Convert streamed WebGL bottom-up raw uint8 pass buffers into standard ComfyUI CPU tensors:
    depth [B,H,W,3], normal [B,H,W,3], shaded [B,H,W,3], textured [B,H,W,3], pose [B,H,W,3], mask [B,H,W].
    """
    frames_dict = job["frames"]
    if not frames_dict:
        raise RuntimeError("Browser completed render job with zero frames.")

    w = int(job["width"])
    h = int(job["height"])
    n_px = w * h
    frame_indices = sorted(frames_dict.keys())
    batch_size = len(frame_indices)

    depth_np = np.empty((batch_size, h, w, 3), dtype=np.float32)
    normal_np = np.empty((batch_size, h, w, 3), dtype=np.float32)
    shaded_np = np.empty((batch_size, h, w, 3), dtype=np.float32)
    textured_np = np.empty((batch_size, h, w, 3), dtype=np.float32)
    pose_np = np.empty((batch_size, h, w, 3), dtype=np.float32)
    mask_np = np.empty((batch_size, h, w), dtype=np.float32)

    inv255 = np.float32(1.0 / 255.0)
    for b, f_idx in enumerate(frame_indices):
        raw = frames_dict[f_idx]
        # Flip vertical scanlines (WebGL readPixels is bottom-up, ComfyUI IMAGE is top-down)
        d = np.frombuffer(raw, dtype=np.uint8, count=n_px, offset=0).reshape(h, w)[::-1]
        depth_np[b, :, :, :] = (d.astype(np.float32) * inv255)[:, :, None]

        nrm = np.frombuffer(raw, dtype=np.uint8, count=n_px * 3, offset=n_px).reshape(h, w, 3)[::-1]
        normal_np[b] = nrm.astype(np.float32) * inv255

        shd = np.frombuffer(raw, dtype=np.uint8, count=n_px * 3, offset=n_px * 4).reshape(h, w, 3)[::-1]
        shaded_np[b] = shd.astype(np.float32) * inv255

        tex = np.frombuffer(raw, dtype=np.uint8, count=n_px * 3, offset=n_px * 7).reshape(h, w, 3)[::-1]
        textured_np[b] = tex.astype(np.float32) * inv255

        msk = np.frombuffer(raw, dtype=np.uint8, count=n_px, offset=n_px * 10).reshape(h, w)[::-1]
        mask_np[b] = msk.astype(np.float32) * inv255

        pse = np.frombuffer(raw, dtype=np.uint8, count=n_px * 3, offset=n_px * 11).reshape(h, w, 3)[::-1]
        pose_np[b] = pse.astype(np.float32) * inv255

    depth_t = torch.from_numpy(depth_np)
    normal_t = torch.from_numpy(normal_np)
    shaded_t = torch.from_numpy(shaded_np)
    textured_t = torch.from_numpy(textured_np)
    pose_t = torch.from_numpy(pose_np)
    mask_t = feather_mask(torch.from_numpy(mask_np), mask_feather)

    metadata_json = json.dumps(job.get("metadata") or {}, separators=(",", ":"))
    return depth_t, normal_t, shaded_t, textured_t, pose_t, mask_t, metadata_json


def build_fallback_passes(
    front_image: torch.Tensor | None,
    driving_video: torch.Tensor | None,
    depth_weight: float,
    render_width: int,
    render_height: int,
    turntable_frames: int,
    mask_feather: int,
    anatomical_guides: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """Generate zero-VRAM CPU fallback passes when executed headless without a browser WebSocket."""
    batch_size = int(driving_video.shape[0]) if driving_video is not None else int(turntable_frames)
    batch_size = max(1, batch_size)

    ys = torch.linspace(-1.0, 1.0, render_height, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, render_width, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    head_r2 = (grid_x / 0.38) ** 2 + ((grid_y + 0.06) / 0.52) ** 2
    inside_head = head_r2 <= 1.0

    if anatomical_guides:
        inside_neck = (grid_x.abs() <= 0.16) & (grid_y >= 0.36) & (grid_y <= 0.85)
        silhouette = (inside_head | inside_neck).to(torch.float32)
    else:
        silhouette = inside_head.to(torch.float32)

    z_surface = torch.sqrt((1.0 - head_r2.clamp(max=1.0)).clamp(min=0.0))
    depth_scale = max(0.0, min(depth_weight, 2.5))
    depth_single = torch.where(
        silhouette > 0.5,
        (z_surface * (0.55 + 0.35 * min(depth_scale, 1.5)) + 0.05).clamp(0.05, 1.0),
        torch.zeros_like(silhouette),
    )

    nx = (grid_x / 0.38) * silhouette
    ny = (-(grid_y + 0.06) / 0.52) * silhouette
    nz = z_surface.clamp(min=0.1) * silhouette + (1.0 - silhouette)
    norm_vec = F.normalize(torch.stack((nx, ny, nz), dim=-1), p=2, dim=-1)
    normal_single = torch.where(
        silhouette.unsqueeze(-1) > 0.5,
        (norm_vec * 0.5) + 0.5,
        torch.zeros((render_height, render_width, 3), dtype=torch.float32),
    )

    if front_image is not None and front_image.numel() > 0:
        avg_color = front_image[0].detach().to("cpu", dtype=torch.float32).mean(dim=(0, 1)).clamp(0.15, 0.95)
    else:
        avg_color = torch.tensor([0.75, 0.62, 0.55], dtype=torch.float32)

    key_light = F.normalize(torch.tensor([0.45, 0.55, 0.70], dtype=torch.float32), dim=0)
    lambert = (norm_vec * key_light).sum(dim=-1).clamp(min=0.0)
    shade = (0.35 + 0.65 * lambert).unsqueeze(-1)
    bg_color = torch.zeros((1, 1, 3), dtype=torch.float32)
    clay_albedo = torch.tensor([0.66, 0.66, 0.66], dtype=torch.float32).view(1, 1, 3)

    shaded_single = torch.where(silhouette.unsqueeze(-1) > 0.5, (clay_albedo * shade).clamp(0.0, 1.0), bg_color)
    textured_single = torch.where(silhouette.unsqueeze(-1) > 0.5, (avg_color.view(1, 1, 3) * shade).clamp(0.0, 1.0), bg_color)
    depth_single_rgb = depth_single.unsqueeze(-1).repeat(1, 1, 3)

    # Headless fallback OpenPose keypoints (Nose, L/R Eye, L/R Ear, optional Neck)
    pose_single = torch.zeros((render_height, render_width, 3), dtype=torch.float32)
    kp_specs = [
        (0.0, -0.02, (1.0, 0.0, 0.0)),      # Nose
        (0.14, -0.14, (1.0, 0.0, 1.0)),     # L_Eye
        (-0.14, -0.14, (0.667, 0.0, 1.0)),  # R_Eye
        (0.32, -0.06, (1.0, 0.0, 0.333)),   # L_Ear
        (-0.32, -0.06, (1.0, 0.0, 0.667)),  # R_Ear
    ]
    if anatomical_guides:
        kp_specs.append((0.0, 0.58, (1.0, 0.333, 0.0)))  # Neck
    for kx, ky, col in kp_specs:
        dot_mask = ((grid_x - kx) ** 2 + (grid_y - ky) ** 2) <= (0.045 ** 2)
        for ch in range(3):
            pose_single[:, :, ch] = torch.where(dot_mask, col[ch], pose_single[:, :, ch])

    depth_batch = depth_single_rgb.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    normal_batch = normal_single.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    shaded_batch = shaded_single.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    textured_batch = textured_single.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    pose_batch = pose_single.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    feathered_mask = feather_mask(silhouette.unsqueeze(0).repeat(batch_size, 1, 1), mask_feather)

    if batch_size > 1 and driving_video is None:
        angles = torch.linspace(-math.pi / 2, math.pi / 2, batch_size, dtype=torch.float32)
        mod = (0.88 + 0.12 * torch.cos(angles)).view(batch_size, 1, 1, 1)
        shaded_batch = (shaded_batch * mod).clamp(0.0, 1.0)
        textured_batch = (textured_batch * mod).clamp(0.0, 1.0)

    metadata = json.dumps({
        "projection": "ortho",
        "resolution": [render_width, render_height],
        "frame_count": batch_size,
        "motion_source": "headless_fallback",
        "depth_mode": "fixed",
        "pose_mode": "head_face",
    })
    return depth_batch, normal_batch, shaded_batch, textured_batch, pose_batch, feathered_mask, metadata


class YedpFace3D(io.ComfyNode):
    """ComfyUI V3 node for multi-view 3D face reconstruction, mocap, clay/relighting, and offscreen pass export."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="YedpFace3D",
            display_name="Yedp Face3D",
            category="Yedp/Face3D",
            description=(
                "Reconstructs a canonical 3D face + welded head shell on CPU, animates via webcam or video, "
                "and exports synchronized Depth, Normal, Shaded Clay, Textured, OpenPose Head+Face, Mask passes, and camera metadata."
            ),
            search_aliases=["face3d", "3d face", "mediapipe", "relight", "face mocap"],
            not_idempotent=True,
            inputs=[
                io.Image.Input(
                    "front_image",
                    optional=True,
                    tooltip="Primary front-facing reference portrait (optional if loaded via viewport stack).",
                ),
                io.Image.Input(
                    "three_quarter_image",
                    optional=True,
                    tooltip="Optional 3/4 view photo for lateral cheek depth refinement.",
                ),
                io.Image.Input(
                    "side_image",
                    optional=True,
                    tooltip="Optional side-profile photo for nose bridge, lips, chin, and jawline depth.",
                ),
                io.Image.Input(
                    "driving_video",
                    optional=True,
                    tooltip="Optional driving video image batch for per-frame facial expression and head pose.",
                ),
                io.Float.Input(
                    "depth_weight",
                    default=1.0,
                    min=0.0,
                    max=2.5,
                    step=0.05,
                    tooltip="Multiplier for reconstructed 3D facial depth relief.",
                ),
                io.Int.Input(
                    "render_width",
                    default=512,
                    min=128,
                    max=2048,
                    step=8,
                    tooltip="Output render width for all exported passes.",
                ),
                io.Int.Input(
                    "render_height",
                    default=512,
                    min=128,
                    max=2048,
                    step=8,
                    tooltip="Output render height for all exported passes.",
                ),
                io.Int.Input(
                    "turntable_frames",
                    default=24,
                    min=1,
                    max=240,
                    step=1,
                    tooltip="Frame count for the turntable sweep when no driving or recorded clip is active.",
                ),
                io.Int.Input(
                    "mask_feather",
                    default=12,
                    min=0,
                    max=64,
                    step=1,
                    tooltip="Soft Gaussian feather radius in pixels along the outer edge of mask_pass.",
                ),
                io.Boolean.Input(
                    "anatomical_guides",
                    default=True,
                    label_on="Head+Neck",
                    label_off="Face Only",
                    tooltip="Render full head shell + ears + neck (Head+Neck) or restrict all passes & viewport to the 478-vertex face (Face Only).",
                ),
            ],
            outputs=[
                io.Image.Output(
                    id="depth_pass",
                    display_name="depth_pass",
                    tooltip="Normalized grayscale Z-depth map batch.",
                ),
                io.Image.Output(
                    id="normal_pass",
                    display_name="normal_pass",
                    tooltip="Unlit camera-space RGB surface normal batch.",
                ),
                io.Image.Output(
                    id="shaded_pass",
                    display_name="shaded_pass",
                    tooltip="Neutral gray Clay render batch.",
                ),
                io.Image.Output(
                    id="textured_pass",
                    display_name="textured_pass",
                    tooltip="Textured and 3D-lit portrait batch.",
                ),
                io.Image.Output(
                    id="pose_pass",
                    display_name="pose_pass",
                    tooltip="Unlit OpenPose Head+Neck skeleton and 70-point facial expression keypoints batch.",
                ),
                io.Mask.Output(
                    id="mask_pass",
                    display_name="mask_pass",
                    tooltip="Soft-feathered head or face silhouette mask batch.",
                ),
                io.String.Output(
                    id="metadata_json",
                    display_name="metadata_json",
                    tooltip="JSON metadata describing projection, resolution, depth bounds, and frame angles.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        depth_weight: float,
        render_width: int,
        render_height: int,
        turntable_frames: int,
        mask_feather: int,
        anatomical_guides: bool,
        front_image: torch.Tensor | None = None,
        three_quarter_image: torch.Tensor | None = None,
        side_image: torch.Tensor | None = None,
        driving_video: torch.Tensor | None = None,
    ) -> io.NodeOutput:
        prompt_server = getattr(PromptServer, "instance", None)
        node_id = getattr(cls.hidden, "unique_id", None)
        has_clients = bool(prompt_server is not None and getattr(prompt_server, "sockets", None))

        if prompt_server is not None and node_id is not None and has_clients:
            driving_frames = _batch_to_data_urls(driving_video, max_side=512)
            driving_batch_id = cache_driving_batch(driving_frames) if driving_frames else None
            job_id, job = create_render_job(render_width, render_height)

            try:
                prompt_server.send_sync(
                    "yedp_face3d_render_job",
                    {
                        "node_id": str(node_id),
                        "job_id": job_id,
                        "front": _tensor_to_data_url(front_image),
                        "three_quarter": _tensor_to_data_url(three_quarter_image),
                        "side": _tensor_to_data_url(side_image),
                        "depth_weight": float(depth_weight),
                        "driving_batch_id": driving_batch_id,
                        "driving_frame_count": len(driving_frames),
                        "render_width": int(render_width),
                        "render_height": int(render_height),
                        "turntable_frames": int(turntable_frames),
                        "anatomical_guides": bool(anatomical_guides),
                    },
                )

                loop = asyncio.get_running_loop()
                completed = await loop.run_in_executor(None, job["event"].wait, _RENDER_TIMEOUT_SECONDS)
                if not completed:
                    raise TimeoutError(
                        f"Timed out waiting for browser WebGL render ({_RENDER_TIMEOUT_SECONDS:.0f}s)."
                    )
                if job.get("error"):
                    raise RuntimeError(f"Browser render failed: {job['error']}")

                depth_t, normal_t, shaded_t, textured_t, pose_t, mask_t, meta = unpack_render_job_tensors(
                    job, mask_feather
                )
                return io.NodeOutput(depth_t, normal_t, shaded_t, textured_t, pose_t, mask_t, meta)
            finally:
                pop_render_job(job_id)

        depth_t, normal_t, shaded_t, textured_t, pose_t, mask_t, meta = build_fallback_passes(
            front_image=front_image,
            driving_video=driving_video,
            depth_weight=depth_weight,
            render_width=render_width,
            render_height=render_height,
            turntable_frames=turntable_frames,
            mask_feather=mask_feather,
            anatomical_guides=anatomical_guides,
        )
        return io.NodeOutput(depth_t, normal_t, shaded_t, textured_t, pose_t, mask_t, meta)
